import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser,JsonOutputParser
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dynamic_sample(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    n = len(df)
    frac = 1.0 if n < 10_000 else 0.1 if n < 100_000 else 0.03 if n < 1_000_000 else 0.01
    return df.sample(frac=frac, random_state=random_state)

class PythonOutputParser(BaseOutputParser):
    """Extract Python code from markdown blocks."""
    def parse(self, text: str) -> str:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text

class EDA_Node:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def perform_eda_analysis(self, state: PythonAnalystState) -> dict:
        """Generates EDA Python function from cleaned data + user query."""
        # Build a safe, limited markdown sample for the prompt
        samples = []
        for i, df in enumerate(state.get("cleaned_data", []) or []):
            if isinstance(df, pd.DataFrame):
                try:
                    md = dynamic_sample(df).to_markdown(index=False)
                except Exception:
                    # fallback to a small string if markdown fails
                    md = dynamic_sample(df).head(10).to_string(index=False)
                samples.append(f"File {i+1} Sample:\n{md}")

        cleaned_data_sample = "\n\n".join(samples) or "No cleaned data samples available."

        # Prompt template (fixed and with closed code fence)
        eda_prompt = PromptTemplate(
            template="""
    You are a senior data analyst.

    CONTEXT (small representative sample in markdown):
    {cleaned_data_sample}

    User question:
    "{user_query}"

    Task:
    Generate one compact, production-grade, fully-executable Python function named exactly `perform_eda(df)`.

    **Mandatory Requirements:**
    1. At the start of the function include:
    import pandas as pd
    import numpy as np
    from datetime import datetime

    2. Never use deprecated NumPy aliases (`np.float`, `np.int`, `np.bool`, `np.object`).
   - Use built-in `float`, `int`, `bool`, `object` or explicit `np.float64`/`np.int64` instead.

    3. Input validation:.

    3. Implement these EDA steps inside the function:
    a. Dataset overview: shape, column names, dtypes.
    b. Descriptive statistics (numeric only): count, mean, std, min, 25%, 50%, 75%, max.
    c. Outlier detection using IQR: counts per numeric column.
    d. Correlation analysis (numeric only): list pairs where abs(corr) > 0.5.
    e. Data quality flags: constant columns, mixed-type columns, missing counts.
    f. Suspicious values: negative ages, future dates, non-positive prices.

    4. Constraints:
    - Use df.shape[0] and df.shape[1] for counts.
    - Only return serializable Python types (dicts, lists, numbers, strings).
    - Do NOT print, plot, or perform file I/O.
    - Use only pandas/numpy/datetime operations.

    OUTPUT:
    Return a single dict named `eda_results` and nothing else. Wrap the function exactly in a fenced python block, for example:

    ```python
    def perform_eda(df):
        ...
        return eda_results
    Do not include any text outside the fenced code block.
    """,
        input_variables=["cleaned_data_sample", "user_query"],
        )   

        chain = eda_prompt | self.llm | PythonOutputParser()
        code = chain.invoke({
            "cleaned_data_sample": cleaned_data_sample,
            "user_query": state.get("question", "")
        })

        code = code.strip()
        if "def perform_eda" not in code:
            raise ValueError("LLM did not produce a function named 'perform_eda'. Received:\n" + code[:1000])

        logger.info("EDA function generated (length %d chars)", len(code))
        return {"eda_code": code}

    def execute_eda_code(self, state: PythonAnalystState) -> dict:

        if "eda_code" not in state:
            raise ValueError("Missing EDA code in state")
        if "raw_data" not in state:
            raise ValueError("Missing raw data in state")

        eda_code = state["eda_code"]
        raw_data = state["raw_data"]
        eda_outputs = []

        # Replace deprecated tokens in generated code
        eda_code = (
            eda_code.replace("np.float", "float")
                    .replace("np.int", "int")
                    .replace("np.bool", "bool")
                    .replace("np.object", "object")
        )

        # Common dtype name -> numpy dtype mapping
        common_dtype_names = [
            "float64", "float32", "int64", "int32", "int16", "int8",
            "uint64", "uint32", "object", "bool"
        ]

        # Helper to build base global environment with dtype aliases
        def make_global_env():
            """Builds a safe execution environment for running generated EDA code without NumPy alias warnings."""
            env = {
                "pd": pd,
                "np": np,
                "datetime": datetime,
                "__builtins__": __builtins__,
                # Use built-in Python types for these
                "float": float,
                "int": int,
                "bool": bool,
                "object": object,
            }

            # Only include valid NumPy dtypes (avoid deprecated aliases like np.object, np.int, np.bool)
            valid_numpy_dtypes = [
                "float64", "float32", "int64", "int32", "int16", "int8",
                "uint64", "uint32", "bool_", "complex64", "complex128"
            ]

            for name in valid_numpy_dtypes:
                try:
                    env[name] = getattr(np, name)
                except AttributeError:
                    pass  # Skip if NumPy version doesn't have this dtype

            return env

        # Attempt execution for each DataFrame
        for i, df in enumerate(raw_data, start=1):
            if not isinstance(df, pd.DataFrame):
                self.logger.warning("Item %d in raw_data is not a DataFrame. Skipping...", i)
                continue

            tried_retry = False
            while True:
                try:
                    global_env = make_global_env()
                    local_vars = {}

                    # Execute the provided EDA code
                    exec(eda_code, global_env, local_vars)

                    # Retrieve the EDA function
                    eda_func = local_vars.get("perform_eda")
                    if not callable(eda_func):
                        raise ValueError("No callable function named 'perform_eda' found in the provided code")

                    # Execute the function with a copy of the DataFrame (non-destructive)
                    result = eda_func(df.copy())

                    if not isinstance(result, dict):
                        raise ValueError("'perform_eda' must return a dictionary")

                    eda_outputs.append(result)
                    self.logger.info("Successfully executed EDA on DataFrame %d", i)
                    break  # success -> exit retry loop

                except NameError as ne:
                    msg = str(ne)
                    self.logger.error("NameError executing EDA on DataFrame %d: %s", i, msg)
                    missing = None
                    m = re.search(r"name '([^']+)' is not defined", msg)
                    if m:
                        missing = m.group(1)
                    if missing and (missing in common_dtype_names) and not tried_retry:
                        self.logger.info("Mapping missing name '%s' to numpy dtype and retrying", missing)
                        tried_retry = True
                        continue
                    else:
                        hint = None
                        if missing:
                            hint = f"Missing name: {missing}. Generated code referenced an unqualified dtype name."
                        eda_outputs.append({"error": msg, "hint": hint, "code": eda_code})
                        break

                except Exception as e:
                    err_msg = str(e)
                    self.logger.error("Failed EDA on DataFrame %d: %s", i, err_msg)
                    hint = None
                    if "cannot be interpreted as an integer" in err_msg or "must be real number" in err_msg:
                        hint = (
                            "Likely cause: The EDA code used a DataFrame/Series where an integer was expected "
                            "(e.g., `range(df)`, `for i in df`, or indexing with a DataFrame). "
                            "Check the generated code for `range(` or integer-context usage of 'df'."
                        )
                    eda_outputs.append({"error": err_msg, "hint": hint, "code": eda_code})
                    break

        return {"eda_result": eda_outputs}
