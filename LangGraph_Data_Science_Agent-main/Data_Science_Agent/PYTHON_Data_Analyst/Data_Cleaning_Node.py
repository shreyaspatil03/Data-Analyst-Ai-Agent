from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import BaseOutputParser
import re
import pandas as pd
from typing import List
import logging
import logging
logger = logging.getLogger(__name__)

def dynamic_sample(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    n = len(df)
    if n < 200:
        frac = 1.0
    elif n <= 500:
        frac = 0.5
    elif n < 10_000:
        frac = 1.0
    elif n < 100_000:
        frac = 0.1
    elif n < 1_000_000:
        frac = 0.03
    else:
        frac = 0.01
    return df.sample(frac=frac, random_state=random_state)

class Routes(BaseModel):
    route : Literal["Valid","Reject"] = Field(description="Return 'Valid' if the data is cleaned, else return 'Reject' to regenerate it.")



class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

class Data_Cleaning_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.router = llm.with_structured_output(Routes)

    def generate_cleaning_code(self, state: PythonAnalystState) -> dict:
        if "raw_data" not in state or not state["raw_data"]:
            raise ValueError("Raw data not found or empty in state")
        if "question" not in state or not state["question"]:
            raise ValueError("User question not found in state")

        # build sample_text from raw_data; ensure columns shown as strings
        sample_parts = []
        for i, df in enumerate(state["raw_data"], start=1):
            if isinstance(df, pd.DataFrame):
                sample_df = dynamic_sample(df)
                # ensure column names are cast to str for the sample display
                sample_df = sample_df.rename(columns={c: str(c) for c in sample_df.columns})
                sample_parts.append(f"File {i} Sample:\n{sample_df.to_string(index=False)}")
            else:
                raise ValueError(f"Item {i} in raw_data is not a valid DataFrame")
        sample_text = "\n\n".join(sample_parts) if sample_parts else "No sample available."

        # Prompt: keep original column names (no renaming). Ask for function named clean_data.
        unified_prompt = PromptTemplate(
            template=(
                "You are a professional data cleaning agent. Think step-by-step to design the most robust and accurate\n"
                "Python function named exactly `clean_data(df)` for strict, production-grade cleaning of a pandas DataFrame.\n\n"
                "Important: DO NOT rename or normalize column names. Keep original column names as-is.\n\n"
                "Reason through each step before coding. Ensure:\n"
                "- Correct handling of missing values, duplicates, types, outliers, inconsistencies, and irrelevant data.\n"
                "- Code is clean, efficient, and executable.\n\n"
                "Sample data:\n{sample_text}\n\n"
                "User question:\n{user_question}\n\n"
                "Cleaning steps to implement (behavioral requirements):\n"
                "1. Drop all rows with any missing values (NaNs).\n"
                "2. Remove all exact duplicate rows.\n"
                "3. DO NOT change column names or case; preserve them exactly as in the DataFrame.\n"
                "4. Detect & handle outliers in numeric columns using the IQR method (clip values).\n"
                "5. Fix inconsistent values in object columns: strip whitespace, unify case if appropriate, normalize common categories (e.g., 'Yes', 'YES' -> 'yes').\n"
                "6. Remove irrelevant columns (entirely empty or constant).\n"
                "7. Reset the index after cleaning.\n\n"
                "At the start, import:\n- pandas as pd\n- numpy as np\n\n"
                "Output in EXACT format: a single fenced python block containing the function definition only.\n\n"
                "```python\n"
                "def clean_data(df):\n"
                "    ...\n"
                "```\n"
            ),
            input_variables=["sample_text", "user_question"]
        )

        chain = unified_prompt | self.llm | PythonOutputParser()
        raw = chain.invoke({
            "sample_text": sample_text,
            "user_question": state["question"]
        })

        self.logger.info("Generated cleaning code length=%d", len(raw) if raw else 0)
        return {"cleaning_code": raw}

    def execute_cleaning_code(self, state: PythonAnalystState) -> dict:
        """
        Execute generated cleaning code, but first cast integer columns to float64 on the copy
        passed into the generated clean_data(df) function to avoid 'incompatible dtype' warnings/errors.
        """
        import numpy as np
        import warnings
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if "raw_data" not in state or not state["raw_data"]:
            raise ValueError("Raw data not found or empty in state")
        if "cleaning_code" not in state or not state["cleaning_code"]:
            raise ValueError("Cleaning code not found in state")

        code = state["cleaning_code"]
        cleaned_dfs: List[pd.DataFrame] = []

        for i, df in enumerate(state["raw_data"], start=1):
            if not isinstance(df, pd.DataFrame):
                logger.warning("Item %s in raw_data is not a DataFrame, skipping...", i)
                continue

            # Execute in isolated namespace to obtain the cleaning function
            ns: dict = {}
            try:
                exec(code, ns, ns)
            except Exception as e:
                logger.error("Error executing cleaning code for DataFrame %s: %s", i, str(e))
                logger.info("Appending original DataFrame %s due to exec error", i)
                cleaned_dfs.append(df)
                continue

            # Find function named clean_data first
            cleaning_func = ns.get("clean_data")
            if not callable(cleaning_func):
                # fallback to any callable found
                for val in ns.values():
                    if callable(val):
                        cleaning_func = val
                        break

            if not callable(cleaning_func):
                logger.error("No callable cleaning function found in executed code for DataFrame %s", i)
                cleaned_dfs.append(df)
                continue

            try:
                df_copy = df.copy()
                int_cols = []
                try:
                    nullable_ints = df_copy.select_dtypes(include=[pd.Int64Dtype()]).columns.tolist()
                except Exception:
                    nullable_ints = []
                numpy_ints = df_copy.select_dtypes(include=[np.integer, "int64", "int32", "int16", "int8", "uint64", "uint32"]).columns.tolist()
                for c in (nullable_ints + numpy_ints):
                    if c not in int_cols:
                        int_cols.append(c)

                if int_cols:
                    logger.info("Casting integer-like columns to float64 before running cleaning function: %s", int_cols)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            df_copy[int_cols] = df_copy[int_cols].astype("float64")
                    except Exception as cast_exc:
                        logger.warning("Casting int->float failed for columns %s: %s", int_cols, cast_exc)

                cleaned = cleaning_func(df_copy)

                if not isinstance(cleaned, pd.DataFrame):
                    raise ValueError("Cleaning function did not return a pandas DataFrame")

                cleaned_dfs.append(cleaned)
                logger.info("Successfully cleaned DataFrame %s", i)

            except Exception as e:
                logger.error("Error running cleaning function on DataFrame %s: %s", i, str(e))
                logger.info("Appending original DataFrame %s due to runtime error", i)
                cleaned_dfs.append(df)

        return {"cleaned_data": cleaned_dfs}
    
    def check(self,state:PythonAnalystState):
        cleaned_preview = []
        for i, df in enumerate(state.get("cleaned_data", []), start=1):
            if isinstance(df, pd.DataFrame):
                preview = dynamic_sample(df).to_string(index=False)
                cleaned_preview.append(f"Table {i}:\n{preview}")
        cleaned_summary = "\n\n".join(cleaned_preview)
        prompt = PromptTemplate(template=("You are a data quality inspector.\n\n"
        "Given this cleaned data: {cleaned_data}\n\n"
        "Determine whether it is properly cleaned.\n"
        "Return only one of the following:\n"
        "- 'Valid' if the data looks clean\n"
        "- 'Reject' if it still looks dirty and needs cleaning again."),
                                input_variables=["cleaned_data"])
        chain = prompt | self.router
        response = chain.invoke({"cleaned_data":cleaned_summary})
        return {"cleaned_or_not":response.route}
    
    def next_route(self, state: PythonAnalystState):
        logger.info(f"[Routing Decision] Cleaned status = {state['cleaned_or_not']}")
                    
        if state["cleaned_or_not"] == "Valid":
            return "Valid"
        else:
            return "Reject"
        



