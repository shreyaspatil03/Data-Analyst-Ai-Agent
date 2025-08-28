from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser,JsonOutputParser
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys
import matplotlib
import tempfile
import os
import uuid
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PythonOutputParser(BaseOutputParser):
    """Parser for extracting Python code from markdown code blocks."""
    
    def parse(self, text: str) -> str:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
def fix_palette_deprecation(code: str, default_color: str = "C0") -> str:
    """
    Fix Seaborn 'palette without hue' deprecation:
      - If x='col' present and hue not present, add hue=x and legend=False (keep palette).
      - Otherwise replace palette=... with color='<default_color>'.
    """
    pattern = r"(sns\.\w+\([^)]*?)palette\s*=\s*([^,)\n]+)([,)\n]?)"

    def repl(m):
        before, palette, trailing = m.groups()
        # find x='col' or x="col" inside the call prefix
        x_match = re.search(r"\bx\s*=\s*['\"]([^'\"]+)['\"]", before)
        if x_match and "hue=" not in before:
            col = x_match.group(1)
            trailing = trailing or ""
            return f"{before}hue='{col}', legend=False, palette={palette}{trailing}"
        # fallback: use single color
        quoted = f"'{default_color}'" if not (default_color.startswith("'") or default_color.startswith('"')) else default_color
        trailing = trailing or ""
        return f"{before}color={quoted}{trailing}"

    try:
        return re.sub(pattern, repl, code, flags=re.S)
    except Exception:
        return code

class Visual_Node:
    """Node for handling data visualization tasks."""
    
    def __init__(self, llm) -> None:
        self.llm = llm

    def generate_visual_code(self, state: PythonAnalystState) -> dict:
        if not state.get("cleaned_data") or not state.get("question"):
            raise ValueError("Missing cleaned_data or question")

        # Convert cleaned_data to readable column summary
        column_summary = "\n".join(
            [f"Table {i+1}: {', '.join(map(str, df.columns))}"
            for i, df in enumerate(state["cleaned_data"]) if isinstance(df, pd.DataFrame)]
        )

        # Step 1: Suggest visualizations
        suggestion_prompt = PromptTemplate(
            template="""
    You are an elite data visualization strategist. Your task is to design only the most critical visualization to help answer the user’s question.
    ---
    ### Inputs:
    - Cleaned Data (columns):  
    {cleaned_data}
    - User Query:  
    "{user_query}"
    - EDA Summary:  
    {eda_result}
    - RCA Summary:  
    {rca_result}
    ---
    ### Output Instructions:
    For **1 visualization**, provide:
    1. **Chart Title**
    2. **Chart Type** – (bar, line, scatter etc.)
    3. **X-Axis**
    4. **Y-Axis**
    5. **Short Description** – in plain English, explain what the chart will show.
    ---
    Output Format:
    ### Suggested Visualization:
    - **Title**: ...
    - **Type**: ...
    - **X**: ...
    - **Y**: ...
    - **Description**: ...
    """,
            input_variables=["user_query", "cleaned_data", "eda_result", "rca_result"]
        )

        suggestion_chain = suggestion_prompt | self.llm | StrOutputParser()
        visual_plan = suggestion_chain.invoke({
            "user_query": state["question"],
            "eda_result": state.get("eda_result", ""),
            "rca_result": state.get("rca_suggestion", ""),
            "cleaned_data": column_summary
        })

        # Step 2: Generate Python code for the visualization
        code_prompt = PromptTemplate(
            template="""
    You are a Python visualization engineer.

    Generate a clean, executable Python function for the visualization below:
    {visual_suggestion}
    ---
    Instructions:
    - Function name: `generate_visualizations(df)`
    - Put all imports inside the function
    - Implement exactly as specified
    - Add descriptive titles and axis labels
    - Call plt.show() (or fig.show() for Plotly) after each chart
    ---
    Rules:
    - No print(), return(), placeholders, markdown, or explanations
    - No hardcoded values — use only df columns
    ---
    Output Format:
    ```python
    def generate_visualizations(df):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Chart
        ...
        plt.show()
    """,
        input_variables=["visual_suggestion"]
        )

        code_chain = code_prompt | self.llm | PythonOutputParser()
        visual_code = code_chain.invoke({"visual_suggestion": visual_plan})

        return {
            "visual_plan": visual_plan,
            "visual_code": visual_code
        }

    
    def execute_visual_code(self, state: PythonAnalystState) -> dict:
        code = state.get("visual_code")
        if not code:
            raise ValueError("No visualization code found in state")
            
        cleaned_dfs = state.get("cleaned_data", [])
        if not cleaned_dfs:
            raise ValueError("No cleaned data found in state")
            
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

        code = fix_palette_deprecation(code)

        image_paths: List[Dict[str, Any]] = []
        sys.modules.setdefault("matplotlib", matplotlib)
        sys.modules.setdefault("matplotlib.pyplot", plt)
        for idx, df in enumerate(cleaned_dfs):
            if not isinstance(df, pd.DataFrame):
                logger.warning("Item %(idx)s in cleaned_data is not a DataFrame. Skipping.", 
                             {"idx": idx + 1})
                continue

            local_vars = {"df": df.copy()}
            original_show = plt.show

            def save_and_track():
                """Save current plot to a temporary file and track its path."""
                try:
                    temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.png")
                    plt.savefig(temp_file, bbox_inches="tight", dpi=300)
                    plt.close("all")
                    
                    # Verify the file was created successfully
                    if not os.path.exists(temp_file):
                        raise IOError("Failed to create image file")
                        
                    image_paths.append(temp_file)
                    logger.info("Successfully saved visualization to %(path)s", {"path": temp_file})
                    
                except (IOError, ValueError) as e:
                    logger.error("Failed to save image: %(error)s", {"error": str(e)})
                    image_paths.append({"error": f"Failed to save image: {str(e)}"})
                except Exception as e:
                    logger.error("Unexpected error while saving image: %(error)s", {"error": str(e)})
                    image_paths.append({"error": f"Unexpected error: {str(e)}"})

            try:
                exec(code, {}, local_vars)
                generate_func = next((val for val in local_vars.values() if callable(val)), None)
                if generate_func is None:
                    raise ValueError("No function found in generated code.")

                # Patch show to save plots
                plt.show = save_and_track
                generate_func(df)

            except Exception as e:
                logger.error(f"Error executing visualization on DataFrame {idx + 1}: {e}")
                image_paths.append({"error": str(e)})
            finally:
                plt.show = original_show

        return {"visual_images": image_paths}