from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class Output_Node:
    def __init__(self, llm, report_base_url: str = None):
        """
        report_base_url: base HTTP URL that serves the report files, e.g. "http://localhost:8001".
                         If None, defaults to "http://localhost:8001".
                         Make sure a file server is serving the folder containing the report files.
        """
        self.llm = llm
        self.report_base_url = (report_base_url.rstrip("/") if report_base_url else "http://localhost:8001")

    def _make_report_url(self, profiling_report_value: str) -> str:
        """
        Convert whatever is in state['profiling_report'] into a clickable HTTP URL.
        - If it's already an http(s) URL, return as-is.
        - If it's a filesystem path, return report_base_url + "/" + basename(path).
        """
        if not profiling_report_value:
            return ""
        val = str(profiling_report_value).strip()

        # If already a URL
        if val.startswith("http://") or val.startswith("https://"):
            return val
        if val.startswith("[REPORT](") and val.endswith(")"):
            inner = val[len("[REPORT]("):-1]
            val = inner
        fn = os.path.basename(val)
        if not fn:
            return ""
        return f"{self.report_base_url}/{fn}"

    def _format_visual_paths(self, visual_images: List[dict]) -> str:

        if not visual_images:
            return "No visuals generated."

        lines = []
        for i, item in enumerate(visual_images, start=1):
            if isinstance(item, dict):
                if "path" in item:
                    lines.append(f"- **Image {i}**: Visualization saved (use the UI to open).")
                elif "error" in item:
                    lines.append(f"- **Image {i}**: Error — {item.get('error')}")
                else:
                    lines.append(f"- **Image {i}**: (unknown item)")
            else:
                # fallback for legacy list of strings
                lines.append(f"- **Image {i}**: {str(item)}")
        return "\n".join(lines)

    def output_parser(self, state: PythonAnalystState) -> dict:
        question = state.get("question", "")
        eda_result = state.get("eda_result", "No EDA results available.")
        rca_result = state.get("rca_suggestion", "No RCA available.")
        # Optional: visual plan or other context can be added if available
        visual_plan = state.get("visual_plan", "")

        final_summary_prompt = PromptTemplate(
        input_variables=["user_query", "eda_result", "rca_result"],
        template="""
    You are a senior data analyst.  
    Using the provided **EDA results** and **RCA insights**, generate a concise, well-structured final report.  
    The output must follow **exactly** the format and headings below.

    ---

    ### **1. Dataset Summary**  
    Write 2–3 sentences describing:  
    - Number of rows & columns  
    - Datatypes (numeric, categorical, etc.)  
    - Domain or context (if inferable from the data)  
    - Key anomalies or data quality issues (e.g., missing values, outliers, imbalance)

    ---

    ### **2. User Query Summary**  
    Write a 2–3 line summary of what is being analyzed based on the user's query.  
    Explicitly mention the **columns** referenced in the query and the main target or context if provided.  
    Do not start with phrases like "The user is asking" or "The user wants to know".  

    ---

    ### **3. Root Cause Analysis**  
    Write **3–4 bullet points** listing the most important patterns, trends, or anomalies from the data.  
    These should be **general insights**, not limited to the user’s query.

    ---

    ### **4. Insights**  
    Write **2–3 bullet points** that directly connect the findings to the user’s requested analysis.

    ---

    ### **5. Recommendations**  
    Write **2–3 actionable suggestions** for next steps, based on the findings.

    ---

    ### **6. Final Takeaway**  
    Write **1–2 sentences** summarizing the single most critical insight and why it matters.

    ---

    **Inputs for reference:**
    - User Query: {user_query}  
    - EDA Result: {eda_result}  
    - RCA Insights: {rca_result}  

    Do not include raw code or unnecessary text. Keep it professional and to the point.
    """
    )

        chain = final_summary_prompt | self.llm | StrOutputParser()
        inputs = {
            "user_query": question,
            "eda_result": str(eda_result),
            "rca_result": str(rca_result),
            "visual_plan": str(visual_plan),
        }

        response = chain.invoke(inputs)
        return {"final_result": response}
