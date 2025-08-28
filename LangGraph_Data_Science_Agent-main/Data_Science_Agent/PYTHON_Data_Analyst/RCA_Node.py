from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
import pandas as pd
from typing import Dict

def dynamic_sample(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    frac = 1.0 if n < 10_000 else 0.1 if n < 100_000 else 0.03 if n < 1_000_000 else 0.01
    return df.sample(frac=frac, random_state=42)

class RCA_Node:
    def __init__(self, llm):
        self.llm = llm

    def rca_node(self, state: PythonAnalystState):
        prompt = PromptTemplate(
            template="""
        You are a senior data analyst tasked with performing Root Cause and Recommendation Analysis (RCA).

        User Question: "{user_query}"

        EDA Summary:
        {eda_result}

        ---

        🎯 Task:
        - Confirm if the user's concern is true using data.
        - Identify top 3–5 contributing factors.
        - Highlight affected segments/timeframes.
        - Propose 2–3 actionable recommendations.

        ---

        ✍️ Guidelines:
        - Be sharp, concise, and specific.
        - Avoid vague or generic language.
        - Use short bullet points with strong verbs and clear metrics.
        - Avoid repeating the user query.

        ---

        📌 Output (markdown, strict format):
        ### ✅ Root Cause Summary
        - (Short 1-liner summary)

        ### 🔍 Contributing Factors
        - Bullet 1 (crisp)
        - Bullet 2 (data-backed)
        - Bullet 3 (sharp)

        ### 📊 Segment/Group Focus
        - Bullet 1 (if applicable)
        - Bullet 2

        ### ⚠️ Data Limitations
        - Bullet 1
        - Bullet 2

        ### 💡 Actionable Recommendations
        - Recommendation 1 (short + direct)
        - Recommendation 2
        """,
            input_variables=["user_query", "eda_result"]
        )


        eda_summary = state.get("eda_result", "")

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
                "user_query": state.get("question", ""),
                "eda_result": eda_summary,
            })

        print("RCA Done")
        return {"rca_suggestion": response}
