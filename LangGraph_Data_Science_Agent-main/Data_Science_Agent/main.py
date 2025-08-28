import os
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
from langsmith import Client
from streamlit.components.v1 import html

from Data_Science_Agent.LLM.gemini import GeminiLLM
from Data_Science_Agent.LLM.groq import GroqLLM
from Data_Science_Agent.GRAPH.Python_Analyst_Graph import Graph_Builder
from Data_Science_Agent.UserInterface.Display_Result import DisplayResultStreamlit
from Data_Science_Agent.UserInterface.Sidebar import SidebarUI

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Data Science Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_app():
    # --- Optional LangSmith Setup ---
    with st.sidebar:
        st.write("## LangSmith Configuration")
        langsmith_api_key = st.text_input("LangSmith API Key (Optional)", type="password")
        if langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGSMITH_PROJECT"] = "Data Science Agent"
            try:
                Client()
                st.success("✅ LangSmith connected!")
            except Exception as e:
                st.error(f"❌ LangSmith Error: {str(e)}")

    ui = SidebarUI()
    user_control_input = ui.Load_UI()
    user_message = st.chat_input("What would you like to analyze?")
    uploaded_files = user_control_input.get("file", [])
    dataframes = []
    for file in uploaded_files:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                elif file.name.endswith(".xlsx"):
                    df = pd.read_excel(file)
                elif file.name.endswith(".json"):
                    df = pd.read_json(file)
                else:
                    st.warning(f"⚠️ Unsupported file format: {file.name}")
                    continue

                # Check for empty or broken DataFrame
                if df.empty or len(df.columns) == 0:
                    st.error(f"❌ {file.name} was read, but contains no usable data (no columns or all empty). Skipping.")
                    continue

                st.success(f"✅ Loaded: {file.name} — Shape: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)

                dataframes.append(df)

            except Exception as e:
                st.error(f"❌ Failed to read {file.name}: {e}")

    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)

        try:
            llm_type = user_control_input.get("llm_type")
            if not llm_type:
                st.error("❌ No LLM selected. Please choose one in the sidebar.")
                return

            # --- LLM Initialization ---

            elif llm_type == "Google Gemini":
                api_key = user_control_input.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("❌ Google Gemini API Key is missing.")
                    return
                llm_object = GeminiLLM(user_contols_input=user_control_input)

            else:
                st.error("❌ Invalid LLM selected. Please choose Groq or Google Gemini.")
                return

            llm = llm_object.get_llm_model()

            # --- Use Case Selection ---
            usecase = user_control_input.get("selected_usecase")
            if not usecase:
                st.error("❌ No use case selected.")
                return

            if usecase == "Data Analyst Agent":
                try:
                    # --- Run the Graph ---
                    graph_builder = Graph_Builder(llm)
                    graph = graph_builder.setup_graph(usecase)
                    
                    DisplayResultStreamlit(usecase, graph, user_message, dataframes).display_result_on_ui()
                except Exception as e:
                    st.error("❌ Error in analysis pipeline.")
                    st.exception(e)
            else:
                st.error(f"❌ Unsupported use case: {usecase}")

        except Exception as e:
            st.error("❌ Fatal error during initialization.")
            st.exception(e)

