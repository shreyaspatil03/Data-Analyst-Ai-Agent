# Data Analyst AI Agent (LangGraph + Streamlit)

A Streamlit application that turns your tabular datasets into insights using an LLM-driven analysis graph built with LangGraph. Upload CSV/XLSX/JSON, ask a question, and the agent will clean data, run EDA, suggest root causes, generate visuals, and produce a final report. Optional Pandas Profiling reports are also generated and linked in the UI.

## Features
- Data upload (CSV, XLSX, JSON) with schema validation preview
- Automated pipeline via LangGraph state machine:
  - Cleaning code generation and execution
  - Exploratory Data Analysis (EDA) and result parsing
  - Root Cause Analysis (RCA) suggestions
  - Visualization code generation and auto-rendering
  - Pandas Profiling report (linked in UI when available)
  - Final insights summary
- LLM backends:
  - Google Gemini (wired in UI)
  - Groq (class present; can be wired similarly)
- Optional LangSmith tracing and project logging

## Tech Stack
- LangChain, LangGraph, LangSmith
- Streamlit UI
- Visualization: matplotlib, seaborn
- Profiling: ydata-profiling

## Repository Structure
```
LangGraph_Data_Science_Agent-main/
├─ app.py                          # Streamlit entrypoint
├─ requirements.txt
├─ Data_Science_Agent/
│  ├─ main.py                      # Streamlit app: UI wiring, file handling, graph run
│  ├─ GRAPH/Python_Analyst_Graph.py# Graph builder with nodes and edges
│  ├─ STATE/Python_Analyst_State.py# Shared graph state definition
│  ├─ LLM/
│  │  ├─ gemini.py                 # Google Gemini LLM wrapper
│  │  └─ groq.py                   # Groq LLM wrapper
│  └─ UserInterface/
│     ├─ Sidebar.py                # Sidebar controls (LLM, data upload)
│     ├─ Display_Result.py         # Streaming results, images, profiling link
│     └─ config.{py,ini}           # UI config
```

## Requirements
- Python 3.10+ recommended
- See `requirements.txt` for full list (LangChain, LangGraph, Streamlit, profiling, viz libs)

Install dependencies:
```bash
pip install -r LangGraph_Data_Science_Agent-main/requirements.txt
```

## Environment Variables
- Required for Google Gemini (currently wired):
  - `GOOGLE_API_KEY`
- Optional LangSmith (for tracing and runs):
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_TRACING_V2=true`
  - `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`
  - `LANGSMITH_PROJECT=Data Science Agent`

You can also paste the Gemini API key directly into the sidebar field.

## Running the App
Launch Streamlit pointing to the app entry:
```bash
streamlit run LangGraph_Data_Science_Agent-main/app.py
```
Then open the provided local URL in your browser (typically http://localhost:8501).

## Usage
1. In the sidebar:
   - Select Usecase: "Data Analyst Agent"
   - Select LLM: "Google Gemini" (enter your API key and pick a model)
   - Upload one or more dataset files (CSV/XLSX/JSON)
2. In the chat input, describe what you want to analyze.
3. The app will stream progress through cleaning, EDA, RCA, visuals, and summary.
4. If a Pandas Profiling report is generated, a link appears under "EDA Report".

## Architecture Overview
- `Python_Analyst_State.py`: Defines the state passed through the graph (question, raw/cleaned data, code snippets, results, images, profiling URL, final result, etc.).
- `Python_Analyst_Graph.py`: Builds a `StateGraph` with nodes:
  - `Clean_Code_Generator` → `Cleaning_Code_Executor` → `Check` (conditional retry)
  - `EDA_Analysis` → `EDA_Code_Executor` → `RCA_Node` → `Visual_Analysis` → `Visual_Code_Executor` → `Output`
  - `Pandas Profiling Report` (optional branch)
- `Display_Result.py`: Streams graph steps, shows progress/status, renders images if produced, and links a profiling report when detected.
- `Sidebar.py`: Collects user controls and file uploads, including Gemini API key and model selection.

## Notes
- LLM wiring: The UI currently enables Google Gemini. A Groq wrapper exists but is not yet wired in the sidebar flow. To add Groq support in the UI, mirror the Gemini code path in `Sidebar.py` and `Data_Science_Agent/main.py`.
- File formats: Unsupported uploads are skipped with a warning.
- Empty/broken datasets: The app validates uploaded DataFrames and skips empty ones.

## Troubleshooting
- No LLM selected: Ensure you selected "Google Gemini" and provided `GOOGLE_API_KEY`.
- Push/Permissions on GitHub: Ensure you have write access and are authenticated.
- Profiling report not found: The app tries to resolve both URLs and local paths. If a path is shown but file is missing, ensure the report file exists in the working directory or `./reports`.

## License
This project is provided as-is without a specific license file. Add a license if required by your usage.

## Acknowledgements
- Built with [LangGraph](https://python.langchain.com/docs/langgraph/), [LangChain](https://python.langchain.com/), and [Streamlit](https://streamlit.io/).