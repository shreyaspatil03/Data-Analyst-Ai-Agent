import os
from typing import List, Any, Optional, Dict
import streamlit as st
import pandas as pd
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState
import streamlit.components.v1 as components
import logging

logger = logging.getLogger(__name__)

class DisplayResultStreamlit:
    def __init__(self, usecase: str, graph: Any, user_message: str, raw_data: List[pd.DataFrame]):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.raw_data = raw_data

    def _extract_report_path_or_url(self, step: Dict) -> Optional[str]:
        if not step:
            return None

        candidates = []
        for key in ("profiling_report_url", "profiling_report", "profiling_report_path", "profiling_reports"):
            if key in step and step[key]:
                val = str(step[key]).strip()
                # If it's a markdown link like [REPORT](path), extract inner
                if val.startswith("[REPORT](") and val.endswith(")"):
                    val = val[len("[REPORT]("):-1].strip()
                candidates.append(val)

        # Prefer first http/https URL if present
        for val in candidates:
            if val.lower().startswith("http://") or val.lower().startswith("https://"):
                return val

        # Next, if any candidate is an absolute path that exists, return it
        for val in candidates:
            if os.path.isabs(val) and os.path.exists(val):
                return os.path.abspath(val)

        # If candidate looks like a relative path under cwd or ./reports or ./static/reports, resolve it
        for val in candidates:
            basename = os.path.basename(val)
            possible = [
                os.path.join(os.getcwd(), val),
                os.path.join(os.getcwd(), "reports", val),
                os.path.join(os.getcwd(), "static", "reports", val),
                os.path.join(os.getcwd(), "reports", basename),
                os.path.join(os.getcwd(), "static", "reports", basename),
                os.path.join(os.getcwd(), basename),
            ]
            for p in possible:
                if os.path.exists(p):
                    return os.path.abspath(p)

        return None

    def _get_image_path(self, item: Any) -> Optional[str]:
        """Normalize image item to a local path if possible."""
        if isinstance(item, dict):
            return item.get("path") or item.get("file") or None
        if isinstance(item, str):
            return item
        return None

    def display_result_on_ui(self):
        """Main entry: stream graph, collect items, and display in Streamlit UI."""
        if self.usecase != "Data Analyst Agent":
            return

        state = {
            "question": self.user_message,
            "raw_data": self.raw_data
        }

        with st.status("🔄 Processing analysis...", expanded=False) as status:
            final_answer = None
            visual_images: List[Any] = []
            final_result = None
            profiling_report_ref: Optional[str] = None  # can be URL or path
            shown_steps = set()

            try:
                # stream through graph steps
                for step in self.graph.stream(state, stream_mode="values"):
                    # basic progress messages
                    if "cleaning_code" in step and "cleaning_code" not in shown_steps:
                        status.write("🧹 Cleaning data...")
                        shown_steps.add("cleaning_code")

                    if "cleaned_data" in step and "cleaned_data" not in shown_steps:
                        status.write("✨ Data cleaned...")
                        shown_steps.add("cleaned_data")

                    if "eda_code" in step and "eda_code" not in shown_steps:
                        status.write("📊 Performing EDA...")
                        shown_steps.add("eda_code")

                    if "eda_result" in step and "eda_result" not in shown_steps:
                        status.write("📈 Processing EDA results...")
                        shown_steps.add("eda_result")

                    if "rca_suggestion" in step and "rca_suggestion" not in shown_steps:
                        status.write("🔍 Analyzing root causes...")
                        shown_steps.add("rca_suggestion")

                    if "answer" in step and "answer" not in shown_steps:
                        final_answer = step["answer"]
                        status.write("📝 Collecting insights...")
                        shown_steps.add("answer")

                    if "visual_images" in step and "visual_images" not in shown_steps:
                        visual_images = step["visual_images"]
                        status.write("🖼️ Loading visualizations...")
                        shown_steps.add("visual_images")

                    if "final_result" in step and "final_result" not in shown_steps:
                        final_result = step["final_result"]
                        status.write("✨ Preparing final summary...")
                        shown_steps.add("final_result")

                    # detect profiling report path/url (first found wins)
                    if not profiling_report_ref:
                        candidate = self._extract_report_path_or_url(step)
                        if candidate:
                            profiling_report_ref = candidate

                status.update(label="✅ Analysis complete!", state="complete")

            except Exception as e:
                st.error(f"❌ Error in pipeline execution: {e}")
                status.update(label="❌ Analysis failed", state="error")
                logger.exception("Error streaming graph: %s", e)
                return

        with st.container():
            st.markdown("## ✅ Final Report")

            if final_result:
                st.markdown(final_result, unsafe_allow_html=True)
            else:
                st.markdown("No textual result produced by the agent.")

            if visual_images:
                st.markdown("---")
                st.markdown("## 🖼️ Visualization")
                for idx, img_item in enumerate(visual_images, start=1):
                    path = self._get_image_path(img_item)
                    if path and os.path.exists(path):
                        try:
                            with open(path, "rb") as img_file:
                                st.image(img_file.read(), caption=f"Image {idx}", width=750)
                        except Exception as e:
                            st.warning(f"⚠️ Failed to open image {path}: {e}")
                    else:
                        st.warning(f"⚠️ Image not found or invalid: {img_item}")

            # --- Pandas Profiling Report ---
            if profiling_report_ref:
                st.markdown("---")
                st.markdown("## 📑 EDA Report")

                if profiling_report_ref.startswith("http://") or profiling_report_ref.startswith("https://"):
                    report_url = profiling_report_ref
                elif os.path.exists(profiling_report_ref):
                    report_filename = os.path.basename(profiling_report_ref)
                    report_url = f"http://localhost:8001/{report_filename}"
                else:
                    st.warning(f"Profiling report found but file missing: {profiling_report_ref}")
                    report_url = None

                if report_url:
                    st.markdown(
                        """This profiling report provides a comprehensive overview of your dataset, detailing aspects such as the distribution and types of variables, correlations between features, and the extent of missing or incomplete data. It serves as a diagnostic summary to help you understand the dataset’s structure, quality, and key relationships before further analysis. """
                        "[🔗 **View Full Report**]({})".format(report_url),
                        unsafe_allow_html=True
                    )
