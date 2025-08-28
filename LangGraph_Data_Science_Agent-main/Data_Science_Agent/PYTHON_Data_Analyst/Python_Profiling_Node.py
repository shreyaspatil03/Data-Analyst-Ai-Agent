# report_generator.py
import os
import uuid
import threading
import socket
import http.server
import socketserver
from functools import partial
from typing import Optional

import pandas as pd
from ydata_profiling import ProfileReport
from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

class Report:
    def __init__(self, llm=None, port: int = 8001, reports_dir: str = "reports"):
        """
        llm: unused here but kept for signature compatibility.
        port: port to serve the reports folder on (http://localhost:<port>/filename.html).
        reports_dir: folder where HTML reports will be saved.
        """
        self.llm = llm
        self.port = port
        self.reports_dir = os.path.abspath(reports_dir)
        os.makedirs(self.reports_dir, exist_ok=True)
        self._server_thread = None
        self._ensure_server_running()

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                return True
            except Exception:
                return False

    def _serve_reports(self, port: int, directory: str):
        # Use a handler that serves the given directory (Python 3.7+ supports directory param)
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
        # Allow reuse address for quick restarts
        class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True

        with ThreadingTCPServer(("0.0.0.0", port), handler) as httpd:
            httpd.serve_forever()

    def _ensure_server_running(self):
        # Start server in a daemon thread if not already listening on the port
        if self._is_port_in_use(self.port):
            return  # already running (could be another process)
        if self._server_thread and self._server_thread.is_alive():
            return

        t = threading.Thread(target=self._serve_reports, args=(self.port, self.reports_dir), daemon=True)
        t.start()
        self._server_thread = t

    def pandas_report(self, state: PythonAnalystState) -> dict:
        """
        Generate a profiling HTML for the first valid DataFrame and return the HTTP URL.
        Returns: {"profiling_report_url": "http://localhost:<port>/profiling_report_xxx.html"} or error info.
        """
        raw_dfs = state.get("raw_data", [])
        if not raw_dfs:
            return {"profiling_report_url": None, "error": "No data available"}

        # ensure server running
        self._ensure_server_running()

        for i, df in enumerate(raw_dfs, start=1):
            if not isinstance(df, pd.DataFrame):
                continue
            try:
                filename = f"profiling_report_{uuid.uuid4().hex[:8]}.html"
                abs_path = os.path.join(self.reports_dir, filename)

                # Create profile (minimal=True for speed — change to False for full report)
                profile = ProfileReport(df, title=f"Pandas Profiling Report {i}", minimal=True)
                profile.to_file(abs_path)

                # Construct URL to served file (use localhost)
                url = f"http://localhost:{self.port}/{filename}"

                return {"profiling_report_url": url}
            except Exception as e:
                return {"profiling_report_url": None, "error": str(e)}

        return {"profiling_report_url": None, "error": "No DataFrame found"}
