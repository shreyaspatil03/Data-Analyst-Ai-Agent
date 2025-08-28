from Data_Science_Agent.STATE.Python_Analyst_State import PythonAnalystState

from Data_Science_Agent.PYTHON_Data_Analyst.Data_Cleaning_Node import Data_Cleaning_Node
from Data_Science_Agent.PYTHON_Data_Analyst.EDA_Node import EDA_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Python_Profiling_Node import Report
from Data_Science_Agent.PYTHON_Data_Analyst.RCA_Node import RCA_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Visual_Node import Visual_Node
from Data_Science_Agent.PYTHON_Data_Analyst.Output_Node import Output_Node

from langgraph.graph import START , END , StateGraph

class Graph_Builder:

    def __init__(self,llm,langsmith_client=None):
        self.llm = llm
        self.langsmith_client = langsmith_client

    def py_graph(self):
        self.graph_builder = StateGraph(PythonAnalystState)

        cleaning_node = Data_Cleaning_Node(self.llm)
        eda_node = EDA_Node(self.llm)
        rca_node = RCA_Node(self.llm)
        visual_node = Visual_Node(self.llm)
        output_node = Output_Node(self.llm)
        report_node = Report(self.llm)

        self.graph_builder.add_node("Clean_Code_Generator", cleaning_node.generate_cleaning_code)
        self.graph_builder.add_node("Cleaning_Code_Executor", cleaning_node.execute_cleaning_code)
        self.graph_builder.add_node("Check",cleaning_node.check)
        self.graph_builder.add_node("EDA_Analysis", eda_node.perform_eda_analysis)
        self.graph_builder.add_node("EDA_Code_Executor", eda_node.execute_eda_code)
        self.graph_builder.add_node("RCA_Node", rca_node.rca_node)
        self.graph_builder.add_node("Visual_Analysis", visual_node.generate_visual_code)
        self.graph_builder.add_node("Visual_Code_Executor", visual_node.execute_visual_code)
        self.graph_builder.add_node("Pandas Profiling Report", report_node.pandas_report)
        self.graph_builder.add_node("Output", output_node.output_parser)

        self.graph_builder.add_edge(START, "Clean_Code_Generator")
        self.graph_builder.add_edge(START, "Pandas Profiling Report")
        self.graph_builder.add_edge("Clean_Code_Generator", "Cleaning_Code_Executor")
        self.graph_builder.add_edge("Cleaning_Code_Executor", "Check")
        self.graph_builder.add_conditional_edges("Check",cleaning_node.next_route,{"Valid":"EDA_Analysis","Reject":"Clean_Code_Generator"})
        self.graph_builder.add_edge("EDA_Analysis", "EDA_Code_Executor")
        self.graph_builder.add_edge("EDA_Code_Executor", "RCA_Node")
        self.graph_builder.add_edge("RCA_Node", "Visual_Analysis")
        self.graph_builder.add_edge("Visual_Analysis", "Visual_Code_Executor")
        self.graph_builder.add_edge("Visual_Code_Executor", "Output")
        self.graph_builder.add_edge("Output", END)
        self.graph_builder.add_edge("Pandas Profiling Report", END)

    def setup_graph(self,usecase : str):
        if usecase == "Data Analyst Agent":
            self.py_graph()
        return self.graph_builder.compile()