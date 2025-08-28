from typing_extensions import Annotated , TypedDict , Literal, Any,List
import pandas as pd
from typing import Union

class PythonAnalystState(TypedDict):

    question: str
    raw_data: List[pd.DataFrame]

    cleaning_code: str
    cleaned_data: List[pd.DataFrame]
    cleaned_or_not : str

    eda_code: str
    eda_result: str
    eda_recheck_suggestions : str
    profiling_report_url : str

    rca_suggestion: str

    visual_code: str
    visual_plan : str
    visual_images: List[Union[str, dict]]  
    
    final_result: str  