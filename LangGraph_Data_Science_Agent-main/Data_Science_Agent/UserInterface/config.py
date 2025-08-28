from configparser import ConfigParser

class Config:

    def __init__(self,config_path = "C:/Users/Dalbir/Downloads/LangGraph_Data_Science_Agent/Data_Science_Agent/UserInterface/config.ini"):
        self.config = ConfigParser()
        self.config.read(config_path)
    
    def get_llms(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")

    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")
    
    def get_gemini_llm(self):
        return self.config["DEFAULT"].get("Gemini_MODEL_OPTIONS").split(", ")
    
    def get_qwen_llm(self):
        return self.config["DEFAULT"].get("QWEN_MODEL_OPTIONS").split(", ")
    
    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")