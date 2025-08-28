import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiLLM:
    def __init__(self,user_contols_input):
        self.user_controls_input=user_contols_input

    def get_llm_model(self):
        try:
            gemini_api_key=self.user_controls_input["GOOGLE_API_KEY"]
            select_gemini_model=self.user_controls_input["select_gemini_model"]
            if gemini_api_key=='' and os.environ["GOOGLE_API_KEY"] =='':
                st.error("Please Enter the Groq API KEY")

            llm=ChatGoogleGenerativeAI(api_key=gemini_api_key,model=select_gemini_model)

        except Exception as e:
            raise ValueError(f"Error Ocuured With Exception : {e}")
        return llm