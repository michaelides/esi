import streamlit as st
from pandasai import SmartDataframe
from pandasai import SmartDataframe
import pandas as pd
import io
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def create_pandas_ai_agent(api_key: str):
    """Creates a PandasAI agent with the given API key."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)
    return SmartDataframe(llm)

def analyze_data(agent: SmartDataframe, data: pd.DataFrame, prompt: str):
    """Analyzes the data using the PandasAI agent and the given prompt."""
    try:
        response = agent.chat(prompt)
        return response
    except Exception as e:
        return f"An error occurred during analysis: {e}"

def load_data(uploaded_file):
    """Loads data from the uploaded file into a Pandas DataFrame."""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == ".xlsx" or file_extension == ".xls":
            df = pd.read_excel(uploaded_file)
        elif file_extension == ".rda" or file_extension == ".rdata":
            import pyreadr
            data = pyreadr.read_r(uploaded_file)
            # Assuming the first object in the R data file is the DataFrame
            df = data[None]  # Access the DataFrame using None key
        elif file_extension == ".sav":
            import pyreadstat
            df, meta = pyreadstat.read_sav(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV, Excel, RData, or SAV file.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
