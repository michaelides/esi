import streamlit as st
from pandasai import SmartDataframe
import pandas as pd
import io
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def create_pandas_ai_agent(api_key: str, df: pd.DataFrame):
    """Creates a PandasAI agent with the given API key."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)
    try:
        if not isinstance(df, pd.DataFrame):
            st.error("Data is not a valid Pandas DataFrame.")
            return None
        return SmartDataframe(df)
    except Exception as e:
        st.error(f"Error initializing SmartDataframe: {e}")
        return None

def analyze_data(agent: SmartDataframe, data: pd.DataFrame, prompt: str):
    """Analyzes the data using the PandasAI agent and the given prompt."""
    if agent is None:
        return "Data analysis agent could not be initialized."
    try:
        # Log the prompt for debugging
        print(f"DEBUG: Sending prompt to PandasAI: {prompt}")
        response = agent.chat(prompt)
        # Log the response type and value for debugging
        print(f"DEBUG: Received response from PandasAI: type={type(response)}, value='{response}'")
        # Check if the response is None, which might be unexpected
        if response is None:
            print("ERROR: PandasAI agent.chat() returned None.")
            return "PandasAI returned an empty response. This might indicate an internal issue. Please check the logs or try rephrasing your request."
        return response
    except Exception as e:
        # Log the full traceback for detailed debugging
        import traceback
        traceback_str = traceback.format_exc()
        print(f"ERROR in analyze_data: {type(e).__name__} - {e}\n{traceback_str}")
        # Return a more informative error message, including the exception type
        return f"An error occurred during analysis: {type(e).__name__} - {e}. Check the application logs (console output) for more details."

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
        return pd.DataFrame(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
