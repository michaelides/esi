import pandas as pd
import io
import os
# Removed pandasai import
# import sklearn # Removed sklearn import
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor # Import for type hinting
from langchain_core.language_models import BaseChatModel # Import for type hinting
# Removed ChatGoogleGenerativeAI import as it's passed from app.py

def load_data(uploaded_file):
    """
    Loads data from an uploaded file (CSV, Excel, RData, SAV) into a pandas DataFrame.
    Handles different file types based on extension.
    """
    filename = uploaded_file.name
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(('.rda', '.rdata')):
            # RData files are tricky in Python. This is a placeholder.
            # You might need a specific library or conversion process.
            # For now, we'll raise an error or return None.
            print(f"Warning: RData file '{filename}' uploaded. Python does not natively support RData. Skipping.")
            return None
        elif filename.endswith('.sav'):
            # Requires pyreadstat or similar
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sav(uploaded_file)
            except ImportError:
                print("Error: 'pyreadstat' library not found. Cannot read .sav files.")
                print("Please install it: pip install pyreadstat")
                return None
            except Exception as e:
                print(f"Error reading .sav file '{filename}': {e}")
                return None
        else:
            print(f"Unsupported file type: {filename}")
            return None

        # Basic cleaning/preparation (optional, can be expanded)
        # Convert column names to strings and strip whitespace
        df.columns = df.columns.astype(str).str.strip()

        return df

    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

# Renamed function and updated signature
def create_pandas_langchain_agent(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    """
    Creates a LangChain agent specifically for interacting with a pandas DataFrame.

    Args:
        llm: The language model to use for the agent.
        df: The pandas DataFrame to analyze.

    Returns:
        An AgentExecutor instance configured for DataFrame analysis.
    """
    try:
        # create_pandas_dataframe_agent automatically sets up tools for DataFrame interaction
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True, # Set to True to see the agent's thought process
            # include_df_in_prompt=True # Consider if you want the full df in the prompt (can be large)
        )
        return agent_executor
    except Exception as e:
        print(f"Error creating pandas LangChain agent: {e}")
        return None

# Renamed function and updated signature
def run_pandas_analysis(agent_executor: AgentExecutor, prompt: str) -> str:
    """
    Runs a data analysis prompt using the pandas LangChain agent.

    Args:
        agent_executor: The initialized LangChain AgentExecutor for the DataFrame.
        prompt: The user's analysis prompt.

    Returns:
        The response from the agent as a string.
    """
    if agent_executor is None:
         return "Data analysis agent could not be initialized."
    try:
        # Log the prompt for debugging
        print(f"DEBUG: Sending prompt to LangChain Pandas Agent: {prompt}")
        # Invoke the agent with the user's prompt
        response = agent_executor.invoke({"input": prompt})
        # The output is typically in the 'output' key for AgentExecutor
        analysis_result = response.get('output', 'No output received from agent.')
        # Log the response type and value for debugging
        print(f"DEBUG: Received response from LangChain Pandas Agent: type={type(analysis_result)}, value='{analysis_result}'")

        # Ensure the result is a string before returning
        return str(analysis_result)

    except Exception as e:
        # Log the full traceback for detailed debugging
        import traceback
        traceback_str = traceback.format_exc()
        print(f"ERROR in run_pandas_analysis: {type(e).__name__} - {e}\n{traceback_str}")
        # Return a more informative error message, including the exception type
        return f"An error occurred during analysis: {type(e).__name__} - {e}. Check the application logs (console output) for more details."


# Example usage (for testing pandas_agent.py directly)
if __name__ == '__main__':
    # Create a dummy DataFrame for testing
    data = {'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]}
    dummy_df = pd.DataFrame(data)

    # Note: To run this __main__ block, you would need to initialize an LLM
    # and potentially mock the uploaded_file object for load_data.
    # This block is primarily for demonstrating the functions' signatures.

    # Example of creating agent (requires LLM)
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # llm_test = ChatGoogleGenerativeAI(model="gemini-pro") # Replace with your LLM setup
    # agent_test = create_pandas_langchain_agent(llm_test, dummy_df)

    # Example of running analysis (requires agent_test)
    # if agent_test:
    #     analysis_result = run_pandas_analysis(agent_test, "summarize the data")
    #     print(analysis_result)

    print("pandas_agent.py functions defined.")
    print("load_data(uploaded_file)")
    print("create_pandas_langchain_agent(llm, df)")
    print("run_pandas_analysis(agent_executor, prompt)")
