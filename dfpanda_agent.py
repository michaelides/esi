import pandas as pd
import io
import os
import traceback # Import traceback for detailed error logging
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor # Import for type hinting
from langchain_core.language_models import BaseChatModel # Import for type hinting
import streamlit as st # Import streamlit for st.error
import matplotlib.pyplot as plt # Import matplotlib

# Define the path where plots should be saved
PLOT_SAVE_DIR = "./exports/charts"
PLOT_SAVE_PATH = os.path.join(PLOT_SAVE_DIR, "temp_chart.png")

# Ensure the plot save directory exists
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)


def load_data(uploaded_file):
    """
    Loads data from an uploaded file (CSV, Excel, RData, SAV) into a pandas DataFrame.
    Handles different file types based on extension.
    """
    filename = uploaded_file.name
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == ".xlsx" or file_extension == ".xls":
            df = pd.read_excel(uploaded_file)
        elif file_extension == ".rda" or file_extension == ".rdata":
            try:
                import pyreadr
                data = pyreadr.read_r(uploaded_file)
                # pyreadr returns an OrderedDict, access the first item's value
                # Assuming the main data object is the first one
                df = next(iter(data.values()))
            except ImportError:
                st.error("Error: 'pyreadr' library not found. Cannot read .rda/.rdata files.")
                st.info("Please install it: pip install pyreadr")
                return None
            except Exception as e:
                st.error(f"Error reading .rda/.rdata file '{filename}': {e}")
                return None
        elif file_extension == ".sav":
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sav(uploaded_file)
            except ImportError:
                st.error("Error: 'pyreadstat' library not found. Cannot read .sav files.")
                st.info("Please install it: pip install pyreadstat")
                return None
            except Exception as e:
                st.error(f"Error reading .sav file '{filename}': {e}")
                return None
        else:
            st.error(f"Unsupported file type: {filename}")
            return None

        # Basic cleaning/preparation (optional, can be expanded)
        # Convert column names to strings and strip whitespace
        df.columns = df.columns.astype(str).str.strip()

        return df

    except Exception as e:
        st.error(f"Error loading data from {filename}: {e}")
        return None

# Function to create the dfpanda agent
def create_dfpanda_agent(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    """
    Creates a LangChain agent specifically for interacting with a pandas DataFrame.
    This agent is named 'dfpanda'.

    Args:
        llm: The language model to use for the agent.
        df: The pandas DataFrame to analyze.

    Returns:
        An AgentExecutor instance configured for DataFrame analysis.
    """
    try:
        # create_pandas_dataframe_agent automatically sets up tools for DataFrame interaction
        # Added allow_dangerous_code=True to enable code execution
        # Added handle_parsing_errors=True to allow the agent to recover from LLM output errors
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True, # Set to True to see the agent's thought process
            allow_dangerous_code=True, # Explicitly allow code execution for data analysis
            handle_parsing_errors=True # Allow the agent to handle and attempt to fix parsing errors
            # include_df_in_prompt=True # Consider if you want the full df in the prompt (can be large)
        )
        return agent_executor
    except Exception as e:
        print(f"Error creating dfpanda agent: {e}")
        st.error(f"Error creating data analysis agent: {e}")
        return None

# Function to run analysis using the dfpanda agent
def run_dfpanda_analysis(agent_executor: AgentExecutor, prompt: str) -> str:
    """
    Runs a data analysis prompt using the dfpanda agent.

    Args:
        agent_executor: The initialized LangChain AgentExecutor for the DataFrame.
        prompt: The user's analysis prompt.

    Returns:
        The response from the agent as a string.
    """
    if agent_executor is None:
         return "Data analysis agent could not be initialized."

    # Add instructions to the prompt about saving plots
    # Tell the agent to save plots to PLOT_SAVE_PATH and NOT use plt.show()
    plot_instruction = f"""
If you generate a plot using matplotlib, save it to the file path '{PLOT_SAVE_PATH}' using `plt.savefig()` and do NOT call `plt.show()`.
After saving the plot, mention in your final answer that a plot was generated and saved to '{PLOT_SAVE_SAVE_PATH}'.
"""
    full_prompt = prompt + plot_instruction

    # Remove any existing plot file before running the analysis
    if os.path.exists(PLOT_SAVE_PATH):
        try:
            os.remove(PLOT_SAVE_PATH)
            print(f"DEBUG: Removed existing plot file: {PLOT_SAVE_PATH}")
        except Exception as e:
            print(f"DEBUG: Could not remove existing plot file {PLOT_SAVE_PATH}: {e}")


    try:
        # Log the prompt for debugging
        print(f"DEBUG: Sending prompt to dfpanda agent: {full_prompt}")
        # Invoke the agent with the user's prompt
        response = agent_executor.invoke({"input": full_prompt})
        # The output is typically in the 'output' key for AgentExecutor
        analysis_result = response.get('output', 'No output received from agent.')
        # Log the response type and value for debugging
        print(f"DEBUG: Received response from dfpanda agent: type={type(analysis_result)}, value='{analysis_result}'")

        # Ensure the result is a string before returning
        return str(analysis_result)

    except Exception as e:
        # Log the full traceback for detailed debugging
        traceback_str = traceback.format_exc()
        print(f"ERROR in run_dfpanda_analysis: {type(e).__name__} - {e}\n{traceback_str}")
        st.error(f"An error occurred during analysis: {type(e).__name__} - {e}. Check the application logs (console output) for more details.")
        # Return a more informative error message, including the exception type
        return f"An error occurred during analysis: {type(e).__name__} - {e}. Check the application logs (console output) for more details."


# Example usage (for testing dfpanda_agent.py directly)
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
    # agent_test = create_dfpanda_agent(llm_test, dummy_df)

    # Example of running analysis (requires agent_test)
    # if agent_test:
    #     analysis_result = run_dfpanda_analysis(agent_test, "summarize the data")
    #     print(analysis_result)

    print("dfpanda_agent.py functions defined.")
    print("load_data(uploaded_file)")
    print("create_dfpanda_agent(llm, df)")
    print("run_dfpanda_analysis(agent_executor, prompt)")
