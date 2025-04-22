import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import random
from langchain_core.runnables import chain
import uuid
from pandas_agent import create_pandas_ai_agent, analyze_data, load_data


st.set_page_config(layout="wide")

def initialize_streamlit():
    """Initializes the Streamlit UI, including title, caption, and session state."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your research.")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial greeting from AI
        st.session_state.messages.append(
            AIMessage(content="Hello! I'm here to help you with your dissertation. How can I assist you today? Feel free to ask about brainstorming ideas, structuring chapters, finding resources, or anything else!")
        )

    # Initialize the agent prompt in session state
    if "agent_prompt" not in st.session_state:
        # Load the system prompt as instructions:
        try:
            with open('esi_agent_instruction.md', 'r') as f:
                instruction = f.read()
        except FileNotFoundError:
            st.error("Could not find esi_agent_instruction.md. Please ensure it is in the correct directory.")
            instruction = ""  # Provide a default value to avoid errors

        # st.session_state.agent_prompt = ChatPromptTemplate.from_messages(prompt_messages)

    # --- Data Loading State (Keep for handle_user_input) ---
    # Note: These might be better placed elsewhere if handle_user_input is refactored
    if "loaded_df" not in st.session_state:
        st.session_state.loaded_df = None
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None


def display_chat_messages():
    """Displays chat messages from history."""
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

def process_user_input(agent_executor, llm, prompt):
    """Processes user input for the Dissertation Agent and generates AI response."""
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response using the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the agent_executor from session state (might include uploaded file tool)
            current_agent_executor = agent_executor
            try:
                response = current_agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages[:-1] # Pass history excluding the current user prompt
                })
                ai_response_content = response.get("output", "Sorry, I encountered an error getting the response.")
            except Exception as e:
                 ai_response_content = f"An error occurred while processing your request: {e}"
                 st.error(ai_response_content) # Display error in UI as well

            st.markdown(ai_response_content)

    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(content=ai_response_content))

    return ai_response_content


# --- Restored Input Handling Logic ---

def handle_user_input(agent_executor, llm):
    """Handles user input using st.chat_input and agent selection."""
    agent_type = st.selectbox("Choose an agent:", ["Dissertation Agent", "Data Analysis Agent"], key="agent_selector")

    if agent_type == "Dissertation Agent":
        # Use chat_input for the dissertation agent
        if prompt := st.chat_input("What's on your mind regarding your dissertation?"):
            process_user_input(agent_executor, llm, prompt) # Use the main agent executor

    elif agent_type == "Data Analysis Agent":
        # File uploader for data analysis
        uploaded_file = st.file_uploader(
            "Upload a CSV, Excel, RData, or SAV file for analysis",
            type=["csv", "xlsx", "xls", "rda", "rdata", "sav"],
            key="data_uploader"
        )

        if uploaded_file is not None:
            # Check if it's a new file or the same one to avoid reloading unnecessarily
            if st.session_state.last_uploaded_filename != uploaded_file.name:
                with st.spinner(f"Loading '{uploaded_file.name}'..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.session_state.loaded_df = df
                        st.session_state.last_uploaded_filename = uploaded_file.name
                        st.success(f"'{uploaded_file.name}' loaded successfully. Preview:")
                        st.dataframe(df.head())
                    else:
                        st.session_state.loaded_df = None # Clear if loading failed
                        st.session_state.last_uploaded_filename = None
            # Display preview if already loaded
            elif st.session_state.loaded_df is not None:
                 st.write("Current data preview:")
                 st.dataframe(st.session_state.loaded_df.head())

            # Only show chat input for analysis *after* data is loaded
            if st.session_state.loaded_df is not None:
                if analysis_prompt := st.chat_input("Enter your data analysis prompt:"):
                    df = st.session_state.loaded_df # Use the loaded df from session state
                    google_api_key = os.getenv("GOOGLE_API_KEY")
                    if not google_api_key:
                        st.error("GOOGLE_API_KEY not found.")
                    else:
                        pandas_ai_agent = create_pandas_ai_agent(google_api_key, df)
                        if pandas_ai_agent:
                            # Add user analysis request to chat history
                            st.session_state.messages.append(HumanMessage(content=f"Analysis request: {analysis_prompt}"))
                            with st.chat_message("user"):
                                st.markdown(f"Analysis request: {analysis_prompt}")

                            # Get analysis result and display it
                            with st.chat_message("assistant"):
                                with st.spinner("Analyzing data..."):
                                    response = analyze_data(pandas_ai_agent, df, analysis_prompt)
                                    st.markdown(response) # Display result directly
                                    # Add AI analysis response to chat history
                                    st.session_state.messages.append(AIMessage(content=response))
                        else:
                            st.error("Failed to initialize data analysis agent.")
        else:
             # Clear data if no file is uploaded
             st.session_state.loaded_df = None
             st.session_state.last_uploaded_filename = None


def display_sidebar():
    """Displays the sidebar with information about the app."""
    with st.sidebar:
        st.header("About")
        st.info("""ESI uses AI to help you navigate the dissertation process.
        It has access to some of the literature in your reading lists and also uses Search tools for web lookups.""")
        st.warning("⚠️ Remember: Always consult your official supervisor for final guidance and decisions.")
