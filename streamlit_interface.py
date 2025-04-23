import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import random
from langchain_core.runnables import chain
import uuid
import re # Import regex for parsing suggestions
# Updated import to use the new dfpanda_agent.py file
from dfpanda_agent import create_dfpanda_agent, run_dfpanda_analysis, load_data, PLOT_SAVE_PATH # Import PLOT_SAVE_PATH
from langchain_core.language_models import BaseChatModel # Import for type hinting
import time # Import time for potential small delay


st.set_page_config(layout="wide")

def initialize_streamlit():
    """Initializes the Streamlit UI, including title, caption, and session state."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your research.")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial greeting from AI - Made more casual
        st.session_state.messages.append(
            AIMessage(content="Hey there! Ready to tackle that dissertation? I'm here to help with brainstorming, structure, resources, or whatever you need. What's on your mind?")
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
    # --- Suggested Prompts State ---
    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = []
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = None
    # --- LLM Temperature State ---
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7 # Default temperature
    # --- Current Agent Temperature State (Used to track if agent needs re-init) ---
    # This is initialized in app.py when the agent is created


def display_chat_messages():
    """Displays chat messages from history."""
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Check if the message content is a string path to an image
                # This is a simple heuristic; a more robust solution might involve
                # storing image paths/data separately in session state.
                if isinstance(message.content, str) and message.content.startswith("PLOT_PATH:"):
                    plot_path = message.content.replace("PLOT_PATH:", "").strip()
                    if os.path.exists(plot_path):
                        try:
                            st.image(plot_path, caption="Generated Plot")
                            # Add download button for the plot
                            with open(plot_path, "rb") as file:
                                st.download_button(
                                    label="Download Plot",
                                    data=file,
                                    file_name=os.path.basename(plot_path),
                                    mime="image/png" # Assuming plots are PNGs
                                )
                            # Optionally remove the file after displaying
                            # os.remove(plot_path) # Decide if you want to keep or remove the file
                        except Exception as e:
                            st.warning(f"Could not display or offer download for plot from {plot_path}: {e}")
                    # else: # Removed this else block to avoid showing the raw path if file is missing
                    #      st.warning(f"Plot file not found at {plot_path}")
                else:
                    st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

def process_user_input(agent_executor, llm: BaseChatModel, prompt: str):
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

    # Clear previous suggestions after processing is done for this turn
    st.session_state.suggested_prompts = []

    return ai_response_content


# --- Suggested Prompts Generation ---

def generate_suggested_prompts(llm: BaseChatModel, chat_history):
    """Generates suggested follow-up prompts using the LLM."""
    if not chat_history:
        return [] # No history, no suggestions

    # Create a concise history string (e.g., last 4 messages)
    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in chat_history[-4:]])

    # Simple prompt asking for 4 follow-up questions
    prompt_text = f"""Based on this recent conversation history:
{history_str}

Suggest exactly 4 concise and relevant follow-up questions a student might ask next about their dissertation.
Format the output as a numbered list, each question on a new line. Do not include any preamble or explanation.
Example:
1. Can you help me refine my research question?
2. Where can I find examples of literature reviews?
3. What are common pitfalls in methodology?
4. How do I structure the introduction chapter?
"""
    try:
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract suggestions using regex (handles potential numbering variations)
        # Looks for lines starting with a number, optional dot/parenthesis, and whitespace
        suggestions = re.findall(r"^\s*\d[\.\)]?\s*(.+)$", content, re.MULTILINE)

        # Return the first 4 suggestions found, stripping whitespace
        return [s.strip() for s in suggestions[:4]]

    except Exception as e:
        print(f"Error generating suggested prompts: {e}")
        return [] # Return empty list on error


# --- Callback for Suggestion Buttons ---
def suggestion_button_callback(prompt_text):
    """Sets the selected prompt and clears suggestions for rerun."""
    st.session_state.selected_prompt = prompt_text
    st.session_state.suggested_prompts = [] # Clear suggestions immediately


# --- Input Handling Logic with Suggestions ---

def handle_user_input(agent_executor, llm: BaseChatModel):
    """Handles user input via chat_input or suggestion buttons, and agent selection."""

    # --- Check for Selected Prompt from Button Click ---
    # We store this to check later if we need to process input
    prompt_from_suggestion_button = st.session_state.get("selected_prompt")
    prompt_processed_this_run = False # Flag to track if input was handled

    if prompt_from_suggestion_button:
        st.session_state.suggested_prompts = [] # Clear suggestions before processing
        prompt_to_process = prompt_from_suggestion_button
        st.session_state.selected_prompt = None # Consume the selected prompt
        # Ensure agent type is Dissertation Agent if a suggestion was clicked
        agent_type = "Dissertation Agent"
        st.session_state.agent_selector = "Dissertation Agent" # Update selectbox state
        process_user_input(agent_executor, llm, prompt_to_process)
        prompt_processed_this_run = True
        # Suggestions will be regenerated later in this run

    # --- Agent Selection ---
    # Use on_change to clear suggestions and data analysis state if agent type changes
    def clear_state_on_agent_change():
        # Clear suggestions if switching away from Dissertation Agent
        if st.session_state.agent_selector != "Dissertation Agent":
             st.session_state.suggested_prompts = []
        # Clear data analysis state if switching away from Data Analysis Agent
        if st.session_state.agent_selector != "Data Analysis Agent":
            st.session_state.loaded_df = None
            st.session_state.last_uploaded_filename = None

    agent_type = st.selectbox(
        "Choose an agent:", # Add the missing label argument
        options=["Dissertation Agent", "Data Analysis Agent"], # Use the 'options' keyword argument
        key="agent_selector",
        on_change=clear_state_on_agent_change
    )

    # --- Agent-Specific UI and Input ---

    # Display chat input based on agent type (always visible)
    prompt_from_chat_input = None
    if agent_type == "Dissertation Agent":
        prompt_from_chat_input = st.chat_input("What's on your mind regarding your dissertation?", key="dissertation_chat_input")
    elif agent_type == "Data Analysis Agent":
        # Clear suggestions explicitly when this agent is active
        st.session_state.suggested_prompts = []
        # File uploader for data analysis
        # Correct indentation for the file uploader call
        uploaded_file = st.file_uploader(
            "Upload a CSV, Excel, RData, or SAV file for analysis",
            type=["csv", "xlsx", "xls", "rda", "rdata", "sav"],
            key="data_uploader"
        )

        # --- Correct indentation for this block ---
        if uploaded_file is not None:
            # Check if it's a new file or the same one to avoid reloading unnecessarily
            if st.session_state.last_uploaded_filename != uploaded_file.name:
                with st.spinner(f"Loading '{uploaded_file.name}'..."):
                        # Indent the following lines to be inside the 'with' block
                        df = load_data(uploaded_file)
                        if df is not None:
                            st.session_state.loaded_df = df
                            st.session_state.last_uploaded_filename = uploaded_file.name
                            st.success(f"'{uploaded_file.name}' loaded successfully. Preview:")
                            st.dataframe(df.head())
                        else:
                            st.session_state.loaded_df = None # Clear if loading failed
                            # Indent this line correctly inside the else block
                            st.session_state.last_uploaded_filename = None
            # Display preview if already loaded
            elif st.session_state.loaded_df is not None:
                 st.write("Current data preview:")
                 st.dataframe(st.session_state.loaded_df.head())

            # Only show chat input for analysis *after* data is loaded
            # Unindent this block to align with the if/elif above (lines 200/214)
            if st.session_state.loaded_df is not None:
                # Use a different key for the data analysis chat input
                analysis_prompt = st.chat_input("Enter your data analysis prompt:", key="data_analysis_chat_input")
                # Assign to prompt_from_chat_input if a value was entered
                if analysis_prompt:
                     prompt_from_chat_input = analysis_prompt


        else:
             # Clear data if no file is uploaded
             st.session_state.loaded_df = None
             st.session_state.last_uploaded_filename = None

    # --- Process Input (only if button wasn't clicked AND chat input has value) ---
    if not prompt_from_suggestion_button and prompt_from_chat_input:
        if agent_type == "Dissertation Agent":
            st.session_state.suggested_prompts = [] # Clear suggestions before processing
            process_user_input(agent_executor, llm, prompt_from_chat_input)
            prompt_processed_this_run = True
        elif agent_type == "Data Analysis Agent" and st.session_state.loaded_df is not None:
            # Data analysis processing logic moved here
            df = st.session_state.loaded_df
            # Removed GOOGLE_API_KEY check as create_pandas_langchain_agent takes LLM directly
            # google_api_key = os.getenv("GOOGLE_API_KEY")
            # if not google_api_key:
            #     st.error("GOOGLE_API_KEY not found.")
            # else:
            # Use the llm object available in this function scope
            # Call the new LangChain agent creation function from dfpanda_agent
            pandas_agent_executor = create_dfpanda_agent(llm, df)
            # Corrected check: check if the agent object is not None
            if pandas_agent_executor is not None:
                # Add user analysis request to chat history
                st.session_state.messages.append(HumanMessage(content=f"Analysis request: {prompt_from_chat_input}"))
                with st.chat_message("user"):
                    st.markdown(f"Analysis request: {prompt_from_chat_input}")

                # Get analysis result and display it
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data..."):
                        # Call the new run_dfpanda_analysis function
                        response = run_dfpanda_analysis(pandas_agent_executor, prompt_from_chat_input)
                        # Convert the response to a string before displaying and adding to history
                        response_str = str(response)
                        st.markdown(response_str) # Display result directly
                        # Add AI analysis response to chat history
                        st.session_state.messages.append(AIMessage(content=response_str)) # Use the string version

                        # --- Check for and display plot ---
                        # Give the file system a moment to catch up
                        time.sleep(0.1) # Small delay
                        if os.path.exists(PLOT_SAVE_PATH):
                            try:
                                st.image(PLOT_SAVE_PATH, caption="Generated Plot")
                                # Add download button for the plot
                                with open(PLOT_SAVE_PATH, "rb") as file:
                                    st.download_button(
                                        label="Download Plot",
                                        data=file,
                                        file_name=os.path.basename(PLOT_SAVE_PATH),
                                        mime="image/png" # Assuming plots are PNGs
                                    )
                                # Add a placeholder message to history indicating a plot was shown
                                # This helps the chat history display the plot on rerun
                                st.session_state.messages.append(AIMessage(content=f"PLOT_PATH:{PLOT_SAVE_PATH}"))
                                # Optionally remove the file after displaying
                                # os.remove(PLOT_SAVE_PATH) # Decide if you want to keep or remove the file
                            except Exception as e:
                                st.warning(f"Could not display or offer download for plot: {e}")
                                st.session_state.messages.append(AIMessage(content=f"Could not display or offer download for plot: {e}"))
                        # --- End plot check ---

                        prompt_processed_this_run = True # Mark that an interaction happened
            else:
                st.error("Failed to initialize data analysis agent.")

    # --- Generate and Display Suggestions (only for Dissertation Agent) ---
    # Generate suggestions if the agent is Dissertation and no suggestions currently exist
    # This runs after any potential input processing within this script run.
    if agent_type == "Dissertation Agent" and not st.session_state.suggested_prompts:
         # Check if there's history to base suggestions on
         if st.session_state.messages:
             with st.spinner("Generating suggestions..."):
                 st.session_state.suggested_prompts = generate_suggested_prompts(llm, st.session_state.messages)

    # Display suggestions if they exist (will only be true if agent is Dissertation)
    if st.session_state.suggested_prompts:
        st.write("Suggestions:") # Add a label
        # Ensure we don't try to create 0 columns if suggestions somehow become empty after generation
        num_suggestions = len(st.session_state.suggested_prompts)
        if num_suggestions > 0:
            cols = st.columns(num_suggestions)
        for i, suggestion in enumerate(st.session_state.suggested_prompts):
            with cols[i]:
                st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    on_click=suggestion_button_callback,
                    args=(suggestion,) # Pass suggestion text to callback
                )


def display_sidebar():
    """Displays the sidebar with information about the app."""
    with st.sidebar:
        st.header("About")
        st.info("""ESI uses AI to help you navigate the dissertation process.
        It has access to some of the literature in your reading lists and also uses Search tools for web lookups.""")
        st.warning("⚠️ Remember: Always consult your official supervisor for final guidance and decisions.")

        # Add the creativity slider
        st.session_state.llm_temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=2.0, # Changed max value to 2.0
            value=st.session_state.get("llm_temperature", 0.7), # Use session state value, default if not set
            step=0.05,
            help="Controls the randomness of the AI's responses. Higher values mean more creative/random."
        )
