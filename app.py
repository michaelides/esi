import streamlit as st
import os
import time
import json
import re # Import regex module for parsing code blocks and markers
import uuid # New import for generating user IDs
import extra_streamlit_components as esc
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole # Import necessary types
import stui
# Import initialize_settings and alias it, and the new orchestrator agent creator
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Load environment variables
load_dotenv()

# Configure page settings - MUST be the first Streamlit command
st.set_page_config(
    page_title="ESI - ESI Scholarly Instructor", # Consistent title
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize cookie manager directly
cookies = esc.CookieManager(key="esi_cookie_manager")

# --- Constants and Configuration ---
# Update DB_PATH default to the new simple vector store persistence directory
# Use path relative to PROJECT_ROOT
SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE) # Resolve to absolute path at runtime
AGENT_SESSION_KEY = "esi_orchestrator_agent" # Key for storing orchestrator agent
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---" # Used by stui.py for display
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---" # Used by stui.py for display

# New: Directory for user chat memories
MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories")

# --- Cached LLM Settings Initialization ---
@st.cache_resource
def setup_global_llm_settings():
    """Initializes global LLM settings using st.cache_resource to run only once."""
    print("Initializing LLM settings (cached)...")
    try:
        initialize_agent_settings()
        print("LLM settings initialized (cached).")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize LLM settings. {e}")
        st.stop()

# --- Cached Agent Initialization ---
@st.cache_resource
def setup_agent():
    """Initializes the orchestrator agent using st.cache_resource to run only once."""
    print("Initializing orchestrator agent (cached)...")
    try:
        agent_instance = create_orchestrator_agent()
        print("Orchestrator agent object initialized (cached) successfully.")
        return agent_instance
    except Exception as e:
        print(f"Error initializing orchestrator agent (cached): {e}")
        st.error(f"Failed to initialize the AI agent. Please check configurations. Error: {e}")
        st.stop()

# --- Cached User ID Retrieval ---
@st.cache_resource
def get_cached_user_id():
    """Retrieves user ID from cookies or creates a new one, cached to run once per session."""
    user_id = cookies.get(cookie="user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        cookies.set(cookie="user_id", val=user_id)
        # cookies.set() triggers a rerun. The cached function will re-execute,
        # but on the next run, cookies.get() should return the newly set user_id.
        # The final return value will be cached.
        print(f"New user ID created and set (cached): {user_id}")
    else:
        print(f"Existing user ID retrieved (cached): {user_id}")
    return user_id

# --- Cached Chat History Loading ---
@st.cache_resource
def get_cached_chat_history(user_id: str) -> List[Dict[str, Any]]:
    """Loads chat history for a given user ID from a JSON file, cached to run once per session."""
    history = load_chat_history_from_file(user_id) # Call the non-cached helper
    if not history:
        # If history is empty, add initial greeting and save it
        history = [{"role": "assistant", "content": generate_llm_greeting()}]
        save_chat_history(user_id, history)
        print(f"New chat history initialized and saved for user {user_id} (cached).")
    else:
        print(f"Chat history loaded for user {user_id} (cached).")
    return history

# --- User ID and Memory Management Functions (non-cached helpers) ---
def get_user_memory_path(user_id: str) -> str:
    """Returns the file path for a user's chat history."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, f"{user_id}.json")

def load_chat_history_from_file(user_id: str) -> List[Dict[str, Any]]:
    """Loads chat history for a given user ID from a JSON file (non-cached helper)."""
    memory_file = get_user_memory_path(user_id)
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                return history
        except json.JSONDecodeError as e:
            print(f"Error decoding chat history for user {user_id}: {e}. Starting fresh.")
            return []
    return []

def save_chat_history(user_id: str, messages: List[Dict[str, Any]]):
    """Saves chat history for a given user ID to a JSON file."""
    memory_file = get_user_memory_path(user_id)
    try:
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        print(f"Saved chat history for user {user_id} to {memory_file}")
    except Exception as e:
        print(f"Error saving chat history for user {user_id}: {e}")

# --- Helper Function for History Formatting ---
def format_chat_history(streamlit_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Converts Streamlit message history to LlamaIndex ChatMessage list."""
    history = []
    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history


# --- Agent Interaction ---
def get_agent_response(query: str, chat_history: List[ChatMessage]) -> str:
    """
    Get a response from the agent stored in the session state using the chat method,
    explicitly passing the conversation history.

    Args:
        query: The user's query
        chat_history: The conversation history as a list of ChatMessage objects.

    Returns:
        The agent's response as a string, or an error message.
    """
    # Agent is now guaranteed to be in session_state because setup_agent is called before this.
    agent = st.session_state[AGENT_SESSION_KEY]

    try:
        # --- Update LLM Temperature from Slider ---
        current_temperature = st.session_state.get("llm_temperature", 0.7)

        if hasattr(agent, '_agent_worker') and hasattr(agent._agent_worker, '_llm'):
            actual_llm_instance = agent._agent_worker._llm
            if hasattr(actual_llm_instance, 'temperature'):
                actual_llm_instance.temperature = current_temperature
                # print(f"Set LLM temperature to: {current_temperature}") # Removed to reduce log spam
            else:
                print(f"Warning: LLM object of type {type(actual_llm_instance)} does not have a 'temperature' attribute.")
        else:
            print("Warning: Could not access LLM object within the agent to set temperature. Agent or worker structure might have changed (_agent_worker or _agent_worker._llm not found).")
        # --- End Temperature Update ---

        with st.spinner("ESI is thinking..."):
            response = agent.chat(query, chat_history=chat_history)

        response_text = response.response if hasattr(response, 'response') else str(response)

        print(f"Orchestrator final response text for UI: \n{response_text[:500]}...") # Log snippet
        return response_text

    except Exception as e:
        # Log the error and return a friendly message
        print(f"Error getting orchestrator agent response: {e}")
        print(f"Error getting agent response: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Technical details: {str(e)}"

def handle_user_input(chat_input_value: str | None):
    """
    Process user input (either from chat box or suggested prompt)
    and update chat with AI response.
    """
    prompt_to_process = None

    # Prioritize suggested prompt if available
    if hasattr(st.session_state, 'prompt_to_use') and st.session_state.prompt_to_use:
        prompt_to_process = st.session_state.prompt_to_use
        st.session_state.prompt_to_use = None  # Clear it after using
    # Otherwise, use the value from the chat input box if it's not None
    elif chat_input_value:
        prompt_to_process = chat_input_value

    if prompt_to_process:
        # Add user message to chat history *before* calling the agent
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Format the *entire* current history (including the new user message)
        # to pass to the agent.chat method.
        formatted_history = format_chat_history(st.session_state.messages)

        # Get AI response using the session agent, passing the history
        response_text = get_agent_response(prompt_to_process, chat_history=formatted_history)

        # Add assistant response (potentially including plot marker from manual execution) to chat history
        # stui.display_chat will handle rendering the text and the plot/download button
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Save chat history after each turn
        save_chat_history(st.session_state.user_id, st.session_state.messages)

        # Update suggested prompts based on new chat history (using original full history)
        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)

        # Force Streamlit to rerun the script immediately to display the new messages
        st.rerun()

# --- UI Callbacks ---
def reset_chat_callback():
    """Resets the chat history and suggested prompts to their initial state, and saves."""
    print("Resetting chat...")
    st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}]
    st.session_state.suggested_prompts = DEFAULT_PROMPTS
    
    if 'prompt_to_use' in st.session_state:
        st.session_state.prompt_to_use = None
    
    # Save the reset history
    save_chat_history(st.session_state.user_id, st.session_state.messages)

    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    # --- Initialize LLM Settings FIRST (cached) ---
    setup_global_llm_settings()

    # Get or create user ID and store in session state (only once per session)
    if "user_id" not in st.session_state:
        st.session_state.user_id = get_cached_user_id() # Use the cached function
        print(f"User ID for this session: {st.session_state.user_id}")
    
    # Initialize agent (cached and stored in session state)
    if AGENT_SESSION_KEY not in st.session_state:
        st.session_state[AGENT_SESSION_KEY] = setup_agent()
        print("Agent for this session is ready.")

    # Handle regeneration request if flag is set
    # This needs to be called before stui.create_interface() so that
    # the messages are updated before being displayed.
    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request() # This function will set do_regenerate to False and rerun

    # Initialize chat history and suggested prompts from memory or fresh
    if "messages" not in st.session_state:
        # Use the cached function for chat history
        st.session_state.messages = get_cached_chat_history(st.session_state.user_id)
        # The print statement is now inside get_cached_chat_history, so no need for one here.

    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = DEFAULT_PROMPTS

    # Create the rest of the interface using stui (displays chat history, sidebar, etc.)
    # Pass the reset_chat_callback to stui.create_interface
    stui.create_interface(reset_callback=reset_chat_callback)

    # Display suggested prompts as buttons below the chat history
    if st.session_state.suggested_prompts:
        st.markdown("---") # Add a separator for visual clarity
        st.subheader("Suggested Prompts:")
        # Create columns for buttons, up to the number of suggested prompts
        cols = st.columns(len(st.session_state.suggested_prompts)) 
        for i, prompt in enumerate(st.session_state.suggested_prompts):
            with cols[i]:
                if st.button(prompt, key=f"suggested_prompt_btn_{i}"):
                    st.session_state.prompt_to_use = prompt
                    st.rerun() # Rerun to process the prompt

    # Render the chat input box at the bottom, capture its value
    chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")

    # Handle user input (either from chat box or a clicked suggested prompt button)
    # If a button was clicked, prompt_to_use is set, otherwise use chat_input_value
    handle_user_input(chat_input_value)


# --- Regeneration Logic ---
def handle_regeneration_request():
    """Handles the request to regenerate the last assistant response."""
    if not st.session_state.get("do_regenerate", False):
        return

    st.session_state.do_regenerate = False # Consume the flag

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'assistant':
        print("Warning: Regeneration called but last message is not from assistant or no messages exist.")
        st.rerun() # Rerun to clear state if needed
        return

    # Case 1: Regenerating the initial greeting
    if len(st.session_state.messages) == 1:
        print("Regenerating initial greeting...")
        new_greeting = generate_llm_greeting()
        st.session_state.messages[0]['content'] = new_greeting
        save_chat_history(st.session_state.user_id, st.session_state.messages) # Save regenerated greeting
        st.rerun()
        return

    # Case 2: Regenerating a response to a user query
    st.session_state.messages.pop() # Remove last assistant message

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'user':
        print("Warning: Cannot regenerate, no preceding user query found after popping assistant message.")
        # Potentially restore the popped message or handle error state more gracefully
        st.rerun()
        return

    print("Regenerating last assistant response to user query...")
    prompt_to_regenerate = st.session_state.messages[-1]['content']
    # The history should include the user message we are responding to
    formatted_history_for_regen = format_chat_history(st.session_state.messages)

    response_text = get_agent_response(prompt_to_regenerate, chat_history=formatted_history_for_regen)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    save_chat_history(st.session_state.user_id, st.session_state.messages) # Save regenerated response
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()


if __name__ == "__main__":
    # Display a warning if environment variables are missing
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")

    main()
