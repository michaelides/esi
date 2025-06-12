import streamlit as st
import os
import json
import re
import uuid
import extra_streamlit_components as esc
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
import stui
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
import shutil # Import shutil for directory deletion

from PyPDF2 import PdfReader
import io # Import io module for BytesIO
from llama_index.core.tools import FunctionTool # Import FunctionTool

# Import necessary libraries for Hugging Face integration
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import HfApi, Repository
import pandas as pd # Import pandas for data manipulation
import os # Import os to access environment variables

# Import HF_USER_MEMORIES_DATASET_ID from config.py
from config import HF_USER_MEMORIES_DATASET_ID

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

cookies = esc.CookieManager(key="esi_cookie_manager")

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

# MEMORY_DIR is no longer used for chat history/metadata storage, as it's now on Hugging Face.
# MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories") # REMOVED THIS LINE

# Import UI_ACCESSIBLE_WORKSPACE from tools.py
from tools import UI_ACCESSIBLE_WORKSPACE

# Constant to control the maximum number of messages sent in chat history to the LLM
MAX_CHAT_HISTORY_MESSAGES = 15 # Keep the last N messages to manage context length

@st.cache_resource
def setup_global_llm_settings() -> tuple[bool, str | None]:
    """Initializes global LLM settings using st.cache_resource to run only once."""
    print("LOG: setup_global_llm_settings() CALLED")
    print("Initializing LLM settings (cached)...")
    try:
        initialize_agent_settings()
        print("LLM settings initialized (cached).")
        return True, None
    except Exception as e:
        error_message = f"Fatal Error: Could not initialize LLM settings. {e}"
        print(error_message) # Keep the log
        return False, error_message

# New cached function for initial greeting
@st.cache_data(show_spinner=False)
def _get_initial_greeting_text():
    """Generates and caches the initial LLM greeting text for startup."""
    return generate_llm_greeting()

# New cached wrapper for suggested prompts
@st.cache_data(show_spinner=False)
def _cached_generate_suggested_prompts(chat_history: List[Dict[str, Any]]) -> List[str]:
    """
    Generates suggested prompts based on chat history, cached to avoid redundant LLM calls.
    The cache key is based on the content of chat_history.
    """
    print("Generating suggested prompts using LLM (cached call)...") # Log when LLM is actually called
    return generate_suggested_prompts(chat_history)

# Define dynamic tool functions that can access st.session_state
def read_uploaded_document_tool_fn(filename: str) -> str:
    """Reads the full text content of a document previously uploaded by the user.
    Input is the exact filename (e.g., 'my_dissertation.pdf')."""
    if "uploaded_documents" not in st.session_state or filename not in st.session_state.uploaded_documents:
        return f"Error: Document '{filename}' not found in uploaded documents. Available documents: {list(st.session_state.uploaded_documents.keys())}"
    return st.session_state.uploaded_documents[filename]

def analyze_dataframe_tool_fn(filename: str, head_rows: int = 5) -> str:
    """Provides summary information (shape, columns, dtypes, head, describe) about a pandas DataFrame
    previously uploaded by the user. Input is the exact filename (e.g., 'my_data.csv').
    For more complex analysis, use the 'code_interpreter' tool."""
    if "uploaded_dataframes" not in st.session_state or filename not in st.session_state.uploaded_dataframes:
        return f"Error: DataFrame '{filename}' not found in uploaded dataframes. Available dataframes: {list(st.session_state.uploaded_dataframes.keys())}"
    
    df = st.session_state.uploaded_dataframes[filename]
    
    info_str = f"DataFrame: {filename}\n"
    info_str += f"Shape: {df.shape}\n"
    info_str += f"Columns: {', '.join(df.columns)}\n"
    info_str += f"Data Types:\n{df.dtypes.to_string()}\n"
    
    # Ensure head_rows is not negative and not too large
    head_rows = max(0, min(head_rows, len(df)))
    if head_rows > 0:
        info_str += f"First {head_rows} rows:\n{df.head(head_rows).to_string()}\n"
    else:
        info_str += "No head rows requested or available.\n"

    info_str += f"Summary Statistics:\n{df.describe().to_string()}\n"
    
    return info_str

@st.cache_resource
def setup_agent(max_search_results: int) -> tuple[Any | None, str | None]:
    """Initializes the orchestrator agent using st.cache_resource to run only once per max_search_results value.
    Returns a tuple (agent_instance, error_message).
    agent_instance is None if an error occurred.
    error_message is None if successful.
    """
    print(f"LOG: setup_agent() CALLED (Cacheable) with max_search_results={max_search_results}")
    print("LOG: Agent not in session state. Initializing agent...") # Moved here
    try:
        # Create dynamic tools here, passing the functions defined above
        uploaded_doc_reader_tool = FunctionTool.from_defaults(
            fn=read_uploaded_document_tool_fn,
            name="read_uploaded_document",
            description="Reads the full text content of a document previously uploaded by the user. Input is the exact filename (e.g., 'my_dissertation.pdf'). Use this to answer questions about the content of uploaded documents."
        )

        dataframe_analyzer_tool = FunctionTool.from_defaults(
            fn=analyze_dataframe_tool_fn,
            name="analyze_uploaded_dataframe",
            description="Provides summary information (shape, columns, dtypes, head, describe) about a pandas DataFrame previously uploaded by the user. Input is the exact filename (e.g., 'my_data.csv'). Use this to understand the structure and basic statistics of uploaded datasets. For more complex analysis, use the 'code_interpreter' tool."
        )

        # Pass these dynamic tools and max_search_results to the agent creation function
        agent_instance = create_orchestrator_agent(
            dynamic_tools=[uploaded_doc_reader_tool, dataframe_analyzer_tool],
            max_search_results=max_search_results # Pass the parameter here
        )
        print("LOG: Orchestrator agent object initialized (cached) successfully.") # Moved here
        return agent_instance, None
    except Exception as e:
        error_message = f"Failed to initialize the AI agent. Please check configurations. Error: {e}"
        print(f"Error initializing orchestrator agent (cached): {e}") # Keep the log
        return None, error_message

def _get_or_create_user_id(long_term_memory_enabled_param: bool) -> tuple[str, str]:
    """
    Determines user ID and necessary cookie action.
    Returns a tuple: (user_id: str, cookie_action_flag: str).
    cookie_action_flag can be "DO_NOTHING", "SET_COOKIE", or "DELETE_COOKIE".
    This function NO LONGER performs cookie operations directly.
    """
    print(f"LOG: _get_or_create_user_id: LTM enabled: {long_term_memory_enabled_param}")
    existing_user_id = cookies.get(cookie="user_id")

    if long_term_memory_enabled_param:
        if existing_user_id:
            print(f"LOG: _get_or_create_user_id: LTM ON. Found existing user_id: {existing_user_id}. Action: DO_NOTHING.")
            return existing_user_id, "DO_NOTHING"
        else:
            new_user_id = str(uuid.uuid4())
            print(f"LOG: _get_or_create_user_id: LTM ON. No existing user_id. Generated new: {new_user_id}. Action: SET_COOKIE.")
            return new_user_id, "SET_COOKIE"
    else: # Long-term memory is disabled
        temporary_user_id = str(uuid.uuid4())
        if existing_user_id:
            print(f"LOG: _get_or_create_user_id: LTM OFF. Found and will discard existing user_id: {existing_user_id}. Generated temp_id: {temporary_user_id}. Action: DELETE_COOKIE.")
            return temporary_user_id, "DELETE_COOKIE"
        else:
            print(f"LOG: _get_or_create_user_id: LTM OFF. No existing user_id. Generated temp_id: {temporary_user_id}. Action: DO_NOTHING.")
            return temporary_user_id, "DO_NOTHING"

@st.cache_resource
def _initialize_user_session_data(long_term_memory_enabled_param: bool) -> tuple[str, Dict[str, Any], Dict[str, Any], str]:
    """
    Initializes user ID, loads chat data from Hugging Face (if long-term memory is enabled),
    and returns the cookie action flag.
    This function is cached to run only once per Streamlit session, or when its parameters change.
    Returns: (user_id, chat_metadata, all_chat_messages, cookie_action_flag)
    """
    print("LOG: _initialize_user_session_data() CALLED (Cacheable)")
    print("LOG: _initialize_user_session_data: Attempting User ID and initial data setup (cached)...")

    user_id, cookie_action_flag = _get_or_create_user_id(long_term_memory_enabled_param)
    print(f"LOG: _initialize_user_session_data: Received user_id: {user_id}, cookie_action_flag: {cookie_action_flag}")

    chat_metadata = {}
    all_chat_messages = {}

    if long_term_memory_enabled_param:
        # Load data from Hugging Face only if memory is enabled
        print(f"LOG: _initialize_user_session_data: LTM ON. Loading data for user {user_id} from Hugging Face.") # Updated log
        user_data = _load_user_data_from_hf(user_id) # This function is not cached, but its call is within a cached function
        chat_metadata = user_data["metadata"]
        all_chat_messages = user_data["messages"]
        print(f"LOG: _initialize_user_session_data: Initial data load complete. Found {len(chat_metadata)} chats for user {user_id}.")
    else:
        # No HF load for disabled memory, data structures remain empty/default
        print(f"LOG: _initialize_user_session_data: LTM OFF. No HF data loaded for temporary user_id {user_id}.")

    print(f"LOG: _initialize_user_session_data: User session data initialized (cached). Final User ID: {user_id}")
    return user_id, chat_metadata, all_chat_messages, cookie_action_flag

def _load_user_data_from_hf(user_id: str) -> Dict[str, Any]:
    """
    Loads all chat metadata and histories for a user from the Hugging Face dataset.
    This function is NOT cached by Streamlit.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set. Cannot load user data from Hugging Face.")
        return {"metadata": {}, "messages": {}}

    all_chat_metadata = {}
    all_chat_messages = {}

    try:
        # Load the entire dataset
        dataset = load_dataset(HF_USER_MEMORIES_DATASET_ID, split="train", token=hf_token)
        print(f"Loaded dataset '{HF_USER_MEMORIES_DATASET_ID}' with {len(dataset)} rows.")

        # Filter data for the current user
        user_data_rows = dataset.filter(lambda row: row["user_id"] == user_id)
        print(f"Found {len(user_data_rows)} rows for user {user_id}.")

        # Process rows to reconstruct chat_metadata and all_chat_messages
        for row in user_data_rows:
            chat_id = row.get("chat_id")
            role = row.get("role")
            content = row.get("content")

            if chat_id == "global_metadata" and role == "metadata":
                try:
                    all_chat_metadata = json.loads(content)
                    print(f"Reconstructed chat metadata for user {user_id}.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding metadata for user {user_id}: {e}. Metadata will be empty.")
                    all_chat_metadata = {}
            elif chat_id and role and content is not None:
                if chat_id not in all_chat_messages:
                    all_chat_messages[chat_id] = []
                all_chat_messages[chat_id].append({"role": role, "content": content})
        
        # Ensure messages are sorted by their original order if possible (not explicitly stored, but usually appended)
        # For now, assume the order from the dataset is sufficient.
        
        # Clean up metadata for chats that no longer have messages (e.g., if messages were manually deleted from HF)
        chats_to_remove_from_metadata = [cid for cid in all_chat_metadata if cid not in all_chat_messages]
        for cid in chats_to_remove_from_metadata:
            print(f"Chat {cid} found in metadata but no messages. Removing from metadata.")
            del all_chat_metadata[cid]

        print(f"Reconstructed {len(all_chat_metadata)} chats and their messages for user {user_id}.")
        return {"metadata": all_chat_metadata, "messages": all_chat_messages}

    except Exception as e:
        print(f"Error loading user data from Hugging Face for user {user_id}: {e}")
        return {"metadata": {}, "messages": {}}

def save_chat_history(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    """
    Saves a specific chat history for a given user ID to the Hugging Face dataset.
    """
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Not saving chat history to Hugging Face.")
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set. Skipping Hugging Face upload for chat history.")
        return

    try:
        # Load the existing dataset
        existing_dataset = load_dataset(HF_USER_MEMORIES_DATASET_ID, split="train", token=hf_token)
        
        # Filter out existing messages for this user_id and chat_id
        # Also keep the metadata row for this user
        filtered_dataset = existing_dataset.filter(
            lambda row: not (row["user_id"] == user_id and row["chat_id"] == chat_id),
            num_proc=os.cpu_count() # Use multiple processes for filtering if dataset is large
        )
        print(f"Filtered out old messages for chat {chat_id} (user {user_id}). Remaining rows: {len(filtered_dataset)}")

        # Prepare new data for the current chat
        new_chat_data = []
        for msg in messages:
            new_chat_data.append({
                "user_id": user_id,
                "chat_id": chat_id,
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", "")
            })
        new_chat_df = pd.DataFrame(new_chat_data)
        new_chat_dataset = Dataset.from_pandas(new_chat_df)

        # Concatenate the filtered existing data with the new chat data
        combined_df = pd.concat([filtered_dataset.to_pandas(), new_chat_dataset.to_pandas()], ignore_index=True)
        combined_dataset = Dataset.from_pandas(combined_df)
        print(f"Combined dataset has {len(combined_dataset)} rows after updating chat {chat_id}.")

        # Push the combined dataset back to the Hugging Face Hub
        combined_dataset.push_to_hub(HF_USER_MEMORIES_DATASET_ID, private=True, token=hf_token)
        print(f"Successfully uploaded updated chat history for chat {chat_id} (user {user_id}) to Hugging Face dataset '{HF_USER_MEMORIES_DATASET_ID}'.") # Updated log

    except Exception as e:
        print(f"Error uploading chat history to Hugging Face for chat {chat_id} (user {user_id}): {e}")
        st.error(f"Error saving chat history to cloud: {e}")

def save_chat_metadata(user_id: str, chat_metadata: Dict[str, str]):
    """Saves the chat metadata (ID to name mapping) for a user to Hugging Face."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Not saving chat metadata to Hugging Face.")
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set. Skipping Hugging Face upload for metadata.")
        return

    try:
        # Load the existing dataset
        existing_dataset = load_dataset(HF_USER_MEMORIES_DATASET_ID, split="train", token=hf_token)

        # Filter out the existing metadata row for this user
        filtered_dataset = existing_dataset.filter(
            lambda row: not (row["user_id"] == user_id and row["chat_id"] == "global_metadata" and row["role"] == "metadata"),
            num_proc=os.cpu_count()
        )
        print(f"Filtered out old metadata for user {user_id}. Remaining rows: {len(filtered_dataset)}")

        # Prepare new metadata row
        new_metadata_data = [{
            "user_id": user_id,
            "chat_id": "global_metadata", # Special chat_id for metadata
            "role": "metadata",
            "content": json.dumps(chat_metadata)
        }]
        new_metadata_df = pd.DataFrame(new_metadata_data)
        new_metadata_dataset = Dataset.from_pandas(new_metadata_df)

        # Concatenate and push
        combined_df = pd.concat([filtered_dataset.to_pandas(), new_metadata_dataset.to_pandas()], ignore_index=True)
        combined_dataset = Dataset.from_pandas(combined_df)
        print(f"Combined dataset has {len(combined_dataset)} rows after updating metadata for user {user_id}.")

        combined_dataset.push_to_hub(HF_USER_MEMORIES_DATASET_ID, private=True, token=hf_token)
        print(f"Saved chat metadata for user {user_id} to Hugging Face dataset '{HF_USER_MEMORIES_DATASET_ID}'.") # Updated log

    except Exception as e:
        print(f"Error saving chat metadata to Hugging Face for user {user_id}: {e}")
        st.error(f"Error saving chat metadata to cloud: {e}")

def format_chat_history(streamlit_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """
    Converts Streamlit message history to LlamaIndex ChatMessage list,
    truncating to the most recent messages to manage context length.
    """
    # Truncate messages to keep only the most recent ones
    # If the list is shorter than MAX_CHAT_HISTORY_MESSAGES, it will take all.
    truncated_messages = streamlit_messages[-MAX_CHAT_HISTORY_MESSAGES:]

    history = []
    for msg in truncated_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history

def get_agent_response(query: str, chat_history: List[ChatMessage]) -> str:
    """
    Get a response from the agent stored in the session state using the chat method,
    explicitly passing the conversation history.
    """
    agent = st.session_state[AGENT_SESSION_KEY]

    try:
        # Get temperature from session state
        current_temperature = st.session_state.get("llm_temperature", 0.7)
        current_verbosity = st.session_state.get("llm_verbosity", 3) # Default to 3 if not found

        # New logic to set temperature using llama_index.core.Settings
        if Settings.llm:
            if hasattr(Settings.llm, 'temperature'):
                Settings.llm.temperature = current_temperature
                print(f"Successfully set Settings.llm.temperature to: {current_temperature}")
            else:
                print(f"Warning: Settings.llm ({type(Settings.llm)}) does not have a 'temperature' attribute.")
        else:
            print("Warning: Settings.llm is not initialized. Cannot set temperature.")

        # Prepend verbosity level to the query
        modified_query = f"Verbosity Level: {current_verbosity}. {query}"
        print(f"Modified query with verbosity: {modified_query}")

        with st.spinner("ESI is thinking..."):
            # Corrected to use the passed chat_history parameter
            response = agent.chat(modified_query, chat_history=chat_history) 

        response_text = response.response if hasattr(response, 'response') else str(response)

        print(f"Orchestrator final response text for UI: \n{response_text[:500]}...")
        return response_text

    except Exception as e:
        print(f"Error getting orchestrator agent response: {type(e).__name__} - {e}")
        print(f"Error getting agent response: {type(e).__name__} - {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Technical details: {str(e)}"

def create_new_chat_session_in_memory():
    """
    Creates a new chat session (ID, name, empty messages) in memory (st.session_state)
    and sets it as the current chat. Does NOT save to Hugging Face immediately.
    """
    new_chat_id = str(uuid.uuid4())
    
    new_chat_name = "Current Session" # Default for disabled memory
    if st.session_state.long_term_memory_enabled:
        existing_idea_nums = []
        for name in st.session_state.chat_metadata.values():
            match = re.match(r"Idea (\d+)", name)
            if match:
                existing_idea_nums.append(int(match.group(1)))
        
        next_idea_num = 1
        if existing_idea_nums:
            next_idea_num = max(existing_idea_nums) + 1
        new_chat_name = f"Idea {next_idea_num}"

    st.session_state.chat_metadata[new_chat_id] = new_chat_name
    # Keep generate_llm_greeting here for new chat creation
    st.session_state.all_chat_messages[new_chat_id] = [{"role": "assistant", "content": _get_initial_greeting_text()}]
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = st.session_state.all_chat_messages[new_chat_id]
    st.session_state.chat_modified = False # New chats are initially unsaved
    
    # Generate initial prompts for the new chat
    st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages)

    print(f"Created new chat in memory: ID={new_chat_id}, Name='{new_chat_name}' (not yet saved to HF)")
    return new_chat_id # Return the new chat ID

def switch_chat(chat_id: str):
    """Switches to an existing chat, ensuring messages are loaded."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Cannot switch to historical chats. Starting a new temporary session.")
        create_new_chat_session_in_memory()
        st.rerun()
        return

    if chat_id not in st.session_state.chat_metadata:
        print(f"Error: Attempted to switch to chat ID '{chat_id}' not found in metadata.")
        return

    # Messages for the target chat_id should already be loaded in st.session_state.all_chat_messages
    # by _initialize_user_session_data or _load_user_data_from_hf.
    # If for some reason they are not, it indicates an issue with the loading logic.
    if st.session_state.all_chat_messages.get(chat_id) is None:
        print(f"WARNING: Messages for current chat ID '{chat_id}' were not loaded by _initialize_user_session_data. This indicates an issue. Setting to empty list.")
        st.session_state.all_chat_messages[chat_id] = [] # Fallback
            
    st.session_state.messages = st.session_state.all_chat_messages.get(chat_id, [])
    
    st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages) # Use cached version
    st.session_state.chat_modified = True # Assume existing chat is modified if switched to (will be saved on next AI response)
    print(f"Switched to chat: ID={chat_id}, Name='{st.session_state.chat_metadata.get(chat_id, 'Unknown')}'")
    st.rerun()

def delete_chat_session(chat_id: str):
    """Deletes a chat history and its metadata from Hugging Face."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Cannot delete historical chats. Resetting current session.")
        if chat_id == st.session_state.current_chat_id:
            create_new_chat_session_in_memory()
            st.rerun()
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set. Skipping Hugging Face deletion.")
        st.error("Cannot delete chat: Hugging Face token not configured.")
        return

    # Check if the chat to be deleted is the currently active one
    is_current_chat = (chat_id == st.session_state.current_chat_id)

    try:
        # Load the existing dataset
        existing_dataset = load_dataset(HF_USER_MEMORIES_DATASET_ID, split="train", token=hf_token)

        # Filter out all rows belonging to the user_id and the specific chat_id
        # Keep the metadata row for this user, unless it's the last chat being deleted
        filtered_dataset = existing_dataset.filter(
            lambda row: not (row["user_id"] == st.session_state.user_id and row["chat_id"] == chat_id),
            num_proc=os.cpu_count()
        )
        print(f"Filtered out chat {chat_id} for user {st.session_state.user_id}. Remaining rows: {len(filtered_dataset)}")

        # Push the filtered dataset back to the Hugging Face Hub
        filtered_dataset.push_to_hub(HF_USER_MEMORIES_DATASET_ID, private=True, token=hf_token)
        print(f"Successfully deleted chat {chat_id} (user {st.session_state.user_id}) from Hugging Face dataset '{HF_USER_MEMORIES_DATASET_ID}'.")

        # Update in-memory session state
        if chat_id in st.session_state.all_chat_messages:
            del st.session_state.all_chat_messages[chat_id]
        if chat_id in st.session_state.chat_metadata:
            del st.session_state.chat_metadata[chat_id]
        
        # Save updated metadata to Hugging Face
        save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
        print(f"Deleted chat: ID={chat_id}")

        # If the deleted chat was the current one, switch to another or create a new one
        if is_current_chat:
            if st.session_state.chat_metadata:
                # Switch to the first available chat
                first_available_chat_id = next(iter(st.session_state.chat_metadata))
                print(f"Deleted current chat. Switching to: {first_available_chat_id}")
                # Call switch_chat to handle updating session state and rerunning
                switch_chat(first_available_chat_id)
            else:
                # No other chats left, set to a "no chat" state
                print("Deleted last chat. Setting to no active chat state.")
                st.session_state.current_chat_id = None
                st.session_state.messages = [{"role": "assistant", "content": _get_initial_greeting_text()}]
                st.session_state.chat_modified = False
                st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages) # Generate prompts for new empty chat
                st.rerun() # Rerun to display the new state
        else:
            # If a non-current chat was deleted, just rerun to update the sidebar
            st.rerun()
    except Exception as e:
        print(f"Error deleting chat {chat_id} from Hugging Face: {e}")
        st.error(f"Error deleting chat from cloud: {e}")
        # No rerun needed if chat_id wasn't found, as nothing changed.

def rename_chat(chat_id: str, new_name: str): # Modified to accept chat_id
    """Renames the specified chat."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Cannot rename chats.")
        return
    if chat_id and new_name and new_name != st.session_state.chat_metadata.get(chat_id):
        st.session_state.chat_metadata[chat_id] = new_name
        save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
        print(f"Renamed chat {chat_id} to '{new_name}'")
        # Removed st.rerun() from here as it causes the "no-op" warning in on_change callbacks.
        # Streamlit will automatically rerun after the on_change event completes.

def get_discussion_markdown(chat_id: str) -> str:
    """Retrieves messages for a given chat_id and formats them into a Markdown string."""
    messages = st.session_state.all_chat_messages.get(chat_id, [])
    markdown_content = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        markdown_content.append(f"**{role}:**\n{content}\n\n---")
    return "\n".join(markdown_content)

def get_discussion_docx(chat_id: str) -> bytes:
    """Retrieves messages for a given chat_id and formats them into a DOCX file."""
    messages = st.session_state.all_chat_messages.get(chat_id, [])
    document = Document()
    
    document.add_heading(f"Chat Discussion: {st.session_state.chat_metadata.get(chat_id, 'Untitled Chat')}", level=1)
    # Use current_chat_name if available, otherwise default to a generic name
    document.add_paragraph(f"Exported on: {st.session_state.chat_metadata.get(chat_id, 'Unknown Chat')}") 

    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        
        document.add_heading(f"{role}:", level=3)
        document.add_paragraph(content)
        document.add_paragraph("---") # Separator

    # Save document to a BytesIO object
    byte_stream = BytesIO()
    document.save(byte_stream)
    byte_stream.seek(0) # Rewind to the beginning of the stream
    return byte_stream.getvalue()

def handle_user_input(chat_input_value: str | None):
    """
    Process user input (either from chat box or suggested prompt)
    and update chat with AI response.
    """
    prompt_to_process = None

    if hasattr(st.session_state, 'prompt_to_use') and st.session_state.prompt_to_use:
        prompt_to_process = st.session_state.prompt_to_use
        st.session_state.prompt_to_use = None
    elif chat_input_value:
        prompt_to_process = chat_input_value

    if prompt_to_process:
        # If this is the first user message in a new, unsaved chat, mark it as modified
        # and save its metadata for the first time.
        if not st.session_state.chat_modified and st.session_state.current_chat_id is None:
            # This means it's the very first message in a fresh session, or after all chats were deleted.
            # Create a new chat session in memory and set it as current.
            # This will also update st.session_state.chat_metadata and st.session_state.all_chat_messages
            # with the new chat's entry.
            new_chat_id = create_new_chat_session_in_memory()
            # Now that the chat is created and current_chat_id is set, save its metadata to Hugging Face.
            # Only save metadata if long-term memory is enabled
            if st.session_state.long_term_memory_enabled:
                save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
            st.session_state.chat_modified = True # Mark as modified for history saving
            print(f"Activated new chat '{st.session_state.chat_metadata.get(st.session_state.current_chat_id)}' with first user input.")
        elif not st.session_state.chat_modified and len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
            # This handles the case where a new chat was created via the "New Chat" button
            # and this is the first user message in it.
            st.session_state.chat_modified = True 
            # Only save metadata if long-term memory is enabled
            if st.session_state.long_term_memory_enabled:
                save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata) # Save metadata now that chat is active
            print(f"Chat '{st.session_state.chat_metadata.get(st.session_state.current_chat_id)}' activated and metadata saved.")


        st.session_state.messages.append({"role": "user", "content": prompt_to_process})

        formatted_history = format_chat_history(st.session_state.messages)
        response_text = get_agent_response(prompt_to_process, chat_history=formatted_history)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Autosave the current chat history after AI response if it's been modified
        if st.session_state.chat_modified and st.session_state.long_term_memory_enabled:
            save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)

        st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages) # Use cached version
        st.rerun()

def reset_chat_callback():
    """Resets the chat by creating a new, unsaved chat session."""
    print("Resetting chat by creating a new session...")
    create_new_chat_session_in_memory() # Create new chat in memory
    st.rerun() # Rerun to display the new chat

def handle_regeneration_request():
    """Handles the request to regenerate the last assistant response."""
    if not st.session_state.get("do_regenerate", False):
        return

    st.session_state.do_regenerate = False

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'assistant':
        print("Warning: Regeneration called but last message is not from assistant or no messages exist.")
        st.rerun()
        return

    if len(st.session_state.messages) == 1:
        print("Regenerating initial greeting...")
        # Keep generate_llm_greeting here as it's an explicit regeneration request
        new_greeting = generate_llm_greeting() 
        st.session_state.messages[0]['content'] = new_greeting
        if st.session_state.long_term_memory_enabled: # Only save if memory is enabled
            save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
        st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages) # Generate prompts for regenerated greeting
        st.rerun()
        return

    print("Regenerating last assistant response to user query...")
    st.session_state.messages.pop() # Remove last assistant message

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'user':
        print("Warning: Cannot regenerate, no preceding user query found after popping assistant message.")
        st.rerun()
        return

    prompt_to_regenerate = st.session_state.messages[-1]['content']
    formatted_history_for_regen = format_chat_history(st.session_state.messages)

    response_text = get_agent_response(prompt_to_regenerate, chat_history=formatted_history_for_regen)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    if st.session_state.long_term_memory_enabled: # Only save if memory is enabled
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
    st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages) # Use cached version
    st.rerun()

def forget_me_and_reset():
    """
    Deletes all user chat histories from the Hugging Face dataset, removes the user ID cookie,
    and resets the Streamlit session state to a fresh start.
    """
    user_id_to_delete = st.session_state.get("user_id")
    hf_token = os.getenv("HF_TOKEN")

    if user_id_to_delete and hf_token:
        try:
            # Load the existing dataset
            existing_dataset = load_dataset(HF_USER_MEMORIES_DATASET_ID, split="train", token=hf_token)

            # Filter out all rows belonging to the user_id to be deleted
            filtered_dataset = existing_dataset.filter(
                lambda row: row["user_id"] != user_id_to_delete,
                num_proc=os.cpu_count()
            )
            print(f"Filtered out all data for user {user_id_to_delete}. Remaining rows: {len(filtered_dataset)}")

            # Push the filtered dataset back to the Hugging Face Hub
            filtered_dataset.push_to_hub(HF_USER_MEMORIES_DATASET_ID, private=True, token=hf_token)
            print(f"Successfully deleted all data for user {user_id_to_delete} from Hugging Face dataset '{HF_USER_MEMORIES_DATASET_ID}'.")

        except Exception as e:
            print(f"Error deleting user data from Hugging Face for user {user_id_to_delete}: {e}")
            st.error(f"Failed to delete user data from cloud: {e}")
    elif not hf_token:
        print("HF_TOKEN environment variable not set. Cannot delete user data from Hugging Face.")
        st.warning("Cannot delete user data from cloud: Hugging Face token not configured.")

    # Delete the user ID cookie
    try:
        cookies.delete(cookie="user_id")
        print(f"Deleted user ID cookie for {user_id_to_delete}")
    except Exception as e:
        print(f"ERROR: Failed to delete user_id cookie for {user_id_to_delete}: {e}")
        st.error(f"Failed to delete user ID cookie: {e}")

    # Reset session state to clear all chat history and user data in memory
    st.session_state.chat_metadata = {}
    st.session_state.all_chat_messages = {}
    st.session_state.current_chat_id = None
    st.session_state.messages = [] # Clear messages, will be re-populated by main's init
    st.session_state.chat_modified = False
    st.session_state.suggested_prompts = DEFAULT_PROMPTS # Reset to default prompts
    st.session_state.renaming_chat_id = None
    st.session_state.uploaded_documents = {}
    st.session_state.uploaded_dataframes = {}
    
    # Crucially, reset the session_control_flags_initialized to force full re-initialization
    # in the main function on the next rerun.
    st.session_state.session_control_flags_initialized = False
    st.session_state._greeting_logic_log_shown_for_current_state = False # Reset for fresh log on next run
    
    # Delete user_id from session state to force re-generation of a temporary one
    if "user_id" in st.session_state:
        del st.session_state.user_id

    # Use JavaScript to clear cookies and force a full page reload
    # This ensures a complete reset from the browser's perspective.
    js_code = """
    <script>
        function deleteAllCookies() {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i];
                const eqPos = cookie.indexOf('=');
                const name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
                document.cookie = name + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
            }
        }
        deleteAllCookies();
        window.location.reload(true); // Force a hard reload from the server
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)

    print(f"Session reset. New temporary user ID will be generated on next run.")
    # No st.rerun() here, as the JavaScript reload will handle it.

def _set_long_term_memory_preference():
    """Callback to save the long_term_memory_enabled state to a cookie."""
    current_value = st.session_state.long_term_memory_enabled
    try:
        cookies.set(cookie="long_term_memory_pref", val=str(current_value))
        print(f"Long-term memory preference saved to cookie: {current_value}")
    except Exception as e:
        print(f"ERROR: Failed to save long-term memory preference to cookie: {e}")
        st.error(f"Failed to save preference: {e}")
    # No st.rerun() here, as the toggle itself triggers a rerun.
    # The main loop's memory state change detection will handle the rest.
    st.session_state._last_memory_state_changed_by_toggle = True

def main():
    """Main function to run the Streamlit app."""
    success, error_message = setup_global_llm_settings()
    if not success:
        st.error(error_message)
        st.stop()

    # --- Long-term memory initialization and change detection ---
    pref_from_cookie = cookies.get(cookie="long_term_memory_pref")

    if "long_term_memory_enabled" not in st.session_state:
        if pref_from_cookie is not None:
            # Convert the cookie value to a string first for robust handling
            pref_str = str(pref_from_cookie).lower()
            
            if pref_str == 'true' or pref_str == '1':
                st.session_state.long_term_memory_enabled = True
            elif pref_str == 'false' or pref_str == '0':
                st.session_state.long_term_memory_enabled = False
            else:
                # Fallback for any unexpected string/value, default to True
                st.session_state.long_term_memory_enabled = True
                print(f"Warning: Unexpected value for long_term_memory_pref cookie: '{pref_from_cookie}'. Defaulting to True.")
            print(f"Loaded long_term_memory_enabled from cookie: {st.session_state.long_term_memory_enabled}")
        else:
            st.session_state.long_term_memory_enabled = True  # Default: enabled
            cookies.set(cookie="long_term_memory_pref", val=str(st.session_state.long_term_memory_enabled))
            print(f"long_term_memory_enabled not in session state or cookie. Defaulting to {st.session_state.long_term_memory_enabled} and saving cookie.")

    if "_last_memory_state_was_enabled" not in st.session_state:
        st.session_state._last_memory_state_was_enabled = st.session_state.long_term_memory_enabled
        print(f"_last_memory_state_was_enabled initialized to match long_term_memory_enabled: {st.session_state._last_memory_state_was_enabled}")

    # --- Handle Memory State Change ---
    memory_state_has_changed_this_run = st.session_state._last_memory_state_was_enabled != st.session_state.long_term_memory_enabled
    if memory_state_has_changed_this_run:
        print(f"LOG: Main: Memory state CHANGE DETECTED or FORCED from {st.session_state._last_memory_state_was_enabled} to {st.session_state.long_term_memory_enabled}.")
        st.session_state._last_memory_state_was_enabled = st.session_state.long_term_memory_enabled
        st.session_state.session_control_flags_initialized = False # Trigger re-init

        if "user_id" in st.session_state:
            del st.session_state.user_id
        
        _initialize_user_session_data.clear()
        print("LOG: Main: Cleared _initialize_user_session_data cache due to memory state change.")
        # REMOVED st.rerun() from here

    if "_last_memory_state_changed_by_toggle" not in st.session_state: # Initialize if not present
        st.session_state._last_memory_state_changed_by_toggle = False

    # Reset the toggle flag after processing potential changes
    st.session_state._last_memory_state_changed_by_toggle = False


    # --- Core Session Variable Initialization (runs once per session OR after memory state change) ---
    if not st.session_state.get("session_control_flags_initialized", False):
        print("LOG: SESSION INIT: Initializing core session variables (first run or after memory state change).")

        st.session_state.initial_greeting_shown_for_session = False # Reset for new session/memory config
        st.session_state.current_chat_id = None
        st.session_state.messages = [] # Will be populated by chat logic below
        st.session_state.chat_modified = False
        st.session_state.suggested_prompts = DEFAULT_PROMPTS
        st.session_state.renaming_chat_id = None
        st.session_state.uploaded_documents = {}
        st.session_state.uploaded_dataframes = {}

        # chat_metadata and all_chat_messages are populated by _initialize_user_session_data.
        # user_id is also populated by _initialize_user_session_data.

        st.session_state.session_control_flags_initialized = True
        print("LOG: SESSION INIT: Core session variables initialized.")
    # else: # This else is too verbose for every run.
        # print("SESSION INFO: Core session variables confirmed initialized.")

    # --- User ID and Chat Data Load (cached, sensitive to memory state) ---
    # This call is critical. It runs if:
    # 1. Not run before in the session (cache miss).
    # 2. `long_term_memory_enabled` parameter to it changes (cache invalidation, forced by .clear() above).
    print(f"LOG: Main: Calling _initialize_user_session_data with LTM_enabled={st.session_state.long_term_memory_enabled}")
    user_id_val, chat_metadata_val, all_chat_messages_val, cookie_action = \
        _initialize_user_session_data(st.session_state.long_term_memory_enabled)

    st.session_state.user_id = user_id_val
    st.session_state.chat_metadata = chat_metadata_val
    st.session_state.all_chat_messages = all_chat_messages_val

    # --- Apply cookie actions based on _initialize_user_session_data result ---
    if cookie_action == "SET_COOKIE":
        cookies.set(cookie="user_id", val=user_id_val)
        print(f"LOG: Main: Set user_id cookie: {user_id_val}")
    elif cookie_action == "DELETE_COOKIE":
        cookies.delete(cookie="user_id")
        print(f"LOG: Main: Deleted user_id cookie.")
    # --- End Apply cookie actions ---

    # --- Agent Initialization (runs once per session) ---
    print(f"LOG: Main: Checking for agent in session. AGENT_SESSION_KEY exists: {AGENT_SESSION_KEY in st.session_state}")
    if AGENT_SESSION_KEY not in st.session_state:
        print(f"LOG: Main: Agent not found. Calling setup_agent().")
        # Pass a default value for max_search_results
        agent_instance, error_message = setup_agent(max_search_results=10) 
        if agent_instance is None:
            st.error(error_message)
            st.stop()
        st.session_state[AGENT_SESSION_KEY] = agent_instance
    # else: # Too verbose
        # print("Agent already initialized.")

    # --- Active Chat and Initial Greeting Logic ---
    # This section determines which chat is active and whether to show an initial greeting.
    # print(f"CHAT LOGIC: Processing. Current chat ID: {st.session_state.current_chat_id}, Memory: {st.session_state.long_term_memory_enabled}, Greeting Shown: {st.session_state.initial_greeting_shown_for_session}, Messages Count: {len(st.session_state.messages)}")

    chat_state_resolved = False

    if st.session_state.long_term_memory_enabled:
        # LTM is ON
        if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_metadata:
            # Valid current_chat_id exists, ensure messages are loaded (they should be by _initialize_user_session_data)
            if st.session_state.all_chat_messages.get(st.session_state.current_chat_id) is None:
                print(f"WARNING: Messages for current chat ID '{st.session_state.current_chat_id}' were not loaded by _initialize_user_session_data. This indicates an issue. Setting to empty list.")
                st.session_state.all_chat_messages[st.session_state.current_chat_id] = [] # Fallback
            
            st.session_state.messages = st.session_state.all_chat_messages.get(st.session_state.current_chat_id, [])
            st.session_state.chat_modified = True # Existing chat is considered modifiable
            chat_state_resolved = True
            print(f"CHAT LOGIC (LTM ON): Active chat is '{st.session_state.current_chat_id}'. Messages: {len(st.session_state.messages)}")

        elif st.session_state.chat_metadata: # No current_chat_id, but other chats exist in metadata
            first_available_chat_id = next(iter(st.session_state.chat_metadata))
            print(f"CHAT LOGIC (LTM ON): No current chat ID. Selecting first available: '{first_available_chat_id}'. Rerunning via switch_chat.")
            # switch_chat handles setting current_chat_id and messages, then reruns.
            # This rerun is acceptable here as it's a one-time setup for the session or view.
            switch_chat(first_available_chat_id)
            # Execution stops here due to rerun in switch_chat
        else: # No current_chat_id and no chats in metadata (e.g., new LTM user)
            if not st.session_state.initial_greeting_shown_for_session:
                print("CHAT LOGIC (LTM ON): No chats exist. Displaying initial greeting.")
                st.session_state.messages = [{"role": "assistant", "content": _get_initial_greeting_text()}]
                st.session_state.initial_greeting_shown_for_session = True
                st.session_state.current_chat_id = None
                st.session_state.chat_modified = False
            chat_state_resolved = True
    else:
        # LTM is OFF - manage a single, temporary chat session
        # If no current_chat_id, or if current_chat_id points to something not in all_chat_messages (e.g. after toggling LTM OFF)
        if not st.session_state.current_chat_id or \
           st.session_state.current_chat_id not in st.session_state.all_chat_messages or \
           not st.session_state.messages: # Also check if messages list is empty, implying a need for greeting

            # Only print creation message if greeting hasn't been shown for this "session" (meaning since LTM was turned off or app start)
            if not st.session_state.initial_greeting_shown_for_session:
                print("CHAT LOGIC (LTM OFF): Creating new temporary session with greeting.")
                create_new_chat_session_in_memory() # This sets up a new chat with greeting
                st.session_state.initial_greeting_shown_for_session = True
            elif not st.session_state.messages: # Greeting was shown, but messages are empty (e.g. user cleared chat)
                 print("CHAT LOGIC (LTM OFF): Messages empty, recreating greeting for temporary session.")
                 create_new_chat_session_in_memory() # This sets up a new chat with greeting
            else:
                # This case means a temporary chat exists (current_chat_id is valid and in all_chat_messages)
                # and messages are already populated. We just ensure st.session_state.messages points to it.
                 st.session_state.messages = st.session_state.all_chat_messages[st.session_state.current_chat_id]
                 print(f"CHAT LOGIC (LTM OFF): Using existing temporary chat '{st.session_state.current_chat_id}'. Messages: {len(st.session_state.messages)}")

        else:
            # Valid temporary chat already exists, ensure messages are correctly assigned
            st.session_state.messages = st.session_state.all_chat_messages[st.session_state.current_chat_id]
            print(f"CHAT LOGIC (LTM OFF): Confirmed existing temporary chat '{st.session_state.current_chat_id}'. Messages: {len(st.session_state.messages)}")
        chat_state_resolved = True

    # Fallback: If messages list is somehow still not a list (should be extremely rare)
    if not isinstance(st.session_state.messages, list):
        print("WARNING: st.session_state.messages was not a list after chat logic. Resetting to empty list and default prompts.")
        st.session_state.messages = []
        st.session_state.suggested_prompts = DEFAULT_PROMPTS
        st.session_state.current_chat_id = None
        st.session_state.chat_modified = False
    elif not st.session_state.messages and not st.session_state.initial_greeting_shown_for_session:
        # If after all logic, messages are still empty AND no greeting has been shown (e.g. a state was missed)
        print("FALLBACK: No messages and no greeting shown. Displaying initial greeting.")
        st.session_state.messages = [{"role": "assistant", "content": _get_initial_greeting_text()}]
        st.session_state.initial_greeting_shown_for_session = True
        st.session_state.current_chat_id = None # No active chat context for this
        st.session_state.chat_modified = False

    # Update suggested prompts based on the final state of messages
    # This ensures prompts are relevant to the current view, whether it's a loaded chat or a new greeting.
    if 'suggested_prompts' not in st.session_state or \
       (st.session_state.messages and st.session_state.suggested_prompts == DEFAULT_PROMPTS and len(st.session_state.messages) > 1) or \
       (not st.session_state.messages and st.session_state.suggested_prompts != DEFAULT_PROMPTS): # Update if messages are empty but prompts are not default
        print("Updating suggested prompts based on current messages state.")
        st.session_state.suggested_prompts = _cached_generate_suggested_prompts(st.session_state.messages if st.session_state.messages else [])


    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request()

    stui.create_interface(
        reset_callback=reset_chat_callback,
        new_chat_callback=lambda: create_new_chat_session_in_memory() and st.rerun(),
        delete_chat_callback=delete_chat_session,
        rename_chat_callback=rename_chat, # Pass the modified rename_chat function
        chat_metadata=st.session_state.chat_metadata,
        current_chat_id=st.session_state.current_chat_id,
        switch_chat_callback=switch_chat,
        get_discussion_markdown_callback=get_discussion_markdown,
        get_discussion_docx_callback=get_discussion_docx, # Pass the new DOCX callback
        suggested_prompts_list=st.session_state.suggested_prompts,
        handle_user_input_callback=handle_user_input,
        long_term_memory_enabled=st.session_state.long_term_memory_enabled, # Pass the new setting
        forget_me_callback=forget_me_and_reset, # Pass the new callback
        set_long_term_memory_callback=_set_long_term_memory_preference # Pass the new callback
    )

    chat_input_for_handler = st.session_state.get("chat_input_value_from_stui")
    if "chat_input_value_from_stui" in st.session_state: # Ensure the key exists before deleting
        del st.session_state.chat_input_value_from_stui # Or set to None: st.session_state.chat_input_value_from_stui = None
    
    # Call handle_user_input if there's a chat input or a suggested prompt pending
    if chat_input_for_handler or st.session_state.get('prompt_to_use'):
        handle_user_input(chat_input_for_handler)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")
    main()
