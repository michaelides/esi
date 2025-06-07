import streamlit as st
import os
import json
import re
import uuid
import extra_streamlit_components as esc
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
import stui
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
import shutil # Import shutil for directory deletion

from PyPDF2 import PdfReader
import io # Import io module for BytesIO
from llama_index.core.tools import FunctionTool # Import FunctionTool

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

cookies = esc.CookieManager(key="esi_cookie_manager")

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories")

# Import UI_ACCESSIBLE_WORKSPACE from tools.py
from tools import UI_ACCESSIBLE_WORKSPACE

# Constant to control the maximum number of messages sent in chat history to the LLM
MAX_CHAT_HISTORY_MESSAGES = 15 # Keep the last N messages to manage context length

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

# New cached function for initial greeting
@st.cache_data(show_spinner=False)
def _get_initial_greeting_text():
    """Generates and caches the initial LLM greeting text for startup."""
    return generate_llm_greeting()

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
def setup_agent():
    """Initializes the orchestrator agent using st.cache_resource to run only once."""
    print("Initializing orchestrator agent (cached)...")
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

        # Pass these dynamic tools to the agent creation function
        agent_instance = create_orchestrator_agent(
            dynamic_tools=[uploaded_doc_reader_tool, dataframe_analyzer_tool]
        )
        print("Orchestrator agent object initialized (cached) successfully.")
        return agent_instance
    except Exception as e:
        print(f"Error initializing orchestrator agent (cached): {e}")
        st.error(f"Failed to initialize the AI agent. Please check configurations. Error: {e}")
        st.stop()

@st.cache_resource
def _get_user_id_from_cookie_cached():
    """Retrieves user ID from cookies, cached to run only once per session."""
    user_id = cookies.get(cookie="user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        # Setting a cookie might trigger a rerun, but the function itself won't re-execute
        cookies.set(cookie="user_id", val=user_id)
        print(f"New user ID created and set in cookie: {user_id}")
    else:
        print(f"Existing user ID retrieved from cookie: {user_id}")
    return user_id

def _load_user_data_from_disk(user_id: str) -> Dict[str, Any]:
    """
    Loads all chat metadata and histories for a user directly from disk.
    This function is NOT cached by Streamlit.
    """
    user_dir = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    chat_metadata_path = os.path.join(user_dir, "chat_metadata.json")
    all_chat_metadata = {}
    if os.path.exists(chat_metadata_path):
        try:
            with open(chat_metadata_path, "r", encoding="utf-8") as f:
                all_chat_metadata = json.load(f)
            print(f"Loaded chat metadata for user {user_id} from disk.")
        except json.JSONDecodeError as e:
            print(f"Error decoding chat metadata for user {user_id}: {e}. Starting fresh metadata.")
            all_chat_metadata = {}
    
    all_chat_messages = {}
    # Iterate over a copy of items for safe deletion during iteration
    for chat_id, chat_name in list(all_chat_metadata.items()): 
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            # Instead of loading messages, set to None for lazy loading
            all_chat_messages[chat_id] = None 
        else:
            print(f"Chat file {chat_file} not found for chat ID {chat_id}. Removing from metadata.")
            # Remove chat_id from all_chat_metadata if its message file doesn't exist
            if chat_id in all_chat_metadata:
                del all_chat_metadata[chat_id]
            # Do not add to all_chat_messages if file is missing
    
    print(f"Processed metadata for {len(all_chat_metadata)} chats. Messages will be lazy-loaded.")
    return {"metadata": all_chat_metadata, "messages": all_chat_messages}

def save_chat_history(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    """Saves a specific chat history for a given user ID to a JSON file."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Not saving chat history to disk.")
        return
    user_dir = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    memory_file = os.path.join(user_dir, f"{chat_id}.json")
    try:
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        print(f"Saved chat history for chat {chat_id} (user {user_id}) to {memory_file}")
    except Exception as e:
        print(f"Error saving chat history for chat {chat_id} (user {user_id}): {e}")

def save_chat_metadata(user_id: str, chat_metadata: Dict[str, str]):
    """Saves the chat metadata (ID to name mapping) for a user."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Not saving chat metadata to disk.")
        return
    user_dir = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    metadata_file = os.path.join(user_dir, "chat_metadata.json")
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chat_metadata, f, indent=2)
        print(f"Saved chat metadata for user {user_id} to {metadata_file}")
    except Exception as e:
        print(f"Error saving chat metadata for user {user_id}): {e}")

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
        current_temperature = st.session_state.get("llm_temperature", 0.7)
        current_verbosity = st.session_state.get("llm_verbosity", 3) # Default to 3 if not found

        # Corrected access path for LLM temperature
        if hasattr(agent, '_agent_worker') and hasattr(agent._agent_worker, 'llm') and hasattr(agent._agent_worker.llm, 'temperature'):
            actual_llm_instance = agent._agent_worker.llm
            actual_llm_instance.temperature = current_temperature
        else:
            print(f"Warning: Could not access LLM object within the agent to set temperature. Agent or LLM structure might have changed (agent._agent_worker.llm or agent._agent_worker.llm.temperature not found).")

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
        print(f"Error getting orchestrator agent response: {e}")
        print(f"Error getting agent response: {e}")
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Technical details: {str(e)}"

def create_new_chat_session_in_memory():
    """
    Creates a new chat session (ID, name, empty messages) in memory (st.session_state)
    and sets it as the current chat. Does NOT save to disk immediately.
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
    st.session_state.all_chat_messages[new_chat_id] = [{"role": "assistant", "content": generate_llm_greeting()}]
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = st.session_state.all_chat_messages[new_chat_id]
    st.session_state.chat_modified = False # New chats are initially unsaved
    
    print(f"Created new chat in memory: ID={new_chat_id}, Name='{new_chat_name}' (not yet saved to disk)")
    return new_chat_id # Return the new chat ID

def switch_chat(chat_id: str):
    """Switches to an existing chat, lazy-loading messages if necessary."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Cannot switch to historical chats. Starting a new temporary session.")
        create_new_chat_session_in_memory()
        st.rerun()
        return

    if chat_id not in st.session_state.chat_metadata:
        print(f"Error: Attempted to switch to chat ID '{chat_id}' not found in metadata.")
        return

    # Lazy load messages if they are not already loaded
    if st.session_state.all_chat_messages.get(chat_id) is None:
        print(f"Messages for chat ID '{chat_id}' not loaded. Loading from disk...")
        user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
        chat_file = os.path.join(user_dir, f"{chat_id}.json")

        if os.path.exists(chat_file):
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    st.session_state.all_chat_messages[chat_id] = json.load(f)
                print(f"Successfully loaded messages for chat ID '{chat_id}'.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for chat {chat_id} (file: {chat_file}): {e}.")
                st.session_state.all_chat_messages[chat_id] = [] # Set to empty list on error
            except Exception as e:
                print(f"An unexpected error occurred while reading chat file {chat_file}: {e}")
                st.session_state.all_chat_messages[chat_id] = [] # Set to empty list on error
        else:
            print(f"Chat file {chat_file} not found for chat ID '{chat_id}'. Setting to empty messages.")
            st.session_state.all_chat_messages[chat_id] = [] # Set to empty list if file not found


    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.all_chat_messages[chat_id]
    # Ensure messages are not None before generating prompts
    if st.session_state.messages is None: 
        # This case should ideally be handled by the loading logic above,
        # setting it to [] if loading fails.
        print(f"Warning: Messages for chat {chat_id} are None even after loading attempt. Defaulting to empty list for prompts.")
        st.session_state.messages = []
        st.session_state.all_chat_messages[chat_id] = []


    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.session_state.chat_modified = True # Assume existing chat is modified if switched to (will be saved on next AI response)
    print(f"Switched to chat: ID={chat_id}, Name='{st.session_state.chat_metadata.get(chat_id, 'Unknown')}'")
    st.rerun()

def delete_chat_session(chat_id: str):
    """Deletes a chat history and its metadata."""
    if not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Cannot delete historical chats from disk. Resetting current session.")
        if chat_id == st.session_state.current_chat_id:
            create_new_chat_session_in_memory()
            st.rerun()
        return

    # Check if the chat to be deleted is the currently active one
    is_current_chat = (chat_id == st.session_state.current_chat_id)

    if chat_id in st.session_state.all_chat_messages:
        # Delete from in-memory session state
        del st.session_state.all_chat_messages[chat_id]
        del st.session_state.chat_metadata[chat_id]
        
        # Delete the physical file
        user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
            print(f"Deleted chat file: {chat_file}")
        
        # Save updated metadata to disk
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
                # Keep generate_llm_greeting here for new session after deletion
                st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}]
                st.session_state.chat_modified = False
                st.rerun() # Rerun to display the new state
        else:
            # If a non-current chat was deleted, just rerun to update the sidebar
            st.rerun()
    else:
        print(f"Attempted to delete non-existent chat ID: {chat_id}")
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
            # Now that the chat is created and current_chat_id is set, save its metadata to disk.
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

        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
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
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()

def forget_me_and_reset():
    """
    Deletes all user chat histories from disk, removes the user ID cookie,
    and resets the Streamlit session state to a fresh start.
    """
    user_id_to_delete = st.session_state.get("user_id")
    if user_id_to_delete:
        user_dir = os.path.join(MEMORY_DIR, user_id_to_delete)
        if os.path.exists(user_dir):
            try:
                shutil.rmtree(user_dir)
                print(f"Successfully deleted user directory: {user_dir}")
            except Exception as e:
                print(f"Error deleting user directory {user_dir}: {e}")
                st.error(f"Failed to delete user data from disk: {e}")
        
        # Delete the user ID cookie
        cookies.delete(cookie="user_id")
        print(f"Deleted user ID cookie for {user_id_to_delete}")

    # Reset session state to clear all chat history and user data in memory
    # This effectively simulates a fresh start for the user.
    st.session_state.chat_metadata = {}
    st.session_state.all_chat_messages = {}
    st.session_state.current_chat_id = None
    # Keep generate_llm_greeting here for new session after full reset
    st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}] # New greeting
    st.session_state.chat_modified = False
    st.session_state.suggested_prompts = DEFAULT_PROMPTS
    st.session_state.renaming_chat_id = None
    st.session_state.uploaded_documents = {}
    st.session_state.uploaded_dataframes = {}
    
    # Crucially, set long_term_memory_enabled to False and update the tracking state
    # so the app starts in a "no memory" mode after "forget me".
    st.session_state.long_term_memory_enabled = False
    st.session_state._last_memory_state_was_enabled = False # Ensure this is updated
    
    # Delete user_id from session state to force re-generation of a temporary one
    if "user_id" in st.session_state:
        del st.session_state.user_id

    # Generate a new temporary user ID for the fresh session
    # This will be handled by the main loop's user ID setup now.
    print(f"Session reset. New temporary user ID will be generated on next run.")

    st.rerun()


def main():
    """Main function to run the Streamlit app."""
    setup_global_llm_settings()

    # Initialize long_term_memory_enabled setting
    if "long_term_memory_enabled" not in st.session_state:
        st.session_state.long_term_memory_enabled = True # Default to enabled

    if "session_control_flags_initialized" not in st.session_state:
        print("First run in session: Initializing session control flags and core variables.")
        st.session_state.long_term_memory_enabled = True  # Default
        st.session_state._last_memory_state_was_enabled = True
        st.session_state.initial_data_load_attempted = False
        st.session_state.initial_greeting_shown_for_session = False

        # Core data structures
        st.session_state.chat_metadata = {}
        st.session_state.all_chat_messages = {}
        st.session_state.current_chat_id = None
        st.session_state.messages = []  # Start empty, will be populated
        st.session_state.chat_modified = False
        st.session_state.suggested_prompts = DEFAULT_PROMPTS
        st.session_state.renaming_chat_id = None
        st.session_state.uploaded_documents = {} # Initialize here for consistency
        st.session_state.uploaded_dataframes = {} # Initialize here for consistency

        st.session_state.session_control_flags_initialized = True
    else:
        print("Session control flags already initialized.")

    # --- Handle Memory State Change ---
    memory_state_changed = st.session_state._last_memory_state_was_enabled != st.session_state.long_term_memory_enabled
    if memory_state_changed:
        print(f"Memory state changed from {st.session_state._last_memory_state_was_enabled} to {st.session_state.long_term_memory_enabled}. Resetting relevant state.")
        # Reset control flags
        st.session_state.initial_data_load_attempted = False
        st.session_state.initial_greeting_shown_for_session = False # Reset greeting flag

        # Clear data structures that depend on memory state
        st.session_state.chat_metadata = {}
        st.session_state.all_chat_messages = {}
        st.session_state.current_chat_id = None
        st.session_state.messages = [] # Clear messages
        st.session_state.chat_modified = False
        st.session_state.suggested_prompts = DEFAULT_PROMPTS
        # Keep uploaded_documents and uploaded_dataframes as they are independent of this memory toggle

        # Crucially, if memory is turned OFF, we need to clear the user_id
        # so a new temporary one is generated.
        if not st.session_state.long_term_memory_enabled:
            if "user_id" in st.session_state:
                del st.session_state.user_id # Force new temporary ID generation

        # Update the tracking flag AFTER processing changes
        st.session_state._last_memory_state_was_enabled = st.session_state.long_term_memory_enabled
        print("Finished resetting state due to memory change.")

    # --- User ID Setup ---
    # This block now only determines if long-term memory is enabled and assigns the user_id
    # The actual retrieval/creation of the user_id is handled by the cached function or direct UUID generation.
    if "user_id" not in st.session_state: # Check if user_id is already set in session state
        print("Attempting User ID setup...")
        if st.session_state.long_term_memory_enabled:
            st.session_state.user_id = _get_user_id_from_cookie_cached() # Call the cached function
            print(f"User ID for long-term memory: {st.session_state.user_id}")
        else:
            st.session_state.user_id = str(uuid.uuid4()) # Temporary ID for non-persistent session
            print(f"User ID for non-persistent session: {st.session_state.user_id}")
        print("User ID setup complete.")
    else:
        print(f"User ID already set in session state: {st.session_state.user_id}")


    # --- Agent Initialization ---
    if AGENT_SESSION_KEY not in st.session_state:
        print("Agent not in session state. Initializing agent...")
        st.session_state[AGENT_SESSION_KEY] = setup_agent()
        print("Agent initialized and stored in session state.")

    # --- Initial Data Loading (from disk) ---
    if st.session_state.long_term_memory_enabled and not st.session_state.initial_data_load_attempted:
        print(f"Long-term memory enabled and initial data load not attempted. Loading for user {st.session_state.user_id}...")
        user_data = _load_user_data_from_disk(st.session_state.user_id)
        st.session_state.chat_metadata = user_data["metadata"]
        st.session_state.all_chat_messages = user_data["messages"]
        st.session_state.initial_data_load_attempted = True
        print(f"Initial data load attempt complete. Found {len(st.session_state.chat_metadata)} chats.")
    elif not st.session_state.long_term_memory_enabled:
        print("Long-term memory disabled. Skipping disk load. Ensuring initial_data_load_attempted is False.")
        # Ensure this is False if memory is off, so if it's turned on later, loading can occur.
        st.session_state.initial_data_load_attempted = False


    # --- Active Chat and Initial Greeting Logic ---
    print(f"Processing active chat and greeting. Current chat ID: {st.session_state.current_chat_id}, Memory enabled: {st.session_state.long_term_memory_enabled}, Initial greeting shown: {st.session_state.initial_greeting_shown_for_session}")

    if st.session_state.long_term_memory_enabled:
        # Case 1: Current chat ID is valid and points to existing metadata
        if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_metadata:
            print(f"Current chat ID '{st.session_state.current_chat_id}' is set and valid.")
            # Lazy load messages if not already loaded
            if st.session_state.all_chat_messages.get(st.session_state.current_chat_id) is None:
                print(f"Lazy loading messages for chat '{st.session_state.current_chat_id}'.")
                user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
                chat_file = os.path.join(user_dir, f"{st.session_state.current_chat_id}.json")
                if os.path.exists(chat_file):
                    try:
                        with open(chat_file, "r", encoding="utf-8") as f:
                            st.session_state.all_chat_messages[st.session_state.current_chat_id] = json.load(f)
                        print("Messages loaded successfully.")
                    except Exception as e:
                        print(f"Error loading messages for chat {st.session_state.current_chat_id}: {e}. Setting to empty list.")
                        st.session_state.all_chat_messages[st.session_state.current_chat_id] = []
                else:
                    print(f"Chat file not found for {st.session_state.current_chat_id}. Setting to empty list.")
                    st.session_state.all_chat_messages[st.session_state.current_chat_id] = []
            st.session_state.messages = st.session_state.all_chat_messages.get(st.session_state.current_chat_id, [])
            st.session_state.chat_modified = True # Assume modification if loaded
        # Case 2: No current chat ID, but other chats exist in metadata -> select first one
        elif not st.session_state.current_chat_id and st.session_state.chat_metadata:
            first_available_chat_id = next(iter(st.session_state.chat_metadata))
            print(f"No current chat ID, selecting first available: '{first_available_chat_id}'.")
            st.session_state.current_chat_id = first_available_chat_id
            # Lazy load its messages (similar to above)
            if st.session_state.all_chat_messages.get(st.session_state.current_chat_id) is None:
                print(f"Lazy loading messages for selected chat '{st.session_state.current_chat_id}'.")
                user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
                chat_file = os.path.join(user_dir, f"{st.session_state.current_chat_id}.json")
                if os.path.exists(chat_file):
                    try:
                        with open(chat_file, "r", encoding="utf-8") as f:
                            st.session_state.all_chat_messages[st.session_state.current_chat_id] = json.load(f)
                    except Exception as e:
                        print(f"Error loading messages for chat {st.session_state.current_chat_id}: {e}. Setting to empty list.")
                        st.session_state.all_chat_messages[st.session_state.current_chat_id] = []
                else:
                    print(f"Chat file not found for {st.session_state.current_chat_id}. Setting to empty list.")
                    st.session_state.all_chat_messages[st.session_state.current_chat_id] = []
            st.session_state.messages = st.session_state.all_chat_messages.get(st.session_state.current_chat_id, [])
            st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
            st.session_state.chat_modified = True
        # Case 3: No chats exist in metadata (e.g., new user, or all chats deleted)
        else: # This also covers if current_chat_id was None and chat_metadata was empty
            print("No existing chats found in metadata for long-term memory user.")
            if not st.session_state.initial_greeting_shown_for_session:
                print("Initial greeting for this session not shown. Generating and displaying.")
                # Use the cached greeting for initial startup
                st.session_state.messages = [{"role": "assistant", "content": _get_initial_greeting_text()}]
                st.session_state.initial_greeting_shown_for_session = True
                st.session_state.current_chat_id = None # No active chat yet
                st.session_state.chat_modified = False # Greeting is not a modification
            else:
                # Greeting already shown, but still no chats. messages should be empty or as is.
                # The greeting should persist in st.session_state.messages from the first run
                # if initial_greeting_shown_for_session is True and current_chat_id is None.
                # Removed the line: if not st.session_state.current_chat_id: st.session_state.messages = []
                print("Initial greeting already shown for session, no new greeting needed. Messages remain as is or empty.")
    else: # Long-term memory is DISABLED
        print("Long-term memory is disabled.")
        # If current_chat_id is None (no temporary session exists yet for this disabled-memory run)
        # or if the current_chat_id (which would be a temp one) is somehow not in all_chat_messages
        # (e.g., after memory state change from enabled to disabled, current_chat_id might have been cleared)
        if st.session_state.current_chat_id is None or \
           st.session_state.current_chat_id not in st.session_state.all_chat_messages:
            print("No active temporary session. Creating one with greeting.")
            new_temp_chat_id = str(uuid.uuid4())
            st.session_state.current_chat_id = new_temp_chat_id
            # Use the cached greeting for initial startup
            st.session_state.messages = [{"role": "assistant", "content": _get_initial_greeting_text()}]
            st.session_state.chat_metadata = {new_temp_chat_id: "Current Session"}
            st.session_state.all_chat_messages = {new_temp_chat_id: st.session_state.messages}
            st.session_state.chat_modified = False # Not saved to disk
            st.session_state.initial_greeting_shown_for_session = True
            print(f"Created new temporary chat {new_temp_chat_id}")
        else:
            # Temporary session already exists, messages should be populated from it.
            # This path is taken on reruns when memory is disabled and a temp chat is active.
            st.session_state.messages = st.session_state.all_chat_messages.get(st.session_state.current_chat_id, [])
            print(f"Continuing with existing temporary chat {st.session_state.current_chat_id}.")

    # Final check: if messages is still uninitialized (should not happen with above logic), set to empty list
    if not isinstance(st.session_state.messages, list):
        print("Warning: st.session_state.messages was not a list. Resetting to empty list.")
        st.session_state.messages = []


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
        forget_me_callback=forget_me_and_reset # Pass the new callback
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
