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
def get_cached_user_id():
    """Retrieves user ID from cookies or creates a new one, cached to run once per session."""
    user_id = cookies.get(cookie="user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        cookies.set(cookie="user_id", val=user_id)
        print(f"New user ID created and set (cached): {user_id}")
    else:
        print(f"Existing user ID retrieved (cached): {user_id}")
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

# Removed initialize_user_session_data function as its logic is now inlined in main().

def save_chat_history(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    """Saves a specific chat history for a given user ID to a JSON file."""
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
    """Converts Streamlit message history to LlamaIndex ChatMessage list."""
    history = []
    for msg in streamlit_messages:
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
    st.session_state.all_chat_messages[new_chat_id] = [{"role": "assistant", "content": generate_llm_greeting()}]
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = st.session_state.all_chat_messages[new_chat_id]
    st.session_state.chat_modified = False # New chats are initially unsaved
    
    # Removed immediate save_chat_metadata call here.
    # Metadata will be saved when the first user message is sent in handle_user_input.
    
    print(f"Created new chat in memory: ID={new_chat_id}, Name='{new_chat_name}' (not yet saved to disk)")
    return new_chat_id # Return the new chat ID

def switch_chat(chat_id: str):
    """Switches to an existing chat, lazy-loading messages if necessary."""
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
            save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
            st.session_state.chat_modified = True # Mark as modified for history saving
            print(f"Activated new chat '{st.session_state.chat_metadata.get(st.session_state.current_chat_id)}' with first user input.")
        elif not st.session_state.chat_modified and len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
            # This handles the case where a new chat was created via the "New Chat" button
            # and this is the first user message in it.
            st.session_state.chat_modified = True 
            save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata) # Save metadata now that chat is active
            print(f"Chat '{st.session_state.chat_metadata.get(st.session_state.current_chat_id)}' activated and metadata saved.")


        st.session_state.messages.append({"role": "user", "content": prompt_to_process})

        formatted_history = format_chat_history(st.session_state.messages)
        response_text = get_agent_response(prompt_to_process, chat_history=formatted_history)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Autosave the current chat history after AI response if it's been modified
        if st.session_state.chat_modified:
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
        new_greeting = generate_llm_greeting()
        st.session_state.messages[0]['content'] = new_greeting
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
    save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    setup_global_llm_settings()

    if "user_id" not in st.session_state:
        st.session_state.user_id = get_cached_user_id()
    
    # Initialize session state for uploaded files
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = {}
    if "uploaded_dataframes" not in st.session_state:
        st.session_state.uploaded_dataframes = {}

    if AGENT_SESSION_KEY not in st.session_state:
        st.session_state[AGENT_SESSION_KEY] = setup_agent()

    # Initialize all necessary session state variables with default empty values
    # This ensures they always exist, preventing AttributeError.
    if "chat_metadata" not in st.session_state:
        st.session_state.chat_metadata = {}
    if "all_chat_messages" not in st.session_state:
        st.session_state.all_chat_messages = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_modified" not in st.session_state:
        st.session_state.chat_modified = False
    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = DEFAULT_PROMPTS
    if 'renaming_chat_id' not in st.session_state:
        st.session_state.renaming_chat_id = None

    # Load user data from disk only if chat_metadata is empty (i.e., first run of a new session)
    # This prevents redundant disk reads on subsequent reruns.
    if not st.session_state.chat_metadata and not st.session_state.all_chat_messages:
        user_data = _load_user_data_from_disk(st.session_state.user_id)
        st.session_state.chat_metadata = user_data["metadata"]
        st.session_state.all_chat_messages = user_data["messages"]
        print(f"Initial load of user data for {st.session_state.user_id}: {len(st.session_state.chat_metadata)} chats.")
    else:
        print(f"User data already in session state. Current chats: {len(st.session_state.chat_metadata)}.")

    # Determine the active chat or present initial greeting
    # This logic ensures a valid current chat is always selected or a new blank state is presented.
    if st.session_state.current_chat_id is None or st.session_state.current_chat_id not in st.session_state.all_chat_messages:
        if st.session_state.chat_metadata:
            # If there are existing chats, switch to the first one
            first_available_chat_id = next(iter(st.session_state.chat_metadata))
            st.session_state.current_chat_id = first_available_chat_id
            
            # Lazy load messages for the first available chat if they are None
            if st.session_state.all_chat_messages.get(first_available_chat_id) is None:
                print(f"Initial load: Messages for chat ID '{first_available_chat_id}' not loaded. Loading from disk...")
                user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
                chat_file = os.path.join(user_dir, f"{first_available_chat_id}.json")
                if os.path.exists(chat_file):
                    try:
                        with open(chat_file, "r", encoding="utf-8") as f:
                            st.session_state.all_chat_messages[first_available_chat_id] = json.load(f)
                        print(f"Successfully loaded messages for initial chat ID '{first_available_chat_id}'.")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON for initial chat {first_available_chat_id} (file: {chat_file}): {e}.")
                        st.session_state.all_chat_messages[first_available_chat_id] = [] # Default to empty
                    except Exception as e:
                        print(f"An unexpected error occurred while reading initial chat file {first_available_chat_id}: {e}")
                        st.session_state.all_chat_messages[first_available_chat_id] = [] # Default to empty
                else:
                    print(f"Initial chat file {chat_file} not found for chat ID '{first_available_chat_id}'. Setting to empty messages.")
                    st.session_state.all_chat_messages[first_available_chat_id] = [] # Default to empty
            
            st.session_state.messages = st.session_state.all_chat_messages[first_available_chat_id]
            st.session_state.chat_modified = True # Existing chats are considered modified for saving
            print(f"No valid current chat found. Switched to first available chat: '{st.session_state.chat_metadata.get(first_available_chat_id, first_available_chat_id)}'")
        else:
            # No chats exist, present a blank slate with initial greeting
            print("No valid current chat found and no existing chats. Presenting initial greeting.")
            st.session_state.current_chat_id = None # Indicate no active chat ID
            st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}]
            st.session_state.chat_modified = False # This state is not yet saved to disk
    else:
        # current_chat_id is valid and in all_chat_messages keys (but messages might be None)
        chat_id_to_load = st.session_state.current_chat_id
        if st.session_state.all_chat_messages.get(chat_id_to_load) is None:
            print(f"Initial load: Messages for current chat ID '{chat_id_to_load}' not loaded. Loading from disk...")
            user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
            chat_file = os.path.join(user_dir, f"{chat_id_to_load}.json")
            if os.path.exists(chat_file):
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        st.session_state.all_chat_messages[chat_id_to_load] = json.load(f)
                    print(f"Successfully loaded messages for current chat ID '{chat_id_to_load}'.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for current chat {chat_id_to_load} (file: {chat_file}): {e}.")
                    st.session_state.all_chat_messages[chat_id_to_load] = [] # Default to empty
                except Exception as e:
                    print(f"An unexpected error occurred while reading current chat file {chat_id_to_load}: {e}")
                    st.session_state.all_chat_messages[chat_id_to_load] = [] # Default to empty
            else:
                print(f"Current chat file {chat_file} not found for chat ID '{chat_id_to_load}'. Setting to empty messages.")
                st.session_state.all_chat_messages[chat_id_to_load] = [] # Default to empty

        st.session_state.messages = st.session_state.all_chat_messages[chat_id_to_load]
        st.session_state.chat_modified = True # Existing chats are considered modified for saving
        print(f"Continuing with chat: '{st.session_state.chat_metadata.get(st.session_state.current_chat_id, st.session_state.current_chat_id)}'")

    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request()

    # --- File Upload Section in Sidebar ---



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
        handle_user_input_callback=handle_user_input
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
