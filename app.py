import streamlit as st
import os
import json
import re
import uuid
import extra_streamlit_components as esc
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Event # For type hinting if needed for handler events
import stui
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting, StreamingOrchestratorAgentWorkflow # Import the workflow
from dotenv import load_dotenv
from docx import Document
from io import BytesIO

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Load system prompt globally
try:
    with open(os.path.join(PROJECT_ROOT, "esi_agent_instruction.md"), "r") as f:
        SYSTEM_PROMPT_TEXT = f.read().strip()
except FileNotFoundError:
    print("Warning: esi_agent_instruction.md not found. Using default system prompt.")
    SYSTEM_PROMPT_TEXT = "You are ESI, an AI assistant for dissertation support. Please be helpful and try your best to assist the user with their queries, using the tools provided when necessary."


cookies = esc.CookieManager(key="esi_cookie_manager")

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories")

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
    # Iterate over a copy of items for safe deletion during iteration
    for chat_id, chat_name in list(all_chat_metadata.items()): 
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            # Instead of loading messages, set to None for lazy loading
            all_chat_messages[chat_id] = None 
            # Instead of loading messages, set to None for lazy loading
            all_chat_messages[chat_id] = None 
        else:
            print(f"Chat file {chat_file} not found for chat ID {chat_id}. Removing from metadata.")
            # Remove chat_id from all_chat_metadata if its message file doesn't exist
            if chat_id in all_chat_metadata:
                del all_chat_metadata[chat_id]
            # Do not add to all_chat_messages if file is missing
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

def format_chat_history(
    streamlit_messages: List[Dict[str, Any]], 
    ensure_system_prompt: bool = True
) -> List[ChatMessage]:
    """
    Converts Streamlit message history to LlamaIndex ChatMessage list.
    Optionally ensures the history starts with the system prompt.
    """
    history = []
    has_system_prompt = False
    if streamlit_messages and streamlit_messages[0]["role"] == "system":
        has_system_prompt = True

    if ensure_system_prompt and not has_system_prompt:
        history.append(ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT_TEXT))

    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else \
               MessageRole.ASSISTANT if msg["role"] == "assistant" else \
               MessageRole.SYSTEM if msg["role"] == "system" else MessageRole.TOOL 
               # Added tool role, and explicit system role handling
        history.append(ChatMessage(role=role, content=str(msg["content"]))) # Ensure content is string
    return history

def get_memory_for_chat(streamlit_messages: List[Dict[str, Any]]) -> ChatMemoryBuffer:
    """
    Creates a LlamaIndex ChatMemoryBuffer from Streamlit messages,
    ensuring the system prompt is the first message.
    """
    # Format messages, ensuring system prompt is first.
    llama_chat_messages = format_chat_history(streamlit_messages, ensure_system_prompt=True)
    # Create and return memory buffer
    return ChatMemoryBuffer.from_messages(llama_chat_messages)


async def get_agent_response_handler(query: str, current_chat_history: List[ChatMessage]):
    """
    Calls the agent workflow with the given query and history.
    Returns the workflow handler.
    Note: `current_chat_history` is List[ChatMessage] from format_chat_history.
    The workflow's `prepare_chat_history` step will add the current `query` to this history.
    """
    agent_workflow: StreamingOrchestratorAgentWorkflow = st.session_state[AGENT_SESSION_KEY]

    # The workflow's StartEvent expects 'user_query' and 'chat_history'
    # `current_chat_history` here is the history *before* the current user query.
    # The workflow's `prepare_chat_history` step will append the user_query.
    try:
        # Note: Gemini LLM for workflow needs temperature set if not default in Settings.llm
        # This is currently handled by initialize_settings and direct modification in old get_agent_response.
        # The workflow uses Settings.llm, so temperature should be configured there.
        # Verbosity is also handled by the LLM based on the prompt content.

        handler = await agent_workflow.arun_async(
            user_query=query,
            chat_history=current_chat_history # Pass the history *before* the current user query
        )
        return handler
    except Exception as e:
        error_message = f"I apologize, but I encountered an error while processing your request. Technical details: {str(e)}"
        print(f"Error getting agent workflow handler: {e}")
        # Need a way to return a handler-like object that streams this error.
        # For now, this will raise, and the calling function needs to handle it or we adjust this.
        # Let's make it return a mock handler that streams the error.
        
        class MockHandler:
            async def arun_async(self, *args, **kwargs): # Match signature if something tries to call run on it
                return {"response_message": ChatMessage(role=MessageRole.ASSISTANT, content=error_message)}

            async def stream_events(self):
                yield Event(type="error", delta=error_message) # Simplified event, adapt as needed by consumer
                # Or more aligned with StreamEvent:
                # yield StreamEvent(delta=error_message)
        
        mock_handler = MockHandler()
        # To make it directly usable by an event stream processor expecting StreamEvent:
        async def error_event_stream_generator(msg):
            from agent import StreamEvent # Local import for clarity
            yield StreamEvent(delta=msg)

        # Instead of returning a mock handler, let's make get_agent_response_handler return
        # something that the event_stream_processor can iterate over.
        # The event_stream_processor in main() expects an object with a `stream_events` async generator.
        # So, the MockHandler approach is better.
        
        # However, the workflow might fail at `arun_async` itself.
        # The current `get_agent_response` returns a stream generator for errors.
        # Let's make this function simpler: it returns handler or raises, caller handles.
        # No, the prompt implies this function should return a handler or a way to stream the error.
        # The previous `get_agent_response` returned a generator.
        # `st.write_stream` takes an async generator or a sync generator.
        # The `event_stream_processor` will take the handler. If handler is None or error, it should cope.
        
        # For now, let get_agent_response_handler raise, and main logic will try-catch it.
        # This is simpler than crafting a perfect mock handler here.
        raise # Re-raise the exception to be caught by the caller in main()

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
    # Initialize with system prompt, then assistant greeting
    st.session_state.all_chat_messages[new_chat_id] = [
        {"role": "system", "content": SYSTEM_PROMPT_TEXT},
        {"role": "assistant", "content": generate_llm_greeting()}
    ]
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
        
        # Prepare for streaming the assistant's response
        st.session_state.stream_next_assistant_response = True
        st.session_state.current_prompt_for_streaming = prompt_to_process
        
        # Remove direct call to get_agent_response and appending assistant message here.
        # No longer generating suggested prompts here as it will be done after stream completion.
        
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
                        print(f"An unexpected error occurred while reading initial chat file {chat_file}: {e}")
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
                    print(f"An unexpected error occurred while reading current chat file {chat_file}: {e}")
                    st.session_state.all_chat_messages[chat_id_to_load] = [] # Default to empty
            else:
                print(f"Current chat file {chat_file} not found for chat ID '{chat_id_to_load}'. Setting to empty messages.")
                st.session_state.all_chat_messages[chat_id_to_load] = [] # Default to empty

        st.session_state.messages = st.session_state.all_chat_messages[chat_id_to_load]
        st.session_state.chat_modified = True # Existing chats are considered modified for saving
        print(f"Continuing with chat: '{st.session_state.chat_metadata.get(st.session_state.current_chat_id, st.session_state.current_chat_id)}'")

    if st.session_state.get("do_regenerate", False):
        handle_regeneration_request()

    # Display chat messages from history
    # Note: st.session_state.messages may be temporarily modified by other callbacks like delete.
    # Ensure it's valid before iterating.
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.all_chat_messages:
        current_messages = st.session_state.all_chat_messages[st.session_state.current_chat_id]
        if current_messages is not None: # Messages might be None if not loaded yet (though main logic tries to load them)
            for msg in current_messages: # Display currently loaded messages for the active chat
                 with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        # If current_messages is None, it implies an issue or that it's a new chat about to be populated.
        # The streaming logic below will handle adding the first assistant message if needed.
    elif not st.session_state.chat_metadata: # No chats at all, show initial greeting
        with st.chat_message("assistant"):
            st.write(generate_llm_greeting())


    # Check if we need to stream a new assistant response
    if st.session_state.get("stream_next_assistant_response", False):
        
        async def run_agent_and_stream():
            prompt = st.session_state.current_prompt_for_streaming
            
            if st.session_state.current_chat_id is None:
                if not st.session_state.chat_metadata:
                     create_new_chat_session_in_memory()
                     # This rerun might short-circuit, handle_user_input will trigger another stream prep.
                     # This is a bit complex; ideally, current_chat_id is always set before streaming.
                     # For now, assume create_new_chat_session_in_memory + rerun correctly sets up for next pass.
                     # Or, we might need to prevent rerun in create_new_chat_session_in_memory if called mid-logic.
                     # Let's assume handle_user_input correctly establishes a chat session ID first.

            st.session_state.stream_next_assistant_response = False # Reset flag early
            st.session_state.current_prompt_for_streaming = None

            history_for_agent_call = st.session_state.messages[:-1]
            llama_chat_history = format_chat_history(history_for_agent_call, ensure_system_prompt=True)

            final_agent_message_content = ""

            try:
                handler = await get_agent_response_handler(prompt, llama_chat_history)

                async def event_stream_processor(handler_obj):
                    async for event in handler_obj.stream_events():
                        # Assuming StreamEvent is defined in agent.py and has 'delta'
                        from agent import StreamEvent as AgentStreamEvent # Avoid conflict if stui has StreamEvent
                        if isinstance(event, AgentStreamEvent) and event.delta is not None:
                            yield str(event.delta)
                
                with st.chat_message("assistant"):
                    # This will render the stream of deltas
                    full_response_text_via_stream = st.write_stream(event_stream_processor(handler))
                
                stop_event_output = await handler 
                final_agent_message_obj = stop_event_output.get("response_message")

                if final_agent_message_obj and hasattr(final_agent_message_obj, 'content'):
                    final_agent_message_content = final_agent_message_obj.content
                else: 
                    final_agent_message_content = full_response_text_via_stream 
                    print("Warning: Could not retrieve final message from StopEvent, using concatenated stream content.")

            except Exception as e:
                print(f"Error during agent response streaming or handling: {e}")
                error_message_for_ui = f"I apologize, but I encountered an error: {str(e)}"
                with st.chat_message("assistant"):
                    st.write(error_message_for_ui)
                final_agent_message_content = error_message_for_ui
            
            # Update message lists and save history
            st.session_state.messages.append({"role": "assistant", "content": final_agent_message_content})
            if st.session_state.current_chat_id:
                 st.session_state.all_chat_messages[st.session_state.current_chat_id] = st.session_state.messages
            if st.session_state.chat_modified and st.session_state.current_chat_id:
                save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
            
            st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
            st.rerun() # Rerun to reflect the completed response and new suggested prompts

        import asyncio
        asyncio.run(run_agent_and_stream())

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
