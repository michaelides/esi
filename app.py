import streamlit as st
import os
import time
import json
import re
import uuid
import extra_streamlit_components as esc
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
import stui
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings, generate_llm_greeting
from dotenv import load_dotenv
from config import get_logger

load_dotenv()
logger = get_logger(__name__)
google_api_key = os.getenv("GOOGLE_API_KEY")
from config import PROJECT_ROOT, DOWNLOAD_MARKER, RAG_SOURCE_MARKER_PREFIX, MEMORY_DIR_NAME

st.set_page_config(
    page_title="ESI - ESI Scholarly Instructor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

cookies = esc.CookieManager(key="esi_cookie_manager")

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"

MEMORY_DIR = os.path.join(PROJECT_ROOT, MEMORY_DIR_NAME)

@st.cache_resource
def setup_global_llm_settings():
    """Initializes global LLM settings using st.cache_resource to run only once."""
    logger.info("Initializing LLM settings (cached)...")
    try:
        initialize_settings()
        logger.info("LLM settings initialized (cached).")
        return True, None # Success, no error message
    except Exception as e:
        error_message = (
            "Fatal Error: Could not initialize the AI's language model (Gemini).\n\n"
            "This is often due to an issue with the GOOGLE_API_KEY setup:\n"
            "1. Ensure the GOOGLE_API_KEY environment variable is set.\n"
            "2. Verify the API key is correct and active in your Google AI Studio or Google Cloud console.\n"
            "3. Check if the API key has permissions for the model being used (e.g., 'gemini-2.5-flash-preview-05-20').\n\n"
            f"Original error details: {e}"
        )
        logger.error(error_message)
        return False, error_message # Failure, with error message

@st.cache_resource
def setup_agent():
    """Initializes the orchestrator agent using st.cache_resource to run only once."""
    logger.info("Initializing orchestrator agent (cached)...")
    try:
        agent_instance = create_orchestrator_agent()
        logger.info("Orchestrator agent object initialized (cached) successfully.")
        return agent_instance, None # Success, no error message
    except Exception as e:
        error_message = f"Failed to initialize the AI agent. Please check configurations. Error: {e}"
        logger.error(f"Error initializing orchestrator agent (cached): {e}")
        return None, error_message # Failure, with error message

@st.cache_resource
def get_cached_user_id():
    """Retrieves user ID from cookies or creates a new one, cached to run once per session."""
    user_id = cookies.get(cookie="user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        cookies.set(cookie="user_id", val=user_id)
        logger.info(f"New user ID created and set (cached): {user_id}")
    else:
        logger.info(f"Existing user ID retrieved (cached): {user_id}")
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
            logger.info(f"Loaded chat metadata for user {user_id} from disk.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding chat metadata for user {user_id}: {e}. Starting fresh metadata.")
            all_chat_metadata = {}
    
    all_chat_messages = {}
    for chat_id, chat_name in list(all_chat_metadata.items()):
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    all_chat_messages[chat_id] = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding chat history for chat {chat_id} (file: {chat_file}): {e}. Removing from metadata.")
                del all_chat_metadata[chat_id]
        else:
            logger.warning(f"Chat file {chat_file} not found for chat ID {chat_id}. Removing from metadata.")
            del all_chat_metadata[chat_id]
    
    logger.info(f"Loaded {len(all_chat_messages)} chat histories for user {user_id} from disk.")
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
        logger.info(f"Saved chat history for chat {chat_id} (user {user_id}) to {memory_file}")
    except Exception as e:
        logger.error(f"Error saving chat history for chat {chat_id} (user {user_id}): {e}")

def save_chat_metadata(user_id: str, chat_metadata: Dict[str, str]):
    """Saves the chat metadata (ID to name mapping) for a user."""
    user_dir = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    metadata_file = os.path.join(user_dir, "chat_metadata.json")
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chat_metadata, f, indent=2)
        logger.info(f"Saved chat metadata for user {user_id} to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving chat metadata for user {user_id}): {e}")

def format_chat_history(streamlit_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Converts Streamlit message history to LlamaIndex ChatMessage list."""
    history = []
    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history

def get_agent_response(query: str, chat_history: List[ChatMessage]): # -> StreamingAgentChatResponse (or similar generator type)
    """
    Get a streaming response from the agent stored in the session state,
    explicitly passing the conversation history.
    """
    agent = st.session_state[AGENT_SESSION_KEY]

    try:
        current_temperature = st.session_state.get("llm_temperature", 0.7)
        current_verbosity = st.session_state.get("llm_verbosity", 3) # Default to 3 if not found

        if hasattr(agent, 'llm') and hasattr(agent.llm, 'temperature'):
            actual_llm_instance = agent.llm # type: ignore
            actual_llm_instance.temperature = current_temperature
        else:
            logger.warning(f"Could not access LLM object within the agent to set temperature. Agent or LLM structure might have changed (agent.llm or agent.llm.temperature not found).")

        # Prepend verbosity level to the query
        modified_query = f"Verbosity Level: {current_verbosity}. {query}"
        logger.info(f"Modified query with verbosity for streaming: {modified_query}")

        # Removed st.spinner from here; it will be handled by the caller.
        streaming_response = agent.stream_chat(modified_query, chat_history=chat_history)
        logger.info(f"Orchestrator is now streaming the response for query: {modified_query[:100]}...")
        return streaming_response

    except Exception as e:
        logger.error(f"Error getting orchestrator agent streaming response: {e}")
        # Re-raise the exception to be handled by the caller
        raise

def create_new_chat_session_in_memory():
    """
    Creates a new chat session (ID, name, empty messages) in memory (st.session_state)
    and sets it as the current chat. Does NOT save to disk immediately.
    """
    new_chat_id = str(uuid.uuid4())
    
    existing_idea_nums = []
    if st.session_state.chat_metadata: # Check if chat_metadata is not empty
        for name in st.session_state.chat_metadata.values():
            match = re.match(r"Idea (\d+)", name)
            if match:
                existing_idea_nums.append(int(match.group(1)))
    
    next_idea_num = 1
    if existing_idea_nums:
        next_idea_num = max(existing_idea_nums) + 1

    new_chat_name = f"Idea {next_idea_num}"

    # Update session state
    st.session_state.chat_metadata[new_chat_id] = new_chat_name
    st.session_state.all_chat_messages[new_chat_id] = [{"role": "assistant", "content": generate_llm_greeting()}]

    # Save metadata to disk immediately
    save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
    logger.info(f"Created new chat and saved metadata: ID={new_chat_id}, Name='{new_chat_name}'")

    # Set as current chat
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = st.session_state.all_chat_messages[new_chat_id]
    st.session_state.chat_modified = False # History is not modified yet
    
    return new_chat_id

def switch_chat(chat_id: str):
    """Switches to an existing chat."""
    if chat_id not in st.session_state.all_chat_messages:
        logger.warning(f"Attempted to switch to non-existent chat ID: {chat_id}")
        return # Or handle error

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.all_chat_messages[chat_id]
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.session_state.chat_modified = True # Assume existing chat is modified if switched to (will be saved on next AI response)
    logger.info(f"Switched to chat: ID={chat_id}, Name='{st.session_state.chat_metadata.get(chat_id, 'Unknown')}'")
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
            logger.info(f"Deleted chat file: {chat_file}")
        
        # Save updated metadata to disk
        save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
        logger.info(f"Deleted chat: ID={chat_id}")

        # If the deleted chat was the current one, switch to another or create a new one
        if is_current_chat:
            if st.session_state.chat_metadata:
                # Switch to the first available chat
                first_available_chat_id = next(iter(st.session_state.chat_metadata))
                logger.info(f"Deleted current chat. Switching to: {first_available_chat_id}")
                # Call switch_chat to handle updating session state and rerunning
                switch_chat(first_available_chat_id)
            else:
                # No other chats left, set to a "no chat" state
                logger.info("Deleted last chat. Setting to no active chat state.")
                st.session_state.current_chat_id = None
                st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}]
                st.session_state.chat_modified = False
                st.rerun() # Rerun to display the new state
        else:
            # If a non-current chat was deleted, just rerun to update the sidebar
            st.rerun()
    else:
        logger.warning(f"Attempted to delete non-existent chat ID: {chat_id}")
        # No rerun needed if chat_id wasn't found, as nothing changed.

def rename_chat(chat_id: str, new_name: str): # Modified to accept chat_id
    """Renames the specified chat."""
    if chat_id and new_name and new_name != st.session_state.chat_metadata.get(chat_id):
        st.session_state.chat_metadata[chat_id] = new_name
        save_chat_metadata(st.session_state.user_id, st.session_state.chat_metadata)
        logger.info(f"Renamed chat {chat_id} to '{new_name}'")
        st.rerun()

def get_discussion_markdown(chat_id: str) -> str:
    """Retrieves messages for a given chat_id and formats them into a Markdown string."""
    messages = st.session_state.all_chat_messages.get(chat_id, [])
    markdown_content = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        markdown_content.append(f"**{role}:**\n{content}\n\n---")
    return "\n".join(markdown_content)

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
        # If no chat is currently active (e.g., first run or after all chats deleted),
        # create a new one.
        if st.session_state.current_chat_id is None:
            logger.info("No current chat ID. Creating a new chat session for the first input.")
            create_new_chat_session_in_memory()
            # create_new_chat_session_in_memory now sets current_chat_id and saves metadata.

        # Append user message and mark chat history as modified
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        st.session_state.chat_modified = True # History is now modified

        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        formatted_history = format_chat_history(st.session_state.messages[:-1]) # Pass history *before* user's latest message for agent

        try:
            with st.spinner("ESI is thinking..."): # Spinner before starting the stream
                stream_generator = get_agent_response(prompt_to_process, chat_history=formatted_history)

            with st.chat_message("assistant"):
                # st.write_stream will render the content as it arrives and returns the full response once done.
                full_response_text = st.write_stream(stream_generator)

            st.session_state.messages.append({"role": "assistant", "content": full_response_text})
            logger.info(f"Full streamed response received and added to history: {full_response_text[:200]}...")

        except Exception as e:
            logger.error(f"Error during response streaming or generation in handle_user_input: {e}")
            error_message = f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Technical details: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)

        # Autosave the current chat history after AI response if it's_been modified
        if st.session_state.chat_modified: # This will be true here
            save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)

        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
        st.rerun()

def reset_chat_callback():
    """Resets the chat by creating a new, unsaved chat session."""
    logger.info("Resetting chat by creating a new session...")
    create_new_chat_session_in_memory() # Create new chat in memory
    st.rerun() # Rerun to display the new chat

def handle_regeneration_request():
    """Handles the request to regenerate the last assistant response."""
    if not st.session_state.get("do_regenerate", False):
        return

    st.session_state.do_regenerate = False

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'assistant':
        logger.warning("Regeneration called but last message is not from assistant or no messages exist.")
        st.rerun()
        return

    if len(st.session_state.messages) == 1:
        logger.info("Regenerating initial greeting...")
        new_greeting = generate_llm_greeting()
        st.session_state.messages[0]['content'] = new_greeting
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
        st.rerun()
        return

    logger.info("Regenerating last assistant response to user query...")
    st.session_state.messages.pop() # Remove last assistant message

    if not st.session_state.messages or st.session_state.messages[-1]['role'] != 'user':
        logger.warning("Cannot regenerate, no preceding user query found after popping assistant message.")
        st.rerun()
        return

    prompt_to_regenerate = st.session_state.messages[-1]['content']
    # formatted_history_for_regen should exclude the last assistant message AND the user message that prompted it,
    # then the user message is passed as the query.
    # However, the current get_agent_response expects the full history *including* the user query that needs a response.
    # So, we pass messages up to and including the last user message.
    history_for_regen = format_chat_history(st.session_state.messages[:-1]) # History before the assistant's message to be regenerated

    try:
        with st.spinner("ESI is rethinking..."):
            stream_generator = get_agent_response(prompt_to_regenerate, chat_history=history_for_regen)

        with st.chat_message("assistant"): # This will replace the previous one due to rerun
            full_response_text = st.write_stream(stream_generator)

        st.session_state.messages.append({"role": "assistant", "content": full_response_text})
        logger.info(f"Full regenerated streamed response received: {full_response_text[:200]}...")

    except Exception as e:
        logger.error(f"Error during response regeneration streaming: {e}")
        error_message = f"I apologize, but I encountered an error while regenerating the response. Please try again. Technical details: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        # No need to display st.error here as st.rerun() will redraw chat from messages

    save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    # Initialize LLM settings and handle potential errors outside cached function
    llm_setup_success, llm_error_msg = setup_global_llm_settings()
    if not llm_setup_success:
        st.error(llm_error_msg)
        st.stop()

    if "user_id" not in st.session_state:
        st.session_state.user_id = get_cached_user_id()
    
    if AGENT_SESSION_KEY not in st.session_state:
        # Initialize orchestrator agent and handle potential errors outside cached function
        agent_instance, agent_error_msg = setup_agent()
        if agent_error_msg:
            st.error(agent_error_msg)
            st.stop()
        st.session_state[AGENT_SESSION_KEY] = agent_instance

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
        logger.info(f"Initial load of user data for {st.session_state.user_id}: {len(st.session_state.chat_metadata)} chats.")
    else:
        logger.info(f"User data already in session state. Current chats: {len(st.session_state.chat_metadata)}.")

    # Determine the active chat or present initial greeting
    # This logic ensures a valid current chat is always selected or a new blank state is presented.
    if st.session_state.current_chat_id is None or st.session_state.current_chat_id not in st.session_state.all_chat_messages:
        if st.session_state.chat_metadata:
            # If there are existing chats, switch to the first one
            first_available_chat_id = next(iter(st.session_state.chat_metadata))
            st.session_state.current_chat_id = first_available_chat_id
            st.session_state.messages = st.session_state.all_chat_messages[first_available_chat_id]
            st.session_state.chat_modified = True # Existing chats are considered modified for saving
            logger.info(f"No valid current chat found. Switched to first available chat: '{st.session_state.chat_metadata.get(first_available_chat_id, first_available_chat_id)}'")
        else:
            # No chats exist, present a blank slate with initial greeting
            logger.info("No valid current chat found and no existing chats. Presenting initial greeting.")
            st.session_state.current_chat_id = None # Indicate no active chat ID
            st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}]
            st.session_state.chat_modified = False # This state is not yet saved to disk
    else:
        # Ensure st.session_state.messages points to the correct chat's messages
        st.session_state.messages = st.session_state.all_chat_messages[st.session_state.current_chat_id]
        st.session_state.chat_modified = True # Existing chats are considered modified for saving
        logger.info(f"Continuing with chat: '{st.session_state.chat_metadata.get(st.session_state.current_chat_id, st.session_state.current_chat_id)}'")

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
        get_discussion_markdown_callback=get_discussion_markdown
    )

    if st.session_state.suggested_prompts:
        st.markdown("---")
        st.subheader("Suggested Prompts:")
        cols = st.columns(len(st.session_state.suggested_prompts)) 
        for i, prompt in enumerate(st.session_state.suggested_prompts):
            with cols[i]:
                if st.button(prompt, key=f"suggested_prompt_btn_{i}"):
                    st.session_state.prompt_to_use = prompt
                    st.rerun()

    chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")
    handle_user_input(chat_input_value)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")
    main()
