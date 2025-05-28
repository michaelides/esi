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
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

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
    for chat_id, chat_name in list(all_chat_metadata.items()):
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    all_chat_messages[chat_id] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding chat history for chat {chat_id} (file: {chat_file}): {e}. Removing from metadata.")
                del all_chat_metadata[chat_id]
        else:
            print(f"Chat file {chat_file} not found for chat ID {chat_id}. Removing from metadata.")
            del all_chat_metadata[chat_id]
    
    print(f"Loaded {len(all_chat_messages)} chat histories for user {user_id} from disk.")
    return {"metadata": all_chat_metadata, "messages": all_chat_messages}

@st.cache_resource
def initialize_user_session_data(user_id: str) -> None:
    """
    Initializes user-specific chat data in st.session_state once per session.
    This function is cached to prevent repeated disk reads and state re-initialization.
    """
    print(f"Initializing user session data for {user_id} (cached)...")
    user_data = _load_user_data_from_disk(user_id)
    st.session_state.chat_metadata = user_data["metadata"]
    st.session_state.all_chat_messages = user_data["messages"]
    print(f"User session data initialized for {user_id}: {len(st.session_state.chat_metadata)} chats.")

    # Set up initial current chat state
    if not st.session_state.chat_metadata:
        # No chats exist, present a blank slate with initial greeting
        print("No existing chats found. Presenting initial greeting.")
        st.session_state.current_chat_id = None # Indicate no active chat ID
        st.session_state.messages = [{"role": "assistant", "content": generate_llm_greeting()}]
        st.session_state.chat_modified = False # This state is not yet saved to disk
    else:
        # If there are existing chats, switch to the first one
        first_available_chat_id = next(iter(st.session_state.chat_metadata))
        st.session_state.current_chat_id = first_available_chat_id
        st.session_state.messages = st.session_state.all_chat_messages[first_available_chat_id]
        st.session_state.chat_modified = True # Existing chats are considered modified for saving
        print(f"Switched to first available chat: '{st.session_state.chat_metadata.get(first_available_chat_id, first_available_chat_id)}'")


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

        if hasattr(agent, 'llm') and hasattr(agent.llm, 'temperature'):
            actual_llm_instance = agent.llm
            actual_llm_instance.temperature = current_temperature
        else:
            print(f"Warning: Could not access LLM object within the agent to set temperature. Agent or LLM structure might have changed (agent.llm or agent.llm.temperature not found).")

        with st.spinner("ESI is thinking..."):
            response = agent.chat(query, chat_history=chat_history)

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
    """Switches to an existing chat."""
    if chat_id not in st.session_state.all_chat_messages:
        print(f"Attempted to switch to non-existent chat ID: {chat_id}")
        return # Or handle error

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.all_chat_messages[chat_id]
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

        with st.chat_message("user"):
            st.markdown(prompt_to_process)

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
    
    if AGENT_SESSION_KEY not in st.session_state:
        st.session_state[AGENT_SESSION_KEY] = setup_agent()

    # Initialize user-specific session data (chat metadata, messages, current chat)
    # This cached function ensures this complex initialization runs only once per session.
    initialize_user_session_data(st.session_state.user_id)

    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = DEFAULT_PROMPTS

    # Initialize renaming state variables
    if 'renaming_chat_id' not in st.session_state:
        st.session_state.renaming_chat_id = None
    # The 'renaming_chat_name_input' is tied to the text_input key, so it's managed by Streamlit.
    # No explicit initialization needed here.

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
