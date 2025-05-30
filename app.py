import streamlit as st
import os
import uuid
import json
from typing import List, Dict, Any, Callable
from datetime import datetime
import pytz

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import AgentRunner

from agent import create_orchestrator_agent, generate_llm_greeting, generate_suggested_prompts
from stui import display_chat # Updated import: only display_chat is needed
from tools import UI_ACCESSIBLE_WORKSPACE_RELATIVE, UI_ACCESSIBLE_WORKSPACE

# Set page config early
st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Cookie manager for user ID persistence
# cookies = esc.CookieManager(key="esi_cookie_manager") # Commented out as esc is not defined in provided context

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories")
os.makedirs(MEMORY_DIR, exist_ok=True)

@st.cache_resource
def setup_global_llm_settings():
    """
    Sets up global LLM settings, including initial session state for chat.
    This runs only once per app deployment.
    """
    # Initialize session state variables if they don't exist
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {} # Stores metadata for all chats
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Current chat messages
    if "llm_greeting" not in st.session_state:
        st.session_state.llm_greeting = generate_llm_greeting()
    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = []
    if "last_chat_history_save" not in st.session_state:
        st.session_state.last_chat_history_save = None
    if "verbosity_level" not in st.session_state: # Initialize verbosity_level
        st.session_state.verbosity_level = 5 # Default value

@st.cache_resource
def setup_agent(verbosity_level: int) -> AgentRunner:
    """
    Sets up and caches the orchestrator agent.
    This function will re-run if verbosity_level changes, effectively re-initializing the agent.
    """
    return create_orchestrator_agent(verbosity_level=verbosity_level) # Pass verbosity_level

@st.cache_resource
def get_cached_user_id():
    """
    Retrieves or generates a unique user ID.
    Uses cookies for persistence if available, otherwise session state.
    """
    # if cookies.ready:
    #     user_id = cookies.get("user_id")
    #     if not user_id:
    #         user_id = str(uuid.uuid4())
    #         cookies.set("user_id", user_id)
    #     return user_id
    # else:
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

def _load_user_data_from_disk(user_id: str) -> Dict[str, Any]:
    """Loads chat metadata and history for a given user from disk."""
    user_memory_path = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_memory_path, exist_ok=True)

    chat_metadata_path = os.path.join(user_memory_path, "chat_metadata.json")
    chat_sessions = {}
    if os.path.exists(chat_metadata_path):
        with open(chat_metadata_path, "r") as f:
            chat_sessions = json.load(f)

    # Load messages for each chat
    for chat_id, metadata in chat_sessions.items():
        chat_history_path = os.path.join(user_memory_path, f"{chat_id}.json")
        if os.path.exists(chat_history_path):
            with open(chat_history_path, "r") as f:
                metadata["messages"] = json.load(f)
        else:
            metadata["messages"] = [] # Ensure messages key exists even if file is missing
    return chat_sessions

def save_chat_history(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    """Saves the chat history for a specific chat ID to disk."""
    user_memory_path = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_memory_path, exist_ok=True)
    chat_history_path = os.path.join(user_memory_path, f"{chat_id}.json")
    with open(chat_history_path, "w") as f:
        json.dump(messages, f, indent=4)
    st.session_state.last_chat_history_save = datetime.now(pytz.utc)

def save_chat_metadata(user_id: str, chat_metadata: Dict[str, str]):
    """Saves the chat metadata (e.g., chat names) for a user to disk."""
    user_memory_path = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_memory_path, exist_ok=True)
    chat_metadata_path = os.path.join(user_memory_path, "chat_metadata.json")
    with open(chat_metadata_path, "w") as f:
        json.dump(chat_metadata, f, indent=4)

def format_chat_history(streamlit_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Converts Streamlit's message format to LlamaIndex ChatMessage format."""
    chat_history = []
    for msg in streamlit_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        chat_history.append(ChatMessage(role=role, content=msg["content"]))
    return chat_history

def get_agent_response(query: str, chat_history: List[ChatMessage]) -> str:
    """Gets a response from the orchestrator agent."""
    agent = st.session_state[AGENT_SESSION_KEY]
    response = agent.chat(query, chat_history=chat_history)
    return response.response

def create_new_chat_session_in_memory():
    """Creates a new chat session in memory and sets it as current."""
    new_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_chat_id] = {
        "name": f"New Chat {len(st.session_state.chat_sessions) + 1}",
        "created_at": datetime.now(pytz.utc).isoformat(),
        "messages": []
    }
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = [] # Clear current messages for the new chat
    st.session_state.suggested_prompts = [] # Clear suggested prompts
    save_chat_metadata(st.session_state.user_id, {k: {"name": v["name"], "created_at": v["created_at"]} for k, v in st.session_state.chat_sessions.items()})
    st.rerun()

def switch_chat(chat_id: str):
    """Switches to an existing chat session."""
    if st.session_state.current_chat_id:
        # Save current chat history before switching
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chat_sessions[chat_id]["messages"]
    st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)
    st.rerun()

def delete_chat_session(chat_id: str):
    """Deletes a chat session from memory and disk."""
    if chat_id in st.session_state.chat_sessions:
        # Delete from disk
        user_memory_path = os.path.join(MEMORY_DIR, st.session_state.user_id)
        chat_history_path = os.path.join(user_memory_path, f"{chat_id}.json")
        if os.path.exists(chat_history_path):
            os.remove(chat_history_path)

        # Delete from memory
        del st.session_state.chat_sessions[chat_id]

        # Update metadata on disk
        save_chat_metadata(st.session_state.user_id, {k: {"name": v["name"], "created_at": v["created_at"]} for k, v in st.session_state.chat_sessions.items()})

        # If the deleted chat was the current one, switch to a new one or clear
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chat_sessions:
                # Switch to the first available chat
                first_chat_id = list(st.session_state.chat_sessions.keys())[0]
                switch_chat(first_chat_id)
            else:
                # No chats left, create a new one
                create_new_chat_session_in_memory()
        else:
            st.rerun() # Rerun to update sidebar

def rename_chat(chat_id: str, new_name: str):
    """Renames a chat session."""
    if chat_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[chat_id]["name"] = new_name
        save_chat_metadata(st.session_state.user_id, {k: {"name": v["name"], "created_at": v["created_at"]} for k, v in st.session_state.chat_sessions.items()})
        st.rerun()

def get_discussion_markdown(chat_id: str) -> str:
    """Generates markdown for a chat discussion."""
    if chat_id not in st.session_state.chat_sessions:
        return "Chat not found."

    chat_name = st.session_state.chat_sessions[chat_id]["name"]
    messages = st.session_state.chat_sessions[chat_id]["messages"]

    markdown_content = f"# Chat Discussion: {chat_name}\n\n"
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        markdown_content += f"## {role}:\n{msg['content']}\n\n"
    return markdown_content

def handle_user_input(chat_input_value: str | None):
    """Handles user input from the chat interface."""
    if chat_input_value:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": chat_input_value})

        with st.spinner("Thinking..."):
            # Get agent response
            formatted_history = format_chat_history(st.session_state.messages)
            response_content = get_agent_response(chat_input_value, formatted_history)

            # Add agent response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            # Update suggested prompts
            st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)

            # Save chat history
            save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)

        st.rerun()

def reset_chat_callback():
    """Callback for resetting the current chat."""
    if st.session_state.current_chat_id:
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
    create_new_chat_session_in_memory()

def handle_regeneration_request():
    """Handles regeneration of the last assistant response."""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        # Remove the last assistant message
        st.session_state.messages.pop()
        # Get the last user message
        last_user_message = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
        if last_user_message:
            # Remove the last user message as well, as handle_user_input will add it back
            st.session_state.messages.pop()
            handle_user_input(last_user_message)
        else:
            st.warning("No user message to regenerate from.")
    else:
        st.warning("Cannot regenerate: last message is not from assistant or no messages exist.")

def main():
    """Main function to run the Streamlit application."""
    setup_global_llm_settings()
    user_id = get_cached_user_id()

    # Load user's chat sessions from disk if not already loaded
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = _load_user_data_from_disk(user_id)
        if not st.session_state.chat_sessions:
            create_new_chat_session_in_memory()
        elif not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chat_sessions:
            # Set current chat to the most recently created one if not set or invalid
            sorted_chats = sorted(st.session_state.chat_sessions.items(), key=lambda item: item[1].get("created_at", ""), reverse=True)
            if sorted_chats:
                st.session_state.current_chat_id = sorted_chats[0][0]
                st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]
            else:
                create_new_chat_session_in_memory()
        else:
            # Ensure messages are loaded for the current chat
            st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]

    # Initialize suggested prompts if current chat has messages
    if not st.session_state.suggested_prompts and st.session_state.messages:
        st.session_state.suggested_prompts = generate_suggested_prompts(st.session_state.messages)

    # Setup the agent with the current verbosity level
    # This will re-run the cached function if st.session_state.verbosity_level changes
    agent = setup_agent(st.session_state.verbosity_level)
    st.session_state[AGENT_SESSION_KEY] = agent

    # Display sidebar and chat interface
    with st.sidebar:
        st.header("LLM Settings")
        st.session_state.verbosity_level = st.slider( # Slider for verbosity
            "Verbosity Level (V)",
            min_value=0,
            max_value=10,
            value=st.session_state.verbosity_level,
            step=1,
            help="Adjust the detail and length of the agent's responses (0: concise, 10: comprehensive)."
        )
        st.markdown("---") # Separator

        st.header("Chat Sessions")
        st.button("➕ New Chat", on_click=create_new_chat_session_in_memory, use_container_width=True)

        # Display existing chat sessions
        chat_ids = list(st.session_state.chat_sessions.keys())
        # Sort chats by creation date, newest first
        sorted_chat_ids = sorted(chat_ids, key=lambda cid: st.session_state.chat_sessions[cid].get("created_at", ""), reverse=True)

        for chat_id in sorted_chat_ids:
            chat_name = st.session_state.chat_sessions[chat_id]["name"]
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col1:
                if st.button(chat_name, key=f"chat_btn_{chat_id}", use_container_width=True,
                             type="primary" if chat_id == st.session_state.current_chat_id else "secondary"):
                    switch_chat(chat_id)
            with col2:
                # Using a popover for rename to avoid immediate text input on button click
                with st.popover("✏️", use_container_width=True, help="Rename chat"):
                    st.write(f"Rename: **{chat_name}**")
                    new_name = st.text_input("New name:", value=chat_name, key=f"rename_input_{chat_id}", label_visibility="collapsed")
                    if st.button("Save Name", key=f"save_rename_btn_{chat_id}"):
                        rename_chat(chat_id, new_name)
                        st.rerun() # Rerun to close popover and update name
            with col3:
                if st.button("🗑️", key=f"delete_btn_{chat_id}", help="Delete chat"):
                    delete_chat_session(chat_id)
        st.markdown("---")

        # Download current chat discussion
        if st.session_state.current_chat_id:
            discussion_markdown = get_discussion_markdown(st.session_state.current_chat_id)
            st.download_button(
                label="Download Current Chat (.md)",
                data=discussion_markdown,
                file_name=f"{st.session_state.chat_sessions[st.session_state.current_chat_id]['name'].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        st.markdown("---")
        st.caption(f"User ID: {user_id}")
        # if cookies.ready:
        #     cookies.save() # Save cookies at the end of the script run

    # Apply CSS globally (moved from stui.py as stui.create_interface is removed)
    CSS = """
    .stExpander > details {
        border: none;
    }
    """
    st.html(f"<style>{CSS}</style>")

    # Display the main chat interface
    display_chat(
        download_marker=DOWNLOAD_MARKER,
        rag_source_marker_prefix=RAG_SOURCE_MARKER_PREFIX,
        ui_accessible_workspace_relative=UI_ACCESSIBLE_WORKSPACE_RELATIVE
    )

    # Handle regeneration request if triggered by the button in display_chat
    if st.session_state.get("do_regenerate", False):
        st.session_state.do_regenerate = False # Reset flag immediately
        handle_regeneration_request() # This will rerun the app

    # Chat input and suggested prompts are below the chat history
    if st.session_state.suggested_prompts:
        st.markdown("---")
        st.subheader("Suggested Prompts:")
        cols = st.columns(len(st.session_state.suggested_prompts)) 
        for i, prompt in enumerate(st.session_state.suggested_prompts):
            with cols[i]:
                if st.button(prompt, key=f"suggested_prompt_btn_{i}"):
                    handle_user_input(prompt) # Directly call handler with suggested prompt

    chat_input_value = st.chat_input("Ask me about dissertations, research methods, academic writing, etc.")
    handle_user_input(chat_input_value)

if __name__ == "__main__":
    main()
