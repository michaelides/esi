import streamlit as st
import os
import uuid
import json
import time
from typing import List, Dict, Any, Callable
from datetime import datetime, timedelta # Added timedelta for cookie expiration

from llama_index.core.llms import ChatMessage
from llama_index.core.agent import AgentRunner
from llama_index.core.memory import ChatMemoryBuffer

from agent import create_orchestrator_agent, generate_llm_greeting, generate_suggested_prompts
from st_pages import add_page_title, hide_pages
import extra_streamlit_components as esc

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

cookies = esc.CookieManager(key="esi_cookie_manager")

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories")

@st.cache_resource
def setup_global_llm_settings():
    """
    Sets up global LLM settings in the sidebar.
    """
    with st.sidebar:
        st.subheader("LLM Settings")
        # Add a slider for verbosity
        st.session_state.llm_verbosity = st.slider(
            "Verbosity Level",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Controls how verbose the LLM's responses are. Higher values mean more detailed responses."
        )
        # Placeholder for other settings if needed
        # st.session_state.llm_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        # st.session_state.llm_top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        # st.session_state.llm_max_tokens = st.slider("Max Tokens", min_value=100, max_value=4096, value=1024, step=100)

@st.cache_resource
def setup_agent():
    """
    Initializes and caches the orchestrator agent.
    """
    return create_orchestrator_agent()

@st.cache_resource
def get_cached_user_id():
    """
    Retrieves or generates a user ID and caches it.
    """
    user_id = cookies.get_cookie("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        cookies.set_cookie("user_id", user_id, expires_at=datetime.now() + timedelta(days=365))
    return user_id

def _load_user_data_from_disk(user_id: str) -> Dict[str, Any]:
    """
    Loads all chat data for a given user from disk.
    """
    user_dir = os.path.join(MEMORY_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        return {"chats": {}, "metadata": {}}

    chat_metadata_path = os.path.join(user_dir, "chat_metadata.json")
    chat_metadata = {}
    if os.path.exists(chat_metadata_path):
        with open(chat_metadata_path, "r") as f:
            chat_metadata = json.load(f)

    chats = {}
    for chat_id in chat_metadata.keys():
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            with open(chat_file, "r") as f:
                chats[chat_id] = json.load(f)
    return {"chats": chats, "metadata": chat_metadata}

def save_chat_history(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    """
    Saves the chat history for a specific chat ID to disk.
    """
    user_dir = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    chat_file = os.path.join(user_dir, f"{chat_id}.json")
    with open(chat_file, "w") as f:
        json.dump(messages, f, indent=4)

def save_chat_metadata(user_id: str, chat_metadata: Dict[str, str]):
    """
    Saves the chat metadata for a user to disk.
    """
    user_dir = os.path.join(MEMORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    chat_metadata_path = os.path.join(user_dir, "chat_metadata.json")
    with open(chat_metadata_path, "w") as f:
        json.dump(chat_metadata, f, indent=4)

def format_chat_history(streamlit_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """
    Formats Streamlit chat messages into LlamaIndex ChatMessage format.
    """
    chat_history = []
    for message in streamlit_messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            chat_history.append(ChatMessage(role="user", content=content))
        elif role == "assistant":
            chat_history.append(ChatMessage(role="assistant", content=content))
    return chat_history

def get_agent_response(query: str, chat_history: List[ChatMessage]) -> str:
    """
    Gets a response from the orchestrator agent.
    """
    agent = st.session_state[AGENT_SESSION_KEY]
    # Update agent's chat memory with current history
    agent.memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history)
    response = agent.query(query)
    return str(response)

def create_new_chat_session_in_memory():
    """
    Creates a new chat session in memory and sets it as the current one.
    """
    new_chat_id = str(uuid.uuid4())
    st.session_state.user_data["chats"][new_chat_id] = []
    st.session_state.user_data["metadata"][new_chat_id] = {
        "name": f"New Chat {len(st.session_state.user_data['chats'])}",
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_chat_id = new_chat_id
    save_chat_metadata(st.session_state.user_id, st.session_state.user_data["metadata"])
    st.rerun()

def switch_chat(chat_id: str):
    """
    Switches the current chat session.
    """
    st.session_state.current_chat_id = chat_id
    st.rerun()

def delete_chat_session(chat_id: str):
    """
    Deletes a chat session from memory and disk.
    """
    if chat_id in st.session_state.user_data["chats"]:
        del st.session_state.user_data["chats"][chat_id]
        del st.session_state.user_data["metadata"][chat_id]
        user_dir = os.path.join(MEMORY_DIR, st.session_state.user_id)
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)
        save_chat_metadata(st.session_state.user_id, st.session_state.user_data["metadata"])

        # If the deleted chat was the current one, switch to a new or existing one
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.user_data["chats"]:
                st.session_state.current_chat_id = list(st.session_state.user_data["chats"].keys())[0]
            else:
                create_new_chat_session_in_memory() # Create a new chat if no others exist
        st.rerun()

def rename_chat(chat_id: str, new_name: str):
    """
    Renames a chat session.
    """
    if chat_id in st.session_state.user_data["metadata"]:
        st.session_state.user_data["metadata"][chat_id]["name"] = new_name
        save_chat_metadata(st.session_state.user_id, st.session_state.user_data["metadata"])
        st.rerun()

def get_discussion_markdown(chat_id: str) -> str:
    """
    Generates markdown for a specific chat discussion.
    """
    messages = st.session_state.user_data["chats"].get(chat_id, [])
    markdown_content = ""
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        markdown_content += f"**{role}:**\n{content}\n\n"
    return markdown_content

def handle_user_input(chat_input_value: str | None):
    """
    Handles user input from the chat interface.
    """
    if chat_input_value:
        current_messages = st.session_state.user_data["chats"][st.session_state.current_chat_id]
        current_messages.append({"role": "user", "content": chat_input_value})
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, current_messages)

        with st.chat_message("user"):
            st.markdown(chat_input_value)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                formatted_history = format_chat_history(current_messages)
                response = get_agent_response(chat_input_value, formatted_history)
                st.markdown(response)
            current_messages.append({"role": "assistant", "content": response})
            save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, current_messages)

def reset_chat_callback():
    """
    Callback for resetting the current chat.
    """
    if st.session_state.current_chat_id in st.session_state.user_data["chats"]:
        st.session_state.user_data["chats"][st.session_state.current_chat_id] = []
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, [])
    st.rerun()

def handle_regeneration_request():
    """
    Handles the request to regenerate the last assistant response.
    """
    current_messages = st.session_state.user_data["chats"][st.session_state.current_chat_id]
    if not current_messages:
        st.warning("No messages to regenerate.")
        return

    # Remove the last assistant message if it exists
    if current_messages[-1]["role"] == "assistant":
        current_messages.pop()
        if not current_messages: # If only assistant message was there, reset completely
            reset_chat_callback()
            return
        last_user_query = current_messages[-1]["content"]
    elif current_messages[-1]["role"] == "user":
        last_user_query = current_messages[-1]["content"]
    else:
        st.warning("Cannot regenerate. Last message is not a user or assistant message.")
        return

    save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, current_messages)

    # Re-run the query for the last user message
    with st.chat_message("assistant"):
        with st.spinner("Regenerating response..."):
            formatted_history = format_chat_history(current_messages)
            response = get_agent_response(last_user_query, formatted_history)
            st.markdown(response)
        current_messages.append({"role": "assistant", "content": response})
        save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, current_messages)
    st.rerun()


def main():
    add_page_title()
    hide_pages(["Thank you"])

    # Initialize session state variables if they don't exist
    if "user_id" not in st.session_state:
        st.session_state.user_id = get_cached_user_id()
        st.session_state.user_data = _load_user_data_from_disk(st.session_state.user_id)
        if not st.session_state.user_data["chats"]:
            create_new_chat_session_in_memory()
        else:
            # Set the current chat to the most recently created one or the first one
            sorted_chats = sorted(st.session_state.user_data["metadata"].items(),
                                  key=lambda item: item[1].get("created_at", ""), reverse=True)
            st.session_state.current_chat_id = sorted_chats[0][0] if sorted_chats else None

    if AGENT_SESSION_KEY not in st.session_state:
        st.session_state[AGENT_SESSION_KEY] = setup_agent()

    # Setup global LLM settings in the sidebar
    setup_global_llm_settings()

    # Display chat interface
    from stui import create_interface
    create_interface(
        reset_callback=reset_chat_callback,
        new_chat_callback=create_new_chat_session_in_memory,
        delete_chat_callback=delete_chat_session,
        rename_chat_callback=rename_chat,
        chat_metadata=st.session_state.user_data["metadata"],
        current_chat_id=st.session_state.current_chat_id,
        switch_chat_callback=switch_chat,
        get_discussion_markdown_callback=get_discussion_markdown
    )

    # Display chat messages from history
    current_messages = st.session_state.user_data["chats"].get(st.session_state.current_chat_id, [])
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initial greeting if chat is empty
    if not current_messages:
        with st.chat_message("assistant"):
            greeting_message = generate_llm_greeting()
            st.markdown(greeting_message)
            current_messages.append({"role": "assistant", "content": greeting_message})
            save_chat_history(st.session_state.user_id, st.session_state.current_chat_id, current_messages)

    # Chat input
    chat_input_key = f"chat_input_{st.session_state.current_chat_id}"
    chat_input_value = st.chat_input("Ask me anything...", key=chat_input_key)
    handle_user_input(chat_input_value)

    # Suggested prompts
    if not current_messages or (len(current_messages) == 1 and current_messages[0]["role"] == "assistant"):
        suggested_prompts = generate_suggested_prompts(current_messages)
        st.markdown("---")
        st.markdown("### Suggested Prompts:")
        cols = st.columns(2)
        for i, prompt in enumerate(suggested_prompts):
            with cols[i % 2]:
                if st.button(prompt, key=f"suggested_prompt_{i}"):
                    handle_user_input(prompt)
                    st.rerun()

    # Regeneration button
    if current_messages and current_messages[-1]["role"] == "assistant":
        st.button("🔄 Regenerate Response", on_click=handle_regeneration_request)


if __name__ == "__main__":
    main()
