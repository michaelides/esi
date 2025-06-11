import os
import json
import uuid
from typing import List, Dict, Any, Tuple
import streamlit as st
# import extra_streamlit_components as esc # No longer needed here
from cookie_utils import cookies # Changed import

from config import MEMORY_DIR

def _get_or_create_user_id(long_term_memory_enabled_param: bool) -> Tuple[str, str]:
    """
    Determines user ID and necessary cookie action.
    Returns a tuple: (user_id: str, cookie_action_flag: str).
    """
    print(f"LOG: session_manager._get_or_create_user_id: LTM enabled: {long_term_memory_enabled_param}")
    try:
        existing_user_id = cookies.get(cookie="user_id")
        # uuid.uuid4() is very unlikely to fail.
        if long_term_memory_enabled_param:
            if existing_user_id:
                print(f"LOG: session_manager._get_or_create_user_id: LTM ON. Found existing user_id: {existing_user_id}. Action: DO_NOTHING.")
                return existing_user_id, "DO_NOTHING"
            else:
                new_user_id = str(uuid.uuid4())
                print(f"LOG: session_manager._get_or_create_user_id: LTM ON. No existing user_id. Generated new: {new_user_id}. Action: SET_COOKIE.")
                return new_user_id, "SET_COOKIE"
        else: # Long-term memory is disabled
            temporary_user_id = str(uuid.uuid4())
            if existing_user_id:
                print(f"LOG: session_manager._get_or_create_user_id: LTM OFF. Found and will discard existing user_id: {existing_user_id}. Generated temp_id: {temporary_user_id}. Action: DELETE_COOKIE.")
                return temporary_user_id, "DELETE_COOKIE"
            else:
                print(f"LOG: session_manager._get_or_create_user_id: LTM OFF. No existing user_id. Generated temp_id: {temporary_user_id}. Action: DO_NOTHING.")
                return temporary_user_id, "DO_NOTHING"
    except Exception as e:
        # This is a fallback for unexpected errors with cookie manager or uuid, though highly unlikely.
        print(f"CRITICAL_ERROR: session_manager._get_or_create_user_id: Failed to get/create user_id. Error: {e}. Falling back to temporary UUID.")
        # Fallback to a temporary user ID if all else fails to prevent app crash.
        return str(uuid.uuid4()), "DO_NOTHING"


def _load_user_data_from_disk(user_id: str) -> Dict[str, Any]:
    """
    Loads all chat metadata and histories for a user directly from disk.
    Returns a dictionary with "metadata" and "messages" keys.
    """
    all_chat_metadata = {}
    all_chat_messages = {}

    if not user_id: # Prevent issues if user_id is somehow None or empty
        print("ERROR: session_manager._load_user_data_from_disk: user_id is None or empty. Cannot load data.")
        return {"metadata": all_chat_metadata, "messages": all_chat_messages}

    user_dir = os.path.join(MEMORY_DIR, user_id)

    try:
        os.makedirs(user_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: session_manager._load_user_data_from_disk: Could not create user directory {user_dir}. Error: {e}")
        # If user_dir can't be created, we can't load or save anything for this user.
        return {"metadata": all_chat_metadata, "messages": all_chat_messages}

    chat_metadata_path = os.path.join(user_dir, "chat_metadata.json")
    if os.path.exists(chat_metadata_path):
        try:
            with open(chat_metadata_path, "r", encoding="utf-8") as f:
                all_chat_metadata = json.load(f)
            print(f"LOG: session_manager.Loaded chat metadata for user {user_id} from disk.")
        except json.JSONDecodeError as e:
            print(f"ERROR: session_manager.Error decoding chat metadata for user {user_id} from {chat_metadata_path}: {e}. Initializing with empty metadata.")
            all_chat_metadata = {} # Reset to empty if corrupt
        except (IOError, OSError) as e:
            print(f"ERROR: session_manager.Could not read chat metadata file {chat_metadata_path} for user {user_id}: {e}. Initializing with empty metadata.")
            all_chat_metadata = {} # Reset to empty if unreadable

    # Iterate over a copy of items for safe deletion during iteration
    for chat_id, chat_name in list(all_chat_metadata.items()):
        chat_file = os.path.join(user_dir, f"{chat_id}.json")
        if os.path.exists(chat_file):
            all_chat_messages[chat_id] = None # Lazy loading marker
        else:
            print(f"LOG: session_manager.Chat file {chat_file} not found for chat ID {chat_id} (name: '{chat_name}'). Removing from metadata.")
            if chat_id in all_chat_metadata:
                del all_chat_metadata[chat_id]

    print(f"LOG: session_manager.Processed metadata for {len(all_chat_metadata)} chats for user {user_id}. Messages will be lazy-loaded.")
    return {"metadata": all_chat_metadata, "messages": all_chat_messages}


@st.cache_resource
def _initialize_user_session_data(long_term_memory_enabled_param: bool) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
    """
    Initializes user ID, loads chat data from disk (if long-term memory is enabled),
    and returns the cookie action flag.
    """
    print("LOG: session_manager._initialize_user_session_data() CALLED (Cacheable)")

    user_id, cookie_action_flag = _get_or_create_user_id(long_term_memory_enabled_param)
    print(f"LOG: session_manager._initialize_user_session_data: User ID='{user_id}', CookieAction='{cookie_action_flag}'")

    chat_metadata: Dict[str, Any] = {}
    all_chat_messages: Dict[str, Any] = {}

    if long_term_memory_enabled_param:
        if user_id: # Only load if user_id is valid
            print(f"LOG: session_manager._initialize_user_session_data: LTM ON. Loading data for user {user_id} from disk.")
            user_data = _load_user_data_from_disk(user_id)
            chat_metadata = user_data["metadata"]
            all_chat_messages = user_data["messages"]
            print(f"LOG: session_manager._initialize_user_session_data: Data load complete. Found {len(chat_metadata)} chats for user {user_id}.")
        else:
            print("ERROR: session_manager._initialize_user_session_data: LTM ON but user_id is invalid. Cannot load data.")
    else:
        print(f"LOG: session_manager._initialize_user_session_data: LTM OFF. No disk data loaded for temporary user_id {user_id}.")

    print(f"LOG: session_manager._initialize_user_session_data: User session data initialized. Final User ID: {user_id}")
    return user_id, chat_metadata, all_chat_messages, cookie_action_flag


def save_chat_history(user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    """Saves a specific chat history for a given user ID to a JSON file."""
    if not st.session_state.get("long_term_memory_enabled", False): # Safely get LTM flag
        print("LOG: session_manager.LTM disabled. Not saving chat history to disk.")
        return

    if not user_id or not chat_id:
        print(f"ERROR: session_manager.save_chat_history: Invalid user_id ('{user_id}') or chat_id ('{chat_id}'). Cannot save history.")
        return

    user_dir = os.path.join(MEMORY_DIR, user_id)
    try:
        os.makedirs(user_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: session_manager.save_chat_history: Could not create user directory {user_dir}. Error: {e}")
        return # Cannot proceed if directory creation fails

    memory_file = os.path.join(user_dir, f"{chat_id}.json")
    try:
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        print(f"LOG: session_manager.Saved chat history for chat {chat_id} (user {user_id}) to {memory_file}")
    except (IOError, OSError) as e:
        print(f"ERROR: session_manager.save_chat_history: Could not write to file {memory_file}. Error: {e}")
    except TypeError as e:
        print(f"ERROR: session_manager.save_chat_history: Could not serialize messages to JSON for chat {chat_id}. Error: {e}")
    except Exception as e: # Catch-all for other unexpected errors
        print(f"ERROR: session_manager.save_chat_history: An unexpected error occurred for chat {chat_id}. Error: {e}")


def save_chat_metadata(user_id: str, chat_metadata: Dict[str, str]):
    """Saves the chat metadata (ID to name mapping) for a user."""
    if not st.session_state.get("long_term_memory_enabled", False): # Safely get LTM flag
        print("LOG: session_manager.LTM disabled. Not saving chat metadata to disk.")
        return

    if not user_id:
        print(f"ERROR: session_manager.save_chat_metadata: Invalid user_id ('{user_id}'). Cannot save metadata.")
        return

    user_dir = os.path.join(MEMORY_DIR, user_id)
    try:
        os.makedirs(user_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: session_manager.save_chat_metadata: Could not create user directory {user_dir}. Error: {e}")
        return

    metadata_file = os.path.join(user_dir, "chat_metadata.json")
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chat_metadata, f, indent=2)
        print(f"LOG: session_manager.Saved chat metadata for user {user_id} to {metadata_file}")
    except (IOError, OSError) as e:
        print(f"ERROR: session_manager.save_chat_metadata: Could not write to file {metadata_file}. Error: {e}")
    except TypeError as e:
        print(f"ERROR: session_manager.save_chat_metadata: Could not serialize metadata to JSON for user {user_id}. Error: {e}")
    except Exception as e: # Catch-all
        print(f"ERROR: session_manager.save_chat_metadata: An unexpected error occurred for user {user_id}. Error: {e}")

print(f"DEBUG: session_manager.py processed with error handling improvements. MEMORY_DIR: {MEMORY_DIR}")
