import streamlit as st
import datetime
import os
import re
import json
from typing import List, Dict, Any, Optional, Callable
from config import PROJECT_ROOT, CODE_DOWNLOAD_MARKER, RAG_SOURCE_MARKER_PREFIX, CODE_WORKSPACE_RELATIVE_PATH, get_logger

logger = get_logger(__name__)

# Helper function to parse assistant message content
def _parse_assistant_message_content(content: str, code_workspace_absolute: str, project_root: str):
    text_to_display = content
    rag_sources_data = []
    
    # Initialize code_download_info with default values
    code_download_info = {
        "filename": None,
        "filepath_relative": None,
        "filepath_absolute": None,
        "is_image": False,
        "exists": False,
        "warning_message": None
    }

    # --- 1. Extract RAG sources using regex ---
    rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER_PREFIX)}({{\"type\":.*?}})", re.DOTALL)
    all_rag_matches = list(rag_source_pattern.finditer(text_to_display))
    processed_text_after_rag = text_to_display
    for match in reversed(all_rag_matches):
        json_str = match.group(1)
        try:
            rag_data = json.loads(json_str)
            rag_sources_data.append(rag_data)
            logger.info(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not decode RAG source JSON: '{json_str}'. Error: {e}")
        processed_text_after_rag = processed_text_after_rag[:match.start()] + processed_text_after_rag[match.end():]
    text_to_display = processed_text_after_rag.strip()

    # --- 2. Extract Code Interpreter download marker ---
    code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
    if code_marker_match:
        extracted_filename = code_marker_match.group(1).strip()
        text_to_display = text_to_display[:code_marker_match.start()].strip() + text_to_display[code_marker_match.end():].strip()

        logger.info(f"Found code download marker. Filename: {extracted_filename}")
        code_download_info["filename"] = extracted_filename
        code_download_info["filepath_relative"] = os.path.join(CODE_WORKSPACE_RELATIVE_PATH, extracted_filename)
        code_download_info["filepath_absolute"] = os.path.join(project_root, code_download_info["filepath_relative"])

        if extracted_filename and os.path.exists(code_download_info["filepath_absolute"]):
            logger.info(f"Code download file exists at: {code_download_info['filepath_absolute']}")
            code_download_info["exists"] = True
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
            if os.path.splitext(extracted_filename)[1].lower() in image_extensions:
                code_download_info["is_image"] = True
                logger.info(f"Detected image file from code interpreter: {extracted_filename}")
        else:
            warning_msg = f"Code download file '{extracted_filename}' NOT found at '{code_download_info['filepath_absolute']}'."
            logger.warning(warning_msg)
            code_download_info["warning_message"] = f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"

    return {
        "processed_text": text_to_display,
        "rag_sources_data": rag_sources_data,
        "code_download_info": code_download_info
    }

# Helper function to display RAG sources
def _display_rag_sources_ui(rag_sources_data: list, msg_idx: int, project_root: str):
    if not rag_sources_data:
        return

    displayed_rag_identifiers = set()
    # Sort rag_sources_data to ensure consistent display order
    rag_sources_data.sort(key=lambda x: x.get('citation_number', float('inf')) if x.get('type') == 'pdf' else x.get('url', x.get('title', '')))

    for rag_idx, rag_data in enumerate(rag_sources_data):
        source_type = rag_data.get("type")
        identifier = None
        display_item = False

        if source_type == "pdf":
            pdf_name = rag_data.get("name", "source.pdf")
            pdf_relative_path = rag_data.get("path")
            identifier = pdf_relative_path
            if identifier and identifier not in displayed_rag_identifiers:
                pdf_absolute_path = os.path.join(project_root, pdf_relative_path) if pdf_relative_path else None
                if pdf_absolute_path and os.path.exists(pdf_absolute_path):
                    try:
                        citation_num = rag_data.get('citation_number')
                        citation_prefix = f"[{citation_num}] " if citation_num else ""
                        button_label = f"{citation_prefix}Download PDF: {pdf_name}"
                        with open(pdf_absolute_path, "rb") as fp:
                            st.download_button(
                                label=button_label, data=fp, file_name=pdf_name,
                                mime="application/pdf", key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}"
                            )
                        logger.info(f"Added download button for RAG PDF: {button_label} (Path: {pdf_absolute_path})")
                        display_item = True
                    except Exception as e:
                        st.error(f"Error creating download button for {pdf_name}: {e}")
                        logger.error(f"Error for RAG PDF '{pdf_name}': {e}")
                elif pdf_relative_path:
                    st.warning(f"Referenced PDF '{pdf_name}' not found.")
                    logger.warning(f"Referenced PDF '{pdf_name}' not found at expected absolute path: {pdf_absolute_path if pdf_absolute_path else 'N/A'}")

        elif source_type == "web":
            url = rag_data.get("url")
            title = rag_data.get("title", url)
            identifier = url
            if identifier and identifier not in displayed_rag_identifiers:
                if url:
                    st.markdown(f"Source: [{title}]({url})")
                    logger.info(f"Added link for RAG web source: {title} (URL: {url})")
                    display_item = True

        if display_item and identifier:
            displayed_rag_identifiers.add(identifier)
            st.divider()

# Helper function to display code output
def _display_code_output_ui(code_download_info: dict, msg_idx: int):
    if not code_download_info or not code_download_info.get("filename"):
        return

    filename = code_download_info["filename"]
    absolute_filepath = code_download_info["filepath_absolute"]
    is_image = code_download_info["is_image"]
    exists = code_download_info["exists"]

    if is_image and exists:
        try:
            st.image(absolute_filepath, caption=filename, use_container_width=True)
            logger.info(f"Successfully displayed image from code interpreter: {filename}")
        except Exception as e:
            st.error(f"Error displaying image {filename}: {e}")
            logger.error(f"Error displaying image {filename}: {e}")
    elif exists: # Not an image but exists, so offer download
        try:
            with open(absolute_filepath, "rb") as fp:
                st.download_button(
                    label=f"Download {filename}", data=fp, file_name=filename,
                    mime="application/octet-stream", key=f"code_dl_{msg_idx}_{filename}"
                )
            logger.info(f"Successfully added download button for code interpreter file: {filename}")
        except Exception as e:
            st.error(f"Error creating download button for {filename}: {e}")
            logger.error(f"Error creating download button for {filename}: {e}")
    # Warning for non-existent file already handled in _parse_assistant_message_content and added to processed_text

# Helper function to display regenerate button
def _display_regenerate_button_ui(message: dict, msg_idx: int):
    if message["role"] == "assistant" and msg_idx == len(st.session_state.messages) - 1:
        can_regenerate = (len(st.session_state.messages) == 1 or
                          (len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user"))
        if can_regenerate:
            if st.button("🔄", key=f"regenerate_{msg_idx}", help="Regenerate Response"):
                st.session_state.do_regenerate = True
                st.rerun()

# Refactored display_chat function
def display_chat():
    """Display the chat messages from the session state, handling file downloads and image display."""
    code_workspace_absolute = os.path.join(PROJECT_ROOT, CODE_WORKSPACE_RELATIVE_PATH)
    os.makedirs(code_workspace_absolute, exist_ok=True)

    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            if message["role"] == "assistant":
                parsed_content = _parse_assistant_message_content(content, code_workspace_absolute, PROJECT_ROOT)
                
                text_to_display = parsed_content["processed_text"]
                if parsed_content["code_download_info"].get("warning_message"):
                     text_to_display += parsed_content["code_download_info"]["warning_message"]

                if text_to_display:
                    st.markdown(text_to_display)

                _display_rag_sources_ui(parsed_content["rag_sources_data"], msg_idx, PROJECT_ROOT)
                _display_code_output_ui(parsed_content["code_download_info"], msg_idx)
                _display_regenerate_button_ui(message, msg_idx)
            else: # User message
                st.markdown(content)

def create_interface(
    reset_callback: Callable,
    new_chat_callback: Callable,
    delete_chat_callback: Callable,
    rename_chat_callback: Callable,
    chat_metadata: Dict[str, str],
    current_chat_id: str,
    switch_chat_callback: Callable,
    get_discussion_markdown_callback: Callable
):
    """Create the Streamlit UI for the chat interface."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")

    # Initialize editing state if not present
    if 'editing_chat_id' not in st.session_state:
        st.session_state.editing_chat_id = None

    with st.sidebar:
        with st.expander("**Chat History**", expanded=True, icon = ":material/forum:"):
            st.info("Conversations are automatically saved and linked to your browser via cookies. Clearing browser data will remove your saved discussions.")
            
            if st.button("➕ New Chat", key="new_chat_button", use_container_width=True):
                # Clear any active editing state when creating a new chat
                st.session_state.editing_chat_id = None
                new_chat_callback()

            # Display existing chats
            sorted_chat_items = sorted(chat_metadata.items(), key=lambda item: item[1].lower())
            
            for chat_id, chat_name in sorted_chat_items:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.session_state.editing_chat_id == chat_id:
                        # Display text input for renaming
                        new_name = st.text_input(
                            "New name:",
                            value=chat_name,
                            key=f"rename_input_{chat_id}",
                            label_visibility="collapsed",
                            on_change=lambda current_chat_id_in_loop=chat_id: (
                                rename_chat_callback(current_chat_id_in_loop, st.session_state[f"rename_input_{current_chat_id_in_loop}"]) if st.session_state[f"rename_input_{current_chat_id_in_loop}"] and st.session_state[f"rename_input_{current_chat_id_in_loop}"] != chat_metadata.get(current_chat_id_in_loop) else None,
                                setattr(st.session_state, 'editing_chat_id', None) # Clear editing state
                            )
                        )
                    else:
                        # Display button for switching chat
                        if st.button(chat_name, key=f"chat_select_{chat_id}", use_container_width=True,
                                    type="primary" if chat_id == current_chat_id else "secondary"):
                            if chat_id != current_chat_id:
                                # Clear any active editing state when switching chats
                                st.session_state.editing_chat_id = None
                                switch_chat_callback(chat_id)
                with col2:
#                    with st.popover("⋮", use_container_width=True):
                    with st.popover("", use_container_width=True):
                        st.write(f"Options for: **{chat_name}**")
                        
                        # Option to download
                        st.download_button(
                            label="⬇️ Download (.md)",
                            data=get_discussion_markdown_callback(chat_id),
                            file_name=f"{chat_name.replace(' ', '_')}.md",
                            mime="text/markdown",
                            key=f"download_listed_{chat_id}",
                            use_container_width=True
                        )
                        
                        # Option to rename (sets editing_chat_id and reruns)
                        if st.button("✏️ Rename", key=f"rename_btn_{chat_id}", use_container_width=True):
                            st.session_state.editing_chat_id = chat_id
                            st.rerun() # Rerun to show the input field

                        # Option to delete
                        if st.button("♻ Delete", key=f"delete_from_popover_{chat_id}", use_container_width=True):
                            # Clear any active editing state if the chat being edited is deleted
                            if st.session_state.editing_chat_id == chat_id:
                                st.session_state.editing_chat_id = None
                            delete_chat_callback(chat_id)

        with st.expander("**LLM Settings**", expanded=False, icon = ":material/tune:"):
            st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("llm_temperature", 0.7),
                step=0.1,
                key="llm_temperature",
                help="Controls the randomness of the AI's responses. Lower values are more focused, higher values are more creative."
            )
            st.slider(
                "Verbosity",
                min_value=1,
                max_value=5,
                value=st.session_state.get("llm_verbosity", 3),
                step=1,
                key="llm_verbosity",
                help="Controls the detail level of the AI's responses. 1 is concise, 5 is very detailed."
            )

        with st.expander("**About ESI**", expanded=False, icon = ":material/info:"):
            st.info("ESI uses AI to help you navigate the dissertation process. It has access to some of the literature in your reading lists and also uses search tools for web lookups.")
            st.warning("⚠️  Remember: Always consult your dissertation supervisor for final guidance and decisions.")
            st.info("Made for NBS7091A and NBS7095x")

    # Apply CSS globally
    CSS = """
    .stExpander > details {
        border: none;
    }
    """
    st.html(f"<style>{CSS}</style>")

    display_chat()
