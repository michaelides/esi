import streamlit as st
import datetime
import os
import re
import json
from typing import List, Dict, Any, Optional, Callable

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def display_chat():
    """Display the chat messages from the session state, handling file downloads and image display."""
    CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
    RAG_SOURCE_MARKER = "---RAG_SOURCE---"
    
    CODE_WORKSPACE_RELATIVE = "./code_interpreter_ws"
    code_workspace_absolute = os.path.join(PROJECT_ROOT, CODE_WORKSPACE_RELATIVE)
    os.makedirs(code_workspace_absolute, exist_ok=True)

    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            text_to_display = content
            rag_sources_data = []
            code_download_filename = None
            code_download_filepath_relative = None
            code_is_image = False

            if message["role"] == "assistant":
                # --- 1. Extract RAG sources using regex ---
                # Regex to find RAG_SOURCE_MARKER followed by a JSON object
                # The JSON object is captured in group 1
                rag_source_pattern = re.compile(rf"{re.escape(RAG_SOURCE_MARKER)}({{\"type\":.*?}})", re.DOTALL)
                
                # Find all matches
                all_rag_matches = list(rag_source_pattern.finditer(text_to_display))
                
                # Extract JSON data and remove markers from text_to_display
                processed_text_after_rag = text_to_display
                for match in reversed(all_rag_matches): # Process in reverse to avoid index issues
                    json_str = match.group(1)
                    try:
                        rag_data = json.loads(json_str)
                        rag_sources_data.append(rag_data)
                        print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")
                    
                    # Remove the entire matched marker and JSON from the text
                    processed_text_after_rag = processed_text_after_rag[:match.start()] + processed_text_after_rag[match.end():]
                
                text_to_display = processed_text_after_rag.strip()

                # --- 2. Extract Code Interpreter download marker ---
                code_marker_match = re.search(rf"^{re.escape(CODE_DOWNLOAD_MARKER)}(.*)$", text_to_display, re.MULTILINE | re.IGNORECASE)
                if code_marker_match:
                    extracted_filename = code_marker_match.group(1).strip()
                    text_to_display = text_to_display[:code_marker_match.start()].strip() + text_to_display[code_marker_match.end():].strip()
                    
                    print(f"Found code download marker. Filename: {extracted_filename}")
                    code_download_filename = extracted_filename
                    code_download_filepath_relative = os.path.join(CODE_WORKSPACE_RELATIVE, extracted_filename)

                    code_download_filepath_absolute = os.path.join(PROJECT_ROOT, code_download_filepath_relative)

                    if extracted_filename and os.path.exists(code_download_filepath_absolute):
                        print(f"Code download file exists at: {code_download_filepath_absolute}")
                        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
                        if os.path.splitext(code_download_filename)[1].lower() in image_extensions:
                            code_is_image = True
                            print(f"Detected image file from code interpreter: {code_download_filename}")
                    else:
                        print(f"Code download file '{extracted_filename}' NOT found at '{code_download_filepath_absolute}'.")
                        text_to_display += f"\n\n*(Warning: The file '{extracted_filename}' mentioned for download could not be found.)*"

            # --- 3. Display main text content ---
            if text_to_display:
                st.markdown(text_to_display)

            # --- 4. Display RAG sources (PDFs and Web Links) - Deduplicated ---
            displayed_rag_identifiers = set()
            # Sort rag_sources_data to ensure consistent display order if multiple sources
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
                        pdf_absolute_path = os.path.join(PROJECT_ROOT, pdf_relative_path) if pdf_relative_path else None

                        if pdf_absolute_path and os.path.exists(pdf_absolute_path):
                            try:
                                citation_num = rag_data.get('citation_number')
                                citation_prefix = f"[{citation_num}] " if citation_num else ""
                                button_label = f"{citation_prefix}Download PDF: {pdf_name}"

                                with open(pdf_absolute_path, "rb") as fp:
                                    st.download_button(
                                        label=button_label,
                                        data=fp,
                                        file_name=pdf_name,
                                        mime="application/pdf",
                                        key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}"
                                    )
                                print(f"Added download button for RAG PDF: {button_label} (Path: {pdf_absolute_path})")
                                display_item = True
                            except Exception as e:
                                st.error(f"Error creating download button for {pdf_name}: {e}")
                                print(f"Error for RAG PDF '{pdf_name}': {e}")
                        elif pdf_relative_path:
                            st.warning(f"Referenced PDF '{pdf_name}' not found.")
                            print(f"Warning: Referenced PDF '{pdf_name}' not found at expected absolute path: {pdf_absolute_path}")
                
                elif source_type == "web":
                    url = rag_data.get("url")
                    title = rag_data.get("title", url)
                    identifier = url
                    if identifier and identifier not in displayed_rag_identifiers:
                        if url:
                            st.markdown(f"Source: [{title}]({url})")
                            print(f"Added link for RAG web source: {title} (URL: {url})")
                            display_item = True
                
                if display_item and identifier:
                    displayed_rag_identifiers.add(identifier)
                    st.divider()

            # --- 5. Display Code Interpreter output (Image or Download Button) ---
            code_download_absolute_filepath = os.path.join(PROJECT_ROOT, code_download_filepath_relative) if code_download_filepath_relative else None

            if code_is_image and code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath):
                try:
                    st.image(code_download_absolute_filepath, caption=code_download_filename, use_container_width=True)
                    print(f"Successfully displayed image from code interpreter: {code_download_filename}")
                except Exception as e:
                    st.error(f"Error displaying image {code_download_filename}: {e}")
                    code_is_image = False
            
            if code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath) and not code_is_image:
                try:
                    with open(code_download_absolute_filepath, "rb") as fp:
                        st.download_button(
                            label=f"Download {code_download_filename}",
                            data=fp,
                            file_name=code_download_filename,
                            mime="application/octet-stream",
                            key=f"code_dl_{msg_idx}_{code_download_filename}"
                        )
                    print(f"Successfully added download button for code interpreter file: {code_download_filename}")
                except Exception as e:
                    st.error(f"Error creating download button for {code_download_filename}: {e}")

            if message["role"] == "assistant" and msg_idx == len(st.session_state.messages) - 1:
                can_regenerate = False
                if len(st.session_state.messages) == 1:
                    can_regenerate = True
                elif len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    can_regenerate = True
                
                if can_regenerate:
                    if st.button("🔄", key=f"regenerate_{msg_idx}", help="Regenerate Response"):
                        st.session_state.do_regenerate = True
                        st.rerun()


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

            st.markdown("---") # Separator for new chat button and list

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
                    with st.popover("⋮", use_container_width=True):
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
