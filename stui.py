import streamlit as st
import os
import re
import json
from typing import List, Dict, Any, Optional, Callable
import html # Import html for escaping HTML content
import pandas as pd
from PyPDF2 import PdfReader # Added for PDF processing in stui.py
from docx import Document # Added for DOCX processing in stui.py
import io # Added for BytesIO in stui.py
import pyreadstat # Ensure pyreadstat is imported if read_spss is used
import app # Import the app module to access process_user_prompt_and_get_response
import extra_streamlit_components as esc # Added for cookies

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
cookies = esc.CookieManager(key="esi_cookie_manager") # Added cookies manager

# Import UI_ACCESSIBLE_WORKSPACE from tools.py
from tools import UI_ACCESSIBLE_WORKSPACE

# Initialize session state for clipboard functionality
if 'text_to_copy_payload' not in st.session_state:
    st.session_state.text_to_copy_payload = None
if 'clipboard_triggered_for_id' not in st.session_state:
    st.session_state.clipboard_triggered_for_id = None

st.set_page_config(
    page_title="ESI - ESI Scholarly Instructor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_chat():
    """Display the chat messages from the session state, handling file downloads and image display."""
    CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
    RAG_SOURCE_MARKER = "---RAG_SOURCE---"
    
    # Use the imported UI_ACCESSIBLE_WORKSPACE directly
    os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)

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
                        # print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}") # Removed verbose log
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
                    
                    # print(f"Found code download marker. Filename: {extracted_filename}") # Removed verbose log
                    code_download_filename = extracted_filename
                    # Use UI_ACCESSIBLE_WORKSPACE for relative path construction
                    code_download_filepath_relative = os.path.relpath(os.path.join(UI_ACCESSIBLE_WORKSPACE, extracted_filename), PROJECT_ROOT)

                    code_download_filepath_absolute = os.path.join(PROJECT_ROOT, code_download_filepath_relative)

                    if extracted_filename and os.path.exists(code_download_filepath_absolute):
                        # print(f"Code download file exists at: {code_download_filepath_absolute}") # Removed verbose log
                        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
                        if os.path.splitext(code_download_filename)[1].lower() in image_extensions:
                            code_is_image = True
                            # print(f"Detected image file from code interpreter: {code_download_filename}") # Removed verbose log
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
                    pdf_source_path = rag_data.get("path") # This will be either http:// or file://
                    citation_num = rag_data.get('citation_number')
                    citation_prefix = f"[{citation_num}] " if citation_num else ""
                    
                    identifier = pdf_source_path # Use the path/URL as identifier for deduplication

                    if identifier and identifier not in displayed_rag_identifiers:
                        if pdf_source_path and pdf_source_path.startswith("http"):
                            # It's a Hugging Face URL, display as a link
                            st.markdown(f"Source: {citation_prefix}[{pdf_name}]({pdf_source_path})")
                            # print(f"Added link for RAG PDF (URL): {citation_prefix}{pdf_name}") # Removed verbose log
                            display_item = True
                        elif pdf_source_path and pdf_source_path.startswith("file://"):
                            # It's a local file path with 'file://' prefix
                            local_file_path = pdf_source_path[len("file://"):] # Remove 'file://'
                            # The path from tools.py is already absolute, so no need to join with PROJECT_ROOT
                            pdf_absolute_path = local_file_path 

                            if os.path.exists(pdf_absolute_path):
                                try:
                                    button_label = f"{citation_prefix}Download PDF: {pdf_name}"
                                    with open(pdf_absolute_path, "rb") as fp:
                                        st.download_button(
                                            label=button_label,
                                            data=fp,
                                            file_name=pdf_name,
                                            mime="application/pdf",
                                            key=f"rag_pdf_{msg_idx}_{rag_idx}_{pdf_name}"
                                        )
                                    # print(f"Added download button for RAG PDF (local file://): {button_label} (Path: {pdf_absolute_path})") # Removed verbose log
                                    display_item = True
                                except Exception as e:
                                    st.error(f"Error creating download button for {pdf_name}: {e}")
                                    print(f"Error for RAG PDF '{pdf_name}': {e}")
                            else:
                                st.warning(f"Referenced PDF '{pdf_name}' not found locally at '{pdf_absolute_path}'.")
                                print(f"Warning: Referenced PDF '{pdf_name}' not found at expected absolute path: {pdf_absolute_path}")
                        else:
                            # Unexpected path format
                            st.warning(f"Referenced PDF '{pdf_name}' has an unsupported path format: '{pdf_source_path}'.")
                            print(f"Warning: Referenced PDF '{pdf_name}' has an unsupported path format: '{pdf_source_path}'.")
                
                elif source_type == "web":
                    url = rag_data.get("url")
                    title = rag_data.get("title", url)
                    identifier = url
                    if identifier and identifier not in displayed_rag_identifiers:
                        if url:
                            st.markdown(f"Source: [{title}]({url})")
                            # print(f"Added link for RAG web source: {title} (URL: {url})") # Removed verbose log
                            display_item = True
                
                if display_item and identifier:
                    displayed_rag_identifiers.add(identifier)
                    st.divider()

            # --- 5. Display Code Interpreter output (Image or Download Button) ---
            code_download_absolute_filepath = os.path.join(PROJECT_ROOT, code_download_filepath_relative) if code_download_filepath_relative else None

            if code_is_image and code_download_absolute_filepath and os.path.exists(code_download_absolute_filepath):
                try:
                    st.image(code_download_absolute_filepath, caption=code_download_filename, use_container_width=True)
                    # print(f"Successfully displayed image from code interpreter: {code_download_filename}") # Removed verbose log
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
                    # print(f"Successfully added download button for code interpreter file: {code_download_filename}") # Removed verbose log
                except Exception as e:
                    st.error(f"Error creating download button for {code_download_filename}: {e}")

            # --- Add Copy to Clipboard and Regenerate Buttons ---
            is_last_assistant_message = (message["role"] == "assistant" and msg_idx == len(st.session_state.messages) - 1)
            
            can_regenerate = False
            if is_last_assistant_message:
                if len(st.session_state.messages) == 1:
                    can_regenerate = True
                elif len(st.session_state.messages) > 1 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    can_regenerate = True

            # The hidden div and content_div_id are no longer needed.

            if can_regenerate:
                col_copy, col_regen, _ = st.columns([0.05, 0.05, 0.9])
            else:
                col_copy, _ = st.columns([0.05, 0.95])

            # Define callback for the copy button
            def _copy_button_callback(text_payload, message_id):
                st.session_state.text_to_copy_payload = text_payload
                st.session_state.clipboard_triggered_for_id = message_id
                # print(f"Copy button clicked for msg {message_id}. Payload set in session state.") # Removed verbose log

            with col_copy:
                # Replace markdown button with st.button
                # text_to_display and msg_idx are from the parent scope of display_chat's loop
                col_copy.button(
                    "📋",
                    key=f"copy_btn_{msg_idx}",
                    help="Copy message to clipboard",
                    on_click=_copy_button_callback,
                    args=(text_to_display, msg_idx) # Corrected: Use text_to_display here
                )
            
            # JS injection for clipboard based on session state
            if st.session_state.get('clipboard_triggered_for_id') == msg_idx:
                text_to_copy_js = st.session_state.get('text_to_copy_payload', "")
                # Using json.dumps to safely escape the text for JavaScript
                escaped_text_for_js = json.dumps(escaped_text_for_js)

                javascript_to_run = f"""
                <script>
                    (function() {{
                        window.focus(); // Attempt to focus the current window/iframe
                        const textToCopy = {escaped_text_for_js};
                        navigator.clipboard.writeText(textToCopy).then(function() {{
                            console.log('Async: Copying to clipboard was successful!', textToCopy);
                            // Toast will be shown from Python side
                        }}, function(err) {{
                            console.error('Async: Could not copy text. Error object:', err); console.error('Error name:', err.name); console.error('Error message:', err.message);
                            alert('Failed to copy text. Check console for errors.');
                        }});
                    }})();
                </script>
                """
                st.components.v1.html(javascript_to_run, height=0, width=0)

                # Display toast message
                st.toast(f"Content from message {msg_idx + 1} copied!", icon="📋")

                # Reset the trigger and payload
                st.session_state.clipboard_triggered_for_id = None
                st.session_state.text_to_copy_payload = None

            if can_regenerate:
                with col_regen:
                    # The button now directly calls handle_regenerate_button_click
                    col_regen.button(
                        "🔄",
                        key=f"regenerate_{msg_idx}",
                        help="Regenerate Response",
                        on_click=handle_regenerate_button_click # No args needed as it reads from session_state
                    )

def handle_regenerate_button_click():
    """
    Handles the button click for regenerating the last assistant response.
    Retrieves necessary state from session_state, calls the app logic,
    and updates session_state with the results.
    """
    print("Regenerate button clicked in stui.py")
    user_id = st.session_state.user_id
    current_chat_id = st.session_state.get("current_chat_id")
    messages = st.session_state.messages # Current list of messages
    long_term_memory_enabled = st.session_state.long_term_memory_enabled
    llm_temperature = st.session_state.get("llm_temperature", 0.7) # Default if not set
    llm_verbosity = st.session_state.get("llm_verbosity", 3) # Default if not set

    try:
        result = app.regenerate_last_response(
            user_id=user_id,
            current_chat_id=current_chat_id,
            messages_input=messages,
            long_term_memory_enabled=long_term_memory_enabled,
            llm_temperature=llm_temperature,
            llm_verbosity=llm_verbosity
        )
    except Exception as e:
        st.error(f"An unexpected error occurred during regeneration: {e}")
        print(f"Critical error in call to app.regenerate_last_response: {e}")
        return

    st.session_state.messages = result.get("updated_messages", messages)
    st.session_state.suggested_prompts = result.get("updated_suggested_prompts", st.session_state.suggested_prompts)

    # Update all_chat_messages if a valid current_chat_id exists
    if current_chat_id and current_chat_id in st.session_state.all_chat_messages:
        st.session_state.all_chat_messages[current_chat_id] = st.session_state.messages

    status_message = result.get("status_message")
    if status_message:
        if "error" in status_message.lower() or "warning" in status_message.lower() or "not from assistant" in status_message.lower() or "no preceding user query" in status_message.lower() :
            st.warning(status_message)
        else:
            st.info(status_message)
        print(f"Status from app.regenerate_last_response: {status_message}")

    st.rerun()

def remove_uploaded_file(file_name: str, file_type: str):
    """Removes an uploaded file from session state and from the workspace."""
    if file_type == "document":
        if file_name in st.session_state.uploaded_documents:
            del st.session_state.uploaded_documents[file_name]
            st.toast(f"Document '{file_name}' removed.", icon="🗑️")
    elif file_type == "dataframe":
        if file_name in st.session_state.uploaded_dataframes:
            del st.session_state.uploaded_dataframes[file_name]
            st.toast(f"Dataset '{file_name}' removed.", icon="🗑️")
    
    # Attempt to delete the physical file from the workspace
    file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
    if os.path.exists(file_path_in_workspace):
        try:
            os.remove(file_path_in_workspace)
            # print(f"Successfully deleted physical file: {file_path_in_workspace}") # Removed verbose log
        except Exception as e:
            print(f"Error deleting physical file '{file_path_in_workspace}': {e}")
            st.error(f"Error deleting physical file '{file_name}': {e}")
    
    st.rerun()


def process_uploaded_file(uploaded_file):
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    # Save the raw file to the UI_ACCESSIBLE_WORKSPACE first
    try:
        os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)
        file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
        with open(file_path_in_workspace, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{file_name}' saved to workspace.")
    except Exception as e:
        st.error(f"Error saving file '{file_name}' to workspace: {e}")
        return None, None # Indicate failure

    # --- ADDED: Warning for research data files ---
    data_file_extensions = [".csv", ".xlsx", ".sav", ".rdata", ".rds"]
    if file_extension in data_file_extensions:
        st.warning(
            "**Important:** If this file contains research data, please ensure you have "
            "obtained all necessary ethical approvals for its use and upload. "
            "Do not upload sensitive or confidential data without proper authorization."
        )
    # --- END ADDED SECTION ---

    # Now, process the file content and store in session state for agent tools
    if file_extension in [".pdf", ".docx", ".md", ".txt"]: # Added .txt
        text_content = ""
        try:
            if file_extension == ".pdf":
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                for page in reader.pages:
                    text_content += page.extract_text() or ""
            elif file_extension == ".docx":
                document = Document(io.BytesIO(uploaded_file.getvalue()))
                for para in document.paragraphs:
                    text_content += para.text + "\n"
            elif file_extension in [".md", ".txt"]: # Handle .md and .txt as plain text
                text_content = uploaded_file.getvalue().decode("utf-8")
            
            st.session_state.uploaded_documents[file_name] = text_content
            st.success(f"Document '{file_name}' processed for agent access.")
            return "document", file_name
        except Exception as e:
            # Corrected typo from file_file_name to file_name
            st.error(f"Error processing document '{file_name}' for agent access: {e}")
            return None, None
    
    elif file_extension in [".csv", ".xlsx", ".sav"]:
        df = None
        try:
            if file_extension == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == ".xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_extension == ".sav":
                # pandas.read_spss requires pyreadstat
                try:
                    # Use the file path from the saved file in the workspace
                    df = pd.read_spss(file_path_in_workspace)
                except ImportError:
                    st.error("`pyreadstat` library not found. Please install it (`pip install pyreadstat`) to read .sav files.")
                    return None, None
            
            if df is not None:
                st.session_state.uploaded_dataframes[file_name] = df
                st.success(f"Dataset '{file_name}' processed for agent access.")
                return "dataframe", file_name
            else:
                st.error(f"Could not load dataframe from '{file_name}'.")
                return None, None
        except Exception as e:
            st.error(f"Error processing dataset '{file_name}' for agent access: {e}")
            return None, None
    
    elif file_extension in [".rdata", ".rds"]:
        st.warning(f"File type '{file_extension}' for '{file_name}' is not directly supported for processing in Python. Please convert it to CSV or XLSX.")
        return None, None
    else:
        st.warning(f"Unsupported file type: {file_extension} for '{file_name}'. File saved to workspace but not processed for agent tools.")
        return None, None

def create_interface(
    reset_callback: Callable,
    new_chat_callback: Callable,
    delete_chat_callback: Callable,
    rename_chat_callback: Callable,
    chat_metadata: Dict[str, str],
    current_chat_id: str,
    switch_chat_callback: Callable,
    get_discussion_markdown_callback: Callable,
    get_discussion_docx_callback: Callable,
    suggested_prompts_list: Optional[List[str]],
    handle_user_input_callback: Callable,
    long_term_memory_enabled: bool, # New parameter
    forget_me_callback: Callable, # New parameter
    set_long_term_memory_callback: Callable # New parameter
):
    """Create the Streamlit UI for the chat interface."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")

    # Initialize editing state if not present
    if 'editing_chat_id' not in st.session_state:
        st.session_state.editing_chat_id = None

    with st.sidebar:
        with st.expander("**Chat History**", expanded=False, icon = ":material/forum:"):
            if not long_term_memory_enabled:
                st.warning("Long-term memory is currently **disabled**. Your chat history will not be saved and will be lost when you close this tab or refresh the page.")
                st.info("To enable long-term memory, check the option in 'LLM Settings'.")
                # Only show "New Chat" button, no list of past chats
                if st.button("➕ New Chat (Temporary)", key="new_chat_button_temp", use_container_width=True):
                    st.session_state.editing_chat_id = None
                    new_chat_callback() # This will create a new in-memory session
            else:
                st.info("Conversations are automatically saved and linked to your browser via cookies. Clearing browser data will remove your saved discussions.")
                
                if st.button("➕ New Chat", key="new_chat_button", use_container_width=True):
                    # Clear any active editing state when creating a new chat
                    st.session_state.editing_chat_id = None
                    new_chat_callback()

                # Display existing chats
                if chat_metadata: # Only iterate if there's metadata
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
                                
                                # Option to download Markdown
                                st.download_button(
                                    label="⬇️ Download (.md)",
                                    data=get_discussion_markdown_callback(chat_id),
                                    file_name=f"{chat_name.replace(' ', '_')}.md",
                                    mime="text/markdown",
                                    key=f"download_listed_md_{chat_id}", # Changed key to be unique
                                    use_container_width=True
                                )

                                # Option to download DOCX
                                st.download_button(
                                    label="⬇️ Download (.docx)",
                                    data=get_discussion_docx_callback(chat_id),
                                    file_name=f"{chat_name.replace(' ', '_')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"download_listed_docx_{chat_id}", # New unique key
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
                                    # delete_chat_callback is now stui.handle_delete_chat_session, passed from app.main
                                    delete_chat_callback(chat_id)
                else:
                    st.info("No saved chats yet. Start a new conversation!")

        with st.expander("**Upload files**", expanded=False, icon = ":material/upload_file:"):
            uploaded_file = st.file_uploader(
                "Upload a document or dataset",
                type=["pdf", "docx", "md", "txt", "csv", "xlsx", "sav", "rdata", "rds"], # Added .txt
                accept_multiple_files=False,
                key="file_uploader"
            )

            if uploaded_file is not None:
                # Check if the file has already been processed in this session
                if uploaded_file.name not in st.session_state.uploaded_documents and \
                uploaded_file.name not in st.session_state.uploaded_dataframes:
                    file_type, file_name = process_uploaded_file(uploaded_file)
                    if file_name:
                        # Add a message to the chat history about the upload
                        if file_type == "document":
                            st.session_state.messages.append({"role": "assistant", "content": f"I've received your document: `{file_name}`. You can now ask me to `read_uploaded_document('{file_name}')`."})
                        elif file_type == "dataframe":
                            st.session_state.messages.append({"role": "assistant", "content": f"I've received your dataset: `{file_name}`. You can now ask me to `analyze_uploaded_dataframe('{file_name}')` or use the `code_interpreter` tool for more complex analysis."})
                        st.rerun() # Rerun to display the new assistant message
                else:
                    st.info(f"File '{uploaded_file.name}' has already been uploaded and processed.")
            
            # Removed the "Uploaded Files" subsection to avoid duplication
            # st.subheader("Uploaded Files")
            # if st.session_state.uploaded_documents or st.session_state.uploaded_dataframes:
            #     st.markdown("---")
            #     if st.session_state.uploaded_documents:
            #         st.markdown("##### Documents:")
            #         for doc_name in st.session_state.uploaded_documents.keys():
            #             col1, col2 = st.columns([0.8, 0.2])
            #             with col1:
            #                 st.write(f"- 📄 {doc_name}")
            #             with col2:
            #                 st.button(
            #                     "🗑️",
            #                     key=f"remove_doc_{doc_name}",
            #                     help=f"Remove {doc_name}",
            #                     on_click=remove_uploaded_file,
            #                     args=(doc_name, "document")
            #                 )
            #     if st.session_state.uploaded_dataframes:
            #         st.markdown("##### Datasets:")
            #         for df_name in st.session_state.uploaded_dataframes.keys():
            #             col1, col2 = st.columns([0.8, 0.2])
            #             with col1:
            #                 st.write(f"- 📊 {df_name}")
            #             with col2:
            #                 st.button(
            #                     "🗑️",
            #                     key=f"remove_df_{df_name}",
            #                     help=f"Remove {df_name}",
            #                     on_click=remove_uploaded_file,
            #                     args=(df_name, "dataframe")
            #                 )
            #     st.markdown("---")
            # else:
            #     st.info("No files uploaded yet.")

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
            st.slider(
                "Number of Search Results",
                min_value=3,
                max_value=15,
                value=st.session_state.get("search_results_count", 5),
                step=1,
                key="search_results_count",
                help="Controls the maximum number of search results returned by search tools (DuckDuckGo, Tavily, Stochasticscholar)."
            )
            st.toggle(
                "Enable Long-term Memory (saves chat history)",
                value=st.session_state.get("long_term_memory_enabled", False), # Default to False
                key="long_term_memory_enabled", # Ensure this key matches the one used in app.py
                help="If enabled, your chat history will be saved and loaded across sessions using browser cookies. If disabled, your chats will be forgotten when you close the browser or refresh the page."
            )

           # Implement the forget me button
            if long_term_memory_enabled: # Corrected variable name from 'on' to 'long_term_memory_enabled'
                # Use a popover for confirmation
                with st.popover("Forget Me (Delete All Data)", use_container_width=True):
                    st.warning("This will permanently delete ALL your saved chat histories and remove your user ID cookie from this browser. This action cannot be undone.")
                    st.write("Are you sure you want to proceed?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, Delete All Data", key="confirm_forget_me_yes", type="primary", use_container_width=True):
                            forget_me_callback() # Call the function passed from app.py
                            # No need for st.success() here, as the page will immediately reload.
                    with col_no:
                        if st.button("No, Cancel", key="confirm_forget_me_no", use_container_width=True):
                            st.info("Deletion cancelled.")
                            # Popover will close automatically
            else:
                st.info("Long-term memory is disabled. No chat history is being saved.")



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
    # The handle_user_input_callback is now handle_user_input_submission from this file (stui.py)
    # It will internally call app.process_user_prompt_and_get_response
    display_main_chat_area(suggested_prompts_list, handle_user_input_callback)

def handle_user_input_submission(chat_input_value: str | None):
    """
    Handles submission from chat input or suggested prompts.
    It retrieves necessary state, calls the core processing logic in app.py,
    and then updates session_state with the results.
    """
    prompt_to_process = None
    source_of_prompt = "chat_input" # Default

    if hasattr(st.session_state, 'prompt_to_use') and st.session_state.prompt_to_use:
        prompt_to_process = st.session_state.prompt_to_use
        st.session_state.prompt_to_use = None # Clear after use
        source_of_prompt = "suggested_prompt"
        print(f"Processing suggested_prompt: '{prompt_to_process[:50]}...'")
    elif chat_input_value:
        prompt_to_process = chat_input_value
        print(f"Processing chat_input_value: '{prompt_to_process[:50]}...'")

    if prompt_to_process:
        # Retrieve all necessary current state from st.session_state
        user_id = st.session_state.user_id
        current_chat_id = st.session_state.get("current_chat_id") # Can be None
        messages = st.session_state.messages # Current list of messages
        chat_metadata = st.session_state.chat_metadata
        long_term_memory_enabled = st.session_state.long_term_memory_enabled

        result = None
        try:
            with st.spinner("ESI is thinking..."):
                result = app.process_user_prompt_and_get_response(
                    prompt_to_process=prompt_to_process,
                    user_id=user_id,
                    current_chat_id=current_chat_id,
                    messages_input=messages,
                    chat_metadata_input=chat_metadata,
                    long_term_memory_enabled=long_term_memory_enabled
                )
        except Exception as e:
            st.error(f"An unexpected error occurred while processing your request: {e}")
            print(f"Critical error in call to app.process_user_prompt_and_get_response: {e}")
            return

        if result is None: # Should not happen if exception handling is correct
            st.error("Failed to get a response from the assistant.")
            return

        # Update st.session_state based on the dictionary returned
        st.session_state.messages = result.get("updated_messages", messages)
        st.session_state.chat_metadata = result.get("updated_chat_metadata", chat_metadata)
        st.session_state.suggested_prompts = result.get("updated_suggested_prompts", st.session_state.suggested_prompts)
        st.session_state.chat_modified = result.get("chat_modified_flag", st.session_state.chat_modified)

        new_chat_id_returned = result.get("new_chat_id")
        processed_chat_id = result.get("current_chat_id_processed")

        if new_chat_id_returned and new_chat_id_returned != current_chat_id:
            st.session_state.current_chat_id = new_chat_id_returned
            # Ensure all_chat_messages is updated for this new chat
            st.session_state.all_chat_messages[new_chat_id_returned] = result.get("updated_messages", [])
            print(f"UI updated to new chat ID: {new_chat_id_returned}")
        elif processed_chat_id and processed_chat_id == current_chat_id :
            # Update messages for the existing current_chat_id in all_chat_messages
             st.session_state.all_chat_messages[current_chat_id] = result.get("updated_messages", messages)
        elif processed_chat_id and processed_chat_id != current_chat_id and new_chat_id_returned is None:
            # This case implies the current_chat_id might have been None, and process_user_prompt created one.
            # This should ideally be covered by new_chat_id_returned logic.
            # However, as a safeguard:
            st.session_state.current_chat_id = processed_chat_id
            st.session_state.all_chat_messages[processed_chat_id] = result.get("updated_messages", [])
            print(f"UI updated to processed chat ID (safeguard): {processed_chat_id}")


        status_message = result.get("status_message")
        if status_message:
            # For now, just print. Could use st.info/st.error if refined.
            print(f"Status from process_user_prompt_and_get_response: {status_message}")
            if "error" in status_message.lower():
                st.error(status_message)
            else:
                st.info(status_message)

        print(f"Rerunning Streamlit after processing prompt '{prompt_to_process[:50]}...' from {source_of_prompt}")
        st.rerun()
    elif source_of_prompt == "chat_input" and not chat_input_value:
        # This handles the case where chat_input is cleared, which causes a rerun.
        # We don't want to do anything here if there's no actual input.
        print("Chat input cleared, no action taken by handle_user_input_submission.")
        pass


def display_main_chat_area(suggested_prompts_list: Optional[List[str]], handle_user_input_callback: Callable):
    """
    Displays the main chat area including suggested prompts and the chat input field.
    The handle_user_input_callback is now stui.handle_user_input_submission.
    """
    if suggested_prompts_list:
        cols = st.columns(len(suggested_prompts_list))
        for i, prompt in enumerate(suggested_prompts_list):
            with cols[i]:
                # When a suggested prompt button is clicked:
                # 1. Set 'prompt_to_use' in session_state.
                # 2. Call the handle_user_input_callback (which is handle_user_input_submission).
                #    handle_user_input_submission will then pick up 'prompt_to_use'.
                if st.button(prompt, key=f"suggested_prompt_btn_{i}"):
                    st.session_state.prompt_to_use = prompt
                    # Call the callback directly. It will handle the rerun.
                    # No separate chat_input_value is provided here as prompt_to_use takes precedence.
                    handle_user_input_callback(None)
    else:
        pass # No suggested prompts to display

    # The st.chat_input widget itself triggers a rerun when the user submits text.
    # The value entered by the user is returned by st.chat_input().
    # This value is then passed to the handle_user_input_callback.
    chat_input_value = st.chat_input(
        "Ask me about dissertations, research methods, academic writing, etc.",
        key="main_chat_input" # Added a key for stability
    )

    # If chat_input_value has content (i.e., user submitted something),
    # call the callback. This happens on the rerun triggered by chat_input.
    if chat_input_value:
        # st.session_state.chat_input_value_from_stui is no longer needed here
        # as we directly pass chat_input_value to the callback.
        handle_user_input_callback(chat_input_value)

def perform_chat_reset():
    """
    Called when the reset chat button (or new chat for temporary LTM off) is pressed.
    Uses app.create_new_chat_session_in_memory and updates session state
    based on the values returned by it.
    """
    print("Performing chat reset via stui.perform_chat_reset...")

    # This function in app.py returns (new_chat_id, new_chat_name, initial_messages)
    # AND also updates several st.session_state keys directly:
    # current_chat_id, messages, chat_metadata[new_chat_id], all_chat_messages[new_chat_id], chat_modified
    new_chat_id, new_chat_name, initial_messages = app.create_new_chat_session_in_memory()

    # The call above already updated st.session_state.current_chat_id,
    # st.session_state.messages, st.session_state.chat_metadata,
    # st.session_state.all_chat_messages, and st.session_state.chat_modified.
    # So, we just need to ensure suggested prompts are also updated based on the new initial messages.
    st.session_state.suggested_prompts = app._cached_generate_suggested_prompts(initial_messages)

    print(f"Chat reset performed by stui. New active chat ID: {st.session_state.current_chat_id}. UI will now update.")
    st.rerun()

def handle_delete_chat_session(chat_id: str):
    """
    Handles the UI part of deleting a chat session.
    Calls the core deletion logic in app.py and then handles UI updates.
    """
    print(f"handle_delete_chat_session called in stui.py for chat_id: {chat_id}")
    error_message = app.delete_chat_session(chat_id) # This now returns str | None

    if error_message:
        st.error(error_message)

    # Always rerun because the list of chats in the sidebar needs to be updated,
    # or the current chat might have changed if the active one was deleted.
    # app.delete_chat_session itself no longer calls rerun.
    st.rerun()

def handle_forget_me_button_click():
    """
    Handles the logic for the "Forget Me" button click.
    Calls data deletion in app.py, then clears cookies and session state.
    """
    print("Forget Me button clicked.")
    user_id_to_delete = st.session_state.get("user_id")
    hf_token = os.getenv("HF_TOKEN")

    if not user_id_to_delete:
        st.warning("No user ID found in session to forget.")
        return

    if not hf_token:
        st.warning("Hugging Face token not configured. Cannot delete cloud data, but will clear local data.")
        # Proceed to clear local data even if cloud deletion isn't possible
    else:
        # Call app.py function to delete data from Hugging Face
        # app.fs should be accessible as it's a global in app.py
        deletion_result = app.perform_forget_me_data_deletion(user_id_to_delete, hf_token, app.fs)
        if not deletion_result["success"]:
            st.warning(f"Cloud data deletion issues: {deletion_result['message']}")
        else:
            st.info(f"Cloud data deletion attempt: {deletion_result['message']}")

    # Delete the user ID cookie using stui's cookie manager
    try:
        cookies.delete(cookie="user_id") # stui.cookies
        print(f"Deleted user ID cookie for '{user_id_to_delete}'")
    except Exception as e:
        error_msg = f"ERROR: Failed to delete user_id cookie for {user_id_to_delete}: {e}"
        print(error_msg)
        st.error(f"Failed to delete user ID cookie: {e}") # Show error in UI

    # Reset session state extensively
    keys_to_clear = [
        "chat_metadata", "all_chat_messages", "current_chat_id",
        "messages", "chat_modified", "suggested_prompts",
        "renaming_chat_id", "uploaded_documents", "uploaded_dataframes",
        "session_control_flags_initialized",
        "_greeting_logic_log_shown_for_current_state",
        "agent_initialized_with_search_count", # ensure agent re-init
        "_last_memory_state_was_enabled",
        "_last_memory_state_changed_by_toggle",
        "initial_greeting_shown_for_session"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            # More robust clearing for different types might be needed,
            # but for now, del is generally effective.
            del st.session_state[key]
            # if isinstance(st.session_state[key], (list, dict)):
            #     st.session_state[key].clear()
            # else:
            #     st.session_state[key] = None

    if "user_id" in st.session_state: # Should be gone if del worked
        del st.session_state.user_id

    # After clearing, some keys might need to be re-initialized to default states
    # for the app to continue running before reload, though reload is imminent.
    # st.session_state.messages = [] # app.main will handle this on reload via session_control_flags_initialized=False
    # st.session_state.suggested_prompts = app.DEFAULT_PROMPTS # app.main will handle
    st.session_state.session_control_flags_initialized = False # Force full re-init in app.py

    print("Local session state and cookies cleared. Triggering page reload.")

    # Use JavaScript to clear all cookies from the browser and force a full page reload
    js_code = """
    <script>
        function deleteAllCookies() {
            const cookies_array = document.cookie.split(';');
            for (let i = 0; i < cookies_array.length; i++) {
                const cookie = cookies_array[i];
                const eqPos = cookie.indexOf('=');
                const name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
                document.cookie = name + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
            }
            console.log("All cookies cleared by JS.");
        }
        deleteAllCookies();
        window.location.reload(true); // Force a hard reload from the server
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)
    # No st.rerun() here as JS handles the reload.

def handle_new_chat_button_click():
    """
    Handles the "New Chat" button click.
    Calls app.create_new_chat_session_in_memory and updates session state.
    """
    print("New chat button clicked in stui.py")
    # app.create_new_chat_session_in_memory updates st.session_state directly for:
    # current_chat_id, messages, chat_metadata[new_chat_id], all_chat_messages[new_chat_id], chat_modified
    new_chat_id, new_chat_name, initial_messages = app.create_new_chat_session_in_memory()

    # Ensure suggested prompts are updated for the new greeting
    st.session_state.suggested_prompts = app._cached_generate_suggested_prompts(initial_messages)

    # Other session state keys like current_chat_id, messages, etc., are already set by
    # app.create_new_chat_session_in_memory() as per its current implementation.
    # If app.create_new_chat_session_in_memory were purely functional, we'd set them here:
    # st.session_state.current_chat_id = new_chat_id
    # st.session_state.messages = initial_messages
    # st.session_state.chat_metadata[new_chat_id] = new_chat_name
    # st.session_state.all_chat_messages[new_chat_id] = initial_messages
    # st.session_state.chat_modified = False

    print(f"New chat session '{new_chat_name}' ({new_chat_id}) created and UI updated by stui.")
    st.rerun()

def handle_rename_chat(chat_id: str, new_name: str):
    """
    Handles renaming a chat.
    Calls app.core_rename_chat and updates session state.
    """
    print(f"handle_rename_chat called in stui.py for chat_id: {chat_id}, new_name: {new_name}")
    user_id = st.session_state.user_id
    current_chat_metadata = st.session_state.chat_metadata
    long_term_memory_enabled = st.session_state.long_term_memory_enabled

    # Assuming app.core_rename_chat exists and is correctly defined in app.py
    # It should return: (updated_metadata_copy_or_none, error_message_str_or_none)
    updated_metadata, error_msg = app.core_rename_chat(
        user_id=user_id,
        chat_id=chat_id,
        new_name=new_name,
        current_chat_metadata=current_chat_metadata,
        long_term_memory_enabled=long_term_memory_enabled
    )

    if updated_metadata is not None:
        st.session_state.chat_metadata = updated_metadata
        print(f"Chat '{chat_id}' renamed to '{new_name}' in session state by stui.")

    if error_msg:
        st.error(error_msg) # Display error in UI
        print(f"Error renaming chat: {error_msg}")

    # The text_input's on_change in create_interface usually handles clearing editing_chat_id and rerunning.
    # If not, st.rerun() might be needed here. For now, assume on_change is sufficient.

def handle_switch_chat(chat_id_to_switch: str):
    """
    Handles switching to a different chat.
    Calls app.core_switch_chat_logic and updates session state.
    """
    print(f"handle_switch_chat called in stui.py for chat_id: {chat_id_to_switch}")

    # Assuming app.core_switch_chat_logic exists and is correctly defined in app.py
    # It returns a dict with keys like: messages_to_display, current_chat_id_to_set, etc.
    result = app.core_switch_chat_logic(
        chat_id_to_switch=chat_id_to_switch,
        current_long_term_memory_enabled=st.session_state.long_term_memory_enabled,
        current_chat_metadata=st.session_state.chat_metadata,
        all_chat_messages_from_session=st.session_state.all_chat_messages
    )

    st.session_state.messages = result.get("messages_to_display", st.session_state.messages)
    st.session_state.current_chat_id = result.get("current_chat_id_to_set", st.session_state.current_chat_id)
    st.session_state.suggested_prompts = result.get("suggested_prompts_to_set", st.session_state.suggested_prompts)
    st.session_state.chat_modified = result.get("chat_modified_to_set", st.session_state.chat_modified)

    # Handle cases where a new temporary chat might have been created by core_switch_chat_logic
    # (e.g., when LTM is off and create_new_chat_session_in_memory was called by core_switch_chat_logic)
    if result.get("chat_metadata_to_set") is not None:
         st.session_state.chat_metadata = result["chat_metadata_to_set"]
    if result.get("all_chat_messages_to_set") is not None:
         st.session_state.all_chat_messages = result["all_chat_messages_to_set"]

    status_message = result.get("status_message")
    if status_message:
        if "Error" in status_message:
            st.warning(status_message)
        else:
            st.info(status_message)
        print(f"Status from core_switch_chat_logic: {status_message}")

    st.rerun()

def handle_set_long_term_memory_preference():
    """
    Handles the toggle for long-term memory.
    Calls app._set_long_term_memory_preference to save the cookie.
    """
    value_to_set = st.session_state.long_term_memory_enabled
    print(f"LTM toggle changed in UI. New value: {value_to_set}. Saving preference via app function.")

    # Assuming app._set_long_term_memory_preference has the signature (cookies_manager, value_to_set) -> str | None
    error_msg = app._set_long_term_memory_preference(cookies, value_to_set) # stui.cookies

    if error_msg:
        st.error(error_msg) # Display error in UI
        print(f"Error setting LTM preference cookie: {error_msg}")

    # This flag is used by app.main to detect if a full session re-initialization is needed
    st.session_state._last_memory_state_changed_by_toggle = True
    # The toggle widget itself causes a rerun, so no explicit st.rerun() here.
    # app.main() will handle the consequences of the memory state change on the next run.

def handle_new_chat_button_click():
    """
    Handles the "New Chat" button click.
    Calls app.create_new_chat_session_in_memory and updates session state.
    Relies on app.create_new_chat_session_in_memory to update relevant
    st.session_state keys like current_chat_id, messages, chat_metadata,
    all_chat_messages, and chat_modified.
    """
    print("New chat button clicked in stui.py")
    new_chat_id, new_chat_name, initial_messages = app.create_new_chat_session_in_memory()

    # create_new_chat_session_in_memory in app.py should have already set:
    # st.session_state.current_chat_id = new_chat_id
    # st.session_state.messages = initial_messages
    # st.session_state.chat_metadata[new_chat_id] = new_chat_name
    # st.session_state.all_chat_messages[new_chat_id] = initial_messages
    # st.session_state.chat_modified = False

    # Ensure suggested prompts are updated for the new greeting
    st.session_state.suggested_prompts = app._cached_generate_suggested_prompts(initial_messages)

    print(f"New chat session '{new_chat_name}' ({new_chat_id}) created and UI updated by stui.")
    st.rerun()

def handle_rename_chat(chat_id: str, new_name: str):
    """
    Handles renaming a chat.
    Calls app.core_rename_chat and updates session state.
    """
    print(f"handle_rename_chat called in stui.py for chat_id: {chat_id}, new_name: {new_name}")
    user_id = st.session_state.user_id
    current_chat_metadata = st.session_state.chat_metadata
    long_term_memory_enabled = st.session_state.long_term_memory_enabled

    # Assuming app.core_rename_chat exists and is correctly defined in app.py
    updated_metadata, error_msg = app.core_rename_chat(
        user_id=user_id,
        chat_id=chat_id,
        new_name=new_name,
        current_chat_metadata=current_chat_metadata,
        long_term_memory_enabled=long_term_memory_enabled
    )

    if updated_metadata is not None:
        st.session_state.chat_metadata = updated_metadata
        print(f"Chat '{chat_id}' renamed to '{new_name}' in session state by stui.")

    if error_msg:
        st.error(error_msg)
        print(f"Error renaming chat: {error_msg}")

    # The text_input's on_change in create_interface handles clearing editing_chat_id and rerunning.
    # No explicit st.rerun() here is needed if that on_change behavior is confirmed.

def handle_switch_chat(chat_id_to_switch: str):
    """
    Handles switching to a different chat.
    Calls app.core_switch_chat_logic and updates session state.
    """
    print(f"handle_switch_chat called in stui.py for chat_id: {chat_id_to_switch}")

    # Assuming app.core_switch_chat_logic exists and is correctly defined in app.py
    result = app.core_switch_chat_logic(
        chat_id_to_switch=chat_id_to_switch,
        current_long_term_memory_enabled=st.session_state.long_term_memory_enabled,
        current_chat_metadata=st.session_state.chat_metadata,
        all_chat_messages_from_session=st.session_state.all_chat_messages
    )

    st.session_state.messages = result.get("messages_to_display", st.session_state.messages)
    st.session_state.current_chat_id = result.get("current_chat_id_to_set", st.session_state.current_chat_id)
    st.session_state.suggested_prompts = result.get("suggested_prompts_to_set", st.session_state.suggested_prompts)
    st.session_state.chat_modified = result.get("chat_modified_to_set", st.session_state.chat_modified)

    # Handle cases where a new temporary chat might have been created by core_switch_chat_logic
    if result.get("chat_metadata_to_set") is not None:
         st.session_state.chat_metadata = result["chat_metadata_to_set"]
    if result.get("all_chat_messages_to_set") is not None:
         st.session_state.all_chat_messages = result["all_chat_messages_to_set"]

    status_message = result.get("status_message")
    if status_message:
        if "Error" in status_message:
            st.warning(status_message)
        else:
            st.info(status_message)
        print(f"Status from core_switch_chat_logic: {status_message}")

    st.rerun()

def handle_set_long_term_memory_preference():
    """
    Handles the toggle for long-term memory.
    Calls app._set_long_term_memory_preference to save the cookie.
    """
    value_to_set = st.session_state.long_term_memory_enabled
    print(f"LTM toggle changed in UI. New value: {value_to_set}. Saving preference via app function.")

    # Assuming app._set_long_term_memory_preference has the signature (cookies_manager, value_to_set)
    error_msg = app._set_long_term_memory_preference(cookies, value_to_set) # stui.cookies

    if error_msg:
        st.error(error_msg)
        print(f"Error setting LTM preference cookie: {error_msg}")

    st.session_state._last_memory_state_changed_by_toggle = True
    # The toggle widget itself causes a rerun.

def handle_new_chat_button_click():
    """
    Handles the "New Chat" button click.
    Calls app.create_new_chat_session_in_memory and updates session state.
    """
    print("New chat button clicked in stui.py")
    # app.create_new_chat_session_in_memory updates st.session_state directly for:
    # chat_metadata, all_chat_messages, current_chat_id, messages, chat_modified
    new_chat_id, new_chat_name, initial_messages = app.create_new_chat_session_in_memory()

    # Ensure suggested prompts are updated for the new greeting
    st.session_state.suggested_prompts = app._cached_generate_suggested_prompts(initial_messages)

    print(f"New chat created by stui. Active chat ID: {new_chat_id}. UI will now update.")
    st.rerun()

def handle_rename_chat(chat_id: str, new_name: str):
    """
    Handles renaming a chat.
    Calls app.core_rename_chat and updates session state.
    """
    print(f"handle_rename_chat called in stui.py for chat_id: {chat_id}, new_name: {new_name}")
    user_id = st.session_state.user_id
    current_chat_metadata = st.session_state.chat_metadata
    long_term_memory_enabled = st.session_state.long_term_memory_enabled

    updated_metadata, error_msg = app.core_rename_chat(
        user_id=user_id,
        chat_id=chat_id,
        new_name=new_name,
        current_chat_metadata=current_chat_metadata,
        long_term_memory_enabled=long_term_memory_enabled
    )

    if updated_metadata is not None:
        st.session_state.chat_metadata = updated_metadata
        print(f"Chat '{chat_id}' renamed to '{new_name}' in session state.")

    if error_msg:
        st.error(error_msg)
        print(f"Error renaming chat: {error_msg}")

    # The text_input's on_change in create_interface already handles clearing editing_chat_id and rerunning.
    # No explicit st.rerun() needed here unless further state changes dependent on this need immediate reflection
    # before the natural rerun from the on_change event.

def handle_switch_chat(chat_id_to_switch: str):
    """
    Handles switching to a different chat.
    Calls app.core_switch_chat_logic and updates session state.
    """
    print(f"handle_switch_chat called in stui.py for chat_id: {chat_id_to_switch}")

    result = app.core_switch_chat_logic(
        chat_id_to_switch=chat_id_to_switch,
        current_long_term_memory_enabled=st.session_state.long_term_memory_enabled,
        current_chat_metadata=st.session_state.chat_metadata,
        all_chat_messages_from_session=st.session_state.all_chat_messages
        # app._get_initial_greeting_text() is not needed by core_switch_chat_logic anymore
    )

    st.session_state.messages = result.get("messages_to_display", st.session_state.messages)
    st.session_state.current_chat_id = result.get("current_chat_id_to_set", st.session_state.current_chat_id)
    st.session_state.suggested_prompts = result.get("suggested_prompts_to_set", st.session_state.suggested_prompts)
    st.session_state.chat_modified = result.get("chat_modified_to_set", st.session_state.chat_modified)

    # If LTM was off and a new temporary chat was effectively created by core_switch_chat_logic (via create_new_chat_session_in_memory)
    if result.get("chat_metadata_to_set") is not None:
         st.session_state.chat_metadata = result["chat_metadata_to_set"]
    if result.get("all_chat_messages_to_set") is not None:
         st.session_state.all_chat_messages = result["all_chat_messages_to_set"]

    status_message = result.get("status_message")
    if status_message:
        # Using st.info for status, could be st.warning/error if the message indicates that
        if "Error" in status_message:
            st.warning(status_message) # Using warning for non-critical errors like chat not found
        else:
            st.info(status_message)
        print(f"Status from core_switch_chat_logic: {status_message}")

    st.rerun()

def handle_set_long_term_memory_preference():
    """
    Handles the toggle for long-term memory.
    Calls app._set_long_term_memory_preference to save the cookie.
    """
    value_to_set = st.session_state.long_term_memory_enabled # This is the new value from the toggle
    print(f"LTM toggle changed in UI. New value: {value_to_set}. Saving preference via app function.")

    # Call the app function, passing the cookie manager and the value to set
    error_msg = app._set_long_term_memory_preference(cookies, value_to_set) # stui.cookies

    if error_msg:
        st.error(error_msg)
        print(f"Error setting LTM preference cookie: {error_msg}")

    # This flag is used by app.main to detect if a full session re-initialization is needed
    st.session_state._last_memory_state_changed_by_toggle = True
    # The toggle widget itself causes a rerun, so no explicit st.rerun() here.
    # app.main() will handle the consequences of the memory state change on the next run.
