import streamlit as st
import datetime
import os
import re # Import regex module
import json # Added for parsing RAG source JSON
from typing import List, Dict, Any, Optional, Callable # Import Callable

# Removed: from agent import generate_llm_greeting, DEFAULT_PROMPTS (no longer needed here)

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# Removed: get_greeting_message() - now handled by app.py
# Removed: init_session_state() - now handled by app.py

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
                # --- 1. Extract RAG sources ---
                current_pos = 0
                temp_text_for_rag_extraction = text_to_display
                processed_text_after_rag = ""
                
                while True:
                    marker_pos = temp_text_for_rag_extraction.find(RAG_SOURCE_MARKER, current_pos)
                    if marker_pos == -1:
                        processed_text_after_rag += temp_text_for_rag_extraction[current_pos:]
                        break 
                        
                    processed_text_after_rag += temp_text_for_rag_extraction[current_pos:marker_pos]
                    
                    json_start_pos = marker_pos + len(RAG_SOURCE_MARKER)
                    json_end_pos = temp_text_for_rag_extraction.find("\n", json_start_pos)
                    if json_end_pos == -1:
                        json_end_pos = len(temp_text_for_rag_extraction)
                    
                    json_str = temp_text_for_rag_extraction[json_start_pos:json_end_pos].strip()
                    consumed_upto = json_end_pos

                    if not json_str.startswith("{"):
                        json_start_pos_next_line = json_end_pos + 1
                        if json_start_pos_next_line < len(temp_text_for_rag_extraction):
                            json_end_pos_next_line = temp_text_for_rag_extraction.find("\n", json_start_pos_next_line)
                            if json_end_pos_next_line == -1:
                                json_end_pos_next_line = len(temp_text_for_rag_extraction)
                            
                            potential_json_str_next_line = temp_text_for_rag_extraction[json_start_pos_next_line:json_end_pos_next_line].strip()
                            if potential_json_str_next_line.startswith("{"):
                                json_str = potential_json_str_next_line
                                consumed_upto = json_end_pos_next_line
                    
                    try:
                        if not json_str:
                            raise json.JSONDecodeError("Extracted JSON string is empty after attempts", json_str, 0)
                        
                        rag_data = json.loads(json_str)
                        rag_sources_data.append(rag_data)
                        print(f"Extracted RAG source: {rag_data.get('name') or rag_data.get('title')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode RAG source JSON: '{json_str}'. Error: {e}")

                    current_pos = consumed_upto

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
                    # If image display fails, fall through to show download button
                    code_is_image = False # Treat as non-image for download button logic
            
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

            # Add regenerate button for the last assistant message
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


def create_interface(reset_callback: Callable): # Accept the callback function
    """Create the Streamlit UI for the chat interface."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your dissertation research")

    # Removed: init_session_state() - now handled by app.py

    with st.sidebar:
        st.header("About ESI")
        st.info("ESI uses AI to help you navigate the dissertation process. It has access to some of the literature in your reading lists and also uses search tools for web lookups.")
        st.warning("⚠️  Remember: Always consult your dissertation supervisor for final guidance and decisions.")

        st.divider()

        st.header("LLM Settings")
        st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get("llm_temperature", 0.7),
            step=0.1,
            key="llm_temperature",
            help="Controls the randomness of the AI's responses. Lower values are more focused, higher values are more creative."
        )

        st.divider()
        st.info("Made for NBS7091A and NBS7095x")
        
        st.divider()
        if st.button("🔄 Reset Chat", key="reset_chat_button", help="Clears the current conversation and starts a new one."):
            reset_callback() # Call the passed callback

    display_chat()




# Removed: reset_chat_callback() - moved to app.py
