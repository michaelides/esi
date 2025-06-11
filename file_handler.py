import os
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List

# Configuration imports
from config import UI_ACCESSIBLE_WORKSPACE # PROJECT_ROOT not directly used here now

# Imports for file processing
import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError # Specific PDF error
from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError as DocxPackageNotFoundError # Specific docx error
import io
try:
    import pyreadstat
    from pyreadstat.pyreadstat import PyreadstatError
except ImportError:
    pyreadstat = None # Make it optional
    PyreadstatError = None # Define for except block even if not available
    print("WARNING: file_handler: pyreadstat library not found. SPSS (.sav) file processing will be unavailable.")


# --- Tool Functions (formerly in app.py) ---

def read_uploaded_document_tool_fn(filename: str) -> str:
    """Reads the full text content of a document previously uploaded by the user."""
    if not filename:
        return "Error: Filename cannot be empty."
    uploaded_docs = st.session_state.get("uploaded_documents", {})
    if filename not in uploaded_docs:
        return f"Error: Document '{filename}' not found. Available documents: {list(uploaded_docs.keys())}"
    return uploaded_docs[filename]

def analyze_dataframe_tool_fn(filename: str, head_rows: int = 5) -> str:
    """Provides summary information about a pandas DataFrame previously uploaded by the user."""
    if not filename:
        return "Error: Filename cannot be empty."
    uploaded_dfs = st.session_state.get("uploaded_dataframes", {})
    if filename not in uploaded_dfs:
        return f"Error: DataFrame '{filename}' not found. Available dataframes: {list(uploaded_dfs.keys())}"

    df = uploaded_dfs[filename]
    if not isinstance(df, pd.DataFrame):
        return f"Error: '{filename}' is not a valid DataFrame object."

    try:
        info_str = f"DataFrame: {filename}\n"
        info_str += f"Shape: {df.shape}\n"
        info_str += f"Columns: {', '.join(df.columns)}\n"
        info_str += f"Data Types:\n{df.dtypes.to_string()}\n"

        # Ensure head_rows is a non-negative integer
        try:
            head_rows = int(head_rows)
            if head_rows < 0: head_rows = 0
        except ValueError:
            head_rows = 5 # Default if conversion fails
            info_str += f"(Invalid head_rows value, defaulted to 5)\n"

        head_rows = min(head_rows, len(df)) # Cap at df length
        if head_rows > 0:
            info_str += f"First {head_rows} rows:\n{df.head(head_rows).to_string()}\n"
        else:
            info_str += "No head rows requested or displayed.\n"

        info_str += f"Summary Statistics:\n{df.describe().to_string()}\n"
        return info_str
    except Exception as e:
        print(f"ERROR: file_handler.analyze_dataframe_tool_fn: Error generating DataFrame summary for '{filename}': {e}")
        return f"Error analyzing DataFrame '{filename}': {e}"


# --- File Processing Functions (formerly in stui.py) ---

def process_uploaded_file(uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    """
    Processes an uploaded file: saves it, extracts content/data for supported types,
    and stores it in session state.
    Returns a tuple (file_type: str|None, file_name: str|None).
    """
    if uploaded_file is None:
        print("ERROR: file_handler.process_uploaded_file: uploaded_file is None.")
        return None, None

    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    # Initialize session state storage if not already present
    if "uploaded_documents" not in st.session_state: st.session_state.uploaded_documents = {}
    if "uploaded_dataframes" not in st.session_state: st.session_state.uploaded_dataframes = {}

    try:
        os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)
        file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
        with open(file_path_in_workspace, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{file_name}' saved to workspace.")
        print(f"LOG: file_handler.File '{file_name}' saved to: {file_path_in_workspace}")
    except (OSError, IOError) as e:
        st.error(f"Error saving file '{file_name}' to workspace: {e}")
        print(f"ERROR: file_handler.Error saving file '{file_name}' to workspace: {e}")
        return None, None
    except Exception as e: # Catch any other unexpected error during save
        st.error(f"Unexpected error saving file '{file_name}': {e}")
        print(f"ERROR: file_handler.Unexpected error saving file '{file_name}': {e}")
        return None, None


    data_file_extensions = [".csv", ".xlsx", ".sav", ".rdata", ".rds"]
    if file_extension in data_file_extensions:
        st.warning(
            "**Important:** If this file contains research data, please ensure you have "
            "obtained all necessary ethical approvals for its use and upload. "
            "Do not upload sensitive or confidential data without proper authorization."
        )

    # Process content for agent access
    if file_extension in [".pdf", ".docx", ".md", ".txt"]:
        text_content = ""
        try:
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            if file_extension == ".pdf":
                reader = PdfReader(file_bytes)
                for page in reader.pages: text_content += (page.extract_text() or "") + "\n"
            elif file_extension == ".docx":
                document = DocxDocument(file_bytes)
                for para in document.paragraphs: text_content += para.text + "\n"
            elif file_extension in [".md", ".txt"]:
                text_content = file_bytes.getvalue().decode("utf-8")

            st.session_state.uploaded_documents[file_name] = text_content
            st.success(f"Document '{file_name}' processed for agent access.")
            print(f"LOG: file_handler.Document '{file_name}' processed.")
            return "document", file_name
        except PdfReadError as e:
            st.error(f"Error reading PDF '{file_name}': {e}. File might be corrupt or password-protected.")
            print(f"ERROR: file_handler.PdfReadError for '{file_name}': {e}")
            return None, None
        except DocxPackageNotFoundError as e:
            st.error(f"Error reading DOCX '{file_name}': {e}. File is not a valid DOCX (zip) file.")
            print(f"ERROR: file_handler.DocxPackageNotFoundError for '{file_name}': {e}")
            return None, None
        except UnicodeDecodeError as e:
            st.error(f"Error decoding text file '{file_name}': {e}. Ensure it's UTF-8 encoded.")
            print(f"ERROR: file_handler.UnicodeDecodeError for '{file_name}': {e}")
            return None, None
        except Exception as e:
            st.error(f"Error processing document '{file_name}': {e}")
            print(f"ERROR: file_handler.Error processing document '{file_name}': {e}")
            return None, None

    elif file_extension in [".csv", ".xlsx", ".sav"]:
        df = None
        file_bytes_for_df = io.BytesIO(uploaded_file.getvalue())
        try:
            if file_extension == ".csv":
                df = pd.read_csv(file_bytes_for_df)
            elif file_extension == ".xlsx":
                df = pd.read_excel(file_bytes_for_df)
            elif file_extension == ".sav":
                if pyreadstat:
                    df, _ = pyreadstat.read_sav(file_bytes_for_df)
                else:
                    st.error("Processing .sav files is disabled because `pyreadstat` library is not available.")
                    return None, None

            if df is not None:
                st.session_state.uploaded_dataframes[file_name] = df
                st.success(f"Dataset '{file_name}' processed for agent access.")
                print(f"LOG: file_handler.Dataset '{file_name}' processed.")
                return "dataframe", file_name
            else: # Should not be reached if pyreadstat error is handled
                st.error(f"Could not load dataframe from '{file_name}'.")
                return None, None
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            st.error(f"Error parsing data file '{file_name}': {e}. File might be empty or malformed.")
            print(f"ERROR: file_handler.Pandas parsing error for '{file_name}': {e}")
            return None, None
        except PyreadstatError as e: # Assuming PyreadstatError is imported or defined
            st.error(f"Error reading SPSS file '{file_name}' with pyreadstat: {e}")
            print(f"ERROR: file_handler.PyreadstatError for '{file_name}': {e}")
            return None, None
        except ImportError as e: # Should be caught by pyreadstat check, but as safeguard
             st.error(f"A required library for {file_extension} is missing: {e}")
             print(f"ERROR: file_handler.ImportError for {file_extension}: {e}")
             return None, None
        except Exception as e:
            st.error(f"Error processing dataset '{file_name}': {e}")
            print(f"ERROR: file_handler.Error processing dataset '{file_name}': {e}")
            return None, None

    elif file_extension in [".rdata", ".rds"]:
        st.warning(f"File type '{file_extension}' for '{file_name}' is saved to workspace but not processed for agent tools. Please convert to CSV or XLSX if analysis is needed.")
        print(f"LOG: file_handler.Unsupported R file type: {file_extension} for '{file_name}'.")
        return "other_saved", file_name
    else:
        st.warning(f"Unsupported file type: {file_extension} for '{file_name}'. File saved to workspace but not processed for agent tools.")
        print(f"LOG: file_handler.Unsupported file type: {file_extension} for '{file_name}'.")
        return "other_saved", file_name

def remove_uploaded_file(file_name: str, file_type: str):
    """
    Removes an uploaded file from session state and from the UI_ACCESSIBLE_WORKSPACE.
    """
    if not file_name:
        print("ERROR: file_handler.remove_uploaded_file: file_name is None or empty.")
        return

    removed_from_session = False
    if file_type == "document":
        if "uploaded_documents" in st.session_state and file_name in st.session_state.uploaded_documents:
            del st.session_state.uploaded_documents[file_name]
            removed_from_session = True
    elif file_type == "dataframe":
        if "uploaded_dataframes" in st.session_state and file_name in st.session_state.uploaded_dataframes:
            del st.session_state.uploaded_dataframes[file_name]
            removed_from_session = True

    if removed_from_session:
        st.toast(f"File '{file_name}' (type: {file_type}) removed from agent access list.", icon="🗑️")
        print(f"LOG: file_handler.File '{file_name}' removed from session state ({file_type}).")

    file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
    if os.path.exists(file_path_in_workspace):
        try:
            os.remove(file_path_in_workspace)
            # Only show toast for successful physical deletion if not already shown for session state removal
            if not removed_from_session:
                 st.toast(f"File '{file_name}' deleted from workspace.", icon="🗑️")
            print(f"LOG: file_handler.Successfully deleted physical file: {file_path_in_workspace}")
        except (OSError, IOError) as e:
            print(f"ERROR: file_handler.Error deleting physical file '{file_path_in_workspace}': {e}")
            st.error(f"Error deleting physical file '{file_name}': {e}")
        except Exception as e: # Catch any other unexpected error
            print(f"ERROR: file_handler.Unexpected error deleting physical file '{file_path_in_workspace}': {e}")
            st.error(f"Unexpected error deleting physical file '{file_name}': {e}")

    else:
        print(f"LOG: file_handler.Physical file '{file_path_in_workspace}' not found for deletion (might have been removed previously or failed to save).")
        if not removed_from_session : # If it was not in session state either, it's truly "not found"
            st.warning(f"File '{file_name}' was not found in workspace or session.")


print(f"DEBUG: file_handler.py processed with error handling improvements. UI_ACCESSIBLE_WORKSPACE: {UI_ACCESSIBLE_WORKSPACE}")
