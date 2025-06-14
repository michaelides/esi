import os
import json
import re
import uuid
import shutil # For file operations
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Path as FastApiPath
from pydantic import BaseModel
import uvicorn
import pandas as pd # For analyze_dataframe_tool
from PyPDF2 import PdfReader # For read_uploaded_document_tool (basic PDF text extraction)

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from agent import create_orchestrator_agent, initialize_settings as initialize_agent_settings
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SESSION_WORKSPACE_ROOT = os.path.join(PROJECT_ROOT, "user_workspaces")
os.makedirs(SESSION_WORKSPACE_ROOT, exist_ok=True)

# Paths and constants
# SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store") # Not directly used in this version
# DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE) # Not directly used
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"
MAX_CHAT_HISTORY_MESSAGES = 15

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ESI Agent API",
    description="API for interacting with the ESI Orchestrator Agent.",
    version="0.2.0", # Version bump for new features
)

# --- Global Variables & In-memory Stores ---
AGENT_INSTANCE = None # Global agent instance
SETTINGS_INITIALIZED = False # Flag for global LLM settings
# Session-specific stores (in-memory, replace with DB for production)
SESSION_SETTINGS: Dict[str, Dict[str, Any]] = {}
# Example: SESSION_SETTINGS = {"session123": {"temperature": 0.5, "verbosity": 4}}
SESSION_UPLOADED_FILES: Dict[str, Dict[str, str]] = {} # session_id -> {filename: filepath}

DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_VERBOSITY = 3
DEFAULT_MAX_SEARCH_RESULTS = 10

# --- Pydantic Models for API ---
class ChatMessageInput(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: str # Make session_id mandatory for chat
    chat_history: Optional[List[ChatMessageInput]] = []
    # user_id: Optional[str] = None # Replaced by session_id for this scope

class ChatResponse(BaseModel):
    assistant_response: str
    updated_chat_history: List[ChatMessageInput]
    session_id: str

class FileUploadResponse(BaseModel):
    message: str
    session_id: str
    filename: str
    filepath: str # Relative to session workspace for confirmation

class LLMSettings(BaseModel):
    temperature: Optional[float] = None
    verbosity: Optional[int] = None
    max_search_results: Optional[int] = None

class SessionLLMSettingsResponse(LLMSettings):
    session_id: str


# --- Session-Aware Tool Functions ---
def read_uploaded_document_tool_fn(tool_input_str: str) -> str:
    """
    Reads the full text content of a document previously uploaded by the user for a given session.
    Input is a JSON string: '{"session_id": "xyz", "filename": "my_doc.pdf"}'
    """
    try:
        tool_input = json.loads(tool_input_str)
        session_id = tool_input.get("session_id")
        filename = tool_input.get("filename")
    except json.JSONDecodeError:
        return "Error: Invalid input format for read_uploaded_document. Expected JSON string."

    if not session_id or not filename:
        return "Error: 'session_id' and 'filename' must be provided in the input JSON for read_uploaded_document."

    if session_id not in SESSION_UPLOADED_FILES or filename not in SESSION_UPLOADED_FILES[session_id]:
        available_files = list(SESSION_UPLOADED_FILES.get(session_id, {}).keys())
        return f"Error: Document '{filename}' not found for session '{session_id}'. Available documents: {available_files}"

    filepath = SESSION_UPLOADED_FILES[session_id][filename]

    try:
        if filename.lower().endswith(".pdf"):
            text_content = ""
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text_content += page.extract_text() or ""
            return f"Content of '{filename}':\n{text_content}"
        # Add more file types (e.g., .txt, .docx) here
        elif filename.lower().endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                return f"Content of '{filename}':\n{f.read()}"
        else:
            return f"Error: File type for '{filename}' not supported by read_uploaded_document. Supported: .pdf, .txt, .md"
    except Exception as e:
        return f"Error reading document '{filename}': {e}"

def analyze_dataframe_tool_fn(tool_input_str: str) -> str:
    """
    Provides summary information about a pandas DataFrame previously uploaded by the user for a given session.
    Input is a JSON string: '{"session_id": "xyz", "filename": "my_data.csv", "head_rows": 5}'
    """
    try:
        tool_input = json.loads(tool_input_str)
        session_id = tool_input.get("session_id")
        filename = tool_input.get("filename")
        head_rows = tool_input.get("head_rows", 5)
    except json.JSONDecodeError:
        return "Error: Invalid input format for analyze_uploaded_dataframe. Expected JSON string."

    if not session_id or not filename:
        return "Error: 'session_id' and 'filename' must be provided in the input JSON for analyze_uploaded_dataframe."

    if session_id not in SESSION_UPLOADED_FILES or filename not in SESSION_UPLOADED_FILES[session_id]:
        available_files = list(SESSION_UPLOADED_FILES.get(session_id, {}).keys())
        return f"Error: DataFrame '{filename}' not found for session '{session_id}'. Available dataframes: {available_files}"

    filepath = SESSION_UPLOADED_FILES[session_id][filename]

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(filepath)
        else:
            return f"Error: File type for '{filename}' not supported for dataframe analysis. Supported: .csv, .xls, .xlsx"

        info_str = f"DataFrame: {filename} (Session: {session_id})\n"
        info_str += f"Shape: {df.shape}\n"
        info_str += f"Columns: {', '.join(df.columns)}\n"
        info_str += f"Data Types:\n{df.dtypes.to_string()}\n"

        head_rows = max(0, min(head_rows, len(df)))
        if head_rows > 0:
            info_str += f"First {head_rows} rows:\n{df.head(head_rows).to_string()}\n"

        info_str += f"Summary Statistics:\n{df.describe(include='all').to_string()}\n"
        return info_str
    except Exception as e:
        return f"Error analyzing dataframe '{filename}': {e}"

# --- Agent Initialization ---
def initialize_global_llm_settings_once():
    global SETTINGS_INITIALIZED
    if not SETTINGS_INITIALIZED:
        print("Initializing global LLM settings for FastAPI app...")
        try:
            initialize_agent_settings() # This function is from agent.py
            SETTINGS_INITIALIZED = True
            print("Global LLM settings initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Fatal Error: Could not initialize LLM settings. {e}") from e

def get_agent_instance_with_dynamic_tools(session_id: Optional[str] = None) -> AgentRunner:
    """
    Initializes the orchestrator agent. If session_id is provided, it attempts to
    create tools that are aware of that session's context (e.g., uploaded files).
    The agent itself is global, but the tools it uses can be made session-specific if designed carefully.
    For LlamaIndex, FunctionTool wraps a Python function. The challenge is making that Python function
    aware of the session_id *at call time* when the agent executes the tool.
    The approach here is that the LLM will be prompted to pass the session_id as part of the tool input.
    """
    global AGENT_INSTANCE
    if AGENT_INSTANCE is None: # Initialize the base agent once
        print("Initializing base AI agent for FastAPI app (first time)...")
        try:
            # Create tool templates. The actual session_id will be part of the input string to the tool.
            uploaded_doc_reader_tool = FunctionTool.from_defaults(
                fn=read_uploaded_document_tool_fn,
                name="read_uploaded_document",
                description="Reads text from an uploaded document (PDF, TXT, MD). Input: JSON string '{\"session_id\": \"<session_id>\", \"filename\": \"<filename>\"}'."
            )
            dataframe_analyzer_tool = FunctionTool.from_defaults(
                fn=analyze_dataframe_tool_fn,
                name="analyze_uploaded_dataframe",
                description="Analyzes an uploaded data file (CSV, XLS, XLSX) and returns summary stats. Input: JSON string '{\"session_id\": \"<session_id>\", \"filename\": \"<filename>\", \"head_rows\": <optional_num_rows>}'."
            )
            dynamic_tools_templates = [uploaded_doc_reader_tool, dataframe_analyzer_tool]

            # Get default settings for the agent (e.g., max_search_results for other tools)
            # These are not session specific at the agent creation level.
            agent_creation_settings = SESSION_SETTINGS.get("global_defaults", {}) # Or some other default mechanism
            max_results = agent_creation_settings.get("max_search_results", DEFAULT_MAX_SEARCH_RESULTS)

            AGENT_INSTANCE = create_orchestrator_agent(
                dynamic_tools=dynamic_tools_templates, # Pass tool templates
                max_search_results=max_results
            )
            print("Base AI agent initialized successfully with tool templates.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the AI agent. Error: {e}") from e

    # The AGENT_INSTANCE is global. Specific session settings (like temperature)
    # will be applied per-call in the chat_endpoint.
    return AGENT_INSTANCE


# --- Helper Functions ---
def format_api_history_to_llama_history(api_chat_history: List[ChatMessageInput]) -> List[ChatMessage]:
    history = []
    truncated_messages = api_chat_history[-MAX_CHAT_HISTORY_MESSAGES:]
    for msg_in in truncated_messages:
        role = MessageRole.USER if msg_in.role == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg_in.content))
    return history

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    print("Application startup...")
    initialize_global_llm_settings_once()
    get_agent_instance_with_dynamic_tools() # Initialize base agent on startup
    print("Application startup complete.")

@app.post("/api/upload_file/{session_id}", response_model=FileUploadResponse)
async def upload_file_endpoint(session_id: str = FastApiPath(...), file: UploadFile = File(...)):
    session_dir = os.path.join(SESSION_WORKSPACE_ROOT, session_id, "uploads")
    os.makedirs(session_dir, exist_ok=True)

    filepath = os.path.join(session_dir, file.filename)

    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if session_id not in SESSION_UPLOADED_FILES:
            SESSION_UPLOADED_FILES[session_id] = {}
        SESSION_UPLOADED_FILES[session_id][file.filename] = filepath

        print(f"File '{file.filename}' uploaded for session '{session_id}' to '{filepath}'")
        return FileUploadResponse(
            message="File uploaded successfully",
            session_id=session_id,
            filename=file.filename,
            filepath=os.path.relpath(filepath, SESSION_WORKSPACE_ROOT) # Return relative path for confirmation
        )
    except Exception as e:
        print(f"Error uploading file '{file.filename}' for session '{session_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")

@app.post("/api/settings/{session_id}", response_model=SessionLLMSettingsResponse)
async def update_llm_settings_endpoint(session_id: str = FastApiPath(...), settings: LLMSettings = Body(...)):
    if session_id not in SESSION_SETTINGS:
        SESSION_SETTINGS[session_id] = {}

    updated_any = False
    if settings.temperature is not None:
        SESSION_SETTINGS[session_id]["temperature"] = settings.temperature
        updated_any = True
    if settings.verbosity is not None:
        SESSION_SETTINGS[session_id]["verbosity"] = settings.verbosity
        updated_any = True
    if settings.max_search_results is not None:
        SESSION_SETTINGS[session_id]["max_search_results"] = settings.max_search_results
        # Note: max_search_results for some tools might be part of agent re-initialization
        # or specific tool call parameters, not just a simple LLM property.
        # For now, we store it; its application might need more work.
        updated_any = True

    if not updated_any:
        raise HTTPException(status_code=400, detail="No settings provided to update.")

    current_settings = {
        "temperature": SESSION_SETTINGS[session_id].get("temperature", DEFAULT_LLM_TEMPERATURE),
        "verbosity": SESSION_SETTINGS[session_id].get("verbosity", DEFAULT_LLM_VERBOSITY),
        "max_search_results": SESSION_SETTINGS[session_id].get("max_search_results", DEFAULT_MAX_SEARCH_RESULTS),
    }
    print(f"Settings updated for session '{session_id}': {current_settings}")
    return SessionLLMSettingsResponse(session_id=session_id, **current_settings)

@app.get("/api/settings/{session_id}", response_model=SessionLLMSettingsResponse)
async def get_llm_settings_endpoint(session_id: str = FastApiPath(...)):
    if session_id not in SESSION_SETTINGS:
        # Return default settings if no specific settings for session
        default_s = {
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "verbosity": DEFAULT_LLM_VERBOSITY,
            "max_search_results": DEFAULT_MAX_SEARCH_RESULTS,
        }
        return SessionLLMSettingsResponse(session_id=session_id, **default_s)

    s = SESSION_SETTINGS[session_id]
    return SessionLLMSettingsResponse(
        session_id=session_id,
        temperature=s.get("temperature", DEFAULT_LLM_TEMPERATURE),
        verbosity=s.get("verbosity", DEFAULT_LLM_VERBOSITY),
        max_search_results=s.get("max_search_results", DEFAULT_MAX_SEARCH_RESULTS),
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    print(f"Received chat request for session_id: {session_id}")

    if not SETTINGS_INITIALIZED:
        raise HTTPException(status_code=500, detail="Global LLM Settings not initialized.")

    agent = get_agent_instance_with_dynamic_tools(session_id) # Pass session_id for context
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    llama_chat_history = format_api_history_to_llama_history(request.chat_history or [])

    session_specific_settings = SESSION_SETTINGS.get(session_id, {})
    current_temp = session_specific_settings.get("temperature", DEFAULT_LLM_TEMPERATURE)
    current_verb = session_specific_settings.get("verbosity", DEFAULT_LLM_VERBOSITY)
    # max_search_results for agent tools is tricky if set per session after agent init.
    # The current agent structure uses max_search_results at init time for some tools.
    # This might require agent re-creation or modification of tool invocation if truly dynamic.
    # For now, agent is initialized with default max_search_results.

    try:
        if Settings.llm:
            if hasattr(Settings.llm, 'temperature'):
                Settings.llm.temperature = current_temp
            else:
                print(f"Warning: Settings.llm ({type(Settings.llm)}) does not have a 'temperature' attribute.")
        else:
            print("Warning: Settings.llm is not initialized. Cannot set temperature.")

        # The agent's system prompt is expected to guide the LLM to include session_id in tool inputs.
        # E.g., "For tools like read_uploaded_document, provide input as JSON: {'session_id': 'CURRENT_SESSION_ID', ...}"
        # We need to ensure the agent's system prompt includes instructions to use the actual session_id.
        # This is a bit indirect. A more robust way would be to curry session_id into tool functions
        # if the agent framework allows, or have tools that can access a request context.
        # For now, we rely on the LLM being prompted correctly.
        
        # Prepending session_id to the query or modifying system prompt per call is complex.
        # The tools themselves are defined to expect session_id in their input string.
        # Prepend session_id to the query so the LLM is aware of it for tool calls.
        # The system prompt will instruct the LLM on how to use this.
        modified_query = f"Current Session ID: {session_id}. Verbosity Level: {current_verb}. {request.query}"
        # The user query should naturally include filenames. The LLM needs to be prompted
        # to format the tool input string for read_uploaded_document or analyze_dataframe
        # to include the Current Session ID from the query and the filename.
        # Example of what LLM should generate as input to read_uploaded_document_tool_fn:
        # '{"session_id": "session123", "filename": "mydoc.pdf"}'
        
        print(f"Processing query for session {session_id}: '{modified_query[:100]}...' with history length: {len(llama_chat_history)}")
        
        agent_response_obj = agent.chat(modified_query, chat_history=llama_chat_history)
        assistant_response_text = agent_response_obj.response if hasattr(agent_response_obj, 'response') else str(agent_response_obj)
        print(f"Agent response for session {session_id}: '{assistant_response_text[:100]}...'")

        updated_api_history = list(request.chat_history or [])
        updated_api_history.append(ChatMessageInput(role="user", content=request.query))
        updated_api_history.append(ChatMessageInput(role="assistant", content=assistant_response_text))

        return ChatResponse(
            assistant_response=assistant_response_text,
            updated_chat_history=updated_api_history,
            session_id=session_id
        )

    except Exception as e:
        print(f"Error during chat processing for session {session_id}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the ESI Agent API. Use /docs for API documentation."}

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not set. The agent may not work properly.")

    try:
        initialize_global_llm_settings_once()
        get_agent_instance_with_dynamic_tools() # Initialize base agent
    except RuntimeError as e:
        print(f"Failed to initialize application: {e}")
        import sys
        sys.exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000)

print("app.py successfully parsed and ready for FastAPI execution with session capabilities.")
