# Configuration settings
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Settings moved from app.py
SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"
MEMORY_DIR = os.path.join(PROJECT_ROOT, "user_memories")
MAX_CHAT_HISTORY_MESSAGES = 15

# Settings moved from agent.py
SUGGESTED_PROMPT_COUNT = 4
DEFAULT_PROMPTS = [
    "Help me brainstorm ideas.",
    "I need to develop my research questions.",
    "I have my topic and I need help with developing hypotheses.",
    "I have my hypotheses and I need help to design the study.",
    "Can you help me design my qualitative study?"
]

# Settings moved from tools.py
HF_DATASET_ID = "gm42/esi_simplevector"
HF_VECTOR_STORE_SUBDIR = "vector_store_data"
UI_ACCESSIBLE_WORKSPACE_RELATIVE = "code_interpreter_ws"
UI_ACCESSIBLE_WORKSPACE = os.path.join(PROJECT_ROOT, UI_ACCESSIBLE_WORKSPACE_RELATIVE)

# Settings moved from ragdb/make_rag.py
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
SOURCE_DATA_DIR_RELATIVE = os.path.join("ragdb", "source_data")
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, SOURCE_DATA_DIR_RELATIVE)
WEB_MARKDOWN_PATH_RELATIVE = os.path.join("ragdb", "web_markdown")
WEB_MARKDOWN_PATH = os.path.join(PROJECT_ROOT, WEB_MARKDOWN_PATH_RELATIVE)
WEBPAGES_FILE_RELATIVE = os.path.join('ragdb', 'webpages.txt')
WEBPAGES_FILE = os.path.join(PROJECT_ROOT, WEBPAGES_FILE_RELATIVE)
