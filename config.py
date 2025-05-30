import os

# Project Root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Agent Constants
SUGGESTED_PROMPT_COUNT = 4
DEFAULT_PROMPTS = [
    "Find recent papers on qualitative data analysis methods.",
    "Explain the structure of a typical literature review.",
    "What are common challenges students face in writing?",
    "Search for university policies on dissertation submission deadlines (uses RAG).",
]
ESI_AGENT_INSTRUCTION_FILE = "esi_agent_instruction.md"

# UI Constants / Markers
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"
CODE_DOWNLOAD_MARKER = "---DOWNLOAD_FILE---" # Used in stui.py, same as DOWNLOAD_MARKER

# File Paths & Directories
SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
# DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE) # app.py uses this, keep it there if only used there or pass PROJECT_ROOT
MEMORY_DIR_NAME = "user_memories"
# MEMORY_DIR = os.path.join(PROJECT_ROOT, MEMORY_DIR_NAME) # app.py uses this

CODE_WORKSPACE_RELATIVE_PATH = "./code_interpreter_ws"
# UI_ACCESSIBLE_WORKSPACE = os.path.join(PROJECT_ROOT, CODE_WORKSPACE_RELATIVE_PATH) # tools.py & stui.py

# RAG Configuration (tools.py)
HF_DATASET_ID = "gm42/esi_simplevector"
HF_VECTOR_STORE_SUBDIR = "vector_store_data"
# HF_PERSIST_PATH = f"datasets/{HF_DATASET_ID}/{HF_VECTOR_STORE_SUBDIR}" # tools.py

# Environment Variables (though these are typically accessed directly via os.getenv)
# GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
# TAVILY_API_KEY_ENV_VAR = "TAVILY_API_KEY"
# HF_TOKEN_ENV_VAR = "HF_TOKEN"

import logging
import sys

# Basic Logging Configuration
LOGGING_LEVEL = logging.INFO # Default level
# Consider making this configurable via an environment variable if needed later
# LOGGING_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout # Output logs to stdout
)

def get_logger(name: str):
    return logging.getLogger(name)
