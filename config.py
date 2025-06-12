import os

# Determine the project root dynamically, assuming config.py is at the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# --- Hugging Face Hub Configuration ---
# It's recommended to set HF_TOKEN as an environment variable (e.g., in a .env file)
# and load it using `python-dotenv`.
HF_TOKEN = os.getenv("HF_TOKEN", None) 

# IMPORTANT: Replace "your-hf-username/your-rag-dataset" with your actual Hugging Face Dataset ID.
# This is where your RAG vector store will be uploaded and persisted.
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "your-hf-username/your-rag-dataset") 
HF_VECTOR_STORE_SUBDIR = "simple_vector_store" # Subdirectory within the HF dataset to store the vector store

# --- RAG Document Processing Configuration ---
CHUNK_SIZE = 512 # Size of text chunks for processing by the node parser
CHUNK_OVERLAP = 20 # Overlap between text chunks to maintain context

# Paths for source data and scraped web content
# These paths are relative to the PROJECT_ROOT
SOURCE_DATA_DIR_RELATIVE = "data" # Directory for local documents (e.g., PDFs, text files)
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, SOURCE_DATA_DIR_RELATIVE)

WEB_MARKDOWN_PATH_RELATIVE = "ragdb/web_markdown" # Directory where scraped web content (markdown) is stored
WEB_MARKDOWN_PATH = os.path.join(PROJECT_ROOT, WEB_MARKDOWN_PATH_RELATIVE)

WEBPAGES_FILE_RELATIVE = "ragdb/webpages.txt" # File containing URLs to scrape, one per line
WEBPAGES_FILE = os.path.join(PROJECT_ROOT, WEBPAGES_FILE_RELATIVE)

# --- Other API Keys (if needed by other modules in your project) ---
# These are included as common examples, ensure they are set as environment variables
# or directly in your .env file.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None) # For GoogleGenAIEmbedding
