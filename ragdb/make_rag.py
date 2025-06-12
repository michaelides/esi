import os
import sys
import asyncio
from urllib.parse import urlparse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from huggingface_hub import HfApi
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables from a .env file if it exists
load_dotenv()

# Add the project root to sys.path to allow importing config.py
# This assumes config.py is in the directory above ragdb/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

# PROJECT_ROOT and other configurations are now imported from config.py
from config import (
    PROJECT_ROOT,
    HF_DATASET_ID,
    HF_VECTOR_STORE_SUBDIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SOURCE_DATA_DIR_RELATIVE, # Keep for reference if SOURCE_DATA_DIR defined here
    SOURCE_DATA_DIR,
    WEB_MARKDOWN_PATH_RELATIVE, # Keep for reference if WEB_MARKDOWN_PATH defined here
    WEB_MARKDOWN_PATH,
    WEBPAGES_FILE_RELATIVE, # Keep for reference if WEBPAGES_FILE defined here
    WEBPAGES_FILE
)

# Ensure HF_TOKEN is set for writing to Hugging Face Hub
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN environment variable is not set. Make sure you are logged in via `huggingface-cli login` or have set HF_TOKEN to write to the Hugging Face Dataset.")

print(f"Target Hugging Face Dataset for RAG persistence: {HF_DATASET_ID}/{HF_VECTOR_STORE_SUBDIR}")

# CHUNK_SIZE, CHUNK_OVERLAP, SOURCE_DATA_DIR, WEB_MARKDOWN_PATH, WEBPAGES_FILE are now imported from config.py

URLS_TO_SCRAPE = []
try:
    with open(WEBPAGES_FILE, 'r') as file:
        # Strip whitespace/newlines from each line
        URLS_TO_SCRAPE = [line.strip() for line in file if line.strip()]
    if not URLS_TO_SCRAPE:
        print(f"Warning: {WEBPAGES_FILE_RELATIVE} is empty. No webpages will be scraped.")
except FileNotFoundError:
    print(f"Warning: Could not find {WEBPAGES_FILE_RELATIVE}. Please create this file in the project root directory and add URLs to scrape, one per line. No webpages will be scraped.")
except Exception as e:
    print(f"Error reading {WEBPAGES_FILE_RELATIVE}: {e}. No webpages will be scraped.")

# url_to_filename and scrape_websites functions have been moved to ragdb.web_scraper
from ragdb.web_scraper import scrape_websites
# Import document processing function
from ragdb.document_processor import load_and_process_documents

# --- Main Script ---
async def main():
    # --- Initialize Embedding Model ---
    embedding_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

    print(f"Configuring RAG to persist to Hugging Face Dataset: {HF_DATASET_ID}, path in repo: {HF_VECTOR_STORE_SUBDIR}")

    # 1. Scrape websites
    # WEB_MARKDOWN_PATH is from config and is used by scrape_websites internally if not passed (though it is passed here)
    # URLS_TO_SCRAPE is defined in this file (make_rag.py)
    await scrape_websites(URLS_TO_SCRAPE, WEB_MARKDOWN_PATH)

    # 2. Load and process documents
    # This function now encapsulates document loading and SentenceSplitter initialization
    # It uses SOURCE_DATA_DIR, WEB_MARKDOWN_PATH, CHUNK_SIZE, CHUNK_OVERLAP from config.py
    all_documents, node_parser = load_and_process_documents()

    if not all_documents:
        print("make_rag.py: No documents loaded by document_processor. Exiting RAG pipeline.")
        return

    # 3. Initialize SimpleVectorStore and Storage Context for local persistence
    # Node parser is now returned by load_and_process_documents
    print("make_rag.py: Initializing SimpleVectorStore for local persistence...")
    vector_store = SimpleVectorStore()
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Create VectorStoreIndex (This performs parsing, embedding, and indexing)
    print(f"Creating index for {len(all_documents)} documents... (This may take a while)")
    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=embedding_model,
        node_parser=node_parser,
        show_progress=True,
    )

    # 6. Persist the index locally to a temporary directory
    local_persist_dir = tempfile.mkdtemp()
    print(f"Persisting index locally to temporary directory: {local_persist_dir}...")
    try:
        index.storage_context.persist(persist_dir=local_persist_dir)
        print("Local persistence successful.")

        # 7. Upload the persisted data to Hugging Face Dataset
        print(f"Uploading persisted index to Hugging Face Dataset: {HF_DATASET_ID}, path in repo: {HF_VECTOR_STORE_SUBDIR}...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set. Upload to Hugging Face Hub will likely fail or use cached credentials.")
            
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=local_persist_dir,
            repo_id=HF_DATASET_ID,
            path_in_repo=HF_VECTOR_STORE_SUBDIR,
            repo_type="dataset",
        )
        print(f"Successfully uploaded index to Hugging Face Dataset: {HF_DATASET_ID}/{HF_VECTOR_STORE_SUBDIR}")

    except Exception as e:
        print(f"An error occurred during local persistence or upload: {e}")
    finally:
        if os.path.exists(local_persist_dir):
            print(f"Cleaning up temporary local persistence directory: {local_persist_dir}")
            shutil.rmtree(local_persist_dir)

    print("RAG database creation script finished.")


if __name__ == "__main__":
   asyncio.run(main())
