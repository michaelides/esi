import os
import asyncio
from typing import List, Tuple
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.markdown import MarkdownReader
from llama_index.readers.json import JSONReader
from llama_index.readers.csv import CSVReader
from llama_index.readers.image import ImageReader
from llama_index.readers.file.docs import DocxReader
from llama_index.readers.file.epub import EpubReader
from llama_index.readers.file.html import HTMLReader
from llama_index.readers.file.ipynb import IPYNBReader
from llama_index.readers.file.mbox import MboxReader
from llama_index.readers.file.mp3 import Mp3Reader
from llama_index.readers.file.odt import ODTReader
from llama_index.readers.file.pdf import PDFReader
from llama_index.readers.file.ppt import PptxReader
from llama_index.readers.file.rtf import RtfReader
from llama_index.readers.file.xml import XMLReader
from llama_index.readers.file.video import VideoReader
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR
from llama_index.readers.file.image_caption import ImageCaptionReader
from llama_index.readers.file.flat_json import FlatReader
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.readers.file.slides import SlidesReader
from llama_index.readers.file.excel import ExcelReader

# Changed to relative imports for modules within the 'ragdb' package
from .document_processor import load_and_process_documents, download_hf_dataset
from .web_scraper import scrape_websites
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAGDB_DIR = os.path.join(PROJECT_ROOT, "ragdb")
WEB_MARKDOWN_DIR = os.path.join(RAGDB_DIR, "web_markdown")
LOCAL_DATA_DIR = os.path.join(RAGDB_DIR, "local_data")
SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
HF_DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "gm42/esi_simplevector/simple_vector_store")
WEB_PAGES_FILE = os.path.join(RAGDB_DIR, "webpages.txt")

# URLs to scrape (can be loaded from a file or defined here)
URLS_TO_SCRAPE = []
# Add a print statement to help debug the path
print(f"Attempting to read webpages from: {WEB_PAGES_FILE}")
if os.path.exists(WEB_PAGES_FILE):
    with open(WEB_PAGES_FILE, 'r') as f:
        URLS_TO_SCRAPE = [line.strip() for line in f if line.strip()]
else:
    print(f"Warning: Could not find {WEB_PAGES_FILE}. Please create this file in the project root directory and add URLs to scrape, one per line. No webpages will be scraped.")

async def main():
    print(f"Target Hugging Face Dataset for RAG persistence: {HF_DATASET_REPO_ID}")

    # 1. Scrape websites if URLs are provided
    if URLS_TO_SCRAPE:
        print(f"Scraping {len(URLS_TO_SCRAPE)} URLs...")
        await scrape_websites(URLS_TO_SCRAPE, WEB_MARKDOWN_DIR)
        print("Web scraping complete.")
    else:
        print("No URLs provided for scraping. Skipping web scraping.")

    # 2. Load and process documents
    print("Loading and processing documents...")
    all_documents, node_parser = load_and_process_documents()
    print(f"Loaded {len(all_documents)} documents.")

    # 3. Initialize embedding model
    print("Initializing embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=os.path.join(PROJECT_ROOT, "model_cache")
    )
    print("Embedding model initialized.")

    # 4. Setup ingestion pipeline
    print("Setting up ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=[node_parser, Settings.embed_model]
    )
    print("Ingestion pipeline set up.")

    # 5. Run ingestion and build index
    print("Running ingestion pipeline and building index...")
    nodes = pipeline.run(documents=all_documents, show_progress=True)
    print(f"Processed {len(nodes)} nodes.")

    vector_store = SimpleVectorStore.from_persist_dir(DB_PATH) if os.path.exists(DB_PATH) else SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print("Index built.")

    # 6. Persist the index locally
    print(f"Persisting index to {DB_PATH}...")
    index.storage_context.persist(persist_dir=DB_PATH)
    print("Index persisted locally.")

    # 7. Upload to Hugging Face Hub
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"Uploading vector store to Hugging Face Hub dataset: {HF_DATASET_REPO_ID}...")
        vector_store.to_hub(
            repo_id=HF_DATASET_REPO_ID,
            commit_message="Update RAG vector store",
            token=hf_token
        )
        print("Vector store uploaded to Hugging Face Hub.")
    else:
        print("HF_TOKEN environment variable not set. Skipping upload to Hugging Face Hub.")

    print("RAG database creation complete!")

if __name__ == "__main__":
    asyncio.run(main())
