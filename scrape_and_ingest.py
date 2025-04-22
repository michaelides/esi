import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models # Import models for VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from crawl4ai import AsyncWebCrawler
import glob
from langchain_community.document_loaders import PyPDFLoader
# import streamlit as st # Removed streamlit import as it's not used in this script

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

QDRANT_DB_PATH = "./qdrant_db"
COLLECTION_NAME = "dissertation_resources"
# Define vector parameters for the collection (based on the embedding model)
# Corrected VECTOR_SIZE based on the error message (768 instead of 3072)
VECTOR_SIZE = 768 # Dimension for models/text-embedding-004 (as reported by error)
DISTANCE_METRIC = models.Distance.COSINE # Common distance metric

# List of URLs to scrape
URLS_TO_SCRAPE = [
#    "https://www.uea.ac.uk/course/postgraduate/msc-management", # Example URL 1
#    "https://www.uea.ac.uk/about/university-information/campus-map", # Example URL 2
#    "https://www.uea.ac.uk/course/postgraduate/msc-organisational-psychology", # Added comma
#    "https://scholar.google.co.uk/citations?user=v9Rzv3kAAAAJ&hl=en", # Added comma
    "https://research-portal.uea.ac.uk/en/persons/kevin-daniels/publications/"
    # Add more relevant URLs here
]

# --- Initialization (Embeddings) ---
try:
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    logging.info("GoogleGenerativeAIEmbeddings initialized.")
except Exception as e:
    logging.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
    exit(1)

# Initialize crawl4ai with Markdown output and PDF extraction
crawler = AsyncWebCrawler(output_format="markdown", extract_pdf=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- Core Logic ---
async def scrape_and_collect_documents():
    """Scrapes URLs using crawl4ai, splits content, and returns documents."""
    logging.info("Starting web scraping process with crawl4ai...")
    all_splits = []
    processed_urls = 0
    failed_urls = []

    for url in URLS_TO_SCRAPE:
        logging.info(f"Attempting to crawl: {url}")
        try:
            crawl_result_container = await crawler.arun(url)

            if crawl_result_container:
                logging.info(f"Successfully crawled content from: {url}.")
                page_content = crawl_result_container.markdown
                source_url = url

                if page_content:
                    # Ensure metadata is pickleable - only keep the source URL string
                    doc = Document(page_content=page_content, metadata={"source": source_url})
                    splits = text_splitter.split_documents([doc])
                    all_splits.extend(splits)
                    processed_urls += 1
                else:
                    logging.warning(f"No markdown content found for URL: {url}")
                    failed_urls.append(url)
            else:
                logging.warning(f"No results returned by crawl4ai for URL: {url}")
                failed_urls.append(url)

        except Exception as crawl_error:
            logging.error(f"Error processing URL {url}: {crawl_error}", exc_info=True)
            failed_urls.append(url)
            continue

    logging.info(f"Finished crawling. Successfully processed {processed_urls} URLs.")
    if failed_urls:
        logging.warning(f"Failed to crawl or get content from {len(failed_urls)} URLs: {failed_urls}")

    logging.info(f"Collected {len(all_splits)} document chunks from web scraping.")
    return all_splits

# --- PDF Ingestion Function (Main Knowledge Base) ---
def check_and_collect_new_pdfs(data_directory: str, db_path: str):
    """Checks for new PDFs in data_directory, loads and splits them, and returns documents."""
    log_file_path = os.path.join(db_path, ".ingested_files.log")
    ingested_files = set()
    all_new_docs = []
    processed_new_files = []

    # Create data directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        logging.info(f"Created data directory: {data_directory}")

    # Read list of already ingested files
    try:
        # Check if log file exists before trying to open
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                ingested_files = set(line.strip() for line in f)
            logging.info(f"Found log of {len(ingested_files)} previously ingested files.")
        else:
             logging.info("No ingestion log file found for the main knowledge base.")

    except Exception as e:
         logging.error(f"Error reading ingestion log file: {e}")

    # Find current PDF files
    current_pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
    current_filenames = set(os.path.basename(f) for f in current_pdf_files)

    # Determine new files to ingest
    new_filenames = current_filenames - ingested_files
    new_file_paths = [os.path.join(data_directory, fname) for fname in new_filenames]

    if not new_file_paths:
        logging.info("Main knowledge base is up-to-date. No new PDFs found to ingest.")
        return [], [] # Return empty list of docs and processed files

    logging.info(f"Found {len(new_file_paths)} new PDF file(s) to ingest into the main knowledge base. Starting process...")

    loaded_documents = []
    for pdf_path in new_file_paths:
        filename = os.path.basename(pdf_path)
        try:
            logging.info(f"  Loading: {filename}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            # Explicitly clean metadata to ensure pickleability
            cleaned_documents = []
            for doc in documents:
                 # Keep only the source path in metadata
                 cleaned_documents.append(Document(page_content=doc.page_content, metadata={"source": pdf_path}))
            loaded_documents.extend(cleaned_documents)
            processed_new_files.append(filename) # Track successfully loaded files
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            continue # Skip to the next file

    if not loaded_documents:
        logging.warning("No new documents were successfully loaded from the PDF files for the main knowledge base.")
        return [], [] # Return empty list of docs and processed files

    logging.info("Splitting new documents into chunks...")
    splits = text_splitter.split_documents(loaded_documents) # Split the cleaned documents

    if not splits:
        logging.warning("Failed to split new documents into chunks for the main knowledge base.")
        return [], [] # Return empty list of docs and processed files

    logging.info(f"Collected {len(splits)} new document chunks from PDFs.")
    return splits, processed_new_files # Return the splits and the list of successfully processed filenames

# --- Main Execution ---
DATA_DIR = "./data"  # Directory to store PDF files

if __name__ == "__main__":
    # Initialize Qdrant client ONCE for the local persistence path
    try:
        client = QdrantClient(path=QDRANT_DB_PATH)
        logging.info(f"Qdrant client initialized at {QDRANT_DB_PATH}.")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
        exit(1)

    # Check if the collection exists
    collection_exists = False
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        collection_exists = True
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
    except ValueError: # Qdrant client raises ValueError if collection not found
        collection_exists = False
        logging.info(f"Collection '{COLLECTION_NAME}' not found.")
    except Exception as e:
        logging.error(f"Error checking for collection existence: {e}", exc_info=True)
        # If we can't even check existence, something is very wrong. Exit.
        exit(1)

    # If collection exists, delete it and the log file to ensure a clean state
    if collection_exists:
        logging.warning(f"Deleting existing collection '{COLLECTION_NAME}' to ensure compatibility...")
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logging.warning(f"Collection '{COLLECTION_NAME}' deleted.")

            # Delete the ingestion log file as the collection is being rebuilt
            log_file_path = os.path.join(QDRANT_DB_PATH, ".ingested_files.log")
            if os.path.exists(log_file_path):
                 logging.info(f"Deleting ingestion log file {log_file_path} due to collection recreation.")
                 os.remove(log_file_path)

        except Exception as e:
             logging.error(f"Error deleting collection or log file: {e}", exc_info=True)
             exit(1) # Exit if deletion fails

    # Collect documents *after* potentially deleting the old log file
    all_documents_to_ingest = []
    successfully_processed_pdf_filenames = []

    # Collect documents from web scraping
    if not URLS_TO_SCRAPE:
        logging.warning("No URLs provided in URLS_TO_SCRAPE list. Skipping web scraping.")
    else:
        scraped_docs = asyncio.run(scrape_and_collect_documents())
        all_documents_to_ingest.extend(scraped_docs)
        logging.info(f"Collected {len(scraped_docs)} documents from web scraping.")

    # Collect documents from PDF ingestion
    logging.info("Checking for new documents to load into the main knowledge base...")
    # Pass the QDRANT_DB_PATH to check_and_collect_new_pdfs so it can find the log file
    pdf_docs, processed_pdf_filenames = check_and_collect_new_pdfs(DATA_DIR, QDRANT_DB_PATH)
    all_documents_to_ingest.extend(pdf_docs)
    successfully_processed_pdf_filenames.extend(processed_pdf_filenames)
    logging.info(f"Collected {len(pdf_docs)} documents from PDF ingestion.")


    if not all_documents_to_ingest:
        logging.info("No new documents found from either web scraping or PDF ingestion. Nothing to add to Qdrant.")
        exit(0) # Exit successfully as there's nothing to do

    logging.info(f"Total documents collected for ingestion: {len(all_documents_to_ingest)}")

    # Create the collection (it won't exist at this point if it was deleted or never existed)
    logging.info(f"Creating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC),
        )
        logging.info(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        logging.error(f"Error creating collection: {e}", exc_info=True)
        exit(1)


    try:
        # Initialize vector store using the existing client and the newly created collection
        vector_store = QdrantVectorStore(
            client=client, # Use the pre-initialized client
            collection_name=COLLECTION_NAME,
            embedding=embedding_function,
        )

        # Add documents to the collection using add_documents
        logging.info(f"Adding {len(all_documents_to_ingest)} documents to collection '{COLLECTION_NAME}'...")
        vector_store.add_documents(all_documents_to_ingest)
        logging.info("Successfully added documents to Qdrant DB.")

        # Update the PDF ingestion log file with newly processed files
        if successfully_processed_pdf_filenames:
            log_file_path = os.path.join(QDRANT_DB_PATH, ".ingested_files.log")
            try:
                # Use 'a' mode to append to the log file. If the collection was recreated,
                # the log file was deleted, so 'a' will create a new one.
                with open(log_file_path, "a") as f:
                    for fname in successfully_processed_pdf_filenames:
                        f.write(f"{fname}\n")
                logging.info(f"Updated ingestion log with {len(successfully_processed_pdf_filenames)} new PDF file(s).")
            except Exception as e:
                logging.error(f"Error writing to ingestion log file: {e}")


    except Exception as e:
        logging.error(f"An error occurred during Qdrant ingestion: {e}", exc_info=True)
        exit(1)

    logging.info("Ingestion script finished.")
