import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant # Changed from langchain_chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from crawl4ai import AsyncWebCrawler # Import crawl4ai
import glob
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st # Keep streamlit import if needed for st.info/warning, otherwise remove

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

QDRANT_DB_PATH = "./qdrant_db" # Changed path name
COLLECTION_NAME = "dissertation_resources" # Assuming the same collection name as in app.py

# List of URLs to scrape
URLS_TO_SCRAPE = [
#    "https://www.uea.ac.uk/course/postgraduate/msc-management", # Example URL 1
#    "https://www.uea.ac.uk/about/university-information/campus-map", # Example URL 2
#    "https://www.uea.ac.uk/course/postgraduate/msc-organisational-psychology", # Added comma
#    "https://scholar.google.co.uk/citations?user=v9Rzv3kAAAAJ&hl=en", # Added comma
    "https://research-portal.uea.ac.uk/en/persons/kevin-daniels/publications/"
    # Add more relevant URLs here
]

# --- Initialization ---
try:
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
    exit(1)

try:
    # Initialize Qdrant vector store for the main knowledge base
    # This uses a local directory for persistence
    vector_store = Qdrant(
        location=QDRANT_DB_PATH, # Use location for local persistence
        collection_name=COLLECTION_NAME,
        embeddings=embedding_function,
    )
    logging.info(f"Connected to Qdrant DB at {QDRANT_DB_PATH} with collection '{COLLECTION_NAME}'")
except Exception as e:
    logging.error(f"Failed to connect to Qdrant DB: {e}")
    exit(1)

# Initialize crawl4ai with Markdown output and PDF extraction
crawler = AsyncWebCrawler(output_format="markdown", extract_pdf=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Keep chunk size reasonable for Markdown

# --- Core Logic ---
async def scrape_and_store():
    """Scrapes URLs using crawl4ai, splits content, and adds to Qdrant DB.""" # Updated description
    logging.info("Starting web scraping process with crawl4ai...")
    all_splits = []
    processed_urls = 0
    failed_urls = []

    try:
        for url in URLS_TO_SCRAPE:
            logging.info(f"Attempting to crawl: {url}")
            try:
                # Use crawl4ai to get content (Markdown + PDFs) for each URL
                # arun returns a CrawlResultContainer
                crawl_result_container = await crawler.arun(url)

                if crawl_result_container:
                    logging.info(f"Successfully crawled content from: {url}.")
                    # Extract markdown content if available
                    page_content = crawl_result_container.markdown # Access the markdown attribute
                    source_url = url  #crawl_result_container.url # Get the specific source URL (could be original or PDF link)

                    if page_content:
                        # Create a Document object for each result (HTML page or PDF)
                        doc = Document(page_content=page_content, metadata={"source": source_url})

                        # Split the document content
                        splits = text_splitter.split_documents([doc]) # Pass as list
                        all_splits.extend(splits)
                        processed_urls += 1 # Mark the original URL as processed if any content was found
                    else:
                        logging.warning(f"No markdown content found for URL: {url}")
                        failed_urls.append(url)

                else:
                    logging.warning(f"No results returned by crawl4ai for URL: {url}")
                    failed_urls.append(url)

            except Exception as crawl_error:
                # Log the specific error, including validation errors
                logging.error(f"Error processing URL {url}: {crawl_error}", exc_info=True)
                failed_urls.append(url)
                continue # Move to the next URL

        logging.info(f"Finished crawling. Successfully processed {processed_urls} URLs.")
        if failed_urls:
            logging.warning(f"Failed to crawl or get content from {len(failed_urls)} URLs: {failed_urls}")

        if not all_splits:
            logging.warning("No content was successfully scraped and split from the provided URLs.")
            return

        logging.info(f"Split content into {len(all_splits)} chunks.")

        # Add documents to Qdrant DB
        logging.info("Adding documents to Qdrant DB...") # Updated message
        # Qdrant's add_documents handles batching internally
        vector_store.add_documents(all_splits)
        logging.info("Successfully added documents to Qdrant DB.") # Updated message

        # Optional: Persist changes explicitly if needed (Qdrant usually auto-persists with local storage)
        # vector_store.persist() # Qdrant doesn't have a .persist() method like Chroma

    except Exception as e:
        logging.error(f"An error occurred during scraping or storing: {e}", exc_info=True)

# --- PDF Ingestion Function (Main Knowledge Base) ---
def check_and_ingest_new_pdfs(data_directory: str, vector_store_instance: Qdrant, db_path: str): # Changed type hint
    """Checks for new PDFs in data_directory and ingests them into the main vector store."""
    log_file_path = os.path.join(db_path, ".ingested_files.log")
    ingested_files = set()

    # Create data directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created data directory: {data_directory}") # Use print for console output

    # Read list of already ingested files
    try:
        with open(log_file_path, "r") as f:
            ingested_files = set(line.strip() for line in f)
        print(f"Found log of {len(ingested_files)} previously ingested files.")
    except FileNotFoundError:
        print("No ingestion log file found for the main knowledge base. Will process all PDFs in data directory.")
    except Exception as e:
         print(f"Error reading ingestion log file: {e}")


    # Find current PDF files
    current_pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
    current_filenames = set(os.path.basename(f) for f in current_pdf_files)

    # Determine new files to ingest
    new_filenames = current_filenames - ingested_files
    new_file_paths = [os.path.join(data_directory, fname) for fname in new_filenames]

    if not new_file_paths:
        print("Main knowledge base is up-to-date. No new PDFs found to ingest.")
        return

    print(f"Found {len(new_file_paths)} new PDF file(s) to ingest into the main knowledge base. Starting process...")

    all_new_docs = []
    processed_new_files = []
    for pdf_path in new_file_paths:
        filename = os.path.basename(pdf_path)
        try:
            print(f"  Loading: {filename}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_new_docs.extend(documents)
            processed_new_files.append(filename) # Track successfully loaded files
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue # Skip to the next file

    if not all_new_docs:
        print("No new documents were successfully loaded from the PDF files for the main knowledge base.")
        return

    print("Splitting new documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_new_docs)

    if not splits:
        print("Failed to split new documents into chunks for the main knowledge base.")
        return

    print(f"Adding {len(splits)} new document chunks to the main vector store...")
    try:
        # Add the new documents to the vector store
        # Qdrant's add_documents handles batching internally
        vector_store_instance.add_documents(splits)

        # Update the log file with newly processed files
        with open(log_file_path, "a") as f:
            for fname in processed_new_files:
                f.write(f"{fname}\n")
        print(f"Successfully ingested {len(processed_new_files)} new PDF file(s) into the main knowledge base!")

    except Exception as e:
        print(f"Error adding new documents to main vector store: {e}")


# --- Main Execution ---
DATA_DIR = "./data"  # Directory to store PDF files
if __name__ == "__main__":
    # Run web scraping and ingestion
    if not URLS_TO_SCRAPE:
        logging.warning("No URLs provided in URLS_TO_SCRAPE list. Skipping web scraping.")
    else:
        # Run the asynchronous function
        asyncio.run(scrape_and_store())
        logging.info("Web scraping script finished.")

    # Run PDF ingestion
    print("Checking for new documents to load into the main knowledge base...")
    check_and_ingest_new_pdfs(DATA_DIR, vector_store, QDRANT_DB_PATH) # Pass the Qdrant instance and new path
    print("PDF ingestion script finished.")

    logging.info("Script finished.")
