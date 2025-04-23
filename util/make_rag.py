import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Removed Qdrant imports
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient, models # Import models for VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from crawl4ai import AsyncWebCrawler
import glob
from langchain_community.document_loaders import PyPDFLoader
# Import LanceDB - Corrected import to only get LanceDB class
from langchain_community.vectorstores import LanceDB # Keep this import for potential future use or reference, though we'll use lancedb client directly for creation
import lancedb # Import lancedb client - This is the correct import for lancedb.connect()
import pyarrow as pa # Import pyarrow for schema definition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Changed path name for LanceDB
LANCEDB_DB_PATH = "../lancedb_db"
COLLECTION_NAME = "dissertation_resources" # This will be the table name in LanceDB

# Define the vector size for the embedding model
# models/text-embedding-004 has a dimension of 768
VECTOR_SIZE = 768

# Removed Qdrant specific vector parameters
# DISTANCE_METRIC = models.Distance.COSINE # Common distance metric

# List of URLs to scrape
URLS_TO_SCRAPE = [
#    "https://www.uea.ac.uk/course/postgraduate/msc-management", # Example URL 1
#    "https://www.uea.ac.uk/about/university-information/campus-map", # Example URL 2
#    "https://www.uea.ac.uk/course/postgraduate/msc-organisational-psychology", # Added comma
    "https://scholar.google.co.uk/citations?user=v9Rzv3kAAAAJ&hl=en", # Added comma
#    "https://research-portal.uea.ac.uk/en/persons/kevin-daniels/publications/"
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
    # Log file path is relative to the DB path
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
    # Pass the LANCEDB_DB_PATH to check_and_collect_new_pdfs so it can find the log file
    pdf_docs, processed_pdf_filenames = check_and_collect_new_pdfs(DATA_DIR, LANCEDB_DB_PATH)
    all_documents_to_ingest.extend(pdf_docs)
    successfully_processed_pdf_filenames.extend(processed_pdf_filenames)
    logging.info(f"Collected {len(pdf_docs)} documents from PDF ingestion.")


    if not all_documents_to_ingest:
        logging.info("No new documents found from either web scraping or PDF ingestion. Nothing to add to LanceDB.")
        exit(0) # Exit successfully as there's nothing to do

    logging.info(f"Total documents collected for ingestion: {len(all_documents_to_ingest)}")

    try:
        # Connect to or create the LanceDB database
        logging.info(f"Connecting to LanceDB database at {LANCEDB_DB_PATH}...")
        # Use the correctly imported lancedb for connect
        db = lancedb.connect(LANCEDB_DB_PATH)
        logging.info("Connected to LanceDB.")

        # --- Prepare data for LanceDB (Generate Embeddings and format as dictionaries) ---
        logging.info(f"Generating embeddings for {len(all_documents_to_ingest)} documents...")
        # Embed all documents in a batch for efficiency
        texts_to_embed = [doc.page_content for doc in all_documents_to_ingest]
        embeddings = embedding_function.embed_documents(texts_to_embed)
        logging.info("Embeddings generated.")

        data_to_add = []
        for i, doc in enumerate(all_documents_to_ingest):
            doc_dict = {
                "vector": embeddings[i],
                "text": doc.page_content,
                # Add metadata fields. Assuming 'source' is the only one based on current code.
                # If there could be other metadata keys, iterate through doc.metadata
                **doc.metadata # Unpack metadata dictionary
            }
            data_to_add.append(doc_dict)
        # --- End Prepare data ---


        # Check if the table exists
        table_names = db.table_names()
        if COLLECTION_NAME in table_names:
            # If table exists, open it and add documents
            logging.info(f"Table '{COLLECTION_NAME}' already exists. Appending documents.")
            table = db.open_table(COLLECTION_NAME)

            # Get the existing schema to ensure data conforms
            existing_schema = table.schema
            # Identify field names in the existing schema (excluding 'text' and 'vector' which are standard)
            # This assumes 'text' and 'vector' are the only standard fields added by LanceDB.from_documents
            # or explicitly defined in the initial creation.
            schema_field_names = {field.name for field in existing_schema}
            # We only care about metadata fields that are actually columns in the table
            # Filter data_to_add to match the existing schema
            filtered_data_to_add = []
            for doc_dict in data_to_add:
                filtered_doc_dict = {}
                for key, value in doc_dict.items():
                    if key in schema_field_names:
                         filtered_doc_dict[key] = value
                    else:
                        # Log a warning if a field is being dropped because it's not in the schema
                        logging.warning(f"Field '{key}' from document is not in the existing table schema ('{COLLECTION_NAME}') and will be dropped.")
                filtered_data_to_add.append(filtered_doc_dict)


            # Add the prepared data to the table
            logging.info(f"Adding {len(filtered_data_to_add)} documents to LanceDB table '{COLLECTION_NAME}'...")
            # Use the filtered data
            table.add(filtered_data_to_add)
            logging.info("Successfully added documents to LanceDB.")

        else:
            # If table does not exist, create it using the lancedb client with an explicit schema
            logging.info(f"Table '{COLLECTION_NAME}' not found. Creating table with explicit schema.")

            # Define the explicit schema
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), list_size=VECTOR_SIZE)),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()) # Explicitly include the 'source' field
            ])

            # Create the table using the lancedb client with the explicit schema
            table = db.create_table(COLLECTION_NAME, schema=schema)
            logging.info(f"Successfully created empty table '{COLLECTION_NAME}' with explicit schema.")

            # Now add the prepared data to the newly created table
            logging.info(f"Adding {len(data_to_add)} initial documents to LanceDB table '{COLLECTION_NAME}'...")
            # Use the original data_to_add which includes 'source'
            table.add(data_to_add)
            logging.info("Successfully added initial documents to LanceDB.")


        # Update the PDF ingestion log file with newly processed files
        if successfully_processed_pdf_filenames:
            # Log file path is relative to the DB path
            log_file_path = os.path.join(LANCEDB_DB_PATH, ".ingested_files.log")
            try:
                # Use 'a' mode to append to the log file.
                with open(log_file_path, "a") as f:
                    for fname in successfully_processed_pdf_filenames:
                        f.write(f"{fname}\n")
                logging.info(f"Updated ingestion log with {len(successfully_processed_pdf_filenames)} new PDF file(s).")
            except Exception as e:
                logging.error(f"Error writing to ingestion log file: {e}")


    except Exception as e:
        logging.error(f"An error occurred during LanceDB ingestion: {e}", exc_info=True)
        exit(1)

    logging.info("Ingestion script finished.")
