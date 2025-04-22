import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
# Removed AsyncHtmlLoader import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from crawl4ai import AsyncWebCrawler # Import crawl4ai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration (Mimicking app.py based on summaries) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY environment variable not set.")
    exit(1)

CHROMA_DB_PATH = "./chroma_db_dissertation"
COLLECTION_NAME = "dissertation_resources" # Assuming the same collection name as in app.py

# List of URLs to scrape
URLS_TO_SCRAPE = [
    "https://www.uea.ac.uk/course/postgraduate/msc-management", # Example URL 1
    "https://www.uea.ac.uk/about/university-information/campus-map", # Example URL 2
    "https://www.uea.ac.uk/course/postgraduate/msc-organisational-psychology", # Added comma
    "https://scholar.google.co.uk/citations?user=v9Rzv3kAAAAJ&hl=en", # Added comma
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
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    logging.info(f"Connected to Chroma DB at {CHROMA_DB_PATH} with collection '{COLLECTION_NAME}'")
except Exception as e:
    logging.error(f"Failed to connect to Chroma DB: {e}")
    exit(1)

# Initialize crawl4ai with Markdown output and PDF extraction
crawler = AsyncWebCrawler(output_format="markdown", extract_pdf=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Keep chunk size reasonable for Markdown

# --- Core Logic ---
async def scrape_and_store():
    """Scrapes URLs using crawl4ai, splits content, and adds to Chroma DB."""
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

                if crawl_result_container and crawl_result_container.results:
                    logging.info(f"Successfully crawled content from: {url}. Found {len(crawl_result_container.results)} result(s).")
                    url_processed = False
                    for result in crawl_result_container.results:
                        # Extract markdown content if available
                        page_content = result.markdown # Access the markdown attribute
                        source_url = result.url # Get the specific source URL (could be original or PDF link)

                        if page_content:
                            # Create a Document object for each result (HTML page or PDF)
                            doc = Document(page_content=page_content, metadata={"source": source_url})

                            # Split the document content
                            splits = text_splitter.split_documents([doc]) # Pass as list
                            all_splits.extend(splits)
                            url_processed = True # Mark the original URL as processed if any content was found
                        else:
                            logging.warning(f"No markdown content found in result for source: {source_url} (original URL: {url})")

                    if url_processed:
                        processed_urls += 1
                    else:
                        logging.warning(f"No processable content (markdown) found for URL: {url}")
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

        # Add documents to Chroma DB
        logging.info("Adding documents to Chroma DB...")
        vector_store.add_documents(all_splits)
        logging.info("Successfully added documents to Chroma DB.")

        # Optional: Persist changes explicitly if needed (Chroma usually auto-persists)
        # vector_store.persist()
        # logging.info("Chroma DB changes persisted.")

    except Exception as e:
        logging.error(f"An error occurred during scraping or storing: {e}", exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    if not URLS_TO_SCRAPE:
        logging.warning("No URLs provided in URLS_TO_SCRAPE list. Exiting.")
    else:
        # Run the asynchronous function
        asyncio.run(scrape_and_store())
        logging.info("Script finished.")
