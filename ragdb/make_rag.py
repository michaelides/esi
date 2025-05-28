import os
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

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Configuration ---
# Hugging Face dataset configuration for SimpleVectorStore persistence
HF_DATASET_ID = "gm42/esi_simplevector"
# Subdirectory within the HF dataset where the vector store files will be saved
HF_VECTOR_STORE_SUBDIR = "vector_store_data" 

# Ensure HF_TOKEN is set for writing to Hugging Face Hub
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN environment variable is not set. Make sure you are logged in via `huggingface-cli login` or have set HF_TOKEN to write to the Hugging Face Dataset.")

print(f"Target Hugging Face Dataset for RAG persistence: {HF_DATASET_ID}/{HF_VECTOR_STORE_SUBDIR}")

CHUNK_SIZE = 512 # Adjusted chunk size, common for LlamaIndex
CHUNK_OVERLAP = 20 # Adjusted chunk overlap
# Define the directory containing source documents for the RAG database relative to PROJECT_ROOT
SOURCE_DATA_DIR_RELATIVE = os.path.join("ragdb", "source_data")
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, SOURCE_DATA_DIR_RELATIVE)
# Define the directory for scraped markdown relative to PROJECT_ROOT
WEB_MARKDOWN_PATH_RELATIVE = os.path.join("ragdb", "web_markdown")
WEB_MARKDOWN_PATH = os.path.join(PROJECT_ROOT, WEB_MARKDOWN_PATH_RELATIVE)

# --- Add URLs to scrape ---
WEBPAGES_FILE_RELATIVE = os.path.join('ragdb', 'webpages.txt')
WEBPAGES_FILE = os.path.join(PROJECT_ROOT, WEBPAGES_FILE_RELATIVE)
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


def url_to_filename(url: str, max_length: int = 200) -> str:
    """Converts a URL to a safe filename for storing markdown, truncating if necessary."""
    parsed_url = urlparse(url)
    filename_base = "_".join(filter(None, [parsed_url.netloc, parsed_url.path.strip('/')]))

    # Replace problematic characters with underscore
    filename_base = "".join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in filename_base)
    
    # Remove consecutive underscores
    while "__" in filename_base:
        filename_base = filename_base.replace("__", "_")

    # Truncate if too long (reserve space for ".md")
    if len(filename_base) > max_length - 3:
        filename_base = filename_base[:max_length - 3]

    # Remove any leading/trailing underscores
    filename_base = filename_base.strip('_')

    if not filename_base:
        filename_base = parsed_url.netloc.replace('.', '_').replace(':', '_') if parsed_url.netloc else "scraped_page"
        if len(filename_base) > max_length -3:
            filename_base = filename_base[:max_length -3]
        filename_base = filename_base.strip('_')

    return f"{filename_base}.md"


async def scrape_websites(urls, output_dir):
    """Scrapes a list of URLs and saves the content as markdown files."""
    print("Running web scraping...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for scraped markdown: {output_dir}")

    if not urls:
        print("No URLs provided for scraping.")
        return

    print(f"Initializing crawler for deep scraping of {len(urls)} seed URLs...")
    crawler = AsyncWebCrawler(should_markdown=True)

    print("Starting deep web scraping...")
    saved_count = 0
    failed_urls_details = []
    MAX_DEPTH = 1
    MAX_PAGES_PER_SEED = 10
    INCLUDE_EXTERNAL = False

    try:
        for start_url in urls:
            print(f"  Initiating deep crawl from: {start_url} (max_depth={MAX_DEPTH}, max_pages_per_seed={MAX_PAGES_PER_SEED})")
            
            config = CrawlerRunConfig(
                deep_crawl_strategy=BFSDeepCrawlStrategy(
                    max_depth=MAX_DEPTH,
                    include_external=INCLUDE_EXTERNAL,
                    max_pages=MAX_PAGES_PER_SEED
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                stream=True,
                verbose=True
            )

            try:
                async for result in await crawler.arun(url=start_url, config=config):
                    current_page_url = result.url
                    print(f"    Processing page from deep crawl: {current_page_url} (Origin: {start_url}, Depth: {result.metadata.get('depth', 'N/A')})")

                    try:
                        if result.success and result.markdown:
                            filename = url_to_filename(current_page_url)
                            filepath = os.path.join(output_dir, filename)
                            try:
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(result.markdown)
                                print(f"      Successfully scraped and saved: {current_page_url} to {filepath}")
                                saved_count += 1
                            except IOError as e:
                                print(f"      Error writing file {filepath} for URL {current_page_url}: {e}")
                                failed_urls_details.append({"url": current_page_url, "error": f"File write error: {e}"})
                        elif result.success and not result.markdown:
                            print(f"      Successfully processed URL: {current_page_url}, but no markdown content was returned.")
                        else:
                            error_message = result.error or "Unknown error during deep crawl of page"
                            print(f"      Failed to scrape page {current_page_url} during deep crawl: {error_message}")
                            failed_urls_details.append({"url": current_page_url, "error": error_message})
                    except Exception as page_processing_e:
                        print(f"      Error processing result for {current_page_url}: {page_processing_e}")
                        failed_urls_details.append({"url": current_page_url, "error": f"Result processing error: {page_processing_e}"})
            
            except Exception as e:
                print(f"    An error occurred while initiating or performing deep crawl for start URL {start_url}: {e}")
                failed_urls_details.append({"url": start_url, "error": f"Deep crawl initiation/execution error: {str(e)}"})
    finally:
        if 'crawler' in locals() and hasattr(crawler, 'close'):
            print("Closing crawler session...")
            await crawler.close()

    print(f"Deep scraping finished. Successfully saved {saved_count} pages from {len(urls)} seed URLs.")
    if failed_urls_details:
        print("Details for failed URLs:")
        for detail in failed_urls_details:
            print(f"  URL: {detail['url']}, Error: {detail['error']}")

    print("Finished web scraping.")

# --- Main Script ---
async def main():
    # --- Initialize Embedding Model ---
    embedding_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

    print(f"Configuring RAG to persist to Hugging Face Dataset: {HF_DATASET_ID}, path in repo: {HF_VECTOR_STORE_SUBDIR}")

    # 1. Scrape websites
    await scrape_websites(URLS_TO_SCRAPE, WEB_MARKDOWN_PATH)

    # 2. Load documents using SimpleDirectoryReader
    print(f"Loading documents from {SOURCE_DATA_DIR_RELATIVE} and {WEB_MARKDOWN_PATH_RELATIVE}...")
    os.makedirs(SOURCE_DATA_DIR, exist_ok=True)
    os.makedirs(WEB_MARKDOWN_PATH, exist_ok=True)

    input_dirs = [SOURCE_DATA_DIR, WEB_MARKDOWN_PATH]
    required_exts = [".pdf", ".txt", ".md", ".csv"]

    all_documents = []

    def get_relative_path_metadata(filename: str) -> dict:
        """Generates metadata including the file path relative to PROJECT_ROOT."""
        try:
            relative_path = os.path.relpath(filename, start=PROJECT_ROOT)
            return {"file_path": relative_path}
        except ValueError:
            print(f"Warning: File {filename} is outside project root. Storing absolute path.")
            return {"file_path": filename}
        except Exception as e:
            print(f"Error generating relative path for {filename}: {e}. Storing absolute path.")
            return {"file_path": filename}


    for input_dir in input_dirs:
        if not os.path.exists(input_dir) or not os.listdir(input_dir):
            print(f"Directory is empty or does not exist, skipping: {input_dir}")
            continue
        try:
            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                required_exts=required_exts,
                recursive=True,
                file_metadata=get_relative_path_metadata
            )
            docs = reader.load_data(show_progress=True)
            if docs:
                print(f"Loaded {len(docs)} documents from {input_dir}")
                all_documents.extend(docs)
            else:
                 print(f"No documents with extensions {required_exts} found in {input_dir}")
        except ValueError as e:
            print(f"Warning: Error reading from directory {input_dir}: {e}")
        except Exception as e:
            print(f"Error loading documents from {input_dir}: {e}")


    print(f"Total documents loaded: {len(all_documents)}")

    if not all_documents:
        print("No documents loaded. Exiting.")
        return

    # 3. Initialize Text Splitter (Node Parser)
    print(f"Initializing node parser (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 4. Initialize SimpleVectorStore and Storage Context for local persistence
    print("Initializing SimpleVectorStore for local persistence...")
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

