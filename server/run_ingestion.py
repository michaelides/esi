import asyncio
import os
import logging
from pathlib import Path
from ingestion import DocumentIngester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Orchestrates the ingestion of PDF files and web content."""
    ingester = DocumentIngester()
    
    # Track ingestion statistics
    stats = {
        "pdf_files_processed": 0,
        "pdf_files_success": 0,
        "pdf_files_failed": 0,
        "websites_processed": 0,
        "websites_success": 0,
        "websites_failed": 0,
        "pdf_files": [],
        "websites": []
    }

    # Ingest PDF files
    pdf_dir = "source_data/pdf"
    if os.path.exists(pdf_dir) and os.listdir(pdf_dir):
        logger.info(f"Ingesting PDF files from {pdf_dir}...")
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith(".pdf"):
                stats["pdf_files_processed"] += 1
                file_path = Path(pdf_dir) / filename
                try:
                    doc_id = await ingester.ingest_pdf(str(file_path))
                    logger.info(f"Successfully ingested PDF: {filename} with ID: {doc_id}")
                    stats["pdf_files_success"] += 1
                    stats["pdf_files"].append({"filename": filename, "id": doc_id})
                except Exception as e:
                    logger.error(f"Failed to ingest PDF {filename}: {e}")
                    stats["pdf_files_failed"] += 1
    else:
        logger.info(f"No PDF files found in {pdf_dir} or directory does not exist. Skipping PDF ingestion.")

    # Ingest websites from markdown file
    websites_file = "source_data/websites_to_ingest.md"
    web_markdown_output_dir = "source_data/web_markdown"
    if os.path.exists(websites_file):
        logger.info(f"Ingesting websites from {websites_file}...")
        try:
            ingested_ids = await ingester.ingest_websites_from_file(websites_file, web_markdown_output_dir)
            stats["websites_processed"] = len(ingested_ids)
            stats["websites_success"] = len(ingested_ids)
            logger.info(f"Successfully ingested {len(ingested_ids)} websites.")
            stats["websites"] = [{"url": "unknown", "id": doc_id} for doc_id in ingested_ids]
        except Exception as e:
            stats["websites_failed"] = stats["websites_processed"]
            logger.error(f"Failed to ingest websites from {websites_file}: {e}")
    else:
        logger.info(f"Websites file not found: {websites_file}. Skipping website ingestion.")
    
    # Print summary report
    logger.info("=" * 50)
    logger.info("INGESTION SUMMARY REPORT")
    logger.info("=" * 50)
    logger.info(f"PDF Files Processed: {stats['pdf_files_processed']}")
    logger.info(f"PDF Files Success: {stats['pdf_files_success']}")
    logger.info(f"PDF Files Failed: {stats['pdf_files_failed']}")
    logger.info(f"Websites Processed: {stats['websites_processed']}")
    logger.info(f"Websites Success: {stats['websites_success']}")
    logger.info(f"Websites Failed: {stats['websites_failed']}")
    
    if stats['pdf_files']:
        logger.info("\nPDF Files Ingested:")
        for pdf in stats['pdf_files']:
            logger.info(f"  - {pdf['filename']} (ID: {pdf['id']})")
    
    if stats['websites']:
        logger.info("\nWebsites Ingested:")
        for website in stats['websites']:
            logger.info(f"  - {website['url']} (ID: {website['id']})")
    
    logger.info("=" * 50)
    logger.info("Ingestion process completed successfully.")
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())