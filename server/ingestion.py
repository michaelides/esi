"""
Document ingestion utilities for RAG implementation.
Handles various document formats and web content ingestion.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import re
from urllib.parse import urlparse
import hashlib
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
import markdown
from markdownify import markdownify
from PyPDF2 import PdfReader

from rag import get_rag_manager
from crawler import SimpleCrawl4AITool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngester:
    """Handles ingestion of various document types into the RAG system."""
    
    def __init__(self):
        """Initialize the document ingester."""
        self.rag_manager = None
    
    async def _get_rag_manager(self):
        """Get or create RAG manager instance."""
        if self.rag_manager is None:
            self.rag_manager = await get_rag_manager()
        return self.rag_manager
    
    async def ingest_text(
        self,
        content: str,
        source_type: str = "text",
        source_url: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Ingest plain text content."""
        rag = await self._get_rag_manager()
        
        doc_metadata = metadata or {}
        doc_metadata['ingestion_method'] = 'text'
        
        return await rag.store_document(
            content=content,
            source_type=source_type,
            source_url=source_url,
            metadata=doc_metadata
        )
    
    async def ingest_markdown(
        self,
        content: str,
        source_url: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Ingest markdown content."""
        # Convert markdown to plain text for embedding
        html = markdown.markdown(content)
        text_content = BeautifulSoup(html, 'html.parser').get_text()
        
        doc_metadata = metadata or {}
        doc_metadata['original_format'] = 'markdown'
        doc_metadata['original_content'] = content
        
        return await self.ingest_text(
            content=text_content,
            source_type='markdown',
            source_url=source_url,
            metadata=doc_metadata
        )
    
    async def ingest_webpage(
        self,
        url: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Ingest content from a webpage."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                    
                    html_content = await response.text()
                    
                    # Parse HTML
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text_content = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_content = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Convert to markdown for better structure
                    markdown_content = markdownify(html_content)
                    
                    doc_metadata = metadata or {}
                    doc_metadata['original_format'] = 'html'
                    doc_metadata['markdown_content'] = markdown_content
                    doc_metadata['title'] = soup.title.string if soup.title else url
                    
                    return await self.ingest_text(
                        content=text_content,
                        source_type='webpage',
                        source_url=url,
                        metadata=doc_metadata
                    )
                    
        except Exception as e:
            logger.error(f"Error ingesting webpage {url}: {e}")
            raise
    
    def _split_text_into_chunks(
        self,
        text: str,
        max_chunk_size: int = 250,
        overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n']
                best_end = end
                
                for ending in sentence_endings:
                    pos = text.rfind(ending, start, end + 50)
                    if pos != -1 and pos > start:
                        best_end = min(best_end, pos + 1)
                
                end = best_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Avoid infinite loops
            if start >= len(text) or len(chunks) > 1000:
                break
        
        return chunks
    
    async def ingest_large_text(
        self,
        content: str,
        source_type: str = "text",
        source_url: str = None,
        metadata: Dict[str, Any] = None,
        max_chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """Ingest large text by splitting into chunks."""
        chunks = self._split_text_into_chunks(content, max_chunk_size, overlap)
        
        doc_ids = []
        base_metadata = metadata or {}
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            chunk_metadata['chunk_size'] = len(chunk)
            
            doc_id = await self.ingest_text(
                content=chunk,
                source_type=source_type,
                source_url=source_url,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        logger.info(f"Ingested {len(chunks)} chunks for document")
        return doc_ids
    
    async def ingest_json_documents(
        self,
        json_content: str,
        source_type: str = "json",
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """Ingest documents from JSON format."""
        try:
            data = json.loads(json_content)
            doc_ids = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        content = item.get('content', '')
                        item_metadata = {**metadata, **item} if metadata else item
                        if 'content' in item_metadata:
                            del item_metadata['content']
                        
                        doc_id = await self.ingest_text(
                            content=content,
                            source_type=source_type,
                            metadata=item_metadata
                        )
                        doc_ids.append(doc_id)
            elif isinstance(data, dict):
                content = data.get('content', str(data))
                doc_metadata = {**metadata, **data} if metadata else data
                if 'content' in doc_metadata:
                    del doc_metadata['content']
                
                doc_id = await self.ingest_text(
                    content=content,
                    source_type=source_type,
                    metadata=doc_metadata
                )
                doc_ids.append(doc_id)
            
            return doc_ids
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    async def ingest_file(
        self,
        file_path: str,
        source_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """Ingest content from a local file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if source_type is None:
            source_type = ext.lstrip('.')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Handle different file types
        if ext == '.md':
            return [await self.ingest_markdown(content, source_url=file_path, metadata=metadata)]
        elif ext == '.json':
            return await self.ingest_json_documents(content, source_type=source_type, metadata=metadata)
        elif ext == '.pdf':
            return [await self.ingest_pdf(file_path, source_url=file_path, metadata=metadata)]
        elif ext in ['.txt', '.py', '.js', '.ts', '.html', '.css']:
            return [await self.ingest_text(content, source_type=source_type, source_url=file_path, metadata=metadata)]
        else:
            # Treat as plain text for unknown formats
            return [await self.ingest_text(content, source_type=source_type, source_url=file_path, metadata=metadata)]

    async def ingest_pdf(
        self,
        file_path: str,
        source_url: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Ingest content from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            
            doc_metadata = metadata or {}
            doc_metadata['original_format'] = 'pdf'
            doc_metadata['file_path'] = file_path
            
            return await self.ingest_large_text(
                content=text_content,
                source_type='pdf',
                source_url=source_url or file_path,
                metadata=doc_metadata
            )
        except Exception as e:
            logger.error(f"Error ingesting PDF {file_path}: {e}")
            raise

    async def ingest_websites_from_file(
        self,
        file_path: str,
        output_dir: str = "source_data/web_markdown",
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Reads a list of URLs from a markdown file, scrapes them using crawl4ai,
        saves the content as markdown files, and then ingests them.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ingested_doc_ids = []
        crawl_tool = SimpleCrawl4AITool()

        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        for url in urls:
            try:
                logger.info(f"Scraping URL: {url}")
                # Use crawl4ai to scrape the content as markdown
                scraped_content = crawl_tool.run({"url": url, "extraction_strategy": "markdown"})
                
                # Generate a filename from the URL
                parsed_url = urlparse(url)
                filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', parsed_url.netloc + parsed_url.path)
                if not filename.endswith(".md"):
                    filename += ".md"
                output_file_path = Path(output_dir) / filename

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(scraped_content)
                logger.info(f"Saved scraped content to {output_file_path}")

                # Ingest the saved markdown file
                doc_id = await self.ingest_markdown(
                    content=scraped_content,
                    source_url=url,
                    metadata={**(metadata or {}), 'original_file': str(output_file_path)}
                )
                ingested_doc_ids.append(doc_id)
                logger.info(f"Ingested scraped content from {url} with ID: {doc_id}")

            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
        
        return ingested_doc_ids

# Convenience functions for direct use
async def ingest_text(content: str, **kwargs) -> str:
    """Ingest plain text content."""
    ingester = DocumentIngester()
    return await ingester.ingest_text(content, **kwargs)

async def ingest_webpage(url: str, **kwargs) -> str:
    """Ingest content from a webpage."""
    ingester = DocumentIngester()
    return await ingester.ingest_webpage(url, **kwargs)

async def ingest_file(file_path: str, **kwargs) -> List[str]:
    """Ingest content from a local file."""
    ingester = DocumentIngester()
    return await ingester.ingest_file(file_path, **kwargs)

async def ingest_large_text(content: str, **kwargs) -> List[str]:
    """Ingest large text by splitting into chunks."""
    ingester = DocumentIngester()
    return await ingester.ingest_large_text(content, **kwargs)