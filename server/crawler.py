import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from langchain.tools import BaseTool
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
import json

class Crawl4AIInput(BaseModel):
    """Input for the Crawl4AI scraper tool."""
    url: str = Field(description="The URL to scrape")
    css_selector: Optional[str] = Field(default=None, description="CSS selector to target specific elements")
    extraction_strategy: Optional[str] = Field(default="text", description="Extraction strategy: 'text', 'markdown', 'structured'")
    word_count_threshold: Optional[int] = Field(default=10, description="Minimum word count for content blocks")
    only_text: Optional[bool] = Field(default=True, description="Extract only text content")

class SimpleCrawl4AITool(BaseTool):
    """Simple web scraper using Crawl4AI."""
    
    name: str = "crawl4ai_scraper"
    description: str = """
    Scrapes web pages using Crawl4AI. Much more reliable than basic scrapers.
    Can extract text, markdown, or structured data from web pages.
    Handles JavaScript-heavy sites automatically.
    """
    args_schema: Type[BaseModel] = Crawl4AIInput
    
    def _run(self, url: str, css_selector: Optional[str] = None, 
             extraction_strategy: str = "text", word_count_threshold: int = 10,
             only_text: bool = True) -> str:
        """Execute web scraping with Crawl4AI."""
        return asyncio.run(self._arun(url, css_selector, extraction_strategy, word_count_threshold, only_text))
    
    async def _arun(self, url: str, css_selector: Optional[str] = None,
                   extraction_strategy: str = "text", word_count_threshold: int = 10,
                   only_text: bool = True) -> str:
        """Async web scraping with Crawl4AI."""
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(
                    url=url,
                    css_selector=css_selector,
                    word_count_threshold=word_count_threshold,
                    only_text=only_text
                )
                
                if result.success:
                    if extraction_strategy == "markdown":
                        return result.markdown
                    elif extraction_strategy == "structured":
                        return f"Title: {result.title}\n\nContent: {result.cleaned_html[:2000]}..."
                    else:
                        return result.cleaned_html
                else:
                    return f"Failed to scrape {url}: {result.error_message}"
                    
        except Exception as e:
            return f"Error scraping with Crawl4AI: {str(e)}"

class AdvancedCrawl4AITool(BaseTool):
    """Advanced Crawl4AI tool with LLM extraction."""
    
    name: str = "advanced_crawl4ai"
    description: str = """
    Advanced web scraper using Crawl4AI with LLM-powered extraction.
    Can extract specific information using natural language instructions.
    Perfect for structured data extraction from web pages.
    """
    args_schema: Type[BaseModel] = Crawl4AIInput
    
    def _run(self, url: str, css_selector: Optional[str] = None,
             extraction_strategy: str = "text", word_count_threshold: int = 10,
             only_text: bool = True) -> str:
        return asyncio.run(self._arun(url, css_selector, extraction_strategy, word_count_threshold, only_text))
    
    async def _arun(self, url: str, css_selector: Optional[str] = None,
                   extraction_strategy: str = "text", word_count_threshold: int = 10,
                   only_text: bool = True) -> str:
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                # Basic extraction
                result = await crawler.arun(
                    url=url,
                    css_selector=css_selector,
                    word_count_threshold=word_count_threshold,
                    only_text=only_text
                )
                
                if result.success:
                    return {
                        "url": url,
                        "title": result.title,
                        "content": result.cleaned_html[:3000],
                        "markdown": result.markdown[:3000] if result.markdown else None,
                        "links": result.links,
                        "media": result.media
                    }
                else:
                    return f"Failed to scrape {url}: {result.error_message}"
                    
        except Exception as e:
            return f"Error in advanced scraping: {str(e)}"

class SmartExtractionTool(BaseTool):
    """Smart extraction tool using Crawl4AI with LLM strategies."""
    
    name: str = "smart_extraction"
    description: str = """
    Intelligent web scraper that can extract specific information using AI.
    Provide extraction instructions and it will find relevant data automatically.
    """
    
    class SmartExtractionInput(BaseModel):
        url: str = Field(description="URL to scrape")
        extraction_prompt: str = Field(description="What specific information to extract")
        
    args_schema: Type[BaseModel] = SmartExtractionInput
    
    def _run(self, url: str, extraction_prompt: str) -> str:
        return asyncio.run(self._arun(url, extraction_prompt))
    
    async def _arun(self, url: str, extraction_prompt: str) -> str:
        try:
            # Create extraction strategy
            extraction_strategy = LLMExtractionStrategy(
                provider="openai/gpt-4",
                api_token="your-openai-key",  # Replace with actual key
                instruction=extraction_prompt
            )
            
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=extraction_strategy
                )
                
                if result.success:
                    return result.extracted_content
                else:
                    return f"Failed to extract from {url}: {result.error_message}"
                    
        except Exception as e:
            return f"Error in smart extraction: {str(e)}"

# Batch processing tool
class BatchCrawl4AITool(BaseTool):
    """Batch web scraper for multiple URLs."""
    
    name: str = "batch_crawl4ai"
    description: str = """
    Scrapes multiple URLs efficiently using Crawl4AI.
    Returns structured data from all URLs.
    """
    
    class BatchInput(BaseModel):
        urls: list[str] = Field(description="List of URLs to scrape")
        max_concurrent: int = Field(default=3, description="Maximum concurrent requests")
        
    args_schema: Type[BaseModel] = BatchInput
    
    def _run(self, urls: list[str], max_concurrent: int = 3) -> str:
        return asyncio.run(self._arun(urls, max_concurrent))
    
    async def _arun(self, urls: list[str], max_concurrent: int = 3) -> str:
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                # Create semaphore for concurrency control
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def scrape_url(url):
                    async with semaphore:
                        result = await crawler.arun(url=url)
                        return {
                            "url": url,
                            "success": result.success,
                            "title": result.title if result.success else None,
                            "content": result.cleaned_html[:1000] if result.success else None,
                            "error": result.error_message if not result.success else None
                        }
                
                # Process all URLs
                tasks = [scrape_url(url) for url in urls]
                results = await asyncio.gather(*tasks)
                
                return json.dumps(results, indent=2)
                
        except Exception as e:
            return f"Error in batch scraping: {str(e)}"

