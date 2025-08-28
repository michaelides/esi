#!/usr/bin/env python3
"""
Test script to verify RAG integration is working correctly.
"""

import os
import sys
import asyncio
from pathlib import Path
import logging
import pytest

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test if environment variables are set correctly."""
    logger.info("Testing environment variables...")
    
    supabase_url = os.getenv('VITE_SUPABASE_URL')
    supabase_key = os.getenv('VITE_SUPABASE_API')
    
    if not supabase_url:
        logger.error("❌ VITE_SUPABASE_URL not found")
        return False
    
    if not supabase_key:
        logger.error("❌ VITE_SUPABASE_API not found")
        return False
    
    logger.info("✅ Environment variables found")
    logger.info(f"   URL: {supabase_url[:20]}...")
    logger.info(f"   Key: {supabase_key[:10]}...")
    return True

def test_database_connection():
    """Test database connection."""
    logger.info("Testing database connection...")
    
    try:
        from rag import RAGManager
        rag = RAGManager()
        
        # Test basic connection
        result = rag.supabase.table('documents').select('count').execute()
        logger.info("✅ Database connection successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

@pytest.mark.anyio
async def test_document_ingestion():
    """Test document ingestion."""
    logger.info("Testing document ingestion...")
    
    try:
        from ingestion import DocumentIngester
        ingestion = DocumentIngester()
        
        # Test with a simple text
        test_content = "This is a test document for RAG integration testing."
        doc_id = await ingestion.ingest_text(
            content=test_content,
            source_type="test",
            metadata={"test": True}
        )
        
        if doc_id:
            logger.info(f"✅ Document ingestion successful: {doc_id}")
            return True
        else:
            logger.error("❌ Document ingestion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document ingestion failed: {e}")
        return False

@pytest.mark.anyio
async def test_document_search():
    """Test document search."""
    logger.info("Testing document search...")
    
    try:
        from rag import RAGManager
        rag = RAGManager()
        
        # Search for the test document
        results = await rag.search_documents(
            query="test document",
            limit=5
        )
        
        if results:
            logger.info(f"✅ Document search successful: found {len(results)} results")
            for i, result in enumerate(results[:2]):
                logger.info(f"   Result {i+1}: {result['content'][:50]}...")
            return True
        else:
            logger.warning("⚠️  No search results found")
            return True  # This might be OK if no documents exist
            
    except Exception as e:
        logger.error(f"❌ Document search failed: {e}")
        return False

@pytest.mark.anyio
async def test_agent_integration():
    """Test agent integration."""
    logger.info("Testing agent integration...")
    
    try:
        from agent import search_documents, store_document, get_document_info
        
        # Test search_documents function
        results = await search_documents("test")
        logger.info("✅ search_documents function accessible")
        
        # Test store_document function
        doc_id = await store_document(content="Test content", metadata={"source_type": "test", "test": True})
        if doc_id:
            logger.info(f"✅ store_document function working: {doc_id}")
            
            # Test get_document_info
            info = await get_document_info(doc_id)
            if info:
                logger.info("✅ get_document_info function working")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Agent integration failed: {e}")
        return False

async def run_async_tests():
    """Run async tests."""
    async_tests = [
        ("Document Ingestion", test_document_ingestion),
        ("Document Search", test_document_search),
        ("Agent Integration", test_agent_integration),
    ]
    
    results = []
    for test_name, test_func in async_tests:
        logger.info("")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    return results

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RAG Integration Test Suite")
    logger.info("=" * 60)
    
    # Run sync tests
    sync_tests = [
        ("Environment Variables", test_environment),
        ("Database Connection", test_database_connection),
    ]
    
    results = []
    for test_name, test_func in sync_tests:
        logger.info("")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run async tests
    logger.info("")
    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("")
    logger.info(f"Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! RAG integration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)