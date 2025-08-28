"""
RAG (Retrieval-Augmented Generation) implementation for the ESI agent.
Provides document storage, retrieval, and semantic search capabilities.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

from supabase import create_client
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    """Manages RAG operations including document storage and retrieval."""
    
    def __init__(self):
        """Initialize RAG manager with Supabase and Google Gemini clients."""
        self.supabase_url = os.getenv('VITE_SUPABASE_URL')
        self.supabase_key = os.getenv('VITE_SUPABASE_API')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'models/text-embedding-004')
        
        if not all([self.supabase_url, self.supabase_key, self.gemini_api_key]):
            raise ValueError("Missing required environment variables. Please check VITE_SUPABASE_URL, VITE_SUPABASE_API, and GEMINI_API_KEY")
        
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        genai.configure(api_key=self.gemini_api_key)
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using Google Gemini."""
        try:
            # Use Google Gemini embeddings
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            
            # Gemini returns the embedding directly
            if 'embedding' in result:
                return result['embedding']
            else:
                logger.error(f"Unexpected Gemini response format: {result}")
                raise ValueError("Unexpected response format from Gemini API")
                
        except Exception as e:
            logger.error(f"Error creating embedding with Gemini: {e}")
            raise
    
    def _generate_document_hash(self, content: str, source_url: str = "") -> str:
        """Generate a unique hash for a document."""
        content_to_hash = f"{content}{source_url}"
        return hashlib.sha256(content_to_hash.encode()).hexdigest()
    
    async def store_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        source_type: str = "text",
        source_url: str = None,
        chunk_index: int = 0
    ) -> str:
        """Store a document in the database."""
        try:
            # Generate document hash
            document_hash = self._generate_document_hash(content, source_url or "")
            
            # Check if document already exists
            existing = self.supabase.table('documents').select('id').eq('document_hash', document_hash).execute()
            if existing.data:
                logger.info(f"Document already exists: {document_hash}")
                return existing.data[0]['id']
            
            # Create embedding
            embedding = await self.create_embedding(content)
            
            # Store document
            document_data = {
                'content': content,
                'metadata': metadata or {},
                'embedding': embedding,
                'source_type': source_type,
                'source_url': source_url,
                'document_hash': document_hash,
                'chunk_index': chunk_index
            }
            
            result = self.supabase.table('documents').insert(document_data).execute()
            
            if result.data:
                doc_id = result.data[0]['id']
                logger.info(f"Document stored successfully: {doc_id}")
                return doc_id
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise
    
    async def search_documents(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        source_type: str = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity."""
        try:
            # Create embedding for the query
            query_embedding = await self.create_embedding(query)
            
            # Build the SQL query
            sql_query = """
            SELECT id, content, metadata, source_type, source_url, created_at,
                   1 - (embedding <=> %s::vector) as similarity
            FROM documents
            WHERE 1 - (embedding <=> %s::vector) > %s
            """
            
            params = [str(query_embedding), str(query_embedding), threshold]
            
            if source_type:
                sql_query += " AND source_type = %s"
                params.append(source_type)
            
            sql_query += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([str(query_embedding), limit])
            
            # Use the built-in vector similarity search
            query_embedding_str = str(query_embedding)
            result = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding_str,
                    'match_threshold': threshold,
                    'match_count': limit
                }
            ).execute()
            
            if result.data:
                documents = []
                for row in result.data:
                    documents.append({
                        'id': row['id'],
                        'content': row['content'],
                        'metadata': row['metadata'] if row['metadata'] else {},
                        'source_type': row['source_type'],
                        'source_url': row['source_url'],
                        'created_at': row['created_at'],
                        'similarity': float(row['similarity'])
                    })
                return documents
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID."""
        try:
            result = self.supabase.table('documents').select('*').eq('id', document_id).execute()
            if result.data:
                doc = result.data[0]
                doc['metadata'] = doc['metadata'] if doc['metadata'] else {}
                return doc
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            result = self.supabase.table('documents').delete().eq('id', document_id).execute()
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    async def list_documents(
        self,
        source_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents with optional filtering."""
        try:
            query = self.supabase.table('documents').select('id, content, metadata, source_type, source_url, created_at')
            
            if source_type:
                query = query.eq('source_type', source_type)
            
            query = query.order('created_at', desc=True).range(offset, offset + limit - 1)
            
            result = query.execute()
            
            documents = []
            for doc in result.data:
                doc['metadata'] = doc['metadata'] if doc['metadata'] else {}
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise

# Global RAG manager instance
_rag_manager = None

async def get_rag_manager() -> RAGManager:
    """Get the global RAG manager instance."""
    global _rag_manager
    if _rag_manager is None:
        if os.getenv('GEMINI_API_KEY') and os.getenv('VITE_SUPABASE_URL') and os.getenv('VITE_SUPABASE_API'):
            try:
                _rag_manager = RAGManager()
            except Exception as e:
                logger.error(f"Failed to initialize RAGManager: {e}")
                class DummyRAGManager:
                    async def search_documents(self, query, limit=5):
                        raise ConnectionError(f"RAG is not configured due to initialization error: {e}")
                    async def store_document(self, content, metadata=None):
                        raise ConnectionError(f"RAG is not configured due to initialization error: {e}")
                    async def get_document_by_id(self, document_id):
                        raise ConnectionError(f"RAG is not configured due to initialization error: {e}")
                _rag_manager = DummyRAGManager()
        else:
            class DummyRAGManager:
                async def search_documents(self, query, limit=5):
                    raise ConnectionError("RAG is not configured. Please provide GEMINI_API_KEY, VITE_SUPABASE_URL, and VITE_SUPABASE_API.")
                async def store_document(self, content, metadata=None):
                    raise ConnectionError("RAG is not configured. Please provide GEMINI_API_KEY, VITE_SUPABASE_URL, and VITE_SUPABASE_API.")
                async def get_document_by_id(self, document_id):
                    raise ConnectionError("RAG is not configured. Please provide GEMINI_API_KEY, VITE_SUPABASE_URL, and VITE_SUPABASE_API.")
            _rag_manager = DummyRAGManager()
    return _rag_manager

# Tool functions for agent integration
async def search_documents_tool(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Tool function to search documents."""
    rag = await get_rag_manager()
    return await rag.search_documents(query, limit=limit)

async def store_document_tool(content: str, metadata: Dict[str, Any] = None) -> str:
    """Tool function to store a document."""
    rag = await get_rag_manager()
    return await rag.store_document(content, metadata=metadata)

async def get_document_tool(document_id: str) -> Optional[Dict[str, Any]]:
    """Tool function to get a document by ID."""
    rag = await get_rag_manager()
    return await rag.get_document_by_id(document_id)