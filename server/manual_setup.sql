-- Manual setup script for RAG implementation
-- Run this in your Supabase SQL editor (Supabase Dashboard > SQL Editor)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    source_type VARCHAR(50),
    source_url TEXT,
    document_hash VARCHAR(64) UNIQUE,
    chunk_index INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_source_hash_idx 
ON documents (document_hash, chunk_index);

CREATE INDEX IF NOT EXISTS documents_source_type_idx 
ON documents (source_type);

CREATE INDEX IF NOT EXISTS documents_created_at_idx 
ON documents (created_at DESC);

-- Create a function to search similar documents
DROP FUNCTION IF EXISTS match_documents(vector, float, int);
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding vector(768),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    source_type varchar,
    source_url text,
    created_at TIMESTAMP WITH TIME ZONE,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        documents.source_type,
        documents.source_url,
        documents.created_at,
        1 - (documents.embedding <=> query_embedding) as similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Test the setup
SELECT 'RAG database setup completed successfully!' as status;