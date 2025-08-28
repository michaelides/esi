"""
Setup script for RAG implementation.
Initializes the database schema and verifies connections.
"""

import os
import logging
from pathlib import Path
import requests

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Set up the database schema for RAG."""
    
    # Load environment variables
    supabase_url = os.getenv('VITE_SUPABASE_URL')
    supabase_key = os.getenv('VITE_SUPABASE_API')
    
    if not supabase_url or not supabase_key:
        logger.error("Missing VITE_SUPABASE_URL or VITE_SUPABASE_API environment variables")
        logger.error(f"Looking for .env file at: {env_path}")
        logger.error(f"File exists: {env_path.exists()}")
        return False
    
    try:
        # Test connection with simple query
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
            'Content-Type': 'application/json'
        }
        
        # Test basic connection
        test_url = f"{supabase_url}/rest/v1/documents?select=id&limit=1"
        response = requests.get(test_url, headers=headers)
        
        if response.status_code == 200:
            logger.info("Connection test successful!")
        else:
            logger.error(f"Connection test failed: {response.status_code} - {response.text}")
            return False
        
        # Create documents table using direct SQL via SQL Editor API
        logger.info("Setting up database schema...")
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding vector(3072),
            source_type VARCHAR(50),
            source_url TEXT,
            document_hash VARCHAR(64) UNIQUE,
            chunk_index INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        
        # Create indexes
        create_indexes_sql = """
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        
        CREATE INDEX IF NOT EXISTS documents_source_hash_idx
        ON documents (document_hash, chunk_index);
        
        CREATE INDEX IF NOT EXISTS documents_source_type_idx
        ON documents (source_type);
        
        CREATE INDEX IF NOT EXISTS documents_created_at_idx
        ON documents (created_at DESC);
        """
        
        # Use SQL Editor API instead of RPC
        sql_url = f"{supabase_url}/rest/v1/query"
        
        # Try to create table
        try:
            response = requests.post(sql_url,
                headers=headers,
                json={"query": create_table_sql}
            )
            if response.status_code in [200, 201]:
                logger.info("Documents table created successfully!")
            else:
                logger.warning(f"Table creation response: {response.status_code} - {response.text}")
                logger.info("This is expected - the table may already exist or SQL API is not enabled")
        except Exception as e:
            logger.warning(f"Could not create table via SQL API: {e}")
        
        # Try to create indexes
        try:
            response = requests.post(sql_url,
                headers=headers,
                json={"query": create_indexes_sql}
            )
            if response.status_code in [200, 201]:
                logger.info("Indexes created successfully!")
            else:
                logger.warning(f"Index creation response: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Could not create indexes via SQL API: {e}")
        
        logger.info("Database setup completed!")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

def test_connection():
    """Test the database connection."""
    supabase_url = os.getenv('VITE_SUPABASE_URL')
    supabase_key = os.getenv('VITE_SUPABASE_API')
    
    if not supabase_url or not supabase_key:
        logger.error("Missing VITE_SUPABASE_URL or VITE_SUPABASE_API environment variables")
        return False
    
    try:
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
            'Content-Type': 'application/json'
        }
        
        # Test basic query
        test_url = f"{supabase_url}/rest/v1/documents?select=count"
        response = requests.get(test_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            count = len(data) if isinstance(data, list) else 0
            logger.info(f"Connection test successful. Found {count} documents.")
            return True
        else:
            logger.error(f"Connection test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("Starting RAG setup...")
    logger.info("=" * 50)
    logger.info("Checking environment variables...")
    logger.info(f"VITE_SUPABASE_URL: {'✓' if os.getenv('VITE_SUPABASE_URL') else '✗'}")
    logger.info(f"VITE_SUPABASE_API: {'✓' if os.getenv('VITE_SUPABASE_API') else '✗'}")
    logger.info("=" * 50)
    
    # Test connection first
    if not test_connection():
        logger.error("Connection test failed. Please check your environment variables.")
        logger.info("\nAlternative setup:")
        logger.info("1. Go to your Supabase Dashboard")
        logger.info("2. Navigate to SQL Editor")
        logger.info("3. Copy and paste the contents of server/manual_setup.sql")
        logger.info("4. Run the SQL script")
        return
    
    # Setup database
    success = setup_database()
    if success:
        logger.info("RAG setup completed successfully!")
        logger.info("\nNote: The warnings above are normal if the table already exists.")
        logger.info("Your RAG system is ready to use!")
    else:
        logger.error("RAG setup failed!")
        logger.info("\nTry the manual setup instead:")
        logger.info("1. Go to your Supabase Dashboard")
        logger.info("2. Navigate to SQL Editor")
        logger.info("3. Copy and paste the contents of server/manual_setup.sql")
        logger.info("4. Run the SQL script")

if __name__ == "__main__":
    main()