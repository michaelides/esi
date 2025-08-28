# Complete RAG Setup Guide

This guide will walk you through setting up the RAG (Retrieval-Augmented Generation) system for your ESI agent.

## Prerequisites

- Python 3.8+
- Supabase account with a project
- OpenAI API key

## Step 1: Environment Setup

### 1.1 Install Dependencies
```bash
cd server
pip install -r requirements.txt
```

### 1.2 Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual credentials
```

Your `.env` file should contain:
```bash
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_API=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
```

## Step 2: Database Setup

### Option A: Manual Setup (Recommended)

1. **Go to your Supabase Dashboard**
   - Navigate to [supabase.com/dashboard](https://supabase.com/dashboard)
   - Select your project

2. **Open SQL Editor**
   - Click **"SQL Editor"** in the left sidebar
   - Click **"New query"**

3. **Run the Setup Script**
   - Copy the entire contents of `server/manual_setup.sql`
   - Paste it into the SQL Editor
   - Click **"Run"** or press `Ctrl+Enter`

4. **Verify Setup**
   - You should see "RAG database setup completed successfully!"

### Option B: Automated Setup (Requires Service Role Key)

1. **Get Service Role Key**
   - In Supabase Dashboard, go to **Settings** > **API**
   - Copy the **service_role** key (NOT the anon key)

2. **Update Environment**
   ```bash
   # Temporarily replace the anon key with service role key
   VITE_SUPABASE_API=your_service_role_key_here
   ```

3. **Run Setup Script**
   ```bash
   python server/setup_rag.py
   ```

4. **Restore Anon Key**
   - After setup, change back to your anon key for regular operations

## Step 3: Test the Integration

### 3.1 Basic Test
```bash
python server/test_rag.py
```

Expected output:
```
✓ Database connection successful
✓ Document storage working
✓ Document search working
✓ Document retrieval working
All tests passed! RAG system is ready.
```

### 3.2 Example Usage
```bash
python server/rag_examples.py
```

## Step 4: Using RAG in Your Agent

The agent now has three new tools available:

### 4.1 Search Documents
```python
# The agent can search through ingested documents
agent.invoke({
    "messages": [{"role": "user", "content": "Find documents about machine learning"}]
})
```

### 4.2 Store Documents
```python
# Store new documents for future retrieval
agent.invoke({
    "messages": [{"role": "user", "content": "Store this: Machine learning is a subset of AI..."}]
})
```

### 4.3 Get Document Info
```python
# Retrieve specific documents by ID
agent.invoke({
    "messages": [{"role": "user", "content": "Get document info for doc_123"}]
})
```

## Step 5: Advanced Usage

### 5.1 Ingesting Files
```python
from server.ingestion import ingest_file, ingest_text

# Ingest a text file
await ingest_file("path/to/document.txt", source_type="text")

# Ingest web content
await ingest_text("Web content here", source_type="web", source_url="https://example.com")
```

### 5.2 Batch Operations
```python
# Ingest multiple documents
documents = [
    {"content": "Doc 1", "metadata": {"type": "guide"}},
    {"content": "Doc 2", "metadata": {"type": "reference"}}
]
for doc in documents:
    await ingest_text(doc["content"], metadata=doc["metadata"])
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
- **Error**: "Connection test failed"
- **Solution**: Check your `.env` file for correct Supabase credentials

#### 2. Permission Denied
- **Error**: "JWT expired" or "permission denied"
- **Solution**: Ensure you're using the correct API key type (anon for queries, service role for setup)

#### 3. Vector Extension Not Found
- **Error**: "type 'vector' does not exist"
- **Solution**: Re-run the database setup to ensure pgvector extension is enabled

#### 4. OpenAI API Errors
- **Error**: "Invalid API key" or rate limiting
- **Solution**: Verify your OpenAI API key and check your usage limits

### Debug Commands
```bash
# Test database connection
python -c "from server.rag import test_connection; import asyncio; asyncio.run(test_connection())"

# Check table structure
python -c "from server.rag import get_document_info; import asyncio; print(asyncio.run(get_document_info('test')))"
```

## Security Notes

- **Never commit `.env` files** to version control
- **Use anon key** for regular operations (not service role key)
- **Keep service role key secure** - only use for initial setup
- **Rotate API keys** regularly

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all environment variables are set correctly
3. Ensure your Supabase project has pgvector extension enabled
4. Check the server logs for detailed error messages

## Next Steps

Once setup is complete, you can:
1. Start ingesting your documents
2. Test the search functionality
3. Integrate with your existing agent workflows
4. Monitor usage and optimize as needed