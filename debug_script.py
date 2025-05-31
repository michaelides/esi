#!/usr/bin/env python3
"""
Debug script to test Gemini configuration independently
Run this to isolate the Gemini setup issue
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test environment variables"""
    print("=== Environment Variables ===")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"GOOGLE_API_KEY: {'SET' if google_key else 'NOT SET'}")
    print(f"OPENAI_API_KEY: {'SET' if openai_key else 'NOT SET'}")
    
    if google_key:
        print(f"GOOGLE_API_KEY length: {len(google_key)}")
        print(f"GOOGLE_API_KEY starts with: {google_key[:10]}...")
    
    return google_key is not None

def test_imports():
    """Test importing required modules"""
    print("\n=== Testing Imports ===")
    
    try:
        from llama_index.core import Settings
        print("✓ llama_index.core imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import llama_index.core: {e}")
        return False
    
    try:
        from llama_index.llms.gemini import Gemini
        print("✓ llama_index.llms.gemini imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Gemini: {e}")
        return False
    
    try:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        print("✓ GoogleGenAIEmbedding imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GoogleGenAIEmbedding: {e}")
        return False
    
    return True

def test_gemini_initialization():
    """Test Gemini initialization"""
    print("\n=== Testing Gemini Initialization ===")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("✗ GOOGLE_API_KEY not found")
        return False
    
    # Clear any existing OpenAI keys that might interfere
    openai_backup = None
    if 'OPENAI_API_KEY' in os.environ:
        openai_backup = os.environ.pop('OPENAI_API_KEY')
        print("Temporarily removed OPENAI_API_KEY from environment")
    
    try:
        from llama_index.llms.gemini import Gemini
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        from llama_index.core import Settings
        
        # Clear settings
        Settings.llm = None
        Settings.embed_model = None
        print("Cleared existing Settings")
        
        # Test embedding model
        print("Initializing GoogleGenAI embedding...")
        embed_model = GoogleGenAIEmbedding(
            model_name="models/text-embedding-004",
            api_key=google_api_key
        )
        print("✓ GoogleGenAI embedding initialized successfully")
        
        # Test LLM
        print("Initializing Gemini LLM...")
        llm = Gemini(
            model_name="models/gemini-2.0-flash-exp",
            api_key=google_api_key,
            temperature=0.7
        )
        print("✓ Gemini LLM initialized successfully")
        
        # Test a simple completion
        print("Testing LLM completion...")
        response = llm.complete("Say 'Hello, World!' in a friendly way.")
        print(f"✓ LLM response: {response.text[:100]}...")
        
        # Set in Settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        print("✓ Settings configured successfully")
        
        # Restore OpenAI key if it existed
        if openai_backup:
            os.environ['OPENAI_API_KEY'] = openai_backup
            print("Restored OPENAI_API_KEY to environment")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during Gemini initialization: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Restore OpenAI key if it existed
        if openai_backup:
            os.environ['OPENAI_API_KEY'] = openai_backup
            print("Restored OPENAI_API_KEY to environment")
        
        return False

def main():
    """Run all tests"""
    print("Gemini Configuration Debug Script")
    print("=" * 40)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment test failed - GOOGLE_API_KEY not found")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        return
    
    # Test Gemini initialization
    if not test_gemini_initialization():
        print("\n❌ Gemini initialization failed")
        return
    
    print("\n✅ All tests passed! Gemini should work correctly.")

if __name__ == "__main__":
    main()
