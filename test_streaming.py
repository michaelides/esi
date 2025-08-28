#!/usr/bin/env python3
"""
Simple test script to verify LLM streaming functionality.
"""

import asyncio
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import AsyncCallbackHandler
import json

load_dotenv()

class TestStreamingHandler(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Token: '{token}'")
        self.tokens.append(token)
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started")
        
    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended. Total tokens: {len(self.tokens)}")

async def test_llm_streaming():
    print("Testing LLM streaming...")
    
    handler = TestStreamingHandler()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        streaming=True,
        callbacks=[handler]
    )
    
    messages = [
        {"role": "user", "content": "Count from 1 to 5, explaining each number briefly."}
    ]
    
    print("Invoking LLM...")
    response = await llm.ainvoke(messages)
    print(f"Response: {response}")
    print(f"Tokens collected: {len(handler.tokens)}")
    
    return handler.tokens

if __name__ == "__main__":
    asyncio.run(test_llm_streaming())