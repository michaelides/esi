#!/usr/bin/env python3
"""
Simple streaming endpoint implementation that works reliably.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import json
import asyncio
import re
from agent import create_agent, get_captured_figures, clear_captured_figures
from config import settings

async def simple_streaming_generator(messages_data, options_data, file_content=None, artifacts=None):
    """Simple pseudo-streaming generator that breaks responses into chunks."""
    
    if artifacts:
        yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifacts})}\n\n"
    
    model = options_data.get("model") or "gemini-2.5-flash"
    temperature = options_data.get("temperature", 0.5)
    verbosity = options_data.get("verbosity", 3)
    debug = options_data.get("debug", False)
    
    try:
        # Create agent
        agent = create_agent(
            temperature=temperature,
            model=model,
            verbosity=verbosity,
            debug=debug,
            file_content=file_content,
            dataframe=None
        )
        
        # Build messages
        if messages_data:
            chat_history = messages_data[:-1]
            input_text = messages_data[-1].get("content", "")
        else:
            chat_history = []
            input_text = ""
        
        messages = []
        messages.append({"role": "system", "content": agent.system_prompt})
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": input_text})
        
        if file_content:
            messages[-1]["content"] = f"{input_text}\n\nFile content:\n{file_content}"
        
        payload = {"messages": messages}
        
        # Emit a status message
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...'})}\n\n"
        
        # Execute agent
        clear_captured_figures()
        result = await asyncio.to_thread(agent.invoke, payload)
        
        # Extract response text
        def extract_text(r):
            if hasattr(r, "messages") and isinstance(r.messages, list):
                for m in reversed(r.messages):
                    if hasattr(m, 'content') and isinstance(m.content, str) and m.content.strip():
                        return m.content.strip()
            
            if hasattr(r, 'content') and isinstance(r.content, str):
                return r.content.strip()
            
            if isinstance(r, dict):
                for k in ("output", "content", "text"):
                    v = r.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            
            return str(r)
        
        response_text = extract_text(result)
        
        if response_text:
            # Split into chunks for pseudo-streaming
            words = response_text.split()
            chunk_size = max(1, len(words) // 20)  # Break into ~20 chunks
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk.strip():
                    yield f"data: {json.dumps({'type': 'delta', 'text': chunk + ' '})}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for streaming effect
        
        # Handle artifacts
        captured_figures = get_captured_figures()
        if captured_figures:
            plot_artifacts = []
            for fig_json in captured_figures:
                try:
                    plot_artifacts.append({"type": "plot", "content": json.loads(fig_json)})
                except json.JSONDecodeError:
                    pass
            if plot_artifacts:
                yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': plot_artifacts})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"