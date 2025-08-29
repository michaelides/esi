import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google.auth.exceptions import DefaultCredentialsError
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
import asyncio
import magic
import pandas as pd
import pyreadstat
import pyreadr
from pypdf import PdfReader
import io
import os
import json
import re
import traceback
from pathlib import Path
from langchain_core.runnables import RunnableConfig


from .agent import create_agent, get_captured_figures, clear_captured_figures, is_mistral_model, MISTRAL_MODEL_MAPPING, load_system_prompt
from .vector_db import get_vector_db
from .openrouter_manager import constrain_temperature_for_model

# Helper function to constrain temperature based on model type
# This is kept for backward compatibility, but now uses the OpenRouter manager
def constrain_temperature_for_model_legacy(temperature, model):
    """Legacy function for temperature constraints - use openrouter_manager instead"""
    return constrain_temperature_for_model(temperature, model)


agent_lock = asyncio.Lock()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the built React app
DIST_PATH = Path(__file__).parent.parent / "dist"
STATIC_PATH = DIST_PATH / "assets"

# Mount static files (CSS, JS, images, etc.) from React build
if STATIC_PATH.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_PATH)), name="assets")


class ChatRequest(BaseModel):
    messages: list[dict] | None = None  # Each: {"role": "user"|"assistant"|"tool", "content": str}
    user_input: str | None = None
    model: str | None = None
    temperature: float | None = None
    verbosity: int | None = None
    debug: bool | None = None


@app.post("/chat")
async def chat(
    messages: str = Form(...),
    options: str = Form('{}'),
    file: UploadFile = File(None)
):
    messages_data = json.loads(messages)
    options_data = json.loads(options)

    # Basic env checks for required keys depending on model type happen in create_agent
    model = options_data.get("model") or "gemini-2.5-flash"
    temperature = constrain_temperature_for_model(options_data.get("temperature", 0.5), model)
    verbosity = options_data.get("verbosity", 3)
    debug = options_data.get("debug", False)

    file_content = None
    artifacts = []
    dataframe = None
    if file:
        file_content, artifacts, dataframe = await process_file(file)

    try:
        agent = create_agent(temperature=temperature, model=model, verbosity=verbosity, debug=debug, file_content=file_content, dataframe=dataframe)
    except Exception as e:
        return {"text": f"Server not configured: {e}", "artifacts": artifacts}

    # Build messages array for the agent
    system_prompt = load_system_prompt()
    final_messages = [{"role": "system", "content": system_prompt}]

    if messages_data:
        if messages_data[0].get("role") == "system":
            final_messages = messages_data
        else:
            final_messages.extend(messages_data)

    if file_content:
        for i in reversed(range(len(final_messages))):
            if final_messages[i].get("role") == "user":
                final_messages[i]["content"] = f"{final_messages[i]['content']}\n\nFile content:\n{file_content}"
                break
    
    payload = {"messages": final_messages}
    
    async with agent_lock:
        clear_captured_figures()
        result = await asyncio.to_thread(agent.invoke, payload)
        captured_figures = get_captured_figures()

    if captured_figures:
        for fig_json in captured_figures:
            try:
                # The figure is a JSON string, so parse it
                artifacts.append({"type": "plot", "content": json.loads(fig_json)})
            except json.JSONDecodeError:
                # Handle cases where the string is not valid JSON
                print(f"Warning: Could not decode captured figure JSON: {fig_json}")

    # Extract clean assistant markdown text
    from typing import Any
    try:
        from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    except Exception:  # Fallback if types unavailable
        AIMessage = ToolMessage = HumanMessage = tuple()

    def extract_markdown(r: Any) -> str:
        # Prefer traversing a message list
        msgs = None
        if hasattr(r, "messages"):
            msgs = getattr(r, "messages")
        elif isinstance(r, dict) and "messages" in r:
            msgs = r["messages"]

        if isinstance(msgs, list):
            last_text = None
            for m in reversed(msgs):  # Start from the last message
                is_ai_message = isinstance(m, AIMessage) or \
                                (isinstance(m, dict) and m.get("role") in ("assistant", "ai"))

                if is_ai_message:
                    content = m.content if isinstance(m, AIMessage) else m.get("content")

                    if isinstance(content, list):
                        # Join list content into a single string, filtering out empty parts
                        joined_content = "\n".join(str(c).strip() for c in content if str(c).strip()).strip()
                        if joined_content:
                            return joined_content  # Return the first valid AI message found
                    elif isinstance(content, str) and content.strip():
                        return content.strip()  # Return the first valid AI message found

            # If no AI message with content is found in the list, fall through

        # Handle direct AIMessage response
        if isinstance(r, AIMessage):
            content = r.content
            if isinstance(content, list):
                return "\n".join(str(c).strip() for c in content if str(c).strip()).strip()
            elif isinstance(content, str) and content.strip():
                return content.strip()

        # Handle dict shapes with direct output/content
        if isinstance(r, dict):
            for k in ("output", "content", "text"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # Fallback: convert the entire raw response to a string
        raw_response_str = str(r)
        if raw_response_str.startswith("{'messages':"):
            return "Oops, it seems that my wires got mixed up... Can you try again?"
        return raw_response_str

    text = extract_markdown(result)
    return {"text": text, "artifacts": artifacts}


# The SSEQueueHandler is no longer needed with astream_log

@app.get("/thinking")
async def thinking():
    import os
    path = os.path.join(os.path.dirname(__file__), "thinking_phrases.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        return {"phrases": lines}
    except Exception as e:
        return {"phrases": ["Thinking…"], "error": str(e)}

@app.post("/chat/stream")
async def chat_stream(
    messages: str = Form(...),
    options: str = Form('{}'),
    file: UploadFile = File(None)
):
    messages_data = json.loads(messages)
    options_data = json.loads(options)

    model = options_data.get("model") or "gemini-2.5-flash"
    temperature = constrain_temperature_for_model(options_data.get("temperature", 0.5), model)
    verbosity = options_data.get("verbosity", 3)
    debug = options_data.get("debug", False)

    file_content = None
    artifacts = []
    dataframe = None
    if file:
        file_content, artifacts, dataframe = await process_file(file)

    async def sse_generator():
        if artifacts:
            yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifacts})}\n\n"

        # No longer need a separate streaming-specific LLM instance
        try:
            agent_local = create_agent(
                temperature=temperature,
                model=model,
                verbosity=verbosity,
                llm=None, # Let create_agent handle it
                debug=debug,
                file_content=file_content,
                dataframe=dataframe
            )
        except (ValueError, DefaultCredentialsError) as e:
            yield f"data: {json.dumps({'type':'error','message': f'Server Configuration Error: {e}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        system_prompt = load_system_prompt()
        final_messages = [{"role": "system", "content": system_prompt}]
        if messages_data:
            if messages_data[0].get("role") == "system":
                final_messages = messages_data
            else:
                final_messages.extend(messages_data)

        if file_content:
            for i in reversed(range(len(final_messages))):
                if final_messages[i].get("role") == "user":
                    final_messages[i]["content"] = f"{final_messages[i]['content']}\n\nFile content:\n{file_content}"
                    break
        
        payload = {"messages": final_messages}

        async with agent_lock:
            clear_captured_figures()
            try:
                config = RunnableConfig()
                config['recursion_limit'] = 10
                # Use astream_log for true streaming of all events
                async for chunk in agent_local.astream_log(payload, config=config, include_names=["ChatOpenAI", "ChatGoogleGenerativeAI", "ChatMistralAI", "ToolsAgentOutputParser"]):
                    for op in chunk.ops:
                        # op: 'add', 'remove', 'replace'
                        # path: '/streamed_output/-', '/logs/.../streamed_output/-'
                        path = op.get("path")
                        
                        if op.get("op") == "add" and path.endswith("/streamed_output/-"):
                            # This is a token from the LLM or final output
                            data = op.get("value")
                            if isinstance(data, dict) and "content" in data:
                                # It's a message chunk
                                token = data.get("content")
                                if token:
                                    yield f"data: {json.dumps({'type': 'delta', 'text': token})}\n\n"
                            elif isinstance(data, str) and data:
                                # It's a direct string output (less common with new agent types)
                                yield f"data: {json.dumps({'type': 'delta', 'text': data})}\n\n"
                        
                        # Check for tool calls
                        if op.get("op") == "add" and "/tool_calls/-" in path:
                            tool_call = op.get("value")
                            if isinstance(tool_call, dict) and "name" in tool_call:
                                tool_name = tool_call.get("name")
                                tool_map = {
                                    "tavily_search": "Searching the web...",
                                    "search_vector_db": "Searching documents...",
                                    "CustomSemanticScholarQueryRun": "Searching academic papers...",
                                    "PythonREPLTool": "Analyzing data...",
                                    "crawl4ai_scraper": "Scraping website...",
                                    "search_documents": "Searching RAG database...",
                                    "store_document": "Storing document...",
                                    "get_document_info": "Retrieving document info..."
                                }
                                message = tool_map.get(tool_name, f"Running tool: {tool_name}...")
                                yield f"data: {json.dumps({'type': 'status', 'message': message})}\n\n"


                # After streaming, check for any captured figures
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
                error_msg = f"Error processing request: {str(e)}"
                print(f"Error during agent execution: {error_msg}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            
            finally:
                # Signal completion
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def process_file(file: UploadFile):
    file_content = await file.read()
    mime_type = magic.from_buffer(file_content, mime=True)

    filename = file.filename
    content_str = f"File: {filename}\n"
    artifacts = []
    dataframe = None

    try:
        if mime_type == 'text/csv':
            df = pd.read_csv(io.BytesIO(file_content))
            buffer = io.StringIO()
            df.info(buf=buffer)
            content_str += f"CSV file loaded as a dataframe with the following info:\n{buffer.getvalue()}\n\nFirst 5 rows:\n{df.head().to_string()}"
            artifacts.append({"type": "dataframe", "content": df.to_html()})
            dataframe = df
        elif mime_type == 'application/pdf':
            import base64
            try:
                reader = PdfReader(io.BytesIO(file_content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                get_vector_db().add_pdf(file_content)
                content_str += f"PDF '{filename}' has been successfully indexed. You can now ask questions about it."

                pdf_data_url = f"data:application/pdf;base64,{base64.b64encode(file_content).decode('utf-8')}"
                artifacts.append({"type": "pdf", "content": pdf_data_url, "filename": filename})
                artifacts.append({"type": "document", "filename": filename, "content": text})
            except Exception as e:
                content_str += f"Error processing PDF file: {e}"
                artifacts.append({"type": "error", "content": f"Error processing PDF {filename}: {e}"})
        elif filename and filename.endswith('.sav'):
            df, meta = pyreadstat.read_sav(io.BytesIO(file_content))
            buffer = io.StringIO()
            df.info(buf=buffer)
            content_str += f"SAV file loaded as a dataframe with the following info:\n{buffer.getvalue()}\n\nFirst 5 rows:\n{df.head().to_string()}"
            artifacts.append({"type": "dataframe", "content": df.to_html()})
            dataframe = df
        elif filename and (filename.endswith('.rdata') or filename.endswith('.rds')):
            try:
                result = pyreadr.read_r(io.BytesIO(file_content))
                if result:
                    first_key = list(result.keys())[0]
                    df = result[first_key]
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    content_str += f"Rdata file loaded. Dataframe '{first_key}' has the following info:\n{buffer.getvalue()}\n\nFirst 5 rows:\n{df.head().to_string()}"
                    artifacts.append({"type": "dataframe", "label": first_key, "content": df.to_html()})
                    dataframe = df
                else:
                    content_str += "Could not read any dataframes from the Rdata file."
                    artifacts.append({"type": "error", "content": "Could not read any dataframes from the Rdata file."})
            except Exception as e:
                content_str += f"Error processing Rdata file: {e}"
                artifacts.append({"type": "error", "content": f"Error processing Rdata {filename}: {e}"})
        else:
            content_str += "Unsupported file type."
            artifacts.append({"type": "error", "content": f"Unsupported file type: {mime_type}"})
    except Exception as e:
        content_str += f"Error processing file: {e}"
        artifacts.append({"type": "error", "content": f"Error processing {filename}: {e}"})

    return content_str, artifacts, dataframe


# Serve React app for all non-API routes (must be last)
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """Serve the React application for all non-API routes"""
    
    # Skip known API routes
    api_routes = ["chat", "thinking"]
    if full_path in api_routes or full_path.startswith("chat/"):
        return {"detail": "API endpoint not found"}
    
    # Check if it's a request for a static file (like favicon.ico, manifest.json, etc.)
    if "." in full_path:
        static_file_path = DIST_PATH / full_path
        if static_file_path.exists() and static_file_path.is_file():
            return FileResponse(str(static_file_path))
    
    # For all other routes, serve the React app (for client-side routing)
    index_path = DIST_PATH / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {"detail": "Frontend application not found"}


# Root route - serve React app
@app.get("/")
async def read_root():
    """Serve the React application at root"""
    index_path = DIST_PATH / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {"detail": "Frontend application not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)