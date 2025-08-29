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
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
import asyncio
import magic
import pandas as pd
import pyreadstat
import pyreadr
from pypdf import PdfReader
import io
import os
import re
from pathlib import Path


from agent import create_agent, get_captured_figures, clear_captured_figures, is_mistral_model, MISTRAL_MODEL_MAPPING, load_system_prompt
from vector_db import get_vector_db
from openrouter_manager import constrain_temperature_for_model

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
    import json

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


class SSEQueueHandler(AsyncCallbackHandler):
    def __init__(self, queue: "asyncio.Queue[str]"):
        self.queue = queue
        self.count = 0

    def on_llm_new_token(self, token: str, **kwargs):  # type: ignore[override]
        # Forward tokens into async queue for SSE loop
        import json
        self.count += 1
        self.queue.put_nowait(json.dumps({'type': 'delta', 'text': token}))

    def on_llm_end(self, response, **kwargs):  # type: ignore[override]
        # Do not signal end of stream here; agent is still running
        pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        import json
        tool_name = serialized.get('name')
        # Map tool names to user-friendly text
        tool_map = {
            "tavily_search": "Searching the web...",
            "search_vector_db": "Searching documents...",
            "CustomSemanticScholarQueryRun": "Searching academic papers...",
            "PythonREPLTool": "Analyzing data...",
            "crawl4ai_scraper": "Scraping website...",
        }
        message = tool_map.get(tool_name, f"Running tool: {tool_name}...")
        self.queue.put_nowait(json.dumps({'type': 'status', 'message': message}))

    # No-ops to satisfy interface without raising
    def on_chat_model_start(self, *args, **kwargs): pass
    def on_chain_end(self, *args, **kwargs): pass
    def on_chain_start(self, *args, **kwargs): pass
    def on_chain_error(self, *args, **kwargs): pass
    def on_llm_start(self, *args, **kwargs): pass
    def on_llm_error(self, *args, **kwargs): pass
    def on_tool_end(self, *args, **kwargs): pass
    def on_tool_error(self, *args, **kwargs): pass
    def on_text(self, *args, **kwargs): pass
    def on_agent_action(self, *args, **kwargs): pass
    def on_agent_finish(self, *args, **kwargs): pass

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
    import json

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
        import json

        if artifacts:
            yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifacts})}\n\n"
        
        q: asyncio.Queue[str | None] = asyncio.Queue()
        handler = SSEQueueHandler(q)

        # Create a streaming-capable LLM without callbacks initially
        if model.startswith("gemini"):
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                streaming=True,
            )
        elif is_mistral_model(model):
            official_model_name = MISTRAL_MODEL_MAPPING.get(model, model)
            llm = ChatMistralAI(
                model=official_model_name,
                temperature=temperature,
                streaming=True,
                mistral_api_key=settings.MISTRAL_API_KEY,
            )
        elif is_mistral_model(model):
            llm = ChatMistralAI(
                model=model,
                temperature=temperature,
                streaming=True,
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            )
        else:
            # Use OpenRouter manager for consistent model handling
            from openrouter_manager import get_openrouter_manager, is_openrouter_model
            
            if is_openrouter_model(model):
                try:
                    manager = get_openrouter_manager()
                    llm = manager.create_llm(
                        model_id=model,
                        temperature=temperature,
                        streaming=True,
                    )
                except Exception as e:
                    manager = get_openrouter_manager()
                    error_msg = manager.handle_api_error(e, model)
                    yield f"data: {json.dumps({'type':'error','message': error_msg})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
            else:
                # Fallback for unknown models
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    streaming=True,
                    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                )
        
        # Create agent with the streaming LLM
        try:
            agent_local = create_agent(
                temperature=temperature,
                model=model,
                verbosity=verbosity,
                llm=llm,
                debug=debug,
                file_content=file_content,
                dataframe=dataframe
            )
        except (ValueError, DefaultCredentialsError) as e:
            # Yield a specific configuration error and stop
            yield f"data: {json.dumps({'type':'error','message': f'Server Configuration Error: {e}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        # Build the messages array for the agent
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
                # Debug: Log the payload being sent to the agent
                print(f"Debug: Streaming agent payload: {payload}")
                
                # Execute agent using the working non-streaming approach
                result = await asyncio.to_thread(agent_local.invoke, payload)
                print(f"Debug: Agent result type: {type(result)}")
                print(f"Debug: Agent result: {str(result)[:200]}...")
                
                captured_figures = get_captured_figures()
                
                # Extract response text using the same logic as non-streaming endpoint
                def extract_markdown(r):
                    msgs = None
                    if hasattr(r, "messages"):
                        msgs = getattr(r, "messages")
                        print(f"Debug: Found messages attribute with {len(msgs)} messages")
                    elif isinstance(r, dict) and "messages" in r:
                        msgs = r["messages"]
                        print(f"Debug: Found messages in dict with {len(msgs)} messages")

                    if isinstance(msgs, list):
                        for i, m in enumerate(reversed(msgs)):
                            try:
                                from langchain_core.messages import AIMessage
                                is_ai_message = isinstance(m, AIMessage)
                            except:
                                is_ai_message = isinstance(m, dict) and m.get("role") in ("assistant", "ai")

                            print(f"Debug: Message {i}: type={type(m)}, is_ai={is_ai_message}")
                            if is_ai_message:
                                try:
                                    content = m.content if hasattr(m, 'content') else m.get("content")
                                except:
                                    content = m.get("content") if isinstance(m, dict) else None

                                print(f"Debug: AI message content: {str(content)[:100]}...")
                                if isinstance(content, list):
                                    joined_content = "\n".join(str(c).strip() for c in content if str(c).strip()).strip()
                                    if joined_content:
                                        return joined_content
                                elif isinstance(content, str) and content.strip():
                                    return content.strip()

                    # Handle dict shapes with direct output/content
                    if isinstance(r, dict):
                        for k in ("output", "content", "text"):
                            v = r.get(k)
                            if isinstance(v, str) and v.strip():
                                print(f"Debug: Found text in dict key '{k}': {v[:100]}...")
                                return v.strip()

                    print(f"Debug: No extractable text found, returning string representation")
                    raw_response_str = str(r)
                    if raw_response_str.startswith("{'messages':"):
                        return "Oops, it seems that my wires got mixed up... Can you try again?"
                    return raw_response_str
                
                def add_markdown_structure(text):
                    """Add proper line breaks to text that lacks markdown structure"""
                    import re as regex_module  # Use alias to avoid any potential conflicts
                    
                    if not text:
                        return text
                    
                    # If the text already has paragraph breaks, return as-is
                    if '\n\n' in text:
                        return text
                    
                    # Add line breaks before markdown headers
                    text = regex_module.sub(r'(\S)\s*(#{1,6}\s)', r'\1\n\n\2', text)
                    
                    # Add line breaks before bullet points
                    text = regex_module.sub(r'(\S)\s*(\*\s\*\*)', r'\1\n\n\2', text)  # * **Bold items
                    text = regex_module.sub(r'(\S)\s*(\*\s(?!\*))', r'\1\n\2', text)  # Regular * items
                    text = regex_module.sub(r'(\S)\s*(-\s)', r'\1\n\2', text)  # - items
                    
                    # Add line breaks before numbered lists
                    text = regex_module.sub(r'(\S)\s*(\d+\.\s)', r'\1\n\2', text)
                    
                    # Add line breaks before blockquotes
                    text = regex_module.sub(r'(\S)\s*(>\s)', r'\1\n\2', text)
                    
                    # Add paragraph breaks after sentences that end sections
                    text = regex_module.sub(r'([.!?])\s+(#{1,6}\s|\*\s|\d+\.\s)', r'\1\n\n\2', text)
                    
                    # Clean up any triple+ newlines
                    text = regex_module.sub(r'\n{3,}', '\n\n', text)
                    
                    return text.strip()
                
                response_text = extract_markdown(result)
                print(f"Debug: Extracted response text length: {len(response_text)}")
                print(f"Debug: Raw response text repr: {repr(response_text[:300])}")
                newline_check = '\n' in response_text
                double_newline_check = '\n\n' in response_text
                print(f"Debug: Contains newlines: {newline_check}")
                print(f"Debug: Contains double newlines: {double_newline_check}")
                
                # Add proper markdown structure if missing
                structured_text = add_markdown_structure(response_text)
                double_newline_after = '\n\n' in structured_text
                print(f"Debug: After adding structure - Contains double newlines: {double_newline_after}")
                print("Debug: Structured text preview:", repr(structured_text[:300]))
                
                if structured_text and structured_text.strip():
                    # Implement smooth word-based streaming with markdown awareness
                    import re
                    
                    def smooth_markdown_streaming(text):
                        """Create smooth streaming chunks while preserving markdown structure"""
                        import re as regex_module  # Use alias to avoid any potential conflicts
                        
                        chunks = []
                        current_chunk = ""
                        in_code_block = False
                        words = []
                        
                        lines = text.split('\n')
                        
                        for line in lines:
                            # Check for code block boundaries
                            if line.strip().startswith('```'):
                                if current_chunk.strip():
                                    # Finish current chunk before code block
                                    chunks.append(current_chunk.rstrip())
                                    current_chunk = ""
                                
                                if in_code_block:
                                    # End of code block - send as single chunk
                                    current_chunk += line + '\n'
                                    chunks.append(current_chunk.rstrip())
                                    current_chunk = ""
                                    in_code_block = False
                                else:
                                    # Start of code block
                                    current_chunk = line + '\n'
                                    in_code_block = True
                            elif in_code_block:
                                # Inside code block - accumulate entire block
                                current_chunk += line + '\n'
                            else:
                                # Regular text - process word by word
                                if line.strip() == "":
                                    # Empty line - preserve but continue chunk
                                    current_chunk += '\n'
                                elif line.startswith('#') or line.startswith('*') or line.startswith('-') or regex_module.match(r'^\d+\.', line.strip()):
                                    # Headers and list items - send accumulated chunk first
                                    if current_chunk.strip():
                                        chunks.append(current_chunk.rstrip())
                                        current_chunk = ""
                                    # Then add the header/list item
                                    current_chunk = line + '\n'
                                else:
                                    # Regular line - split into words for smooth streaming
                                    words = line.split(' ')
                                    word_chunk = ""
                                    
                                    for i, word in enumerate(words):
                                        word_chunk += word
                                        if i < len(words) - 1:
                                            word_chunk += " "
                                        
                                        # Create chunks of 3-8 words for smooth streaming
                                        if len(word_chunk.split()) >= 5 or i == len(words) - 1:
                                            current_chunk += word_chunk
                                            if len(current_chunk.split()) >= 12:  # Send chunk every ~12 words
                                                chunks.append(current_chunk.rstrip())
                                                current_chunk = ""
                                            word_chunk = ""
                                    
                                    current_chunk += '\n'
                        
                        # Add any remaining content
                        if current_chunk.strip():
                            chunks.append(current_chunk.rstrip())
                        
                        return [chunk for chunk in chunks if chunk.strip()]
                    
                    chunks = smooth_markdown_streaming(structured_text)
                    
                    print(f"Debug: Streaming {len(chunks)} smooth chunks")
                    
                    for i, chunk in enumerate(chunks):
                        if chunk:
                            # Determine chunk type for appropriate spacing
                            import re as regex_module  # Use alias to avoid any potential conflicts
                            
                            is_code_block = '```' in chunk
                            is_header = chunk.strip().startswith('#')
                            is_list = chunk.strip().startswith(('*', '-')) or regex_module.match(r'^\d+\.', chunk.strip())
                            
                            # Add appropriate spacing
                            if i < len(chunks) - 1:
                                next_chunk = chunks[i + 1] if i + 1 < len(chunks) else ""
                                next_is_header = next_chunk.strip().startswith('#')
                                next_is_list = next_chunk.strip().startswith(('*', '-')) or regex_module.match(r'^\d+\.', next_chunk.strip())
                                next_is_code = '```' in next_chunk
                                
                                if is_code_block or next_is_header or next_is_code:
                                    chunk_text = chunk + "\n\n"
                                elif next_is_list and not is_list:
                                    chunk_text = chunk + "\n\n"
                                else:
                                    chunk_text = chunk + " "
                            else:
                                chunk_text = chunk
                            
                            # Dynamic delay based on chunk type and size
                            if is_code_block:
                                delay = 0.4  # Longer delay for code blocks
                            elif is_header:
                                delay = 0.2  # Medium delay for headers
                            elif len(chunk.split()) > 15:
                                delay = 0.15  # Slightly longer for bigger chunks
                            else:
                                delay = 0.08  # Fast for small word chunks
                            
                            print(f"Debug: Sending chunk {i+1} ({len(chunk.split())} words): {chunk_text[:50]}...")
                            yield f"data: {json.dumps({'type': 'delta', 'text': chunk_text})}\n\n"
                            await asyncio.sleep(delay)
                else:
                    # If no response text, send a fallback message
                    print(f"Debug: No response text, sending fallback")
                    fallback_text = "I apologize, but I couldn't generate a response. Please try again."
                    yield f"data: {json.dumps({'type': 'delta', 'text': fallback_text})}\n\n"
                
                # Handle artifacts from captured figures
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
                # Send error message as streaming response
                error_msg = f"Error processing request: {str(e)}"
                print(f"Debug: Exception occurred: {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'delta', 'text': 'I apologize, but I encountered an error. Please try again.'})}\n\n"
            
            # Signal completion
            print(f"Debug: Streaming completed")
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