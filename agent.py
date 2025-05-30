import os
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from typing import Any, List, Dict
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM # Import base LLM type for type hinting

from tools import (
    get_search_tools, 
    get_semantic_scholar_tool_for_agent,
    get_web_scraper_tool_for_agent,
    get_rag_tool_for_agent,
    get_coder_tools
)
from dotenv import load_dotenv

load_dotenv()

# Determine project root based on the script's location
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# --- Constants ---
SUGGESTED_PROMPT_COUNT = 4

# --- Global Settings ---
def initialize_settings():
    """Initializes LlamaIndex settings with Gemini LLM and Embedding model."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Use Google Generative AI Embeddings
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=google_api_key)
    # Use a potentially more stable model name and set a default temperature
    Settings.llm = Gemini(model_name="models/gemini-2.5-flash-preview-04-17",
                          api_key=google_api_key,
                          temperature=0.7) 


# --- Greeting Generation ---
def generate_llm_greeting() -> str:
    """Generates a dynamic greeting message using the configured LLM."""
    static_fallback = "Hello! I'm ESI, your AI assistant for dissertation support. How can I help you today?"
    try:
        # Ensure settings are initialized (might be redundant if called elsewhere first, but safe)
        if not Settings.llm:
            initialize_settings()

        llm = Settings.llm
        if not isinstance(llm, LLM): # Basic check
             print("Warning: LLM not configured correctly for greeting generation.")
             return static_fallback

        # Use a simple prompt for a greeting
        prompt = """Generate a short, friendly greeting (1-2 sentences) for ESI, an AI dissertation assistant. Mention ESI by name and offer help. Provide only the greeting."""
        response = llm.complete(prompt)
        greeting = response.text.strip()

        # Basic validation
        if not greeting or len(greeting) < 10:
            print("Warning: LLM generated an empty or too short greeting. Falling back.")
            return static_fallback

        print(f"Generated LLM Greeting: {greeting}")
        return greeting

    except Exception as e:
        print(f"Error generating LLM greeting: {e}. Falling back to static message.")
        return static_fallback


# --- Comprehensive Agent Definition ---
def create_orchestrator_agent() -> AgentRunner:
    """
    Creates a single comprehensive agent that has access to all specialized tools.
    This agent will act as the primary interface, leveraging various tools as needed.
    """
    initialize_settings() # Ensure LLM settings are initialized

    print("Initializing all tools for the comprehensive agent...")

    # Initialize all tools
    search_tools = get_search_tools()
    lit_reviewer_tool = get_semantic_scholar_tool_for_agent() # This returns a single FunctionTool
    web_scraper_tool = get_web_scraper_tool_for_agent() # This returns a single FunctionTool
    rag_tool = get_rag_tool_for_agent() # This returns a single FunctionTool
    coder_tools = get_coder_tools()

    # Combine all tools into a single list
    all_tools = []
    if search_tools:
        all_tools.extend(search_tools)
    if lit_reviewer_tool:
        all_tools.append(lit_reviewer_tool)
    if web_scraper_tool:
        all_tools.append(web_scraper_tool)
    if rag_tool:
        all_tools.append(rag_tool)
    if coder_tools:
        all_tools.extend(coder_tools)

    if not all_tools:
        raise RuntimeError("No tools could be initialized for the comprehensive agent. Agent cannot function.")

    print(f"Initialized {len(all_tools)} tools for the comprehensive agent.")

    try:
        with open("esi_agent_instruction.md", "r") as f:
             system_prompt_base = f.read().strip()
    except FileNotFoundError:
        print("Warning: esi_agent_instruction.md not found. Using default base prompt for the comprehensive agent.")
        system_prompt_base = "You are ESI, an AI assistant for dissertation support."

    comprehensive_system_prompt = f"""{system_prompt_base}
Your role is to understand the user's query and use the available tools to gather information, perform tasks, and synthesize a comprehensive final answer.
You have access to the following tools:
*   **Search Tools (DuckDuckGo, Tavily, Wikipedia)**: For general web searches, current events, or broad topics.
*   **Literature Review Tool (Semantic Scholar)**: For finding academic papers and scholarly articles.
*   **Web Scraper Tool**: To fetch the textual content of a specific webpage URL.
*   **RAG Tool (rag_dissertation_retriever)**: For queries about specific institutional knowledge, previously saved research, or topics likely covered in the local dissertation knowledge base. Use this first for university-specific questions.
*   **Coder Tool (code_interpreter)**: To write and execute Python code for tasks like data analysis, plotting, complex calculations, or file generation.

The level of detail of your responses is controlled by the user from a verbosity parameter called V which varies from 1 to 5. 
Level 1 is the least verbose where you need to avoid any details or additional context and get straight to the response, while Level 5 is the most detailed and 
extremely verbose response. The level V for each new response will be provided as part of the context 

Your process:
1.  Analyze the user's query carefully.
2.  Determine which tool(s) are best suited to handle the query or parts of it. You can use multiple tools sequentially if needed.
3.  Formulate clear and concise inputs for each chosen tool.
4.  Call the tool(s).
5.  Review the responses from the tool(s).
6.  Synthesize all gathered information into a single, coherent, and helpful final answer for the user based *only* on the current query and the information received from the tools for *this query*.
7.  **Crucially, you MUST ensure the following markers from tools are included in YOUR final synthesized response to the user, if the information is used**:
    *   If the `rag_dissertation_retriever` tool provides `---RAG_SOURCE---{{...}}` markers in its response, you MUST include these exact markers (each on its own new line) in your final answer.
    *   If the `code_interpreter` tool's response indicates a file has been saved and provides a `---DOWNLOAD_FILE---filename.ext` marker, you MUST include this exact marker (on its own new line) at the end of your final answer.
    *   If `code_interpreter` provides Python code it executed, include this code in your final response, formatted with Markdown (e.g., ```python\ncode here\n```).

Be proactive and thorough. Cite sources when possible (based on information from tools).
If a tool returns an error or no useful information, acknowledge this politely in your final response. You may try another tool or ask the user for clarification if necessary to fulfill the original request.
Your final output to the user should be a single, complete response directly addressing their query using the tool information. Avoid conversational filler about the chat history (e.g., don't say "As we discussed before..." or "Like I said earlier..."). Focus on delivering the answer.
"""

    comprehensive_agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=all_tools,
        llm=Settings.llm,
        system_prompt=comprehensive_system_prompt,
        verbose=True, # Set to False for production
    )
    comprehensive_agent_runner = AgentRunner(comprehensive_agent_worker)
    print("Comprehensive agent created with all expert tools.")
    return comprehensive_agent_runner

# --- Suggested Prompts ---
DEFAULT_PROMPTS = [
    "Find recent papers on qualitative data analysis methods.",
    "Explain the structure of a typical literature review.",
    "What are common challenges students face in writing?",
    "Search for university policies on dissertation submission deadlines (uses RAG).",
]

def generate_suggested_prompts(chat_history: List[Dict[str, Any]]) -> List[str]:
    """
    Generates concise suggested prompts.
     """
     # --- LLM-based Generation ---
    try:
        # Ensure LLM is initialized
        if not Settings.llm:
            print("Warning: LLM not initialized for suggested prompts. Returning defaults.")
            return DEFAULT_PROMPTS

        llm = Settings.llm

        # Create context from the last few messages (e.g., last 4)
        context_messages = chat_history[-4:] # Get last 4 messages
        context_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in context_messages])

        # Construct the prompt for the LLM
        prompt = f"""Based on the recent conversation context below, suggest exactly {SUGGESTED_PROMPT_COUNT} concise follow-up prompts (under 15 words each) for a user working on a dissertation.
--- CONTEXT START ---
{context_str}
--- CONTEXT END ---
Output ONLY the prompts, each on a new line, without numbering or introductory text.
Example:
Find papers on topic X.
Explain concept Y.
How do I structure section Z?
"""

        print("Generating suggested prompts using LLM...")
        response = llm.complete(prompt)
        suggestions_text = response.text.strip()

        # Parse the response (split by newline, remove empty lines)
        suggested_prompts = [line.strip() for line in suggestions_text.split('\n') if line.strip()]

        # Validate the output
        if len(suggested_prompts) == SUGGESTED_PROMPT_COUNT and all(suggested_prompts):
            print(f"LLM generated prompts: {suggested_prompts}")
            return suggested_prompts
        else:
            print(f"Warning: LLM generated unexpected output for suggestions: '{suggestions_text}'. Falling back to defaults.")
            return DEFAULT_PROMPTS

    except Exception as e:
        print(f"Error generating suggested prompts with LLM: {e}. Falling back to defaults.")
        return DEFAULT_PROMPTS
