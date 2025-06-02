import os
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from typing import Any, List, Dict, Optional, Type
from llama_index.core.tools import FunctionTool, ToolOutput
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.base.llms.types import ToolCall # Corrected import path for ToolCall
from llama_index.core.memory import ChatMemoryBuffer # For chat history management

# Workflow specific imports
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    Event,
    Context,
    step,
)


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
    Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", # Updated model name
                          api_key=google_api_key,
                          temperature=0.7)


# --- Workflow Event Classes ---
class InputEvent(Event):
    input: List[ChatMessage] # Expecting chat history as input

class StreamEvent(Event):
    delta: str # The delta content from the LLM stream

class ToolCallEvent(Event): # Redefined
    tool_calls: List[ToolCall] # Changed from OpenAIToolCall to ToolCall
    history_before_llm_call: List[ChatMessage]
    llm_tool_request_message: ChatMessage # The LLM message that contained the tool call requests
    # If Gemini has a different ToolCall object, OpenAIToolCall might need adjustment.

# --- Custom Workflow for Streaming Agent ---
class StreamingOrchestratorAgentWorkflow(Workflow):
    def __init__(self, llm: LLM, tools: List[FunctionTool], verbose: bool = True, timeout: int = 120):
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = llm
        self.tools = tools
        self.tools_dict = {tool.metadata.name: tool for tool in tools}
        self._verbose = verbose # Store verbose for logging within steps if needed

    @step
    async def prepare_chat_history(self, ev: StartEvent) -> InputEvent:
        """
        Prepares chat history from the input event.
        The StartEvent is expected to have 'user_query' and 'chat_history'.
        """
        user_query = ev.get("user_query", None)
        chat_history = ev.get("chat_history", [])

        if not user_query:
            raise ValueError("StartEvent must contain 'user_query'.")

        # Add the new user query to the chat history
        current_chat_history = list(chat_history) # Make a copy
        current_chat_history.append(ChatMessage(role=MessageRole.USER, content=user_query))

        if self._verbose:
            print(f"Workflow Step: prepare_chat_history. Input query: {user_query}")
            print(f"Workflow Step: prepare_chat_history. History length: {len(current_chat_history)}")

        return InputEvent(input=current_chat_history)

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> Event:
        """
        Handles LLM interaction, streams response, and identifies tool calls.
        """
        if self._verbose:
            print("Workflow Step: handle_llm_input. Calling LLM with tools...")

        # Use astream_chat_with_tools for streaming and tool calling
        response_stream = await self.llm.astream_chat_with_tools(
            tools=self.tools,
            chat_history=ev.input, # ev.input is already List[ChatMessage]
        )

        full_response_message = None
        async for response_chunk in response_stream:
            # Stream out delta
            # Ensure delta is a string, handle None or other types if necessary
            delta_content = response_chunk.delta if response_chunk.delta is not None else ""
            ctx.write_event_to_stream(StreamEvent(delta=delta_content))

            # Store the last message chunk, which should contain the full response and tool calls
            full_response_message = response_chunk

        if self._verbose:
            print(f"Workflow Step: handle_llm_input. LLM stream complete. Final message: {full_response_message}")

        if full_response_message is None:
            # This case should ideally not happen if LLM always returns something
            print("Workflow Step: handle_llm_input. Warning: LLM stream ended without a final message.")
            return StopEvent(result={"response_message": ChatMessage(content="No response from LLM.", role=MessageRole.ASSISTANT)})

        # Add the full assistant response (including potential tool call requests) to history for the next turn
        # The actual adding to memory will be managed by the caller (app.py) after the stream is consumed.
        # Here, we just prepare for the next step in the workflow based on this response.

        tool_calls = self.llm.get_tool_calls_from_response(
            full_response_message,
            error_on_no_tool_call=False
        )

        if tool_calls:
            if self._verbose:
                print(f"Workflow Step: handle_llm_input. Tool calls identified: {tool_calls}")
            # Pass the history that led to this LLM call, and the LLM's response that contains the tool requests
            return ToolCallEvent(
                tool_calls=tool_calls,
                history_before_llm_call=ev.input, # History used for the LLM call
                llm_tool_request_message=full_response_message # LLM's message asking for tools
            )
        else:
            if self._verbose:
                print("Workflow Step: handle_llm_input. No tool calls. Stopping.")
            return StopEvent(result={"response_message": full_response_message})

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """
        Executes tool calls and prepares history for the next LLM interaction.
        """
        if self._verbose:
            print(f"Workflow Step: handle_tool_calls. Executing tools: {ev.tool_calls}")

        tool_output_messages: List[ChatMessage] = []
        for tool_call in ev.tool_calls:
            tool = self.tools_dict.get(tool_call.name)
            if not tool:
                print(f"Warning: Tool '{tool_call.name}' not found. Skipping.")
                tool_output_messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=f"Error: Tool '{tool_call.name}' not found.",
                        additional_kwargs={"tool_call_id": tool_call.id, "name": tool_call.name}
                    )
                )
                continue

            try:
                if self._verbose:
                    print(f"Calling tool: {tool_call.name} with args: {tool_call.arguments}")
                # Assuming tool.call might be synchronous, wrap if necessary or use async versions if available
                # For now, direct call. If tools are async, this needs `await`.
                # LlamaIndex tools' `__call__` is typically synchronous.
                output = tool(**tool_call.arguments) # Pass arguments as kwargs

                if self._verbose:
                    print(f"Tool {tool_call.name} output: {output.content[:100]}...")

                tool_output_messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=str(output.content), # Ensure content is string
                        additional_kwargs={"tool_call_id": tool_call.id, "name": tool_call.name}
                    )
                )
            except Exception as e:
                print(f"Error executing tool {tool_call.name}: {e}")
                tool_output_messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=f"Error executing tool {tool_call.name}: {str(e)}",
                        additional_kwargs={"tool_call_id": tool_call.id, "name": tool_call.name}
                    )
                )

        # Construct the new history for the next LLM call
        # History up to (and including) user message -> LLM message requesting tools -> Tool outputs
        new_history = list(ev.history_before_llm_call) # This history already includes the user query that led to LLM call.
        new_history.append(ev.llm_tool_request_message) # Add the LLM's message that requested the tools
        new_history.extend(tool_output_messages) # Add all tool outputs

        if self._verbose:
            print(f"Workflow Step: handle_tool_calls. New history length for next LLM call: {len(new_history)}")

        return InputEvent(input=new_history) # Loop back to handle_llm_input

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
        # Updated prompt to include the specific open-ended question from esi_agent_instruction.md
        prompt = """Generate a short, friendly greeting (1-2 sentences) for ESI, an AI dissertation assistant. Mention ESI by name and offer help.
        After the greeting, include one of these open-ended questions:
        - "So, what's on your mind about your dissertation today?"
        - "How can I lend a hand with your research?"
        Combine the greeting and the question naturally. Provide only the combined greeting and question."""
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
def create_orchestrator_agent() -> StreamingOrchestratorAgentWorkflow: # Return type changed
    """
    Creates a comprehensive agent workflow that has access to all specialized tools.
    This workflow will act as the primary interface, leveraging various tools as needed.
    """
    initialize_settings() # Ensure LLM settings are initialized

    print("Initializing all tools for the Streaming Orchestrator Agent Workflow...")

    # Initialize all tools
    search_tools = get_search_tools()
    lit_reviewer_tool = get_semantic_scholar_tool_for_agent()
    web_scraper_tool = get_web_scraper_tool_for_agent()
    rag_tool = get_rag_tool_for_agent()
    coder_tools = get_coder_tools()

    all_tools_list: List[FunctionTool] = [] # Explicitly type hint
    if search_tools:
        all_tools_list.extend(search_tools)
    if lit_reviewer_tool:
        all_tools_list.append(lit_reviewer_tool)
    if web_scraper_tool:
        all_tools_list.append(web_scraper_tool)
    if rag_tool:
        all_tools_list.append(rag_tool)
    if coder_tools:
        all_tools_list.extend(coder_tools)

    if not all_tools_list:
        raise RuntimeError("No tools could be initialized for the agent workflow. Workflow cannot function.")

    print(f"Initialized {len(all_tools_list)} tools for the agent workflow.")

    # The system prompt is now part of the LLM calls within the workflow,
    # typically by being the first system message in the chat history.
    # The Gemini LLM in LlamaIndex might handle system prompts differently (e.g., via generate_kwargs).
    # For FunctionCalling/ReAct agents, a system prompt is often passed to the worker.
    # For a workflow, the LLM's behavior is guided by the history and the tools provided.
    # The detailed instructions previously in `comprehensive_system_prompt` should ideally be
    # part of the LLM's configuration or the initial messages if needed.
    # For now, we rely on the tools' descriptions and the LLM's function calling capabilities.
    # The verbosity control and specific marker handling (RAG_SOURCE, DOWNLOAD_FILE)
    # will need to be managed by the LLM's responses based on its general instructions
    # or by post-processing in app.py if the workflow itself doesn't enforce it.
    # The workflow's main job is to orchestrate LLM calls and tool executions.

    # The `comprehensive_system_prompt` used with AgentRunner might need to be adapted.
    # Gemini models are often used with a system message at the start of the chat history.
    # We will assume `Settings.llm` is configured to understand tool usage.
    # The specific instructions about markers (RAG_SOURCE, DOWNLOAD_FILE) and verbosity
    # are more about the *content* the LLM generates. The workflow ensures tools are called.
    # These instructions should be part of the initial system message given to the LLM,
    # which means `app.py` should ensure `esi_agent_instruction.md` is the first system message.
    # The workflow itself doesn't directly use a "system_prompt" argument like AgentWorker.

    workflow = StreamingOrchestratorAgentWorkflow(
        llm=Settings.llm,
        tools=all_tools_list,
        verbose=True # Set to False for production
    )

    print("Streaming Orchestrator Agent Workflow created.")
    return workflow

# --- Suggested Prompts ---
DEFAULT_PROMPTS = [
    "Help me brainstorm ideas.",
    "I need to develop my research questions.",
    "I have my topic but I need help with developing hypotheses.",
    "I have my hypotheses but I am need help to design the study.",
    "Can you help me design my qualitative study?"
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
        suggestions_text = response.strip()

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
