import streamlit as st
import os
from dotenv import load_dotenv
import io  # Import io for handling file streams
import uuid  # Import uuid for generating unique IDs

# Import chromadb client
import chromadb

# Use Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain import hub
import glob
from langchain_community.document_loaders import PyPDFLoader
from tavily import TavilyClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from crawl4ai import AsyncWebCrawler
import streamlit as st
from streamlit_interface import display_chat_messages, handle_user_input, display_sidebar, initialize_streamlit
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_interface import initialize_streamlit
from streamlit_interface import generate_followup_suggestions


# --- Configuration ---
load_dotenv()
DATA_DIR = "./data"  # Directory to store PDF files

# Check for necessary API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Load the system prompt as instructions:
try:
    with open('esi_agent_instruction.md', 'r') as f:
        instruction = f.read()
except FileNotFoundError:
    st.error("Could not find esi_agent_instruction.md. Please ensure it is in the correct directory.")
    instruction = ""

# Define the system message and prompt structure globally
system_message = f"""{instruction}
You are a helpful AI assistant designed to support university students with their dissertations.
Your goal is to help them brainstorm research ideas, structure their work, understand methodologies, and overcome challenges.

When you use tools, ALWAYS cite the source URL if one is provided.

**Tool Use Instructions:**

1.  When a student refers to the \"module\" they are referring to the MSc dissertation module at UEA, called NBS-7095x. You have access to information about this module via the `dissertation_resource_retriever` tool.
2.  **You MUST always use** the `dissertation_resource_retriever` tool first to find relevant information from the knowledge base (e.g., module deadlines, procedures, milestones, specific writing guides, methodology examples, previously discussed concepts). Cite information retrieved using this tool.
3.  Use the `duckduckgo_search` tool to find recent research papers, news, or general information not present in the knowledge base. Cite information retrieved using this tool.
4.  If the `tavily_search` tool is available, use it to supplement the `duckduckgo_search` for broader or more in-depth searches. It returns the most relevant search results with snippets. Cite information retrieved using this tool.
5.  Use the `crawl4ai` tool to crawl a specific website and extract its content. Only use this tool if you need to get information directly from a specific website. Be specific about the URL you want to crawl.
6.  If unsure about a specific academic convention, first search for information using the `duckduckgo_search` tool, the `tavily_search` tool (if available), the `dissertation_resource_retriever`, and the `crawl4ai` tool (if a specific website is relevant), and if unable to find the answer, advise the student to consult their supervisor or university guidelines.

"""

# Define the prompt message structure
prompt_messages = [
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]


# Using DuckDuckGo Search as a free alternative
search = DuckDuckGoSearchRun()
duckduckgo_search_tool = Tool(
    name="duckduckgo_search",
    func=search.run,
    description="Use this tool to search the internet for information. Use it to find recent research papers, news, or general information not present in the knowledge base. If the user is asking about something that is not specific to the module, use this tool.",
)

# --- Tavily Tool Setup ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    st.warning("TAVILY_API_KEY not found. Tavily Search tool will be disabled. Set TAVILY_API_KEY in .env if needed.")
    tavily_search_tool = None
else:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    tavily_search = Tool(
        name="tavily_search",
        func=tavily_client.search,
        description="Use this tool to search the internet for information. It is good for general information, academic papers, and current events. It returns the most relevant search results with snippets.",
    )

# --- Crawl4AI Tool Setup ---

crawl4ai = AsyncWebCrawler()
crawl4ai_tool = Tool(
    name="crawl4ai",
    func=crawl4ai.arun,
    description="Use this tool to crawl a website and extract its content. Input should be a valid URL. Only use this tool if you need to get information directly from a specific website. Be specific about the URL you want to crawl.",
)

# --- RAG Setup (Main Dissertation Knowledge Base) ---
# Define the path for the persistent ChromaDB database
CHROMA_DB_PATH = "./chroma_db_dissertation"
# Use Google Generative AI embeddings
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") # Updated model name

# Initialize ChromaDB client for the main knowledge base
# This uses the default persistent client
vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_function,
    collection_name="dissertation_resources" # You can name your collection
)

# Create a retriever from the main vector store
# `k=3` means it will retrieve the top 3 most relevant documents
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create a RAG tool for the main knowledge base
rag_tool = create_retriever_tool(
    retriever,
    "dissertation_resource_retriever",
    """Use this tool to retrieve information about the MSc dissertation module at UEA, called NBS-7091A.
    Specifically, use this tool to find information about module deadlines, procedures, milestones, specific writing guides, methodology examples, previously discussed concepts, scales, questionnaires, and instruments.
    If the question is at all related to the module requirements, use this tool first.""",
)

# --- LLM Setup ---
# Initialize the LLM (e.g., Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) 

# --- Agent Setup ---
# Define the base tools the agent can use (main knowledge base and search)
base_tools = [rag_tool, duckduckgo_search_tool, crawl4ai_tool]
if tavily_search:
    base_tools.append(tavily_search)

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define the system message and prompt structure globally
system_message = f"""{instruction}
You are a helpful AI assistant designed to support university students with their dissertations.
Your goal is to help them brainstorm research ideas, structure their work, understand methodologies, and overcome challenges.

When you use tools, ALWAYS cite the source URL if one is provided.

**Tool Use Instructions:**

1.  When a student refers to the \"module\" they are referring to the MSc dissertation module at UEA, called NBS-7095x. You have access to information about this module via the `dissertation_resource_retriever` tool.
2.  **You MUST always use** the `dissertation_resource_retriever` tool first to find relevant information from the knowledge base (e.g., module deadlines, procedures, milestones, specific writing guides, methodology examples, previously discussed concepts). Cite information retrieved using this tool.
3.  Use the `duckduckgo_search` tool to find recent research papers, news, or general information not present in the knowledge base. Cite information retrieved using this tool.
4.  If the `tavily_search` tool is available, use it to supplement the `duckduckgo_search` for broader or more in-depth searches. It returns the most relevant search results with snippets. Cite information retrieved using this tool.
5.  Use the `crawl4ai` tool to crawl a specific website and extract its content. Only use this tool if you need to get information directly from a specific website. Be specific about the URL you want to crawl.
6.  If unsure about a specific academic convention, first search for information using the `duckduckgo_search` tool, the `tavily_search` tool (if available), the `dissertation_resource_retriever`, and the `crawl4ai` tool (if a specific website is relevant), and if unable to find the answer, advise the student to consult their supervisor or university guidelines.

"""

# Define the prompt message structure
prompt_messages = [
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]


# --- PDF Ingestion Function (Main Knowledge Base) ---
def check_and_ingest_new_pdfs(data_directory: str, vector_store_instance: Chroma, db_path: str):
    """Checks for new PDFs in data_directory and ingests them into the main vector store."""
    log_file_path = os.path.join(db_path, ".ingested_files.log")
    ingested_files = set()

    # Create data directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        st.info(f"Created data directory: {data_directory}")

    # Read list of already ingested files
    try:
        with open(log_file_path, "r") as f:
            ingested_files = set(line.strip() for line in f)
        # st.info(f"Found log of {len(ingested_files)} previously ingested files.") # Keep this less verbose
    except FileNotFoundError:
        st.info("No ingestion log file found for the main knowledge base. Will process all PDFs in data directory.")

    # Find current PDF files
    current_pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
    current_filenames = set(os.path.basename(f) for f in current_pdf_files)

    # Determine new files to ingest
    new_filenames = current_filenames - ingested_files
    new_file_paths = [os.path.join(data_directory, fname) for fname in new_filenames]

    if not new_file_paths:
        st.success("Main knowledge base is up-to-date. No new PDFs found to ingest.")
        return

    st.info(f"Found {len(new_file_paths)} new PDF file(s) to ingest into the main knowledge base. Starting process...")

    all_new_docs = []
    processed_new_files = []
    for pdf_path in new_file_paths:
        filename = os.path.basename(pdf_path)
        try:
            st.write(f"  Loading: {filename}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_new_docs.extend(documents)
            processed_new_files.append(filename) # Track successfully loaded files
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            continue # Skip to the next file

    if not all_new_docs:
        st.error("No new documents were successfully loaded from the PDF files for the main knowledge base.")
        return

    st.write("Splitting new documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_new_docs)

    if not splits:
        st.error("Failed to split new documents into chunks for the main knowledge base.")
        return

    st.write(f"Adding {len(splits)} new document chunks to the main vector store...")
    try:
        # Add only the new documents to the vector store
        vector_store_instance.add_documents(splits)

        # Update the log file with newly processed files
        with open(log_file_path, "a") as f:
            for fname in processed_new_files:
                f.write(f"{fname}\n")
        st.success(f"Successfully ingested {len(processed_new_files)} new PDF file(s) into the main knowledge base!")

    except Exception as e:
        st.error(f"Error adding new documents to main vector store: {e}")


# def update_agent_executor():
#     """Recreates the agent executor with the current set of tools."""
#     current_tools = list(base_tools) # Start with the base tools
#     if "uploaded_retriever_tool" in st.session_state and st.session_state.uploaded_retriever_tool:
#         current_tools.append(st.session_state.uploaded_retriever_tool)
#         st.info(f"Agent tools updated: {', '.join([tool.name for tool in current_tools])}")
#     else:
#          st.info(f"Agent tools: {', '.join([tool.name for tool in current_tools])}")

#     # Retrieve the prompt from session state
#     agent_prompt = st.session_state.agent_prompt

#     # Create and store the new agent executor instance
#     st.session_state.agent_executor = AgentExecutor(
#         agent=create_tool_calling_agent(llm, current_tools, agent_prompt), # Use prompt from session state
#         tools=current_tools,
#         verbose=True # Set verbose=True for debugging
# --- Automatic PDF Ingestion on Startup (Main Knowledge Base) ---
with st.spinner("Checking for new documents to load into the main knowledge base..."):
    check_and_ingest_new_pdfs(DATA_DIR, vector_store, CHROMA_DB_PATH)

# Initialize Streamlit UI and session state
initialize_streamlit()

# Initialize the agent prompt in session state
if "agent_prompt" not in st.session_state:
    st.session_state.agent_prompt = ChatPromptTemplate.from_messages(prompt_messages)

if "agent_executor" not in st.session_state:
    # Initialize the agent executor with base tools on first run
    agent_prompt = st.session_state.agent_prompt
    st.session_state.agent_executor = AgentExecutor(
        agent=create_tool_calling_agent(llm, base_tools, agent_prompt),
        tools=base_tools,
        verbose=True
    )

# Display chat messages from history
display_chat_messages()

# Handle user input and generate AI response
handle_user_input(st.session_state.agent_executor, llm)

# Display the sidebar
display_sidebar()
