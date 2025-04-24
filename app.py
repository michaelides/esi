import streamlit as st
import os
from dotenv import load_dotenv
import io  # Import io for handling file streams
import uuid  # Import uuid for generating unique IDs
# Import LanceDB
from langchain_community.vectorstores import LanceDB
import lancedb # Import lancedb client

# Use Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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

from streamlit_interface import display_chat_messages, handle_user_input, display_sidebar, initialize_streamlit
from langchain_core.messages import AIMessage, HumanMessage


# --- Configuration ---
loaded_env = load_dotenv()
DATA_DIR = "./data"  # Directory to store PDF files

# Check for necessary API keys
if loaded_env:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
else:
    GOOGLE_API_KEY = st.secrets.esi.GOOGLE_API_KEY
    GOOGLE_CSE_ID = st.secrets.esi.GOOGLE_CSE_ID
    TAVILY_API_KEY = st.secrets.esi.TAVILY_API_KEY



# --- Embedding Model Setup ---
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

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

When you use the duckduckgo_search and tavily_search tools, ALWAYS cite the source URL if one is provided. This should be inline citation (e.g. evidence show xyz[X],
where X is a number. The url for each of these citations should be provided at the end.

**Tool Use Instructions:**

1.  When a student refers to the \"module\" they are referring to the MSc dissertation module at UEA, called NBS-7095x. You have access to information about this module via the `dissertation_resource_retriever` tool.
2.  **CRITICAL INSTRUCTION: For EVERY user query, your FIRST action MUST be to use the `dissertation_resource_retriever` tool.** Check if the knowledge base contains relevant information about the query (e.g., module deadlines, procedures, milestones, specific writing guides, methodology examples, previously discussed concepts, scales, questionnaires, instruments). **DO NOT skip this step.** Only if the retriever returns no relevant information should you consider other tools or answer directly. Always cite information retrieved using this tool.
3.  Use the `duckduckgo_search` tool to find recent research papers, news, or general information not present in the knowledge base. Cite information retrieved using this tool.
4.  If the `tavily_search` tool is available, use it to supplement the `duckduckgo_search` for broader or more in-depth searches. It returns the most relevant search results with snippets. Cite information retrieved using this tool.
5.  If unsure about a specific academic convention, first search for information using the `duckduckgo_search` tool, the `tavily_search` tool (if available), and the `dissertation_resource_retriever`, and if unable to find the answer, advise the student to consult their supervisor or university guidelines.
6.  When asked to search for something or asked to find or reccomend literature you should use all of your tools
7.  When asked to find information or literature about a specific author, you should use all of your tools.
8.  If you want to mention the "dissertation_knowledge_retriever", refer to it as your "knowledge base".
9.  If you want to mention the "duckduckgo_search" or the "tavily_search" refer to them as your "search engine".
10. If you come accross any journal references or full papers when using the "dissertation_knowledge_retriever", you should cite them and provide the references in APA format

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
duckduckgo_search = Tool(
    name="duckduckgo_search",
    func=search.run,
    description="Use this tool to search the internet for information. Use it to find recent research papers, news, or general information not present in the knowledge base. If the user is asking about something that is not specific to the module, use this tool.",
)

# --- Tavily Tool Setup ---
if not TAVILY_API_KEY:
    st.warning("TAVILY_API_KEY not found. Tavily Search tool will be disabled. Set TAVILY_API_KEY in .env if needed.")
    tavily_search = None
else:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    tavily_search = Tool(
        name="tavily_search",
        func=tavily_client.search,
        description="""Use this tool to search the internet for information. It is good for general information, academic papers, academic authors, and current events.
        It returns the most relevant search results with snippets.""",
    )

# --- RAG Setup (Main Dissertation Knowledge Base) ---
# Define the path for the persistent LanceDB database
LANCEDB_DB_PATH = "./lance" # Changed path name to match scrape_and_ingest.py
COLLECTION_NAME = "dissertation_resources" # This will be the table name in LanceDB

# Initialize LanceDB vector store for the main knowledge base
try:
    # Connect to the LanceDB database
    db = lancedb.connect(LANCEDB_DB_PATH)

    # Check if the table exists before trying to initialize the vector store
    table_names = db.table_names()
    if COLLECTION_NAME not in table_names:
        st.error(f"LanceDB table '{COLLECTION_NAME}' not found at {LANCEDB_DB_PATH}.")
        st.info("Please run the ingestion script (`python scrape_and_ingest.py`) first to create and populate the database.")
        st.stop() # Stop the app if the database/table is not ready

    # Initialize LanceDB vector store using the connection and table name
    # Pass the connection object, table name, and embedding function
    vector_store = LanceDB(
        connection=db, # Pass the LanceDB connection object
        table_name=COLLECTION_NAME, # Pass the table name
        embedding=embedding_function # Pass the embedding function
    )

    st.info(f"Connected to LanceDB database at {LANCEDB_DB_PATH} and table '{COLLECTION_NAME}'")

except FileNotFoundError:
    st.error(f"LanceDB database directory not found at {LANCEDB_DB_PATH}.")
    st.info("Please run the ingestion script (`python scrape_and_ingest.py`) first to create and populate the database.")
    st.stop() # Stop the app if the database directory doesn't exist
except Exception as e:
    st.error(f"Failed to initialize LanceDB vector store: {e}")
    st.stop() # Stop the app on other initialization errors


# Create a retriever from the main vector store
# `k=5` means it will retrieve the top 5 most relevant documents
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create a RAG tool for the main knowledge base
rag_tool = create_retriever_tool(
    retriever,
    "dissertation_resource_retriever",
    """Use this tool to retrieve information about literature, references, authors, people, and the MSc dissertation module at UEA, called NBS-7091A.
    Use this tool to find information about module deadlines, procedures, milestones, specific writing guides,
    methodology examples, previously discussed concepts, scales, questionnaires, and instruments.
    If the question is at all related to the module requirements, use this tool first.
    Use this tool for anything related to the UEA, research ethics, and ethics applications.
    Use this tool when asked to search for publications of different authors. """,
)


# --- Agent Setup ---
# Define the base tools the agent can use (main knowledge base and search)
# Ensure tavily_search is included only if it was successfully initialized
base_tools = [rag_tool, duckduckgo_search]
if tavily_search:
    base_tools.append(tavily_search)


# Initialize Streamlit UI and session state (This must run before LLM/Agent init)
initialize_streamlit()

# Initialize the agent prompt in session state
if "agent_prompt" not in st.session_state:
    st.session_state.agent_prompt = ChatPromptTemplate.from_messages(prompt_messages)

# --- LLM Setup ---
# Initialize the LLM (e.g., Gemini) using the temperature from session state
# This LLM object will be used regardless of whether the agent is re-initialized.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=st.session_state.get("llm_temperature", 0.7) # Use the current session state value
)

# --- Agent Executor Setup ---
# Re-initialize the agent executor if it doesn't exist or if the temperature slider value changes
# We store the temperature used for the current agent in session state.
if "agent_executor" not in st.session_state or \
   "current_agent_temperature" not in st.session_state or \
   st.session_state.current_agent_temperature != st.session_state.llm_temperature:

    print(f"DEBUG: Re-initializing agent with temperature: {st.session_state.llm_temperature}")

    agent_prompt = st.session_state.agent_prompt
    st.session_state.agent_executor = AgentExecutor(
        agent=create_tool_calling_agent(llm, base_tools, agent_prompt), # Use the 'llm' object created above
        tools=base_tools,
        verbose=True
    )
    # Store the temperature used to create this agent
    st.session_state.current_agent_temperature = st.session_state.llm_temperature
else:
    # If temperature hasn't changed, use the existing agent from session state.
    # The 'llm' object created before the if condition already has the correct temperature.
    print(f"DEBUG: Using existing agent with temperature: {st.session_state.current_agent_temperature}")
    pass # No action needed if agent is already correct


# Display the sidebar first (This is where the slider updates session state)
display_sidebar()

# Display chat messages from history
display_chat_messages()

# Handle user input using the restored function (which includes agent selection and chat input)
# Pass the agent_executor and the 'llm' object created earlier.
handle_user_input(st.session_state.agent_executor, llm) # Pass the 'llm' object directly


# --- PDF Ingestion Function (Main Knowledge Base) ---
# Moved this function here from scrape_and_ingest.py to be accessible by app.py
# It's better to have ingestion as a separate script, but for simplicity in this context,
# we'll keep the check/ingest logic here if it's intended to run on app startup.
# However, running ingestion on every app startup is inefficient.
# A better approach is to run scrape_and_ingest.py separately.
# For now, I will keep the function here but comment out the call at the end of the file.
# The call is moved to scrape_and_ingest.py's __main__ block.

# def check_and_ingest_new_pdfs(data_directory: str, vector_store_instance: LanceDB, db_path: str): # Changed type hint
#     """Checks for new PDFs in data_directory and ingests them into the main vector store."""
#     log_file_path = os.path.join(db_path, ".ingested_files.log")
#     ingested_files = set()

#     # Create data directory if it doesn't exist
#     if not os.path.exists(data_directory):
#         os.makedirs(data_directory)
#         print(f"Created data directory: {data_directory}") # Use print for console output

#     # Read list of already ingested files
#     try:
#         with open(log_file_path, "r") as f:
#             ingested_files = set(line.strip() for line in f)
#         print(f"Found log of {len(ingested_files)} previously ingested files.")
#     except FileNotFoundError:
#         print("No ingestion log file found for the main knowledge base. Will process all PDFs in data directory.")
#     except Exception as e:
#          print(f"Error reading ingestion log file: {e}")


#     # Find current PDF files
#     current_pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
#     current_filenames = set(os.path.basename(f) for f in current_pdf_files)

#     # Determine new files to ingest
#     new_filenames = current_filenames - ingested_files
#     new_file_paths = [os.path.join(data_directory, fname) for fname in new_filenames]

#     if not new_file_paths:
#         print("Main knowledge base is up-to-date. No new PDFs found to ingest.")
#         return

#     print(f"Found {len(new_file_paths)} new PDF file(s) to ingest into the main knowledge base. Starting process...")

#     all_new_docs = []
#     processed_new_files = []
#     for pdf_path in new_file_paths:
#         filename = os.path.basename(pdf_path)
#         try:
#             print(f"  Loading: {filename}")
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             all_new_docs.extend(documents)
#             processed_new_files.append(filename) # Track successfully loaded files
#         except Exception as e:
#             print(f"Error loading {filename}: {e}")
#             continue # Skip to the next file

#     if not all_new_docs:
#         print("No new documents were successfully loaded from the PDF files for the main knowledge base.")
#         return

#     print("Splitting new documents into chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(all_new_docs)

#     if not splits:
#         print("Failed to split new documents into chunks for the main knowledge base.")
#         return

#     print(f"Adding {len(splits)} new document chunks to the main vector store...")
#     try:
#         # Add the new documents to the vector store
#         # LanceDB's add_documents handles batching internally
#         vector_store_instance.add_documents(splits)

#         # Update the log file with newly processed files
#         with open(log_file_path, "a") as f:
#             for fname in processed_new_files:
#                 f.write(f"{fname}\n")
#         print(f"Successfully ingested {len(processed_new_files)} new PDF file(s) into the main knowledge base!")

#     except Exception as e:
#         print(f"Error adding new documents to main vector store: {e}")

# print("Checking for new documents to load into the main knowledge base...")
# check_and_ingest_new_pdfs(DATA_DIR, vector_store, LANCEDB_DB_PATH) # Pass the LanceDB instance and new path
