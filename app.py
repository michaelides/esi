import streamlit as st
import os
from dotenv import load_dotenv
import io # Import io for handling file streams
import tempfile # Import tempfile for creating temporary files
import shutil # Import shutil for directory cleanup
import uuid # Import uuid for generating unique IDs

# Import chromadb client
import chromadb

# Use Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma # Updated Chroma import
from langchain_community.tools import DuckDuckGoSearchRun # Using DuckDuckGo as a free alternative first
# from langchain_community.utilities import GoogleSerperAPIWrapper # Option for Google Search via Serper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool # Import Tool class
from langchain import hub # To pull prompts easily, e.g., for agent scratchpad
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough # Import for potential future use or clarity


# --- Configuration ---
load_dotenv()
DATA_DIR = "./data" # Directory to store PDF files

# Check for necessary API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Uncomment if using Google Serper

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in your .env file (GOOGLE_API_KEY=...).")
    st.stop()

# Configure Google API key for LangChain components
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Optional: Check for Serper key if using Google Search
# if not SERPER_API_KEY:
#     st.warning("SERPER API key not found. Google Search tool will be disabled. Set SERPER_API_KEY in .env if needed.")
#     google_search_tool = None
# else:
#     search = GoogleSerperAPIWrapper()
#     google_search_tool = Tool(
#         name="google_search",
#         func=search.run,
#         description="Useful for when you need to answer questions about current events or look up information on the internet.",
#     )

# Using DuckDuckGo Search as a free alternative
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="duckduckgo_search",
    func=search.run,
    description="Useful for when you need to answer questions about current events or look up information on the internet."
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
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create a RAG tool for the main knowledge base
rag_tool = create_retriever_tool(
    retriever,
    "dissertation_resource_retriever",
    "Searches and returns relevant information from loaded dissertation guides, research papers, and academic resources. Use this to find specific details, methodologies, or examples from the knowledge base.",
)

# --- LLM Setup ---
# Initialize the LLM (e.g., Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7) # Removed deprecated convert_system_message_to_human

# --- Agent Setup ---
# Define the base tools the agent can use (main knowledge base and search)
base_tools = [search_tool, rag_tool]

# Define the system message and prompt structure globally
system_message = """You are a helpful AI assistant designed to support university students with their dissertations. Your goal is to help them brainstorm research ideas, structure their work, understand methodologies, and overcome challenges.

Instructions for interacting with students:
1.  Be encouraging, patient, and constructive. Avoid overly critical language.
2.  Ask clarifying questions to understand the student's specific field, topic, and progress.
3.  When brainstorming, suggest diverse ideas but encourage the student to evaluate them based on feasibility, interest, and academic contribution.
4.  Use the 'dissertation_resource_retriever' tool to find relevant information from the knowledge base (e.g., specific writing guides, methodology examples, previously discussed concepts). Cite information retrieved using this tool.
5.  Use the search tool ('duckduckgo_search') to find recent research papers, news, or general information not present in the knowledge base. Cite information retrieved using this tool.
6.  Break down complex tasks into smaller, manageable steps.
7.  If unsure about a specific academic convention, advise the student to consult their supervisor or university guidelines.
8.  Maintain a conversational and supportive tone. Remember the student might be feeling overwhelmed.
9.  **IMPORTANT:** If the user has uploaded files and asks questions specifically about their content, use the 'uploaded_document_retriever' tool to answer those questions. Prioritize this tool for questions directly related to the uploaded documents. If the question is general or about the main knowledge base, use the other tools.
"""

# Define the prompt message structure
prompt_messages = [
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"), # Use MessagesPlaceholder
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Use MessagesPlaceholder
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


# --- Uploaded File Processing Functions ---

def process_uploaded_files(uploaded_files):
    """Processes uploaded files, creates a temporary vector store, and updates the agent."""
    if not uploaded_files:
        st.warning("No files uploaded.")
        return

    # Clear any previous uploaded file context before processing new ones
    clear_uploaded_files()

    st.session_state.processed_files = []
    all_uploaded_docs = []
    temp_dir = None # Initialize temp_dir outside the try block

    try:
        # Create a temporary directory to save uploaded files
        temp_dir = tempfile.mkdtemp()
        st.info(f"Processing {len(uploaded_files)} uploaded file(s)...")

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            st.write(f"  Loading: {filename}")

            # Save the uploaded file to the temporary directory
            temp_filepath = os.path.join(temp_dir, filename)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                # Use PyPDFLoader to load the saved file
                loader = PyPDFLoader(temp_filepath)
                documents = loader.load()
                all_uploaded_docs.extend(documents)
                st.session_state.processed_files.append(filename) # Track successfully loaded files
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
                # Continue processing other files even if one fails
                continue

        if not all_uploaded_docs:
            st.error("No documents were successfully loaded from the uploaded files.")
            # Clear processed files list if none were successful
            st.session_state.processed_files = []
            return

        st.write("Splitting uploaded documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_uploaded_docs)

        if not splits:
            st.error("Failed to split uploaded documents into chunks.")
            # Clear processed files list if splitting fails
            st.session_state.processed_files = []
            return

        st.write(f"Adding {len(splits)} document chunks to a temporary vector store...")

        # --- Explicitly create an in-memory Chroma client and add documents ---
        try:
            # Create an in-memory client - **Explicitly initialize with default settings**
            uploaded_client = chromadb.Client() # Keep it as default in-memory client
            # Get or create the collection
            uploaded_collection = uploaded_client.get_or_create_collection(name="uploaded_resources")

            # Prepare documents for adding
            docs_to_add = [split.page_content for split in splits]
            metadatas_to_add = [split.metadata for split in splits]
            ids_to_add = [str(uuid.uuid4()) for _ in splits] # Generate unique IDs

            # Add documents to the collection
            uploaded_collection.add(
                documents=docs_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )

            # Create the LangChain Chroma wrapper around the explicit client and collection
            uploaded_vector_store = Chroma(
                client=uploaded_client,
                collection_name="uploaded_resources",
                embedding_function=embedding_function # Use the same embedding function
            )

            # Store the temporary vector store and tool in session state
            st.session_state.uploaded_vector_store = uploaded_vector_store
            st.session_state.uploaded_retriever_tool = create_retriever_tool(
                uploaded_vector_store.as_retriever(search_kwargs={"k": 3}),
                "uploaded_document_retriever",
                "Searches and returns relevant information ONLY from the documents the user has uploaded during this session. Use this tool specifically when the user asks questions about the content of their uploaded files.",
            )

            # Update the agent executor to include the new tool
            update_agent_executor()

            st.success(f"Successfully processed {len(st.session_state.processed_files)} uploaded file(s)! You can now ask questions about their content.")

        except Exception as e:
            st.error(f"An error occurred during Chroma processing for uploaded files: {e}")
            # Ensure session state is clean if processing fails
            clear_uploaded_files()
            # Re-raise the exception to show the full traceback if needed for debugging
            # raise e # Uncomment for detailed debugging
        # --- End of explicit Chroma client creation ---


    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        # Ensure session state is clean if processing fails
        clear_uploaded_files()
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                # st.info(f"Cleaned up temporary directory: {temp_dir}") # Optional: keep this less verbose
            except Exception as e:
                st.warning(f"Could not clean up temporary directory {temp_dir}: {e}")


def clear_uploaded_files():
    """Clears the temporary vector store and removes the uploaded file tool from the agent."""
    st.info("Clearing uploaded files context...")
    if "uploaded_vector_store" in st.session_state and st.session_state.uploaded_vector_store:
        try:
            # If the vector store exists, try to get the client and delete the collection
            uploaded_vector_store = st.session_state.uploaded_vector_store
            if hasattr(uploaded_vector_store, '_client') and uploaded_vector_store._client:
                 try:
                     uploaded_vector_store._client.delete_collection(name="uploaded_resources")
                     st.info("Deleted 'uploaded_resources' collection from temporary client.")
                 except Exception as e:
                     st.warning(f"Could not delete 'uploaded_resources' collection: {e}")

            # For in-memory Chroma, setting to None should be sufficient for garbage collection
            st.session_state.uploaded_vector_store = None
        except Exception as e:
            st.warning(f"Could not clear temporary vector store reference: {e}")

    st.session_state.uploaded_retriever_tool = None
    st.session_state.processed_files = []

    # Update the agent executor to remove the uploaded file tool
    update_agent_executor()

    st.success("Uploaded files context cleared.")


def update_agent_executor():
    """Recreates the agent executor with the current set of tools."""
    current_tools = list(base_tools) # Start with the base tools
    if "uploaded_retriever_tool" in st.session_state and st.session_state.uploaded_retriever_tool:
        current_tools.append(st.session_state.uploaded_retriever_tool)
        st.info(f"Agent tools updated: {', '.join([tool.name for tool in current_tools])}")
    else:
         st.info(f"Agent tools: {', '.join([tool.name for tool in current_tools])}")

    # Retrieve the prompt from session state
    agent_prompt = st.session_state.agent_prompt

    # Create and store the new agent executor instance
    st.session_state.agent_executor = AgentExecutor(
        agent=create_tool_calling_agent(llm, current_tools, agent_prompt), # Use prompt from session state
        tools=current_tools,
        verbose=True # Set verbose=True for debugging
    )


# --- Streamlit UI ---
st.title("🎓 ESI: ESI Scholarly Instructor")
st.caption("Your AI partner for brainstorming and structuring your research.")

# --- Automatic PDF Ingestion on Startup (Main Knowledge Base) ---
# Check for new PDFs and ingest them right after vector store is initialized
with st.spinner("Checking for new documents to load into the main knowledge base..."):
    check_and_ingest_new_pdfs(DATA_DIR, vector_store, CHROMA_DB_PATH)

# --- Session State Initialization ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting from AI
    st.session_state.messages.append(
        AIMessage(content="Hello! I'm here to help you with your dissertation. How can I assist you today? Feel free to ask about brainstorming ideas, structuring chapters, finding resources, or anything else!")
    )

# Initialize session state for uploaded files and agent executor
if "uploaded_vector_store" not in st.session_state:
    st.session_state.uploaded_vector_store = None
if "uploaded_retriever_tool" not in st.session_state:
    st.session_state.uploaded_retriever_tool = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Initialize the agent prompt in session state
if "agent_prompt" not in st.session_state:
     st.session_state.agent_prompt = ChatPromptTemplate.from_messages(prompt_messages)


if "agent_executor" not in st.session_state:
    # Initialize the agent executor with base tools on first run
    update_agent_executor()


# Display chat messages from history
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

# React to user input
if prompt := st.chat_input("What's on your mind regarding your dissertation?"):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response using the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the agent_executor from session state (might include uploaded file tool)
            current_agent_executor = st.session_state.agent_executor
            try:
                response = current_agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages[:-1] # Pass history excluding the current user prompt
                })
                ai_response_content = response.get("output", "Sorry, I encountered an error getting the response.")
            except Exception as e:
                 ai_response_content = f"An error occurred while processing your request: {e}"
                 st.error(ai_response_content) # Display error in UI as well

            st.markdown(ai_response_content)

    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(content=ai_response_content))

# Add a sidebar for potential future options or info
with st.sidebar:
    st.header("About")
    st.info("""ESI uses AI to help you navigate the dissertation process.
    It has access to some of the literature in your reading lists and also uses Search tools for web lookups.""")
    st.warning("⚠️ Remember: Always consult your official supervisor for final guidance and decisions.")

    st.divider()
    st.header("Discuss Uploaded Files")
    uploaded_files = st.file_uploader(
        "Upload PDF files to discuss (temporary for this session)",
        accept_multiple_files=True,
        type=['pdf'] # Initially restrict to PDF
    )

    # Process files button
    # Only show process button if new files are selected
    if uploaded_files:
        if st.button("Process Uploaded Files", key="process_files_button"):
             process_uploaded_files(uploaded_files)

    # Display processed files and clear button only if files have been processed
    if st.session_state.processed_files:
        st.caption("Currently discussing:")
        for fname in st.session_state.processed_files:
            st.write(f"- {fname}")
        if st.button("Clear Uploaded Files Context", key="clear_files_button"):
            clear_uploaded_files()
            # Rerun the app to clear the display immediately
            st.rerun() # Use st.rerun() to refresh the UI state
