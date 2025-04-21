import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun # Using DuckDuckGo as a free alternative first
# from langchain_community.utilities import GoogleSerperAPIWrapper # Option for Google Search via Serper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool # Import Tool class
from langchain import hub # To pull prompts easily, e.g., for agent scratchpad

# --- Configuration ---
load_dotenv()

# Check for necessary API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Uncomment if using Google Serper

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in your .env file (OPENAI_API_KEY=sk-...).")
    st.stop()

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


# --- RAG Setup ---
# Define the path for the persistent ChromaDB database
CHROMA_DB_PATH = "./chroma_db_dissertation"
# Use OpenAI embeddings (or choose another embedding model)
embedding_function = OpenAIEmbeddings()

# Initialize ChromaDB client
# This will create the directory if it doesn't exist.
# In a real scenario, you'd have a separate script to ingest documents.
vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_function,
    collection_name="dissertation_resources" # You can name your collection
)

# Create a retriever from the vector store
# `k=3` means it will retrieve the top 3 most relevant documents
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create a RAG tool
# NOTE: Adjust the description to be informative for the agent.
rag_tool = create_retriever_tool(
    retriever,
    "dissertation_resource_retriever",
    "Searches and returns relevant information from loaded dissertation guides, research papers, and academic resources. Use this to find specific details, methodologies, or examples from the knowledge base.",
)

# --- LLM and Agent Setup ---
# Initialize the LLM (e.g., GPT-4o)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)

# Define the tools the agent can use
tools = [search_tool, rag_tool]
# Filter out None tools if any API keys were missing
# tools = [tool for tool in [google_search_tool, rag_tool] if tool is not None]


# Define the custom system prompt
# Using the newer `create_tool_calling_agent` which requires a specific prompt structure
# You can pull a base prompt and customize it, or create one from scratch.
# Let's pull a base prompt suitable for tool calling agents
prompt = hub.pull("hwchase17/openai-tools-agent")

# Customize the system message part of the prompt template
# The base prompt structure is usually [SystemMessage, MessagesPlaceholder, AIMessage]
# We find the SystemMessage and modify its content.
# This assumes the first message in the template is the system message.
if prompt.messages and hasattr(prompt.messages[0], 'prompt'):
     # Accessing the template string within the PromptTemplate of the SystemMessagePromptTemplate
     original_system_template = prompt.messages[0].prompt.template
     custom_system_message = f"""You are a helpful AI assistant designed to support university students with their dissertations. Your goal is to help them brainstorm research ideas, structure their work, understand methodologies, and overcome challenges.

     Instructions for interacting with students:
     1.  Be encouraging, patient, and constructive. Avoid overly critical language.
     2.  Ask clarifying questions to understand the student's specific field, topic, and progress.
     3.  When brainstorming, suggest diverse ideas but encourage the student to evaluate them based on feasibility, interest, and academic contribution.
     4.  Use the 'dissertation_resource_retriever' tool to find relevant information from the knowledge base (e.g., specific writing guides, methodology examples, previously discussed concepts). Cite information retrieved using this tool.
     5.  Use the search tool ('duckduckgo_search') to find recent research papers, news, or general information not present in the knowledge base. Cite information retrieved using this tool.
     6.  Break down complex tasks into smaller, manageable steps.
     7.  If unsure about a specific academic convention, advise the student to consult their supervisor or university guidelines.
     8.  Maintain a conversational and supportive tone. Remember the student might be feeling overwhelmed.

     {original_system_template} # Include the original agent instructions for tool use etc.
     """
     prompt.messages[0].prompt.template = custom_system_message
else:
     st.warning("Could not customize the default agent prompt. Using the default.")


# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set verbose=True for debugging

# --- Streamlit UI ---
st.title("🎓 Dissertation Advisor Chatbot")
st.caption("Your AI partner for brainstorming and structuring your research.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting from AI
    st.session_state.messages.append(
        AIMessage(content="Hello! I'm here to help you with your dissertation. How can I assist you today? Feel free to ask about brainstorming ideas, structuring chapters, finding resources, or anything else!")
    )


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
            # We need to pass the chat history to the agent_executor
            # The format depends on the agent type. Tool-calling agents often expect 'chat_history' and 'input'.
            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": st.session_state.messages[:-1] # Pass history excluding the current user prompt
            })
            ai_response_content = response.get("output", "Sorry, I encountered an error.")
            st.markdown(ai_response_content)

    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(content=ai_response_content))

# Add a sidebar for potential future options or info
with st.sidebar:
    st.header("About")
    st.info("This chatbot uses AI to help you navigate the dissertation process. Use the RAG tool for internal resources and the Search tool for web lookups.")
    st.warning("⚠️ Remember: Always consult your official supervisor for final guidance and decisions.")

    # Placeholder for document upload/management in the future
    # st.header("Manage RAG Documents")
    # uploaded_file = st.file_uploader("Upload PDF or TXT documents", accept_multiple_files=True)
    # if uploaded_file:
    #     # Add logic here to process and ingest uploaded documents into ChromaDB
    #     st.success("Document processing placeholder.")
