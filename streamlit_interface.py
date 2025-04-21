import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

def initialize_streamlit():
    """Initializes the Streamlit UI, including title, caption, and session state."""
    st.title("🎓 ESI: ESI Scholarly Instructor")
    st.caption("Your AI partner for brainstorming and structuring your research.")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial greeting from AI
        st.session_state.messages.append(
            AIMessage(content="Hello! I'm here to help you with your dissertation. How can I assist you today? Feel free to ask about brainstorming ideas, structuring chapters, finding resources, or anything else!")
        )

    # Initialize the agent prompt in session state
    if "agent_prompt" not in st.session_state:
        # Load the system prompt as instructions:
        try:
            with open('esi_agent_instruction.md', 'r') as f:
                instruction = f.read()
        except FileNotFoundError:
            st.error("Could not find esi_agent_instruction.md. Please ensure it is in the correct directory.")
            instruction = ""  # Provide a default value to avoid errors

        system_message = f"""{instruction}
        You are a helpful AI assistant designed to support university students with their dissertations.
        Your goal is to help them brainstorm research ideas, structure their work, understand methodologies, and overcome challenges.

        When you use tools, ALWAYS cite the source URL if one is provided.

        **Tool Use Instructions:**

        1.  When a student refers to the \"module\" they are referring to the MSc dissertation module at UEA, called NBS-7091A. You have access to information about this module via the `dissertation_resource_retriever` tool.
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
        st.session_state.agent_prompt = ChatPromptTemplate.from_messages(prompt_messages)


def display_chat_messages():
    """Displays chat messages from history."""
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

def handle_user_input(agent_executor):
    """Handles user input and generates AI response."""
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
                current_agent_executor = agent_executor
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

def display_sidebar():
    """Displays the sidebar with information about the app."""
    with st.sidebar:
        st.header("About")
        st.info("""ESI uses AI to help you navigate the dissertation process.
        It has access to some of the literature in your reading lists and also uses Search tools for web lookups.""")
        st.warning("⚠️ Remember: Always consult your official supervisor for final guidance and decisions.")
