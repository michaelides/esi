@st.cache_resource
def setup_global_llm_settings():
    """
    Sets up global LLM settings in the sidebar.
    """
    with st.sidebar:
        st.subheader("LLM Settings")
        # Add a slider for verbosity
        st.session_state.llm_verbosity = st.slider(
            "Verbosity Level",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Controls how verbose the LLM's responses are. Higher values mean more detailed responses."
        )
        # Placeholder for other settings if needed
        # st.session_state.llm_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        # st.session_state.llm_top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        # st.session_state.llm_max_tokens = st.slider("Max Tokens", min_value=100, max_value=4096, value=1024, step=100)
