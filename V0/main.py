import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/langchain-ai/langchain",
        "Report a bug": "https://github.com/langchain-ai/langchain/issues",
        "About": "Production RAG Interface v2.0",
    },
)

# ============================================================================
# CUSTOM CSS & THEME
# ============================================================================
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Styling */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #1e252b;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #2d3748;
    }
    
    [data-testid="stChatMessageContent"] {
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a202c;
        border-right: 1px solid #2d3748;
    }

    /* Buttons */
    .stButton button {
        background-color: #3182ce;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #2b6cb0;
        transform: translateY(-1px);
    }

    /* Inputs */
    .stTextInput input, .stSelectbox, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #4a5568;
        background-color: #2d3748;
        color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #63b3ed;
        font-weight: 700;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d3748;
        border-radius: 8px;
        color: #e2e8f0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1a202c; 
    }
    ::-webkit-scrollbar-thumb {
        background: #4a5568; 
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #718096; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time.time())

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Connection Settings
    with st.expander("🔌 Connection", expanded=False):
        api_base_url = st.text_input(
            "API URL", value="http://localhost:7777", help="Backend API address"
        )

    # Model Configuration
    st.markdown("#### Model Configuration")
    try:
        models_response = requests.get(f"{api_base_url}/models", timeout=2)
        if models_response.status_code == 200:
            models_data = models_response.json()
            providers = list(models_data.get("providers", {}).keys())

            selected_provider = st.selectbox(
                "Provider",
                providers,
                index=providers.index("groq") if "groq" in providers else 0,
                format_func=lambda x: x.upper(),
            )

            # Update models based on provider
            provider_models = models_data["providers"][selected_provider]["models"]
            selected_model = st.selectbox("Model", provider_models)

            chain_types = list(models_data.get("chain_types", {}).keys())
            selected_chain = st.selectbox(
                "Chain Strategy",
                chain_types,
                index=chain_types.index("advanced") if "advanced" in chain_types else 0,
                help="Advanced: Uses query expansion. Simple: Faster.",
            )
        else:
            st.warning("⚠️ Using offline defaults")
            selected_provider = "groq"
            selected_model = "llama-3.3-70b-versatile"
            selected_chain = "advanced"
    except Exception:
        # st.sidebar.error("Could not connect to backend")
        selected_provider = "groq"
        selected_model = "llama-3.3-70b-versatile"
        selected_chain = "advanced"

    st.divider()

    # RAG Parameters
    st.markdown("#### RAG Parameters")
    temperature = st.slider(
        "Creativity (Temperature)",
        0.0,
        1.0,
        0.1,
        help="Higher = more creative, Lower = more factual",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        use_reranking = st.toggle(
            "Reranking", value=True, help="Re-order results for relevance"
        )
    with col_b:
        use_query_expansion = st.toggle(
            "Expansion", value=True, help="Generate query variations"
        )

    if use_query_expansion:
        num_variations = st.slider(
            "Variations", 1, 5, 2, help="Number of query variations"
        )
    else:
        num_variations = 0

    top_k = st.slider("Context Limit (Top K)", 1, 20, 5)
    include_confidence = st.toggle("Show Confidence", value=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("✨ RAG AI Assistant")
st.caption(f"Powered by {selected_provider.upper()} • {selected_model}")

# Tabs for different modes
tab_chat, tab_ingest, tab_batch, tab_system = st.tabs(
    ["💬 Chat", "📄 Knowledge Base", "🔄 Batch Process", "📊 Status"]
)

# --- CHAT TAB ---
with tab_chat:
    # Message Container
    chat_container = st.container()

    # Display Chat History
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style='text-align: center; color: #718096; padding: 2rem;'>
                    <h3>👋 Welcome!</h3>
                    <p>Ask me anything about your documents.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Metadata display
                if message.get("role") == "assistant":
                    meta_cols = st.columns([1, 1, 1])
                    if message.get("confidence"):
                        with meta_cols[0]:
                            st.caption(
                                f"🎯 Confidence: {message['confidence']['confidence']:.2f}"
                            )
                    if message.get("latency"):
                        with meta_cols[1]:
                            st.caption(f"⚡ {message['latency']}ms")

                    if message.get("sources"):
                        with st.expander("📚 Analyzed Documents"):
                            st.json(message["sources"])

                    if message.get("variations"):
                        with st.expander("🔍 Query Variations"):
                            st.write(message["variations"])

    # Chat Input
    if prompt := st.chat_input("What would you like to know?"):
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Assistant Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            payload = {
                "question": prompt,
                "llm_provider": selected_provider,
                "model": selected_model,
                "temperature": temperature,
                "use_reranking": use_reranking,
                "use_query_expansion": use_query_expansion,
                "num_query_variations": num_variations,
                "chain_type": selected_chain,
                "top_k": top_k,
                "stream": True,  # Default to streaming
                "include_confidence": include_confidence,
            }

            try:
                # Streaming Output
                response = requests.post(
                    f"{api_base_url}/query/stream",
                    json=payload,
                    stream=True,
                    timeout=60,
                )

                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                chunk = decoded_line[6:]
                                if chunk.startswith("ERROR:"):
                                    st.error(chunk)
                                    full_response = "I encountered an error."
                                    break
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # Store in history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().isoformat(),
                            # Note: Sources are tricky with streaming unless we do a separate fetch or improved protocol
                        }
                    )

                else:
                    st.error(f"Server Error: {response.status_code}")

            except Exception as e:
                st.error(f"Connection Error: {str(e)}")


# --- INGEST TAB ---
with tab_ingest:
    st.markdown("### 📥 Add Knowledge")
    st.markdown("Upload documents to expand the AI's knowledge base.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "md"])

    with st.expander("Advanced Ingestion Settings"):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size", 100, 4000, 1000)
            chunk_strategy = st.selectbox(
                "Strategy", ["recursive", "semantic", "parent_child"]
            )
        with col2:
            chunk_overlap = st.number_input("Overlap", 0, 1000, 200)
            force_reingest = st.checkbox("Force Re-process", value=False)

    if uploaded_file and st.button("🚀 Process Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                os.makedirs("temp_uploads", exist_ok=True)
                file_path = os.path.join("temp_uploads", uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                payload = {
                    "file_path": os.path.abspath(file_path),
                    "chunking_strategy": chunk_strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "force_reingest": force_reingest,
                }

                response = requests.post(f"{api_base_url}/ingest", json=payload)
                if response.status_code == 200:
                    st.success(f"Successfully processed: {uploaded_file.name}")
                    st.json(response.json())
                else:
                    st.error(f"Processing failed: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- BATCH TAB ---
with tab_batch:
    st.markdown("### 🔄 Bulk Processing")

    batch_input = st.text_area(
        "Questions (one per line)",
        height=200,
        placeholder="What is the revenue?\nWhat are the risks?\nWho is the CEO?",
    )

    if st.button("Run Batch Analysis", type="primary"):
        questions = [q.strip() for q in batch_input.split("\n") if q.strip()]

        if not questions:
            st.warning("Please enter at least one question.")
        else:
            with st.status("Processing Batch...", expanded=True):
                st.write(f"Analyzing {len(questions)} queries...")
                payload = {
                    "questions": questions,
                    "llm_provider": selected_provider,
                    "chain_type": selected_chain,
                    "use_query_expansion": use_query_expansion,
                    "use_reranking": False,
                }

                try:
                    response = requests.post(
                        f"{api_base_url}/query/batch", json=payload
                    )
                    if response.status_code == 200:
                        st.write("✅ Complete!")
                        results = response.json().get("results", [])
                        df = pd.DataFrame(results)
                        st.dataframe(df)
                    else:
                        st.error("Batch failed")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- SYSTEM TAB ---
with tab_system:
    st.markdown("### 🖥️ System Status")

    if st.button("Refresh", key="refresh_system"):
        st.rerun()

    col1, col2, col3 = st.columns(3)

    try:
        stats = requests.get(f"{api_base_url}/stats", timeout=2).json()
        sys_stats = stats.get("system", {})

        with col1:
            st.metric("CPU Usage", f"{sys_stats.get('cpu_percent')}%")
        with col2:
            st.metric("Memory Usage", f"{sys_stats.get('memory_percent')}%")
        with col3:
            st.metric("Disk Usage", f"{sys_stats.get('disk_percent')}%")

        st.json(stats)
    except:
        st.error("Could not fetch system stats")
