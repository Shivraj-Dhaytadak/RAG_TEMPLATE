# streamlit_app.py
# Streamlit Frontend for HR Handbook Chatbot
#
# Run with: streamlit run streamlit_app.py

import streamlit as st
import time
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Import from the optimized chatbot
from chatbot import (
    BASE_DIR, PDF_DIR, MODELS_DIR, INDEX_DIR,
    FAISS_INDEX_PATH, META_PATH, BM25_PATH,
    EMBED_MODEL_NAME, RERANKER_MODEL_NAME, USE_RERANKER,
    MAX_SOURCE_EXCERPTS_TO_PRINT,
    Chunk, SentenceTransformer, Reranker,
    build_or_load, retrieve, build_context_block,
    answer_question, find_gguf_model, make_llm,
    delete_index_files
)

# Page config
st.set_page_config(
    page_title="HR Handbook Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .response-time {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.embed_model = None
        st.session_state.faiss_index = None
        st.session_state.chunks = None
        st.session_state.bm25 = None
        st.session_state.reranker = None
        st.session_state.llm = None
        st.session_state.chat_history = []
        st.session_state.last_selected = []
        st.session_state.messages = []
        st.session_state.loading_complete = False

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the embedding model"""
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_reranker_model():
    """Load and cache the reranker model"""
    if USE_RERANKER:
        return Reranker(RERANKER_MODEL_NAME)
    return None

@st.cache_resource(show_spinner=False)
def load_llm_model():
    """Load and cache the LLM"""
    model_path = find_gguf_model(MODELS_DIR)
    return make_llm(model_path), model_path.name

def initialize_models():
    """Initialize all models and indices"""
    if st.session_state.initialized:
        return
    
    with st.spinner("🔄 Loading models and building index... This may take a minute on first run."):
        try:
            # Load embedding model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading embedding model...")
            st.session_state.embed_model = load_embedding_model()
            progress_bar.progress(25)
            
            # Build or load index
            status_text.text("Building/loading vector index...")
            faiss_index, chunks, bm25 = build_or_load(st.session_state.embed_model)
            st.session_state.faiss_index = faiss_index
            st.session_state.chunks = chunks
            st.session_state.bm25 = bm25
            progress_bar.progress(50)
            
            # Load reranker
            status_text.text("Loading reranker model...")
            st.session_state.reranker = load_reranker_model()
            progress_bar.progress(75)
            
            # Load LLM
            status_text.text("Loading language model...")
            st.session_state.llm, model_name = load_llm_model()
            st.session_state.model_name = model_name
            progress_bar.progress(100)
            
            status_text.text("✅ All models loaded successfully!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.initialized = True
            st.session_state.loading_complete = True
            
        except Exception as e:
            st.error(f"❌ Error initializing models: {str(e)}")
            st.stop()

def format_sources(chunks: List[Chunk], max_display: int = 5) -> str:
    """Format source chunks as HTML"""
    if not chunks:
        return "<p>No sources available</p>"
    
    html = ""
    for i, chunk in enumerate(chunks[:max_display], 1):
        excerpt = chunk.text[:300].replace("\n", " ")
        if len(chunk.text) > 300:
            excerpt += "..."
        
        html += f"""
        <div class="source-box">
            <strong>Source {i}:</strong> {chunk.source_file} (Page {chunk.page})<br>
            <small>{excerpt}</small>
        </div>
        """
    
    return html

def get_answer(question: str) -> Tuple[str, float, List[Chunk]]:
    """Get answer from the chatbot"""
    start_time = time.time()
    
    # Retrieve relevant chunks
    selected = retrieve(
        question,
        st.session_state.embed_model,
        st.session_state.faiss_index,
        st.session_state.chunks,
        st.session_state.bm25,
        st.session_state.reranker
    )
    
    # Build context
    context = build_context_block(selected)
    
    # Generate answer
    answer = answer_question(
        st.session_state.llm,
        question,
        context,
        st.session_state.chat_history
    )
    
    elapsed = time.time() - start_time
    
    return answer, elapsed, selected

def main():
    """Main Streamlit app"""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📚 HR Handbook Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your company policies and benefits</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check if PDFs exist
        pdf_files = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
        
        st.subheader("📄 Documents")
        if pdf_files:
            for pdf in pdf_files:
                st.text(f"✓ {pdf.name}")
        else:
            st.warning("⚠️ No PDFs found in ./handbooks/")
            st.info("Add PDF files to the ./handbooks/ folder and restart the app.")
        
        st.divider()
        
        # Model info
        st.subheader("🤖 Model Info")
        if st.session_state.initialized:
            st.success(f"Model: {st.session_state.model_name}")
            st.info(f"Chunks indexed: {len(st.session_state.chunks)}")
        else:
            st.info("Models not loaded yet")
        
        st.divider()
        
        # Actions
        st.subheader("🔧 Actions")
        
        if st.button("🔄 Rebuild Index", help="Rebuild the search index from PDFs"):
            with st.spinner("Rebuilding index..."):
                delete_index_files()
                # Force reload by clearing cache
                st.cache_resource.clear()
                st.session_state.initialized = False
                st.rerun()
        
        if st.button("🗑️ Clear Chat History", help="Clear all messages"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.last_selected = []
            st.rerun()
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages) // 2)
        with col2:
            st.metric("Sources", len(st.session_state.last_selected))
        
        st.divider()
        
        # Show sources toggle
        show_sources = st.checkbox("Show sources with answers", value=True)
        
        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Type your question in the chat input
            2. Press Enter or click Send
            3. View the answer and sources
            
            **Example questions:**
            - What is the PTO policy?
            - Am I eligible for health insurance?
            - How do I enroll in 401k?
            - What are the parental leave benefits?
            """)
    
    # Main content area
    if not pdf_files:
        st.error("❌ No PDF documents found. Please add PDF files to the ./handbooks/ folder.")
        st.stop()
    
    # Initialize models
    if not st.session_state.initialized:
        initialize_models()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available and enabled
            if message["role"] == "assistant" and show_sources and "sources" in message:
                with st.expander("📚 View Sources"):
                    st.markdown(format_sources(message["sources"]), unsafe_allow_html=True)
            
            # Show response time
            if message["role"] == "assistant" and "time" in message:
                st.markdown(f'<p class="response-time">Response time: {message["time"]:.1f}s</p>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about HR policies...", key="chat_input"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, elapsed, selected = get_answer(prompt)
                    
                    # Store in history
                    st.session_state.chat_history.append((prompt, answer))
                    st.session_state.last_selected = selected
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show sources
                    if show_sources and selected:
                        with st.expander("📚 View Sources"):
                            st.markdown(format_sources(selected), unsafe_allow_html=True)
                    
                    # Show response time
                    st.markdown(f'<p class="response-time">Response time: {elapsed:.1f}s</p>', unsafe_allow_html=True)
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": selected,
                        "time": elapsed
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.caption("Powered by LLaMA-CPP & Streamlit")

if __name__ == "__main__":
    main()