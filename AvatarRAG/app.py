# streamlit_enterprise_ui.py
# Enterprise-Grade HR Handbook Chatbot UI
# Modern, Smart, User-Focused Design
#
# Run with: streamlit run streamlit_enterprise_ui.py

import streamlit as st
import time
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import json

# Import from the optimized chatbot
from chatbot import (
    PDF_DIR, MODELS_DIR,
    Chunk, SentenceTransformer, Reranker, QueryCache,
    build_or_load, retrieve, build_context_block,
    answer_question, find_gguf_model, make_llm,
    delete_index_files, EMBED_MODEL_NAME, RERANKER_MODEL_NAME, USE_RERANKER
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="HR Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ASSETS_DIR = Path(__file__).parent
BOT_AVATAR = "image/bot.gif"
# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Chat container */
    .stApp {
        background: transparent;
    }
    
    /* Custom header */
    .custom-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Message bubbles */
    .stChatMessage {
        background: transparent;
        padding: 1rem 0;
    }
    
    /* User message styling */
    div[data-testid="stChatMessageContent"] {
        padding: 1rem 1.5rem;
        border-radius: 20px;
    }
    
    /* Input area */
    .stChatInputContainer {
        background: white;
        border-radius: 24px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        max-width: 900px;
        margin: 1rem auto;
    }
    
    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .source-header {
        font-weight: 600;
        color: #334155;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .source-text {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 12px;
        font-weight: 600;
        color: #334155;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        padding: 2rem 1rem;
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Response time badge */
    .response-badge {
        display: inline-block;
        background: #ecfdf5;
        color: #059669;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-text {
        color: #64748b;
        font-size: 1rem;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        background: #94a3b8;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'initialized': False,
        'embed_model': None,
        'faiss_index': None,
        'chunks': None,
        'bm25': None,
        'reranker': None,
        'llm': None,
        'query_cache': None,
        'chat_history': [],
        'messages': [],
        'conversations': [],
        'current_conversation_id': None,
        'show_sources': True,
        'total_queries': 0,
        'last_sources': [],
        'suggested_questions': [
            "What is our PTO policy?",
            "How do I enroll in health insurance?",
            "What are the 401(k) contribution limits?",
            "Tell me about parental leave benefits",
            "What is the remote work policy?"
        ]
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all models with progress tracking"""
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    faiss_index, chunks, bm25 = build_or_load(embed_model)
    reranker = Reranker(RERANKER_MODEL_NAME) if USE_RERANKER else None
    model_path = find_gguf_model(MODELS_DIR)
    llm = make_llm(model_path)
    query_cache = QueryCache(max_size=100)
    
    return {
        'embed_model': embed_model,
        'faiss_index': faiss_index,
        'chunks': chunks,
        'bm25': bm25,
        'reranker': reranker,
        'llm': llm,
        'query_cache': query_cache,
        'model_name': model_path.name
    }

def initialize_models():
    """Initialize models with beautiful loading screen"""
    if st.session_state.initialized:
        return
    
    loading_container = st.empty()
    
    with loading_container.container():
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🚀</div>
            <h2 style='color: #334155; margin-bottom: 1rem;'>Initializing AI Assistant</h2>
            <p style='color: #64748b;'>Setting up your intelligent HR companion...</p>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            status.markdown("**Loading AI models...**")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            status.markdown("**Building knowledge base...**")
            progress_bar.progress(50)
            
            models = load_all_models()
            
            status.markdown("**Optimizing search...**")
            progress_bar.progress(75)
            time.sleep(0.2)
            
            for key, value in models.items():
                st.session_state[key] = value
            
            status.markdown("**Ready! ✅**")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            st.stop()
    
    loading_container.empty()

# ==================== HELPER FUNCTIONS ====================
def format_single_source_html(chunk: Chunk, index: int) -> str:
    """Format a single source as HTML"""
    excerpt = chunk.text.replace("\n", " ")
    
    html = f"""
    <div class="source-card">
        <div class="source-header">
            📄 Source {index}: {chunk.source_file} · Page {chunk.page}
        </div>
        <div class="source-text">{excerpt}</div>
    </div>
    """
    return html

def get_conversation_title(message: str) -> str:
    """Generate a short title from the first message"""
    return message[:50] + ("..." if len(message) > 50 else "")

def save_conversation():
    """Save current conversation"""
    if st.session_state.messages:
        conversation = {
            'id': datetime.now().isoformat(),
            'title': get_conversation_title(st.session_state.messages[0]['content']),
            'timestamp': datetime.now().strftime("%b %d, %I:%M %p"),
            'messages': st.session_state.messages.copy(),
            'chat_history': st.session_state.chat_history.copy()
        }
        st.session_state.conversations.insert(0, conversation)
        st.session_state.conversations = st.session_state.conversations[:20]

def load_conversation(conv_id: str):
    """Load a saved conversation"""
    for conv in st.session_state.conversations:
        if conv['id'] == conv_id:
            st.session_state.messages = conv['messages'].copy()
            st.session_state.chat_history = conv['chat_history'].copy()
            st.session_state.current_conversation_id = conv_id
            st.rerun()

def new_conversation():
    """Start a new conversation"""
    if st.session_state.messages:
        save_conversation()
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.session_state.current_conversation_id = None
    st.rerun()

def get_answer(question: str) -> Tuple[str, float, List[Chunk]]:
    """Get answer from chatbot"""
    start_time = time.time()
    
    selected = retrieve(
        question,
        st.session_state.embed_model,
        st.session_state.faiss_index,
        st.session_state.chunks,
        st.session_state.bm25,
        st.session_state.reranker,
        st.session_state.query_cache
    )
    
    context = build_context_block(selected)
    answer = answer_question(
        st.session_state.llm,
        question,
        context,
        st.session_state.chat_history
    )
    
    elapsed = time.time() - start_time
    return answer, elapsed, selected

# ==================== MAIN APP ====================
def main():
    init_session_state()
    
    # Check for PDFs
    pdf_files = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    
    if not pdf_files:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📁</div>
            <div class="empty-state-title">No Documents Found</div>
            <div class="empty-state-text">
                Please add PDF files to the <code>./handbooks/</code> folder
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Initialize models
    if not st.session_state.initialized:
        initialize_models()
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="custom-header">
        <div class="header-title">
            💼 HR Assistant
        </div>
        <div class="header-subtitle">
            Your intelligent companion for HR policies and benefits
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown('<div class="sidebar-header">💬 Conversations</div>', unsafe_allow_html=True)
        
        if st.button("➕ New Conversation", use_container_width=True):
            new_conversation()
        
        st.markdown("---")
        
        if st.session_state.conversations:
            for conv in st.session_state.conversations:
                is_active = conv['id'] == st.session_state.current_conversation_id
                
                if st.button(
                    f"{'🟢 ' if is_active else ''}**{conv['title']}**\n{conv['timestamp']}",
                    key=conv['id'],
                    use_container_width=True
                ):
                    load_conversation(conv['id'])
        else:
            st.markdown("""
            <div style='text-align: center; padding: 2rem 0; color: #94a3b8;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>💭</div>
                <div style='font-size: 0.9rem;'>No conversations yet</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;'>SESSION STATS</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{st.session_state.total_queries}</div>
            <div style='font-size: 0.8rem; color: #64748b;'>Total Queries</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.session_state.show_sources = st.checkbox(
                "Show sources",
                value=st.session_state.show_sources
            )
            
            if st.button("🔄 Rebuild Knowledge Base"):
                with st.spinner("Rebuilding..."):
                    delete_index_files()
                    st.cache_resource.clear()
                    st.session_state.initialized = False
                    st.rerun()
    
    # ==================== MAIN CHAT AREA ====================
    
    # Empty state with suggested questions
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">👋</div>
            <div class="empty-state-title">Welcome to HR Assistant</div>
            <div class="empty-state-text">
                Ask me anything about company policies, benefits, and procedures
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Try asking:")
        cols = st.columns(2)
        
        for idx, question in enumerate(st.session_state.suggested_questions):
            col = cols[idx % 2]
            with col:
                if st.button(question, key=f"suggest_{idx}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()
    
    # Display all chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else BOT_AVATAR):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant":
                if "time" in message:
                    st.markdown(f'<span class="response-badge">⚡ {message["time"]:.1f}s</span>', unsafe_allow_html=True)
                
                if st.session_state.show_sources and "sources" in message and message["sources"]:
                    with st.expander(f"📚 View {len(message['sources'])} Sources", expanded=False):
                        # Display each source in its own nested expander
                        for source_idx, source in enumerate(message["sources"], 1):
                            source_title = f"📄 {source.source_file} - Page {source.page}"
                            with st.expander(source_title, expanded=False):
                                st.markdown(format_single_source_html(source, source_idx), unsafe_allow_html=True)
    
    # Handle pending question from suggestions
    if hasattr(st.session_state, 'pending_question'):
        prompt = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
        
        # Process the question
        process_user_query(prompt)
    
    # Chat input
    if prompt := st.chat_input("Ask me about HR policies, benefits, procedures...", key="chat_input"):
        process_user_query(prompt)

def process_user_query(prompt: str):
    """Process user query and generate response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar= BOT_AVATAR): # type: ignore
        # Typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            answer, elapsed, selected = get_answer(prompt)
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Display answer
            st.markdown(answer)
            st.markdown(f'<span class="response-badge">⚡ {elapsed:.1f}s</span>', unsafe_allow_html=True)
            
            # Show sources with nested expanders
            if st.session_state.show_sources and selected:
                with st.expander(f"📚 View {len(selected)} Sources", expanded=False):
                    for source_idx, source in enumerate(selected, 1):
                        source_title = f"📄 {source.source_file} - Page {source.page}"
                        with st.expander(source_title, expanded=False):
                            st.markdown(format_single_source_html(source, source_idx), unsafe_allow_html=True)
            
            # Update state
            st.session_state.chat_history.append((prompt, answer))
            st.session_state.last_sources = selected
            st.session_state.total_queries += 1
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": selected,
                "time": elapsed
            })
            
            # Save conversation
            if st.session_state.current_conversation_id is None:
                st.session_state.current_conversation_id = datetime.now().isoformat()
            
        except Exception as e:
            typing_placeholder.empty()
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()