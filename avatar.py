import streamlit as st
import requests
import time
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="RAG AI Assistant with Lip Sync Avatar",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for avatar animation
st.markdown("""
<style>
    .avatar-container {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }
    
    .avatar-face {
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        position: relative;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .avatar-eyes {
        position: absolute;
        top: 60px;
        width: 100%;
        display: flex;
        justify-content: space-around;
        padding: 0 40px;
    }
    
    .eye {
        width: 20px;
        height: 20px;
        background: white;
        border-radius: 50%;
        position: relative;
    }
    
    .pupil {
        width: 10px;
        height: 10px;
        background: #333;
        border-radius: 50%;
        position: absolute;
        top: 5px;
        left: 5px;
    }
    
    .avatar-mouth {
        position: absolute;
        bottom: 50px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 30px;
        border: 3px solid white;
        border-top: none;
        border-radius: 0 0 60px 60px;
        transition: all 0.1s ease;
    }
    
    .mouth-open {
        height: 40px !important;
        border-radius: 0 0 80px 80px !important;
    }
    
    .mouth-closed {
        height: 5px !important;
        border-radius: 0 0 30px 30px !important;
    }
    
    @keyframes blink {
        0%, 100% { height: 20px; }
        50% { height: 2px; }
    }
    
    .blinking {
        animation: blink 3s infinite;
    }
    
    .text-display {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        min-height: 100px;
        font-size: 18px;
        line-height: 1.6;
        margin-top: 20px;
    }
    
    .speaking-indicator {
        width: 20px;
        height: 20px;
        background: #4CAF50;
        border-radius: 50%;
        display: inline-block;
        margin-left: 10px;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time.time())
if "is_speaking" not in st.session_state:
    st.session_state.is_speaking = False

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    with st.expander("Server Settings", expanded=True):
        api_base_url = st.text_input("API Base URL", value="http://localhost:7777")
        
    try:
        models_response = requests.get(f"{api_base_url}/models", timeout=2)
        if models_response.status_code == 200:
            models_data = models_response.json()
            providers = list(models_data.get("providers", {}).keys())
            
            selected_provider = st.selectbox(
                "LLM Provider", 
                providers, 
                index=providers.index("groq") if "groq" in providers else 0
            )
            
            provider_models = models_data["providers"][selected_provider]["models"]
            selected_model = st.selectbox("Model", provider_models)
            
            chain_types = list(models_data.get("chain_types", {}).keys())
            selected_chain = st.selectbox(
                "Chain Type", 
                chain_types, 
                index=chain_types.index("advanced") if "advanced" in chain_types else 0
            )
        else:
            st.error("Could not fetch models from server")
            selected_provider = "groq"
            selected_model = "llama-3.3-70b-versatile"
            selected_chain = "advanced"
    except Exception as e:
        st.warning(f"Using default settings (server not available)")
        selected_provider = "groq"
        selected_model = "llama-3.3-70b-versatile"
        selected_chain = "advanced"

    with st.expander("Advanced Parameters", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        use_reranking = st.toggle("Use Reranking", value=True)
        use_query_expansion = st.toggle("Use Query Expansion", value=True)
        num_variations = st.slider("Query Variations", 1, 5, 2, disabled=not use_query_expansion)
        top_k = st.slider("Top K Documents", 1, 20, 5)
        
    with st.expander("Avatar Settings", expanded=True):
        words_per_mouth_movement = st.slider("Words per mouth movement", 1, 5, 2)
        mouth_open_duration = st.slider("Mouth open duration (ms)", 50, 300, 100)

# Main Content
st.title("🤖 RAG Assistant with Lip Sync Avatar")

# Create two columns: avatar on left, chat on right
col_avatar, col_chat = st.columns([1, 2])

with col_avatar:
    st.subheader("AI Avatar")
    avatar_placeholder = st.empty()
    
    # Initial avatar state
    speaking_status = "Speaking..." if st.session_state.is_speaking else "Idle"
    status_color = "#4CAF50" if st.session_state.is_speaking else "#999"
    
    avatar_placeholder.markdown(f"""
    <div class="avatar-container">
        <div class="avatar-face">
            <div class="avatar-eyes">
                <div class="eye blinking">
                    <div class="pupil"></div>
                </div>
                <div class="eye blinking">
                    <div class="pupil"></div>
                </div>
            </div>
            <div class="avatar-mouth" id="mouth"></div>
        </div>
    </div>
    <div style="text-align: center; margin-top: 10px; color: {status_color};">
        <strong>{speaking_status}</strong>
        {f'<span class="speaking-indicator"></span>' if st.session_state.is_speaking else ''}
    </div>
    """, unsafe_allow_html=True)

with col_chat:
    st.subheader("Conversation")
    
    # Control buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        enable_streaming = st.toggle("Enable Streaming", value=True)
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources & Metadata"):
                        st.json(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response with lip sync
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
                "stream": enable_streaming,
                "include_confidence": False
            }

            try:
                if enable_streaming:
                    st.session_state.is_speaking = True
                    
                    # Streaming with lip sync
                    response = requests.post(
                        f"{api_base_url}/query/stream", 
                        json=payload, 
                        stream=True,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        word_count = 0
                        mouth_state = "closed"
                        
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    chunk = line[6:]
                                    if chunk.startswith("ERROR:"):
                                        st.error(chunk)
                                        break
                                    
                                    full_response += chunk
                                    message_placeholder.markdown(full_response + "▌")
                                    
                                    # Lip sync logic: toggle mouth every N words
                                    words = chunk.split()
                                    word_count += len(words)
                                    
                                    if word_count >= words_per_mouth_movement:
                                        mouth_state = "open" if mouth_state == "closed" else "closed"
                                        word_count = 0
                                        
                                        # Update avatar mouth
                                        with col_avatar:
                                            avatar_placeholder.markdown(f"""
                                            <div class="avatar-container">
                                                <div class="avatar-face">
                                                    <div class="avatar-eyes">
                                                        <div class="eye blinking">
                                                            <div class="pupil"></div>
                                                        </div>
                                                        <div class="eye blinking">
                                                            <div class="pupil"></div>
                                                        </div>
                                                    </div>
                                                    <div class="avatar-mouth mouth-{mouth_state}"></div>
                                                </div>
                                            </div>
                                            <div style="text-align: center; margin-top: 10px; color: #4CAF50;">
                                                <strong>Speaking...</strong>
                                                <span class="speaking-indicator"></span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        time.sleep(mouth_open_duration / 1000)
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.is_speaking = False
                        
                        # Reset avatar to idle
                        with col_avatar:
                            avatar_placeholder.markdown(f"""
                            <div class="avatar-container">
                                <div class="avatar-face">
                                    <div class="avatar-eyes">
                                        <div class="eye blinking">
                                            <div class="pupil"></div>
                                        </div>
                                        <div class="eye blinking">
                                            <div class="pupil"></div>
                                        </div>
                                    </div>
                                    <div class="avatar-mouth"></div>
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 10px; color: #999;">
                                <strong>Idle</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        st.session_state.is_speaking = False
                else:
                    # Non-streaming request
                    st.session_state.is_speaking = True
                    
                    with st.spinner("Thinking..."):
                        response = requests.post(f"{api_base_url}/query", json=payload, timeout=60)
                        
                    if response.status_code == 200:
                        data = response.json()
                        full_response = data["answer"]
                        sources = data.get("sources", [])
                        
                        # Simulate lip sync for non-streaming
                        words = full_response.split()
                        displayed_text = ""
                        word_count = 0
                        mouth_state = "closed"
                        
                        for word in words:
                            displayed_text += word + " "
                            word_count += 1
                            message_placeholder.markdown(displayed_text + "▌")
                            
                            if word_count >= words_per_mouth_movement:
                                mouth_state = "open" if mouth_state == "closed" else "closed"
                                word_count = 0
                                
                                # Update avatar
                                with col_avatar:
                                    avatar_placeholder.markdown(f"""
                                    <div class="avatar-container">
                                        <div class="avatar-face">
                                            <div class="avatar-eyes">
                                                <div class="eye blinking">
                                                    <div class="pupil"></div>
                                                </div>
                                                <div class="eye blinking">
                                                    <div class="pupil"></div>
                                                </div>
                                            </div>
                                            <div class="avatar-mouth mouth-{mouth_state}"></div>
                                        </div>
                                    </div>
                                    <div style="text-align: center; margin-top: 10px; color: #4CAF50;">
                                        <strong>Speaking...</strong>
                                        <span class="speaking-indicator"></span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                time.sleep(mouth_open_duration / 1000)
                            
                            time.sleep(0.05)  # Small delay between words
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.is_speaking = False
                        
                        # Reset avatar
                        with col_avatar:
                            avatar_placeholder.markdown(f"""
                            <div class="avatar-container">
                                <div class="avatar-face">
                                    <div class="avatar-eyes">
                                        <div class="eye blinking">
                                            <div class="pupil"></div>
                                        </div>
                                        <div class="eye blinking">
                                            <div class="pupil"></div>
                                        </div>
                                    </div>
                                    <div class="avatar-mouth"></div>
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 10px; color: #999;">
                                <strong>Idle</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if sources:
                            with st.expander("📚 Sources & Metadata"):
                                st.json(sources)
                                
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response,
                            "sources": sources
                        })
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        st.session_state.is_speaking = False
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.is_speaking = False

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        💡 The avatar's mouth moves in sync with the text generation<br>
        Adjust the "Words per mouth movement" slider in the sidebar to control lip sync speed
    </div>
    """, 
    unsafe_allow_html=True
)