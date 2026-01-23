import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",

    
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time.time())

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Server Settings
    with st.expander("Server Settings", expanded=True):
        api_base_url = st.text_input("API Base URL", value="http://localhost:7777")
        
    # Fetch available models/providers
    try:
        models_response = requests.get(f"{api_base_url}/models")
        if models_response.status_code == 200:
            models_data = models_response.json()
            providers = list(models_data.get("providers", {}).keys())
            
            selected_provider = st.selectbox("LLM Provider", providers, index=providers.index("groq") if "groq" in providers else 0)
            
            provider_models = models_data["providers"][selected_provider]["models"]
            selected_model = st.selectbox("Model", "meta-llama/llama-4-scout-17b-16e-instruct")
            
            chain_types = list(models_data.get("chain_types", {}).keys())
            selected_chain = st.selectbox("Chain Type", chain_types, index=chain_types.index("advanced") if "advanced" in chain_types else 0)
        else:
            st.error("Could not fetch models from server")
            selected_provider = "groq"
            selected_model = "llama-3.3-70b-versatile"
            selected_chain = "advanced"
    except Exception as e:
        st.error(f"Connection error: {e}")
        selected_provider = "groq"
        selected_model = "llama-3.3-70b-versatile"
        selected_chain = "advanced"

    # Advanced Settings
    with st.expander("Advanced Parameters", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        use_reranking = st.toggle("Use Reranking", value=True)
        use_query_expansion = st.toggle("Use Query Expansion", value=True)
        num_variations = st.slider("Query Variations", 1, 5, 2, disabled=not use_query_expansion)
        top_k = st.slider("Top K Documents", 1, 20, 5)
        include_confidence = st.toggle("Show Confidence Score", value=False)

# Main Content
st.title("🤖 Production RAG Assistant")

# Tabs
tab_chat, tab_ingest, tab_batch, tab_system = st.tabs(["💬 Chat", "📄 Ingest Documents", "🔄 Batch Query", "📊 System Health"])

# --- Chat Tab ---
with tab_chat:
    # Chat settings
    col1, col2 = st.columns([1, 1])
    with col1:
        enable_streaming = st.toggle("Enable Streaming", value=True)
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources & Metadata"):
                    st.json(message["sources"])
            if "confidence" in message and message["confidence"]:
                st.info(f"Confidence Score: {message['confidence']}")
            if "variations" in message and message["variations"]:
                st.caption(f"Query Variations: {', '.join(message['variations'])}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
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
                "include_confidence": include_confidence
            }

            try:
                if enable_streaming:
                    # Streaming request
                    response = requests.post(f"{api_base_url}/query/stream", json=payload, stream=True)
                    
                    if response.status_code == 200:
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
                        message_placeholder.markdown(full_response)
                        
                        # Note: Streaming endpoint currently doesn't return sources/metadata in the stream
                        # You might need a separate call or protocol update to get sources with streaming
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                else:
                    # Non-streaming request
                    with st.spinner("Thinking..."):
                        response = requests.post(f"{api_base_url}/query", json=payload)
                        
                    if response.status_code == 200:
                        data = response.json()
                        full_response = data["answer"]
                        sources = data.get("sources", [])
                        confidence = data.get("confidence")
                        variations = data.get("query_variations")
                        
                        message_placeholder.markdown(full_response)
                        
                        if sources:
                            with st.expander("📚 Sources & Metadata"):
                                st.json(sources)
                        if confidence:
                            st.info(f"Confidence Score: {confidence}")
                        if variations:
                            st.caption(f"Query Variations: {', '.join(variations)}")
                            
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response,
                            "sources": sources,
                            "confidence": confidence,
                            "variations": variations
                        })
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- Ingest Tab ---
with tab_ingest:
    st.header("📄 Ingest Documents")
    
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx", "md"])
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", 100, 4000, 1000)
        chunk_strategy = st.selectbox("Chunking Strategy", ["recursive", "semantic", "parent_child"])
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", 0, 1000, 200)
        force_reingest = st.checkbox("Force Re-ingest", value=False)
        
    if st.button("Start Ingestion", disabled=not uploaded_file):
        if uploaded_file:
            # Save file temporarily
            import os
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"File saved to {file_path}. Sending to API...")
            
            payload = {
                "file_path": os.path.abspath(file_path),
                "chunking_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "force_reingest": force_reingest
            }
            
            try:
                response = requests.post(f"{api_base_url}/ingest", json=payload)
                if response.status_code == 200:
                    st.success(f"Ingestion started! Task ID: {response.json().get('task_id')}")
                    st.json(response.json())
                else:
                    st.error(f"Ingestion failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Batch Query Tab ---
with tab_batch:
    st.header("🔄 Batch Query Processing")
    
    batch_input = st.text_area("Enter questions (one per line)", height=200, placeholder="What is the summary?\nWho are the key players?\nWhat are the risks?")
    
    if st.button("Run Batch"):
        questions = [q.strip() for q in batch_input.split("\n") if q.strip()]
        
        if not questions:
            st.warning("Please enter at least one question.")
        elif len(questions) > 10:
            st.warning("Maximum 10 questions allowed per batch.")
        else:
            payload = {
                "questions": questions,
                "llm_provider": selected_provider,
                "chain_type": selected_chain,
                "use_query_expansion": use_query_expansion,
                "use_reranking": False # Recommended false for batch speed
            }
            
            with st.spinner(f"Processing {len(questions)} queries..."):
                try:
                    response = requests.post(f"{api_base_url}/query/batch", json=payload)
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        
                        # Display as dataframe
                        df = pd.DataFrame(results)
                        st.dataframe(df)
                        
                        # Detailed view
                        for res in results:
                            with st.expander(f"Q: {res['question']}"):
                                st.markdown(f"**Answer:** {res['answer']}")
                                st.caption(f"Latency: {res['latency_ms']}ms | Status: {res['status']}")
                    else:
                        st.error(f"Batch processing failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- System Health Tab ---
with tab_system:
    st.header("📊 System Status")
    
    if st.button("Refresh Status"):
        st.rerun()
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Health Check")
        try:
            health = requests.get(f"{api_base_url}/health").json()
            st.json(health)
        except Exception as e:
            st.error(f"Health check failed: {e}")
            
    with col2:
        st.subheader("System Stats")
        try:
            stats = requests.get(f"{api_base_url}/stats").json()
            st.json(stats)
            
            if "system" in stats:
                sys_stats = stats["system"]
                st.progress(sys_stats.get("cpu_percent", 0) / 100, text=f"CPU Usage: {sys_stats.get('cpu_percent')}%")
                st.progress(sys_stats.get("memory_percent", 0) / 100, text=f"Memory Usage: {sys_stats.get('memory_percent')}%")
                st.progress(sys_stats.get("disk_percent", 0) / 100, text=f"Disk Usage: {sys_stats.get('disk_percent')}%")
        except Exception as e:
            st.error(f"Stats check failed: {e}")

    st.subheader("Available Models Configuration")
    try:
        models_info = requests.get(f"{api_base_url}/models").json()
        st.json(models_info)
    except:
        st.warning("Could not load model info")
