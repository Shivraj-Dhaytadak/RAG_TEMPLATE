from __future__ import annotations
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ANSI Colors for terminal
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "handbooks"
INDEX_STORE = BASE_DIR / "index_store_langchain"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
INDEX_STORE.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Model Configs
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------------------------------------------
# LLM Backend Configuration
# Options: "vllm", "ollama", "llamacpp"
# -----------------------------------------------------------------
LLM_BACKEND = "ollama"  # Change this to switch backends

# 1. vLLM Server Config (OpenAI Compatible)
# Ensure vLLM is running: vllm serve Qwen/Qwen1.5-7B-Chat --host 0.0.0.0 --port 8000
VLLM_MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

# 2. Ollama Config
# Ensure Ollama is running: ollama serve
# Pull model first: ollama pull llama3
OLLAMA_MODEL_NAME = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# 3. LlamaCpp Config (GGUF)
# Path to your GGUF file
LLAMACPP_MODEL_PATH = BASE_DIR / "models" / "Qwen3-4B-Q4_0.gguf"
LLAMACPP_N_GPU_LAYERS = 35  # Adjust based on your GPU VRAM
LLAMACPP_N_CTX = 2048

# Retrieval Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVER = 10  # For individual retrievers (before ensemble)
TOP_K_FINAL = 5  # After reranking


# ----------------------------
# 1. Ingestion & Indexing
# ----------------------------
def ingest_documents() -> List[Any]:
    """Load PDFs from the PDF_DIR"""
    pdf_files = sorted(list(PDF_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"{Colors.FAIL}❌ No PDF files found in {PDF_DIR}{Colors.ENDC}")
        return []

    print(f"{Colors.CYAN}Found {len(pdf_files)} PDFs. Loading...{Colors.ENDC}")
    all_docs = []

    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()
            print(f"  - Loaded {pdf_path.name}: {len(docs)} pages")
            all_docs.extend(docs)
        except Exception as e:
            print(
                f"{Colors.WARNING}  ! Failed to load {pdf_path.name}: {e}{Colors.ENDC}"
            )

    return all_docs


def create_or_load_index(force_rebuild: bool = False):
    """Create or load vector store and BM25 retriever"""

    # Paths for persistence
    faiss_path = INDEX_STORE / "faiss_index"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # Use 'cuda' if GPU available for embeddings
        encode_kwargs={"normalize_embeddings": True},
    )

    if not force_rebuild and faiss_path.exists():
        print(f"{Colors.GREEN}Loading existing FAISS index...{Colors.ENDC}")
        try:
            vector_store = FAISS.load_local(
                str(faiss_path), embeddings, allow_dangerous_deserialization=True
            )

            # For BM25, we need the documents again to rebuild it efficiently or pickle it.
            # To keep it simple and stateless for this demo, we'll re-ingest for BM25
            # or we could save/load it. For now, let's re-ingest texts for BM25.
            # Ideally, BM25 should also be persisted.
            print(f"{Colors.BLUE}Re-scanning documents for BM25 (fast)...{Colors.ENDC}")
            docs = ingest_documents()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = TOP_K_RETRIEVER

            return vector_store, bm25_retriever

        except Exception as e:
            print(
                f"{Colors.WARNING}Failed to load index: {e}. Rebuilding...{Colors.ENDC}"
            )

    # Rebuild
    print(f"{Colors.BLUE}Building new index...{Colors.ENDC}")
    docs = ingest_documents()
    if not docs:
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # Build FAISS
    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(str(faiss_path))

    # Build BM25
    print("Building BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = TOP_K_RETRIEVER

    return vector_store, bm25_retriever


# ----------------------------
# 2. Setup Retrieval Chain
# ----------------------------
def setup_retrieval_chain(vector_store, bm25_retriever):
    """Configure the hybrid retrieval + reranking chain"""

    # 1. Base Retrievers
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVER})

    # 2. Ensemble (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    # 3. Reranking
    print(f"Loading reranker: {RERANK_MODEL_NAME}...")
    model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL_NAME)
    compressor = CrossEncoderReranker(model=model, top_n=TOP_K_FINAL)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compression_retriever


# ----------------------------
# 3. vLLM Setup
# ----------------------------
def setup_vllm_client():
    """Setup OpenAI-compatible client for vLLM"""
    print(
        f"{Colors.HEADER}Connecting to vLLM server at {VLLM_API_BASE}...{Colors.ENDC}"
    )
    try:
        llm = ChatOpenAI(
            model=VLLM_MODEL_NAME,
            base_url=VLLM_API_BASE,
            api_key=VLLM_API_KEY, # type: ignore
            temperature=0.2,
            max_completion_tokens=512,
            top_p=0.95,
        )
        return llm
    except Exception as e:
        print(f"{Colors.FAIL}Failed to connect to vLLM: {e}{Colors.ENDC}")
        raise e


def setup_ollama_client():
    """Setup Ollama client"""
    print(f"{Colors.HEADER}Connecting to Ollama at {OLLAMA_BASE_URL}...{Colors.ENDC}")
    try:
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
            top_p=0.95,
        )
        return llm
    except Exception as e:
        print(f"{Colors.FAIL}Failed to connect to Ollama: {e}{Colors.ENDC}")
        raise e


def setup_llamacpp_client():
    """Setup LlamaCpp client"""
    print(
        f"{Colors.HEADER}Loading LlamaCpp model from {LLAMACPP_MODEL_PATH}...{Colors.ENDC}"
    )
    if not LLAMACPP_MODEL_PATH.exists():
        print(
            f"{Colors.FAIL}Model file not found at {LLAMACPP_MODEL_PATH}{Colors.ENDC}"
        )
        sys.exit(1)

    try:
        llm = LlamaCpp(
            model_path=str(LLAMACPP_MODEL_PATH),
            n_gpu_layers=LLAMACPP_N_GPU_LAYERS,
            n_ctx=LLAMACPP_N_CTX,
            temperature=0.2,
            max_tokens=512,
            top_p=0.95,
            verbose=False,
        )
        return llm
    except Exception as e:
        print(f"{Colors.FAIL}Failed to initialize LlamaCpp: {e}{Colors.ENDC}")
        raise e


def setup_llm():
    """Initialize LLM based on configuration"""
    print(f"{Colors.BOLD}Initializing LLM Backend: {LLM_BACKEND}{Colors.ENDC}")

    if LLM_BACKEND == "vllm":
        return setup_vllm_client()
    elif LLM_BACKEND == "ollama":
        return setup_ollama_client()
    elif LLM_BACKEND == "llamacpp":
        return setup_llamacpp_client()
    else:
        print(f"{Colors.FAIL}Invalid LLM_BACKEND: {LLM_BACKEND}{Colors.ENDC}")
        sys.exit(1)


# ----------------------------
# 4. RAG Chain
# ----------------------------
def format_docs(docs):
    """Format matching documents for the prompt"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("file_path", "unknown")).name
        page = doc.metadata.get("page", "?")
        content = doc.page_content.replace("\n", " ")
        formatted.append(f"[Source {i}: {source} (Page {page})]\n{content}\n")
    return "\n".join(formatted)


def create_chain(retriever, llm):
    """Create the final RAG chain"""

    template = """You are an expert HR Assistant. Answer the question based ONLY on the following context:

{context}

Question: {question}

Instructions:
1. Be specific and citation-focused.
2. If the answer is not in the context, say "I cannot find this information in the handbook."
3. Do not make up information.

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ----------------------------
# Main Loop
# ----------------------------
def main():
    print(f"{Colors.HEADER}Using LLM Backend: {LLM_BACKEND}{Colors.ENDC}\n")

    # 1. Indexing
    if "--reindex" in sys.argv:
        try:
            shutil.rmtree(INDEX_STORE)
        except:
            pass
        vector_store, bm25_retriever = create_or_load_index(force_rebuild=True)
    else:
        vector_store, bm25_retriever = create_or_load_index(force_rebuild=False)

    # 2. Retrieval Setup
    retriever = setup_retrieval_chain(vector_store, bm25_retriever)

    # 3. LLM Setup
    llm = setup_llm()

    # 4. Chain Setup
    chain = create_chain(retriever, llm)

    print(f"\n{Colors.GREEN}✓ System Ready! Type 'exit' to quit.{Colors.ENDC}")
    print("-" * 50)

    while True:
        try:
            query = input(f"\n{Colors.BOLD}Question: {Colors.ENDC}").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print(f"{Colors.CYAN}Thinking...{Colors.ENDC}")

            # Stream response if possible (vLLM supports it, but standard invoke is safer for now)
            try:
                # We can also get sources by modifying the chain to return them,
                # but for simplicity we'll just print the answer first.
                # To debug sources, we can invoke the retriever separately.

                # Debug sources option
                # docs = retriever.invoke(query)
                # print(format_docs(docs))

                response = chain.invoke(query)
                print(f"\n{Colors.GREEN}Answer:{Colors.ENDC}\n{response}")

            except Exception as e:
                print(f"{Colors.FAIL}Error generating response: {e}{Colors.ENDC}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
