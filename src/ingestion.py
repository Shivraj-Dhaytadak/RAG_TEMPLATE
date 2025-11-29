from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from typing import List, Optional
import pickle
from pathlib import Path
from tqdm import tqdm
import hashlib
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
PATH = "../data/ISLP_website.pdf"
CHROMA_DIR = '../stats/chroma_semantic'
BM25_DIR = "../stats/bm25_retriever.pkl"
METADATA_DIR = "../stats/doc_metadata.json"
COLLECTION_NAME = "agentic_rag_collection"  # Explicit collection name

# ============================================================================
# EMBEDDINGS
# ============================================================================
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# ============================================================================
# OPTIMIZED DOCUMENT LOADING
# ============================================================================
def load_documents(file_path: str) -> List[Document]:
    """
    Load documents with rich metadata (LangChain 0.3+ compatible)
    """
    logger.info(f"Loading documents from {file_path}")
    
    loader = PyMuPDFLoader(
        file_path=file_path,
        extract_images=True,
        extract_tables='markdown'
    )
    
    docs = loader.load()
    
    # Add rich metadata for better filtering
    for i, doc in enumerate(docs):
        doc.metadata.update({
            'page': i + 1,
            'source': file_path,
            'doc_id': hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
            'file_name': Path(file_path).name,
            'total_pages': len(docs)
        })
    
    logger.info(f"✅ Loaded {len(docs)} pages")
    return docs

# ============================================================================
# ADVANCED CHUNKING STRATEGIES
# ============================================================================

def semantic_chunking(docs: List[Document], **kwargs) -> List[Document]:
    """
    Semantic chunking - groups text by semantic similarity
    BEST FOR: Documents with clear semantic boundaries
    SPEED: Slower (requires embeddings)
    """
    breakpoint_threshold_type = kwargs.get('breakpoint_threshold_type', 'percentile')
    breakpoint_threshold_amount = kwargs.get('breakpoint_threshold_amount', 95)
    
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )
    
    chunks = splitter.split_documents(docs)
    logger.info(f"✅ Created {len(chunks)} semantic chunks")
    return chunks

def recursive_chunking(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Recursive character splitting (RECOMMENDED for production)
    BEST FOR: General purpose, consistent chunk sizes
    SPEED: Fast
    
    OPTIMIZATION: Increased overlap from 100 to 200 for better context continuity
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Prefer natural boundaries
    )
    
    chunks = splitter.split_documents(docs)
    logger.info(f"✅ Created {len(chunks)} recursive chunks")
    return chunks

def parent_child_chunking(
    docs: List[Document],
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 500
) -> tuple[List[Document], List[Document]]:
    """
    Two-level chunking for better context (LangChain 0.3+ pattern)
    
    BEST FOR: Maximum accuracy - retrieve with small chunks, return parent context
    SPEED: Moderate
    
    How it works:
    - Small chunks (children) used for embedding/retrieval (better precision)
    - Large chunks (parents) returned as context (better recall)
    """
    # Parent chunks (larger context)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=400
    )
    parent_chunks = parent_splitter.split_documents(docs)
    
    # Child chunks (for retrieval)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=100
    )
    
    child_chunks = []
    for parent in parent_chunks:
        # Generate unique parent ID
        parent_id = hashlib.md5(parent.page_content.encode()).hexdigest()[:8]
        
        # Split parent into children
        children = child_splitter.split_documents([parent])
        
        for child in children:
            # Link child to parent
            child.metadata['parent_id'] = parent_id
            child.metadata['is_child'] = True
            child_chunks.append(child)
    
    logger.info(f"✅ Created {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")
    return child_chunks, parent_chunks

# ALTERNATIVE: Contextual chunking (requires LLM - slower but more accurate)
def contextual_chunking(docs: List[Document], llm) -> List[Document]:
    """
    Add document-level context to each chunk using LLM
    
    BEST FOR: Maximum retrieval accuracy
    SPEED: Very slow (requires LLM calls per chunk)
    
    Example: "This chunk discusses pricing in a product manual"
    """
    from langchain.prompts import ChatPromptTemplate
    
    # Get document summary
    doc_text = "\n".join([doc.page_content for doc in docs[:10]])
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this document in 2 sentences:\n\n{text}"
    )
    
    doc_context = (summary_prompt | llm).invoke({"text": doc_text[:4000]})
    
    # Chunk normally
    chunks = recursive_chunking(docs)
    
    # Add context to each chunk (expensive!)
    logger.info("Adding contextual information to chunks...")
    for chunk in tqdm(chunks, desc="Contextualizing"):
        context_prompt = ChatPromptTemplate.from_template(
            """Document context: {doc_context}

Chunk: {chunk}

Provide a 1-sentence situating context:"""
        )
        
        chunk_context = (context_prompt | llm).invoke({
            "doc_context": doc_context,
            "chunk": chunk.page_content[:500]
        })
        
        # Prepend context
        chunk.page_content = f"[Context: {chunk_context}]\n\n{chunk.page_content}"
    
    logger.info(f"✅ Created {len(chunks)} contextual chunks")
    return chunks

# ============================================================================
# VECTOR STORE CREATION (LangChain 0.3+ Compatible)
# ============================================================================
def create_vector_store(
    chunks: List[Document],
    persist_dir: str,
    collection_name: str = COLLECTION_NAME
) -> Chroma:
    """
    Create Chroma vector store (LangChain 0.3+ pattern)
    
    OPTIMIZATION: Batch processing for faster ingestion
    """
    logger.info(f"Creating vector store with {len(chunks)} chunks...")
    
    # Create vector store with explicit collection name
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    logger.info(f"✅ Vector store created at {persist_dir}")
    return vector_db

# ALTERNATIVE: FAISS for faster retrieval (good for large datasets)
def create_faiss_store(chunks: List[Document], save_path: str = "./stats/faiss_index"):
    """
    FAISS vector store - faster than Chroma for large datasets (100k+ docs)
    
    PROS: Very fast retrieval, efficient memory usage
    CONS: Requires manual persistence, less feature-rich
    """
    from langchain_community.vectorstores import FAISS
    
    logger.info(f"Creating FAISS vector store...")
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save to disk
    vector_db.save_local(save_path)
    logger.info(f"✅ FAISS vector store saved to {save_path}")
    return vector_db

# ============================================================================
# SPARSE RETRIEVER (BM25) - LangChain 0.3+ Compatible
# ============================================================================
def create_bm25_retriever(chunks: List[Document], save_path: str) -> BM25Retriever:
    """
    Create BM25 retriever for keyword-based search
    
    OPTIMIZATION: Tune k parameter for number of results
    """
    logger.info("Creating BM25 retriever...")
    
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=15  # Return top 15 results
    )
    
    # OPTIONAL: Tune BM25 parameters (advanced)
    # bm25_retriever.k1 = 1.5  # Term frequency saturation (default: 1.2)
    # bm25_retriever.b = 0.75  # Length normalization (default: 0.75)
    
    # Save to disk
    with open(save_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    
    logger.info(f"✅ BM25 retriever saved to {save_path}")
    return bm25_retriever

# ============================================================================
# METADATA TRACKING (for versioning & incremental updates)
# ============================================================================
def save_metadata(chunks: List[Document], metadata_path: str, file_path: str):
    """
    Save ingestion metadata for tracking and updates
    """
    metadata = {
        "num_chunks": len(chunks),
        "chunk_ids": [c.metadata.get('doc_id', '') for c in chunks],
        "sources": list(set([c.metadata.get('source', '') for c in chunks])),
        "timestamp": str(Path(file_path).stat().st_mtime),
        "file_size": Path(file_path).stat().st_size,
        "ingestion_date": str(Path(metadata_path).stat().st_mtime) if Path(metadata_path).exists() else None
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Metadata saved to {metadata_path}")

def check_if_reingestion_needed(file_path: str, metadata_path: str) -> bool:
    """
    Check if document has changed since last ingestion
    """
    if not Path(metadata_path).exists():
        return True
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    current_timestamp = str(Path(file_path).stat().st_mtime)
    return current_timestamp != metadata.get("timestamp")

# ============================================================================
# DEDUPLICATION
# ============================================================================
def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """
    Remove duplicate or near-duplicate chunks
    """
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    removed = len(chunks) - len(unique_chunks)
    if removed > 0:
        logger.info(f"🗑️ Removed {removed} duplicate chunks")
    
    return unique_chunks

# ============================================================================
# MAIN INGESTION PIPELINE (LangChain 0.3+ Compatible)
# ============================================================================
def ingest_documents(
    file_path: str,
    chunking_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_deduplication: bool = True,
    force_reingest: bool = False
):
    """
    Main ingestion pipeline with LangChain 0.3+ best practices
    
    Args:
        file_path: Path to PDF file
        chunking_strategy: Type of chunking (semantic, recursive, parent_child)
        chunk_size: Size of chunks (for recursive strategy)
        chunk_overlap: Overlap between chunks
        use_deduplication: Remove duplicate chunks
        force_reingest: Force reingestion even if file hasn't changed
    """
    logger.info("=" * 80)
    logger.info("STARTING DOCUMENT INGESTION (LangChain 0.3+ Compatible)")
    logger.info("=" * 80)
    
    # Check if reingestion needed
    if not force_reingest and not check_if_reingestion_needed(file_path, METADATA_DIR):
        logger.info("✅ Document unchanged. Skipping reingestion.")
        return
    
    # Step 1: Load documents
    logger.info("\n📄 Step 1: Loading documents...")
    docs = load_documents(file_path)
    
    # Step 2: Chunk documents
    logger.info(f"\n✂️ Step 2: Chunking with strategy: {chunking_strategy}")
    
    if chunking_strategy == "semantic":
        chunks = semantic_chunking(docs)
        
    elif chunking_strategy == "recursive":
        chunks = recursive_chunking(docs, chunk_size, chunk_overlap)
        
    elif chunking_strategy == "parent_child":
        child_chunks, parent_chunks = parent_child_chunking(docs)
        chunks = child_chunks
        
        # Save parent chunks separately for retrieval
        parent_path = "../stats/parent_chunks.pkl"
        with open(parent_path, "wb") as f:
            pickle.dump(parent_chunks, f)
        logger.info(f"💾 Saved parent chunks to {parent_path}")
        
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
    
    # Step 3: Deduplication
    if use_deduplication:
        logger.info("\n🔍 Step 3: Deduplicating chunks...")
        chunks = deduplicate_chunks(chunks)
    
    # Step 4: Create vector store (LangChain 0.3+ pattern)
    logger.info("\n🗄️ Step 4: Creating vector store...")
    vector_db = create_vector_store(chunks, CHROMA_DIR, COLLECTION_NAME)
    
    # ALTERNATIVE: Use FAISS for very large datasets
    # vector_db = create_faiss_store(chunks)
    
    # Step 5: Create BM25 retriever
    logger.info("\n🔎 Step 5: Creating BM25 retriever...")
    bm25_retriever = create_bm25_retriever(chunks, BM25_DIR)
    
    # Step 6: Save metadata
    logger.info("\n💾 Step 6: Saving metadata...")
    save_metadata(chunks, METADATA_DIR, file_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ INGESTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"📊 Total chunks created: {len(chunks)}")
    logger.info(f"🗄️ Vector store: {CHROMA_DIR}")
    logger.info(f"🔎 BM25 retriever: {BM25_DIR}")
    logger.info(f"💾 Metadata: {METADATA_DIR}")

# ============================================================================
# INCREMENTAL UPDATES (Add new docs without full reingestion)
# ============================================================================
def incremental_update(new_file_path: str):
    """
    Add new documents to existing vector store (LangChain 0.3+ pattern)
    """
    logger.info(f"🔄 Incremental update: Adding {new_file_path}")
    
    # Load existing vector store
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    
    # Load and chunk new document
    new_docs = load_documents(new_file_path)
    new_chunks = recursive_chunking(new_docs)
    
    # Add to vector store
    vector_db.add_documents(new_chunks)
    logger.info(f"✅ Added {len(new_chunks)} new chunks to vector store")
    
    # Update BM25 (requires full rebuild)
    logger.info("🔄 Rebuilding BM25 retriever...")
    with open(BM25_DIR, "rb") as f:
        existing_retriever = pickle.load(f)
    
    # Combine old and new chunks
    all_chunks = list(existing_retriever.docs) + new_chunks
    create_bm25_retriever(all_chunks, BM25_DIR)
    
    logger.info(f"✅ Incremental update complete")

# ============================================================================
# BATCH INGESTION (Multiple files)
# ============================================================================
def batch_ingest(file_paths: List[str], **kwargs):
    """
    Ingest multiple files at once
    """
    logger.info(f"📚 Batch ingestion: {len(file_paths)} files")
    
    all_chunks = []
    
    for file_path in file_paths:
        logger.info(f"\n📄 Processing {file_path}...")
        docs = load_documents(file_path)
        chunks = recursive_chunking(docs, **kwargs)
        all_chunks.extend(chunks)
    
    # Deduplicate across all files
    all_chunks = deduplicate_chunks(all_chunks)
    
    # Create stores
    create_vector_store(all_chunks, CHROMA_DIR, COLLECTION_NAME)
    create_bm25_retriever(all_chunks, BM25_DIR)
    
    logger.info(f"\n✅ Batch ingestion complete: {len(all_chunks)} total chunks")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Example 1: Standard ingestion (RECOMMENDED)
    # print("\n📝 Example 1: Standard Recursive Chunking")
    # ingest_documents(
    #     file_path=PATH,
    #     chunking_strategy="recursive",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     use_deduplication=True
    # )
    
    # Example 2: Semantic chunking (for documents with clear sections)
    # print("\n📝 Example 2: Semantic Chunking")
    ingest_documents(
        file_path=PATH,
        chunking_strategy="semantic",
        use_deduplication=True
    )
    
    # Example 3: Parent-child chunking (maximum accuracy)
    # print("\n📝 Example 3: Parent-Child Chunking")
    # ingest_documents(
    #     file_path=PATH,
    #     chunking_strategy="parent_child",
    #     use_deduplication=True
    # )
    
    # Example 4: Incremental update
    # print("\n📝 Example 4: Incremental Update")
    # incremental_update("new_document.pdf")
    
    # Example 5: Batch ingestion
    # print("\n📝 Example 5: Batch Ingestion")
    # batch_ingest(["doc1.pdf", "doc2.pdf", "doc3.pdf"])