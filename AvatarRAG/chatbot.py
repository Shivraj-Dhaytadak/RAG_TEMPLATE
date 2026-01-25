from __future__ import annotations
import os
import re
import json
import time
import pickle
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
import numpy as np
from tqdm import tqdm
import fitz  # PyMuPDF
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ----------------------------
# Config - OPTIMIZED FOR QWEN
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "handbooks"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "index_store"
CACHE_DIR = BASE_DIR / "cache"

# Create directories
for dir_path in [INDEX_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "handbook.faiss"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNK_HASH_PATH = INDEX_DIR / "chunk_hash.txt"

# Embeddings - Using multilingual model for better performance
# EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"  # Better for diverse content
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Alternative: faster, good quality

# Reranker
USE_RERANKER = True
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking settings - IMPROVED
CHUNK_CHARS = 800  # Smaller chunks for better precision
CHUNK_OVERLAP = 150  # Balanced overlap
MIN_CHUNK_CHARS = 100  # Filter out tiny chunks

# Retrieval settings - OPTIMIZED
TOP_K_VECTOR = 20  # Cast wider net initially
TOP_K_BM25 = 20
TOP_K_PRE_RERANK = 25
TOP_K_FINAL = 6  # Final selection after reranking

# LLM settings - QWEN OPTIMIZED
TEMPERATURE = 0.2  # Slightly higher for Qwen (less repetitive)
MAX_NEW_TOKENS = 512  # Qwen can handle longer, quality responses
TOP_P = 0.8  # Nucleus sampling
TOP_K_SAMPLING = 40  # Top-k sampling
REPEAT_PENALTY = 1.15  # Qwen specific: reduce repetition
FREQUENCY_PENALTY = 0.1  # Penalize frequent tokens
PRESENCE_PENALTY = 0.1  # Encourage diversity

# Context settings
LLM_N_CTX = 40960 # Qwen models support larger context
LLM_N_BATCH = 512
MAX_CONTEXT_CHARS = 4500  # Use more context with Qwen

# GPU settings
USE_GPU = True
N_GPU_LAYERS = 0  # Auto-detect or set manually (e.g., 35 for RTX 3090)

# Qwen-specific chat template
USE_QWEN_TEMPLATE = True  # Enable Qwen chat template

# Caching
ENABLE_QUERY_CACHE = True
MAX_CACHE_SIZE = 100

# UI settings
PRINT_SOURCES_EVERY_ANSWER = False
MAX_SOURCE_EXCERPTS_TO_PRINT = 4

# IMPROVED: Qwen-specific system prompt
SYSTEM_PROMPT = """You are an expert HR Benefits Assistant with deep knowledge of company policies.

Your task: Answer employee questions using ONLY the provided handbook excerpts below.

Critical Rules:
1. BE SPECIFIC: Include exact numbers, dates, percentages, and requirements
2. CITE SOURCES: Always reference [source: <file> p.<page>] for every claim
3. COMPLETE ANSWERS: Cover eligibility, process, deadlines, and exceptions
4. ADMIT GAPS: If information is missing, say "The handbook doesn't specify this"
5. NO ASSUMPTIONS: Don't infer or assume information not in the excerpts

Format:
- Use bullet points for multiple requirements or steps
- Quote exact policy language when relevant
- Highlight important deadlines or restrictions

Remember: Employees rely on accurate information for important decisions."""

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    chunk_id: int
    text: str
    source_file: str
    page: int
    char_count: int = 0
    
    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.text)

# ----------------------------
# Helpers - IMPROVED
# ----------------------------
def clean_text(s: str) -> str:
    """Enhanced text cleaning"""
    # Remove soft hyphens and other unicode artifacts
    s = s.replace("\u00ad", "")
    s = s.replace("\ufeff", "")  # BOM
    s = s.replace("\xa0", " ")  # Non-breaking space
    
    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    
    # Remove page numbers and headers/footers (common pattern)
    s = re.sub(r"^\s*\d+\s*$", "", s, flags=re.MULTILINE)
    
    return s.strip()

def tokenize(s: str) -> List[str]:
    """Improved tokenization"""
    s = s.lower()
    # Keep important punctuation for better BM25
    s = re.sub(r"[^a-z0-9\-]+", " ", s)
    return [t for t in s.split() if len(t) > 1]  # Filter single chars

def chunk_text(text: str, max_chars: int, overlap: int, min_chars: int = 100) -> List[str]:
    """
    IMPROVED: Better chunking with sentence awareness and filtering
    """
    # Split into sentences first for better boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks: List[str] = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence exceeds max, save current chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0:
                # Take last N characters for overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap//5:])  # Approx word-based overlap
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= min_chars:
        chunks.append(current_chunk.strip())
    
    return chunks

def compute_chunk_hash(chunks: List[Chunk]) -> str:
    """Compute hash of chunks to detect changes"""
    content = "".join(f"{c.source_file}:{c.page}:{c.text[:100]}" for c in chunks)
    return hashlib.md5(content.encode()).hexdigest()

# ----------------------------
# PDF ingestion - IMPROVED
# ----------------------------
def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Enhanced PDF extraction with better error handling"""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Warning: Could not open {pdf_path.name}: {e}")
        return []
    
    pages: List[Tuple[int, str]] = []
    
    for i in range(doc.page_count):
        try:
            # Try multiple extraction methods
            txt = doc.load_page(i).get_text("text")
            
            # If text extraction fails, try OCR or skip
            if not txt or len(txt.strip()) < 20: # type: ignore
                # Could add OCR here with pytesseract if needed
                continue
            
            txt = clean_text(txt) # type: ignore
            if txt:
                pages.append((i + 1, txt))
        except Exception as e:
            print(f"Warning: Error extracting page {i+1} from {pdf_path.name}: {e}")
            continue
    
    doc.close()
    return pages

def build_chunks(pdf_dir: Path) -> List[Chunk]:
    """Build chunks with improved filtering"""
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir.resolve()}")
    
    chunks: List[Chunk] = []
    cid = 0
    
    print(f"Processing {len(pdfs)} PDF(s)...")
    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        pages = extract_pdf_pages(pdf)
        
        for page_num, page_text in pages:
            page_chunks = chunk_text(page_text, CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_CHARS)
            
            for c_text in page_chunks:
                chunks.append(Chunk(
                    chunk_id=cid,
                    text=c_text,
                    source_file=pdf.name,
                    page=page_num,
                    char_count=len(c_text)
                ))
                cid += 1
    
    print(f"Created {len(chunks)} chunks from {len(pdfs)} PDF(s)")
    return chunks

# ----------------------------
# Persistence - IMPROVED
# ----------------------------
def save_meta(chunks: List[Chunk], meta_path: Path) -> None:
    """Save with better serialization"""
    with meta_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

def load_meta(meta_path: Path) -> List[Chunk]:
    """Load with validation"""
    chunks: List[Chunk] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                o = json.loads(line)
                chunks.append(Chunk(**o))
            except Exception as e:
                print(f"Warning: Skipping corrupted chunk at line {line_num}: {e}")
                continue
    return chunks

def save_bm25(bm25: BM25Okapi, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bm25(path: Path) -> BM25Okapi:
    with path.open("rb") as f:
        return pickle.load(f)

# ----------------------------
# Embeddings + FAISS - IMPROVED
# ----------------------------
def embed_passages(model: SentenceTransformer, passages: List[str], batch_size: int = 32) -> np.ndarray:
    """Optimized embedding with error handling"""
    # For multilingual-e5, use "passage: " prefix
    prefixed = [f"passage: {p[:512]}" for p in passages]  # Limit length
    
    try:
        emb = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        print(f"Error during embedding: {e}")
        # Fallback: process one by one
        embeddings = []
        for p in prefixed:
            try:
                emb = model.encode([p], normalize_embeddings=True)
                embeddings.append(emb[0])
            except:
                embeddings.append(np.zeros(model.get_sentence_embedding_dimension())) # type: ignore
        return np.array(embeddings, dtype=np.float32)

@lru_cache(maxsize=1000)
def embed_query_cached(model_name: str, query: str) -> bytes:
    """Cached query embedding"""
    # This is a placeholder - actual caching happens in retrieve()
    return query.encode()

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    """Query embedding with prefix"""
    emb = model.encode([f"query: {q}"], normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build optimized FAISS index"""
    dim = embeddings.shape[1]
    n_embeddings = embeddings.shape[0]
    
    # Use IVF for large datasets (faster search)
    if n_embeddings > 1000:
        nlist = min(int(np.sqrt(n_embeddings)), 100)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings) # type: ignore
        index.add(embeddings) # type: ignore
        index.nprobe = min(10, nlist)  # Search quality vs speed
    else:
        # Simple flat index for small datasets
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings) # type: ignore
    
    return index

# ----------------------------
# BM25
# ----------------------------
def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    """Build BM25 with improved tokenization"""
    corpus = [tokenize(c.text) for c in chunks]
    return BM25Okapi(corpus, k1=1.5, b=0.75)  # Tuned parameters

# ----------------------------
# Reranker - IMPROVED
# ----------------------------
class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.ce = None
    
    def load(self) -> None:
        if self.ce is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder(self.model_name, max_length=512)
            print(f"Loaded reranker: {self.model_name}")
        except Exception as e:
            print(f"[Reranker disabled] Could not load '{self.model_name}'. Reason: {e}")
            self.ce = None
    
    def rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:
        self.load()
        
        if self.ce is None or not candidates:
            return candidates[:top_k]
        
        # Batch reranking for efficiency
        pairs = [(query, c.text[:512]) for c in candidates]  # Limit length
        
        try:
            scores = self.ce.predict(pairs, batch_size=32, show_progress_bar=False)
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: float(x[1]), reverse=True)
            return [c for c, _ in scored[:top_k]]
        except Exception as e:
            print(f"Reranking error: {e}, returning original order")
            return candidates[:top_k]

# ----------------------------
# Query Cache - NEW
# ----------------------------
class QueryCache:
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, List[Chunk]] = {}
        self.max_size = max_size
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[Chunk]]:
        key = self._hash_query(query)
        return self.cache.get(key)
    
    def set(self, query: str, chunks: List[Chunk]) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        key = self._hash_query(query)
        self.cache[key] = chunks

# ----------------------------
# Retrieval - IMPROVED
# ----------------------------
def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    faiss_index: faiss.Index,
    chunks: List[Chunk],
    bm25: BM25Okapi,
    reranker: Optional[Reranker],
    cache: Optional[QueryCache] = None
) -> List[Chunk]:
    """
    IMPROVED: Hybrid retrieval with caching and better fusion
    """
    # Check cache
    if cache and ENABLE_QUERY_CACHE:
        cached = cache.get(query)
        if cached is not None:
            return cached
    
    # Vector search
    q_emb = embed_query(embed_model, query)
    vec_scores, vec_idxs = faiss_index.search(q_emb, TOP_K_VECTOR) # type: ignore
    
    # BM25 search
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_top_idxs = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]
    
    # IMPROVED: Reciprocal Rank Fusion (RRF) for better merging
    rrf_scores: Dict[int, float] = {}
    k = 60  # RRF constant
    
    # Add vector scores
    for rank, idx in enumerate(vec_idxs[0]):
        if idx >= 0:
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (k + rank + 1)
    
    # Add BM25 scores
    for rank, idx in enumerate(bm25_top_idxs):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
    
    # Sort by RRF score
    sorted_idxs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_idxs = sorted_idxs[:TOP_K_PRE_RERANK]
    
    candidates = [chunks[i] for i in top_idxs]
    
    # Rerank if available
    if reranker is not None and USE_RERANKER:
        candidates = reranker.rerank(query, candidates, top_k=TOP_K_FINAL)
    else:
        candidates = candidates[:TOP_K_FINAL]
    
    # Cache result
    if cache and ENABLE_QUERY_CACHE:
        cache.set(query, candidates)
    
    return candidates

def build_context_block(selected: List[Chunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build context with smart truncation"""
    blocks = []
    total_chars = 0
    
    for k, ch in enumerate(selected, start=1):
        # More concise citation for Qwen
        block = f"[EXCERPT {k} - {ch.source_file} p.{ch.page}]\n{ch.text}\n"
        
        if total_chars + len(block) > max_chars:
            # Try to fit truncated version
            remaining = max_chars - total_chars
            if remaining > 200:
                truncated = f"[EXCERPT {k} - {ch.source_file} p.{ch.page}]\n{ch.text[:remaining-50]}...\n"
                blocks.append(truncated)
            break
        
        blocks.append(block)
        total_chars += len(block)
    
    return "\n".join(blocks).strip()

# ----------------------------
# LLM loading - QWEN OPTIMIZED
# ----------------------------
def find_gguf_model(models_dir: Path) -> Path:
    """Find GGUF model with Qwen preference"""
    ggufs = sorted(models_dir.glob("*.gguf"))
    
    if not ggufs:
        raise SystemExit(
            f"No .gguf files found in {models_dir.resolve()}\n"
            "Download a Qwen model from HuggingFace"
        )
    
    # Prefer Qwen models
    for p in ggufs:
        if "qwen" in p.name.lower():
            return p
    
    # Prefer Q4_K_M quantization
    for p in ggufs:
        if "q4_k_m" in p.name.lower():
            return p
    
    return ggufs[0]

def detect_gpu_layers() -> int:
    """Auto-detect optimal GPU layers"""
    if not USE_GPU:
        return 0
    
    try:
        import torch
        if torch.cuda.is_available():
            # Estimate based on VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if vram_gb >= 24:
                return 50  # High-end GPU
            elif vram_gb >= 12:
                return 35  # Mid-range GPU
            elif vram_gb >= 8:
                return 25  # Entry-level GPU
            else:
                return 15  # Low VRAM
        else:
            print("CUDA not available, using CPU")
            return 0
    except:
        return 0

def make_llm(model_path: Path) -> Llama:
    """Create LLM with Qwen-optimized settings"""
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path.resolve()}")
    
    cpu_count = os.cpu_count() or 8
    n_threads = max(4, cpu_count - 2)
    
    # Auto-detect GPU layers
    n_gpu = N_GPU_LAYERS if N_GPU_LAYERS > 0 else detect_gpu_layers()
    
    print(f"Loading {model_path.name}...")
    print(f"Threads: {n_threads}, GPU layers: {n_gpu}, Context: {LLM_N_CTX}")
    
    # Qwen-optimized parameters
    return Llama(
        model_path=str(model_path),
        n_ctx=LLM_N_CTX,
        n_threads=n_threads,
        n_batch=LLM_N_BATCH,
        n_gpu_layers=n_gpu,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
        # Qwen works well with these
        rope_freq_base=1000000.0,  # Extended context
        rope_freq_scale=1.0,
    )

# ----------------------------
# Answering - QWEN OPTIMIZED
# ----------------------------
def answer_question(
    llm: Llama,
    query: str,
    context: str,
    chat_history: List[Tuple[str, str]]
) -> str:
    """Generate answer with Qwen-optimized prompting"""
    
    # Keep last 3 exchanges (Qwen handles context well)
    trimmed = chat_history[-3:]
    
    if USE_QWEN_TEMPLATE:
        # Qwen-specific chat template
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })
        
        # Chat history
        for u, a in trimmed:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        
        # Current query with context
        user_content = f"""Question: {query}

Handbook Excerpts:
{context}

Instructions: Answer the question using ONLY the excerpts above. Be specific and cite sources."""
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            out = llm.create_chat_completion(
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
                top_p=TOP_P,
                top_k=TOP_K_SAMPLING,
                repeat_penalty=REPEAT_PENALTY,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                stop=["<|endoftext|>", "<|im_end|>", "User:", "\n\nQuestion:"],
            )
            return out["choices"][0]["message"]["content"].strip() # type: ignore
        
        except Exception as e:
            print(f"Chat completion failed ({e}), trying fallback...")
            # Fallback to text completion
            pass
    
    # Fallback: Direct text completion
    prompt = f"""{SYSTEM_PROMPT}

Handbook Excerpts:
{context}

Question: {query}

Answer:"""
    
    try:
        out = llm(
            prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K_SAMPLING,
            repeat_penalty=REPEAT_PENALTY,
            stop=["Question:", "Handbook Excerpts:", "\n\n\n"]
        )
        return out["choices"][0]["text"].strip() # type: ignore
    
    except Exception as e:
        return f"Error generating response: {e}"

# ----------------------------
# Index build/load - IMPROVED
# ----------------------------
def build_or_load(embed_model: SentenceTransformer) -> Tuple[faiss.Index, List[Chunk], BM25Okapi]:
    """Build or load index with hash validation"""
    
    # Check if index exists and is valid
    if (FAISS_INDEX_PATH.exists() and META_PATH.exists() and 
        BM25_PATH.exists() and CHUNK_HASH_PATH.exists()):
        
        # Load existing
        chunks = load_meta(META_PATH)
        
        # Verify hash
        current_hash = compute_chunk_hash(chunks)
        stored_hash = CHUNK_HASH_PATH.read_text().strip()
        
        if current_hash == stored_hash:
            print("Loading existing index...")
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            bm25 = load_bm25(BM25_PATH)
            print(f"Loaded {len(chunks)} chunks")
            return index, chunks, bm25
        else:
            print("Hash mismatch - rebuilding index...")
    
    # Build new index
    print("Building index from PDFs...")
    chunks = build_chunks(PDF_DIR)
    
    if not chunks:
        raise SystemExit("No chunks created - check your PDFs")
    
    save_meta(chunks, META_PATH)
    
    passages = [c.text for c in chunks]
    
    # Embed in batches
    print("Creating embeddings...")
    sample = embed_passages(embed_model, [passages[0]], batch_size=1)
    dim = int(sample.shape[1])
    embeddings = np.zeros((len(passages), dim), dtype=np.float32)
    
    batch = 32
    for start in tqdm(range(0, len(passages), batch), desc="Embedding"):
        end = min(start + batch, len(passages))
        embeddings[start:end] = embed_passages(embed_model, passages[start:end], batch_size=batch)
    
    # Build FAISS
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    # Build BM25
    print("Building BM25 index...")
    bm25 = build_bm25(chunks)
    save_bm25(bm25, BM25_PATH)
    
    # Save hash
    chunk_hash = compute_chunk_hash(chunks)
    CHUNK_HASH_PATH.write_text(chunk_hash)
    
    print(f"Index built: {len(chunks)} chunks")
    return index, chunks, bm25

# ----------------------------
# UI helpers - IMPROVED
# ----------------------------
def print_sources(selected: List[Chunk], max_to_print: int = 5) -> None:
    """Enhanced source display"""
    print("\n" + "="*60)
    print("📚 SOURCES USED:")
    print("="*60)
    
    for i, ch in enumerate(selected[:max_to_print], 1):
        print(f"\n[{i}] {ch.source_file} - Page {ch.page}")
        print(f"    Chunk ID: {ch.chunk_id} | Length: {ch.char_count} chars")
        
        excerpt = ch.text.strip().replace("\n", " ")
        excerpt = excerpt[:300] + ("..." if len(excerpt) > 300 else "")
        print(f"    Preview: {excerpt}")
    
    if len(selected) > max_to_print:
        print(f"\n... and {len(selected) - max_to_print} more sources")
    
    print("="*60)

def help_text() -> str:
    return """
╔══════════════════════════════════════════════════════════╗
║              HR HANDBOOK Q&A - COMMANDS                  ║
╠══════════════════════════════════════════════════════════╣
║  /help      Show this help message                       ║
║  /sources   Show sources from last answer                ║
║  /reindex   Rebuild index (run after updating PDFs)      ║
║  /clear     Clear chat history                           ║
║  /stats     Show performance statistics                  ║
║  /exit      Quit the chatbot                             ║
╚══════════════════════════════════════════════════════════╝
"""

def delete_index_files() -> None:
    """Delete all index files"""
    for p in [FAISS_INDEX_PATH, META_PATH, BM25_PATH, CHUNK_HASH_PATH]:
        if p.exists():
            try:
                p.unlink()
                print(f"Deleted: {p.name}")
            except Exception as e:
                print(f"Could not delete {p.name}: {e}")

# ----------------------------
# Main - IMPROVED
# ----------------------------
def main() -> None:
    """Main application with better UX"""
    
    # Startup banner
    print("="*60)
    print("  HR HANDBOOK Q&A CHATBOT - QWEN OPTIMIZED")
    print("="*60)
    
    # Validate directories
    for dir_path, name in [(PDF_DIR, "handbooks"), (MODELS_DIR, "models")]:
        if not dir_path.exists():
            raise SystemExit(f"❌ Missing folder: {dir_path.resolve()}\n"
                           f"   Create ./{name}/ and add required files")
    
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"❌ No PDFs found in {PDF_DIR.resolve()}")
    
    # Find model
    model_path = find_gguf_model(MODELS_DIR)
    print(f"✓ Model: {model_path.name}")
    print(f"✓ PDFs: {len(pdfs)} document(s)")
    
    # Load components
    print("\n🔄 Initializing...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    faiss_index, chunks, bm25 = build_or_load(embed_model)
    
    reranker = Reranker(RERANKER_MODEL_NAME) if USE_RERANKER else None
    query_cache = QueryCache(MAX_CACHE_SIZE) if ENABLE_QUERY_CACHE else None
    
    llm = make_llm(model_path)
    
    print("\n✅ Ready!")
    print(help_text())
    
    # State
    chat_history: List[Tuple[str, str]] = []
    last_selected: List[Chunk] = []
    stats = {"queries": 0, "total_time": 0.0}
    
    # Main loop
    while True:
        try:
            q = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not q:
            continue
        
        low = q.lower().strip()
        
        # Commands
        if low in {"/exit", "exit", "quit", "q"}:
            print("Goodbye! 👋")
            break
        
        if low == "/help":
            print(help_text())
            continue
        
        if low == "/sources":
            if last_selected:
                print_sources(last_selected, MAX_SOURCE_EXCERPTS_TO_PRINT)
            else:
                print("ℹ️  No sources yet - ask a question first")
            continue
        
        if low == "/clear":
            chat_history.clear()
            last_selected.clear()
            if query_cache:
                query_cache.cache.clear()
            print("✓ Chat history cleared")
            continue
        
        if low == "/reindex":
            print("🔄 Rebuilding index...")
            delete_index_files()
            faiss_index, chunks, bm25 = build_or_load(embed_model)
            if query_cache:
                query_cache.cache.clear()
            print("✓ Index rebuilt")
            continue
        
        if low == "/stats":
            avg_time = stats["total_time"] / max(stats["queries"], 1)
            print(f"\n📊 STATISTICS:")
            print(f"   Queries: {stats['queries']}")
            print(f"   Avg response time: {avg_time:.2f}s")
            print(f"   Cache size: {len(query_cache.cache) if query_cache else 0}")
            print(f"   Chunks indexed: {len(chunks)}")
            continue
        
        # Process query
        t0 = time.time()
        
        try:
            selected = retrieve(
                q, embed_model, faiss_index, chunks, bm25, reranker, query_cache
            )
            last_selected = selected
            
            context = build_context_block(selected)
            answer = answer_question(llm, q, context, chat_history)
            
            dt = time.time() - t0
            stats["queries"] += 1
            stats["total_time"] += dt
            
            print(f"\n🤖 Assistant ({dt:.1f}s):")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            
            if PRINT_SOURCES_EVERY_ANSWER:
                print_sources(selected, MAX_SOURCE_EXCERPTS_TO_PRINT)
            
            chat_history.append((q, answer))
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()