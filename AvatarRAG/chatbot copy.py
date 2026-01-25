# chatbot.py

# Local HR handbook Q&A chatbot (offline LLM + local RAG over 2 PDFs)

#

# Folder structure (relative to this file):

#  ./handbooks/  -> put your 2 PDFs here

#  ./models/   -> put your GGUF model here

#  ./index_store/ -> auto-created on first run (FAISS + metadata)

#

# Requirements (PowerShell):

#  python -m pip install pymupdf sentence-transformers faiss-cpu llama-cpp-python numpy tqdm rank-bm25

# Optional (reranker, improves precision):

#  python -m pip install torch

#

# Run:

#  python chatbot.py

from __future__ import annotations
import os
import re
import json
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import fitz # PyMuPDF
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
# ----------------------------

# Config (edit only if needed)

# ----------------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "handbooks"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "index_store"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = INDEX_DIR / "handbook.faiss"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"
# Embeddings (fast + good retrieval)
EMBED_MODEL_NAME = "intfloat/e5-small-v2"
# Optional reranker for higher precision
USE_RERANKER = True
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Chunking settings
CHUNK_CHARS = 1100
CHUNK_OVERLAP = 200
# Retrieval settings
TOP_K_VECTOR = 16
TOP_K_BM25 = 16
TOP_K_PRE_RERANK = 20
TOP_K_FINAL = 8
# LLM settings
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 380
# LLM context window (8192 is a safe default for GGUF CPU use)
LLM_N_CTX = 8192
# Print sources behavior
PRINT_SOURCES_EVERY_ANSWER = False
MAX_SOURCE_EXCERPTS_TO_PRINT = 5
SYSTEM_PROMPT = """You are an HR Benefits Assistant.
Answer the user's question using ONLY the provided handbook excerpts.

Rules:
- Be direct and specific. Do not give generic advice.
- If the excerpts do not contain the answer, say: "I couldn't find that in the handbook excerpts I have."
- Cite the exact source and page like: [source: <file> p.<page> chunk:<id>]
- Include eligibility rules, deadlines, amounts, exceptions, and steps if present.
- If the user asks "can I" or "am I eligible", answer by stating the exact eligibility rules in the excerpts.
"""
# ----------------------------
# Data structures
# ----------------------------

@dataclass

class Chunk:
  chunk_id: int
  text: str
  source_file: str
  page: int # 1-based page number

# ----------------------------

# Helpers

# ----------------------------

def clean_text(s: str) -> str:
  s = s.replace("\u00ad", "") # soft hyphen
  s = re.sub(r"[ \t]+", " ", s)
  s = re.sub(r"\n{3,}", "\n\n", s)
  return s.strip()

def tokenize(s: str) -> List[str]:
  s = s.lower()
  s = re.sub(r"[^a-z0-9]+", " ", s)
  return [t for t in s.split() if t]

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
  paras = [p.strip() for p in text.split("\n\n") if p.strip()]
  chunks: List[str] = []
  cur = ""
  def push(cur_text: str):
    if cur_text.strip():
      chunks.append(cur_text.strip())

  for p in paras:
    if len(cur) + len(p) + 2 <= max_chars:
      cur = (cur + "\n\n" + p).strip() if cur else p

    else:
      push(cur)
      if overlap > 0 and cur:
        tail = cur[-overlap:]
        cur = (tail + "\n\n" + p).strip()

      else:
        cur = p

  push(cur)
  return chunks


# ----------------------------

# PDF ingestion

# ----------------------------

def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
  doc = fitz.open(pdf_path)
  pages: List[Tuple[int, str]] = []

  for i in range(doc.page_count):

    txt = doc.load_page(i).get_text("text")

    txt = clean_text(txt) # type: ignore

    if txt:

      pages.append((i + 1, txt))

  return pages

def build_chunks(pdf_dir: Path) -> List[Chunk]:

  pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])

  if not pdfs:

    raise SystemExit(f"No PDFs found in {pdf_dir.resolve()}")

  chunks: List[Chunk] = []

  cid = 0

  for pdf in pdfs:

    pages = extract_pdf_pages(pdf)

    for page_num, page_text in pages:

      for c in chunk_text(page_text, CHUNK_CHARS, CHUNK_OVERLAP):

        chunks.append(Chunk(chunk_id=cid, text=c, source_file=pdf.name, page=page_num))

        cid += 1

  return chunks


# ----------------------------

# Persistence

# ----------------------------

def save_meta(chunks: List[Chunk], meta_path: Path) -> None:

  with meta_path.open("w", encoding="utf-8") as f:

    for ch in chunks:

      f.write(json.dumps({

        "chunk_id": ch.chunk_id,

        "source_file": ch.source_file,

        "page": ch.page,

        "text": ch.text

      }, ensure_ascii=False) + "\n")

def load_meta(meta_path: Path) -> List[Chunk]:

  chunks: List[Chunk] = []

  with meta_path.open("r", encoding="utf-8") as f:

    for line in f:

      o = json.loads(line)

      chunks.append(Chunk(

        chunk_id=o["chunk_id"],

        source_file=o["source_file"],

        page=o["page"],

        text=o["text"]

      ))

  return chunks

def save_bm25(bm25: BM25Okapi, path: Path) -> None:

  with path.open("wb") as f:

    pickle.dump(bm25, f)

def load_bm25(path: Path) -> BM25Okapi:

  with path.open("rb") as f:

    return pickle.load(f)


# ----------------------------

# Embeddings + FAISS

# ----------------------------

def embed_passages(model: SentenceTransformer, passages: List[str], batch_size: int = 64) -> np.ndarray:

  prefixed = [f"passage: {p}" for p in passages] # E5 format

  emb = model.encode(prefixed, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

  return np.asarray(emb, dtype=np.float32)

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:

  emb = model.encode([f"query: {q}"], normalize_embeddings=True) # E5 format

  return np.asarray(emb, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:

  dim = embeddings.shape[1]

  index = faiss.IndexFlatIP(dim) # cosine via inner product on normalized vectors

  index.add(embeddings) # type: ignore

  return index


# ----------------------------

# BM25

# ----------------------------

def build_bm25(chunks: List[Chunk]) -> BM25Okapi:

  corpus = [tokenize(c.text) for c in chunks]

  return BM25Okapi(corpus)


# ----------------------------

# Optional reranker

# ----------------------------

class Reranker:

  def __init__(self, model_name: str):

    self.model_name = model_name

    self.ce = None

  def load(self) -> None:

    if self.ce is not None:

      return

    try:

      from sentence_transformers import CrossEncoder # type: ignore

      self.ce = CrossEncoder(self.model_name)

    except Exception as e:

      print(f"[Reranker disabled] Could not load '{self.model_name}'. Reason: {e}")

      self.ce = None

  def rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:

    self.load()

    if self.ce is None or not candidates:

      return candidates[:top_k]

    pairs = [(query, c.text) for c in candidates]

    scores = self.ce.predict(pairs)

    scored = list(zip(candidates, scores))

    scored.sort(key=lambda x: float(x[1]), reverse=True)

    return [c for c, _ in scored[:top_k]]


# ----------------------------

# Retrieval

# ----------------------------

def retrieve(

  query: str,

  embed_model: SentenceTransformer,

  faiss_index: faiss.Index,

  chunks: List[Chunk],

  bm25: BM25Okapi,

  reranker: Optional[Reranker]

) -> List[Chunk]:

 # Vector search

  q_emb = embed_query(embed_model, query)

  scores, idxs = faiss_index.search(q_emb, TOP_K_VECTOR) # type: ignore

  vec_idxs = [int(i) for i in idxs[0] if i >= 0]

 # BM25 search

  bm25_scores = bm25.get_scores(tokenize(query))

  bm25_top = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]

  bm25_idxs = [int(i) for i in bm25_top]

 # Merge unique up to TOP_K_PRE_RERANK

  merged: List[int] = []

  seen = set()

  for i in vec_idxs + bm25_idxs:

    if i not in seen:

      merged.append(i)

      seen.add(i)

    if len(merged) >= TOP_K_PRE_RERANK:

      break

  candidates = [chunks[i] for i in merged]

  if reranker is not None and USE_RERANKER:

    candidates = reranker.rerank(query, candidates, top_k=TOP_K_FINAL)

  return candidates[:TOP_K_FINAL]

def build_context_block(selected: List[Chunk]) -> str:

  blocks = []

  for k, ch in enumerate(selected, start=1):

    blocks.append(

      f"EXCERPT {k}\n"

      f"[source: {ch.source_file} p.{ch.page} chunk:{ch.chunk_id}]\n"

      f"{ch.text}\n"

    )

  return "\n".join(blocks).strip()


# ----------------------------

# LLM loading

# ----------------------------

def find_gguf_model(models_dir: Path) -> Path:

  ggufs = sorted(models_dir.glob("*.gguf"))

  if not ggufs:

    raise SystemExit(

      f"No .gguf files found in {models_dir.resolve()}\n"

      "Put your GGUF model file into ./models/"

    )

#  Prefer q4_k_m if present, else first file

  for p in ggufs:

    if "q4_k_m" in p.name.lower():

      return p

 # Prefer 00001-of-00002 if present (split models)

  for p in ggufs:

    if "00001-of-00002" in p.name.lower():

      return p

  return ggufs[0]

def make_llm(model_path: Path) -> Llama:

  if not model_path.exists():

    raise SystemExit(f"Model not found: {model_path.resolve()}")

  n_threads = max(4, (os.cpu_count() or 8) - 2)

 # CPU-only defaults that work well on laptops

  return Llama(

    model_path=str(model_path),
    n_ctx=LLM_N_CTX,
    n_threads=n_threads,
    n_batch=256,
    use_mmap=True,
    # flash_attn=True,
    verbose=False,

  )


# ----------------------------

# Answering

# ----------------------------

def answer_question(llm: Llama, query: str, context: str, chat_history: List[Tuple[str, str]]) -> str:

  trimmed = chat_history[-6:]

  messages = [{"role": "system", "content": SYSTEM_PROMPT}]

  for u, a in trimmed:

    messages.append({"role": "user", "content": u})

    messages.append({"role": "assistant", "content": a})

  user_prompt = (

    f"User question:\n{query}\n\n"

    f"Handbook excerpts:\n{context}\n\n"

    "Now answer the user. Remember: be direct, policy-specific, and cite pages."

  )

  messages.append({"role": "user", "content": user_prompt})

  try:

    out = llm.create_chat_completion(

      messages=messages, # type: ignore

      temperature=TEMPERATURE,

      max_tokens=MAX_NEW_TOKENS,

    )

    return out["choices"][0]["message"]["content"].strip() # type: ignore

  except Exception:

   # Fallback for models/templates that don't support chat completion cleanly

    plain = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nAssistant:"

    out = llm(

      plain,

      temperature=TEMPERATURE,

      max_tokens=MAX_NEW_TOKENS,

      stop=["\nUser question:", "\n\nUser:"]

    )

    return out["choices"][0]["text"].strip() # type: ignore


# ----------------------------

# Index build/load

# ----------------------------

def build_or_load(embed_model: SentenceTransformer) -> Tuple[faiss.Index, List[Chunk], BM25Okapi]:

  if FAISS_INDEX_PATH.exists() and META_PATH.exists() and BM25_PATH.exists():

    chunks = load_meta(META_PATH)

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    bm25 = load_bm25(BM25_PATH)

    return index, chunks, bm25

  print("Building index from PDFs (first run only)...")

  chunks = build_chunks(PDF_DIR)

  save_meta(chunks, META_PATH)

  passages = [c.text for c in chunks]

 # Determine embedding dim dynamically

  sample = embed_passages(embed_model, [passages[0]], batch_size=1)

  dim = int(sample.shape[1])

  embeddings = np.zeros((len(passages), dim), dtype=np.float32)

  batch = 64

  for start in tqdm(range(0, len(passages), batch), desc="Embedding"):

    end = min(start + batch, len(passages))

    embeddings[start:end] = embed_passages(embed_model, passages[start:end], batch_size=batch)

  index = build_faiss_index(embeddings)

  faiss.write_index(index, str(FAISS_INDEX_PATH))

  bm25 = build_bm25(chunks)

  save_bm25(bm25, BM25_PATH)

  print(f"Index built: {len(chunks)} chunks")

  return index, chunks, bm25


# ----------------------------

# UI helpers

# ----------------------------

def print_sources(selected: List[Chunk], max_to_print: int = 5) -> None:

  print("\nSources used (top excerpts):")

  for ch in selected[:max_to_print]:

    print(f"- {ch.source_file} p.{ch.page} chunk:{ch.chunk_id}")

    excerpt = ch.text.strip().replace("\n", " ")

    excerpt = excerpt[:420] + ("..." if len(excerpt) > 420 else "")

    print(f" {excerpt}")

def help_text() -> str:

  return (

    "\nCommands:\n"

    " /help    Show commands\n"

    " /sources  Show sources for the last answer\n"

    " /reindex  Rebuild index (if PDFs changed)\n"

    " /exit    Quit\n"

  )

def delete_index_files() -> None:

  for p in [FAISS_INDEX_PATH, META_PATH, BM25_PATH]:

    if p.exists():

      try:

        p.unlink()

      except Exception:

        pass


# ----------------------------

# Main

# ----------------------------

def main() -> None:

  if not PDF_DIR.exists():

    raise SystemExit(f"Missing folder: {PDF_DIR.resolve()} (create ./handbooks and put PDFs inside)")

  if not MODELS_DIR.exists():

    raise SystemExit(f"Missing folder: {MODELS_DIR.resolve()} (create ./models and put a .gguf inside)")

  pdfs = list(PDF_DIR.glob("*.pdf"))

  if not pdfs:

    raise SystemExit(f"No PDFs found in {PDF_DIR.resolve()}")

  model_path = find_gguf_model(MODELS_DIR)

  print(f"Using model: {model_path.name}")

 # Load embedding model

  embed_model = SentenceTransformer(EMBED_MODEL_NAME)

 # Build/load index

  faiss_index, chunks, bm25 = build_or_load(embed_model)

 # Load optional reranker

  reranker = Reranker(RERANKER_MODEL_NAME) if USE_RERANKER else None

 # Load LLM

  llm = make_llm(model_path)

  print("\nHR Handbook Q&A Chatbot (local)")

  print(help_text())

  chat_history: List[Tuple[str, str]] = []

  last_selected: List[Chunk] = []

  while True:

    q = input("You: ").strip()

    if not q:

      continue

    low = q.lower().strip()

    if low in {"/exit", "exit", "quit"}:

      break

    if low == "/help":

      print(help_text())

      continue

    if low == "/sources":

      if last_selected:

        print_sources(last_selected, MAX_SOURCE_EXCERPTS_TO_PRINT)

      else:

        print("No sources yet. Ask a question first.")

      continue

    if low == "/reindex":

      print("Rebuilding index...")

      delete_index_files()

      faiss_index, chunks, bm25 = build_or_load(embed_model)

      print("Done.")

      continue

    t0 = time.time()

    selected = retrieve(q, embed_model, faiss_index, chunks, bm25, reranker)

    last_selected = selected

    context = build_context_block(selected)

    a = answer_question(llm, q, context, chat_history)

    dt = time.time() - t0

    print(f"\nAssistant ({dt:.1f}s):\n{a}\n")

    if PRINT_SOURCES_EVERY_ANSWER:

      print_sources(selected, MAX_SOURCE_EXCERPTS_TO_PRINT)

    chat_history.append((q, a))


if __name__ == "__main__":

  main()