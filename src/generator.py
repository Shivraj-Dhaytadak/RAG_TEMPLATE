from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma  # Updated from langchain_chroma.vectorstores
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import List, Dict, Optional
from langchain.schema import Document
from pydantic import BaseModel, Field
from rich import print
from dotenv import load_dotenv
import pickle
import os
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL = os.getenv('MODEL_ID', 'llama-3.3-70b-versatile')
DB_URL = os.getenv("DB")
PER_DIR = os.getenv("PERSISTENT_DIR", "stats/chroma_semantic")
BM25_DIR = "stats/bm25_retriever.pkl"

# ============================================================================
# MULTI-LLM PROVIDER FACTORY (LangChain 0.3+ Compatible)
# ============================================================================
class LLMProvider:
    """Factory for creating different LLM providers following LangChain 0.3+ patterns"""
    
    @staticmethod
    def get_llm(provider: str = "groq", **kwargs):
        """
        Get LLM based on provider (LangChain 0.3+ compatible)
        Options: groq, openrouter, google_ai, ollama
        """
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        if provider == "groq":
            return ChatGroq(
                model=kwargs.get('model', "llama-3.3-70b-versatile"),
                temperature=temperature,
                api_key=GROQ_API_KEY, # type: ignore
                # max_tokens=max_tokens
            )
        
        elif provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            return ChatOpenAI(
                model=kwargs.get('model', "anthropic/claude-3.5-sonnet"),
                temperature=temperature,
                api_key=OPENROUTER_API_KEY, # type: ignore
                base_url="https://openrouter.ai/api/v1",
                # max_tokens=max_tokens
            )
        
        elif provider == "google_ai":
            return ChatGoogleGenerativeAI(
                model=kwargs.get('model', "gemini-1.5-pro"),
                temperature=temperature,
                google_api_key=GEMINI_API_KEY,
                # max_output_tokens=max_tokens
            )
        
        elif provider == "ollama":
            return ChatOllama(
                model=kwargs.get('model', "llama3.1:8b"),
                temperature=temperature,
                base_url=kwargs.get('base_url', "http://localhost:11434")
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")

# Initialize default LLM
llm = LLMProvider.get_llm("groq", temperature=0.1)

# ============================================================================
# EMBEDDINGS & RETRIEVAL SETUP (LangChain 0.3+ Compatible)
# ============================================================================
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Use langchain_chroma.Chroma (not langchain_community)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=PER_DIR,
    collection_name="agentic_rag_collection"  # Explicit collection name
)

# OPTIMIZATION: Use similarity_score_threshold for better filtering
semantic_retriever = vector_store.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={
        "k": 15,  # Retrieve top 15 documents
        "score_threshold": 0.7  # Only docs with score > 0.5
    }
)

# ALTERNATIVE: Use MMR for diversity
# semantic_retriever = vector_store.as_retriever(
#     search_type='mmr',
#     search_kwargs={
#         "k": 15,
#         "fetch_k": 30,  # Fetch 30, return 15 with diversity
#         "lambda_mult": 0.7  # Balance relevance vs diversity
#     }
# )

# Load BM25 retriever
with open(BM25_DIR, "rb") as f:
    bm25_retriever = pickle.load(f)

# OPTIMIZATION: Weighted RRF with EnsembleRetriever (LangChain 0.3+)
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.7, 0.3],  # Favor semantic search
    c=60  # RRF constant (default)
)

# ============================================================================
# OPTIMIZED PROMPTS (Following LangChain 0.3+ Best Practices)
# ============================================================================
RAG_PROMPT_TEMPLATE = """You are an AI assistant that answers questions using ONLY the provided context documents.

Core rules:
- Answer based solely on the context provided
- If information is not in the context, say "I don't know based on the provided documents"
- Never fabricate information, citations, or URLs
- Be concise and accurate

When answering:
1. Identify relevant information in the context
2. Synthesize information from multiple sources if needed
3. Cite sources using [doc_i] notation

Context documents:
{context}

Question: {input}

Answer:"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# ALTERNATIVE: Multi-turn conversation prompt
CONVERSATIONAL_PROMPT = """You are an AI assistant helping with questions based on documents.

Context:
{context}

Chat History:
{chat_history}

Current Question: {input}

Answer based on the context and chat history:"""

# ============================================================================
# RERANKING STRATEGIES
# ============================================================================

class RatingScore(BaseModel):
    """Pydantic model for structured output"""
    relevance_score: int = Field(
        ..., 
        description="Relevance score (1-10) of document to query",
        ge=1, 
        le=10
    )

# OPTIMIZATION 1: Fast RRF Reranking (No LLM calls - 10-50x faster)
def rerank_with_rrf(query: str, docs: List[Document], top_n: int = 10, k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion reranking - much faster than LLM reranking
    Uses embedding similarity for ranking
    """
    if len(docs) <= top_n:
        return docs
    
    # Get semantic similarity scores
    query_embedding = embeddings.embed_query(query)
    
    rrf_scores = {}
    for idx, doc in enumerate(docs):
        # Calculate semantic similarity
        doc_embedding = embeddings.embed_documents([doc.page_content])[0]
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        # RRF score
        rrf_scores[idx] = 1.0 / (k + idx + 1) + similarity * 0.5
    
    # Sort by RRF score
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [docs[i] for i in sorted_indices[:top_n]]

# OPTIMIZATION 2: LLM-based reranking (slower but more accurate)
# def rerank_with_llm(query: str, docs: List[Document], top_n: int = 10) -> List[Document]:
#     """
#     LLM-based reranking using structured output (LangChain 0.3+ pattern)
#     """
#     if len(docs) <= top_n:
#         return docs
    
#     # Use lightweight model for reranking
#     rerank_llm = LLMProvider.get_llm("groq", model="llama-3.1-8b-instant", temperature=0)
    
#     rerank_prompt = ChatPromptTemplate.from_template(
#         """Rate the relevance of this document to the query on a scale of 1-10.

# Query: {query}
# Document: {doc}

# Provide only the relevance score (1-10):"""
#     )
    
#     # Use with_structured_output (LangChain 0.3+ pattern)
#     structured_llm = rerank_llm.with_structured_output(RatingScore)
#     chain = rerank_prompt | structured_llm
    
#     scored_docs = []
#     start = time.time()
    
#     # Parallel processing for speed
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = []
#         for doc in docs:
#             future = executor.submit(
#                 chain.invoke,
#                 {"query": query, "doc": doc.page_content[:500]}
#             )
#             futures.append((doc, future))
        
#         for doc, future in futures:
#             try:
#                 result = future.result(timeout=5)
#                 score = float(result.relevance_score)
#             except Exception as e:
#                 print(f"Reranking error: {e}")
#                 score = 5.0
#             scored_docs.append((doc, score))
    
#     print(f"⏱️ Reranking time: {time.time() - start:.2f}s")
    
#     # Sort and return top_n
#     reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
#     return [doc for doc, _ in reranked[:top_n]]

# OPTIMIZATION 3: Cohere Reranker (production-grade, requires API key)
# from langchain.retrievers.document_compressors import CohereRerank
# from langchain.retrievers import ContextualCompressionRetriever
# 
# cohere_reranker = CohereRerank(
#     model="rerank-english-v3.0",
#     top_n=10,
#     cohere_api_key=os.getenv("COHERE_API_KEY")
# )
# 
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=cohere_reranker,
#     base_retriever=ensemble_retriever
# )

# ============================================================================
# CONTEXT BUILDING
# ============================================================================
def build_context(inputs: Dict) -> Dict:
    """
    Build context with reranking
    
    Strategy selection:
    1. rerank_with_rrf: Fast, good for production (recommended)
    2. rerank_with_llm: Slower, more accurate
    3. No reranking: Fastest, lower accuracy
    """
    question = inputs["input"]
    docs = list(inputs.get("docs", []))
    
    if not docs:
        return {"input": question, "context": "No relevant documents found."}
    
    # Use RRF reranking (10-50x faster than LLM)
    top_docs = rerank_with_rrf(query=question, docs=docs, top_n=15)
    
    # ALTERNATIVE: LLM reranking for higher accuracy
    # top_docs = rerank_with_llm(query=question, docs=docs, top_n=10)
    
    # ALTERNATIVE: No reranking (fastest)
    # top_docs = docs[:10]
    
    # Format context
    context_text = "\n\n".join(
        f"[doc_{i+1}]\nSource: {doc.metadata.get('source', 'unknown')}\nContent:\n{doc.page_content}"
        for i, doc in enumerate(top_docs)
    )
    
    return {
        "input": question,
        "context": context_text,
    }

# ============================================================================
# RAG CHAINS (LangChain 0.3+ Patterns)
# ============================================================================

# METHOD 1: Using create_retrieval_chain (Recommended for LangChain 0.3+)
def create_advanced_rag_chain(llm_provider: str = "groq"):
    """
    Create RAG chain using LangChain 0.3+ recommended pattern
    with create_retrieval_chain
    """
    current_llm = LLMProvider.get_llm(llm_provider)
    
    # Create the question-answer chain
    qa_chain = create_stuff_documents_chain(current_llm, prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(ensemble_retriever, qa_chain)
    
    return rag_chain

# METHOD 2: Custom LCEL chain with reranking (More control)
def create_custom_rag_chain(llm_provider: str = "groq"):
    """
    Custom RAG chain with reranking using LCEL
    """
    current_llm = LLMProvider.get_llm(llm_provider)
    
    chain = (
        {
            "input": RunnablePassthrough(),
            "docs": RunnablePassthrough() | ensemble_retriever,
        }
        | RunnableLambda(build_context)
        | prompt
        | current_llm
        | StrOutputParser()
    )
    
    return chain

# METHOD 3: Simple RAG without reranking (Fastest)
def create_simple_rag_chain(llm_provider: str = "groq"):
    """
    Simple RAG chain without reranking
    """
    current_llm = LLMProvider.get_llm(llm_provider)
    
    def format_docs(docs):
        return "\n\n".join(
            f"[doc_{i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
    
    chain = (
        {
            "context": ensemble_retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | current_llm
        | StrOutputParser()
    )
    
    return chain

# Initialize default chains
advanced_rag_chain = create_advanced_rag_chain("groq")
custom_rag_chain = create_custom_rag_chain("groq")
simple_rag_chain = create_simple_rag_chain("groq")

# ============================================================================
# STREAMING SUPPORT (LangChain 0.3+ Async Pattern)
# ============================================================================
async def stream_rag_response(question: str, llm_provider: str = "groq"):
    """
    Stream RAG response for better UX
    Uses LangChain 0.3+ streaming pattern
    """
    stream_llm = LLMProvider.get_llm(llm_provider)
    
    # Create streaming chain
    stream_chain = create_custom_rag_chain(llm_provider)
    
    async for chunk in stream_chain.astream(question):
        if isinstance(chunk, str):
            yield chunk
        elif hasattr(chunk, 'content'):
            yield chunk.content

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================
class ConfidenceScore(BaseModel):
    score: float = Field(..., description="Confidence score 0-1", ge=0, le=1)
    reasoning: str = Field(..., description="Brief reasoning")

def get_answer_confidence(question: str, answer: str, context: str) -> Dict:
    """
    Evaluate confidence using structured output
    """
    confidence_prompt = ChatPromptTemplate.from_template(
        """Rate your confidence (0-1) that this answer is fully supported by the context.

Question: {question}
Answer: {answer}
Context: {context}

Provide confidence score and reasoning:"""
    )
    
    structured_llm = llm.with_structured_output(ConfidenceScore)
    chain = confidence_prompt | structured_llm
    
    result = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context[:2000]
    })
    
    return {
        "confidence": result.score, # type: ignore
        "reasoning": result.reasoning # type: ignore
    }

# ============================================================================
# QUERY EXPANSION (For better retrieval coverage)
# ============================================================================
def expand_query(question: str, num_queries: int = 2) -> List[str]:
    """
    Generate multiple query variations for better retrieval
    """
    expansion_prompt = ChatPromptTemplate.from_template(
        """Generate {num_queries} alternative phrasings of this question:

Question: {question}

Alternatives (one per line):"""
    )
    
    chain = expansion_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "num_queries": num_queries})
    
    alternatives = [q.strip() for q in result.split("\n") if q.strip()]
    return [question] + alternatives[:num_queries]

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    question = "what is K means Clustering and algorithm to create a model ?"
    
    print("=" * 80)
    print("RAG CHAIN EXAMPLES (LangChain 0.3+ Compatible)")
    print("=" * 80)
    
    # Example 1: Advanced RAG chain (Recommended)
    print("\n1️⃣ Advanced RAG Chain (create_retrieval_chain pattern):")
    start = time.time()
    response = advanced_rag_chain.invoke({"input": question})
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response['answer']}")
    
    # Example 2: Custom chain with reranking
    print("\n2️⃣ Custom RAG Chain (with RRF reranking):")
    start = time.time()
    response = custom_rag_chain.invoke(question)
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response}")
    
    # Example 3: Switch LLM provider
    # print("\n3️⃣ Using different LLM provider (Google Gemini):")
    # gemini_chain = create_simple_rag_chain("google_ai")
    # start = time.time()
    # response = gemini_chain.invoke(question)
    # print(f"⏱️ Latency: {time.time() - start:.2f}s")
    # print(f"📝 Answer: {response[:200]}...")
    
    # Example 4: Query expansion
    print("\n4️⃣ Query Expansion:")
    expanded_queries = expand_query(question)
    print(f"🔍 Expanded queries: {expanded_queries}")