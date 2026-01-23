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
from typing import List, Dict, Optional, TypedDict
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
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # max_tokens = kwargs.get('max_tokens', 2048)
        
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
    weights=[0.6, 0.4],  # Favor semantic search
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
def rerank_with_llm(query: str, docs: List[Document], top_n: int = 10) -> List[Document]:
    """
    LLM-based reranking using structured output (LangChain 0.3+ pattern)
    """
    if len(docs) <= top_n:
        return docs
    
    # Use lightweight model for reranking
    rerank_llm = LLMProvider.get_llm("groq", model="llama-3.1-8b-instant", temperature=0)
    
    rerank_prompt = ChatPromptTemplate.from_template(
        """Rate the relevance of this document to the query on a scale of 1-10.

Query: {query}
Document: {doc}

Provide only the relevance score (1-10):"""
    )
    
    # Use with_structured_output (LangChain 0.3+ pattern)
    structured_llm = rerank_llm.with_structured_output(RatingScore)
    chain = rerank_prompt | structured_llm
    
    scored_docs = []
    start = time.time()
    
    # Parallel processing for speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for doc in docs:
            future = executor.submit(
                chain.invoke,
                {"query": query, "doc": doc.page_content[:500]}
            )
            futures.append((doc, future))
        
        for doc, future in futures:
            try:
                result = future.result(timeout=5)
                score = float(result.relevance_score)
            except Exception as e:
                print(f"Reranking error: {e}")
                score = 5.0
            scored_docs.append((doc, score))
    
    print(f"⏱️ Reranking time: {time.time() - start:.2f}s")
    
    # Sort and return top_n
    reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_n]]

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
# CONTEXT BUILDING WITH QUERY EXPANSION
# ============================================================================
def build_context_with_expansion(inputs: Dict, use_expansion: bool = True) -> Dict:
    """
    Build context with OPTIONAL query expansion + reranking
    
    This function now supports query expansion at the retrieval stage
    
    Args:
        inputs: Dict with "input" and optionally "docs"
        use_expansion: Enable query expansion (recommended for better recall)
    
    Strategy selection:
    1. With query expansion + RRF reranking (BEST ACCURACY - recommended)
    2. With query expansion + LLM reranking (HIGHEST ACCURACY but slower)
    3. Without expansion, with reranking (BALANCED)
    4. Without expansion, without reranking (FASTEST)
    """
    question = inputs["input"]
    
    # If docs are already provided (from retriever), use them
    # Otherwise, retrieve with query expansion
    if "docs" in inputs and inputs["docs"]:
        docs = list(inputs["docs"])
        logger.info(f"📚 Using pre-retrieved docs: {len(docs)}")
    else:
        # This branch is used when calling build_context directly
        if use_expansion:
            docs = retrieve_with_query_expansion(
                question,
                ensemble_retriever,
                num_queries=2,
                use_llm_expansion=True,
                fusion_method="rrf"
            )
        else:
            docs = ensemble_retriever.invoke(question)
    
    if not docs:
        return {"input": question, "context": "No relevant documents found."}
    
    # Apply reranking to the fused/retrieved documents
    # Use RRF reranking (10-50x faster than LLM)
    top_docs = rerank_with_rrf(query=question, docs=docs, top_n=10)
    
    # ALTERNATIVE: LLM reranking for higher accuracy (slower)
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
# RAG CHAINS (LangChain 0.3+ Patterns) - ALL WITH QUERY EXPANSION SUPPORT
# ============================================================================

# METHOD 1: Advanced chain with query expansion (RECOMMENDED for best accuracy)
def create_advanced_rag_chain_with_expansion(
    llm_provider: str = "groq",
    use_query_expansion: bool = True,
    num_queries: int = 2
):
    """
    Create RAG chain with query expansion using LangChain 0.3+ patterns
    
    This uses create_retrieval_chain with query expansion in the retrieval step
    BEST FOR: Maximum accuracy, production use
    """
    current_llm = LLMProvider.get_llm(llm_provider)
    
    if use_query_expansion:
        # Custom retriever with query expansion
        def expanded_retriever_func(query: str) -> List[Document]:
            return retrieve_with_query_expansion(
                question=query,
                retriever=ensemble_retriever,
                num_queries=num_queries,
                use_llm_expansion=True,
                fusion_method="rrf"
            )
        
        # Wrap in RunnableLambda for LCEL compatibility
        expanded_retriever = RunnableLambda(expanded_retriever_func)
    else:
        expanded_retriever = ensemble_retriever
    
    # Create the question-answer chain
    qa_chain = create_stuff_documents_chain(current_llm, prompt)
    
    # Create retrieval chain with expanded retriever
    rag_chain = create_retrieval_chain(expanded_retriever, qa_chain) # type: ignore
    
    return rag_chain

# METHOD 2: Custom LCEL chain with query expansion (More control)
def create_custom_rag_chain_with_expansion(
    llm_provider: str = "groq",
    use_query_expansion: bool = True,
    use_reranking: bool = True
):
    """
    Custom RAG chain with query expansion + reranking using LCEL
    BEST FOR: Maximum control over the pipeline
    """
    current_llm = LLMProvider.get_llm(llm_provider)
    
    if use_query_expansion:
        # Retrieval with query expansion
        def retrieve_and_expand(question: str) -> List[Document]:
            return retrieve_with_query_expansion(
                question=question,
                retriever=ensemble_retriever,
                num_queries=2,
                use_llm_expansion=True,
                fusion_method="rrf"
            )
        
        retriever_step = RunnableLambda(retrieve_and_expand)
    else:
        retriever_step = ensemble_retriever
    
    # Build chain with optional reranking
    if use_reranking:
        chain = (
            {
                "input": RunnablePassthrough(),
                "docs": RunnablePassthrough() | retriever_step,
            }
            | RunnableLambda(lambda x: build_context_with_expansion(x, use_expansion=False))  # Expansion already done # type: ignore
            | prompt
            | current_llm
            | StrOutputParser()
        )
    else:
        # Simple chain without reranking
        def format_docs(docs):
            return "\n\n".join(
                f"[doc_{i+1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
        
        chain = (
            {
                "context": retriever_step | format_docs,
                "input": RunnablePassthrough()
            }
            | prompt
            | current_llm
            | StrOutputParser()
        )
    
    return chain

# METHOD 3: Simple RAG with optional query expansion (Balanced)
def create_simple_rag_chain_with_expansion(
    llm_provider: str = "groq",
    use_query_expansion: bool = False  # Default OFF for speed
):
    """
    Simple RAG chain with optional query expansion
    BEST FOR: Fast responses when expansion is not needed
    """
    current_llm = LLMProvider.get_llm(llm_provider)
    
    def format_docs(docs):
        return "\n\n".join(
            f"[doc_{i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
    
    if use_query_expansion:
        retriever_step = RunnableLambda(
            lambda q: retrieve_with_query_expansion(
                q, ensemble_retriever, num_queries=2,  # type: ignore
                use_llm_expansion=False,  # Use rule-based for speed
                fusion_method="simple"
            )
        )
    else:
        retriever_step = ensemble_retriever
    
    chain = (
        {
            "context": retriever_step | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | current_llm
        | StrOutputParser()
    )
    
    return chain

# ============================================================================
# INITIALIZE DEFAULT CHAINS WITH QUERY EXPANSION
# ============================================================================

# RECOMMENDED: Advanced chain with query expansion (best accuracy)
advanced_rag_chain = create_advanced_rag_chain_with_expansion(
    "groq",
    use_query_expansion=True,  # Enable query expansion by default
    num_queries=2
)

# Custom chain with full control
custom_rag_chain = create_custom_rag_chain_with_expansion(
    "groq",
    use_query_expansion=True,  # Enable query expansion
    use_reranking=True  # Enable reranking
)

# Simple chain (fast, expansion optional)
simple_rag_chain = create_simple_rag_chain_with_expansion(
    "groq",
    use_query_expansion=False  # Disabled for speed
)

# ALTERNATIVE: Create chains without query expansion (for backward compatibility)
def create_advanced_rag_chain(llm_provider: str = "groq"):
    """Backward compatible - without query expansion"""
    return create_advanced_rag_chain_with_expansion(llm_provider, use_query_expansion=False)

def create_custom_rag_chain(llm_provider: str = "groq"):
    """Backward compatible - without query expansion"""
    return create_custom_rag_chain_with_expansion(llm_provider, use_query_expansion=False, use_reranking=True)

def create_simple_rag_chain(llm_provider: str = "groq"):
    """Backward compatible - without query expansion"""
    return create_simple_rag_chain_with_expansion(llm_provider, use_query_expansion=False)

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
        "input": question,
        "answer": answer,
        "context": context[:2000]
    })
    
    return {
        "confidence": result.score, # type: ignore
        "reasoning": result.reasoning # type: ignore
    }

# ============================================================================
# QUERY EXPANSION (For better retrieval coverage) - INTEGRATED INTO ALL METHODS
# ============================================================================
def expand_query(question: str, num_queries: int = 2, use_llm: bool = True) -> List[str]:
    """
    Generate multiple query variations for better retrieval coverage
    
    Two strategies:
    1. LLM-based expansion (more semantic, slower)
    2. Rule-based expansion (faster, simpler)
    """
    if use_llm:
        expansion_prompt = ChatPromptTemplate.from_template(
            """Generate {num_queries} alternative phrasings of this question that capture different aspects:

Question: {question}

Requirements:
- Each alternative should focus on a different angle
- Keep the core intent but vary the wording
- Make them suitable for search

Alternatives (one per line):"""
        )
        
        chain = expansion_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "num_queries": num_queries})
        
        alternatives = [q.strip() for q in result.split("\n") if q.strip()]
        return [question] + alternatives[:num_queries]
    else:
        # Rule-based expansion (faster)
        alternatives = [question]
        
        # Add variations with different question words
        if question.lower().startswith(("what", "how", "why", "when", "where", "who")):
            # Extract the core topic after question word
            core = " ".join(question.split()[1:])
            alternatives.append(f"Explain {core}")
            alternatives.append(f"Information about {core}")
        
        return alternatives[:num_queries + 1]

# OPTIMIZATION: Multi-query retrieval with fusion
def retrieve_with_query_expansion(
    question: str,
    retriever,
    num_queries: int = 2,
    use_llm_expansion: bool = True,
    fusion_method: str = "rrf"  # "rrf" or "simple"
) -> List[Document]:
    """
    Retrieve documents using multiple query variations and fuse results
    
    This is the CORE function that integrates query expansion into retrieval
    
    Args:
        question: Original user question
        retriever: Any LangChain retriever (ensemble, semantic, BM25, etc.)
        num_queries: Number of alternative queries to generate
        use_llm_expansion: Use LLM for expansion (slower but better)
        fusion_method: "rrf" for Reciprocal Rank Fusion or "simple" for deduplication
    
    Returns:
        Fused list of documents from all query variations
    """
    # Generate query variations
    queries = expand_query(question, num_queries, use_llm=use_llm_expansion)
    logger.info(f"🔍 Query expansion: {len(queries)} variations generated")
    
    if fusion_method == "rrf":
        # Reciprocal Rank Fusion - better quality
        doc_scores = {}  # doc_id -> score
        doc_map = {}     # doc_id -> Document
        
        k = 60  # RRF constant
        
        for query_idx, query in enumerate(queries):
            try:
                docs = retriever.invoke(query)
                
                for rank, doc in enumerate(docs):
                    # Create unique doc ID
                    doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                    
                    # RRF score: 1 / (k + rank)
                    score = 1.0 / (k + rank + 1)
                    
                    if doc_id in doc_scores:
                        doc_scores[doc_id] += score
                    else:
                        doc_scores[doc_id] = score
                        doc_map[doc_id] = doc
                        
            except Exception as e:
                logger.warning(f"⚠️ Query expansion retrieval failed for '{query}': {e}")
                continue
        
        # Sort by fused scores
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        fused_docs = [doc_map[doc_id] for doc_id in sorted_doc_ids[:20]]
        
        logger.info(f"✅ RRF fusion: {len(fused_docs)} unique documents")
        return fused_docs
        
    else:
        # Simple deduplication - faster
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                docs = retriever.invoke(query)
                
                for doc in docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
                        
            except Exception as e:
                logger.warning(f"⚠️ Query expansion retrieval failed for '{query}': {e}")
                continue
        
        logger.info(f"✅ Simple fusion: {len(all_docs[:20])} unique documents")
        return all_docs[:20]

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
      # Need to import for query expansion
    
    question = "What are the main features of the product?"
    
    print("=" * 80)
    print("RAG CHAIN EXAMPLES (LangChain 0.3+ with Query Expansion)")
    print("=" * 80)
    
    # Example 1: Advanced RAG with query expansion (RECOMMENDED)
    print("\n1️⃣ Advanced RAG Chain WITH Query Expansion (Best Accuracy):")
    start = time.time()
    response = advanced_rag_chain.invoke({"input": question})
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response['answer'][:200]}...")
    
    # Example 2: Custom chain with expansion + reranking
    print("\n2️⃣ Custom RAG Chain (Query Expansion + RRF Reranking):")
    start = time.time()
    response = custom_rag_chain.invoke(question)
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response[:200]}...")
    
    # Example 3: Simple chain without expansion (fastest)
    print("\n3️⃣ Simple RAG Chain (No Query Expansion - Fastest):")
    start = time.time()
    response = simple_rag_chain.invoke(question)
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response[:200]}...")
    
    # Example 4: Demonstrate query expansion
    print("\n4️⃣ Query Expansion Demo:")
    expanded_queries = expand_query(question, num_queries=2, use_llm=True)
    print(f"🔍 Original: {question}")
    print(f"🔍 Expanded queries:")
    for i, q in enumerate(expanded_queries[1:], 1):
        print(f"   {i}. {q}")
    
    # Example 5: Create chain with specific configuration
    print("\n5️⃣ Custom Configuration Example:")
    custom_config_chain = create_advanced_rag_chain_with_expansion(
        llm_provider="groq",
        use_query_expansion=True,
        num_queries=3  # Generate 3 query variations
    )
    start = time.time()
    response = custom_config_chain.invoke({"input": question})
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response['answer'][:200]}...")
    
    # Example 6: Switch LLM provider with expansion
    print("\n6️⃣ Using different LLM provider (Google Gemini) with Query Expansion:")
    gemini_chain = create_custom_rag_chain_with_expansion(
        "google_ai",
        use_query_expansion=True,
        use_reranking=True
    )
    start = time.time()
    response = gemini_chain.invoke(question)
    print(f"⏱️ Latency: {time.time() - start:.2f}s")
    print(f"📝 Answer: {response[:200]}...")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION:")
    print("   - Use advanced_rag_chain for best accuracy (query expansion ON)")
    print("   - Use custom_rag_chain for full control (expansion + reranking)")
    print("   - Use simple_rag_chain for fastest responses (expansion OFF)")
    print("=" * 80)