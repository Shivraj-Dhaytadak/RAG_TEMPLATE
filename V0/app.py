from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
from langchain_core.documents import Document
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
import asyncio

# Import RAG components (ensure these imports work with LangChain 0.3+)
from src.generator import (
    create_advanced_rag_chain_with_expansion,
    create_custom_rag_chain_with_expansion,
    create_simple_rag_chain_with_expansion,
    LLMProvider,
    create_simple_rag_chain,
    stream_rag_response,
    get_answer_confidence,
    expand_query,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN MANAGEMENT (FastAPI 0.100+ pattern)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management - LangChain 0.3+ compatible
    Load models on startup, cleanup on shutdown
    """
    logger.info("🚀 Starting RAG API server (LangChain 0.3+ compatible)...")

    # Warm up models for faster first request
    try:
        logger.info("🔥 Warming up models...")
        test_chain = create_simple_rag_chain("groq")
        _ = test_chain.invoke("test warmup query")
        logger.info("✅ Models warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model warmup failed: {e}")

    yield

    # Cleanup on shutdown
    logger.info("🛑 Shutting down RAG API server...")


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Production RAG API",
    description="Production-grade RAG API with LangChain 0.3+ and multiple LLM providers",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your security needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS (Pydantic V2 compatible)
# ============================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1)
    llm_provider: Optional[str] = Field(
        "groq", description="LLM provider: groq, openrouter, google_ai, ollama"
    )
    model: Optional[str] = Field(
        "llama-3.3-70b-versatile", description="Specific model name"
    )
    temperature: Optional[float] = Field(0.1, ge=0, le=1)
    use_reranking: Optional[bool] = Field(
        True, description="Use reranking for better accuracy"
    )
    use_query_expansion: Optional[bool] = Field(
        True, description="Use query expansion for better recall (RECOMMENDED)"
    )
    num_query_variations: Optional[int] = Field(
        2, ge=1, le=5, description="Number of query variations to generate"
    )
    chain_type: Optional[str] = Field(
        "advanced", description="Chain type: advanced, custom, simple"
    )
    top_k: Optional[int] = Field(
        10, ge=1, le=50, description="Number of documents to retrieve"
    )
    stream: Optional[bool] = Field(False, description="Stream response")
    include_confidence: Optional[bool] = Field(
        False, description="Include confidence score"
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[Document] = []
    latency_ms: float
    timestamp: str
    confidence: Optional[Dict] = None
    metadata: Optional[Dict] = None
    query_variations: Optional[List[str]] = None  # Added to show expanded queries


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    langchain_version: Optional[str] = None


class IngestRequest(BaseModel):
    file_path: str
    chunking_strategy: Optional[str] = Field(
        "recursive", description="Chunking strategy: recursive, semantic, parent_child"
    )
    chunk_size: Optional[int] = Field(1000, ge=100, le=4000)
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000)
    force_reingest: Optional[bool] = False


class IngestResponse(BaseModel):
    message: str
    file_path: str
    status: str
    task_id: Optional[str] = None


# ============================================================================
# RATE LIMITING (Simple in-memory implementation)
# ============================================================================
from collections import defaultdict
from datetime import timedelta


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests = defaultdict(list)
        self.limit = requests_per_minute

    def is_allowed(self, client_id: str) -> bool:
        now = datetime.now()
        # Remove old requests
        self.requests[client_id] = [
            req_time
            for req_time in self.requests[client_id]
            if now - req_time < timedelta(minutes=1)
        ]

        if len(self.requests[client_id]) >= self.limit:
            return False

        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=60)


# ============================================================================
# DEPENDENCY INJECTION (for LLM caching with query expansion support)
# ============================================================================
@lru_cache(maxsize=20)
def get_cached_chain(
    chain_type: str,
    provider: str,
    use_query_expansion: bool = True,
    use_reranking: bool = True,
    num_queries: int = 2,
    model: Optional[str] = None,
):
    """
    Cache chain instances to avoid repeated initialization
    LangChain 0.3+ compatible with query expansion support
    """
    logger.info(
        f"🔧 Creating chain: type={chain_type}, provider={provider}, expansion={use_query_expansion}, reranking={use_reranking}"
    )

    if chain_type == "advanced":
        return create_advanced_rag_chain_with_expansion(
            provider, use_query_expansion=use_query_expansion, num_queries=num_queries
        )
    elif chain_type == "custom":
        return create_custom_rag_chain_with_expansion(
            provider,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
        )
    else:  # simple
        return create_simple_rag_chain_with_expansion(
            provider, use_query_expansion=use_query_expansion
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Try to get real IP if behind proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host if request.client else "unknown"


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/", response_model=Dict)
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Production RAG API - LangChain 0.3+ Compatible",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with LangChain version info
    """
    import langchain

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        langchain_version=langchain.__version__,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request_data: QueryRequest, req: Request):
    """
    Main RAG query endpoint with query expansion support (LangChain 0.3+ compatible)

    Features:
    - Query expansion for better recall (RECOMMENDED)
    - Multiple chain types (advanced, custom, simple)
    - Multiple LLM providers
    - Optional reranking for accuracy
    - Confidence scoring

    Example:
    ```json
    {
        "question": "What is the main topic?",
        "llm_provider": "groq",
        "chain_type": "advanced",
        "use_query_expansion": true,
        "num_query_variations": 2,
        "use_reranking": true
    }
    ```
    """
    start_time = time.time()

    # Rate limiting
    client_id = get_client_id(req)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        logger.info(
            f"📝 Query received: {request_data.question[:100]}... from {client_id}"
        )
        logger.info(
            f"⚙️ Config: expansion={request_data.use_query_expansion}, reranking={request_data.use_reranking}, chain={request_data.chain_type}"
        )

        # Generate query variations if expansion is enabled (for transparency)
        query_variations = None
        if request_data.use_query_expansion:
            query_variations = expand_query(
                request_data.question,
                num_queries=request_data.num_query_variations,  # type: ignore
                use_llm=True,
            )
            logger.info(f"🔍 Query variations: {query_variations}")

        # Get or create chain with query expansion configuration
        chain = get_cached_chain(
            request_data.chain_type,
            request_data.llm_provider,
            use_query_expansion=request_data.use_query_expansion,
            use_reranking=request_data.use_reranking,
            num_queries=request_data.num_query_variations,
            model=request_data.model,
        )

        # Execute query based on chain type
        if request_data.chain_type == "advanced":
            # Advanced chain returns dict with 'answer' key
            result = chain.invoke({"input": request_data.question})
            answer = result.get("answer", str(result))  # type: ignore
            sources = result.get("context", [])  # type: ignore
        else:
            # Custom/simple chains return string directly
            answer = chain.invoke(request_data.question)
            sources = []

        latency_ms = (time.time() - start_time) * 1000

        response = QueryResponse(
            answer=answer,
            sources=sources if isinstance(sources, list) else [],
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat(),
            query_variations=query_variations,  # Include query variations
            metadata={
                "chain_type": request_data.chain_type,
                "provider": request_data.llm_provider,
                "model": request_data.model,
                "query_expansion": request_data.use_query_expansion,
                "reranking": request_data.use_reranking,
                "num_variations": (
                    request_data.num_query_variations
                    if request_data.use_query_expansion
                    else 0
                ),
            },
        )

        # Add confidence score if requested
        if request_data.include_confidence and isinstance(answer, str):
            try:
                confidence = get_answer_confidence(
                    request_data.question, answer, str(sources)[:2000]
                )
                response.confidence = confidence
            except Exception as e:
                logger.warning(f"⚠️ Confidence scoring failed: {e}")

        logger.info(f"✅ Query completed in {latency_ms:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream")
async def query_stream(request_data: QueryRequest):
    """
    Streaming query endpoint for real-time response (LangChain 0.3+ compatible)

    Returns Server-Sent Events (SSE) stream

    Example:
    ```json
    {
        "question": "What is the main topic?",
        "llm_provider": "groq",
        "stream": true
    }
    ```
    """

    async def generate():
        try:
            logger.info(f"🌊 Streaming query: {request_data.question[:100]}...")

            async for chunk in stream_rag_response(
                request_data.question, request_data.llm_provider or "groq"
            ):
                # Send as SSE format
                yield f"data: {chunk}\n\n"

        except Exception as e:
            logger.error(f"❌ Streaming failed: {str(e)}")
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# @app.post("/ingest", response_model=IngestResponse)
# async def ingest_document(
#     request_data: IngestRequest, background_tasks: BackgroundTasks
# ):
#     """
#     Trigger document ingestion (runs in background)

#     Example:
#     ```json
#     {
#         "file_path": "path/to/document.pdf",
#         "chunking_strategy": "recursive",
#         "chunk_size": 1000,
#         "chunk_overlap": 200
#     }
#     ```
#     """
#     import uuid
#     from ingestion import ingest_documents

#     task_id = str(uuid.uuid4())

#     def run_ingestion():
#         try:
#             logger.info(f"🔄 Starting ingestion task {task_id}")
#             ingest_documents(
#                 file_path=request_data.file_path,
#                 chunking_strategy=request_data.chunking_strategy,  # type: ignore
#                 chunk_size=request_data.chunk_size,  # type: ignore
#                 chunk_overlap=request_data.chunk_overlap,  # type: ignore
#                 force_reingest=request_data.force_reingest,  # type: ignore
#             )
#             logger.info(f"✅ Ingestion task {task_id} completed")
#         except Exception as e:
#             logger.error(f"❌ Ingestion task {task_id} failed: {str(e)}")

#     background_tasks.add_task(run_ingestion)

#     return IngestResponse(
#         message="Ingestion started in background",
#         file_path=request_data.file_path,
#         status="processing",
#         task_id=task_id,
#     )


@app.get("/models")
async def list_models():
    """
    List available LLM providers, models, and features
    """
    return {
        "providers": {
            "groq": {
                "models": [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                ],
                "default": "llama-3.3-70b-versatile",
                "speed": "fastest",
            },
            "openrouter": {
                "models": [
                    "anthropic/claude-3.5-sonnet",
                    "openai/gpt-4-turbo",
                    "meta-llama/llama-3.1-70b-instruct",
                ],
                "default": "anthropic/claude-3.5-sonnet",
                "speed": "moderate",
            },
            "google_ai": {
                "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
                "default": "gemini-1.5-pro",
                "speed": "moderate",
            },
            "ollama": {
                "models": ["llama3.1:8b", "mistral", "mixtral"],
                "default": "llama3.1:8b",
                "speed": "variable (depends on local hardware)",
                "note": "Requires Ollama running locally",
            },
        },
        "chain_types": {
            "advanced": {
                "description": "Using create_retrieval_chain (LangChain 0.3+ recommended)",
                "features": [
                    "Query expansion",
                    "Document retrieval",
                    "Answer generation",
                ],
                "best_for": "Maximum accuracy",
            },
            "custom": {
                "description": "Custom LCEL chain with reranking",
                "features": ["Query expansion", "RRF/LLM reranking", "Full control"],
                "best_for": "Balanced accuracy and speed",
            },
            "simple": {
                "description": "Basic RAG without reranking (fastest)",
                "features": ["Optional query expansion", "Direct retrieval"],
                "best_for": "Speed and simplicity",
            },
        },
        "features": {
            "query_expansion": {
                "description": "Generate multiple query variations for better recall",
                "impact": "10-20% accuracy improvement, +0.5-1s latency",
                "recommended": True,
                "options": {
                    "num_variations": "1-5 (default: 2)",
                    "llm_based": "More semantic (slower)",
                    "rule_based": "Faster but simpler",
                },
            },
            "reranking": {
                "description": "Reorder retrieved documents by relevance",
                "methods": {
                    "rrf": "Reciprocal Rank Fusion (fast, 10-50x faster than LLM)",
                    "llm": "LLM-based scoring (more accurate, slower)",
                },
                "impact": "5-15% accuracy improvement, +0.2-2s latency",
                "recommended": True,
            },
        },
        "performance_guide": {
            "maximum_accuracy": {
                "chain_type": "advanced",
                "query_expansion": True,
                "num_variations": 3,
                "reranking": True,
                "expected_latency": "3-5s",
            },
            "balanced": {
                "chain_type": "custom",
                "query_expansion": True,
                "num_variations": 2,
                "reranking": True,
                "expected_latency": "1-3s",
            },
            "maximum_speed": {
                "chain_type": "simple",
                "query_expansion": False,
                "reranking": False,
                "expected_latency": "0.5-1.5s",
            },
        },
    }


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        import psutil

        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except ImportError:
        return {
            "message": "Install psutil for system stats: pip install psutil",
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# BATCH PROCESSING
# ============================================================================
class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., min_items=1, max_items=10)  # type: ignore
    llm_provider: Optional[str] = "groq"
    chain_type: Optional[str] = "simple"
    use_query_expansion: Optional[bool] = True  # Enable by default
    use_reranking: Optional[bool] = False  # Disable for batch speed


@app.post("/query/batch")
async def query_batch(request_data: BatchQueryRequest):
    """
    Process multiple queries in batch with query expansion support (max 10 at once)

    Example:
    ```json
    {
        "questions": ["Question 1?", "Question 2?"],
        "llm_provider": "groq",
        "chain_type": "simple",
        "use_query_expansion": true,
        "use_reranking": false
    }
    ```
    """
    results = []

    # Get chain once for all queries with expansion configuration
    chain = get_cached_chain(
        request_data.chain_type,
        request_data.llm_provider,
        use_query_expansion=request_data.use_query_expansion,
        use_reranking=request_data.use_reranking,
    )

    for question in request_data.questions:
        try:
            start_time = time.time()

            if request_data.chain_type == "advanced":
                result = chain.invoke({"input": question})
                answer = result.get("answer", str(result))  # type: ignore
            else:
                answer = chain.invoke(question)

            latency_ms = (time.time() - start_time) * 1000

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "latency_ms": round(latency_ms, 2),
                    "status": "success",
                }
            )
        except Exception as e:
            logger.error(f"❌ Batch query failed for '{question}': {str(e)}")
            results.append(
                {
                    "question": question,
                    "answer": None,
                    "latency_ms": 0,
                    "status": "error",
                    "error": str(e),
                }
            )

    return {
        "results": results,
        "total": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "config": {
            "query_expansion": request_data.use_query_expansion,
            "reranking": request_data.use_reranking,
        },
    }


# ============================================================================
# FEEDBACK ENDPOINT
# ============================================================================
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comments: Optional[str] = None


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for answer quality

    Can be used to:
    1. Track answer quality over time
    2. Fine-tune prompts
    3. Identify problematic queries

    Example:
    ```json
    {
        "question": "What is X?",
        "answer": "X is...",
        "rating": 5,
        "comments": "Great answer!"
    }
    ```
    """
    import uuid

    feedback_id = str(uuid.uuid4())
    feedback_data = {
        "feedback_id": feedback_id,
        "question": feedback.question,
        "answer": feedback.answer,
        "rating": feedback.rating,
        "comments": feedback.comments,
        "timestamp": datetime.now().isoformat(),
    }

    # TODO: Store feedback in database (e.g., PostgreSQL, MongoDB)
    # For now, just log it
    logger.info(f"📊 Feedback received: Rating {feedback.rating}/5 (ID: {feedback_id})")

    return {"message": "Feedback submitted successfully", "feedback_id": feedback_id}


# ============================================================================
# QUERY EXPANSION TEST ENDPOINT
# ============================================================================
class QueryExpansionRequest(BaseModel):
    question: str
    num_variations: Optional[int] = Field(2, ge=1, le=5)
    use_llm: Optional[bool] = True


@app.post("/query/expand")
async def test_query_expansion(request_data: QueryExpansionRequest):
    """
    Test query expansion feature

    Returns the original query plus generated variations

    Example:
    ```json
    {
        "question": "What are the main features?",
        "num_variations": 3,
        "use_llm": true
    }
    ```
    """
    try:
        start_time = time.time()

        variations = expand_query(
            request_data.question,
            num_queries=request_data.num_variations,  # type: ignore
            use_llm=request_data.use_llm,  # type: ignore
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "original_query": request_data.question,
            "variations": variations[1:],  # Exclude original
            "all_queries": variations,
            "method": "llm" if request_data.use_llm else "rule-based",
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Query expansion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query expansion failed: {str(e)}")


# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    """
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP exception handler
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()},
    )


# ============================================================================
# STARTUP EVENT (for additional initialization)
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """
    Additional startup tasks
    """
    logger.info("🎯 RAG API is ready to serve requests!")


# ============================================================================
# MAIN (for development)
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    # Development server with auto-reload
    uvicorn.run(
        "app:app",
        # host="0.0.0.0",
        port=7777,
        reload=True,
        log_level="info",
        access_log=True,
    )

    # Production deployment (use this command):
    # uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
