# Step 1 & Pipeline B Complete: API Engine & Conversational Retrieval Streaming Loop
import os
import uuid
import time
import json
import shutil
import requests
from pathlib import Path
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse 
import redis
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
EMBEDDING_SERVER_URL = os.getenv("Embedding_Server_URL", "http://localhost:8080/embed")

app = FastAPI(title="RAG Ingestion & Conversational Retrieval API")

# --- CONFIGURATION & CONNECTIONS ---
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

EMBEDDING_SERVER_URL = os.getenv("Embedding_Server_URL", "http://localhost:8080/embed")
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "pdf_documents"

UPLOAD_DIR = Path("./temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
QUEUE_NAME = "pdf_ingestion_queue"

# LLM Configuration (Hugging Face Inference)
hf_client = InferenceClient(model=MODEL_ID, token=HF_TOKEN if HF_TOKEN else None)

SIMILARITY_THRESHOLD = 0
GROUNDING_REFUSAL_MESSAGE = (
    "I'm sorry, but I couldn't find sufficient relevant information in your uploaded document "
    "to confidently answer this question."
)

SYSTEM_INSTRUCTIONS = """You are a helpful, precise AI document assistant.
Answer the user's question ONLY using the provided Context Chunks below.

CRITICAL RULES:
1. Do NOT use outside knowledge or make assumptions not directly supported by the context.
2. If the context does not contain enough information to answer the question completely, clearly state: "I cannot answer this question based on the provided document context."
3. Cite relevant details directly from the text where applicable."""


# --- PYDANTIC SCHEMAS ---
class JobStatusResponse(BaseModel):
    job_id: str
    filename: Optional[str] = None
    status: str
    error: Optional[str] = None
    created_at: str
    file_path: Optional[str] = None

class QueryRequest(BaseModel):
    job_id: str
    query: str
    top_k: int = 5


# --- PIPELINE A: INGESTION ENDPOINTS ---

@app.post("/v1/document/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...)):
    # 1. Validation
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file type. Only PDF documents are allowed."
        )
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
            detail="File is too large. Maximum allowed size is 20MB."
        )
        
    # 2. Generate ID & save file
    job_id = str(uuid.uuid4())
    temp_file_path = UPLOAD_DIR / f"{job_id}.pdf"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Write to Redis
        job_key = f"job:{job_id}"
        redis_client.hset(job_key, mapping={
            "job_id": job_id,
            "filename": file.filename,
            "status": "PENDING",
            "created_at": str(time.time()),
            "file_path": str(temp_file_path)
        })
        
        # 4. Queue the job
        redis_client.lpush(QUEUE_NAME, job_id)
        
        # 5. Handshake response
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "Accepted",
                "job_id": job_id,
                "message": "Your file is queued. Use the job_id to track progress."
            }
        )
        
    except Exception as e:
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to initialize upload: {str(e)}"
        )


@app.get("/v1/document/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job_key = f"job:{job_id}"
    job_data = redis_client.hgetall(job_key)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    return job_data


# --- PIPELINE B: RETRIEVAL & VECTOR SEARCH ---

def vectorize_query(query_text: str) -> list[float]:
    """Transforms plain text into a vector using the local Hugging Face embedding server."""
    payload = {"inputs": [query_text]}
    try:
        response = requests.post(EMBEDDING_SERVER_URL, json=payload, timeout=10)
        if response.status_code == 200:
            embeddings = response.json()
            return embeddings[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Embedding server error: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Embedding server unreachable: {str(e)}"
        )


def retrieve_context_chunks(job_id: str, query_vector: list[float], top_k: int = 5) -> tuple[list[dict], bool]:
    """Executes ANN search in Qdrant and applies similarity cutoff guardrail."""
    job_filter = Filter(
        must=[
            FieldCondition(
                key="job_id",
                match=MatchValue(value=job_id)
            )
        ]
    )

    try:
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=job_filter,
            limit=top_k,
            with_payload=True
        )

        context_chunks = []
        highest_score = 0.0

        for point in search_results.points:
            score = round(point.score, 4)
            if score > highest_score:
                highest_score = score

            context_chunks.append({
                "score": score,
                "chunk_id": point.id,
                "text": point.payload.get("text", ""),
                "page_number": point.payload.get("page_number", None),
                "filename": point.payload.get("filename", "")
            })

        # Guardrail evaluation
        if not context_chunks or highest_score in (0.45, 0.55):
            return [], False

        return context_chunks, True

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qdrant retrieval error: {str(e)}"
        )


# --- STAGE 3: PROMPT ASSEMBLY ---

def assemble_rag_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """Formats retrieved vector matches and system instructions into standard message structures."""
    formatted_blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        block = f"--- Chunk {i} [Score: {chunk['score']}] ---\nContent:\n{chunk['text']}\n"
        formatted_blocks.append(block)

    full_context = "\n".join(formatted_blocks)

    return [
        {
            "role": "system",
            "content": f"{SYSTEM_INSTRUCTIONS}\n\n=== RETRIEVED CONTEXT CHUNKS ===\n{full_context}"
        },
        {"role": "user", "content": query}
    ]


# --- STAGE 4: STREAMING GENERATOR & MAIN COMPLETION ENDPOINT ---

async def stream_llm_response(prompt_messages: list[dict]) -> AsyncGenerator[str, None]:
    """Yields tokens from Hugging Face Inference API as Server-Sent Events (SSE)."""
    try:
        stream = hf_client.chat_completion(
            messages=prompt_messages,
            max_tokens=512,
            temperature=0.2,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token_text = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'token': token_text})}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completion_stream(payload: QueryRequest):
    """
    Complete Pipeline B Execution:
    1. Embeds user query (Stage 1)
    2. Searches Qdrant & applies cutoff score guardrail (Stage 2)
    3. Formats prompt with grounded instructions (Stage 3)
    4. Streams response tokens over SSE (Stage 4)
    """
    query_vector = vectorize_query(payload.query)

    chunks, pass_guardrail = retrieve_context_chunks(
        job_id=payload.job_id,
        query_vector=query_vector,
        top_k=payload.top_k
    )

    if not pass_guardrail:
        async def refusal_stream():
            yield f"data: {json.dumps({'token': GROUNDING_REFUSAL_MESSAGE})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(refusal_stream(), media_type="text/event-stream")

    prompt_messages = assemble_rag_prompt(payload.query, chunks)

    return StreamingResponse(
        stream_llm_response(prompt_messages),
        media_type="text/event-stream"
    )


# --- DIAGNOSTIC & TEST ENDPOINTS ---

@app.post("/v1/chat/vectorize-test")
async def test_query_vectorization(payload: QueryRequest):
    vector = vectorize_query(payload.query)
    return {
        "job_id": payload.job_id,
        "query": payload.query,
        "vector_dimensions": len(vector),
        "sample_vector_values": vector[:5]
    }


@app.post("/v1/chat/search-test")
async def test_vector_search(payload: QueryRequest):
    query_vector = vectorize_query(payload.query)

    matches, pass_guardrail = retrieve_context_chunks(
        job_id=payload.job_id,
        query_vector=query_vector,
        top_k=payload.top_k
    )

    if not pass_guardrail:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "REFUSED",
                "reason": "Highest match score fell below threshold.",
                "message": GROUNDING_REFUSAL_MESSAGE
            }
        )
        
    return {
        "status": "PASSED",
        "job_id": payload.job_id,
        "query": payload.query,
        "top_match_score": matches[0]["score"] if matches else 0,
        "total_matches_found": len(matches),
        "top_matches": matches
    }