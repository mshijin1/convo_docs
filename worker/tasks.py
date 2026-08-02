# worker/tasks.py
import time
import os
import redis
import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from pymongo import MongoClient

# Database Connections
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
QUEUE_NAME = "pdf_ingestion_queue"
EMBEDDING_SERVER_URL = "http://localhost:8080/embed"

qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "pdf_documents"

mongo_client = MongoClient("mongodb://admin:secretpassword@localhost:27017/")
db = mongo_client["rag_db"]
documents_collection = db["documents"]

def ensure_qdrant_collection():
    """Creates the Qdrant collection configured for 768-dim vectors if missing."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

# --- FAILURE INTERCEPTOR: PURGE SEQUENCE ---
def purge_partial_data(job_id: str):
    """
    Cleans up partial document fragments from MongoDB and Qdrant
    if processing fails mid-execution. Prevents orphan records.
    """
    print(f"[🧹] [Purge Sequence] Triggered cleanup for failed job: {job_id}")
    
    # 1. Purge partial document entries from MongoDB
    try:
        mongo_result = documents_collection.delete_many({"job_id": job_id})
        print(f"    └─ Removed {mongo_result.deleted_count} record(s) from MongoDB.")
    except Exception as e:
        print(f"    └─ MongoDB purge warning: {str(e)}")

    # 2. Purge partial vector points from Qdrant using a job_id payload filter
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="job_id",
                        match=MatchValue(value=job_id)
                    )
                ]
            )
        )
        print(f"    └─ Cleared associated vector points from Qdrant.")
    except Exception as e:
        print(f"    └─ Qdrant purge warning: {str(e)}")

def recursive_token_splitter(text: str, max_tokens: int = 150, overlap_tokens: int = 15) -> list[str]:
    words = text.strip().split()
    chunks = []
    stride = max_tokens - overlap_tokens
    if stride <= 0: stride = max_tokens
    for i in range(0, len(words), stride):
        chunk_words = words[i:i + max_tokens]
        chunk_text = " ".join(chunk_words).strip()
        if len(chunk_text) > 20:
            chunks.append(chunk_text)
    return chunks

def generate_embeddings(text_chunks: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        payload = {"inputs": batch}
        try: 
            response = requests.post(EMBEDDING_SERVER_URL, json=payload, timeout=30)
            if response.status_code == 200:
                all_embeddings.extend(response.json())
            else:
                raise RuntimeError(f"Embedding failed at batch {i}: {response.text}")
        except requests.exceptions.RequestException as e:
            # Re-raise as RuntimeError so Stage 6 catches it cleanly
            raise RuntimeError(f"Embedding server connection lost: {str(e)}") from e
    return all_embeddings

def start_worker():
    ensure_qdrant_collection()
    print(f"[*] Background worker container initialized with Failure Interceptor.")
    print(f"[*] Awaiting payload drop tokens on Redis queue...\n")
    
    while True:
        try:
            result = redis_client.brpop(QUEUE_NAME, timeout=0)
            if not result:
                continue

            _, job_id = result
            print(f"\n[+] [Queue Pickup] Snatched active Job ID: {job_id}")
            
            job_key = f"job:{job_id}"
            job_data = redis_client.hgetall(job_key)
            if not job_data:
                continue
            
            # --- STAGE 2: PROCESSING TRANSITION ---
            redis_client.hset(job_key, "status", "PROCESSING")
            file_path = job_data.get("file_path")
            filename = job_data.get("filename", "unknown.pdf")
            
            # --- MAIN PROCESSING BLOCK WRAPPED IN FAILURE INTERCEPTOR ---
            try:
                if not file_path or not os.path.exists(file_path):
                    raise FileNotFoundError(f"Target PDF missing at path: {file_path}")

                # --- STAGE 3: PARSING & CHUNKING ---
                print(f"[➔] [Parsing] Extracting text layer...")
                reader = PdfReader(file_path)
                raw_text = ""
                for page in reader.pages:
                    text_layer = page.extract_text()
                    if text_layer: 
                        raw_text += text_layer + "\n"
                
                if not raw_text.strip():
                    raise ValueError("Uploaded PDF contains no readable text layers.")

                text_chunks = recursive_token_splitter(raw_text, max_tokens=150, overlap_tokens=15)
                
                # --- STAGE 4: VECTOR GENERATION ---
                print(f"[➔] [Embeddings] Generating 768-dim vectors for {len(text_chunks)} chunks...")
                vectors = generate_embeddings(text_chunks)
                
                # --- STAGE 5: INDEXATION & STORAGE ---
                print(f"[➔] [Indexation] Writing vectors and payload strings to Qdrant...")
                points = []
                for idx, (chunk_text, vector) in enumerate(zip(text_chunks, vectors)):
                    points.append(
                        PointStruct(
                            id=hash(f"{job_id}_{idx}") & 0x7FFFFFFFFFFFFFFF,
                            vector=vector,
                            payload={
                                "job_id": job_id,
                                "filename": filename,
                                "chunk_index": idx,
                                "text": chunk_text
                            }
                        )
                    )
                
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                
                print(f"[➔] [MongoDB] Saving document record to database...")
                documents_collection.insert_one({
                    "job_id": job_id,
                    "filename": filename,
                    "total_chunks": len(text_chunks),
                    "status": "COMPLETED",
                    "processed_at": time.time()
                })
                
                redis_client.hset(name=job_key, mapping={
                    "status": "COMPLETED",
                    "total_chunks": str(len(text_chunks))
                })
                print(f"[✓] Document Processing Completed! Job {job_id} ready.")

            except Exception as pipeline_error:
                # --- STAGE 6: FAILURE INTERCEPTOR CATCH ---
                error_msg = str(pipeline_error)
                print(f"[!] [Failure Interceptor] Crash caught: {error_msg}")
                
            # Update fields individually or using standard mapping syntax
                redis_client.hset(job_key, "status", "FAILED")
                redis_client.hset(job_key, "error", error_msg)
                
                # 2. Run automated purge sequence
                purge_partial_data(job_id)
                
                        # --- ADD THIS CLEANUP STEP HERE ---
            finally:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"[🧹] [File Cleanup] Removed temporary file: {file_path}")
                    except Exception as clean_err:
                        print(f"[!] [File Cleanup Warning] Could not remove {file_path}: {clean_err}")

        except redis.ConnectionError:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n[-] Shutting down worker gracefully.")
            break


if __name__ == "__main__":
    start_worker()