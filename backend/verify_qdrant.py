# verify_qdrant.py
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "pdf_documents"

# Get collection info
info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"[*] Total Vectors Stored in Qdrant: {info.points_count}")

# Fetch 1 sample record to inspect the payload and vector
records, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=1,
    with_payload=True,
    with_vectors=True
)

if records:
    sample = records[0]
    print("\n--- SAMPLE STORED POINT ---")
    print(f"Point ID: {sample.id}")
    print(f"Filename: {sample.payload.get('filename')}")
    print(f"Chunk Index: {sample.payload.get('chunk_index')}")
    print(f"Text Snippet: {sample.payload.get('text')[:100]}...")
    print(f"Vector Dimensions: {len(sample.vector)} (Should be 768)")