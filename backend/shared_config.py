# shared_config.py
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = "pdf_ingestion_queue"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:secretpassword@localhost:27017/?authSource=admin")
MONGO_DB = "rag_database"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8080")
UPLOAD_DIR = "/tmp/rag_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)