# verify_mongo.py
from pymongo import MongoClient

mongo_client = MongoClient("mongodb://admin:secretpassword@localhost:27017/")
db = mongo_client["rag_db"]
collection = db["documents"]

# Fetch all processed documents
documents = list(collection.find({}, {"_id": 0}))

print(f"[*] Total Document Records in MongoDB: {len(documents)}\n")
for doc in documents:
    print(f"Job ID: {doc.get('job_id')}")
    print(f"Filename: {doc.get('filename')}")
    print(f"Total Chunks: {doc.get('total_chunks')}")
    print(f"Status: {doc.get('status')}")
    print("-" * 40)