"""
Level 4 RAG System using:
- MySQL (via MySQL Workbench)
- Direct DB connection (mysql-connector)
- E5 embeddings
- Chunking
- Semantic similarity search

Simple, stable, industry-usable
"""

import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# DATABASE CONFIGURATION (YOUR WAY - CORRECT)
# -------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root12345",   # change if needed
    "database": "rag_db"
}

# -------------------------------------------------
# Load Embedding Model
# -------------------------------------------------
embedding_model = SentenceTransformer("intfloat/e5-small")

# -------------------------------------------------
# Fetch Documents from MySQL
# -------------------------------------------------
def fetch_documents():
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()

    cursor.execute("SELECT title, content FROM documents")
    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    documents = []
    for title, content in rows:
        documents.append({
            "title": title,
            "content": content
        })

    return documents

# -------------------------------------------------
# Chunking Logic
# -------------------------------------------------
def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks

# -------------------------------------------------
# Generate Embeddings
# -------------------------------------------------
def generate_embeddings(texts):
    return embedding_model.encode(texts)

# -------------------------------------------------
# Semantic Search
# -------------------------------------------------
def semantic_search(query, chunks, embeddings):
    query_embedding = generate_embeddings([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    best_index = np.argmax(scores)
    return chunks[best_index], scores[best_index]

# -------------------------------------------------
# MAIN RAG PIPELINE
# -------------------------------------------------
def main():
    print("🔹 Fetching data from MySQL...")
    documents = fetch_documents()

    all_chunks = []

    print("🔹 Chunking documents...")
    for doc in documents:
        chunks = chunk_text(doc["content"])
        all_chunks.extend(chunks)

    print("🔹 Generating embeddings...")
    chunk_embeddings = generate_embeddings(all_chunks)

    # User Query
    user_query = "What skills are required for AI Engineer?"

    print("🔹 Performing semantic search...")
    best_chunk, score = semantic_search(
        user_query,
        all_chunks,
        chunk_embeddings
    )

    print("\n================ RAG RESULT ================")
    print("User Question:", user_query)
    print("Best Match:", best_chunk)
    print("Similarity Score:", round(float(score), 3))
    print("===========================================")

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
