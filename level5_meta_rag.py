"""
Level 5 RAG System with Metadata Filtering

Features:
- MySQL (MySQL Workbench)
- Metadata filtering (category, location)
- Chunking
- E5 embeddings
- Semantic similarity search

Industry-used pattern
"""

import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# DATABASE CONFIGURATION
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
# Fetch Documents WITH METADATA FILTERING
# -------------------------------------------------
def fetch_documents(category=None, location=None):
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()

    query = "SELECT title, content FROM documents WHERE 1=1"
    params = []

    if category:
        query += " AND category = %s"
        params.append(category)

    if location:
        query += " AND location = %s"
        params.append(location)

    cursor.execute(query, params)
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
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

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
    # Simulating extracted metadata from user query
    category = "job"
    location = "Bangalore"

    print("🔹 Applying metadata filtering...")
    documents = fetch_documents(category=category, location=location)

    if not documents:
        print("❌ No documents found after filtering.")
        return

    all_chunks = []

    print("🔹 Chunking documents...")
    for doc in documents:
        all_chunks.extend(chunk_text(doc["content"]))

    print("🔹 Generating embeddings...")
    chunk_embeddings = generate_embeddings(all_chunks)

    user_query = "What skills are required for AI Engineer?"

    print("🔹 Performing semantic search...")
    best_chunk, score = semantic_search(
        user_query,
        all_chunks,
        chunk_embeddings
    )

    print("\n================ LEVEL 5 RESULT ================")
    print("Category:", category)
    print("Location:", location)
    print("User Query:", user_query)
    print("Best Match:", best_chunk)
    print("Similarity Score:", round(float(score), 3))
    print("===============================================")

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
