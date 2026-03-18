from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("intfloat/e5-small")

# Sample real-life data (FAQ style)
documents = [
    "AI course duration is 6 months",
    "Eligibility is graduation with basic math",
    "Fees is 50,000 INR"
]

# Create embeddings
doc_embeddings = model.encode(documents)

# User query
query = "What is the total duration?"
query_embedding = model.encode([query])

# Similarity search
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Best match
best_index = scores.argmax()
print("Answer:", documents[best_index])
