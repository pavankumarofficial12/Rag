from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
model = SentenceTransformer("intfloat/e5-small")

documents = [
    "AI course duration is 6 months",
    "Eligibility is graduation with basic math",
    "Fees is 50,000 INR"
]

doc_embeddings = model.encode(documents)

@app.post("/ask")
def ask(question: str):
    query_embedding = model.encode([question])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    best_index = scores.argmax()
    return {"answer": documents[best_index]}
