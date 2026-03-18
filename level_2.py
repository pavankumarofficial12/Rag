from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load model
model = SentenceTransformer("intfloat/e5-small")

# Load PDF
loader = PyPDFLoader("/Users/sai/Downloads/Pavan_Gelli_AIEngineer.pdf")  # put a real PDF here
docs = loader.load()

# Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

texts = [chunk.page_content for chunk in chunks]

# Create embeddings
embeddings = model.encode(texts)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Query
query = "tell me the name of the?"
query_embedding = model.encode([query])

# Search
D, I = index.search(np.array(query_embedding), k=2)

print("Retrieved context:")
for i in I[0]:
    print("-", texts[i])
