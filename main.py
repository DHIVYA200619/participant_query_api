from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int

class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]

# ✅ Lightweight model for Render free tier (CPU optimized)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Demo corpus
documents = [
    "Artificial intelligence simulates human intelligence processes using machines.",
    "Machine learning is a subset of AI that learns from experience.",
    "Deep learning employs neural networks for complex data representations.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information from the world.",
    "Reinforcement learning uses rewards and punishments to teach models optimal actions."
]

corpus_embeddings = model.encode(documents, convert_to_tensor=True)

@app.post("/")
async def participant_query(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(request.top_k, len(documents))
    top_indices = np.argsort(-scores)[:top_k]
    contexts = [documents[i] for i in top_indices]
    answer = f"Based on my knowledge, {contexts[0]}"
    return QueryResponse(answer=answer, contexts=contexts)
