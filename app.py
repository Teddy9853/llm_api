from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from openai import OpenAI
import math
import uuid

app = FastAPI()

client = OpenAI()

# =========================
# In-memory stores
# =========================

sessions: Dict[str, List[dict]] = {}
chunks_store: List[dict] = []
doc_index: Dict[str, List[str]] = {}

# =========================
# Models
# =========================

class SessionResponse(BaseModel):
    session_id: str


class IngestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class Citation(BaseModel):
    chunk_id: str
    score: float


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str
    message: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    turn_count: int


# =========================
# Config
# =========================

CHAT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# =========================
# Helpers
# =========================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step

    return chunks


def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must have the same length")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def remove_existing_doc_chunks(doc_id: str) -> None:
    global chunks_store

    existing_chunk_ids = set(doc_index.get(doc_id, []))
    if not existing_chunk_ids:
        return

    chunks_store = [c for c in chunks_store if c["chunk_id"] not in existing_chunk_ids]
    doc_index[doc_id] = []


def retrieve_top_k(query: str, k: int) -> List[dict]:
    if not chunks_store:
        return []

    query_embedding = get_embedding(query)
    scored_results = []

    for chunk in chunks_store:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "score": score
        })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:k]


def build_grounded_messages(history: List[dict], retrieved_chunks: List[dict], user_message: str) -> List[dict]:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[{chunk['chunk_id']} | score={chunk['score']:.4f}]\n{chunk['text']}"
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

    grounded_system = {
        "role": "system",
        "content": (
            "You are a helpful CS teaching assistant. "
            "Answer using the retrieved context. "
            "If the answer is not supported by the retrieved context, say that it is not available in the ingested documents. "
            "Do not make up facts. Give concise explanations."
        )
    }

    context_message = {
        "role": "system",
        "content": (
            "Retrieved context:\n\n"
            f"{context_text}\n\n"
            "Use this context together with the conversation history to answer the latest user message."
        )
    }

    prior_turns = [msg for msg in history if msg["role"] != "system"]

    return [grounded_system, context_message] + prior_turns + [
        {"role": "user", "content": user_message}
    ]


# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {"message": "RAG Q&A Service Running"}


@app.post("/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())

    sessions[session_id] = [
        {
            "role": "system",
            "content": "You are a helpful CS teaching assistant. Give concise explanations."
        }
    ]

    return {"session_id": session_id}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        remove_existing_doc_chunks(req.doc_id)

        text_chunks = chunk_text(req.text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Document text produced no chunks")

        added_chunk_ids = []

        for i, chunk_text_value in enumerate(text_chunks):
            chunk_id = f"{req.doc_id}#{i}"
            embedding = get_embedding(chunk_text_value)

            chunk_record = {
                "chunk_id": chunk_id,
                "doc_id": req.doc_id,
                "text": chunk_text_value,
                "embedding": embedding
            }

            chunks_store.append(chunk_record)
            added_chunk_ids.append(chunk_id)

        doc_index[req.doc_id] = added_chunk_ids

        return {
            "doc_id": req.doc_id,
            "chunks_added": len(added_chunk_ids)
        }

    except HTTPException:
        raise
    except Exception as e:
        print("INGEST ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Document ingestion failed")


@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., min_length=1), k: int = Query(3, ge=1, le=10)):
    try:
        if not chunks_store:
            return {"query": query, "results": []}

        top_chunks = retrieve_top_k(query, k)

        return {
            "query": query,
            "results": [
                {
                    "chunk_id": item["chunk_id"],
                    "score": round(item["score"], 4),
                    "text": item["text"]
                }
                for item in top_chunks
            ]
        }

    except Exception as e:
        print("SEARCH ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if not chunks_store:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet")

    try:
        history = sessions[req.session_id]

        # Step 1: retrieve top-k chunks using the current message
        retrieved_chunks = retrieve_top_k(req.message, req.k)

        # Step 2: build grounded prompt with context + conversation history
        grounded_messages = build_grounded_messages(
            history=history,
            retrieved_chunks=retrieved_chunks,
            user_message=req.message
        )

        # Step 3: save user message in session history
        history.append({
            "role": "user",
            "content": req.message
        })

        # Step 4: call LLM
        response = client.responses.create(
            model=CHAT_MODEL,
            input=grounded_messages
        )

        assistant_text = response.output_text

        # Step 5: save assistant reply
        history.append({
            "role": "assistant",
            "content": assistant_text
        })

        # Count turns excluding the original system prompt
        turn_count = len(history) - 1

        return {
            "answer": assistant_text,
            "citations": [
                {
                    "chunk_id": item["chunk_id"],
                    "score": round(item["score"], 4)
                }
                for item in retrieved_chunks
            ],
            "turn_count": turn_count
        }

    except HTTPException:
        raise
    except Exception as e:
        print("CHAT ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Chat request failed")