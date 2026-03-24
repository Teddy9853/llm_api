from typing import Dict, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from openai import OpenAI
import math
import uuid
import ast
import json

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


class AgentStep(BaseModel):
    action: str
    input: str
    output: str


class AgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str
    query: str = Field(min_length=1, max_length=1000)
    max_steps: int = Field(default=5, ge=1, le=10)
    k: int = Field(default=4, ge=1, le=10)


class AgentResponse(BaseModel):
    answer: str
    steps: List[AgentStep]


# =========================
# Config
# =========================

CHAT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

AGENT_SYSTEM_PROMPT = """
You are a helpful CS teaching assistant and tool-using agent.

You may answer directly when the question is simple and does not require tools.

You must use:
- calculator for arithmetic or numeric computation
- kb_search for questions that depend on ingested documents
- both tools for mixed questions that need both math and document knowledge

Rules:
- Do not make up facts.
- If knowledge is not in the KB, say so.
- Prefer concise answers.
- Maintain multi-step reasoning by using tools when needed.
- When enough information is available, produce a final answer for the user.
""".strip()

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Constant,
)

AGENT_TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression. Use this for math questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression using +, -, *, /, and parentheses"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "kb_search",
        "description": "Search the ingested knowledge base for relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the knowledge base"
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of chunks to retrieve"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
]

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


def safe_eval_arithmetic(expression: str) -> float:
    if not expression or len(expression) > 100:
        raise ValueError("Invalid expression length")

    allowed_chars = set("0123456789+-*/(). ")
    if any(ch not in allowed_chars for ch in expression):
        raise ValueError("Expression contains invalid characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid arithmetic syntax")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError("Unsupported operation in expression")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")

    result = eval(compile(tree, filename="<calc>", mode="eval"), {"__builtins__": {}}, {})

    if not isinstance(result, (int, float)):
        raise ValueError("Expression did not produce a numeric result")

    return result


def calculator_tool(expression: str) -> str:
    value = safe_eval_arithmetic(expression)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def kb_search_tool(query: str, k: int = 4) -> List[dict]:
    return retrieve_top_k(query, k)


def format_kb_results(results: List[dict]) -> str:
    if not results:
        return "No relevant KB results found."

    lines = []
    for item in results:
        lines.append(
            f"[{item['chunk_id']} | doc={item['doc_id']} | score={item['score']:.4f}]\n{item['text']}"
        )
    return "\n\n".join(lines)


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

        retrieved_chunks = retrieve_top_k(req.message, req.k)

        grounded_messages = build_grounded_messages(
            history=history,
            retrieved_chunks=retrieved_chunks,
            user_message=req.message
        )

        history.append({
            "role": "user",
            "content": req.message
        })

        response = client.responses.create(
            model=CHAT_MODEL,
            input=grounded_messages
        )

        assistant_text = response.output_text

        history.append({
            "role": "assistant",
            "content": assistant_text
        })

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


@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if not chunks_store:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet")

    history = sessions[req.session_id]
    steps: List[dict] = []

    try:
        messages: List[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]

        for msg in history:
            if msg["role"] != "system":
                messages.append(msg)

        messages.append({"role": "user", "content": req.query})

        for _ in range(req.max_steps):
            response = client.responses.create(
                model=CHAT_MODEL,
                input=messages,
                tools=AGENT_TOOLS
            )

            response_items = getattr(response, "output", [])
            function_calls = [item for item in response_items if item.type == "function_call"]

            if not function_calls:
                final_answer = response.output_text.strip()

                history.append({"role": "user", "content": req.query})
                history.append({"role": "assistant", "content": final_answer})

                return {
                    "answer": final_answer,
                    "steps": steps
                }

            messages.extend(response.output)

            for call in function_calls:
                tool_name = call.name

                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "calculator":
                    expression = str(args.get("expression", "")).strip()
                    tool_output = calculator_tool(expression)

                    steps.append({
                        "action": "calculator",
                        "input": expression,
                        "output": tool_output
                    })

                elif tool_name == "kb_search":
                    search_query = str(args.get("query", "")).strip()
                    search_k = int(args.get("k", req.k))
                    search_k = max(1, min(search_k, 10))

                    results = kb_search_tool(search_query, search_k)
                    tool_output = format_kb_results(results)

                    steps.append({
                        "action": "kb_search",
                        "input": search_query,
                        "output": tool_output
                    })

                else:
                    tool_output = "Unsupported tool"

                    steps.append({
                        "action": tool_name,
                        "input": json.dumps(args),
                        "output": tool_output
                    })

                messages.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": tool_output
                })

        fallback_answer = "I could not complete the agent workflow within the step limit."

        history.append({"role": "user", "content": req.query})
        history.append({"role": "assistant", "content": fallback_answer})

        return {
            "answer": fallback_answer,
            "steps": steps
        }

    except HTTPException:
        raise
    except Exception as e:
        print("AGENT ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Agent request failed")