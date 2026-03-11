from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from openai import OpenAI
import uuid

app = FastAPI()

client = OpenAI()  # reads OPENAI_API_KEY from environment

# In-memory session storage
sessions = {}

# =========================
# Models
# =========================

class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str
    message: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    response: str
    turn_count: int


# =========================
# Basic Root (keep from your version)
# =========================

@app.get("/")
def root():
    return {"message": "Conversational LLM API Running"}


# =========================
# Create Session Endpoint
# =========================

@app.post("/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())

    # Initialize conversation with system prompt
    sessions[session_id] = [
        {
            "role": "system",
            "content": "You are a helpful CS teaching assistant. Give concise explanations."
        }
    ]

    return {"session_id": session_id}


# =========================
# Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    # Validate session exists
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    try:
        history = sessions[req.session_id]

        # Append user message
        history.append({
            "role": "user",
            "content": req.message
        })

        # Send full conversation to LLM
        response = client.responses.create(
            model="gpt-5.2",
            input=history
        )

        # Extract text
        assistant_text = response.output_text

        # Append assistant reply
        history.append({
            "role": "assistant",
            "content": assistant_text
        })

        # Count turns (excluding system)
        turn_count = len(history) - 1

        return {
            "response": assistant_text,
            "turn_count": turn_count
        }

    except Exception as e:
        print("LLM ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="LLM request failed")