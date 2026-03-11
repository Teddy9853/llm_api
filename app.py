from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from openai import OpenAI

app = FastAPI()

client = OpenAI()  # reads OPENAI_API_KEY from environment


class PromptRequest(BaseModel):
    # Step 10: strip whitespace automatically
    model_config = ConfigDict(str_strip_whitespace=True)

    # Step 10: enforce non-empty prompt
    prompt: str = Field(min_length=1)


@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}


@app.post("/hello")
def hello(req: PromptRequest):
    try:
        # LLM call
        response = client.responses.create(
            model="gpt-5.2",
            input=req.prompt,
        )

        # Extract model text output
        llm_text = response.output_text

        return {
            "input": req.prompt,
            "llm_output": llm_text,
        }

    except Exception as e:
        # Step 11: log internally, but don't expose details to clients
        print("LLM ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="LLM request failed")