from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph import graph

app = FastAPI(title="Agentic AI RAG Chatbot")


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = graph.invoke({"question": question})
    return {
        "answer": result["answer"],
        # "retrieved_chunks": result["chunks"],
        "confidence_score": result["confidence"],
    }