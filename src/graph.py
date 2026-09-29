from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src import config
from src.ingestion import get_embedder


class RagState(TypedDict):
    question: str
    chunks: list[dict]
    confidence: float
    answer: str


_store = FAISS.load_local(
    str(config.INDEX_DIR),
    get_embedder(),
    allow_dangerous_deserialization=True,
)
_llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0)


def retrieve(state: RagState):
    hits = _store.similarity_search_with_score(state["question"], k=config.TOP_K)

    chunks = []
    for doc, dist in hits:
        sim = 1 - float(dist) / 2
        chunks.append({
            "text": doc.page_content,
            "page": doc.metadata.get("page"),
            "similarity": round(sim, 4),
        })

    best = max((c["similarity"] for c in chunks), default=0.0)
    return {"chunks": chunks, "confidence": round(max(best, 0.0), 4)}


def gate(state: RagState):
    return "generate" if state["confidence"] >= config.MIN_SIMILARITY else "refuse"


def generate(state: RagState):
    passages = "\n\n".join(
        f"[page {c['page']}] {c['text']}" for c in state["chunks"]
    )
    prompt = (
        "Answer the question using only the passages below, which come from "
        "an eBook on Agentic AI. Do not add outside knowledge. If the passages "
        f"do not contain the answer, reply exactly: {config.REFUSAL}\n\n"
        f"Passages:\n{passages}\n\n"
        f"Question: {state['question']}\n"
        "Answer in a few clear sentences:"
    )
    reply = _llm.invoke(prompt)
    return {"answer": reply.content.strip()}


def refuse(state: RagState):
    return {"answer": config.REFUSAL}


builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("refuse", refuse)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", gate, ["generate", "refuse"])
builder.add_edge("generate", END)
builder.add_edge("refuse", END)

graph = builder.compile()