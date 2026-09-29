# RAG Chatbot for the Agentic AI eBook

A Retrieval-Augmented Generation chatbot that answers questions strictly from the Agentic AI eBook. It is built with Python, LangGraph, FAISS, Hugging Face embeddings, Ollama and FastAPI.

## Tech stack

| Part | Choice |
|------|--------|
| Orchestration | LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face, local) |
| Vector store | FAISS (local) |
| LLM | `qwen2.5:1.5b` via Ollama (local) |
| API | FastAPI |

## Why Hugging Face, FAISS and Ollama instead of OpenAI and Pinecone

The reference guide suggests OpenAI for embeddings and the LLM, and Pinecone for the vector index. I did not use them for these reasons:

1. **No paid credits.** The OpenAI API needs a funded account, and I have no credits available for this assignment. A free stack lets anyone run the project without paying.
2. **Reproducibility.** Reviewers can clone the repo and run it without creating accounts or handling API keys. There is no secret to leak or expire.
3. **Privacy and offline use.** The document, embeddings and questions stay on the local machine.
4. **Same architecture.** The pipeline is unchanged: load, chunk, embed, store, retrieve, generate with a strict grounding prompt. Only the providers differ, and each sits behind one small function.
5. **Pinecone is not needed at this scale.** The eBook produces a few hundred chunks, which FAISS searches instantly on a laptop. A hosted vector database adds signup, network latency and an API key without any benefit here.

Trade-offs I accept:

1. `qwen2.5:1.5b` is much smaller than `gpt-4o-mini`, so answers are less fluent and it can miss subtle points. The retrieval step and strict prompt reduce hallucination, but the model quality has a ceiling.
2. MiniLM embeddings (384 dimensions) are a bit weaker than `text-embedding-3-small` (1536 dimensions).
3. FAISS is local and file-based, so it does not scale across machines the way a managed service does.

Moving to OpenAI and Pinecone later means changing `get_embedder()` in `src/ingestion.py`, the `_store` setup and `_llm` in `src/graph.py`, and re-creating the index with dimension 1536.

## How it works

1. **Ingestion** (`src/ingestion.py`): loads the PDF with `PyPDFLoader`, splits it with `RecursiveCharacterTextSplitter`, embeds the chunks with normalized MiniLM vectors and saves a FAISS index to `data/faiss_index`.
2. **Graph** (`src/graph.py`): a LangGraph `StateGraph` with these nodes:
   - `retrieve`: finds the top-k chunks and computes cosine similarity for each.
   - a conditional gate: if the best similarity is below `MIN_SIMILARITY`, the question goes to `refuse`, and the LLM is never called.
   - `generate`: asks the LLM to answer using only the retrieved passages.
   - `refuse`: returns a fixed "cannot answer" message.
3. **API** (`app.py`): `POST /chat` runs the graph and returns the answer, retrieved chunks and a confidence score.

The confidence score is the highest cosine similarity between the question and the retrieved chunks. Embeddings are normalized, so the cosine is computed from the FAISS distance as `1 - distance / 2`.

## Project structure

```
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
├── app.py
├── requirements.txt
├── .env.example
├── tests_sample_queries.py
└── README.md
```

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com).

1. Clone and enter the repo:
```bash
git clone https://github.com/hrudhay05/rag-agentic-ai-bot.git
cd rag-agentic-ai-bot
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

3. Pull the LLM:
```bash
ollama pull qwen2.5:1.5b
```

4. Make sure the PDF is at `data/Ebook-Agentic-AI.pdf`, then build the index:
```bash
python -m src.ingestion
```

5. Start the API:
```bash
uvicorn app:app --reload
```

## Usage

Interactive docs are at http://127.0.0.1:8000/docs.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```

Response format:
```json
{
  "answer": "...",
  "confidence_score": 0.61
}
```

## Testing

With the server running:
```bash
python tests_sample_queries.py
```

This runs six queries: five about the eBook and one off-topic (FIFA World Cup). The off-topic query should return `I cannot answer based on the provided document.`

## Configuration

Settings live in `src/config.py`: chunk size, overlap, `TOP_K`, `MIN_SIMILARITY` and the model names. If valid questions get refused, lower `MIN_SIMILARITY`. If off-topic questions get answered, raise it.

## Limitations

1. The small local LLM can produce shorter or less polished answers than a hosted model.
2. The confidence score measures retrieval similarity, not the correctness of the generated answer.
3. The index is rebuilt manually with `python -m src.ingestion` when the PDF changes.


## Tests_Sample_Queries_Output
[1] What is Agentic AI according to the eBook?
Answer: Agentic AI is defined as a shift from reactive to proactive problem-solving, where it promises to enhance businesses' ability to anticipate and address issues before they occur.The eBook emphasizes that while Agentic AI is often misunderstood, it is not a magic solution that can solve all problems. It provides a guide to understanding what Agentic AI is, how it stands apart from other AI, what it can do, and how businesses are using it in the real world.
Confidence: 0.823

[2] How do AI agents differ from traditional automation systems?
Answer: AI agents differ from traditional automation systems in that they are goal-driven systems capable of performing actions autonomously in a dynamic environment, whereas traditional automation systems are rule-based and lack autonomy.
Confidence: 0.6909

[3] What are the core components of an Agentic Architecture?
Answer: The core components of an Agentic Architecture include:

- Perception Layer: Captures data from the environment using sensors and input devices.
- Reasoning Layer: Analyzes the data to make decisions and predictions.
- Planning Layer: Defines the actions and strategies to achieve goals.
- Learning Layer: Trains the system to adapt and improve over time.
- Execution Layer: Carries out the actions based on the decisions and plans.
Confidence: 0.6541

[4] What role does memory play in Agentic AI workflows?
Answer: Memory plays a crucial role in Agentic AI workflows by serving as a repository for past interactions and successful past methods of completing tasks. It helps reduce the amount of computing needed to complete new tasks by referencing relevant past plans and actions. Additionally, memory acts as the place where an agent can store human demonstrations it has seen, which can expedite its progress without as rigorous a planning and reasoning loop.
Confidence: 0.7374

[5] What are the main challenges of deploying agentic systems?
Answer: The main challenges of deploying agentic systems include complex system design, interoperability issues with existing ERP, WMS, and TMS systems, data security concerns, conflictresolution, scalability problems, high initial investment in development and deployment, andthe need for advanced orchestration.
Confidence: 0.6021

[6] Who won the 2022 FIFA World Cup?
Answer: I cannot answer based on the provided document.
Confidence: 0.1078