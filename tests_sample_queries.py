import requests

URL = "http://127.0.0.1:8000/chat"

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What are the main challenges of deploying agentic systems?",
    "Who won the 2022 FIFA World Cup?",
]


def run():
    for i, q in enumerate(QUERIES, 1):
        resp = requests.post(URL, json={"query": q}, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        print(f"\n[{i}] {q}")
        print(f"Answer: {data['answer']}")
        print(f"Confidence: {data['confidence_score']}")
        # pages = [c["page"] for c in data["retrieved_chunks"]]
        # print(f"Chunk pages: {pages}")


if __name__ == "__main__":
    run()