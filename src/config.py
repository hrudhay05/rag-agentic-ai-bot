from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_FILE = ROOT / "data" / "Ebook-Agentic-AI.pdf"
INDEX_DIR = ROOT / "data" / "faiss_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:1.5b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
MIN_SIMILARITY = 0.30

REFUSAL = "I cannot answer based on the provided document."