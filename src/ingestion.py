from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def get_embedder():
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index():
    if not config.PDF_FILE.exists():
        raise FileNotFoundError(f"Missing PDF: {config.PDF_FILE}")

    pages = PyPDFLoader(str(config.PDF_FILE)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        c for c in splitter.split_documents(pages)
        if len(c.page_content.strip()) > 50
    ]

    index = FAISS.from_documents(chunks, get_embedder())
    index.save_local(str(config.INDEX_DIR))
    print(f"{len(pages)} pages -> {len(chunks)} chunks saved to {config.INDEX_DIR}")


if __name__ == "__main__":
    build_index()