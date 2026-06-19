import json
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print

from app.config import settings


def ingest_resume() -> int:
    pdf_path = settings.resume_path

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume not found at: {pdf_path}")

    print(f"📄 Loading: {pdf_path}")

    # Step 1 — Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   {len(documents)} page(s) loaded")

    # Step 2 — Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(documents)
    print(f"   {len(chunks)} chunks created")

    # Step 3 — Embed + Store in ChromaDB
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
    print("   ChromaDB updated")

    # Step 4 — Save raw chunks for BM25 (new)
    bm25_data = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    os.makedirs("./data", exist_ok=True)
    with open("./data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, indent=2)

    print("   BM25 chunks saved to data/chunks.json")  # noqa: F541
    print(f"✅ Done. {len(chunks)} chunks ready")

    return len(chunks)


if __name__ == "__main__":
    ingest_resume()