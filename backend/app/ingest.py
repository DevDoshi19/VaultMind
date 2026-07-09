import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
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

    # Connect to Pinecone and clear existing vectors before re-ingesting.
    # This prevents duplicate chunks building up across multiple ingest runs.

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        index.delete(delete_all=True)
        print("   Pinecone namespace cleared")
    else:
        print("   Pinecone index is empty — skipping clear")
        
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
        pinecone_api_key=settings.pinecone_api_key,
    )
    print("   Pinecone index updated")

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