import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import settings 

def ingest_resume() -> int :
    pdf_path = settings.resume_path

    if not os.path.exists(pdf_path) :
        raise FileNotFoundError(f"Resume not found at :{pdf_path}")
    
    # LOAD THE PDF 
    print(f"📄 Loading : {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   {len(documents)} page(s) loaded")

    # chunk 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f" {len(chunks)} chunks created")

    # Embed + store 
    embeddings = OpenAIEmbeddings(
        model = settings.embedding_model,
        openai_api_key = settings.OPENAI_API_KEY ,
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name= settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir
    )

    print(f"✅ Done. {len(chunks)} chunks stored in ChromaDB")
    return len(chunks)


if __name__ == "__main__":
    ingest_resume()