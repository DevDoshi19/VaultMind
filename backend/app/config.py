from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "vaultmind"

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    # chroma_persist_dir: str = "./chroma_db"
    # chroma_collection_name: str = "vaultmind_resume"

    pinecone_api_key: str 
    pinecone_index_name: str ="vaultmind"

    # Resume
    resume_path: str = "./data/resume.pdf"

    # Retrieval
    retrieval_k: int = 3
    # Pinecone uses cosine similarity — higher score = more relevant.
    # This replaces ChromaDB's L2 distance threshold of 1.6.
    # Cosine scores range from 0.0 to 1.0, we keep chunks above 0.7.
    similarity_threshold: float = 0.3

    max_context_tokens: int = 1000
    llm_timeout: float = 30.0
    max_llm_retries: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()