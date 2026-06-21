from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "vaultmind_resume"

    # Resume
    resume_path: str = "./data/resume.pdf"

    # Retrieval
    retrieval_k: int = 3
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()