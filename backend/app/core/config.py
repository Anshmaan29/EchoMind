from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application configuration managed via Environment variables and .env file.
    Follows 12-factor app principles.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "EchoMind"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True

    # PostgreSQL Database
    POSTGRES_USER: str = "echomind_user"
    POSTGRES_PASSWORD: str = "echomind_password"
    POSTGRES_DB: str = "echomind_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://echomind_user:echomind_password@localhost:5432/echomind_db"
    )

    # Qdrant Vector Store
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "echomind_memories"

    # Neo4j Knowledge Graph Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "echomind_neo4j_password"
    NEO4J_DATABASE: str = "neo4j"
    GRAPH_STORE_TYPE: Literal["neo4j", "mock"] = "neo4j"

    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: Literal["mock", "openai", "sentence_transformers", "bge-m3", "qwen", "qwen3", "nomic"] = "mock"
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-8B"
    EMBEDDING_DIMENSION: int = 384
    OPENAI_API_KEY: str | None = None

    # LLM Provider Configuration (RAG Pipeline)
    LLM_PROVIDER: Literal["mock", "openai", "anthropic", "openrouter"] = "mock"
    LLM_MODEL_NAME: str = "mock-gpt-4o"
    LLM_API_KEY: str | None = None

    @property
    def async_database_url(self) -> str:
        """Constructs an asyncpg dialect URL for SQLAlchemy."""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL

settings = Settings()
