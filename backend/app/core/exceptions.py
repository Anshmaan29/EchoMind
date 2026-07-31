from typing import Any

class EchoMindException(Exception):
    """Base exception for all domain and application level errors in EchoMind."""
    def __init__(self, message: str, status_code: int = 500, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class DocumentNotFoundError(EchoMindException):
    """Raised when a requested document entity is not found in storage."""
    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' was not found.",
            status_code=404,
            details={"document_id": document_id}
        )

class IngestionException(EchoMindException):
    """Raised when an error occurs during parsing, chunking, or embedding ingestion."""
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"Ingestion Pipeline Error: {message}",
            status_code=422,
            details=details or {}
        )

class VectorStoreException(EchoMindException):
    """Raised when a vector store operation (Qdrant) fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"Vector Store Error: {message}",
            status_code=502,
            details=details or {}
        )

class DatabaseException(EchoMindException):
    """Raised when a database persistence transaction fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"Database Transaction Error: {message}",
            status_code=500,
            details=details or {}
        )
