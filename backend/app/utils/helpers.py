import hashlib
import uuid

def generate_uuid() -> str:
    """Generates a standard UUID4 string representation."""
    return str(uuid.uuid4())

def compute_hash(content: bytes | str) -> str:
    """Computes SHA-256 hash string for deduplication."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()

def clean_whitespace(text: str) -> str:
    """Removes excessive newlines and spaces from string content."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
