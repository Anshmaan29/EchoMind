import re
from app.extraction.base import BaseEntityExtractor
from app.schemas.entity import EntityCreate, EntityType

class EntityExtractor(BaseEntityExtractor):
    """
    Rule-based and NLP pattern Entity Extraction Engine supporting 23 Entity Types.
    """
    def __init__(self) -> None:
        # Pattern rules matching tech, programming languages, repos, versions, dates, etc.
        self.tech_patterns: list[tuple[str, EntityType]] = [
            (r"\b(FastAPI|Django|Flask|React|Next\.js|Vue|Angular|Spring|Express)\b", "Framework"),
            (r"\b(Python|JavaScript|TypeScript|Rust|Go|C\+\+|Java|Kotlin|Scala|Swift|SQL)\b", "Programming Language"),
            (r"\b(PostgreSQL|Postgres|Qdrant|Neo4j|Redis|MongoDB|Pinecone|Milvus|SQLite)\b", "Technology"),
            (r"\b(Docker|Kubernetes|AWS|GCP|Azure|Terraform|Helm)\b", "Technology"),
            (r"\b(PyTorch|TensorFlow|Scikit-Learn|HuggingFace|LangChain|LlamaIndex)\b", "Library"),
            (r"\b(GitHub|GitLab|Bitbucket)\b", "Repository"),
            (r"\b(v\d+\.\d+(?:\.\d+)?|\d+\.\d+\.\d+)\b", "Version"),
            (r"\bhttps?://[^\s/$.?#].[^\s]*\b", "URL"),
            (r"\bPR-\d+|\b[A-Z]+-\d+\b|\bIssue #?\d+\b", "Issue"),
            (r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b", "Date"),
        ]

    async def extract_entities(self, text: str, source_document_id: str | None = None) -> list[EntityCreate]:
        entities: list[EntityCreate] = []
        seen_names: set[tuple[str, str]] = set()

        if not text.strip():
            return []

        # 1. Match specific regex patterns
        for pattern, entity_type in self.tech_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(0).strip()
                key = (name.lower(), entity_type)
                if key not in seen_names and len(name) > 1:
                    seen_names.add(key)
                    entities.append(
                        EntityCreate(
                            name=name,
                            type=entity_type,
                            aliases=[name.lower()],
                            description=f"Extracted {entity_type} '{name}'",
                            confidence=0.95,
                            source_document_id=source_document_id
                        )
                    )

        # 2. Extract Capitalized Names & Concepts (Person, Organization, Project, Concept)
        capitalized_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b", text)
        for phrase in capitalized_phrases:
            phrase_clean = phrase.strip()
            if len(phrase_clean) <= 2 or phrase_clean in ["The", "A", "An", "In", "On", "This", "That", "EchoMind"]:
                continue

            entity_type: EntityType = "Concept"
            if any(w in phrase_clean for w in ["Inc", "Corp", "Ltd", "LLC", "Google", "OpenAI", "Microsoft"]):
                entity_type = "Organization"
            elif any(w in phrase_clean for w in ["Engine", "System", "Platform", "Memory", "Store", "EchoMind"]):
                entity_type = "Project"

            key = (phrase_clean.lower(), entity_type)
            if key not in seen_names:
                seen_names.add(key)
                entities.append(
                    EntityCreate(
                        name=phrase_clean,
                        type=entity_type,
                        aliases=[phrase_clean.lower()],
                        description=f"Extracted {entity_type} phrase '{phrase_clean}'",
                        confidence=0.85,
                        source_document_id=source_document_id
                    )
                )

        return entities
