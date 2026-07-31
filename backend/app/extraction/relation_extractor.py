import re
from app.extraction.base import BaseRelationshipExtractor
from app.schemas.entity import EntityCreate
from app.schemas.relationship import RelationshipCreate, RelationType

class RelationshipExtractor(BaseRelationshipExtractor):
    """
    Relationship Extractor Engine discovering directed relations across entities.
    """
    def __init__(self) -> None:
        self.relation_verbs: list[tuple[str, RelationType]] = [
            (r"\b(created|built|developed|authored|designed)\b", "CREATED"),
            (r"\b(uses|utilizes|leverages|adopts)\b", "USES"),
            (r"\b(depends on|requires|relies on)\b", "DEPENDS_ON"),
            (r"\b(part of|belongs to|component of)\b", "PART_OF"),
            (r"\b(owns|maintains|manages)\b", "OWNS"),
            (r"\b(worked on|contributed to)\b", "WORKED_ON"),
            (r"\b(related to|associated with)\b", "RELATED_TO"),
            (r"\b(mentions|cites)\b", "MENTIONS"),
            (r"\b(references|links to)\b", "REFERENCES"),
            (r"\b(generates|produces)\b", "GENERATES"),
            (r"\b(implements|fulfills)\b", "IMPLEMENTS"),
            (r"\b(fixes|resolves|closes)\b", "FIXES"),
            (r"\b(discussed in|debated in)\b", "DISCUSSED_IN"),
            (r"\b(located in|hosted on)\b", "LOCATED_IN"),
            (r"\b(trained on|fine-tuned on)\b", "TRAINED_ON"),
            (r"\b(connected to|interfaced with)\b", "CONNECTED_TO"),
        ]

    async def extract_relationships(
        self,
        text: str,
        entities: list[EntityCreate],
        source_document_id: str | None = None
    ) -> list[RelationshipCreate]:
        relationships: list[RelationshipCreate] = []
        seen_pairs: set[tuple[str, str, str]] = set()

        if len(entities) < 2 or not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            # Check entity pairs co-occurring in sentence
            matched_in_sent = [e for e in entities if e.name.lower() in sent_clean.lower()]
            if len(matched_in_sent) < 2:
                continue

            for i in range(len(matched_in_sent)):
                for j in range(i + 1, len(matched_in_sent)):
                    e1 = matched_in_sent[i]
                    e2 = matched_in_sent[j]

                    rel_type: RelationType = "CONNECTED_TO"
                    confidence = 0.70

                    # Find explicit verb in sentence
                    for verb_pattern, relation in self.relation_verbs:
                        if re.search(verb_pattern, sent_clean, re.IGNORECASE):
                            rel_type = relation
                            confidence = 0.90
                            break

                    key = (e1.name.lower(), rel_type, e2.name.lower())
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        relationships.append(
                            RelationshipCreate(
                                source_id=e1.name,  # Mapped to entity name / ID during pipeline creation
                                target_id=e2.name,
                                relation_type=rel_type,
                                confidence=confidence,
                                evidence=sent_clean[:250],
                                source_document_id=source_document_id
                            )
                        )

        return relationships
