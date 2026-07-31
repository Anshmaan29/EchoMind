import json
import os
from typing import Any
from app.core.logging import logger

class JSONLBackupWriter:
    """
    Local JSONL Streamer writing generated vector payloads to disk for data backup and auditability.
    Contains: id, source, content, embedding_model, embedding_vector, metadata.
    """
    def __init__(self, backup_filepath: str = "data/embeddings_backup.jsonl") -> None:
        self.backup_filepath = backup_filepath
        os.makedirs(os.path.dirname(self.backup_filepath), exist_ok=True)

    def write_records(
        self,
        records: list[dict[str, Any]]
    ) -> int:
        """
        Appends vector records line-by-line in JSONL format.
        
        :param records: List of record dictionaries.
        :return: Number of written records.
        """
        if not records:
            return 0

        count = 0
        try:
            with open(self.backup_filepath, "a", encoding="utf-8") as f:
                for rec in records:
                    line = json.dumps({
                        "id": rec.get("id"),
                        "source": rec.get("source", "unknown"),
                        "content": rec.get("content", ""),
                        "embedding_model": rec.get("embedding_model", "default"),
                        "embedding_vector": rec.get("embedding_vector", []),
                        "metadata": rec.get("metadata", {})
                    })
                    f.write(line + "\n")
                    count += 1
            logger.info(f"Appended {count} embedding records to local backup '{self.backup_filepath}'.")
        except Exception as e:
            logger.error(f"Error writing to JSONL backup file '{self.backup_filepath}': {e}")

        return count
