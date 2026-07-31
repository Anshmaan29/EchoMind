import os
import sqlite3
from typing import Set
from app.core.logging import logger

class CheckpointManager:
    """
    Lightweight SQLite-backed checkpoint manager tracking processed item IDs.
    Enables instant pipeline resumption after network interruptions or system restarts.
    """
    def __init__(self, db_path: str = ".checkpoints/embedding_checkpoint.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processed_items (
                        item_id TEXT PRIMARY KEY,
                        source TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing CheckpointManager DB: {e}")

    def is_processed(self, item_id: str) -> bool:
        """Checks if an item ID has already been successfully processed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM processed_items WHERE item_id = ?", (item_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Checkpoint check error: {e}")
            return False

    def get_processed_ids(self) -> Set[str]:
        """Returns set of all processed item IDs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT item_id FROM processed_items")
                rows = cursor.fetchall()
                return {r[0] for r in rows}
        except Exception as e:
            logger.error(f"Error fetching processed IDs: {e}")
            return set()

    def mark_processed_batch(self, item_ids: list[str], source: str = "unknown") -> None:
        """Marks a batch of item IDs as processed."""
        if not item_ids:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                records = [(item_id, source) for item_id in item_ids]
                conn.executemany(
                    "INSERT OR IGNORE INTO processed_items (item_id, source) VALUES (?, ?)",
                    records
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording checkpoint batch: {e}")

    def reset(self) -> None:
        """Resets all checkpoint state."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM processed_items")
                conn.commit()
            logger.info("Reset CheckpointManager ledger.")
        except Exception as e:
            logger.error(f"Error resetting checkpoint ledger: {e}")
