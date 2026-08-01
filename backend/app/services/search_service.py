import json
import os
import re
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.factory import embedding_provider
from app.vector.base import BaseVectorStore
from app.vector.factory import vector_store

# Paths that are always deprioritised no matter their cosine score
_NOISE_PATH_FRAGMENTS = {
    ".pytest_cache",
    "__pycache__",
    "/tests/",
    "\\tests\\",
    "/node_modules/",
    "/.venv/",
    "/dist/",
    "/build/",
    ".mypy_cache",
    "/experiments/outputs/",
    "embeddings_backup.jsonl",
    ".checkpoints",
}


class SearchResult(BaseModel):
    """
    Standardized Hybrid Search Result container preserving file paths and line numbers.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    filepath: str
    filename: str
    start_line: int
    end_line: int
    score: float
    content: str
    source: str
    meta_data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hybrid Scoring Helpers
# ---------------------------------------------------------------------------

def _tokenize_query(query: str) -> list[str]:
    """Lowercased words + split CamelCase parts."""
    raw = re.findall(r"[A-Za-z0-9_]+", query)
    tokens: list[str] = []
    for word in raw:
        tokens.append(word.lower())
        # Split CamelCase e.g. QwenEmbeddingProvider -> qwen embedding provider
        parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", word)
        for p in parts:
            if len(p) >= 3:
                tokens.append(p.lower())
    return list(set(tokens))


def _is_noise_path(filepath: str) -> bool:
    fp_norm = filepath.replace("\\", "/")
    for frag in _NOISE_PATH_FRAGMENTS:
        if frag.replace("\\", "/") in fp_norm:
            return True
    return False


def _compute_hybrid_boost(
    query_tokens: list[str],
    query_raw: str,
    rec: dict[str, Any],
    meta: dict[str, Any],
) -> float:
    """
    Returns a hybrid boost score [0.0, 1.0] based on:
      - filename token match
      - class / function name match
      - defined symbols match
      - exact content token match
    """
    boost = 0.0

    filepath = str(meta.get("filepath", "")).replace("\\", "/")
    filename = os.path.basename(filepath).lower()
    content = str(rec.get("content", "")).lower()

    # 1. Filename match  (weight 0.20)
    for tok in query_tokens:
        if len(tok) >= 3 and tok in filename:
            boost += 0.20
            break

    # 2. Class / function / defined symbols / note metadata match  (weight 0.20)
    class_names = [c.lower() for c in meta.get("class_names", [])]
    function_names = [f.lower() for f in meta.get("function_names", [])]
    file_class_names = [c.lower() for c in meta.get("file_class_names", [])]
    defined_syms = [s.lower() for s in meta.get("defined_symbols", [])]
    note_title = str(meta.get("title", "")).lower()
    note_tags = [t.lower() for t in meta.get("tags", [])]
    note_headings = [h.lower() for h in meta.get("headings", [])]

    all_syms = set(
        class_names + function_names + file_class_names + defined_syms +
        ([note_title] if note_title else []) + note_tags + note_headings
    )
    query_lower = query_raw.lower()

    for sym in all_syms:
        if sym and sym in query_lower:
            boost += 0.20
            break
    for tok in query_tokens:
        if len(tok) >= 3 and any(tok in sym for sym in all_syms):
            boost += 0.10
            break

    # 3. Exact identifier in content  (weight 0.15)
    for tok in query_tokens:
        if len(tok) >= 4 and tok in content:
            boost += 0.15
            break

    # 4. Full query phrase in content  (weight 0.10)
    if query_lower.replace(" ", "").replace("_", "") in content.replace(" ", "").replace("_", ""):
        boost += 0.10

    # 5. Noise path penalty  (-0.40)
    if _is_noise_path(filepath):
        boost -= 0.40

    return boost


class SearchService:
    """
    Hybrid Vector Search Service with structural re-ranking.
    Pipeline:
      1. Cosine similarity over JSONL / Qdrant  (semantic signal)
      2. Hybrid boost:  filename + class/func names + exact token matching
      3. Final score = 0.55 * cosine + boost  (capped at 1.0, floored at 0.0)
      4. Re-sort by final score
    """
    def __init__(
        self,
        embedder: BaseEmbeddingProvider = None,
        vector_store_inst: BaseVectorStore = None,
        backup_filepath: str = None
    ) -> None:
        self.embedder = embedder or embedding_provider
        self.vector_store = vector_store_inst or vector_store
        self.backup_filepath = backup_filepath or self._discover_backup_filepath()

    def _discover_backup_filepath(self) -> str:
        candidates = [
            "data/embeddings_backup.jsonl",
            "../data/embeddings_backup.jsonl",
            "embeddings_backup.jsonl",
            "../embeddings_backup.jsonl",
            "/Users/anshmaansingh/Echomind/data/embeddings_backup.jsonl",
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 0:
                return c
        return "data/embeddings_backup.jsonl"

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        collection_name: str = None
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        target_collection = collection_name or settings.QDRANT_COLLECTION_NAME
        query_vec = await self.embedder.embed_single(query)
        query_tokens = _tokenize_query(query)

        # 1. Try Qdrant
        qdrant_results: list[SearchResult] = []
        try:
            is_qdrant_available = getattr(self.vector_store, "is_available", None)
            if is_qdrant_available is not False:
                hits = await self.vector_store.search(
                    collection_name=target_collection,
                    query_vector=query_vec,
                    limit=top_k * 4,
                    score_threshold=0.0
                )
                raw: list[tuple[float, SearchResult]] = []
                for hit in hits:
                    payload = hit.payload or {}
                    fp = payload.get("filepath", payload.get("filename", "unknown"))
                    meta = payload
                    boost = _compute_hybrid_boost(query_tokens, query, {"content": payload.get("content", "")}, meta)
                    final = min(1.0, max(0.0, 0.55 * hit.score + boost))
                    raw.append((final, SearchResult(
                        id=hit.id,
                        filepath=fp,
                        filename=payload.get("filename", "unknown"),
                        start_line=payload.get("start_line", 1),
                        end_line=payload.get("end_line", 1),
                        score=round(final, 4),
                        content=payload.get("content", ""),
                        source=payload.get("source", "unknown"),
                        meta_data=meta,
                    )))
                raw.sort(key=lambda x: x[0], reverse=True)
                seen: set[tuple[str, int, int]] = set()
                for final_score, res in raw:
                    key = (res.filepath, res.start_line, res.end_line)
                    if key in seen or final_score < min_score:
                        continue
                    seen.add(key)
                    qdrant_results.append(res)
                    if len(qdrant_results) >= top_k:
                        break
        except Exception:
            pass

        if qdrant_results:
            return qdrant_results

        # 2. JSONL fallback with hybrid re-ranking
        return self._search_jsonl_backup(
            query_raw=query,
            query_tokens=query_tokens,
            query_vec=query_vec,
            top_k=top_k,
            min_score=min_score
        )

    def _search_jsonl_backup(
        self,
        query_raw: str,
        query_tokens: list[str],
        query_vec: list[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        if not os.path.exists(self.backup_filepath):
            return []

        records: list[dict[str, Any]] = []
        vectors: list[list[float]] = []

        try:
            with open(self.backup_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    vec = rec.get("embedding_vector")
                    if vec and len(vec) == len(query_vec):
                        records.append(rec)
                        vectors.append(vec)
        except Exception:
            return []

        if not vectors:
            return []

        # Cosine similarity
        matrix = np.array(vectors, dtype=np.float32)
        q_vec = np.array(query_vec, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec) or 1e-9
        norm_matrix = np.linalg.norm(matrix, axis=1)
        norm_matrix[norm_matrix == 0] = 1e-9
        cosine_scores = np.dot(matrix, q_vec) / (norm_matrix * norm_q)

        # Hybrid final scores
        final_scores: list[float] = []
        for idx, cos in enumerate(cosine_scores):
            rec = records[idx]
            meta = rec.get("metadata", {})
            boost = _compute_hybrid_boost(query_tokens, query_raw, rec, meta)
            final = min(1.0, max(-1.0, 0.55 * float(cos) + boost))
            final_scores.append(final)

        ranked_indices = np.argsort(final_scores)[::-1]

        results: list[SearchResult] = []
        seen: set[tuple[str, int, int]] = set()

        for idx in ranked_indices:
            final = final_scores[idx]
            if final < min_score:
                continue
            rec = records[idx]
            meta = rec.get("metadata", {})
            fp = str(meta.get("filepath", meta.get("filename", "unknown")))
            s_line = int(meta.get("start_line", 1))
            e_line = int(meta.get("end_line", 1))

            key = (fp, s_line, e_line)
            if key in seen:
                continue
            seen.add(key)

            results.append(SearchResult(
                id=str(rec.get("id", f"backup_{idx}")),
                filepath=fp,
                filename=str(meta.get("filename", "unknown")),
                start_line=s_line,
                end_line=e_line,
                score=round(final, 4),
                content=str(rec.get("content", "")),
                source=str(rec.get("source", "unknown")),
                meta_data=meta,
            ))

            if len(results) >= top_k:
                break

        return results
