# EchoMind Git Memory — docs/git_memory.md

## Overview

EchoMind can index the entire **Git commit history** of any repository into its vector search engine. Once indexed, you can ask natural language questions over your commit history using the Ask CLI:

```bash
python -m app.cli.ask --query "What changed today?"
python -m app.cli.ask --query "What was my latest embedding commit?"
python -m app.cli.ask --query "Which files changed in the hybrid retrieval commit?"
```

---

## Architecture

```
Git Repository (local)
        │
        ▼
GitConnector (subprocess + git CLI)
  ├── commit hash
  ├── author / email
  ├── commit date (ISO 8601)
  ├── branch
  ├── commit message
  ├── files changed
  └── added / deleted lines
        │
        ▼
EmbeddingItem  (source="timeline")
        │
        ▼
GenericEmbeddingPipeline
  ├── MockEmbeddingProvider (local dev)
  └── QwenEmbeddingProvider (AI Kosh GPU)
        │
        ▼
embeddings_backup.jsonl  +  Qdrant (when available)
        │
        ▼
SearchService (hybrid ranker)
        │
        ▼
AskService  →  CLI answer with evidence
```

---

## Indexing Commands

### Index full history (default: last 500 commits)

```bash
cd backend
python -m app.cli.git_index
```

### Index from a specific date

```bash
python -m app.cli.git_index --since 2026-07-01
```

### Index a specific repo path

```bash
python -m app.cli.git_index --repo /path/to/other/repo
```

### Force re-index (ignore checkpoint)

```bash
python -m app.cli.git_index --no-resume
```

---

## Commit Metadata

Every commit is stored as an `EmbeddingItem` with source `"timeline"` and the following metadata fields:

| Field            | Type         | Description                           |
|------------------|--------------|---------------------------------------|
| `commit`         | `str`        | Full 40-char SHA hash                 |
| `short_hash`     | `str`        | First 8 chars of hash                 |
| `author`         | `str`        | Author full name                      |
| `author_email`   | `str`        | Author email                          |
| `date`           | `str`        | ISO 8601 timestamp with timezone      |
| `date_human`     | `str`        | `YYYY-MM-DD HH:MM UTC` readable form  |
| `branch`         | `str`        | Branch name at time of commit         |
| `message`        | `str`        | Full commit message subject           |
| `files_changed`  | `list[str]`  | Relative paths of all modified files  |
| `added_lines`    | `int`        | Total lines added                     |
| `deleted_lines`  | `int`        | Total lines deleted                   |

---

## Searching Git Memory

After indexing, Git commits are searchable alongside code chunks via the same `SearchService`:

```bash
python -m app.cli.search --query "What changed today?"
python -m app.cli.search --query "hybrid retrieval commit"
python -m app.cli.search --query "embedding pipeline changes"
```

---

## Ask CLI Examples

```bash
python -m app.cli.ask --query "What changed today?"
# → cites git://commit/<hash> with date, author, files, message

python -m app.cli.ask --query "What was my latest embedding commit?"
# → cites the most relevant commit mentioning embedding pipeline changes

python -m app.cli.ask --query "Which files were modified in the RAG milestone?"
# → retrieves commits whose message/files match RAG-related terms
```

---

## No New Dependencies

The Git connector uses Python's built-in `subprocess` module and the system `git` CLI. No GitPython or other packages are required.
