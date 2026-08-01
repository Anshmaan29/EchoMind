# CLI & API Reference

EchoMind exposes both Command Line Interfaces (CLIs) and HTTP endpoints for interacting with personal memory data.

---

## Command Line Interfaces (CLI)

All CLI commands are executed from the `backend/` directory using Python module execution syntax (`python -m app.cli.<module>`).

### 1. Ask RAG CLI (`app.cli.ask`)
Answers user queries by performing hybrid retrieval over vector stores, notes, git history, and timelines.

```bash
python -m app.cli.ask --query "What did I work on today?"
python -m app.cli.ask --query "What notes do I have about RAG?" --top-k 5
python -m app.cli.ask --query "Which commits changed the embedding provider?"
```

#### Options:
- `--query`, `-q` *(required)*: The query string.
- `--top-k`, `-k`: Number of evidence items to retrieve (default: 5).
- `--provider`, `-p`: LLM provider override (`mock`, `openai`).
- `--stream`, `-s`: Enable token streaming output.

---

### 2. Timeline CLI (`app.cli.timeline`)
Displays chronologically sorted project activity from git commits, note updates, and document modifications.

```bash
# Display today's events
python -m app.cli.timeline --today

# Display events from the past 7 days
python -m app.cli.timeline --week

# Display events from the past 30 days
python -m app.cli.timeline --month
```

#### Options:
- `--today`: Filter events from the current day.
- `--week`: Filter events from the past 7 days.
- `--month`: Filter events from the past 30 days.
- `--limit`, `-l`: Maximum number of events to display (default: 20).

---

### 3. Git Indexing CLI (`app.cli.git_index`)
Indexes local git repository commit history into Qdrant and local backup vector files.

```bash
python -m app.cli.git_index --repo-path .
```

#### Options:
- `--repo-path`, `-r`: Path to git repository directory (default: auto-discovered project root).
- `--provider`, `-p`: Embedding provider override (`mock`, `qwen`, `bge`, `openai`).
- `--batch-size`, `-b`: Custom batch size (default: auto -> CUDA: 8, CPU: 32).
- `--no-resume`: Reprocess all items, ignoring existing checkpoints.

---

### 4. Notes Indexing CLI (`app.cli.notes_index`)
Recursively scans Markdown (`.md`) and text (`.txt`) notes directories and generates vector embeddings.

```bash
# Automatically scans project root notes/ directory
python -m app.cli.notes_index

# Override notes directory path
python -m app.cli.notes_index --notes-dir /path/to/my/notes
```

#### Options:
- `--notes-dir`, `-n`: Path to notes directory (default: `<project_root>/notes`).
- `--provider`, `-p`: Embedding provider override (`mock`, `qwen`, `bge`, `openai`).
- `--batch-size`, `-b`: Custom batch size (default: auto -> CUDA: 8, CPU: 32).

---

### 5. Verify LLM Gateway (`app.cli.verify_llm`)
Validates runtime connectivity, streaming latency, and token response parameters for configured OpenAI-compatible endpoints.

```bash
python -m app.cli.verify_llm
```

---

## References

- [Architecture Guide](architecture.md)
- [Configuration Reference](configuration.md)
