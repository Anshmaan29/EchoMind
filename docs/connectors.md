# Connectors Overview

EchoMind uses modular data connectors to ingest digital artifacts from software development workflows and note repositories.

---

## Supported Connectors

| Source | Status | Connector Module | Output Schema |
| :--- | :--- | :--- | :--- |
| **Git Repositories** | Supported | `app.ingestion.git_connector.GitConnector` | `SourceType.GITHUB` / `SourceType.TIMELINE` |
| **Personal Notes** | Supported | `app.ingestion.note_connector.NoteConnector` | `SourceType.NOTES` |
| **PDF Documents** | Supported | `app.ingestion.loaders.pdf_loader.PDFLoader` | `SourceType.PDF` |
| **Markdown Files** | Supported | `app.ingestion.note_connector.NoteConnector` | `SourceType.NOTES` |
| **WhatsApp Exports** | Planned | `app.ingestion.whatsapp_connector` | `SourceType.CHAT` |
| **Audio Files** | Planned | `app.ingestion.audio_connector` | `SourceType.AUDIO` |
| **Calendar** | Planned | `app.ingestion.calendar_connector` | `SourceType.CALENDAR` |
| **Gmail** | Planned | `app.ingestion.gmail_connector` | `SourceType.GMAIL` |

---

## Technical Details

### Git Connector
The `GitConnector` scans local git repositories to extract:
- Commit messages and unique commit hashes (`git://commit/<hash>`)
- Author identity and commit timestamps
- Inserted/deleted line stats and modified file paths
- Granular file diffs for vector indexing and timeline discovery

### Personal Notes Connector
The `NoteConnector` scans directory trees (`notes/` by default) for `.md` and `.txt` files to extract:
- Document title from top heading (`# Title`) or filename
- Section headings and structure
- Frontmatter tags (`tags: [...]`) and internal Markdown links (`[[link]]` / `[text](url)`)
- File creation and modification dates from filesystem stat metadata
- Clean text chunks formatted into standardized `EmbeddingItem` payloads

### PDF Connector
The `PDFLoader` and `PDFParser` extract structured body text, page metadata, and layout information from PDF documents, producing chunked text blocks for indexing.

---

## References

- [Architecture Guide](architecture.md)
- [API & CLI Reference](api.md)
