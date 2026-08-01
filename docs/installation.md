# Installation & Setup Guide

This guide covers setting up EchoMind for local development and testing.

---

## Prerequisites

- **Python**: 3.11 or higher
- **Git**: 2.30 or higher
- **Docker & Docker Compose** *(Optional)*: For running Qdrant, PostgreSQL, and Neo4j locally.

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Anshmaan29/EchoMind.git
cd EchoMind
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the backend package in editable mode:

```bash
cd backend
pip install -e .
```

### 4. Configure Environment Variables

Create your local `.env` file from the example template:

```bash
cp .env.example .env
```

### 5. Launch Infrastructure Services (Optional)

To start PostgreSQL, Qdrant, and Neo4j containerized services:

```bash
cd ..
docker compose up -d
```

*Note: If Docker services are not running, EchoMind automatically falls back to local JSONL vector storage and in-memory mock stores for zero-dependency local development.*

---

## Verifying Setup

Run the baseline RAG interface to verify system connectivity:

```bash
cd backend
python -m app.cli.ask --query "System status check"
```

---

## Next Steps

- [Configuration Reference](configuration.md)
- [API & CLI Reference](api.md)
- [Connectors Overview](connectors.md)
