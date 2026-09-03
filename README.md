# RAG Citation Agent

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions from your documents with cited sources. Ask anything — the agent retrieves the most relevant passages, generates a grounded answer, and tells you exactly which document and page it came from.

## How it works

```
Document → Chunk → Embed → Store in Qdrant
                                    ↓
User Query → Multi-Query Expansion → Embed → Search → Deduplicate → Rerank → Generate → Cited Answer
```

The pipeline has two stages:

**Ingestion (offline):** Documents are split into overlapping chunks using a recursive text splitter that respects natural language boundaries. Each chunk is embedded using Google's `gemini-embedding-001` model and stored in a Qdrant vector database.

**Retrieval & Generation (online):** When a user asks a question, the agent generates 3-5 alternative phrasings of the query using Gemini, runs vector search for each phrasing, deduplicates the combined candidate set, and passes the top results to the generator. The generator produces a grounded answer with explicit `[Source N]` citations.

## Architecture

```
rag-citation-agent/
├── src/
│   ├── config.py        # Pydantic settings, environment variable loading
│   ├── chunker.py       # Recursive text chunker with overlap
│   ├── embedder.py      # Gemini embedding model wrapper
│   ├── vectorstore.py   # Qdrant vector store (in-memory)
│   ├── retriever.py     # Multi-query retrieval with deduplication
│   └── generator.py     # Cited answer generation
├── tests/
│   ├── test_chunker.py
│   └── test_retriever.py
├── data/                # Place your documents here
├── main.py              # Entry point
├── requirements.txt
└── .env.example
```

## Key concepts demonstrated

- **Recursive chunking** — splits at paragraph → sentence → word boundaries, preserving natural language structure with configurable overlap
- **Multi-query retrieval** — generates alternative query phrasings to cast a wider retrieval net, then deduplicates results
- **Vector similarity search** — cosine similarity over 3072-dimensional Gemini embeddings stored in Qdrant
- **Citation grounding** — every answer includes `[Source N]` references with filename and page number
- **Production config pattern** — all settings via environment variables, fails loudly at startup if secrets are missing

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/SahilSastakar/rag-citation-agent.git
cd rag-citation-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key. Get one free at [aistudio.google.com](https://aistudio.google.com).

```
GEMINI_API_KEY=your_key_here
GENERATION_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=models/gemini-embedding-001
```

**4. Run**
```bash
python main.py
```

The agent will create a sample document automatically on first run and answer three questions against it.

## Configuration

All settings are controlled via `.env` — no code changes needed to tune behaviour:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | required | Your Gemini API key |
| `GENERATION_MODEL` | `gemini-3.6-flash` | Model for answer generation |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model for embeddings |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |
| `TOP_K_RETRIEVAL` | `20` | Candidates retrieved per query |
| `TOP_K_RERANK` | `5` | Final chunks passed to generator |

## Using your own documents

Replace the sample document path in `main.py`:

```python
ingest_document("data/your_document.txt", vectorstore)
```

Then ask questions:

```python
ask("Your question here", retriever, generator)
```

## Tech stack

| Tool | Purpose |
|---|---|
| Google Gemini | Embeddings and answer generation |
| Qdrant | Vector database (in-memory) |
| Pydantic v2 | Settings management and validation |
| tiktoken | Token counting for chunk sizing |
| loguru | Structured logging |
| pytest | Testing |

## Example output

```
============================================================
ANSWER
============================================================
AI alignment is the challenge of ensuring that AI systems pursue goals
that humans actually want [Source 1]. It matters because misaligned AI
systems might pursue proxy goals that diverge from true human
intentions [Source 1].

============================================================
SOURCES
============================================================
[Source 1] sample.txt (Page N/A, Chunk 0)

============================================================
Chunks used: 1
============================================================
```

## Project roadmap

This is Project 2 of a 12-project AI/ML engineering portfolio. The roadmap covers RAG, agentic systems, multi-agent orchestration, and production ML deployment.
