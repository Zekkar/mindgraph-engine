# MindGraph Engine

LLM-powered knowledge graph with hybrid RAG retrieval. Turn any folder of markdown notes into a searchable, interconnected wiki — with graph traversal, TF-IDF keyword search, and pgvector semantic search.

## Goals

- **Domain-agnostic**: works with any knowledge domain (trading notes, engineering docs, research papers, personal wikis)
- **Provider-agnostic**: swap LLM and embedding providers via config (Gemini, OpenAI, Anthropic)
- **Idempotent pipeline**: re-run `ingest` anytime; unchanged files skipped with zero API calls
- **Hybrid retrieval**: keyword (TF-IDF) + vector (pgvector cosine) + graph (1-hop boost)
- **No lock-in**: plain markdown in, plain markdown out; graph rebuilt on-the-fly from files

---

## Architecture

```
Data Sources
  filesystem adapter / custom adapter / future: HTTP, DB
       |
       | fetch() -> RawDocument[]
       v
raw/  (source of truth — never modified by LLM)
  raw/notes/*.md  raw/articles/*.md
       |
       | LLMProvider.complete(rewrite_prompt)
       v
wiki/  (LLM-compiled structured pages)
  Pages with ## Core Conclusions + [[wikilinks]]
       |                              |
       | WikiGraphEngine.build()      | EmbeddingStore.upsert()
       v                              v
  NetworkX Graph               PostgreSQL pgvector
  + TF-IDF index  <--boost---  wiki.embeddings table
       |                              |
       | KeywordRetriever             | VectorRetriever
       +------------------------------+
                    |
                    | two_stage_hybrid()
                    v
             Ranked ScoredDoc[]
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Graph backend | NetworkX in-memory | Zero dependencies; <500 pages = millisecond rebuild |
| Search | TF-IDF + pgvector cosine | Keyword finds exact terms; vector finds semantic matches |
| Community detection | Louvain | Built into NetworkX; meaningful at 20+ nodes |
| Edge confidence | Auto-inferred | EXTRACTED (wikilinks) > INFERRED (shared tags) |
| State tracking | `.ingest-state.json` + content hash | Re-run is free; only changed files call the LLM |
| Atomic state writes | `os.replace()` temp-file swap | State file never corrupted on crash mid-ingest |

---

## Package Structure

```
mindgraph-engine/
├── mindgraph/
│   ├── config.py          # MindGraphConfig — mindgraph.yml parser
│   ├── providers/
│   │   ├── base.py        # LLMProvider, EmbeddingProvider ABC
│   │   ├── gemini.py      # Gemini LLM + embedding
│   │   └── openai.py      # OpenAI / Anthropic LLM + embedding
│   ├── adapters/
│   │   ├── base.py        # DataSourceAdapter ABC, RawDocument dataclass
│   │   └── filesystem.py  # FileSystemAdapter
│   ├── graph.py           # WikiGraphEngine (NetworkX + TF-IDF + Louvain)
│   ├── store.py           # EmbeddingStore (pgvector), iter_sections()
│   ├── retrievers.py      # KeywordRetriever, VectorRetriever, two_stage_hybrid()
│   ├── router.py          # IntentRouter (pattern-based query classification)
│   ├── ingest.py          # run_ingest() pipeline
│   └── cli.py             # typer CLI entry point
├── tests/                 # 19 pytest tests
├── raw/                   # your source notes (gitignored: wiki/)
├── mindgraph.yml.example  # config template
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/you/mindgraph-engine
cd mindgraph-engine
pip install -e .
```

### 2. Configure

```bash
cp mindgraph.yml.example mindgraph.yml
cp .env.example .env
# Edit both files
```

`mindgraph.yml`:

```yaml
domain: "my-notes"
language: "en"

llm:
  provider: anthropic        # openai | gemini | anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY

embedding:
  provider: gemini
  model: gemini-embedding-001
  api_key_env: GEMINI_API_KEY

database:
  host: localhost
  port: 5432
  name: mindgraph
  user: mindgraph
  password_env: MINDGRAPH_DB_PASSWORD

adapters:
  - type: filesystem
    source: ./raw/notes
    target: raw/notes
```

### 3. Add notes and run

```bash
# Drop .md files into raw/notes/
cp ~/my-notes/*.md raw/notes/

# Run ingest: raw/ -> LLM rewrite -> wiki/
mindgraph ingest

# Check result
mindgraph check
mindgraph stats
mindgraph search "your query"
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `mindgraph ingest` | Run adapters, LLM-rewrite raw/ into wiki/ |
| `mindgraph embed` | Embed wiki/ sections into pgvector (idempotent; `--prune` sweeps orphans) |
| `mindgraph stats` | Show graph stats (pages, edges, communities) |
| `mindgraph search QUERY` | TF-IDF keyword search; `--hybrid` for intent-aware graph+vector |
| `mindgraph check` | wiki/ integrity + diagnostics (orphans, broken links, missing frontmatter); `--strict` exits 1 on broken links |
| `mindgraph coverage CODEBASE` | Measure how many code modules are covered by wiki knowledge |
| `mindgraph timeline CONCEPT` | How a concept evolved over time across raw/ dated notes |
| `mindgraph serve` | Run the REST API (FastAPI); binds `127.0.0.1` by default |
| `mindgraph mcp` | Run the MCP server (stdio) for Claude Code |

All commands accept `--config PATH` (default: `mindgraph.yml`) and `--base-dir PATH` (default: `.`).

### Ingest Pipeline

```
adapters.fetch()
  -> raw/{target}/*.md           (written once per file)
  -> llm.complete(rewrite_prompt)
  -> wiki/{stem}.md              (written when hash changes)
  -> .ingest-state.json          (atomic update after each file)
```

Second run: unchanged files → `skipped` counter goes up, zero API calls made.

### Embed

Splits each wiki page at H2/H3/H4 headings into `Section` chunks, embeds each chunk, upserts into `wiki.embeddings` with `UNIQUE(concept_name, section_id, model_version)`.

### Coverage

```bash
mindgraph coverage CODEBASE [--base-dir .] [--min-mentions 1] [--service-level] [--json]
```

Scans CODEBASE for `.py` files and reports which are referenced in `wiki/` pages.

**Coverage = modules with >= min-mentions wiki references / total modules scanned**

Options:
- `--service-level` — scan top-level directories as service units (not individual files)
- `--min-mentions N` — threshold for "covered" (default: 1)
- `--exclude` — comma-separated stems to skip (default: `__init__,test,conftest,migration,setup,manage`)
- `--json` — machine-readable output for CI pipelines

Examples:

```bash
# File-level scan
mindgraph coverage ./IBAPI/backend --base-dir .

# Service-level scan of ShioajiPy microservices
mindgraph coverage ./ShioajiPy --service-level --base-dir .

# CI gate
mindgraph coverage ./src --json | jq '.coverage_pct'
```

This implements the principle: **knowledge coverage = whether code modules are referenced in first-layer wiki knowledge**, not the volume of wiki pages.

---

## Server: REST API + MCP

```bash
mindgraph serve          # REST API (FastAPI) — binds 127.0.0.1 by default
mindgraph mcp            # MCP server over stdio (for Claude Code)
```

The REST API exposes the graph + hybrid search over HTTP:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness + page count |
| `GET /api/search?q=` | TF-IDF keyword search |
| `GET /api/smart_search?q=` | intent-aware graph+vector hybrid (degrades to keyword if no DB/keys) |
| `GET /api/concept/{name}` | full page + metadata + neighbors |
| `GET /api/concept/{name}/sections` · `/section/{id}` | section navigation |
| `GET /api/related/{name}?depth=` | graph traversal |
| `GET /api/communities` · `/api/god-nodes` · `/api/stats` · `/api/path` | graph analytics |
| `GET /api/timeline/{concept}` | chronological mentions of a concept across raw/ notes |

The MCP server registers 8 tools (`search`, `smart_search`, `get_concept`, `related`, `communities`, `stats`, `god_nodes`, `timeline`) so an MCP client like Claude Code can query the knowledge base as external memory.

> ⚠️ **Security**: the API has **no authentication** and serves the entire knowledge base read-only. `serve` binds `127.0.0.1` by default and the Docker compose publishes the port on host loopback only. Use `--host 0.0.0.0` — behind an auth proxy — only when you deliberately want network exposure. `cors_origins` defaults to `[]`; never set `"*"` without an auth layer.

---

## Provider Support

### LLM

| `provider` value | Supported models |
|-----------------|-----------------|
| `anthropic` | claude-sonnet-4-6, claude-haiku-4-5-20251001, claude-opus-4-8 |
| `openai` | gpt-4o-mini, gpt-4o |
| `gemini` | gemini-2.0-flash, gemini-pro |

### Embedding

| `provider` value | Model | Vector dim |
|-----------------|-------|-----------|
| `gemini` | gemini-embedding-001 | 768 |
| `openai` | text-embedding-3-small | 1536 |
| `openai` | text-embedding-3-large | 3072 |

---

## Adapter System

Adapters decouple data sources from the ingest pipeline.

### FileSystemAdapter (built-in)

```yaml
adapters:
  - type: filesystem
    source: ./raw/notes     # any local path
    target: raw/notes       # destination inside raw/
```

### Custom Adapter

```python
from mindgraph.adapters.base import DataSourceAdapter, RawDocument

class NotionAdapter(DataSourceAdapter):
    def fetch(self) -> list[RawDocument]:
        # pull pages from Notion API
        return [RawDocument(content=..., filename=..., metadata={...})]

    def get_target_dir(self) -> str:
        return "raw/notion"
```

```yaml
adapters:
  - type: custom
    module: myproject.adapters.NotionAdapter
```

---

## Graph Model

Pages = nodes. Edges auto-created in three tiers:

| Confidence | Source | Example |
|-----------|--------|---------|
| `EXTRACTED` | `[[wikilinks]]` in content | `[[PageB]]` found in PageA |
| `EXTRACTED` | `related:` in frontmatter | `related: [PageB]` |
| `INFERRED` | Shared non-generic tags | Both have tag `machine-learning` |

Louvain community detection groups nodes automatically. See communities with `mindgraph stats`.

---

## Ingest State Machine

```
[new/changed file]
       |
       v
  pending ──(LLM ok)──> ingested ──(same hash next run)──> skipped
       |
       └──(LLM error)──> failed ──(retried next run)──> ...
```

State stored per-file in `.ingest-state.json`:

```json
{
  "version": 1,
  "files": {
    "raw/notes/my-note.md": {
      "hash": "a1b2c3d4",
      "status": "ingested",
      "ingested_at": "2026-05-31T01:00:00+00:00"
    }
  }
}
```

---

## Docker

```bash
docker compose up -d
```

Services:
- `mindgraph` — engine container, mounts project dir as `/data`
- `postgres` — `pgvector/pgvector:pg16`, healthcheck included

Set env via `.env` file (see `.env.example`).

---

## Development

```bash
pip install -e .
pytest tests/ -v        # 72 tests
```

| Test file | Covers |
|-----------|--------|
| `test_config.py` | YAML parsing, defaults, intent patterns |
| `test_providers.py` | ABC enforcement, Gemini/OpenAI/Anthropic mocks, factory dispatch |
| `test_adapters.py` | FileSystemAdapter, empty dirs |
| `test_graph.py` | build, TF-IDF search, concept lookup, incremental rebuild, diagnostics |
| `test_ingest.py` | Pipeline write, idempotency |
| `test_search.py` | SmartSearchService: intent routing, degradation, failure boost, Pass-2 expansion |
| `test_retrievers.py` | keyword normalization, graph boost_fn, vector degradation, embed_query |
| `test_store.py` | section chunking, content hash, model_version format, build_dsn |
| `test_server.py` | REST endpoints (TestClient) + MCP tool registration |
| `test_cli.py` | serve / mcp / search --hybrid / embed idempotency / timeline / check via CliRunner |
| `test_timeline.py` | concept timeline: date resolution, chronological ordering, snippets |

---

## Roadmap

### Phase 1 — MVP ✅ (v0.1.0)
- [x] CLI: ingest / embed / stats / search / check / coverage
- [x] Provider abstraction (Gemini, OpenAI, Anthropic)
- [x] FileSystemAdapter + custom adapter protocol
- [x] WikiGraphEngine (NetworkX + TF-IDF + Louvain)
- [x] EmbeddingStore (pgvector, ensure_schema, mark-and-sweep)
- [x] Idempotent ingest with atomic state file

### Phase 2 — Server ✅ (v0.2.0)
- [x] REST API (`mindgraph serve`) — 10 endpoints, localhost-bound by default
- [x] MCP server for Claude Code integration (`mindgraph mcp`, 7 tools)
- [x] Migrate `google.generativeai` → `google.genai`
- [x] `mindgraph check` extended diagnostics (orphans, broken wikilinks, missing frontmatter, `--strict`)

### Phase 3 — Scale ✅ (v0.2.0)
- [x] Hybrid retrieval (`two_stage_hybrid`) wired into `search --hybrid` + REST/MCP `smart_search`
- [x] IntentRouter wired to search (Pass 1 regex classify + Pass 2 LLM expansion)
- [x] Incremental graph rebuild (mtime file-diff: re-parse only changed files)
- [x] Concept evolution timeline (`mindgraph timeline` + `/api/timeline`, devdiary-aware)

---

## License

MIT
