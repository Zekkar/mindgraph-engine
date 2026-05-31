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
| `mindgraph embed` | Embed wiki/ sections into pgvector |
| `mindgraph stats` | Show graph stats (pages, edges, communities) |
| `mindgraph search QUERY` | TF-IDF keyword search |
| `mindgraph check` | Verify wiki/ integrity; exits 1 if empty |

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
pytest tests/ -v        # 19 tests
```

| Test file | Covers |
|-----------|--------|
| `test_config.py` | YAML parsing, defaults, intent patterns |
| `test_providers.py` | ABC enforcement, Gemini/OpenAI mock calls |
| `test_adapters.py` | FileSystemAdapter, empty dirs |
| `test_graph.py` | Build, TF-IDF search, concept lookup |
| `test_ingest.py` | Pipeline write, idempotency |

---

## Roadmap

### Phase 1 — MVP (v0.1.0, current)
- [x] CLI: ingest / embed / stats / search / check
- [x] Provider abstraction (Gemini, OpenAI, Anthropic)
- [x] FileSystemAdapter + custom adapter protocol
- [x] WikiGraphEngine (NetworkX + TF-IDF + Louvain)
- [x] EmbeddingStore (pgvector, ensure_schema, mark-and-sweep)
- [x] Idempotent ingest with atomic state file

### Phase 2 — Server
- [ ] REST API (`mindgraph serve`)
- [ ] MCP server for Claude Code integration
- [ ] Migrate `google.generativeai` → `google.genai`
- [ ] `mindgraph check` extended diagnostics

### Phase 3 — Scale
- [ ] Hybrid retrieval endpoint (`two_stage_hybrid` via REST)
- [ ] IntentRouter wired to search
- [ ] Incremental graph rebuild (file-diff based)
- [ ] Concept evolution timeline (devdiary integration)

---

## License

MIT
