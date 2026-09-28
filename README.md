# MindGraph Engine

LLM-powered knowledge graph with hybrid RAG retrieval. Turn any folder of markdown notes into a searchable, interconnected wiki. Retrieval combines keyword search, pgvector semantic search, recency weighting and graph co-occurrence. MindGraph also recalls past failure modes to your AI agent at the moments it edits, commits or deploys.

## Goals

- **Domain-agnostic**: works with any knowledge domain (engineering docs, research notes, personal wikis)
- **Provider-agnostic**: swap LLM and embedding providers via config (Gemini, OpenAI, Anthropic)
- **Idempotent pipeline**: re-run `ingest` / `embed` anytime; unchanged content costs zero API calls
- **Hybrid retrieval**: keyword (TF-IDF + CJK bigrams) + vector (pgvector) → recency decay → graph co-occurrence boost
- **Knowledge at decision points**: Claude Code hooks surface relevant failure modes when a file is first edited and before a commit, push or deploy
- **Only what earns its keep**: v0.3.0 dropped every feature with no real usage (see [Roadmap](#roadmap))

---

## Architecture

```
raw/  (immutable source notes)
   │  mindgraph ingest   (LLM rewrite, content-hash incremental)
   ▼
wiki/<category>/<concept>.md   (one concept per page; folder = knowledge category)
   │                                   │
   │ WikiGraphEngine                   │ mindgraph embed
   ▼                                   ▼
NetworkX graph + TF-IDF         wiki.embeddings (pgvector, H2–H4 sections)
   │                              content_changed_at + category
   │                                   │
   └──────────── two_stage_hybrid ─────┘
        1. recall  keyword ∪ vector
        2. weight  kw 0.4 / vec 0.6 (intent-dependent)
        3. recency × [floor … 1] by age and category   ← after weighting, before boost
        4. graph   +0.1 if >1 neighbours co-occur in the top candidates
        5. cutoff  score ≥ 0.15
   │
   ├── REST  /api/search · /api/hybrid_search · /api/smart_search · /api/concept …
   ├── MCP   search · get_concept · list_sections · get_section · stats · code_search
   └── hooks mindgraph hook prefetch | decision-point   (Claude Code)

Python repo ── mindgraph code sync ──► code.symbols ──► code.wiki_links (name / embedding match)
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Retrieval unit | H2–H4 sections, best section per concept | Summing several sections of one page lets long hub pages crowd out precise hits |
| Recency | `factor = floor + (1−floor)·e^(−λ·age)` at ranking time | Explainable and tunable per category; keeps the vector space clean |
| Two timestamps | `updated_at` (liveness) ≠ `content_changed_at` (freshness) | If one column holds both, every page is 0 days old and recency silently does nothing |
| Lessons never decay | failure-mode / decision folders use λ = 0 | A lesson doesn't expire because it is old |
| Evaluation | rank-based hit rate (target in top 3) | Recency rescales scores, so a fixed score threshold would report fake regressions |
| Code coverage | counts `embedding_match` links only | Name matching produces collision noise that inflates coverage |
| Write paths | `ingest`, `embed` and `code sync` are CLI-only | The REST API is unauthenticated, so it stays read-only |

---

## Package Structure

```
mindgraph/
├── config.py          # mindgraph.yml → MindGraphConfig (incl. recency / recall sections)
├── providers/         # LLMProvider / EmbeddingProvider (Gemini, OpenAI, Anthropic)
├── adapters/          # DataSourceAdapter + FileSystemAdapter
├── ingest.py          # raw/ → wiki/ LLM rewrite pipeline
├── graph.py           # WikiGraphEngine (NetworkX + TF-IDF + section navigation)
├── store.py           # EmbeddingStore (pgvector), iter_sections()
├── recency.py         # RecencyWeighter — time decay shared by every retrieval path
├── retrievers.py      # KeywordRetriever, VectorRetriever, two_stage_hybrid()
├── router.py          # IntentRouter (regex intent + LLM query expansion)
├── search.py          # SmartSearchService (intent routing + hybrid)
├── evaluate.py        # rank-based hit-rate evaluation
├── recall.py          # decision-point recall for Claude Code hooks
├── code/              # Python AST indexer, code↔wiki linker, coverage
├── server/            # REST (FastAPI) + MCP (FastMCP), code routes
└── cli.py
```

---

## Quick Start

```bash
pip install -e .
cp mindgraph.yml.example mindgraph.yml     # edit providers / database
export MINDGRAPH_DB_PASSWORD=...           # never commit secrets

cp ~/my-notes/*.md raw/notes/
mindgraph ingest                           # raw/ → wiki/
mindgraph embed --prune                    # wiki/ → pgvector (sections, freshness, category)
mindgraph search "your query" --hybrid
mindgraph eval                             # can the KB find its own concepts?
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `mindgraph ingest` | Run adapters, then LLM-rewrite raw/ into wiki/ |
| `mindgraph embed [--prune]` | Embed wiki sections incrementally; `--prune` sweeps deleted sections |
| `mindgraph search QUERY [--hybrid]` | Keyword search, or intent-aware hybrid with recency |
| `mindgraph check [--strict]` | Report orphans, broken wikilinks and missing frontmatter |
| `mindgraph stats` | Show graph statistics |
| `mindgraph eval [--sample 20] [--top-n 3] [--save]` | Rank-based retrieval hit rate; `--save` writes `wiki.eval_history` |
| `mindgraph code sync REPO_PATH [--repo NAME]` | Index a Python repo, then rebuild its code↔wiki links |
| `mindgraph code status REPO` | Compare the indexed git HEAD with the current HEAD; exits 1 when stale |
| `mindgraph code coverage REPO` | Semantic coverage of public functions and methods |
| `mindgraph code search QUERY` | Fuzzy symbol search with linked wiki concepts |
| `mindgraph hook prefetch` / `hook decision-point` | Claude Code hook entry points (read hook JSON on stdin) |
| `mindgraph serve` / `mindgraph mcp` | Start the REST API (binds 127.0.0.1) / the MCP server (stdio) |

### Recency

```yaml
recency:
  enabled: true
  base_lambda: 0.003            # per day; ~231-day half-life above the floor
  floor: 0.5                    # old but relevant knowledge never drops below half weight
  category_ratio:               # multiplier on base_lambda, keyed by wiki folder name
    failure-modes: 0            # lessons don't expire
    decisions: 0
    concepts: 0.33
    news: 1.67
  evergreen: [core-principles]  # concepts that never decay
```

Each result explains its score in `retriever_breakdown`: `pre_recency_score`, `age_days`, `decay_lambda` and `recency_factor`. Upgrading from v0.2: the next `embed` run adds the new columns in place. Old rows get `content_changed_at` from the file's mtime.

### Code graph

```bash
mindgraph embed                                   # 1. wiki vectors must be current
mindgraph code sync ./my-service --repo my-svc    # 2. index + link (always together)
mindgraph code coverage my-svc
```

Re-indexing regenerates symbol ids, which drops the existing links, so `sync` always re-links right after it indexes. `embedding_match` needs docstrings that describe **what the code does in business terms** (more than 50 characters). Placeholder docstrings won't reach the 0.72 similarity threshold.

---

## Claude Code Hooks

Retrieval that only runs when you ask a question misses mistakes made mid-task. Register the hooks in `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "mindgraph hook prefetch"}]}],
    "PostToolUse": [{"matcher": "Edit|Write",
                     "hooks": [{"type": "command", "command": "mindgraph hook decision-point"}]}],
    "PreToolUse": [{"matcher": "Bash",
                    "hooks": [{"type": "command", "command": "mindgraph hook decision-point"}]}]
  }
}
```

| Moment | Behaviour |
|--------|-----------|
| Each prompt | Injects up to 6 related pages (score ≥ 0.55) as `<wiki-context>` |
| First edit of a code file | Injects related **failure-mode** pages only |
| `git commit` / `git push` / deploy | Blocks once with related failure modes; resend to proceed (30-min cooldown per kind) |

Pages are de-duplicated per session. If the server is down, the hooks stay silent and never block. They need `mindgraph serve` running, and they read the `recall:` section of `mindgraph.yml` (`api_url`, `failure_categories`, thresholds).

---

## Server: REST API + MCP

| Endpoint | Purpose |
|----------|---------|
| `GET /health` · `/api/stats` | liveness; stats plus structural health (broken links, missing frontmatter) |
| `GET /api/search?q=` | keyword search |
| `GET /api/hybrid_search?q=` | hybrid + recency, no intent routing (used by hooks) |
| `GET /api/smart_search?q=` | intent-routed hybrid with query expansion |
| `GET /api/concept/{name}` · `/sections` · `/section/{id}` | page and section reads |
| `GET /api/code/search` · `/api/code/wiki-links` · `/api/code/coverage/{repo}` · `/api/code/status/{repo}` | code graph (read-only) |

MCP tools: `search` (intent-routed hybrid), `get_concept`, `list_sections`, `get_section`, `stats`, `code_search`. REST and MCP share one `SmartSearchService`, so their ranking rules can't drift apart.

> ⚠️ **Security**: the API has **no authentication**. `serve` binds `127.0.0.1` by default; expose it only behind an auth proxy. `cors_origins` defaults to `[]`.

---

## Provider Support

| Role | `provider` | Models |
|------|-----------|--------|
| LLM | `anthropic` / `openai` / `gemini` | any chat model of that provider |
| Embedding | `gemini` | gemini-embedding-001 (768-d) |
| Embedding | `openai` | text-embedding-3-small (1536-d) / -large (3072-d) |

The embedding `model_version` string is derived once from config. The writer (`embed`) and every reader (vector retrieval, recency, code linker) use that same value.

## Adapter System

```yaml
adapters:
  - type: filesystem
    source: ./raw/notes
    target: raw/notes
  - type: custom
    module: myproject.adapters.NotionAdapter   # subclass DataSourceAdapter.fetch()
```

## Graph Model

| Confidence | Source |
|-----------|--------|
| `EXTRACTED` | `[[wikilinks]]` in content, `related:` in frontmatter |
| `INFERRED` | a non-generic tag shared by 2–16 pages (more than 16 = too broad, it would turn into a hub) |

The graph is used for ranking (co-occurrence boost) and for neighbour lists. It does not generate answers.

---

## Docker

```bash
docker compose up -d     # mindgraph + pgvector/pgvector:pg16
```

## Development

```bash
pip install -e .
pytest -q                                   # unit tests
MINDGRAPH_TEST_DSN="host=... dbname=mindgraph_test ..." pytest -q   # + real-Postgres round trip
```

---

## Roadmap

### v0.1–v0.2 ✅
Providers, adapters, idempotent ingest, graph + TF-IDF, pgvector store, REST + MCP, intent routing, incremental graph rebuild.

### v0.3.0 ✅ — aligned with the production reference implementation
- [x] Recency weighting with separate liveness and freshness timestamps, per-category λ and an evergreen list
- [x] Code graph: Python AST indexer, code↔wiki linking, semantic coverage, stale-index detection
- [x] Decision-point recall hooks for Claude Code
- [x] Rank-based hit-rate evaluation
- [x] `/api/hybrid_search`; results carry `category`; stats report structural health
- [x] **Removed** (0–5 calls in months of production use): communities, god nodes, shortest path, related, timeline, text-mention `coverage`
- [ ] Not ported on purpose: weekly automatic λ calibration (only ever run once in production, so its benefit is unproven)

## License

MIT
