from __future__ import annotations
import os
from pathlib import Path
import typer

app = typer.Typer(help="MindGraph — LLM-powered knowledge graph with hybrid RAG")


@app.command()
def ingest(
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
):
    """Run adapters then LLM-rewrite raw/ into wiki/."""
    from mindgraph.config import MindGraphConfig
    from mindgraph.providers import get_llm_provider
    from mindgraph.adapters import get_adapters
    from mindgraph.ingest import IngestConfig, run_ingest

    cfg = MindGraphConfig.from_yaml(config)
    base = Path(base_dir)
    ingest_cfg = IngestConfig(
        raw_root=base / "raw",
        wiki_root=base / "wiki",
        domain=cfg.domain,
        language=cfg.language,
        state_file=base / ".ingest-state.json",
    )
    stats = run_ingest(ingest_cfg, get_llm_provider(cfg), get_adapters(cfg))
    typer.echo(f"Ingest complete: {stats}")


@app.command()
def embed(
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    prune: bool = typer.Option(
        False, "--prune",
        help="Delete orphaned embeddings whose wiki sections no longer exist",
    ),
):
    """Embed wiki/ sections into pgvector (idempotent: unchanged sections skipped)."""
    from mindgraph.config import MindGraphConfig
    from mindgraph.providers import get_embedding_provider
    from mindgraph.store import EmbeddingStore, iter_sections, model_version, build_dsn
    from datetime import datetime, timezone

    cfg = MindGraphConfig.from_yaml(config)
    base = Path(base_dir)

    # dry-run only previews which sections would be embedded — no provider/key needed
    if dry_run:
        n = 0
        for section in iter_sections(base / "wiki"):
            typer.echo(f"DRY: {section.concept_name}#{section.section_id}")
            n += 1
        typer.echo(f"Embed dry-run: {n} sections")
        return

    emb = get_embedding_provider(cfg)
    model_ver = model_version(cfg.embedding.provider, cfg.embedding.model, emb.dimension)
    dsn = build_dsn(cfg.database)

    embedded = skipped = 0
    with EmbeddingStore(dsn) as store:
        store.ensure_schema(emb.dimension, model_ver)
        ingest_ts = datetime.now(timezone.utc)
        for section in iter_sections(base / "wiki"):
            existing = store.get_existing_hash(
                section.concept_name, section.section_id, model_ver
            )
            if existing == section.content_hash:
                hint = (datetime.fromtimestamp(section.source_mtime, timezone.utc)
                        if section.source_mtime else None)
                store.touch(section.concept_name, section.section_id, model_ver, ingest_ts,
                            category=section.category, changed_hint=hint)
                skipped += 1
            else:
                store.upsert(section, emb.embed(section.chunk_text), model_ver, ingest_ts)
                embedded += 1
            if (embedded + skipped) % 20 == 0:
                store.commit()
        store.commit()
        orphans = 0
        if prune:
            if embedded + skipped == 0:
                # guard: 0 sections would make EVERY stored embedding an "orphan"
                typer.echo(
                    "WARNING: 0 sections found in wiki/ — refusing to --prune (would delete "
                    "ALL embeddings for this model). Check the wiki path / config.",
                    err=True,
                )
            else:
                orphans = store.mark_sweep_orphans(model_ver, ingest_ts)
                store.commit()
    msg = f"Embed complete: {embedded} embedded, {skipped} skipped"
    typer.echo(msg + (f", {orphans} orphans pruned" if prune else ""))


@app.command()
def stats(
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
):
    """Show knowledge graph statistics."""
    import json
    from mindgraph.config import MindGraphConfig
    from mindgraph.graph import WikiGraphEngine

    cfg = MindGraphConfig.from_yaml(config)
    engine = WikiGraphEngine(str(Path(base_dir) / "wiki"))
    engine.build()
    typer.echo(json.dumps(engine.get_stats(), ensure_ascii=False, indent=2))


@app.command()
def search(
    query: str = typer.Argument(...),
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
    limit: int = typer.Option(10),
    hybrid: bool = typer.Option(
        False, "--hybrid",
        help="Intent-aware graph+vector hybrid retrieval (degrades to keyword if no DB/keys)",
    ),
):
    """Search the knowledge graph.

    Default: TF-IDF keyword search (no external deps).
    --hybrid: intent-aware graph + pgvector hybrid retrieval via SmartSearchService.
    """
    import json
    from mindgraph.config import MindGraphConfig
    from mindgraph.graph import WikiGraphEngine

    cfg = MindGraphConfig.from_yaml(config)
    engine = WikiGraphEngine(str(Path(base_dir) / "wiki"))
    engine.build()

    if not hybrid:
        typer.echo(json.dumps(engine.search(query, limit), ensure_ascii=False, indent=2))
        return

    from mindgraph.search import build_smart_search

    svc = build_smart_search(engine, cfg)
    typer.echo(json.dumps(svc.search(query, limit), ensure_ascii=False, indent=2))


@app.command()
def check(
    base_dir: str = typer.Option("."),
    strict: bool = typer.Option(False, "--strict", help="Exit 1 if broken wikilinks exist"),
):
    """Verify wiki/ integrity: stats + diagnostics (orphans, broken wikilinks, missing frontmatter)."""
    import json
    from mindgraph.graph import WikiGraphEngine

    engine = WikiGraphEngine(str(Path(base_dir) / "wiki"))
    engine.build()
    s = engine.get_stats()
    diag = engine.get_diagnostics()
    typer.echo(json.dumps({"stats": s, "diagnostics": diag}, ensure_ascii=False, indent=2))
    if s["total_pages"] == 0:
        typer.echo("WARNING: no wiki pages found", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"OK: {s['total_pages']} pages, {s['total_edges']} edges | "
        f"{diag['orphan_count']} orphans, {diag['broken_count']} broken links, "
        f"{diag['no_frontmatter_count']} no-frontmatter"
    )
    if strict and diag["broken_count"] > 0:
        typer.echo(f"STRICT FAIL: {diag['broken_count']} broken wikilinks", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
    host: str = typer.Option(
        "127.0.0.1", help="Bind address; use 0.0.0.0 to expose on LAN (NO AUTH!)"
    ),
    port: int = typer.Option(0, help="Override server.rest_port from config"),
):
    """Run the REST API server (FastAPI + uvicorn).

    The API has NO authentication and exposes the entire knowledge base read-only.
    Default binds to localhost; binding 0.0.0.0 exposes it to the whole network —
    put it behind a reverse proxy / auth layer first.
    """
    import uvicorn
    from mindgraph.config import MindGraphConfig
    from mindgraph.server.rest import build_app

    cfg = MindGraphConfig.from_yaml(config)
    application = build_app(base_dir=base_dir, config_path=config)
    bind_port = port or cfg.server.rest_port
    if host == "0.0.0.0":
        typer.echo(
            "WARNING: 0.0.0.0 exposes the UNAUTHENTICATED knowledge base to the whole "
            "network. Use --host 127.0.0.1 or front it with an auth proxy.",
            err=True,
        )
    typer.echo(f"MindGraph REST API on http://{host}:{bind_port}")
    uvicorn.run(application, host=host, port=bind_port)


@app.command()
def mcp(
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
):
    """Run the MCP server over stdio (for Claude Code integration)."""
    from mindgraph.server.mcp_server import build_mcp

    server = build_mcp(base_dir=base_dir, config_path=config)
    server.run(transport="stdio")


@app.command("eval")
def evaluate_cmd(
    config: str = typer.Option("mindgraph.yml"),
    base_dir: str = typer.Option("."),
    sample: int = typer.Option(20, help="Number of wiki concepts to sample"),
    top_n: int = typer.Option(3, help="A hit = target concept ranked within top N"),
    seed: int = typer.Option(None, help="Random seed for a reproducible sample"),
    save: bool = typer.Option(False, "--save", help="Write the result to wiki.eval_history"),
):
    """Measure retrieval hit rate: can each sampled concept be found by asking what it is?

    Rank-based (not score-threshold) so recency's score rescaling can't fake a regression.
    """
    import json
    from mindgraph.server import build_runtime
    from mindgraph.evaluate import evaluate_hit_rate, save_hit_rate
    from mindgraph.store import build_dsn

    cfg, engine, svc = build_runtime(base_dir, config)
    concepts = sorted(n for n in engine.pages if n not in {"index", "log"})
    result = evaluate_hit_rate(svc, concepts, sample=sample, top_n=top_n, seed=seed)
    result["vector_enabled"] = svc.vector_retriever is not None
    result["recency_enabled"] = svc.recency is not None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if save:
        save_hit_rate(build_dsn(cfg.database), result)
        typer.echo("saved to wiki.eval_history")


# ─── 程式碼圖 ────────────────────────────────────────────────────────────────
code_app = typer.Typer(help="Code graph: index a Python repo and link symbols to wiki concepts")
app.add_typer(code_app, name="code")


def _code_conn(config: str):
    """code 子指令共用：讀設定並開一條到程式碼圖資料庫的連線，回傳 (cfg, conn)。"""
    import psycopg
    from mindgraph.config import MindGraphConfig
    from mindgraph.store import build_dsn

    cfg = MindGraphConfig.from_yaml(config)
    return cfg, psycopg.connect(build_dsn(cfg.database))


@code_app.command("sync")
def code_sync(
    repo_path: str = typer.Argument(..., help="Path to the Python repository"),
    repo: str = typer.Option(None, help="Repo name (default: directory name)"),
    config: str = typer.Option("mindgraph.yml"),
    exclude: list[str] = typer.Option([], "--exclude", help="Extra fnmatch patterns to skip"),
    methods: str = typer.Option("all", help="all | name_only | embedding_only"),
):
    """Index the repo, then rebuild code↔wiki links — always both, in this order.

    Re-indexing regenerates symbol ids and drops existing links, so linking is never optional.
    Run `mindgraph embed` first so links see the current wiki.
    """
    import json
    from mindgraph.code import index_repository, link_wiki
    from mindgraph.store import build_dsn, model_version

    cfg, conn = _code_conn(config)
    name = repo or Path(repo_path).resolve().name
    provider = mv = recency_fn = None
    if methods != "name_only":
        from mindgraph.providers import get_embedding_provider
        provider = get_embedding_provider(cfg)
        mv = model_version(cfg.embedding.provider, cfg.embedding.model, provider.dimension)
        if cfg.recency.enabled:
            from mindgraph.recency import make_db_weighter
            recency_fn = make_db_weighter(build_dsn(cfg.database), mv, cfg.recency).factor_for
    with conn:
        idx = index_repository(conn, repo_path, name, exclude=exclude)
        links = link_wiki(conn, name, embedding_provider=provider, model_version=mv,
                          methods=methods, recency_fn=recency_fn)
    typer.echo(json.dumps({"index": idx, "links": links}, ensure_ascii=False, indent=2, default=str))
    cleared_emb = (links.get("cleared_links") or {}).get("embedding_match", 0)
    if methods == "name_only" and cleared_emb:
        typer.echo(f"WARNING: name_only cleared {cleared_emb} embedding_match links", err=True)


@code_app.command("status")
def code_status(
    repo: str = typer.Argument(...),
    config: str = typer.Option("mindgraph.yml"),
):
    """Show index freshness: indexed git HEAD vs current HEAD (stale index = silent gap)."""
    import json
    from mindgraph.code import index_status

    _cfg, conn = _code_conn(config)
    with conn:
        st = index_status(conn, repo)
    typer.echo(json.dumps(st, ensure_ascii=False, indent=2, default=str))
    if st.get("stale"):
        raise typer.Exit(1)


@code_app.command("coverage")
def code_coverage(
    repo: str = typer.Argument(...),
    config: str = typer.Option("mindgraph.yml"),
    exclude: list[str] = typer.Option([], "--exclude", help="Path patterns to drop from both sides"),
):
    """Semantic coverage: share of public functions/methods with an embedding_match wiki link."""
    import json
    from mindgraph.code import coverage

    _cfg, conn = _code_conn(config)
    with conn:
        typer.echo(json.dumps(coverage(conn, repo, exclude_paths=exclude),
                              ensure_ascii=False, indent=2, default=str))


@code_app.command("search")
def code_search_cmd(
    query: str = typer.Argument(...),
    repo: str = typer.Option(None),
    limit: int = typer.Option(20),
    config: str = typer.Option("mindgraph.yml"),
):
    """Fuzzy-search code symbols with their linked wiki concepts."""
    import json
    from mindgraph.config import MindGraphConfig
    from mindgraph.server.code_routes import code_search_payload
    from mindgraph.store import build_dsn

    cfg = MindGraphConfig.from_yaml(config)
    typer.echo(json.dumps(code_search_payload(build_dsn(cfg.database), query, repo, limit),
                          ensure_ascii=False, indent=2, default=str))


# ─── Claude Code hooks ───────────────────────────────────────────────────────
hook_app = typer.Typer(help="Claude Code hook entry points (read hook JSON on stdin)")
app.add_typer(hook_app, name="hook")


def _run_hook(config: str, handler: str) -> None:
    """共用 hook 外殼：讀 stdin JSON → 呼叫 RecallClient → 印出結果；任何例外都靜默放行。"""
    import json
    import sys
    try:
        from mindgraph.recall import RecallClient, is_decision_command
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
        # 快速放行：絕大多數 Bash 呼叫不是決策點，不必載入設定檔
        if handler != "prefetch" and payload.get("tool_name") == "Bash" and not is_decision_command(
                str((payload.get("tool_input") or {}).get("command", ""))):
            return
        from mindgraph.config import MindGraphConfig, RecallConfig
        try:
            rc = MindGraphConfig.from_yaml(config).recall
        except Exception:
            rc = RecallConfig()  # 沒有設定檔也能用預設值運作
        client = RecallClient(rc)
        if handler == "prefetch":
            out = client.prefetch(payload)
        else:
            event, tool = payload.get("hook_event_name"), payload.get("tool_name")
            if event == "PostToolUse" and tool in ("Edit", "Write"):
                out = client.on_edit(payload)
            elif event == "PreToolUse" and tool == "Bash":
                out = client.on_bash(payload)
            else:
                out = None
        if out:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stdout.write(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))
    except Exception:
        pass


@hook_app.command("prefetch")
def hook_prefetch(config: str = typer.Option("mindgraph.yml")):
    """UserPromptSubmit: inject related wiki pages as <wiki-context>."""
    _run_hook(config, "prefetch")


@hook_app.command("decision-point")
def hook_decision_point(config: str = typer.Option("mindgraph.yml")):
    """PostToolUse(Edit|Write) / PreToolUse(Bash): recall related failure modes at decision points."""
    _run_hook(config, "decision-point")


if __name__ == "__main__":
    app()
