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
):
    """Embed wiki/ sections into pgvector."""
    from mindgraph.config import MindGraphConfig
    from mindgraph.providers import get_embedding_provider
    from mindgraph.store import EmbeddingStore, iter_sections
    from datetime import datetime, timezone

    cfg = MindGraphConfig.from_yaml(config)
    base = Path(base_dir)
    emb = get_embedding_provider(cfg)
    db = cfg.database
    pw = os.environ.get(db.password_env, "")
    dsn = f"host={db.host} port={db.port} dbname={db.name} user={db.user}" + (
        f" password={pw}" if pw else ""
    )
    model_ver = f"{cfg.embedding.provider}-{cfg.embedding.model}"

    with EmbeddingStore(dsn) as store:
        store.ensure_schema(emb.dimension, model_ver)
        ingest_ts = datetime.now(timezone.utc)
        count = 0
        for section in iter_sections(base / "wiki"):
            if dry_run:
                typer.echo(f"DRY: {section.concept_name}#{section.section_id}")
            else:
                store.upsert(section, emb.embed(section.chunk_text), model_ver, ingest_ts)
            count += 1
            if count % 20 == 0 and not dry_run:
                store.commit()
        if not dry_run:
            store.commit()
    typer.echo(f"Embed complete: {count} sections")


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
):
    """Quick keyword search."""
    import json
    from mindgraph.config import MindGraphConfig
    from mindgraph.graph import WikiGraphEngine

    cfg = MindGraphConfig.from_yaml(config)
    engine = WikiGraphEngine(str(Path(base_dir) / "wiki"))
    engine.build()
    typer.echo(json.dumps(engine.search(query, limit), ensure_ascii=False, indent=2))


@app.command()
def check(
    base_dir: str = typer.Option("."),
):
    """Verify wiki/ integrity: parse all pages and report errors."""
    import json
    from mindgraph.graph import WikiGraphEngine

    engine = WikiGraphEngine(str(Path(base_dir) / "wiki"))
    engine.build()
    s = engine.get_stats()
    typer.echo(json.dumps(s, ensure_ascii=False, indent=2))
    if s["total_pages"] == 0:
        typer.echo("WARNING: no wiki pages found", err=True)
        raise typer.Exit(1)
    typer.echo(f"OK: {s['total_pages']} pages, {s['total_edges']} edges")


if __name__ == "__main__":
    app()
