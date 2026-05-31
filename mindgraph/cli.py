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


@app.command()
def coverage(
    codebase: str = typer.Argument(..., help="Path to codebase root to scan"),
    base_dir: str = typer.Option(".", help="MindGraph knowledge base root (contains wiki/)"),
    min_mentions: int = typer.Option(1, help="Min wiki mentions to count as covered"),
    exclude: str = typer.Option(
        "__init__,test,conftest,migration,setup,manage",
        help="Comma-separated stems to exclude",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    service_level: bool = typer.Option(
        False, "--service-level",
        help="Scan top-level directories as service units instead of individual files",
    ),
):
    """Measure how many codebase modules are covered by wiki knowledge.

    Coverage = modules with at least --min-mentions wiki page references
             / total modules scanned.

    Example:
        mindgraph coverage ./IBAPI/backend --base-dir .
        mindgraph coverage ./ShioajiPy --service-level --base-dir .
    """
    import json as json_mod
    import re

    wiki_root = Path(base_dir) / "wiki"
    code_root = Path(codebase)

    if not code_root.exists():
        typer.echo(f"ERROR: codebase path not found: {code_root}", err=True)
        raise typer.Exit(1)
    if not wiki_root.exists():
        typer.echo(f"ERROR: wiki/ not found at {wiki_root}", err=True)
        raise typer.Exit(1)

    exclude_stems = {s.strip() for s in exclude.split(",")}

    # ── Collect wiki text corpus ──
    wiki_corpus = ""
    for md in wiki_root.rglob("*.md"):
        try:
            wiki_corpus += md.read_text(encoding="utf-8", errors="ignore") + "\n"
        except Exception:
            pass

    # ── Collect modules to scan ──
    if service_level:
        # Top-level directories as service units
        modules = [
            (d.name, d.name)
            for d in sorted(code_root.iterdir())
            if d.is_dir() and not d.name.startswith((".", "_", "__"))
        ]
    else:
        # Individual .py files
        modules = []
        for py in sorted(code_root.rglob("*.py")):
            if "__pycache__" in str(py) or "__pycache__" in py.parts:
                continue
            stem = py.stem
            if stem in exclude_stems:
                continue
            rel = str(py.relative_to(code_root))
            modules.append((stem, rel))

    # ── Score each module ──
    covered = []
    uncovered = []
    for stem, label in modules:
        pattern = re.compile(re.escape(stem), re.IGNORECASE)
        count = len(pattern.findall(wiki_corpus))
        entry = {"module": label, "stem": stem, "mentions": count}
        if count >= min_mentions:
            covered.append(entry)
        else:
            uncovered.append(entry)

    total = len(modules)
    cov_pct = round(len(covered) / total * 100, 1) if total else 0.0

    result = {
        "codebase": str(code_root),
        "wiki": str(wiki_root),
        "total_modules": total,
        "covered": len(covered),
        "uncovered": len(uncovered),
        "coverage_pct": cov_pct,
        "min_mentions": min_mentions,
        "uncovered_modules": [u["module"] for u in uncovered],
        "covered_modules": [c["module"] for c in covered],
    }

    if json_output:
        typer.echo(json_mod.dumps(result, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"Coverage: {len(covered)}/{total} ({cov_pct}%)")
        typer.echo(f"Wiki: {wiki_root}")
        typer.echo(f"Codebase: {code_root}")
        if uncovered:
            typer.echo(f"\nUncovered ({len(uncovered)}):")
            for u in sorted(uncovered, key=lambda x: x["module"]):
                typer.echo(f"  ✗ {u['module']}")
        else:
            typer.echo("\nAll modules covered!")


if __name__ == "__main__":
    app()
