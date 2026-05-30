from __future__ import annotations
import hashlib
import json
import logging
import tempfile
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mindgraph.adapters.base import DataSourceAdapter
from mindgraph.providers.base import LLMProvider

logger = logging.getLogger("mindgraph.ingest")

REWRITE_PROMPT = """You are a knowledge curator. Domain: {domain}. Language: {language}.
Read the raw content and rewrite it as a structured wiki page with:
- A clear title (# heading)
- A "## Core Conclusions" section with key insights
- Additional sections as appropriate
Output ONLY the markdown.

Raw content:
{content}"""


@dataclass
class IngestConfig:
    raw_root: Path
    wiki_root: Path
    domain: str = "general"
    language: str = "en"
    state_file: Path = None

    def __post_init__(self):
        if self.state_file is None:
            self.state_file = self.raw_root.parent / ".ingest-state.json"


def _file_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


def _atomic_write(path: Path, data: dict):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_ingest(
    cfg: IngestConfig,
    llm_provider: LLMProvider,
    adapters: list[DataSourceAdapter],
    rewrite_prompt: str = REWRITE_PROMPT,
) -> dict:
    state = (
        json.loads(cfg.state_file.read_text(encoding="utf-8"))
        if cfg.state_file.exists()
        else {"version": 1, "files": {}}
    )
    stats = {"raw_written": 0, "wiki_written": 0, "skipped": 0, "errors": 0}

    # Phase A: adapters → raw/
    for adapter in adapters:
        target = cfg.raw_root / adapter.get_target_dir().removeprefix("raw/")
        target.mkdir(parents=True, exist_ok=True)
        for doc in adapter.fetch():
            (target / doc.filename).write_text(doc.content, encoding="utf-8")
            stats["raw_written"] += 1

    # Phase B+D: raw/ → LLM → wiki/
    cfg.wiki_root.mkdir(parents=True, exist_ok=True)
    for raw_file in sorted(cfg.raw_root.rglob("*.md")):
        rel = str(raw_file.relative_to(cfg.raw_root.parent))
        content = raw_file.read_text(encoding="utf-8")
        fhash = _file_hash(content)
        existing = state["files"].get(rel, {})
        if existing.get("hash") == fhash and existing.get("status") == "ingested":
            stats["skipped"] += 1
            continue
        try:
            wiki_content = llm_provider.complete(
                rewrite_prompt.format(
                    domain=cfg.domain,
                    language=cfg.language,
                    content=content[:4000],
                )
            )
            wiki_file = cfg.wiki_root / f"{raw_file.stem}.md"
            wiki_file.write_text(wiki_content, encoding="utf-8")
            stats["wiki_written"] += 1
            state["files"][rel] = {
                "hash": fhash,
                "status": "ingested",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("ingest failed %s: %s", raw_file, e)
            stats["errors"] += 1
            state["files"][rel] = {"hash": fhash, "status": "failed", "error": str(e)}

        _atomic_write(cfg.state_file, state)

    return stats
