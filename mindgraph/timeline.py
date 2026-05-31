"""
Concept evolution timeline — track how a concept appears over time in raw/ notes.

Domain-agnostic: scans raw/ markdown for dated mentions of a concept and returns a
chronological list, so you can see how an idea evolved across your dated notes
(devdiary entries, research logs, meeting notes...). Date is resolved from the
frontmatter `date:` field, else a YYYY-MM-DD prefix in the filename, else file mtime.
"""
from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

logger = logging.getLogger("mindgraph.timeline")

_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _resolve_date(md_path: Path, meta: dict) -> str:
    d = meta.get("date")
    if d:
        return str(d)[:10]
    m = _DATE_IN_NAME.search(md_path.name)
    if m:
        return m.group(1)
    return datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).date().isoformat()


def _snippet(content: str, concept: str, width: int = 160) -> str:
    i = content.lower().find(concept.lower())
    if i < 0:
        return ""
    start = max(0, i - width // 2)
    return content[start:start + width].replace("\n", " ").strip()


def concept_timeline(raw_root, concept: str, limit: int = 50) -> dict:
    """掃描 raw/ 找出含 concept 的有日期記錄，依時間排序，呈現概念隨時間的演化軌跡。

    回傳 {concept, mentions, timeline:[{date, source, snippet}, ...]}；raw/ 不存在則 mentions=0。
    """
    raw_root = Path(raw_root)
    needle = concept.lower()
    entries: list[dict] = []
    if raw_root.exists():
        for md in raw_root.rglob("*.md"):
            try:
                post = frontmatter.load(str(md))
            except Exception as e:
                logger.warning("timeline parse failed %s: %s", md, e)
                continue
            content = post.content
            if needle not in content.lower() and needle not in md.stem.lower():
                continue
            try:
                source = str(md.relative_to(raw_root.parent))
            except ValueError:
                source = str(md)
            entries.append({
                "date": _resolve_date(md, dict(post.metadata)),
                "source": source,
                "snippet": _snippet(content, concept),
            })
    entries.sort(key=lambda e: e["date"])
    return {"concept": concept, "mentions": len(entries), "timeline": entries[:limit]}
