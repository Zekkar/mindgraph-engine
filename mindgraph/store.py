from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import psycopg
import frontmatter
from pgvector.psycopg import register_vector

logger = logging.getLogger("mindgraph.store")

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")


@dataclass(slots=True)
class Section:
    concept_name: str
    section_id: str
    chunk_text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.chunk_text.encode("utf-8")).hexdigest()


def iter_sections(wiki_root: Path) -> Iterator[Section]:
    for md_path in sorted(wiki_root.rglob("*.md")):
        if md_path.name in {"index.md", "log.md"}:
            continue
        try:
            post = frontmatter.load(md_path)
            content = post.content
        except Exception as e:
            logger.warning("parse failed: %s (%s)", md_path, e)
            continue

        concept_name = md_path.stem
        lines = content.split("\n")
        markers: list[tuple[int, int, str]] = []
        for i, line in enumerate(lines):
            m = _HEADING_RE.match(line)
            if m:
                markers.append((i, len(m.group(1)), m.group(2).strip()))

        if not markers:
            text = content.strip()
            if text:
                yield Section(concept_name, "__page__", text)
            continue

        for idx, (lineno, level, heading_text) in enumerate(markers):
            end = len(lines)
            for next_lineno, next_level, _ in markers[idx + 1:]:
                if next_level <= level:
                    end = next_lineno
                    break
            chunk_text = "\n".join(lines[lineno:end]).strip()
            if chunk_text:
                yield Section(concept_name, heading_text, chunk_text)


class EmbeddingStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn: psycopg.Connection | None = None

    def connect(self):
        self.conn = psycopg.connect(self.dsn, autocommit=False)
        register_vector(self.conn)
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        if self.conn is not None:
            self.conn.close()

    def ensure_schema(self, dimension: int, model_version: str):
        with self.conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS wiki")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS wiki.embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    concept_name TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    embedding vector({dimension}),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(concept_name, section_id, model_version)
                )
            """)
        self.conn.commit()

    def get_existing_hash(self, concept: str, section_id: str, model: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM wiki.embeddings "
                "WHERE concept_name=%s AND section_id=%s AND model_version=%s",
                (concept, section_id, model),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def touch(self, concept: str, section_id: str, model: str, ingest_ts):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE wiki.embeddings SET updated_at=%s "
                "WHERE concept_name=%s AND section_id=%s AND model_version=%s",
                (ingest_ts, concept, section_id, model),
            )

    def upsert(self, section: Section, embedding: list[float], model: str, ingest_ts):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wiki.embeddings
                  (concept_name, section_id, chunk_text, content_hash,
                   model_version, embedding, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (concept_name, section_id, model_version) DO UPDATE
                SET chunk_text   = EXCLUDED.chunk_text,
                    content_hash = EXCLUDED.content_hash,
                    embedding    = EXCLUDED.embedding,
                    updated_at   = EXCLUDED.updated_at
                """,
                (section.concept_name, section.section_id, section.chunk_text,
                 section.content_hash, model, embedding, ingest_ts),
            )

    def mark_sweep_orphans(self, model: str, ingest_ts) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM wiki.embeddings WHERE model_version=%s AND updated_at < %s",
                (model, ingest_ts),
            )
            return cur.rowcount

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()
