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

from mindgraph.graph import category_from_path

logger = logging.getLogger("mindgraph.store")

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")


def model_version(provider: str, model: str, dimension: int) -> str:
    """產生 embedding 的 model_version 識別字串。

    embed 寫入與 VectorRetriever 查詢必須用同一個 model_version，否則查詢命不中任何向量。
    格式：{provider}-{model}-d{dimension}，例如 gemini-gemini-embedding-001-d768。
    """
    return f"{provider}-{model}-d{dimension}"


def build_dsn(db) -> str:
    """從 DatabaseConfig 安全組出 libpq conninfo。

    用 psycopg 的 make_conninfo 取代 f-string 拼接：密碼含空白/引號/反斜線時不會損壞 DSN，
    也避免把密碼塞進可能被 log 的字串。密碼從 env var 讀取，缺省則不帶 password 欄位。
    """
    import os
    from psycopg.conninfo import make_conninfo

    pw = os.environ.get(db.password_env, "")
    params = {"host": db.host, "port": db.port, "dbname": db.name, "user": db.user}
    if pw:
        params["password"] = pw
    return make_conninfo(**params)


@dataclass(slots=True)
class Section:
    """wiki 概念頁的一個 H2–H4 章節（向量檢索的最小單位）。

    category 是該頁所在的 wiki 資料夾名稱（＝知識種類），供時間遞減依種類決定衰減速度。
    """
    concept_name: str
    section_id: str
    chunk_text: str
    category: str | None = None
    source_mtime: float | None = None  # 來源檔 mtime（epoch 秒），只用來補舊列的 content_changed_at

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.chunk_text.encode("utf-8")).hexdigest()


def iter_sections(wiki_root: Path) -> Iterator[Section]:
    """逐頁把 wiki 切成 H2–H4 章節（無標題的頁整頁一塊，section_id='__page__'），
    供 embed 增量寫入向量庫；跳過 index.md / log.md。"""
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
        category = category_from_path(md_path, wiki_root)
        mtime = md_path.stat().st_mtime
        lines = content.split("\n")
        markers: list[tuple[int, int, str]] = []
        for i, line in enumerate(lines):
            m = _HEADING_RE.match(line)
            if m:
                markers.append((i, len(m.group(1)), m.group(2).strip()))

        if not markers:
            text = content.strip()
            if text:
                yield Section(concept_name, "__page__", text, category, mtime)
            continue

        for idx, (lineno, level, heading_text) in enumerate(markers):
            end = len(lines)
            for next_lineno, next_level, _ in markers[idx + 1:]:
                if next_level <= level:
                    end = next_lineno
                    break
            chunk_text = "\n".join(lines[lineno:end]).strip()
            if chunk_text:
                yield Section(concept_name, heading_text, chunk_text, category, mtime)


class EmbeddingStore:
    """wiki 章節向量庫（pgvector）的寫入端：供 `mindgraph embed` 增量同步 wiki/ 章節。

    兩個時間欄位語意嚴格分開（不變式 I-1）：
    - updated_at：存活標記，每次同步都刷新，只給 mark-and-sweep 清孤兒用
    - content_changed_at：內容最後改變時間，只在 content_hash 真的變了才更新，給時間遞減算年齡
    若混用成一個欄位，所有頁年齡永遠是 0 天，時間遞減會靜默失效而且不報錯。
    """

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
        """建立（或就地升級）wiki.embeddings 與 wiki.eval_history；冪等，可重複執行。"""
        dimension = int(dimension)  # DDL cannot parameterize vector(dim); keep SQL int-only
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
                    content_changed_at TIMESTAMPTZ,
                    category TEXT,
                    UNIQUE(concept_name, section_id, model_version)
                )
            """)
            # v0.2.x 既有表就地升級
            cur.execute("ALTER TABLE wiki.embeddings ADD COLUMN IF NOT EXISTS content_changed_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE wiki.embeddings ADD COLUMN IF NOT EXISTS category TEXT")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wiki.eval_history (
                    eval_date DATE PRIMARY KEY,
                    hit_rate DOUBLE PRECISION NOT NULL,
                    total_q INTEGER NOT NULL,
                    hits INTEGER NOT NULL,
                    top_n INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
        self.conn.commit()

    def get_existing_hash(self, concept: str, section_id: str, model: str) -> str | None:
        """查某章節目前存的內容指紋；不存在回 None（代表要新增）。"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM wiki.embeddings "
                "WHERE concept_name=%s AND section_id=%s AND model_version=%s",
                (concept, section_id, model),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def touch(self, concept: str, section_id: str, model: str, ingest_ts,
              category: str | None = None, changed_hint=None):
        """內容未變的章節：只刷新存活標記 updated_at，順手補寫 category。

        content_changed_at 刻意不動；唯一例外是它還是 NULL（v0.2 升級上來的舊列）時，
        用 changed_hint（通常是檔案 mtime）補一次，讓舊資料也能參與時間遞減。
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE wiki.embeddings SET updated_at=%s, "
                "category=COALESCE(%s, category), "
                "content_changed_at=COALESCE(content_changed_at, %s) "
                "WHERE concept_name=%s AND section_id=%s AND model_version=%s",
                (ingest_ts, category, changed_hint, concept, section_id, model),
            )

    def upsert(self, section: Section, embedding: list[float], model: str, ingest_ts):
        """寫入新章節或更新已變動章節的向量；content_changed_at 只在指紋真的改變時更新。"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wiki.embeddings
                  (concept_name, section_id, chunk_text, content_hash,
                   model_version, embedding, updated_at, category, content_changed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (concept_name, section_id, model_version) DO UPDATE
                SET chunk_text   = EXCLUDED.chunk_text,
                    embedding    = EXCLUDED.embedding,
                    updated_at   = EXCLUDED.updated_at,
                    category     = EXCLUDED.category,
                    content_changed_at = CASE
                        WHEN wiki.embeddings.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                        THEN EXCLUDED.content_changed_at
                        ELSE COALESCE(wiki.embeddings.content_changed_at,
                                      EXCLUDED.content_changed_at)
                    END,
                    content_hash = EXCLUDED.content_hash
                """,
                (section.concept_name, section.section_id, section.chunk_text,
                 section.content_hash, model, embedding, ingest_ts,
                 section.category, ingest_ts),
            )

    def mark_sweep_orphans(self, model: str, ingest_ts) -> int:
        """刪除本次同步沒刷新到存活標記的列（章節已不存在），回傳刪除筆數。"""
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
