"""MindGraph 程式碼圖層的 PostgreSQL schema 定義。

所有程式碼圖相關資料表集中在 `code` schema 下，與 wiki 知識層（`wiki` schema）
分開，但透過 `code.wiki_links.wiki_concept_name` 對應到 `wiki.embeddings.concept_name`，
形成「程式碼 symbol ↔ wiki 概念」的跨層連結。

資料表：
- code.symbols      ：索引器抽出的 module / class / function / method
- code.call_edges   ：symbol 之間的呼叫邊（目前只寫入保存，不提供查詢）
- code.import_edges ：模組層級的 import 紀錄（目前只寫入保存，不提供查詢）
- code.wiki_links   ：symbol ↔ wiki 概念連結（name_match / embedding_match）
- code.index_state  ：每個 repo 最近一次索引的 git HEAD 與時間，供偵測索引過期
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

# 連結類型常數：覆蓋率只認 embedding_match（不變式 I-9）
LINK_NAME_MATCH = "name_match"
LINK_EMBEDDING_MATCH = "embedding_match"

_DDL: tuple[str, ...] = (
    "CREATE SCHEMA IF NOT EXISTS code",
    """
    CREATE TABLE IF NOT EXISTS code.symbols (
        id              BIGSERIAL PRIMARY KEY,
        repo            TEXT NOT NULL,
        qualified_name  TEXT NOT NULL,
        short_name      TEXT NOT NULL,
        symbol_type     TEXT NOT NULL,
        language        TEXT NOT NULL DEFAULT 'python',
        file_path       TEXT NOT NULL,
        lineno          INTEGER,
        docstring       TEXT,
        decorators      JSONB NOT NULL DEFAULT '[]'::jsonb,
        git_head        TEXT,
        indexed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (repo, qualified_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_code_symbols_repo ON code.symbols (repo)",
    "CREATE INDEX IF NOT EXISTS idx_code_symbols_short_name ON code.symbols (lower(short_name))",
    """
    CREATE TABLE IF NOT EXISTS code.call_edges (
        caller_id   BIGINT NOT NULL REFERENCES code.symbols(id) ON DELETE CASCADE,
        callee_id   BIGINT NOT NULL REFERENCES code.symbols(id) ON DELETE CASCADE,
        call_type   TEXT NOT NULL DEFAULT 'direct',
        PRIMARY KEY (caller_id, callee_id, call_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_code_call_edges_callee ON code.call_edges (callee_id)",
    """
    CREATE TABLE IF NOT EXISTS code.import_edges (
        importer_id     BIGINT NOT NULL REFERENCES code.symbols(id) ON DELETE CASCADE,
        imported_module TEXT NOT NULL,
        is_type_only    BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_code_import_edges_importer ON code.import_edges (importer_id)",
    """
    CREATE TABLE IF NOT EXISTS code.wiki_links (
        symbol_id           BIGINT NOT NULL REFERENCES code.symbols(id) ON DELETE CASCADE,
        wiki_concept_name   TEXT NOT NULL,
        link_type           TEXT NOT NULL,
        confidence_score    REAL NOT NULL,
        UNIQUE (symbol_id, wiki_concept_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_code_wiki_links_concept ON code.wiki_links (wiki_concept_name)",
    """
    CREATE TABLE IF NOT EXISTS code.index_state (
        repo              TEXT PRIMARY KEY,
        repo_path         TEXT,
        git_head          TEXT,
        indexed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        symbol_count      INTEGER NOT NULL DEFAULT 0,
        call_edge_count   INTEGER NOT NULL DEFAULT 0,
        import_edge_count INTEGER NOT NULL DEFAULT 0,
        error_count       INTEGER NOT NULL DEFAULT 0
    )
    """,
)


@contextmanager
def atomic(conn) -> Iterator[None]:
    """在 MindGraph 程式碼圖層的寫入流程中提供「全有或全無」的交易邊界。

    索引寫入（刪除 repo → 批次寫入）與 wiki 連結重建（刪除 repo 連結 → 寫入）
    都必須在單一交易內完成，讀取端才不會看到半套資料。psycopg 的
    `conn.transaction()` 在 autocommit 連線上是真正的 BEGIN/COMMIT；在非
    autocommit 連線上若已處於交易中則只是 savepoint，因此離開區塊後對非
    autocommit 連線再補一次 `commit()`，確保資料確實落地。
    """
    with conn.transaction():
        yield
    if not conn.autocommit:
        conn.commit()


def ensure_code_schema(conn) -> None:
    """建立 MindGraph 程式碼圖層所需的 `code` schema 與全部資料表（冪等）。

    供索引（index_repository）與 wiki 連結（link_wiki）流程在寫入前呼叫；
    全部使用 CREATE ... IF NOT EXISTS，可重複執行而不影響既有資料。
    不依賴 pg_trgm / pgvector 以外的擴充套件（本函式本身不需要任何擴充）。
    """
    with atomic(conn):
        for stmt in _DDL:
            conn.execute(stmt)
