"""程式碼圖查詢測試：純函式與 mock 連線；需要真 PostgreSQL 的整合測試以 MINDGRAPH_TEST_DSN 啟用。"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from mindgraph.code.query import _escape_like, code_search, index_status


def test_escape_like():
    assert _escape_like("a%b_c\\d") == "a\\%b\\_c\\\\d"


def test_code_search_empty_query_skips_db():
    conn = MagicMock()
    assert code_search(conn, "   ")["results"] == []
    conn.execute.assert_not_called()


def _status_conn(row):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    return conn


def test_index_status_never_indexed():
    assert index_status(_status_conn(None), "demo") == {"repo": "demo", "indexed": False}


def test_index_status_stale_detection(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("f" * 40, encoding="utf-8")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    fresh = index_status(_status_conn((str(tmp_path), "f" * 40, ts, 10, 2, 3, 0)), "demo")
    assert fresh["stale"] is False and fresh["current_head"] == "f" * 40
    assert fresh["indexed_at"] == ts.isoformat()

    stale = index_status(_status_conn((str(tmp_path), "e" * 40, ts, 10, 2, 3, 0)), "demo")
    assert stale["stale"] is True

    unknown = index_status(_status_conn((str(tmp_path / "missing"), "e" * 40, ts, 1, 0, 0, 0)), "demo")
    assert unknown["stale"] is None


# ---------------------------------------------------------------------------
# 真資料庫整合測試
# ---------------------------------------------------------------------------

DSN = os.environ.get("MINDGRAPH_TEST_DSN")


@pytest.mark.skipif(not DSN, reason="需要 MINDGRAPH_TEST_DSN 指向測試用 PostgreSQL")
def test_index_link_query_roundtrip(tmp_path):
    import psycopg

    from mindgraph.code import (
        code_search as search, coverage, ensure_code_schema, index_repository,
        index_status as status, link_wiki, wiki_links,
    )

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "linker.py").write_text(
        'class WikiLinker:\n    """連結器。"""\n    def rebuild_links(self):\n'
        '        return self._inner()\n    def _inner(self):\n        return 1\n',
        encoding="utf-8",
    )
    repo = f"test-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN) as conn:
        try:
            ensure_code_schema(conn)
            ensure_code_schema(conn)  # 冪等
            stats = index_repository(conn, tmp_path, repo)
            assert stats["symbols"] >= 5 and stats["call_edges"] == 1

            hits = search(conn, "wikilinker", repo=repo)["results"]
            assert hits[0]["short_name"] == "WikiLinker" and hits[0]["score"] == 1.0

            st = status(conn, repo)
            assert st["indexed"] and st["stale"] is None  # tmp_path 不是 git repo

            linked = link_wiki(conn, repo, methods="name_only")
            assert linked["repo"] == repo
            wiki_links(conn, repo=repo)

            cov = coverage(conn, repo)
            assert cov["public_symbols"] == 1        # rebuild_links；_inner 與 class 不計
            assert cov["embedding_linked"] == 0      # name_match 不計入覆蓋率
            assert coverage(conn, repo, exclude_paths=["pkg/"])["public_symbols"] == 0
        finally:
            conn.rollback()
            conn.execute("DELETE FROM code.symbols WHERE repo = %s", (repo,))
            conn.execute("DELETE FROM code.index_state WHERE repo = %s", (repo,))
            conn.commit()
