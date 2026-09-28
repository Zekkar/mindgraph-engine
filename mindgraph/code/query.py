"""MindGraph 程式碼圖層的唯讀查詢：symbol 搜尋、wiki 連結檢視、覆蓋率與索引狀態。

只提供實際有使用量的查詢；呼叫圖影響分析（impact / trace）刻意不提供，
呼叫邊仍由索引器寫入 code.call_edges，保留未來擴充空間。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from mindgraph.code.indexer import read_git_head
from mindgraph.code.schema import LINK_EMBEDDING_MATCH

_MAX_LIMIT = 500


def _escape_like(text: str) -> str:
    """跳脫 LIKE/ILIKE 的萬用字元，讓使用者輸入的 % 與 _ 以字面比對。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _clamp(limit: int) -> int:
    return max(1, min(int(limit), _MAX_LIMIT))


def code_search(conn, query: str, repo: Optional[str] = None, limit: int = 20) -> dict:
    """在 MindGraph 程式碼圖中模糊搜尋 symbol，供「這段功能在哪裡實作」類查詢使用。

    以不分大小寫的子字串比對 short_name、qualified_name 與 docstring，依命中位置給分：
    短名完全相同 1.0 > 短名前綴 0.8 > 短名包含 0.6 > 完整名稱包含 0.4 > 僅 docstring
    包含 0.2；同分再依短名長度（越短越貼近）與名稱排序。不需要 pg_trgm 擴充。
    可用 repo 限定範圍。回傳 {query, repo, results: [{qualified_name, short_name,
    symbol_type, file_path, lineno, docstring, repo, git_head, score}]}。
    """
    q = (query or "").strip()
    if not q:
        return {"query": query, "repo": repo, "results": []}
    esc = _escape_like(q)
    contains, prefix = f"%{esc}%", f"{esc}%"
    rows = conn.execute(
        """
        SELECT qualified_name, short_name, symbol_type, file_path, lineno,
               docstring, repo, git_head,
               CASE
                 WHEN lower(short_name) = lower(%(q)s)            THEN 1.0
                 WHEN short_name ILIKE %(prefix)s ESCAPE '\\'     THEN 0.8
                 WHEN short_name ILIKE %(contains)s ESCAPE '\\'   THEN 0.6
                 WHEN qualified_name ILIKE %(contains)s ESCAPE '\\' THEN 0.4
                 ELSE 0.2
               END AS score
        FROM code.symbols
        WHERE (%(repo)s::text IS NULL OR repo = %(repo)s)
          AND (short_name ILIKE %(contains)s ESCAPE '\\'
               OR qualified_name ILIKE %(contains)s ESCAPE '\\'
               OR docstring ILIKE %(contains)s ESCAPE '\\')
        ORDER BY score DESC, length(short_name), qualified_name
        LIMIT %(limit)s
        """,
        {"q": q, "prefix": prefix, "contains": contains, "repo": repo, "limit": _clamp(limit)},
    ).fetchall()
    keys = ("qualified_name", "short_name", "symbol_type", "file_path", "lineno",
            "docstring", "repo", "git_head", "score")
    results = [dict(zip(keys, r)) for r in rows]
    for r in results:
        r["score"] = float(r["score"])
    return {"query": query, "repo": repo, "results": results}


def wiki_links(
    conn,
    symbol: Optional[str] = None,
    concept: Optional[str] = None,
    link_type: Optional[str] = None,
    repo: Optional[str] = None,
    limit: int = 200,
) -> dict:
    """列出 MindGraph 已建立的程式碼 ↔ wiki 概念連結，依信心分數降序。

    篩選條件皆可省略並可組合：
    - symbol：比對 qualified_name（完全相同）或 short_name（完全相同）
    - concept：wiki 概念名稱（完全相同），用於「這個概念由哪些程式碼實作」
    - link_type：`name_match` 或 `embedding_match`
    - repo：限定 repo
    回傳 {count, links: [{qualified_name, short_name, symbol_type, file_path, repo,
    wiki_concept_name, link_type, confidence_score}]}。
    """
    rows = conn.execute(
        """
        SELECT s.qualified_name, s.short_name, s.symbol_type, s.file_path, s.repo,
               l.wiki_concept_name, l.link_type, l.confidence_score
        FROM code.wiki_links l
        JOIN code.symbols s ON s.id = l.symbol_id
        WHERE (%(symbol)s::text IS NULL
               OR s.qualified_name = %(symbol)s OR s.short_name = %(symbol)s)
          AND (%(concept)s::text IS NULL OR l.wiki_concept_name = %(concept)s)
          AND (%(link_type)s::text IS NULL OR l.link_type = %(link_type)s)
          AND (%(repo)s::text IS NULL OR s.repo = %(repo)s)
        ORDER BY l.confidence_score DESC, s.qualified_name, l.wiki_concept_name
        LIMIT %(limit)s
        """,
        {"symbol": symbol, "concept": concept, "link_type": link_type,
         "repo": repo, "limit": _clamp(limit)},
    ).fetchall()
    keys = ("qualified_name", "short_name", "symbol_type", "file_path", "repo",
            "wiki_concept_name", "link_type", "confidence_score")
    links = [dict(zip(keys, r)) for r in rows]
    for link in links:
        link["confidence_score"] = float(link["confidence_score"])
    return {"count": len(links), "links": links}


def coverage(conn, repo: str, exclude_paths: Iterable[str] = ()) -> dict:
    """計算某個 repo 的程式碼 ↔ wiki 語意覆蓋率，衡量程式碼有多少已對上知識概念。

    口徑（不變式 I-9）：
    - 分子：至少有一筆 `embedding_match` 連結的 public symbol 數。name_match 只是字面
      撞名，不具語意鑑別力，**不計入**覆蓋率；其數量另列於 name_match_symbols 供參考。
    - 分母：public function + method（short_name 不以 `_` 開頭，dunder 亦排除）。
    - 以 EXISTS 在 symbol 層級計數，避免一對多 JOIN 把「連結數」膨脹成「symbol 數」。
    - exclude_paths：repo 相對路徑前綴（如 `tests/`、`vendor/`），分子分母同步排除，
      用於剔除測試或第三方程式碼等已知污染源。
    回傳 {repo, public_symbols, embedding_linked, name_match_symbols, coverage}；
    coverage 為 0–1 的比例，分母為 0 時為 None。
    """
    prefixes = [f"{_escape_like(p)}%" for p in exclude_paths]
    row = conn.execute(
        """
        WITH pub AS (
            SELECT id FROM code.symbols
            WHERE repo = %(repo)s
              AND symbol_type IN ('function', 'method')
              AND left(short_name, 1) <> '_'
              AND NOT (file_path ILIKE ANY(%(prefixes)s::text[]))
        )
        SELECT
          (SELECT count(*) FROM pub),
          (SELECT count(*) FROM pub p WHERE EXISTS (
              SELECT 1 FROM code.wiki_links l
              WHERE l.symbol_id = p.id AND l.link_type = %(emb)s)),
          (SELECT count(*) FROM pub p WHERE EXISTS (
              SELECT 1 FROM code.wiki_links l
              WHERE l.symbol_id = p.id AND l.link_type <> %(emb)s))
        """,
        {"repo": repo, "prefixes": prefixes, "emb": LINK_EMBEDDING_MATCH},
    ).fetchone()
    total, emb, name = (int(x) for x in row)
    return {
        "repo": repo,
        "public_symbols": total,
        "embedding_linked": emb,
        "name_match_symbols": name,
        "coverage": (emb / total) if total else None,
    }


def index_status(conn, repo: str, repo_path=None) -> dict:
    """回報某個 repo 的程式碼索引狀態，並判斷索引是否已落後 repo 目前的 git HEAD（不變式 I-7）。

    讀取 code.index_state 的 git_head / indexed_at / 各類筆數，再讀取 repo 目前 HEAD
    （repo_path 省略時使用索引當下記錄的路徑）比對：
    - stale=True：兩個 HEAD 都可取得且不同 → 程式碼已前進、索引過期，應重跑索引與連結
    - stale=False：HEAD 相同
    - stale=None：任一方 HEAD 無法取得（非 git repo、路徑不在本機等），無法判定
    從未索引過時回傳 {repo, indexed: False}。
    """
    row = conn.execute(
        """SELECT repo_path, git_head, indexed_at, symbol_count, call_edge_count,
                  import_edge_count, error_count
           FROM code.index_state WHERE repo = %s""",
        (repo,),
    ).fetchone()
    if row is None:
        return {"repo": repo, "indexed": False}
    stored_path, git_head, indexed_at, n_sym, n_call, n_imp, n_err = row
    path = repo_path if repo_path is not None else stored_path
    current = read_git_head(Path(path)) if path else None
    stale = None if (git_head is None or current is None) else (git_head != current)
    return {
        "repo": repo,
        "indexed": True,
        "repo_path": str(path) if path else None,
        "git_head": git_head,
        "current_head": current,
        "stale": stale,
        "indexed_at": indexed_at.isoformat() if indexed_at else None,
        "symbol_count": n_sym,
        "call_edge_count": n_call,
        "import_edge_count": n_imp,
        "error_count": n_err,
    }
