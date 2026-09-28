"""
程式碼圖的唯讀 REST 端點與 MCP payload

只開放查詢：code_search、wiki_links、coverage、index_status。
索引與重建連結（會寫 DB、會讀任意本機路徑）只走 CLI `mindgraph code sync`，
不經過無認證的 REST，避免把檔案系統掃描能力暴露到網路上。
"""
from __future__ import annotations

from fastapi import FastAPI, Query


def _connect(dsn: str):
    import psycopg
    return psycopg.connect(dsn)


def code_search_payload(dsn: str, query: str, repo: str | None = None, limit: int = 20) -> dict:
    """模糊搜尋已索引的 symbol，並附上每個 symbol 連到的 wiki 概念；REST 與 MCP 共用。"""
    from mindgraph.code import code_search, wiki_links

    with _connect(dsn) as conn:
        res = code_search(conn, query, repo=repo, limit=limit)
        for r in res.get("results", []):
            links = wiki_links(conn, symbol=r["qualified_name"], repo=r.get("repo"), limit=5)
            r["wiki_concepts"] = [
                {"concept": l["wiki_concept_name"], "link_type": l["link_type"],
                 "score": l["confidence_score"]}
                for l in links.get("links", [])
            ]
    return res


def register_code_routes(app: FastAPI, dsn: str) -> None:
    """在 app 上註冊 /api/code/* 唯讀端點。"""

    @app.get("/api/code/search")
    def code_search(q: str = Query(..., min_length=1), repo: str | None = None,
                    limit: int = Query(20, ge=1, le=100)):
        """模糊搜尋 symbol（function / class / method），附帶其連結的 wiki 概念。"""
        return code_search_payload(dsn, q, repo, limit)

    @app.get("/api/code/wiki-links")
    def code_wiki_links(symbol: str | None = None, concept: str | None = None,
                        link_type: str | None = None, repo: str | None = None,
                        limit: int = Query(200, ge=1, le=1000)):
        """查 code↔wiki 連結：從 symbol 找設計文件／踩坑紀錄，或從概念找實作位置。"""
        from mindgraph.code import wiki_links
        with _connect(dsn) as conn:
            return wiki_links(conn, symbol=symbol, concept=concept, link_type=link_type,
                              repo=repo, limit=limit)

    @app.get("/api/code/coverage/{repo}")
    def code_coverage(repo: str):
        """語意覆蓋率：public function+method 中有 embedding_match 連結的比例（不含撞名雜訊）。"""
        from mindgraph.code import coverage
        with _connect(dsn) as conn:
            return coverage(conn, repo)

    @app.get("/api/code/status/{repo}")
    def code_status(repo: str):
        """索引狀態：索引時的 git HEAD、時間與筆數，用來發現「索引過期但連結仍回報成功」。"""
        from mindgraph.code import index_status
        with _connect(dsn) as conn:
            return index_status(conn, repo)
