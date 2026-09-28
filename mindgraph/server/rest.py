"""
MindGraph REST API（FastAPI）

給 hook、腳本與其他服務直接呼叫（不需 MCP client）。只保留有實際使用量的端點
（v0.3.0 依參考實作的 transcript 統計篩選）：關鍵字搜尋、混合搜尋、意圖路由搜尋、
概念／章節讀取、健康統計、程式碼圖。社群、god nodes、最短路徑、timeline、related
使用量為 0～5 次，已移除。

所有搜尋端點與 MCP 共用同一個 SmartSearchService（同一組 retriever 與時間遞減加權器），
避免某條路徑規則不同步（不變式 I-5）。
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from mindgraph import __version__
from mindgraph.search import degraded_keyword_result as _degraded


def create_app(engine, smart_service=None, cors_origins: list[str] | None = None,
               code_dsn: str | None = None) -> FastAPI:
    """以已建好的 engine（與可選的 smart_service、程式碼圖 DSN）組出 FastAPI app；可注入物件以利測試。"""
    app = FastAPI(title="MindGraph REST API", version=__version__)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health():
        """存活檢查：回傳頁數。"""
        return {"status": "ok", "pages": engine.get_stats()["total_pages"]}

    @app.get("/api/stats")
    def stats():
        """圖譜統計＋結構健康度（斷鏈數、缺 frontmatter 頁數），用來抓靜默失效。"""
        return engine.get_health_stats()

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), limit: int = 10):
        """關鍵字（TF-IDF＋CJK bigram）搜尋；字面精準、零外部依賴，腳本最常用。"""
        return {"query": q, "results": engine.search(q, limit)}

    @app.get("/api/hybrid_search")
    def hybrid_search(q: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=50)):
        """兩階段混合檢索（關鍵字＋向量＋時間遞減＋圖共現），不做意圖路由；決策點 hook 用。"""
        if smart_service is None:
            return _degraded(engine, q, top_k, "smart_search_unavailable")
        return smart_service.hybrid(q, top_k)

    @app.get("/api/smart_search")
    def smart_search(q: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=50)):
        """意圖路由搜尋：依查詢意圖選權重，failure 意圖加權失敗模式頁，低信心時做查詢擴展。"""
        if smart_service is None:
            return _degraded(engine, q, top_k, "smart_search_unavailable")
        return smart_service.search(q, top_k)

    @app.get("/api/concept/{name}")
    def concept(name: str):
        """讀取完整概念頁（內容、frontmatter、圖鄰居）。"""
        c = engine.get_concept(name)
        if not c:
            raise HTTPException(status_code=404, detail=f"concept not found: {name}")
        return c

    @app.get("/api/concept/{name}/sections")
    def sections(name: str):
        """列出概念頁的章節大綱，先看大綱再只讀需要的段落以節省 token。"""
        return engine.get_sections(name)

    @app.get("/api/concept/{name}/section/{section_id}")
    def section(name: str, section_id: str):
        """只讀概念頁的單一章節。"""
        return engine.get_section(name, section_id)

    if code_dsn:
        from mindgraph.server.code_routes import register_code_routes
        register_code_routes(app, code_dsn)

    return app


def build_app(base_dir: str = ".", config_path: str = "mindgraph.yml") -> FastAPI:
    """CLI `serve` 用：建好 runtime 後組出 app（含 CORS 與程式碼圖路由）。"""
    from mindgraph.server import build_runtime
    from mindgraph.store import build_dsn

    cfg, engine, smart = build_runtime(base_dir, config_path)
    return create_app(engine, smart, cors_origins=cfg.server.cors_origins,
                      code_dsn=build_dsn(cfg.database))
