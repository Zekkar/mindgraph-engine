"""
MindGraph REST API（FastAPI）

把 WikiGraphEngine 的查詢能力與 SmartSearchService 的意圖感知混合檢索，
以 HTTP endpoint 對外暴露，供任何服務直接呼叫（不需 MCP client）。
端點命名對齊既有 trading-wiki API 慣例（/api/search、/api/smart_search、/api/related 等）。
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


def create_app(engine, smart_service=None, cors_origins: list[str] | None = None) -> FastAPI:
    """以已建好的 engine（與可選的 smart_service）組出 FastAPI app；可注入物件以利測試。"""
    app = FastAPI(title="MindGraph REST API", version="0.2.0")
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "pages": engine.get_stats()["total_pages"]}

    @app.get("/api/stats")
    def stats():
        return engine.get_stats()

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), limit: int = 10):
        return {"query": q, "results": engine.search(q, limit)}

    @app.get("/api/smart_search")
    def smart_search(q: str = Query(..., min_length=1), limit: int = 10):
        if smart_service is None:
            from mindgraph.search import keyword_to_scored_dicts
            return {
                "query": q,
                "results": keyword_to_scored_dicts(engine.search(q, limit)),
                "degradation": "smart_search_unavailable",
            }
        return smart_service.search(q, limit)

    @app.get("/api/concept/{name}")
    def concept(name: str):
        c = engine.get_concept(name)
        if not c:
            raise HTTPException(status_code=404, detail=f"concept not found: {name}")
        return c

    @app.get("/api/concept/{name}/sections")
    def sections(name: str):
        return engine.get_sections(name)

    @app.get("/api/concept/{name}/section/{section_id}")
    def section(name: str, section_id: str):
        return engine.get_section(name, section_id)

    @app.get("/api/related/{name}")
    def related(name: str, depth: int = 2):
        return engine.get_related(name, depth)

    @app.get("/api/communities")
    def communities():
        return {"communities": engine.get_communities()}

    @app.get("/api/god-nodes")
    def god_nodes(limit: int = 5):
        return {"god_nodes": engine.get_god_nodes(limit)}

    @app.get("/api/path")
    def path(source: str = Query(...), target: str = Query(...)):
        return engine.shortest_path(source, target)

    return app


def build_app(base_dir: str = ".", config_path: str = "mindgraph.yml") -> FastAPI:
    """CLI `serve` 用：建好 runtime 後組出 app（含 CORS 設定）。"""
    from mindgraph.server import build_runtime

    cfg, engine, smart = build_runtime(base_dir, config_path)
    return create_app(engine, smart, cors_origins=cfg.server.cors_origins)
