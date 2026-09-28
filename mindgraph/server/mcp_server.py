"""
MindGraph MCP server（FastMCP / stdio）

給 Claude Code 等 AI 代理直接呼叫。工具組依參考實作的實際呼叫量篩選（v0.3.0）：
search（意圖路由混合檢索，最常用）、get_concept、list_sections / get_section（只讀一段省 token）、
stats、code_search。related / communities / god_nodes / timeline 使用量近 0，已移除。

MCP 與 REST 共用同一個 SmartSearchService，檢索規則只有一份（不變式 I-5）。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def create_mcp(engine, smart_service=None, code_dsn: str | None = None) -> FastMCP:
    """以已建好的 engine（與可選的 smart_service、程式碼圖 DSN）註冊 MCP tools；可注入物件以利測試。"""
    server = FastMCP(
        "mindgraph",
        instructions="MindGraph knowledge base: intent-aware hybrid search with recency, "
        "concept and section reads, code symbol search. Read large pages via "
        "list_sections then get_section.",
    )

    @server.tool()
    def search(query: str, limit: int = 10) -> dict:
        """Intent-aware hybrid search (keyword + vector + recency + graph co-occurrence).
        Each result carries retriever_breakdown explaining its score."""
        if smart_service is None:
            from mindgraph.search import degraded_keyword_result
            return degraded_keyword_result(engine, query, limit, "smart_search_unavailable")
        return smart_service.search(query, limit)

    @server.tool()
    def get_concept(name: str) -> dict:
        """Fetch a full wiki page: content + metadata + graph neighbors."""
        return engine.get_concept(name) or {"error": f"not found: {name}"}

    @server.tool()
    def list_sections(name: str) -> dict:
        """List a page's section outline (read this before get_section on large pages)."""
        return engine.get_sections(name) or {"error": f"not found: {name}"}

    @server.tool()
    def get_section(name: str, section_id: str) -> dict:
        """Read a single section of a page."""
        return engine.get_section(name, section_id)

    @server.tool()
    def stats() -> dict:
        """Knowledge base statistics and structural health (broken links, missing frontmatter)."""
        return engine.get_health_stats()

    if code_dsn:
        @server.tool()
        def code_search(query: str, repo: str | None = None, limit: int = 20) -> dict:
            """Fuzzy-search indexed code symbols (function/class/method) and their linked wiki concepts."""
            from mindgraph.server.code_routes import code_search_payload
            return code_search_payload(code_dsn, query, repo, limit)

    return server


def build_mcp(base_dir: str = ".", config_path: str = "mindgraph.yml") -> FastMCP:
    """CLI `mcp` 用：建好 runtime 後註冊 tools。"""
    from mindgraph.server import build_runtime
    from mindgraph.store import build_dsn

    cfg, engine, smart = build_runtime(base_dir, config_path)
    return create_mcp(engine, smart, code_dsn=build_dsn(cfg.database))
