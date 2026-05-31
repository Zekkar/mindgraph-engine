"""
MindGraph MCP server（FastMCP / stdio）

把知識圖譜查詢以 MCP tools 暴露給 Claude Code 等 MCP client：
search / smart_search / get_concept / related / communities / stats / god_nodes。
讓 Agent 在對話中直接把私有知識庫當外部記憶查詢。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def create_mcp(engine, smart_service=None) -> FastMCP:
    """以已建好的 engine（與可選的 smart_service）註冊 MCP tools；可注入物件以利測試。"""
    server = FastMCP(
        "mindgraph",
        instructions="MindGraph knowledge graph: keyword + intent-aware hybrid search, "
        "concept lookup, graph relations, communities.",
    )

    @server.tool()
    def search(query: str, limit: int = 10) -> list:
        """Keyword (TF-IDF) search over wiki pages. Returns ranked page summaries."""
        return engine.search(query, limit)

    @server.tool()
    def smart_search(query: str, limit: int = 10) -> dict:
        """Intent-aware graph + vector hybrid search. Best for ambiguous or conceptual queries."""
        if smart_service is None:
            from mindgraph.search import keyword_to_scored_dicts
            return {
                "query": query,
                "results": keyword_to_scored_dicts(engine.search(query, limit)),
                "degradation": "smart_search_unavailable",
            }
        return smart_service.search(query, limit)

    @server.tool()
    def get_concept(name: str) -> dict:
        """Fetch a full wiki page: content + metadata + graph neighbors."""
        return engine.get_concept(name) or {"error": f"not found: {name}"}

    @server.tool()
    def related(name: str, depth: int = 2) -> dict:
        """Graph-traverse concepts related to `name` up to `depth` hops."""
        return engine.get_related(name, depth)

    @server.tool()
    def communities() -> list:
        """List auto-detected concept communities (Louvain)."""
        return engine.get_communities()

    @server.tool()
    def stats() -> dict:
        """Knowledge graph statistics: pages, edges, communities, density."""
        return engine.get_stats()

    @server.tool()
    def god_nodes(limit: int = 5) -> list:
        """Most-connected hub concepts (highest degree)."""
        return engine.get_god_nodes(limit)

    return server


def build_mcp(base_dir: str = ".", config_path: str = "mindgraph.yml") -> FastMCP:
    """CLI `mcp` 用：建好 runtime 後註冊 tools。"""
    from mindgraph.server import build_runtime

    _cfg, engine, smart = build_runtime(base_dir, config_path)
    return create_mcp(engine, smart)
