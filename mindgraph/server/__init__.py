"""MindGraph server layer — REST API + MCP server sharing one runtime."""
from __future__ import annotations

from pathlib import Path


def build_runtime(base_dir: str = ".", config_path: str = "mindgraph.yml"):
    """組裝 server 共用的執行期物件：config + 建好的 WikiGraphEngine + SmartSearchService。

    REST (`serve`) 與 MCP server 都呼叫這個，確保兩者背後是同一套圖引擎與降級後的混合搜尋。
    回傳 (cfg, engine, smart_search)。
    """
    from mindgraph.config import MindGraphConfig
    from mindgraph.graph import WikiGraphEngine
    from mindgraph.search import build_smart_search

    cfg = MindGraphConfig.from_yaml(config_path)
    engine = WikiGraphEngine(str(Path(base_dir) / "wiki"))
    engine.build()
    smart = build_smart_search(engine, cfg)
    return cfg, engine, smart
