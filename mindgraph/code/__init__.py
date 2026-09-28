"""MindGraph 程式碼圖層：把程式碼 repo 索引成 symbol 圖，並連結到 wiki 知識概念。

典型流程（同一個 psycopg 連線）：
    index_repository(conn, "/path/to/repo", "my-repo")        # 1. 索引，記錄 git HEAD
    link_wiki(conn, "my-repo", provider, model_version)       # 2. 重建 code↔wiki 連結
    coverage(conn, "my-repo")                                  # 3. 語意覆蓋率
    index_status(conn, "my-repo")                              # 偵測索引是否過期
"""

from mindgraph.code.indexer import (
    IndexResult,
    SymbolInfo,
    extract_repository,
    index_repository,
    persist_index,
    read_git_head,
)
from mindgraph.code.linker import link_wiki
from mindgraph.code.query import code_search, coverage, index_status, wiki_links
from mindgraph.code.schema import ensure_code_schema

__all__ = [
    "IndexResult",
    "SymbolInfo",
    "code_search",
    "coverage",
    "ensure_code_schema",
    "extract_repository",
    "index_repository",
    "index_status",
    "link_wiki",
    "persist_index",
    "read_git_head",
    "wiki_links",
]
