"""
SmartSearchService — 意圖感知的混合檢索協調層

把先前各自獨立的元件接成一條可執行的搜尋管線：
WikiGraphEngine（關鍵字 + 圖 boost）+ VectorRetriever（pgvector 語意）
+ IntentRouter（Pass 1 意圖分類 / Pass 2 LLM 查詢擴展）+ two_stage_hybrid（合併排序）。

設計鐵律：任何下游元件（向量庫、LLM）不可用時靜默降級，不中斷搜尋。
向量庫掛掉 → 退化成關鍵字 + 圖；LLM 不可用 → 跳過 Pass 2 擴展。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mindgraph.router import IntentRouter
from mindgraph.retrievers import KeywordRetriever, two_stage_hybrid

logger = logging.getLogger("mindgraph.search")

# 各意圖的 keyword/vector 權重預設：location 偏關鍵字、concept/failure 偏語意
INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "location": {"keyword": 0.7, "vector": 0.3},
    "concept": {"keyword": 0.3, "vector": 0.7},
    "relation": {"keyword": 0.4, "vector": 0.6},
    "failure": {"keyword": 0.3, "vector": 0.7},
    "default": {"keyword": 0.4, "vector": 0.6},
}

DEFAULT_FAILURE_CATEGORIES: tuple[str, ...] = ("失敗模式", "failure-modes", "failures")
FAILURE_BOOST = 1.5
# substrings in a page NAME that hint the page is itself a failure/bug note
_FAILURE_NAME_HINTS: tuple[str, ...] = ("fail", "error", "timeout", "bug", "失敗", "錯誤")


@dataclass
class SmartSearchService:
    """意圖感知混合檢索服務，協調圖/向量/意圖三路並支援 LLM 查詢擴展與降級保護。"""

    graph_engine: object  # WikiGraphEngine：提供 KeywordRetriever 與 get_related 圖 boost
    vector_retriever: object | None = None  # VectorRetriever；None → 降級為關鍵字+圖
    llm_provider: object | None = None  # 供 IntentRouter Pass 2 擴展；None → 跳過
    router: IntentRouter = field(default_factory=IntentRouter)
    failure_categories: tuple[str, ...] = DEFAULT_FAILURE_CATEGORIES
    enable_expansion: bool = True

    def search(self, query: str, top_k: int = 10) -> dict:
        """執行意圖感知混合搜尋，回傳含 results / intent / 降級狀態的結構化結果。"""
        if hasattr(self.graph_engine, "ensure_fresh"):
            self.graph_engine.ensure_fresh()

        intent = self.router.classify(query)
        weights = INTENT_WEIGHTS.get(intent.intent, INTENT_WEIGHTS["default"])
        keyword_ret = KeywordRetriever(self.graph_engine)
        boost_fn = self._failure_boost_fn() if intent.intent == "failure" else None

        result = two_stage_hybrid(
            query, top_k, keyword_ret, self.vector_retriever, self.graph_engine,
            weights=weights, boost_fn=boost_fn,
        )

        top1 = result["results"][0]["score"] if result["results"] else 0.0
        if (
            self.enable_expansion
            and self.llm_provider is not None
            and self.router.needs_expansion(top1, query)
        ):
            result = self._expand_and_merge(
                query, top_k, keyword_ret, weights, result, boost_fn
            )

        result["intent"] = {"type": intent.intent, "confidence": round(intent.confidence, 3)}
        return result

    def _failure_boost_fn(self):
        """產生 boost 函式：對 failure-mode 類頁面（類別 / 標籤 / 名稱命中）乘上 FAILURE_BOOST。"""
        cats = self.failure_categories
        pages = getattr(self.graph_engine, "pages", {})

        def fn(doc) -> float:
            page = pages.get(doc.concept_name, {})
            category = page.get("category", "")
            tags = page.get("metadata", {}).get("tags", []) or []
            name = (doc.concept_name or "").lower()
            if (
                category in cats
                or any(t in cats for t in tags)
                or any(hint in name for hint in _FAILURE_NAME_HINTS)
            ):
                return FAILURE_BOOST
            return 1.0

        return fn

    def _expand_and_merge(self, query, top_k, keyword_ret, weights, base_result, boost_fn=None) -> dict:
        """Pass 2：請 LLM 生成補充查詢，逐一檢索後與原結果合併取較高分。

        必須把同一個 boost_fn 傳給子查詢，否則經擴展才找到的（例如 failure-mode）頁面會以
        未加權分數進入合併池，與已加權的 base 結果排序尺度不一致。
        """
        extra_queries = self.router.expand_query(
            query, base_result["results"], self.llm_provider
        )
        if not extra_queries:
            return base_result

        merged = {r["concept_name"]: r for r in base_result["results"]}
        for eq in extra_queries:
            try:
                sub = two_stage_hybrid(
                    eq, top_k, keyword_ret, self.vector_retriever, self.graph_engine,
                    weights=weights, boost_fn=boost_fn,
                )
            except Exception as e:  # 擴展查詢失敗不影響主結果
                logger.warning("expansion sub-query failed: %s", e)
                continue
            for r in sub["results"]:
                cn = r["concept_name"]
                if cn not in merged or r["score"] > merged[cn]["score"]:
                    merged[cn] = r

        base_result["results"] = sorted(
            merged.values(), key=lambda x: x["score"], reverse=True
        )[:top_k]
        base_result["expanded_queries"] = extra_queries
        return base_result


def keyword_to_scored_dicts(raw: list[dict]) -> list[dict]:
    """把 WikiGraphEngine.search 的關鍵字結果轉成與 SmartSearch 一致的 ScoredDoc dict 形狀。

    讓 smart_search 在降級（無 smart_service）時回傳的 results 仍含 concept_name 等欄位，
    下游解析不會因降級邊界而 KeyError。
    """
    return [
        {
            "concept_name": r.get("name", ""),
            "section_id": None,
            "chunk_text": r.get("snippet", ""),
            "score": r.get("score", 0.0),
            "retriever_breakdown": {"keyword_raw": r.get("score", 0.0)},
        }
        for r in raw
    ]


def build_smart_search(engine, cfg) -> SmartSearchService:
    """從 config 盡力組裝 SmartSearchService：金鑰/DB 缺失時自動降級（vector/LLM 設 None）。

    供 CLI `search --hybrid`、REST `serve`、MCP server 共用，集中降級邏輯避免重複。
    """
    from mindgraph.retrievers import VectorRetriever
    from mindgraph.store import model_version, build_dsn

    vector_ret = None
    try:
        from mindgraph.providers import get_embedding_provider
        emb = get_embedding_provider(cfg)
        mv = model_version(cfg.embedding.provider, cfg.embedding.model, emb.dimension)
        vector_ret = VectorRetriever(
            build_dsn(cfg.database), embedding_provider=emb, model_version=mv
        )
    except Exception as e:
        logger.warning("vector retriever unavailable, degrading to keyword+graph: %s", e)

    llm = None
    try:
        from mindgraph.providers import get_llm_provider
        llm = get_llm_provider(cfg)
    except Exception as e:
        logger.info("LLM provider unavailable, Pass 2 expansion disabled: %s", e)

    return SmartSearchService(graph_engine=engine, vector_retriever=vector_ret, llm_provider=llm)
