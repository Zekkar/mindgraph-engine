from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import psycopg
from pgvector.psycopg import register_vector

from mindgraph.store import model_version as _model_version

logger = logging.getLogger("mindgraph.retrievers")

# Derive from the single source of truth so the reader default can never drift
# from the writer format used by `embed` / build_smart_search.
DEFAULT_MODEL_VERSION = _model_version("gemini", "gemini-embedding-001", 768)


@dataclass(slots=True)
class ScoredDoc:
    concept_name: str
    section_id: str | None
    chunk_text: str
    score: float
    retriever_breakdown: dict[str, Any] = field(default_factory=dict)
    category: str | None = None  # wiki 資料夾（知識種類）；hook 靠它挑出失敗模式頁

    def to_dict(self) -> dict:
        return {
            "concept_name": self.concept_name,
            "category": self.category,
            "section_id": self.section_id,
            "chunk_text": self.chunk_text,
            "score": round(self.score, 4),
            "retriever_breakdown": self.retriever_breakdown,
        }


class BaseRetriever(Protocol):
    name: str

    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]: ...


class KeywordRetriever:
    name = "keyword"

    def __init__(self, engine):
        self.engine = engine

    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]:
        raw = self.engine.search(query, top_k)
        if not isinstance(raw, list):
            return []
        max_score = max((r.get("score", 0) for r in raw), default=1.0) or 1.0
        return [
            ScoredDoc(
                concept_name=r.get("name") or "?",
                section_id=None,
                chunk_text=r.get("snippet") or r.get("preview") or "",
                score=r.get("score", 0) / max_score,
                retriever_breakdown={"keyword_raw": r.get("score", 0)},
            )
            for r in raw
        ]


class VectorRetriever:
    name = "vector"

    def __init__(self, dsn: str, embedding_provider=None,
                 model_version: str = DEFAULT_MODEL_VERSION):
        self.dsn = dsn
        self._embedding_provider = embedding_provider
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        """查詢時比對的 embedding model_version（須與 embed 寫入端相同）。"""
        return self._model_version

    def embed_query(self, query: str) -> list[float]:
        if self._embedding_provider is not None:
            return self._embedding_provider.embed(query)
        import os
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set and no embedding_provider supplied")
        client = genai.Client(api_key=api_key)
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        return list(resp.embeddings[0].values)

    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]:
        try:
            q_vec = self.embed_query(query)
        except Exception as e:
            logger.warning("embed_query failed: %s", e)
            return []

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT concept_name, section_id, chunk_text,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM wiki.embeddings
                    WHERE model_version = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_vec, self._model_version, q_vec, top_k),
                )
                rows = cur.fetchall()

        return [
            ScoredDoc(
                concept_name=name,
                section_id=section,
                chunk_text=chunk[:1000],
                score=float(sim),
                retriever_breakdown={"vector_cosine": float(sim)},
            )
            for name, section, chunk, sim in rows
        ]


def two_stage_hybrid(
    query: str,
    top_k: int,
    keyword_ret: KeywordRetriever,
    vector_ret: VectorRetriever | None,
    graph_engine,
    weights: dict | None = None,
    min_score: float = 0.15,
    boost_fn=None,
    recency=None,
) -> dict:
    """兩階段混合檢索：關鍵字＋向量召回 → 加權合併 → 時間遞減 → 圖共現加分 → 門檻。

    時間遞減（recency，RecencyWeighter）的位置不可移動（不變式 I-4）：
    放在 retriever 內部會被關鍵字再正規化抵銷；放在圖加分之後，候選挑選就不看新舊。
    """
    t0 = time.time()
    weights = weights or {"keyword": 0.4, "vector": 0.6}
    over_k = max(top_k * 3, 15)
    retrievers_run = []
    degradation = None

    try:
        kw_docs = keyword_ret.retrieve(query, over_k)
        retrievers_run.append("keyword")
    except Exception as e:
        logger.warning("keyword failed: %s", e)
        kw_docs = []

    if vector_ret is None:
        # explicit degraded path: no vector retriever wired (no DB/keys)
        vec_docs = []
        degradation = "vector_unavailable_fallback_to_keyword"
    else:
        try:
            vec_docs = vector_ret.retrieve(query, over_k)
            retrievers_run.append("vector")
        except Exception as e:
            logger.warning("vector failed: %s", e)
            vec_docs = []
            degradation = "vector_unavailable_fallback_to_keyword"

    # Merge by concept_name. NOTE: each retriever's hits are section-level, so a
    # single concept can appear more than once (multiple sections matched). We
    # keep only the best-scoring section per concept per retriever — summing
    # every hit with `+=` would let concepts with many matched sections
    # out-rank a concept with a single, precisely-matching section, and
    # unconditionally overwriting chunk_text/section_id on every hit would
    # leave whichever section was processed last (not necessarily the best)
    # as the representative content. See trading-wiki devdiary 2026-09-05 /
    # mindgraph-code-exists-not-wired-adjacent bug: this file was copied from
    # trading-wiki's mcp-server/retrievers.py before that fix landed there.
    pool: dict[str, ScoredDoc] = {}
    kw_best: dict[str, float] = {}
    vec_best: dict[str, float] = {}

    kw_max = max((d.score for d in kw_docs), default=1.0) or 1.0
    for rank, d in enumerate(kw_docs, 1):
        key = d.concept_name
        norm_score = d.score / kw_max
        if key not in pool:
            pool[key] = ScoredDoc(
                concept_name=d.concept_name, section_id=d.section_id,
                chunk_text=d.chunk_text, score=0.0, retriever_breakdown={},
            )
        if norm_score > kw_best.get(key, -1.0):
            kw_best[key] = norm_score
            pool[key].retriever_breakdown.update({"keyword_rank": rank,
                                                  "keyword_score": round(d.score, 4)})

    for rank, d in enumerate(vec_docs, 1):
        key = d.concept_name
        if key not in pool:
            pool[key] = ScoredDoc(
                concept_name=d.concept_name, section_id=d.section_id,
                chunk_text=d.chunk_text, score=0.0, retriever_breakdown={},
            )
        if d.score > vec_best.get(key, -1.0):
            vec_best[key] = d.score
            pool[key].section_id = d.section_id
            pool[key].chunk_text = d.chunk_text
            pool[key].retriever_breakdown.update({"vector_rank": rank,
                                                  "vector_score": round(d.score, 4)})

    for key, doc in pool.items():
        doc.score = weights["keyword"] * kw_best.get(key, 0.0) + weights["vector"] * vec_best.get(key, 0.0)

    pages = getattr(graph_engine, "pages", None) or {}
    for doc in pool.values():
        doc.category = (pages.get(doc.concept_name) or {}).get("category")

    if recency is not None:
        try:
            recency.apply(pool.values())
            retrievers_run.append("recency")
        except Exception as e:  # 降級：時間遞減失敗不影響檢索
            logger.warning("recency failed: %s", e)

    stage1_top = sorted(pool.values(), key=lambda x: x.score, reverse=True)[: top_k * 2]
    if graph_engine and hasattr(graph_engine, "neighbors"):
        retrievers_run.append("graph_post")
        top_names = {d.concept_name for d in stage1_top}
        for doc in stage1_top:
            try:
                neighbors = graph_engine.neighbors(doc.concept_name)
            except Exception:
                neighbors = set()
            cooccur = neighbors & top_names
            # 至少 1 個「真正的」鄰居也在候選內就加分。舊版 get_related 把概念自己算進鄰居，
            # 寫成 len > 1 實際等於 ≥1 個真鄰居；參考實作的線上行為與命中率都建立在這個語意上。
            if cooccur:
                doc.score += 0.1
                doc.retriever_breakdown["graph_boost"] = 0.1
                doc.retriever_breakdown["cooccur_neighbors"] = list(cooccur)[:3]

    # intent-driven re-scoring (e.g. boost failure-mode pages) before final ranking
    if boost_fn is not None:
        for doc in pool.values():
            try:
                factor = boost_fn(doc)
            except Exception:
                factor = 1.0
            if factor and factor != 1.0:
                doc.score *= factor
                doc.retriever_breakdown["intent_boost"] = round(factor, 3)

    final = sorted(pool.values(), key=lambda x: x.score, reverse=True)
    above_threshold = [d for d in final if d.score >= min_score][:top_k]
    if not above_threshold and final:
        degradation = degradation or "no_match_above_threshold"

    elapsed_ms = int((time.time() - t0) * 1000)
    if above_threshold:
        top1 = above_threshold[0]
        interpretation = (
            f"找到 {len(above_threshold)} 筆相關概念，最高分 [{top1.concept_name}]"
            + (f"#{top1.section_id}" if top1.section_id else "")
            + f" (score={top1.score:.2f})。"
        )
    else:
        interpretation = "未找到分數超過閾值的相關概念，建議改寫查詢或新建概念。"

    suggestions = []
    if above_threshold:
        suggestions.append(f"深入查詢：get_concept(\"{above_threshold[0].concept_name}\")")
        if above_threshold[0].section_id:
            suggestions.append(
                f"只讀該段：get_section(\"{above_threshold[0].concept_name}\", "
                f"\"{above_threshold[0].section_id}\")"
            )

    return {
        "query": query,
        "results": [d.to_dict() for d in above_threshold],
        "interpretation": interpretation,
        "suggestions": suggestions,
        "cache_hit": False,
        "degradation": degradation,
        "stats": {
            "latency_ms": elapsed_ms,
            "retrievers_run": retrievers_run,
            "stage1_candidates": len(pool),
            "stage2_boosted": sum(
                1 for d in final if "graph_boost" in d.retriever_breakdown
            ),
        },
    }
