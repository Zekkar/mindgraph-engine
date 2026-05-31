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

    def to_dict(self) -> dict:
        return {
            "concept_name": self.concept_name,
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
) -> dict:
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

    pool: dict[str, ScoredDoc] = {}
    kw_max = max((d.score for d in kw_docs), default=1.0) or 1.0
    for rank, d in enumerate(kw_docs, 1):
        key = d.concept_name
        if key not in pool:
            pool[key] = ScoredDoc(
                concept_name=d.concept_name, section_id=d.section_id,
                chunk_text=d.chunk_text, score=0.0, retriever_breakdown={},
            )
        pool[key].score += weights["keyword"] * (d.score / kw_max)
        pool[key].retriever_breakdown.update({"keyword_rank": rank,
                                              "keyword_score": round(d.score, 4)})

    for rank, d in enumerate(vec_docs, 1):
        key = d.concept_name
        if key not in pool:
            pool[key] = ScoredDoc(
                concept_name=d.concept_name, section_id=d.section_id,
                chunk_text=d.chunk_text, score=0.0, retriever_breakdown={},
            )
        else:
            pool[key].section_id = d.section_id
            pool[key].chunk_text = d.chunk_text
        pool[key].score += weights["vector"] * d.score
        pool[key].retriever_breakdown.update({"vector_rank": rank,
                                              "vector_score": round(d.score, 4)})

    stage1_top = sorted(pool.values(), key=lambda x: x.score, reverse=True)[: top_k * 2]
    if graph_engine and hasattr(graph_engine, "get_related"):
        retrievers_run.append("graph_post")
        top_names = {d.concept_name for d in stage1_top}
        for doc in stage1_top:
            try:
                rel = graph_engine.get_related(doc.concept_name, depth=1)
                neighbors = {n.get("name") for n in rel.get("nodes", []) if isinstance(n, dict)}
            except Exception:
                neighbors = set()
            cooccur = neighbors & top_names
            if cooccur and len(cooccur) > 1:
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
        if len(above_threshold) >= 2:
            suggestions.append(
                f"對照查詢：related(\"{above_threshold[0].concept_name}\", depth=2)"
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
