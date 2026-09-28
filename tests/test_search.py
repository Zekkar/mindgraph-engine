import pytest
from unittest import mock

from mindgraph.graph import WikiGraphEngine
from mindgraph.search import SmartSearchService, INTENT_WEIGHTS, FAILURE_BOOST
from mindgraph.retrievers import ScoredDoc


def _make_wiki(tmp_path):
    wiki = tmp_path / "wiki"
    (wiki / "failures").mkdir(parents=True)
    (wiki / "dependency-injection.md").write_text(
        "---\ntags: [pattern]\n---\n# Dependency Injection\n\n## Core\n\n"
        "Dependency injection wires components together at runtime.\n",
        encoding="utf-8",
    )
    (wiki / "cache-layer.md").write_text(
        "---\ntags: [infra]\nrelated: [data-layer]\n---\n# Cache Layer\n\n"
        "The cache layer sits in front of the data layer.\n",
        encoding="utf-8",
    )
    (wiki / "data-layer.md").write_text(
        "---\ntags: [infra]\n---\n# Data Layer\n\nThe data layer persists records.\n",
        encoding="utf-8",
    )
    (wiki / "failures" / "connection-timeout.md").write_text(
        "---\ntags: [bug]\n---\n# Connection Timeout\n\n"
        "Connection timeout error happens when the socket exceeds the deadline.\n",
        encoding="utf-8",
    )
    return wiki


@pytest.fixture
def engine(tmp_path):
    _make_wiki(tmp_path)
    e = WikiGraphEngine(str(tmp_path / "wiki"))
    e.build()
    return e


def test_intent_weight_presets_sane():
    # location should favour keyword; concept should favour vector
    assert INTENT_WEIGHTS["location"]["keyword"] > INTENT_WEIGHTS["location"]["vector"]
    assert INTENT_WEIGHTS["concept"]["vector"] > INTENT_WEIGHTS["concept"]["keyword"]


def test_degrades_to_keyword_without_vector(engine):
    svc = SmartSearchService(graph_engine=engine, vector_retriever=None, llm_provider=None)
    res = svc.search("dependency injection", top_k=5)
    assert res["degradation"] == "vector_unavailable_fallback_to_keyword"
    assert "intent" in res
    assert "dependency-injection" in [r["concept_name"] for r in res["results"]]


def test_hybrid_merges_vector_results(engine):
    vec = mock.MagicMock()
    vec.retrieve.return_value = [
        ScoredDoc(concept_name="data-layer", section_id="Core",
                  chunk_text="the data layer persists records", score=0.9)
    ]
    svc = SmartSearchService(graph_engine=engine, vector_retriever=vec, llm_provider=None)
    res = svc.search("data layer 是什麼", top_k=5)
    assert res["intent"]["type"] == "concept"
    assert res["degradation"] is None
    assert "data-layer" in [r["concept_name"] for r in res["results"]]
    vec.retrieve.assert_called()


def test_failure_intent_boosts_failure_page(engine):
    svc = SmartSearchService(graph_engine=engine, vector_retriever=None, llm_provider=None)
    res = svc.search("connection timeout error 怎麼解決", top_k=5)
    assert res["intent"]["type"] == "failure"
    boosted = [
        r for r in res["results"]
        if r["concept_name"] == "connection-timeout"
        and r["retriever_breakdown"].get("intent_boost") == FAILURE_BOOST
    ]
    assert boosted, "failure-mode page should receive the intent boost"


def test_pass2_expansion_triggers_on_weak_retrieval(engine):
    mock_llm = mock.MagicMock()
    mock_llm.complete.return_value = "dependency injection\ncache layer"
    svc = SmartSearchService(graph_engine=engine, vector_retriever=None, llm_provider=mock_llm)
    res = svc.search("zzz-nonexistent-term", top_k=5)  # top1=0 < 0.5 → expansion
    assert mock_llm.complete.called
    assert "expanded_queries" in res
    # expanded sub-queries surface real pages that the original query missed
    assert any(
        n in [r["concept_name"] for r in res["results"]]
        for n in ("dependency-injection", "cache-layer")
    )


def test_no_expansion_when_llm_absent(engine):
    svc = SmartSearchService(graph_engine=engine, vector_retriever=None, llm_provider=None)
    res = svc.search("zzz-nonexistent-term", top_k=5)
    assert "expanded_queries" not in res


def test_build_smart_search_degrades_without_keys(engine, monkeypatch):
    from mindgraph.search import build_smart_search
    from mindgraph.config import MindGraphConfig, LLMConfig, EmbeddingConfig
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = MindGraphConfig(
        domain="t",
        llm=LLMConfig("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        embedding=EmbeddingConfig("openai", "text-embedding-3-small", "OPENAI_API_KEY"),
    )
    svc = build_smart_search(engine, cfg)
    assert svc.vector_retriever is None
    assert svc.llm_provider is None


def test_build_smart_search_wires_recency_with_same_model_version(engine, monkeypatch):
    """守接線（不只守邏輯）：有向量庫時 build_smart_search 必須建出 recency，
    且 recency 讀的 model_version 與 VectorRetriever 相同（I-2 / I-5）。"""
    from unittest import mock
    from mindgraph import recency as rec_mod
    from mindgraph.search import build_smart_search
    from mindgraph.config import MindGraphConfig, LLMConfig, EmbeddingConfig, RecencyConfig

    fake_emb = mock.MagicMock(dimension=768)
    monkeypatch.setattr("mindgraph.providers.get_embedding_provider", lambda c: fake_emb)
    seen = {}

    def fake_make(dsn, mv, cfg=None):
        seen["mv"] = mv
        return rec_mod.RecencyWeighter(lambda: {})

    monkeypatch.setattr(rec_mod, "make_db_weighter", fake_make)
    cfg = MindGraphConfig(
        domain="t",
        llm=LLMConfig("openai", "gpt-4o-mini", "NO_SUCH_KEY_ENV"),
        embedding=EmbeddingConfig("gemini", "gemini-embedding-001", "X"),
    )
    svc = build_smart_search(engine, cfg)
    assert svc.recency is not None
    assert seen["mv"] == svc.vector_retriever.model_version == "gemini-gemini-embedding-001-d768"

    cfg.recency = RecencyConfig(enabled=False)
    assert build_smart_search(engine, cfg).recency is None


def test_keyword_to_scored_dicts_shape():
    from mindgraph.search import keyword_to_scored_dicts
    out = keyword_to_scored_dicts([{"name": "a", "snippet": "s", "score": 0.5}])
    assert out[0]["concept_name"] == "a"
    assert out[0]["section_id"] is None
    assert "retriever_breakdown" in out[0]
