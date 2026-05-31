from unittest import mock

from mindgraph.retrievers import KeywordRetriever, VectorRetriever, two_stage_hybrid


class _FakeEngine:
    def __init__(self, results):
        self._r = results

    def search(self, q, k):
        return self._r

    def get_related(self, name, depth=1):
        return {"nodes": []}


KW_ONLY = {"keyword": 1.0, "vector": 0.0}


def test_keyword_normalization():
    eng = _FakeEngine([
        {"name": "a", "score": 4.0, "snippet": "x"},
        {"name": "b", "score": 2.0, "snippet": "y"},
    ])
    docs = KeywordRetriever(eng).retrieve("q", 10)
    scores = {d.concept_name: d.score for d in docs}
    assert scores["a"] == 1.0 and scores["b"] == 0.5


def test_keyword_all_zero_no_div_by_zero():
    eng = _FakeEngine([{"name": "a", "score": 0, "snippet": ""}])
    docs = KeywordRetriever(eng).retrieve("q", 10)
    assert docs[0].score == 0.0


def test_keyword_non_list_returns_empty():
    assert KeywordRetriever(_FakeEngine("not-a-list")).retrieve("q", 10) == []


def test_boost_fn_applied_directly():
    eng = _FakeEngine([
        {"name": "hit", "score": 1.0, "snippet": ""},
        {"name": "miss", "score": 1.0, "snippet": ""},
    ])
    res = two_stage_hybrid(
        "q", 10, KeywordRetriever(eng), None, None, weights=KW_ONLY,
        boost_fn=lambda d: 2.0 if d.concept_name == "hit" else 1.0,
    )
    by = {r["concept_name"]: r for r in res["results"]}
    assert by["hit"]["retriever_breakdown"]["intent_boost"] == 2.0
    assert by["hit"]["score"] > by["miss"]["score"]


def test_boost_fn_raising_is_safe():
    eng = _FakeEngine([{"name": "a", "score": 1.0, "snippet": ""}])

    def boom(_doc):
        raise ValueError("nope")

    res = two_stage_hybrid("q", 10, KeywordRetriever(eng), None, None,
                           weights=KW_ONLY, boost_fn=boom)
    assert res["results"][0]["retriever_breakdown"].get("intent_boost") is None


def test_none_vector_degrades():
    eng = _FakeEngine([{"name": "a", "score": 1.0, "snippet": ""}])
    res = two_stage_hybrid("q", 10, KeywordRetriever(eng), None, None, weights=KW_ONLY)
    assert res["degradation"] == "vector_unavailable_fallback_to_keyword"


def test_raising_vector_degrades():
    eng = _FakeEngine([{"name": "a", "score": 1.0, "snippet": ""}])
    bad_vec = mock.MagicMock()
    bad_vec.retrieve.side_effect = RuntimeError("db down")
    res = two_stage_hybrid("q", 10, KeywordRetriever(eng), bad_vec, None, weights=KW_ONLY)
    assert res["degradation"] == "vector_unavailable_fallback_to_keyword"


def test_embed_query_forwards_to_provider():
    prov = mock.MagicMock()
    prov.embed.return_value = [0.1, 0.2]
    vr = VectorRetriever("dsn", embedding_provider=prov, model_version="m")
    assert vr.embed_query("hello") == [0.1, 0.2]
    prov.embed.assert_called_once_with("hello")
