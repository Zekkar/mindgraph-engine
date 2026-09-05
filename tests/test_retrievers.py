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


# --- Regression: Stage 1 merge must not accumulate scores across sections ---
# This file was copied from trading-wiki's mcp-server/retrievers.py before a
# 2026-09-05 fix landed there (found via real RAGAS evaluation). The bug:
# `pool[key].score += ...` on a `+=` keyed only by concept_name let a concept
# with several matched sections out-score a concept with a single, precisely
# matching section; the representative chunk_text/section_id was also
# unconditionally overwritten by whichever section was processed last (not
# necessarily the best-matching one). Fixed by tracking each retriever's
# best-per-concept score separately and computing the final weighted sum once.

def _vec_docs(docs):
    """Build a fake vector retriever returning a fixed ScoredDoc list."""
    from mindgraph.retrievers import ScoredDoc
    vr = mock.MagicMock()
    vr.retrieve.return_value = [ScoredDoc(*d) for d in docs]
    return vr


def test_multiple_vector_hits_same_concept_do_not_accumulate():
    eng = _FakeEngine([])
    # concept "A" has two sections matched (simulating a "popular, many-section"
    # page); concept "B" has a single, more precisely matching section.
    vec = _vec_docs([
        ("B", "only_section", "B's single relevant section", 0.8, {}),
        ("A", "section_1", "A's first section", 0.6, {}),
        ("A", "section_2", "A's second section", 0.55, {}),
    ])
    res = two_stage_hybrid("q", 10, KeywordRetriever(eng), vec, None,
                           weights={"keyword": 0.0, "vector": 1.0})
    by = {r["concept_name"]: r for r in res["results"]}
    # Pre-fix: A = 0.6 + 0.55 = 1.15 (wrongly beats B = 0.8). Post-fix: A keeps
    # only its best section (0.6), so B correctly outranks A.
    assert by["B"]["score"] > by["A"]["score"]


def test_representative_section_is_best_scoring_not_last_processed():
    vec = _vec_docs([
        ("A", "best_section", "the section with the actual answer", 0.9, {}),
        ("A", "worse_section", "an unrelated section", 0.3, {}),
    ])
    res = two_stage_hybrid("q", 10, KeywordRetriever(_FakeEngine([])), vec, None,
                           weights={"keyword": 0.0, "vector": 1.0})
    doc = res["results"][0]
    assert doc["section_id"] == "best_section"
    assert doc["chunk_text"] == "the section with the actual answer"
