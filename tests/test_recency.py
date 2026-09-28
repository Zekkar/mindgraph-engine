from datetime import datetime, timedelta, timezone

import pytest

from mindgraph.recency import PageRecency, RecencyWeighter, decay_factor
from mindgraph.retrievers import ScoredDoc, two_stage_hybrid

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)


def _weighter(pages, **kw):
    return RecencyWeighter(lambda: pages, now=lambda: NOW, **kw)


def _doc(name, score):
    return ScoredDoc(concept_name=name, section_id=None, chunk_text="", score=score)


def test_decay_factor_bounds():
    assert decay_factor(0, 0.003) == pytest.approx(1.0)
    assert decay_factor(-5, 0.003) == pytest.approx(1.0)  # 負年齡視為 0
    assert decay_factor(10_000, 0.003) == pytest.approx(0.5, abs=1e-6)  # 永不低於地板


def test_newer_page_outranks_older_on_equal_score():
    pages = {
        "old": PageRecency("系統", NOW - timedelta(days=400)),
        "new": PageRecency("系統", NOW - timedelta(days=1)),
    }
    ranked = _weighter(pages).apply([_doc("old", 0.8), _doc("new", 0.8)])
    assert [d.concept_name for d in ranked] == ["new", "old"]
    assert ranked[1].retriever_breakdown["pre_recency_score"] == 0.8


def test_failure_modes_and_evergreen_never_decay():
    old = NOW - timedelta(days=1000)
    pages = {"lesson": PageRecency("failure-modes", old), "rule": PageRecency("系統", old)}
    w = _weighter(pages, evergreen=["rule"])
    ranked = w.apply([_doc("lesson", 0.7), _doc("rule", 0.6)])
    assert ranked[0].score == pytest.approx(0.7)
    assert ranked[1].score == pytest.approx(0.6)
    assert ranked[0].retriever_breakdown["decay_lambda"] == 0.0


def test_custom_category_ratio_overrides_default():
    old = NOW - timedelta(days=300)
    w = _weighter({"a": PageRecency("news", old)}, category_ratio={"news": 5.0})
    assert w.factor_for("a") < _weighter({"a": PageRecency("x", old)}).factor_for("a")


def test_missing_age_keeps_score_and_loader_failure_degrades():
    w = _weighter({})
    [d] = w.apply([_doc("unknown", 0.5)])
    assert d.score == 0.5

    def boom():
        raise RuntimeError("db down")

    broken = RecencyWeighter(boom, now=lambda: NOW)
    [d2] = broken.apply([_doc("x", 0.4)])
    assert d2.score == 0.4  # 從未成功載入 → 不衰減、不中斷


def test_snapshot_is_cached_within_ttl():
    calls = []

    def loader():
        calls.append(1)
        return {}

    t = [0.0]
    w = RecencyWeighter(loader, ttl_s=300, clock=lambda: t[0], now=lambda: NOW)
    w.apply([]); w.apply([])
    t[0] = 301
    w.apply([])
    assert len(calls) == 2


class _KW:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, q, k):
        return [_doc(n, s) for n, s in self.docs]


def test_hybrid_applies_recency_after_weighting_before_threshold():
    """不變式 I-4：時間遞減在加權合併之後套用，會真的改變最終排序。"""
    pages = {
        "old": PageRecency("系統", NOW - timedelta(days=2000)),
        "new": PageRecency("系統", NOW),
    }
    kw = _KW([("old", 1.0), ("new", 0.9)])
    res = two_stage_hybrid("q", 5, kw, None, None, recency=_weighter(pages))
    names = [r["concept_name"] for r in res["results"]]
    assert names == ["new", "old"]
    assert "recency" in res["stats"]["retrievers_run"]
