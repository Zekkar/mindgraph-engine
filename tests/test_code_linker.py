"""code↔wiki 連結器測試：純函式 + 以 monkeypatch 替換資料庫存取的流程測試。"""

from unittest.mock import MagicMock

import pytest

from mindgraph.code import linker
from mindgraph.code.linker import (
    embedding_text,
    link_wiki,
    max_containment,
    merge_links,
    name_matches,
    rank_concepts,
    tokenize_name,
)


def test_tokenize_camel_snake_kebab():
    assert tokenize_name("CodeIndexer") == {"code", "indexer"}
    assert tokenize_name("HTTPServerPool") == {"http", "server", "pool"}
    assert tokenize_name("link_wiki_concepts") == {"link", "wiki", "concepts"}
    assert tokenize_name("wiki-linker-design-v2") == {"wiki", "linker", "design"}
    assert tokenize_name("to_id") == set()           # 長度 < 3 的 token 丟棄
    assert tokenize_name("程式碼圖") == set()          # 非 ASCII 不產生 token


def test_max_containment():
    assert max_containment({"wiki", "linker"}, {"wiki", "linker", "design"}) == 1.0
    assert max_containment({"a1x", "b2x"}, {"a1x", "c3x", "d4x", "e5x"}) == 0.5
    assert max_containment({"abc"}, {"xyz"}) == 0.0
    assert max_containment(set(), {"abc"}) == 0.0


def test_name_matches_threshold():
    rows = name_matches(
        [(1, "WikiLinker"), (2, "run"), (3, "parse_config_file")],
        ["wiki-linker-design", "config-loader", "unrelated-topic", "中文概念"],
    )
    got = {(r[0], r[1]): r[3] for r in rows}
    assert got[(1, "wiki-linker-design")] == 1.0
    assert got[(3, "config-loader")] == pytest.approx(0.5)
    assert all(r[2] == "name_match" for r in rows)
    assert not any(k[0] == 2 for k in got)


def test_embedding_text_min_length():
    assert embedding_text("f", "short") is None
    assert embedding_text("f", "x" * 50) is None           # 恰 50 字元不收（需 > 50）
    assert embedding_text("f", None) is None
    text = embedding_text("f", "y" * 1000)
    assert text == "f: " + "y" * 800


def test_rank_concepts_threshold_topk_and_max_per_concept():
    rows = [("A", 0.80), ("A", 0.90), ("B", 0.75), ("C", 0.73),
            ("D", 0.719), ("E", 0.85)]
    ranked = rank_concepts(rows)
    assert [c for c, _ in ranked] == ["A", "E", "B"]      # 每 symbol 最多 3
    assert ranked[0][1] == pytest.approx(0.90)             # 同概念取最高
    assert "D" not in dict(rank_concepts(rows, top_k=10))  # 0.719 < 0.72


def test_rank_concepts_recency_applies_after_raw_threshold():
    rows = [("old", 0.95), ("new", 0.80), ("low", 0.70)]
    recency = {"old": 0.5, "new": 1.0, "low": 2.0}.get
    ranked = rank_concepts(rows, recency_fn=recency)
    # old 的原始相似度過門檻，折扣後 0.475 仍保留；low 原始 0.70 < 0.72，即使 ×2 也不收
    assert ranked == [("new", pytest.approx(0.80)), ("old", pytest.approx(0.475))]


def test_merge_links_keeps_higher_score():
    merged = merge_links([
        (1, "A", "name_match", 0.5),
        (1, "A", "embedding_match", 0.8),
        (1, "B", "name_match", 1.0),
        (1, "B", "embedding_match", 0.9),
        (2, "A", "name_match", 0.6),
    ])
    got = {(r[0], r[1]): (r[2], r[3]) for r in merged}
    assert got[(1, "A")] == ("embedding_match", 0.8)
    assert got[(1, "B")] == ("name_match", 1.0)
    assert got[(2, "A")] == ("name_match", 0.6)


@pytest.fixture
def patched_db(monkeypatch):
    """把 link_wiki 的資料庫存取替換成記憶體假資料。"""
    state = {"written": None}
    symbols = [
        (1, "WikiLinker", "把程式碼 symbol 連結到 wiki 知識概念的元件，負責重建跨層連結。" * 2),
        (2, "tiny", "太短"),
        (3, "boom", "這個 symbol 的 embedding 會失敗，用來確認單筆失敗不影響整批處理。" * 2),
    ]
    monkeypatch.setattr(linker, "ensure_code_schema", lambda conn: None)
    monkeypatch.setattr(linker, "_wiki_table_exists", lambda conn: True)
    monkeypatch.setattr(linker, "_fetch_symbols", lambda conn, repo: symbols)
    monkeypatch.setattr(linker, "_fetch_concepts", lambda conn, mv: ["wiki-linker", "other"])
    monkeypatch.setattr(
        linker, "_nearest_sections",
        lambda conn, vec, mv: [("wiki-linker", 0.9), ("other", 0.8), ("far", 0.5)],
    )

    def fake_replace(conn, repo, links):
        state["written"] = links
        return {"embedding_match": 7}

    monkeypatch.setattr(linker, "_replace_links", fake_replace)
    return state


def _provider():
    p = MagicMock()

    def embed(text):
        if text.startswith("boom"):
            raise RuntimeError("api down")
        return [0.1, 0.2]

    p.embed.side_effect = embed
    return p


def test_link_wiki_all(patched_db):
    provider = _provider()
    out = link_wiki(MagicMock(), "demo", provider, "prov-model-d2",
                    recency_fn=lambda c: 0.5 if c == "other" else 1.0)
    assert provider.embed.call_count == 2          # tiny 被略過
    assert out["embedded_symbols"] == 1 and out["embed_failures"] == 1
    written = {(r[0], r[1]): (r[2], r[3]) for r in patched_db["written"]}
    # name_match 1.0 與 embedding 0.9 衝突 → 保留 name_match 1.0
    assert written[(1, "wiki-linker")] == ("name_match", 1.0)
    assert written[(1, "other")] == ("embedding_match", pytest.approx(0.4))
    assert (1, "far") not in written
    assert out["by_link_type"] == {"name_match": 1, "embedding_match": 1}
    assert out["total_links"] == 2
    assert out["cleared_links"] == {"embedding_match": 7}


def test_link_wiki_name_only_needs_no_provider(patched_db):
    out = link_wiki(MagicMock(), "demo", methods="name_only")
    assert out["by_link_type"] == {"name_match": 1, "embedding_match": 0}
    assert out["embedded_symbols"] == 0


def test_link_wiki_embedding_only(patched_db):
    out = link_wiki(MagicMock(), "demo", _provider(), "mv", methods="embedding_only")
    assert out["by_link_type"]["name_match"] == 0
    assert out["by_link_type"]["embedding_match"] == 2


def test_link_wiki_validates_arguments(patched_db):
    with pytest.raises(ValueError):
        link_wiki(MagicMock(), "demo", methods="bogus")
    with pytest.raises(ValueError):
        link_wiki(MagicMock(), "demo", methods="all")   # 缺 provider / model_version
