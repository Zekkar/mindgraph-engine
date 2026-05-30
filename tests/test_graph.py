import pytest
from mindgraph.graph import WikiGraphEngine


@pytest.fixture
def wiki_dir(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "PageA.md").write_text(
        "---\ntype: concept\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\n# Page A\n[[PageB]]",
        encoding="utf-8",
    )
    (wiki / "PageB.md").write_text(
        "---\ntype: concept\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\n# Page B\nCore content.",
        encoding="utf-8",
    )
    return wiki


def test_graph_builds(wiki_dir):
    engine = WikiGraphEngine(str(wiki_dir))
    engine.build()
    assert engine.get_stats()["total_pages"] == 2


def test_search(wiki_dir):
    engine = WikiGraphEngine(str(wiki_dir))
    engine.build()
    assert len(engine.search("content", 5)) > 0


def test_get_concept(wiki_dir):
    engine = WikiGraphEngine(str(wiki_dir))
    engine.build()
    assert engine.get_concept("PageA") is not None
