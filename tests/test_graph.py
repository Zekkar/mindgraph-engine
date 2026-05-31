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


def test_incremental_picks_up_new_file(wiki_dir):
    engine = WikiGraphEngine(str(wiki_dir))
    engine.build()
    assert engine.get_stats()["total_pages"] == 2
    (wiki_dir / "PageC.md").write_text(
        "---\ntype: concept\n---\n# Page C\n[[PageA]]\n", encoding="utf-8"
    )
    engine.ensure_fresh()  # triggers build_incremental
    assert engine.get_stats()["total_pages"] == 3
    assert "PageC" in engine.pages


def test_incremental_drops_deleted_file(wiki_dir):
    engine = WikiGraphEngine(str(wiki_dir))
    engine.build()
    (wiki_dir / "PageB.md").unlink()
    engine.ensure_fresh()
    assert "PageB" not in engine.pages
    assert engine.get_stats()["total_pages"] == 1


def test_diagnostics_reports_broken_and_orphans(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "A.md").write_text(
        "---\ntype: concept\n---\n# A\n\nlinks to [[Nonexistent]] page\n", encoding="utf-8"
    )
    (wiki / "Lonely.md").write_text("# Lonely\n\nno links and no frontmatter\n", encoding="utf-8")
    engine = WikiGraphEngine(str(wiki))
    engine.build()
    diag = engine.get_diagnostics()
    assert diag["broken_count"] >= 1
    assert any(b["link"] == "Nonexistent" for b in diag["broken_wikilinks"])
    assert "Lonely" in diag["orphan_pages"]
    assert "Lonely" in diag["pages_without_frontmatter"]
