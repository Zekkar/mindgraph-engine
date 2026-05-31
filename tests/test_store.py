import pytest

from mindgraph.store import iter_sections, model_version, build_dsn, Section
from mindgraph.config import DatabaseConfig


def test_model_version_format_golden():
    # locks the writer format; the embed/VectorRetriever mismatch bug came from drift here
    assert model_version("gemini", "gemini-embedding-001", 768) == "gemini-gemini-embedding-001-d768"
    mv = model_version("openai", "text-embedding-3-small", 1536)
    assert mv.startswith("openai") and mv.endswith("-d1536")


def test_default_model_version_matches_writer():
    # retrievers' reader default MUST equal the writer format (regression guard)
    from mindgraph.retrievers import DEFAULT_MODEL_VERSION
    assert DEFAULT_MODEL_VERSION == model_version("gemini", "gemini-embedding-001", 768)


def test_iter_sections_chunking(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "p.md").write_text(
        "---\ntags: [x]\n---\n# Title\n\n## A\n\nalpha body\n\n## B\n\nbeta body\n",
        encoding="utf-8",
    )
    secs = list(iter_sections(wiki))
    ids = {s.section_id for s in secs}
    assert "A" in ids and "B" in ids
    assert all(s.concept_name == "p" for s in secs)


def test_iter_sections_headingless_fallback(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "flat.md").write_text("just text, no level-2 headings here\n", encoding="utf-8")
    secs = list(iter_sections(wiki))
    assert len(secs) == 1 and secs[0].section_id == "__page__"


def test_iter_sections_skips_index_and_log(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text("# idx\n\n## S\n\nx\n", encoding="utf-8")
    (wiki / "log.md").write_text("# log\n\n## S\n\nx\n", encoding="utf-8")
    assert list(iter_sections(wiki)) == []


def test_content_hash_sensitivity():
    a = Section("c", "s", "hello")
    b = Section("c", "s", "hello")
    c = Section("c", "s", "hello!")
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_build_dsn_includes_params_without_password(monkeypatch):
    monkeypatch.delenv("MINDGRAPH_DB_PASSWORD", raising=False)
    db = DatabaseConfig(host="h", port=5599, name="db", user="u",
                        password_env="MINDGRAPH_DB_PASSWORD")
    dsn = build_dsn(db)
    assert "host=h" in dsn and "port=5599" in dsn and "dbname=db" in dsn and "user=u" in dsn
    assert "password" not in dsn


def test_build_dsn_quotes_special_password(monkeypatch):
    # make_conninfo must round-trip a password with spaces/quotes without corrupting the DSN
    monkeypatch.setenv("MINDGRAPH_DB_PASSWORD", "p ass'wd")
    db = DatabaseConfig(password_env="MINDGRAPH_DB_PASSWORD")
    from psycopg.conninfo import conninfo_to_dict
    assert conninfo_to_dict(build_dsn(db))["password"] == "p ass'wd"
