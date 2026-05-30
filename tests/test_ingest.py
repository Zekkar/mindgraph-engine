from mindgraph.ingest import run_ingest, IngestConfig
from mindgraph.adapters.base import DataSourceAdapter, RawDocument


class MockAdapter(DataSourceAdapter):
    def fetch(self):
        return [RawDocument(content="# Test\n\nContent.", filename="test.md")]

    def get_target_dir(self):
        return "raw/notes"


def test_ingest_writes_wiki(tmp_path, mock_llm):
    cfg = IngestConfig(
        raw_root=tmp_path / "raw",
        wiki_root=tmp_path / "wiki",
        domain="test",
        language="en",
        state_file=tmp_path / ".ingest-state.json",
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    stats = run_ingest(cfg, llm_provider=mock_llm, adapters=[MockAdapter()])
    assert stats["raw_written"] >= 1
    assert stats["wiki_written"] >= 1


def test_ingest_idempotent(tmp_path, mock_llm):
    cfg = IngestConfig(
        raw_root=tmp_path / "raw",
        wiki_root=tmp_path / "wiki",
        domain="test",
        language="en",
        state_file=tmp_path / ".ingest-state.json",
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    run_ingest(cfg, llm_provider=mock_llm, adapters=[MockAdapter()])
    stats2 = run_ingest(cfg, llm_provider=mock_llm, adapters=[MockAdapter()])
    assert stats2["wiki_written"] == 0
    assert stats2["skipped"] >= 1
