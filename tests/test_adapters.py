import pytest
from mindgraph.adapters.base import DataSourceAdapter, RawDocument
from mindgraph.adapters.filesystem import FileSystemAdapter


def test_raw_document():
    doc = RawDocument(content="# Title", filename="test.md", metadata={})
    assert doc.filename == "test.md"


def test_filesystem_fetch(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nContent", encoding="utf-8")
    (notes / "b.md").write_text("# B\n\nContent", encoding="utf-8")
    (notes / "ignore.txt").write_text("plain", encoding="utf-8")
    adapter = FileSystemAdapter(source=str(notes), target="raw/notes")
    docs = adapter.fetch()
    assert len(docs) == 2
    assert adapter.get_target_dir() == "raw/notes"


def test_filesystem_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert FileSystemAdapter(source=str(empty), target="raw/notes").fetch() == []
