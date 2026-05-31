from unittest import mock

from typer.testing import CliRunner

import mindgraph.cli as cli

runner = CliRunner()


def _setup(tmp_path):
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "a.md").write_text(
        "# A\n\nbody about alpha and caching\n", encoding="utf-8"
    )
    (tmp_path / "mindgraph.yml").write_text(
        "domain: t\nlanguage: en\n"
        "llm:\n  provider: openai\n  model: gpt-4o-mini\n  api_key_env: OPENAI_API_KEY\n"
        "embedding:\n  provider: openai\n  model: text-embedding-3-small\n  api_key_env: OPENAI_API_KEY\n"
        "server:\n  rest_port: 8401\n",
        encoding="utf-8",
    )
    return str(tmp_path), str(tmp_path / "mindgraph.yml")


def test_embed_dry_run_needs_no_key(tmp_path, monkeypatch):
    base, cfg = _setup(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = runner.invoke(cli.app, ["embed", "--dry-run", "--base-dir", base, "--config", cfg])
    assert r.exit_code == 0
    assert "dry-run" in r.stdout


def test_search_default_keyword(tmp_path):
    base, cfg = _setup(tmp_path)
    r = runner.invoke(cli.app, ["search", "alpha", "--base-dir", base, "--config", cfg])
    assert r.exit_code == 0
    assert "alpha" in r.stdout


def test_search_hybrid_routes_through_smart(tmp_path, monkeypatch):
    base, cfg = _setup(tmp_path)
    fake = mock.MagicMock()
    fake.search.return_value = {"results": [{"concept_name": "SENTINEL"}], "degradation": None}
    monkeypatch.setattr("mindgraph.search.build_smart_search", lambda engine, cfg: fake)
    r = runner.invoke(cli.app, ["search", "x", "--hybrid", "--base-dir", base, "--config", cfg])
    assert r.exit_code == 0
    assert "SENTINEL" in r.stdout
    fake.search.assert_called_once()


def test_serve_port_and_host(tmp_path, monkeypatch):
    base, cfg = _setup(tmp_path)
    monkeypatch.setattr("mindgraph.server.rest.build_app", lambda base_dir, config_path: object())
    run_mock = mock.MagicMock()
    monkeypatch.setattr("uvicorn.run", run_mock)

    r = runner.invoke(cli.app, ["serve", "--base-dir", base, "--config", cfg])
    assert r.exit_code == 0
    assert run_mock.call_args.kwargs["port"] == 8401          # default → config rest_port
    assert run_mock.call_args.kwargs["host"] == "127.0.0.1"   # secure default bind

    run_mock.reset_mock()
    r = runner.invoke(cli.app, ["serve", "--port", "9999", "--base-dir", base, "--config", cfg])
    assert run_mock.call_args.kwargs["port"] == 9999


def test_mcp_runs_stdio(tmp_path, monkeypatch):
    base, cfg = _setup(tmp_path)
    fake_server = mock.MagicMock()
    monkeypatch.setattr("mindgraph.server.mcp_server.build_mcp",
                        lambda base_dir, config_path: fake_server)
    r = runner.invoke(cli.app, ["mcp", "--base-dir", base, "--config", cfg])
    assert r.exit_code == 0
    fake_server.run.assert_called_once_with(transport="stdio")


class _FakeStore:
    """In-memory stand-in for EmbeddingStore to test embed idempotency without a DB."""
    store: dict = {}
    upserts = 0
    touches = 0

    def __init__(self, dsn):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def ensure_schema(self, dim, mv):
        pass

    def get_existing_hash(self, c, s, m):
        return _FakeStore.store.get((c, s))

    def touch(self, c, s, m, ts):
        _FakeStore.touches += 1

    def upsert(self, section, emb, m, ts):
        _FakeStore.store[(section.concept_name, section.section_id)] = section.content_hash
        _FakeStore.upserts += 1

    def mark_sweep_orphans(self, m, ts):
        return 0

    def commit(self):
        pass


def test_embed_idempotency_skips_unchanged(tmp_path, monkeypatch):
    base, cfg = _setup(tmp_path)
    fake_emb = mock.MagicMock()
    fake_emb.embed.return_value = [0.0] * 8
    fake_emb.dimension = 8
    monkeypatch.setattr("mindgraph.providers.get_embedding_provider", lambda c: fake_emb)
    monkeypatch.setattr("mindgraph.store.EmbeddingStore", _FakeStore)
    _FakeStore.store.clear()
    _FakeStore.upserts = 0
    _FakeStore.touches = 0

    args = ["embed", "--base-dir", base, "--config", cfg]
    r1 = runner.invoke(cli.app, args)
    assert r1.exit_code == 0, r1.output
    first = _FakeStore.upserts
    assert first >= 1

    r2 = runner.invoke(cli.app, args)
    assert r2.exit_code == 0
    assert _FakeStore.upserts == first   # nothing re-embedded
    assert _FakeStore.touches >= 1       # unchanged section touched instead


def test_timeline_command(tmp_path):
    base, _cfg = _setup(tmp_path)
    raw = tmp_path / "raw" / "devdiary"
    raw.mkdir(parents=True)
    (raw / "2026-02-02.md").write_text("# d\n\nnote about alpha caching\n", encoding="utf-8")
    r = runner.invoke(cli.app, ["timeline", "alpha", "--base-dir", base])
    assert r.exit_code == 0
    assert "2026-02-02" in r.stdout
    assert "mentions" in r.stdout


def test_check_reports_diagnostics(tmp_path):
    base, _cfg = _setup(tmp_path)
    (tmp_path / "wiki" / "b.md").write_text("# B\n\n[[ghost]] missing link\n", encoding="utf-8")
    r = runner.invoke(cli.app, ["check", "--base-dir", base])
    assert r.exit_code == 0
    assert "diagnostics" in r.stdout
    assert "broken links" in r.stdout


def test_check_strict_fails_on_broken_links(tmp_path):
    base, _cfg = _setup(tmp_path)
    (tmp_path / "wiki" / "b.md").write_text("# B\n\n[[ghost]] missing link\n", encoding="utf-8")
    r = runner.invoke(cli.app, ["check", "--strict", "--base-dir", base])
    assert r.exit_code == 1
