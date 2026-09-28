import json

from typer.testing import CliRunner

from mindgraph import cli
from mindgraph.config import RecallConfig
from mindgraph.evaluate import evaluate_hit_rate, is_hit
from mindgraph.recall import RecallClient, clean_prompt


def _results(*rows):
    return {"results": [
        {"concept_name": n, "category": c, "section_id": "s", "chunk_text": "## h\nbody", "score": sc}
        for n, c, sc in rows
    ]}


def _client(tmp_path, rows, clock=lambda: 1000.0):
    urls = []

    def fetch(url):
        urls.append(url)
        return _results(*rows)

    c = RecallClient(RecallConfig(), state_dir=tmp_path, fetch=fetch, clock=clock)
    return c, urls


def test_clean_prompt_strips_harness_and_channel_attrs():
    raw = ('<channel source="tg" chat_id="123" message_id="9">'
           '幫我看部署為什麼失敗</channel><system-reminder>noise</system-reminder>')
    assert clean_prompt(raw) == "幫我看部署為什麼失敗"


def test_prefetch_injects_and_dedups_within_session(tmp_path):
    c, _ = _client(tmp_path, [("alpha", "系統", 0.9), ("low", "系統", 0.2)])
    payload = {"prompt": "請解釋 alpha 模組的設計理由", "session_id": "s1"}
    out = c.prefetch(payload)
    assert "**alpha**" in out and "low" not in out
    assert c.prefetch(payload) is None  # 同 session 不重複注入


def test_prefetch_skips_short_and_slash_prompts(tmp_path):
    c, urls = _client(tmp_path, [("alpha", "系統", 0.9)])
    assert c.prefetch({"prompt": "ok", "session_id": "s"}) is None
    assert c.prefetch({"prompt": "/commit something long enough", "session_id": "s"}) is None
    assert urls == []


def test_edit_recall_only_failure_modes_and_once_per_file(tmp_path):
    c, _ = _client(tmp_path, [("design-page", "系統", 0.9), ("lesson", "failure-modes", 0.6)])
    p = {"session_id": "s", "hook_event_name": "PostToolUse", "tool_name": "Edit",
         "tool_input": {"file_path": "C:\\repo\\svc\\store.py", "new_string": "x = 1"}}
    out = c.on_edit(p)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "lesson" in ctx and "design-page" not in ctx
    assert c.on_edit(p) is None


def test_edit_ignores_non_code_and_wiki_paths(tmp_path):
    c, urls = _client(tmp_path, [("lesson", "failure-modes", 0.9)])
    for fp in ("notes.md", "/kb/wiki/x.py"):
        assert c.on_edit({"session_id": "s", "tool_input": {"file_path": fp}}) is None
    assert urls == []


def test_bash_gate_blocks_once_then_cools_down(tmp_path):
    t = [1000.0]
    c, _ = _client(tmp_path, [("lesson", "failure-modes", 0.7)], clock=lambda: t[0])
    p = {"session_id": "s", "tool_input": {"command": 'git commit -m "fix store"'}}
    out = c.on_bash(p)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    t[0] += 60
    assert c.on_bash(p) is None  # 冷卻中放行
    assert c.on_bash({"session_id": "s", "tool_input": {"command": "ls"}}) is None


def test_service_down_is_silent(tmp_path):
    def boom(url):
        raise OSError("refused")

    c = RecallClient(RecallConfig(), state_dir=tmp_path, fetch=boom)
    assert c.prefetch({"prompt": "這段設計為什麼這樣寫呢", "session_id": "s"}) is None


def test_hook_cli_never_fails_on_garbage_stdin(tmp_path):
    r = CliRunner().invoke(cli.app, ["hook", "decision-point", "--config", str(tmp_path / "none.yml")],
                           input="not json")
    assert r.exit_code == 0 and r.output == ""


def test_hook_cli_end_to_end_emits_recall(tmp_path, monkeypatch):
    """守住 fail-safe 外殼：寬 except 不能把整條正常路徑吞掉。真的 Edit payload 必須產出注入。"""
    monkeypatch.setattr(RecallClient, "_http_get",
                        lambda self, url: _results(("lesson", "failure-modes", 0.8)))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    payload = {"session_id": "cli", "hook_event_name": "PostToolUse", "tool_name": "Edit",
               "tool_input": {"file_path": "/repo/svc/store.py", "new_string": "x = 1"}}
    r = CliRunner().invoke(cli.app, ["hook", "decision-point", "--config", str(tmp_path / "none.yml")],
                           input=json.dumps(payload))
    assert r.exit_code == 0
    out = json.loads(r.output)
    assert "lesson" in out["hookSpecificOutput"]["additionalContext"]


def test_is_hit_is_rank_based():
    res = [{"concept_name": "a", "score": 0.1}, {"concept_name": "b"}, {"concept_name": "c"},
           {"concept_name": "target", "score": 0.99}]
    assert is_hit("a", res, 3)  # 低分但排第一 → 命中
    assert not is_hit("target", res, 3)  # 高分但排第四 → 未命中


def test_evaluate_hit_rate_reports_misses():
    class Svc:
        def search(self, q, k):
            name = q.split(" ")[0]
            return {"results": [{"concept_name": name}]} if name != "lost" else {"results": []}

    r = evaluate_hit_rate(Svc(), ["found", "lost"], sample=2, seed=1)
    assert r["hits"] == 1 and r["misses"] == ["lost"] and r["hit_rate"] == 0.5
