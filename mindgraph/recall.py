"""
決策點主動召回（Claude Code hook 用；`mindgraph hook prefetch|decision-point`）

知識最常見的失效形態不是「找不到」，而是「找得到但沒在做決定時出現」。只在使用者提問時
檢索一次，攔不到執行途中的錯誤。因此在三個時點主動推送：

1. prefetch（UserPromptSubmit）：使用者訊息 → 混合檢索 → 注入 <wiki-context>
2. decision-point edit（PostToolUse Edit/Write）：某程式檔在 session 內第一次被改 →
   只帶入相關**失敗模式**頁
3. decision-point bash（PreToolUse Bash）：git commit / push / 部署 → **擋一次**並附上
   相關失敗模式，模型確認後重送即放行（同類決策點冷卻時間內不再擋）

政策（不變式 I-14）：只帶失敗模式頁、限量、同 session 去重，否則注入洗版會讓 AI 忽略全部。
fail-safe：服務不通或任何例外 → 靜默放行，絕不阻擋使用者或工具。
只用標準函式庫呼叫 REST，hook 啟動不載入重依賴。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# harness 包裝：整段移除（內容不是使用者寫的）
_DROP_BLOCKS = [
    r"<system-reminder>.*?</system-reminder>",
    r"<task-notification>.*?</task-notification>",
    r"<local-command-stdout>.*?</local-command-stdout>",
    r"<local-command-caveat>.*?</local-command-caveat>",
    r"<command-(?:name|message|args)>.*?</command-(?:name|message|args)>",
]
# 通道外層標籤：只去掉標籤（含屬性），保留正文——屬性值會污染查詢詞
_UNWRAP_TAGS = r"</?(?:channel|pasted_content)\b[^>]*>"

CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".sql",
            ".yml", ".yaml", ".sh", ".ps1", ".toml", ".conf"}
SKIP_PATH_PARTS = ("/raw/", "/wiki/", "/node_modules/", "/.git/", "/scratchpad/")

BASH_DECISIONS = [
    ("commit", re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?commit\b")),
    ("push", re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?push\b")),
    ("deploy", re.compile(r"docker[\s-]compose\b[^|;&]*\bup\b|\bkubectl\s+apply\b|\bdeploy\b")),
]
BASH_QUERY_HINT = {
    "commit": "commit unverified half-done 提交前 未驗證",
    "push": "push deploy verify 推送 部署 驗證",
    "deploy": "deploy health check regression 部署 健康檢查 回歸",
}
MIN_PROMPT_LEN = 12


def is_decision_command(cmd: str) -> bool:
    """Bash 指令是否屬於決策點（commit / push / 部署）；hook 用它在載入設定前快速放行。"""
    return any(pat.search(cmd) for _, pat in BASH_DECISIONS)


def clean_prompt(text: str) -> str:
    """去掉 harness 包裝與通道標籤，只留使用者真正寫的內容作為查詢詞。"""
    if not isinstance(text, str):
        return ""
    t = text
    for pat in _DROP_BLOCKS:
        t = re.sub(pat, " ", t, flags=re.S)
    t = re.sub(_UNWRAP_TAGS, " ", t)
    return re.sub(r"\s+", " ", t).strip()


class RecallClient:
    """決策點召回的 REST 用戶端＋每 session 狀態（去重、冷卻、已改檔清單）。"""

    def __init__(self, cfg, state_dir: Path | None = None, fetch=None, clock=time.time):
        self.cfg = cfg  # RecallConfig
        self._fail_cats = frozenset(cfg.failure_categories)
        self.state_dir = Path(state_dir or Path.home() / ".mindgraph" / "state")
        self._fetch = fetch or self._http_get
        self._clock = clock

    # ── 搜尋 ──
    def _http_get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "mindgraph-recall/1"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """呼叫 /api/hybrid_search，回傳 [{name, category, section, snippet, score}]；失敗回 []。"""
        if not query:
            return []
        url = (f"{self.cfg.api_url.rstrip('/')}/api/hybrid_search?"
               f"q={urllib.parse.quote(query[:300])}&top_k={top_k}")
        try:
            data = self._fetch(url)
        except Exception:
            return []
        out = []
        for r in (data.get("results") or []) if isinstance(data, dict) else []:
            if not r.get("concept_name"):
                continue
            chunk = re.sub(r"^#+\s.*\n", "", r.get("chunk_text") or "").strip()
            out.append({
                "name": r["concept_name"],
                "category": r.get("category"),
                "section": r.get("section_id") or "",
                "snippet": re.sub(r"\s+", " ", chunk)[:180],
                "score": float(r.get("score") or 0.0),
            })
        return out

    # ── 狀態 ──
    def _state_file(self, sid: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", sid)[:80] or "unknown"
        return self.state_dir / f"recall-{safe}.json"

    def load_state(self, sid: str) -> dict:
        try:
            st = json.loads(self._state_file(sid).read_text(encoding="utf-8"))
        except Exception:
            st = {}
        st.setdefault("injected", []); st.setdefault("files", []); st.setdefault("gates", {})
        return st

    def save_state(self, sid: str, st: dict) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._state_file(sid).write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def pick(self, hits: list[dict], st: dict, min_score: float, only_failures: bool) -> list[dict]:
        """依分數門檻挑頁：排除本 session 已注入過的、同頁只留一段、可限定只要失敗模式頁。"""
        seen = set(st["injected"])
        picked, names = [], set()
        for h in hits:
            n = h["name"]
            if h["score"] < min_score or n in seen or n in names:
                continue
            if only_failures and h.get("category") not in self._fail_cats:
                continue
            names.add(n)
            picked.append(h)
            if len(picked) >= self.cfg.max_inject:
                break
        st["injected"] = sorted(seen | names)
        return picked

    # ── 三個時點 ──
    def prefetch(self, payload: dict) -> str | None:
        """UserPromptSubmit：使用者內容夠長才查；回傳要印到 stdout 的 <wiki-context> 或 None。"""
        text = clean_prompt(payload.get("prompt") or "")
        if len(text) < MIN_PROMPT_LEN or text.startswith("/"):
            return None
        sid = payload.get("session_id", "unknown")
        st = self.load_state(sid)
        picked = self.pick(self.hybrid_search(text, 12), st, self.cfg.min_score, False)
        if not picked:
            return None
        self.save_state(sid, st)
        body = self._render(picked, "MindGraph 預取（本 session 已注入過的頁不重複）：",
                            tag_failures=True)
        return f"<wiki-context>\n{body}\n⚠️ 只是段落摘要；引用前請 get_concept 讀全文。\n</wiki-context>"

    def on_edit(self, payload: dict) -> dict | None:
        """PostToolUse(Edit|Write)：程式檔在本 session 第一次被改時，注入相關失敗模式。"""
        ti = payload.get("tool_input") or {}
        fp = str(ti.get("file_path", "")).replace("\\", "/")
        if not fp or os.path.splitext(fp)[1].lower() not in CODE_EXT:
            return None
        if any(part in fp for part in SKIP_PATH_PARTS):
            return None
        sid = payload.get("session_id", "unknown")
        st = self.load_state(sid)
        if fp in st["files"]:
            return None
        st["files"].append(fp)
        parts = fp.split("/")
        change = re.sub(r"\s+", " ", (ti.get("new_string") or ti.get("content") or "")[:1000])[:200]
        query = f"{' '.join(parts[-3:-1])} {os.path.splitext(parts[-1])[0]} {change}"
        picked = self.pick(self.hybrid_search(query, 15), st, self.cfg.decision_min_score, True)
        self.save_state(sid, st)
        if not picked:
            return None
        return {"hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": self._render(
                picked, f"<decision-point-recall> 你剛開始修改 `{parts[-1]}`，過去相關的失敗模式：")
            + "\n</decision-point-recall>",
        }}

    def on_bash(self, payload: dict) -> dict | None:
        """PreToolUse(Bash)：commit / push / 部署第一次出現時擋一次並附上相關失敗模式。"""
        cmd = str((payload.get("tool_input") or {}).get("command", ""))
        kind = next((k for k, pat in BASH_DECISIONS if pat.search(cmd)), None)
        if not kind:
            return None
        sid = payload.get("session_id", "unknown")
        st = self.load_state(sid)
        now = self._clock()
        last = st["gates"].get(kind)
        if last is not None and now - last < self.cfg.gate_cooldown_s:
            return None
        st["gates"][kind] = now
        touched = " ".join(os.path.splitext(f.split("/")[-1])[0] for f in st["files"][-5:])
        msg = re.search(r"-m\s+[\"']([^\"']{0,160})", cmd)
        query = f"{BASH_QUERY_HINT[kind]} {touched} {msg.group(1) if msg else ''}"
        picked = self.pick(self.hybrid_search(query, 15), st, self.cfg.decision_min_score, True)
        self.save_state(sid, st)
        if not picked:
            return None
        reason = self._render(picked, f"決策點召回（{kind}）：執行前先對照這些相關失敗模式——") + \
            "\n確認已避開後重送同一指令即放行。"
        return {
            "decision": "block",
            "reason": reason,
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
        }

    def _render(self, picked: list[dict], header: str, tag_failures: bool = False) -> str:
        """把挑中的頁排成編號清單；tag_failures 時在失敗模式頁前加標記。"""
        lines = [header]
        for i, h in enumerate(picked, 1):
            tag = "［失敗模式］" if tag_failures and h.get("category") in self._fail_cats else ""
            lines.append(f"{i}. {tag}**{h['name']}** (score={h['score']:.2f}) § {h['section']} — {h['snippet'][:140]}")
        return "\n".join(lines)
