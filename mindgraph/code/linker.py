"""MindGraph 程式碼 ↔ wiki 跨層連結器。

把程式碼圖中的 symbol 連到 wiki 知識層的概念，兩種方法：

- name_match（字面比對，零外部呼叫）：把 symbol 短名與 wiki 概念名拆成英文 token
  （CamelCase / snake_case / kebab-case），以 max-containment
  `max(|A∩B|/|A|, |A∩B|/|B|)` 評分，≥ 0.5 即收。只能抓到撞名，語意鑑別力低。
- embedding_match（語意比對）：以 `"{short_name}: {docstring[:800]}"` 產生向量，
  對 wiki.embeddings（相同 model_version）取最近 30 個章節，每個概念取最高相似度，
  **原始相似度 ≥ 0.72** 才收，每個 symbol 最多 3 個概念；最終分數 = 相似度 ×
  recency_fn(概念)（未提供時乘 1）。只處理 docstring 長度 > 50 字元的 symbol。

寫入採破壞性重建（不變式 I-8）：先算完全部候選，再於單一交易內刪除該 repo 既有
連結並寫入，同一 (symbol, 概念) 衝突時保留較高分者。
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Optional

from mindgraph.code.schema import (
    LINK_EMBEDDING_MATCH,
    LINK_NAME_MATCH,
    atomic,
    ensure_code_schema,
)

logger = logging.getLogger("mindgraph.code.linker")

NAME_MATCH_THRESHOLD = 0.5
EMBEDDING_THRESHOLD = 0.72
EMBEDDING_TOP_K = 3
EMBEDDING_NEIGHBOURS = 30
MIN_DOCSTRING_CHARS = 50
DOCSTRING_EMBED_CHARS = 800

VALID_METHODS = ("all", "name_only", "embedding_only")

RecencyFn = Callable[[str], float]
# (symbol_id, concept_name, link_type, score)
LinkRow = tuple[int, str, str, float]


# ---------------------------------------------------------------------------
# 純函式：name_match
# ---------------------------------------------------------------------------

def tokenize_name(name: str) -> set[str]:
    """把 symbol 名稱或 wiki 概念名稱拆成小寫英文 token 集合，供 name_match 比對。

    支援 CamelCase（`CodeIndexer` → code, indexer）、連續大寫縮寫
    （`HTTPServer` → http, server）、snake_case 與 kebab-case；先移除版本字樣
    （v2、v1.3），只保留長度 ≥ 3 的 token 以壓低 get/set/id 之類的撞名雜訊。
    非 ASCII 字元（例如中文概念名）不產生 token。
    """
    name = re.sub(r"\bv\d[\d.]*\b", "", name, flags=re.IGNORECASE)
    tokens: list[str] = []
    for part in re.findall(r"[a-zA-Z]+", name):
        part = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
        part = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", part)
        tokens.extend(part.split())
    return {t.lower() for t in tokens if len(t) >= 3}


def max_containment(a: set[str], b: set[str]) -> float:
    """計算兩個 token 集合的 max-containment 分數 `max(|A∩B|/|A|, |A∩B|/|B|)`。

    用於 name_match：只要一方名稱幾乎被另一方包含就給高分（例如 `WikiLinker`
    對 `wiki-linker-design`）。任一方為空或無交集時回傳 0。
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return max(inter / len(a), inter / len(b)) if inter else 0.0


def name_matches(
    symbols: Iterable[tuple[int, str]],
    concepts: Iterable[str],
    threshold: float = NAME_MATCH_THRESHOLD,
) -> list[LinkRow]:
    """對 (symbol_id, short_name) 與 wiki 概念名清單做字面 token 比對，產生 name_match 連結候選。

    每一對 max-containment ≥ threshold 即產生一筆 (symbol_id, 概念, 'name_match', 分數)。
    無 token 的 symbol 或概念直接略過。純函式，不觸碰資料庫。
    """
    concept_tokens = [(c, tokenize_name(c)) for c in concepts]
    concept_tokens = [(c, t) for c, t in concept_tokens if t]
    out: list[LinkRow] = []
    for sym_id, short_name in symbols:
        toks = tokenize_name(short_name)
        if not toks:
            continue
        for concept, ctoks in concept_tokens:
            score = max_containment(toks, ctoks)
            if score >= threshold:
                out.append((sym_id, concept, LINK_NAME_MATCH, float(score)))
    return out


# ---------------------------------------------------------------------------
# 純函式：embedding_match
# ---------------------------------------------------------------------------

def embedding_text(short_name: str, docstring: Optional[str]) -> Optional[str]:
    """產生 symbol 送去 embedding 的文字 `"{short_name}: {docstring[:800]}"`。

    docstring 去除首尾空白後長度 ≤ 50 字元時回傳 None：占位符式短 docstring
    對語意比對幾乎沒有訊號，只會產生雜訊連結，因此不送 embedding。
    """
    doc = (docstring or "").strip()
    if len(doc) <= MIN_DOCSTRING_CHARS:
        return None
    return f"{short_name}: {doc[:DOCSTRING_EMBED_CHARS]}"


def rank_concepts(
    rows: Iterable[tuple[str, float]],
    threshold: float = EMBEDDING_THRESHOLD,
    top_k: int = EMBEDDING_TOP_K,
    recency_fn: Optional[RecencyFn] = None,
) -> list[tuple[str, float]]:
    """把單一 symbol 的最近鄰 wiki 章節彙整成要連結的概念排名（embedding_match 核心規則）。

    rows 為 (concept_name, 相似度) 的章節列；同一概念多個章節只取最高相似度。
    門檻看**原始相似度**（≥ threshold），避免較舊的概念因 recency 折扣掉出門檻；
    通過門檻後分數 = 相似度 × recency_fn(concept)（未提供則 ×1），依分數降序
    取前 top_k 個。回傳 [(concept_name, score)]。
    """
    best: dict[str, float] = {}
    for concept, sim in rows:
        sim = float(sim)
        if sim > best.get(concept, float("-inf")):
            best[concept] = sim
    ranked = [
        (concept, sim * (recency_fn(concept) if recency_fn else 1.0))
        for concept, sim in best.items()
        if sim >= threshold
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def merge_links(rows: Iterable[LinkRow]) -> list[LinkRow]:
    """合併多種方法產生的連結候選：同一 (symbol_id, 概念) 只保留分數較高的那一筆。

    分數相同時保留先出現者（呼叫端先放 name_match 再放 embedding_match，同分時 name_match 勝出）。
    寫入前已在此去重，所以 _replace_links 的 INSERT 不需要 ON CONFLICT。
    """
    best: dict[tuple[int, str], LinkRow] = {}
    for row in rows:
        key = (row[0], row[1])
        if key not in best or row[3] > best[key][3]:
            best[key] = row
    return list(best.values())


def _vector_literal(vec: Iterable[float]) -> str:
    """把向量轉成 pgvector 的文字字面值 `[x,y,...]`，不依賴 pgvector 的 psycopg adapter。"""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# ---------------------------------------------------------------------------
# 資料庫存取（拆成小函式，方便測試時替換）
# ---------------------------------------------------------------------------

def _wiki_table_exists(conn) -> bool:
    return conn.execute("SELECT to_regclass('wiki.embeddings') IS NOT NULL").fetchone()[0]


def _fetch_concepts(conn, model_version: Optional[str]) -> list[str]:
    if not _wiki_table_exists(conn):
        return []
    if model_version:
        rows = conn.execute(
            "SELECT DISTINCT concept_name FROM wiki.embeddings WHERE model_version = %s",
            (model_version,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT DISTINCT concept_name FROM wiki.embeddings").fetchall()
    return [r[0] for r in rows]


def _fetch_symbols(conn, repo: str) -> list[tuple[int, str, Optional[str]]]:
    return conn.execute(
        "SELECT id, short_name, docstring FROM code.symbols WHERE repo = %s ORDER BY id",
        (repo,),
    ).fetchall()


def _nearest_sections(conn, vec: list[float], model_version: str) -> list[tuple[str, float]]:
    lit = _vector_literal(vec)
    return conn.execute(
        """SELECT concept_name, 1 - (embedding <=> %s::vector) AS sim
           FROM wiki.embeddings
           WHERE model_version = %s
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (lit, model_version, lit, EMBEDDING_NEIGHBOURS),
    ).fetchall()


def _replace_links(conn, repo: str, links: list[LinkRow]) -> dict[str, int]:
    """單一交易內刪除 repo 既有連結並寫入新連結；回傳被刪除連結的 link_type 分布。"""
    with atomic(conn):
        removed = conn.execute(
            "WITH d AS (DELETE FROM code.wiki_links WHERE symbol_id IN "
            "(SELECT id FROM code.symbols WHERE repo = %s) RETURNING link_type) "
            "SELECT link_type, count(*) FROM d GROUP BY link_type",
            (repo,),
        ).fetchall()
        if links:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO code.wiki_links
                       (symbol_id, wiki_concept_name, link_type, confidence_score)
                       VALUES (%s, %s, %s, %s)""",
                    links,
                )
    return {link_type: n for link_type, n in removed}


# ---------------------------------------------------------------------------
# 對外入口
# ---------------------------------------------------------------------------

def link_wiki(
    conn,
    repo: str,
    embedding_provider=None,
    model_version: Optional[str] = None,
    methods: str = "all",
    recency_fn: Optional[RecencyFn] = None,
) -> dict:
    """重建某個 repo 的程式碼 ↔ wiki 概念連結，是 MindGraph 跨層連結的對外入口。

    參數：
    - embedding_provider：實作 `embed(text) -> list[float]` 的 EmbeddingProvider；
      methods 含 embedding 時必填。
    - model_version：必須與寫入 wiki.embeddings 時相同（見 store.model_version），
      否則查不到任何章節；methods 含 embedding 時必填。name_match 若有提供也只比對
      該版本的概念。
    - methods：`all`（預設）/ `name_only` / `embedding_only`。
    - recency_fn：`concept_name -> 係數` 的時間衰減函式，只作用於 embedding 分數；
      可直接傳入 `mindgraph.recency.RecencyWeighter.factor_for`。

    行為：先計算全部候選，再於單一交易內刪除該 repo 既有連結並寫入（不變式 I-8），
    因此即使只跑 name_only 也會清掉舊的 embedding_match 連結——回傳的 cleared_links
    會列出被刪除連結的類型分布，呼叫端可據此警示語意層被清空。單一 symbol 的
    embedding 失敗只記 warning 並略過。前置條件：repo 已用 index_repository 索引到最新。

    回傳 {repo, methods, model_version, symbols, embedded_symbols, embed_failures,
    total_links, by_link_type: {name_match: n, embedding_match: n},
    cleared_links: {link_type: 被刪除筆數}}。
    """
    if methods not in VALID_METHODS:
        raise ValueError(f"methods 必須是 {VALID_METHODS} 之一，收到 {methods!r}")
    use_name = methods in ("all", "name_only")
    use_embed = methods in ("all", "embedding_only")
    if use_embed and (embedding_provider is None or not model_version):
        raise ValueError("embedding_match 需要 embedding_provider 與 model_version")

    ensure_code_schema(conn)
    symbols = _fetch_symbols(conn, repo)
    candidates: list[LinkRow] = []

    if use_name:
        concepts = _fetch_concepts(conn, model_version)
        candidates.extend(name_matches(((sid, name) for sid, name, _ in symbols), concepts))

    embedded = failures = 0
    if use_embed and _wiki_table_exists(conn):
        for sid, short_name, doc in symbols:
            text = embedding_text(short_name, doc)
            if text is None:
                continue
            try:
                vec = embedding_provider.embed(text)
            except Exception:  # noqa: BLE001 — 單一 symbol 失敗不影響整批
                failures += 1
                logger.warning("embed 失敗 repo=%s symbol=%s", repo, short_name, exc_info=True)
                continue
            embedded += 1
            for concept, score in rank_concepts(
                _nearest_sections(conn, vec, model_version), recency_fn=recency_fn
            ):
                candidates.append((sid, concept, LINK_EMBEDDING_MATCH, float(score)))

    links = merge_links(candidates)
    cleared = _replace_links(conn, repo, links)

    by_type = {LINK_NAME_MATCH: 0, LINK_EMBEDDING_MATCH: 0}
    for row in links:
        by_type[row[2]] = by_type.get(row[2], 0) + 1
    logger.info("link_wiki done repo=%s methods=%s links=%s", repo, methods, by_type)
    return {
        "repo": repo,
        "methods": methods,
        "model_version": model_version,
        "symbols": len(symbols),
        "embedded_symbols": embedded,
        "embed_failures": failures,
        "total_links": len(links),
        "by_link_type": by_type,
        "cleared_links": cleared,
    }
