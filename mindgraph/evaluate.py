"""
檢索命中率評估（`mindgraph eval`）

隨機抽 N 個 wiki 概念，以「{概念名} 是什麼」查意圖路由搜尋，看該概念是否出現在前 top_n 名。
**以排名判定，不用絕對分數門檻**：時間遞減會改變分數尺度，固定分數門檻會把找得到的
舊頁判成失敗（不變式 I-6）。

結果可寫入 wiki.eval_history，供週與週比較、抓檢索品質靜默退化。
參考實作的 λ 自動校準刻意不移植：上線後只跑過 1 次、沒有證據顯示它改善了什麼。
"""
from __future__ import annotations

import random
from datetime import date


def is_hit(concept: str, results: list[dict], top_n: int = 3) -> bool:
    """判斷目標概念是否出現在搜尋結果前 top_n 名（看排名、不看分數）。"""
    target = concept.lower()
    return any((r.get("concept_name") or "").lower() == target for r in results[:top_n])


def evaluate_hit_rate(smart_service, concepts: list[str], sample: int = 20, top_n: int = 3,
             seed: int | None = None, query_template: str = "{concept} 是什麼") -> dict:
    """對抽樣概念逐一自我查詢，回傳命中率與未命中清單（未命中清單是最有用的除錯線索）。"""
    rng = random.Random(seed)
    picked = rng.sample(concepts, min(sample, len(concepts)))
    misses = []
    hits = 0
    for c in picked:
        try:
            res = smart_service.search(query_template.format(concept=c), top_n)
            ok = is_hit(c, res.get("results", []), top_n)
        except Exception:
            ok = False
        if ok:
            hits += 1
        else:
            misses.append(c)
    total = len(picked)
    return {
        "eval_date": date.today().isoformat(),
        "total_q": total,
        "hits": hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "top_n": top_n,
        "misses": misses,
    }


def save_hit_rate(dsn: str, result: dict) -> None:
    """把評估結果 upsert 進 wiki.eval_history（同一天重跑覆蓋）。"""
    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.execute(
            """
            INSERT INTO wiki.eval_history (eval_date, hit_rate, total_q, hits, top_n)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (eval_date) DO UPDATE
            SET hit_rate = EXCLUDED.hit_rate, total_q = EXCLUDED.total_q,
                hits = EXCLUDED.hits, top_n = EXCLUDED.top_n, created_at = now()
            """,
            (result["eval_date"], result["hit_rate"], result["total_q"],
             result["hits"], result["top_n"]),
        )
