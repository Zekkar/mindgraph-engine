"""
時間遞減（Recency Weighting）— 讓同一主題較新的知識排在前面

    factor = floor + (1 - floor) × exp(-λ × age_days)

- 年齡取 wiki.embeddings.content_changed_at（只在章節內容指紋改變時更新），
  **不是** updated_at；後者每次同步都被刷新，拿來算年齡會讓所有頁永遠 0 天（不變式 I-1）。
- floor（預設 0.5）：衰減只負責排序，不讓高度相關的舊知識整個消失。
- λ = base_lambda × 該知識種類（wiki 資料夾）的倍率；教訓與決策類預設倍率 0（不衰減），
  evergreen 名單上的概念一律不衰減。
- 載入失敗時沿用上一份成功快照，從未成功則不衰減（降級而非中斷檢索）。

所有檢索路徑（CLI / REST / MCP / eval / code linker）共用同一個 RecencyWeighter，
確保時間語意一致（不變式 I-5）。
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

logger = logging.getLogger("mindgraph.recency")

DEFAULT_BASE_LAMBDA = 0.003  # per day；地板以上部分的半衰期約 231 天
DEFAULT_FLOOR = 0.5
# 教訓與決策類不因時間失效（中英資料夾名都收）；失敗種類與 search / recall 共用同一份
FAILURE_CATEGORIES: tuple[str, ...] = ("失敗模式", "failure-modes", "failures")
DECISION_CATEGORIES: tuple[str, ...] = ("架構決策", "decisions", "adr")
DEFAULT_CATEGORY_RATIO: dict[str, float] = {
    c: 0.0 for c in FAILURE_CATEGORIES + DECISION_CATEGORIES
}


def decay_factor(age_days: float, lam: float, floor: float = DEFAULT_FLOOR) -> float:
    """依頁面年齡計算時間遞減係數（介於 floor 與 1 之間），負年齡視為 0。"""
    age = max(0.0, float(age_days))
    return floor + (1.0 - floor) * math.exp(-lam * age)


@dataclass(slots=True)
class PageRecency:
    """單一概念頁的時間資訊：知識種類與內容最後改變時間。"""
    category: str | None
    changed_at: datetime | None


class RecencyWeighter:
    """檢索結果的時間遞減加權器：把「同主題新知識權重較重」套用到 ScoredDoc 清單。

    loader 回傳 {概念名: PageRecency}，結果快取 ttl_s 秒；
    category_ratio 決定各知識種類相對 base_lambda 的倍率（未列出的種類倍率 1），
    evergreen 內的概念一律不衰減。
    """

    def __init__(
        self,
        loader: Callable[[], dict[str, PageRecency]],
        base_lambda: float = DEFAULT_BASE_LAMBDA,
        floor: float = DEFAULT_FLOOR,
        category_ratio: dict[str, float] | None = None,
        evergreen: Iterable[str] = (),
        ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._loader = loader
        self.base_lambda = base_lambda
        self.floor = floor
        # 使用者設定逐 key 覆蓋預設（不整份取代），避免改一個種類就讓教訓頁開始衰減
        self.category_ratio = {**DEFAULT_CATEGORY_RATIO, **(category_ratio or {})}
        self.evergreen = set(evergreen)
        self._ttl_s = ttl_s
        self._clock = clock
        self._now = now
        self._snapshot: dict[str, PageRecency] | None = None
        self._loaded_at: float | None = None

    def lambda_for(self, concept: str, category: str | None) -> float:
        """決定某概念的衰減係數 λ：evergreen 為 0，否則 base × 種類倍率。"""
        if concept in self.evergreen:
            return 0.0
        return self.base_lambda * self.category_ratio.get(category or "", 1.0)

    def _get_snapshot(self) -> dict[str, PageRecency] | None:
        t = self._clock()
        if self._loaded_at is not None and t - self._loaded_at < self._ttl_s:
            return self._snapshot
        try:
            self._snapshot = self._loader()
        except Exception as e:
            logger.warning("recency snapshot load failed, keeping previous: %s", e)
        self._loaded_at = t
        return self._snapshot

    def _decay(self, snap: dict[str, PageRecency], concept: str):
        """回傳 (age_days, λ, factor)；沒有年齡資料時回 None。"""
        info = snap.get(concept)
        if info is None or info.changed_at is None:
            return None
        age = self._age_days(info.changed_at)
        lam = self.lambda_for(concept, info.category)
        return age, lam, decay_factor(age, lam, self.floor)

    def factor_for(self, concept: str) -> float:
        """單一概念目前的時間係數（無年齡資料回 1.0）；供 code↔wiki 連結分數加權使用。"""
        d = self._decay(self._get_snapshot() or {}, concept)
        return d[2] if d else 1.0

    def _age_days(self, changed: datetime) -> float:
        if changed.tzinfo is None:
            changed = changed.replace(tzinfo=timezone.utc)
        return (self._now() - changed).total_seconds() / 86400.0

    def apply(self, docs: Iterable) -> list:
        """就地把時間係數乘進每筆結果的 score，並在 retriever_breakdown 記下
        pre_recency_score / age_days / decay_lambda / recency_factor 供解釋；
        回傳依新分數降序的清單。沒有年齡資料的概念不動分數。"""
        docs = list(docs)
        snap = self._get_snapshot()
        if not snap:
            return docs
        for d in docs:
            decay = self._decay(snap, d.concept_name)
            if decay is None:
                continue
            age, lam, factor = decay
            d.retriever_breakdown["pre_recency_score"] = round(d.score, 4)
            d.retriever_breakdown["age_days"] = round(max(0.0, age), 2)
            d.retriever_breakdown["decay_lambda"] = round(lam, 6)
            d.retriever_breakdown["recency_factor"] = round(factor, 4)
            d.score = d.score * factor
        docs.sort(key=lambda x: x.score, reverse=True)
        return docs


def load_snapshot(dsn: str, model_version: str) -> dict[str, PageRecency]:
    """從向量庫讀出每個概念的知識種類與內容最後改變時間（取各章節最新者）。

    model_version 必須與 embed 寫入端相同（由呼叫端從同一份設定算出，不在此寫死）。
    """
    import psycopg

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            """
            SELECT concept_name, MAX(category), MAX(content_changed_at)
            FROM wiki.embeddings
            WHERE model_version = %s
            GROUP BY concept_name
            """,
            (model_version,),
        ).fetchall()
    return {name: PageRecency(cat, changed) for name, cat, changed in rows}


def make_db_weighter(dsn: str, model_version: str, cfg) -> RecencyWeighter:
    """依 RecencyConfig 建立讀取向量庫的 RecencyWeighter。"""
    return RecencyWeighter(lambda: load_snapshot(dsn, model_version),
                           base_lambda=cfg.base_lambda, floor=cfg.floor,
                           category_ratio=cfg.category_ratio, evergreen=cfg.evergreen)
