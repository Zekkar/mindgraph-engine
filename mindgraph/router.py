from __future__ import annotations
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger("mindgraph.router")


@dataclass
class IntentResult:
    intent: str
    confidence: float


class IntentRouter:
    PATTERNS: dict[str, list[str]] = {
        "location": [
            r"在哪", r"哪個檔案", r"路徑", r"哪裡定義",
            r"哪個.{0,5}class", r"定義在", r"哪個模組", r"哪裡",
        ],
        "concept": [
            r"是什麼", r"解釋", r"原理", r"設計", r"為什麼",
            r"怎麼運作", r"什麼意思", r"定義",
        ],
        "relation": [
            r"關係", r"連結", r"影響", r"相關", r"跟.{0,10}有何",
            r"和.{0,10}的關係", r"怎麼互動",
        ],
        "failure": [
            r"錯誤", r"出錯", r"失敗", r"修復",
            r"Exception", r"exception", r"bug", r"error",
            r"問題", r"怎麼解決",
        ],
    }

    MULTI_CONCEPT_MARKERS = [
        r"跟.{1,15}的", r"和.{1,15}的", r"vs\.?",
        r"或者", r"，.{1,20}，", r".+,.+,.+",
    ]

    def __init__(self, custom_patterns: dict[str, list[str]] | None = None):
        if custom_patterns:
            self.PATTERNS = {**self.PATTERNS, **custom_patterns}

    def classify(self, query: str) -> IntentResult:
        scores: dict[str, int] = {k: 0 for k in self.PATTERNS}
        for intent, patterns in self.PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query, re.IGNORECASE):
                    scores[intent] += 1

        best = max(scores, key=lambda k: scores[k])
        best_score = scores[best]
        if best_score == 0:
            return IntentResult(intent="default", confidence=0.0)

        confidence = min(best_score / 2, 1.0)
        logger.debug("intent=%s conf=%.2f q=%s", best, confidence, query[:50])
        return IntentResult(intent=best, confidence=confidence)

    def needs_expansion(self, top1_score: float, query: str) -> bool:
        if top1_score < 0.5:
            return True
        for marker in self.MULTI_CONCEPT_MARKERS:
            if re.search(marker, query, re.IGNORECASE):
                return True
        return False

    def expand_query(
        self, query: str, initial_results: list[dict], llm_provider
    ) -> list[str]:
        top_concepts = [
            r.get("concept_name", "")
            for r in initial_results[:3]
            if r.get("concept_name")
        ]
        context = "、".join(top_concepts) if top_concepts else "無初始結果"
        prompt = (
            f"你是知識庫搜尋輔助。原始查詢：{query}\n"
            f"初步命中概念：{context}\n"
            f"生成 2-3 個補充查詢（不同角度，每行一個，不加編號）："
        )
        try:
            text = llm_provider.complete(prompt)
            lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
            return [l[:120] for l in lines][:3]
        except Exception as e:
            logger.warning("expand_query failed: %s", e)
            return []
