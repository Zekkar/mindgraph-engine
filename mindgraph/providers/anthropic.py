from __future__ import annotations
import anthropic as anthropic_lib
from mindgraph.providers.base import LLMProvider

DEFAULT_MAX_TOKENS = 4096


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude 文字生成供應商（Messages API）。

    負責 ingest pipeline 中把 raw/ 原始筆記重寫成結構化 wiki 頁面，
    支援 claude-sonnet / claude-haiku / claude-opus 系列。Anthropic 不提供 embedding API，
    故僅實作 LLMProvider；向量嵌入仍由 Gemini 或 OpenAI 供應商負責。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self._client = anthropic_lib.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
