from __future__ import annotations
from google import genai
from google.genai import types
from mindgraph.providers.base import LLMProvider, EmbeddingProvider

EMBED_DIM = 768


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini 向量嵌入供應商（google.genai 新 SDK）。

    把 wiki 頁面區塊文字轉成 768 維向量，供 EmbeddingStore 寫入 pgvector、
    讓 VectorRetriever 以 cosine 相似度做語意檢索。取代已棄用的 google.generativeai。
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM,
            ),
        )
        return list(resp.embeddings[0].values)

    @property
    def dimension(self) -> int:
        return EMBED_DIM


class GeminiLLMProvider(LLMProvider):
    """Gemini 文字生成供應商（google.genai 新 SDK）。

    負責 ingest pipeline 中把 raw/ 原始筆記重寫成結構化 wiki 頁面（含 ## Core Conclusions），
    以及 IntentRouter Pass 2 的查詢擴展。取代已棄用的 google.generativeai。
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, **kwargs) -> str:
        resp = self._client.models.generate_content(model=self._model, contents=prompt)
        return resp.text or ""
