from __future__ import annotations
import google.generativeai as genai
from mindgraph.providers.base import LLMProvider, EmbeddingProvider

EMBED_DIM = 768


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        genai.configure(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        resp = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=EMBED_DIM,
        )
        return resp["embedding"]

    @property
    def dimension(self) -> int:
        return EMBED_DIM


class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str, **kwargs) -> str:
        resp = self._model.generate_content(prompt)
        return resp.text or ""
