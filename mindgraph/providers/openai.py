from __future__ import annotations
import openai as openai_lib
from mindgraph.providers.base import LLMProvider, EmbeddingProvider

_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._client = openai_lib.OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding

    @property
    def dimension(self) -> int:
        return _MODEL_DIMS.get(self._model, 1536)


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = openai_lib.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""
