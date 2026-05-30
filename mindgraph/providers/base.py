from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Single completion call. Returns generated text."""
        ...


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return embedding vector for the given text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension (must match pgvector schema)."""
        ...
