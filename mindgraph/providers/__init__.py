from __future__ import annotations
import os
from mindgraph.config import MindGraphConfig
from mindgraph.providers.base import LLMProvider, EmbeddingProvider


def get_llm_provider(config: MindGraphConfig) -> LLMProvider:
    p = config.llm.provider
    key = os.environ.get(config.llm.api_key_env, "")
    if not key:
        raise EnvironmentError(
            f"Environment variable '{config.llm.api_key_env}' is not set. "
            f"Required for LLM provider '{p}'."
        )
    if p == "gemini":
        from mindgraph.providers.gemini import GeminiLLMProvider
        return GeminiLLMProvider(api_key=key, model=config.llm.model)
    if p == "openai":
        from mindgraph.providers.openai import OpenAILLMProvider
        return OpenAILLMProvider(api_key=key, model=config.llm.model)
    if p == "anthropic":
        from mindgraph.providers.anthropic import AnthropicLLMProvider
        return AnthropicLLMProvider(api_key=key, model=config.llm.model)
    raise ValueError(f"Unsupported LLM provider: '{p}'. Supported: gemini, openai, anthropic")


def get_embedding_provider(config: MindGraphConfig) -> EmbeddingProvider:
    p = config.embedding.provider
    key = os.environ.get(config.embedding.api_key_env, "")
    if not key:
        raise EnvironmentError(
            f"Environment variable '{config.embedding.api_key_env}' is not set. "
            f"Required for embedding provider '{p}'."
        )
    if p == "gemini":
        from mindgraph.providers.gemini import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider(api_key=key, model=config.embedding.model)
    if p == "openai":
        from mindgraph.providers.openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(api_key=key, model=config.embedding.model)
    raise ValueError(f"Unsupported embedding provider: '{p}'. Supported: gemini, openai")
