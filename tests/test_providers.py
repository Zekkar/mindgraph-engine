import pytest
import unittest.mock as mock
from mindgraph.providers.base import LLMProvider, EmbeddingProvider


def test_llm_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()


def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_concrete_llm_provider():
    class MyLLM(LLMProvider):
        def complete(self, prompt, **kwargs):
            return "response"
    assert MyLLM().complete("hello") == "response"


def test_concrete_embedding_provider():
    class MyEmb(EmbeddingProvider):
        def embed(self, text):
            return [0.1] * 1536
        @property
        def dimension(self):
            return 1536
    p = MyEmb()
    assert len(p.embed("test")) == 1536
    assert p.dimension == 1536


def test_gemini_embedding_calls_api():
    # new google.genai SDK: patch the Client bound in the gemini module
    with mock.patch("mindgraph.providers.gemini.genai.Client") as MockClient:
        inst = MockClient.return_value
        inst.models.embed_content.return_value.embeddings = [
            mock.MagicMock(values=[0.1] * 768)
        ]
        from mindgraph.providers.gemini import GeminiEmbeddingProvider
        p = GeminiEmbeddingProvider(api_key="fake", model="gemini-embedding-001")
        result = p.embed("test text")
        assert len(result) == 768
        inst.models.embed_content.assert_called_once()


def test_gemini_llm_calls_api():
    # new google.genai SDK: patch the Client bound in the gemini module
    with mock.patch("mindgraph.providers.gemini.genai.Client") as MockClient:
        inst = MockClient.return_value
        inst.models.generate_content.return_value.text = "generated text"
        from mindgraph.providers.gemini import GeminiLLMProvider
        p = GeminiLLMProvider(api_key="fake", model="gemini-2.0-flash")
        assert p.complete("prompt") == "generated text"


def test_anthropic_llm_calls_api():
    # patch Anthropic as bound (aliased anthropic_lib) in the anthropic module
    with mock.patch("mindgraph.providers.anthropic.anthropic_lib.Anthropic") as MockClient:
        inst = MockClient.return_value
        inst.messages.create.return_value.content = [
            mock.MagicMock(type="text", text="claude answer")
        ]
        from mindgraph.providers.anthropic import AnthropicLLMProvider
        p = AnthropicLLMProvider(api_key="sk-ant-fake", model="claude-sonnet-4-6")
        assert p.complete("question") == "claude answer"


def test_openai_embedding_calls_api():
    # patch OpenAI as bound in the openai module (aliased as openai_lib)
    with mock.patch("mindgraph.providers.openai.openai_lib.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.embeddings.create.return_value.data = [
            mock.MagicMock(embedding=[0.2] * 1536)
        ]
        from mindgraph.providers.openai import OpenAIEmbeddingProvider
        p = OpenAIEmbeddingProvider(api_key="sk-fake", model="text-embedding-3-small")
        result = p.embed("test")
        assert len(result) == 1536


def test_openai_llm_calls_api():
    # patch OpenAI as bound in the openai module (aliased as openai_lib)
    with mock.patch("mindgraph.providers.openai.openai_lib.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create.return_value.choices = [
            mock.MagicMock(message=mock.MagicMock(content="answer"))
        ]
        from mindgraph.providers.openai import OpenAILLMProvider
        p = OpenAILLMProvider(api_key="sk-fake", model="gpt-4o-mini")
        result = p.complete("question")
        assert result == "answer"


# --- provider factory dispatch ---

def _cfg(llm_provider, llm_key_env, emb_provider="gemini", emb_key_env="GEMINI_API_KEY"):
    from mindgraph.config import MindGraphConfig, LLMConfig, EmbeddingConfig
    return MindGraphConfig(
        domain="t",
        llm=LLMConfig(llm_provider, "m", llm_key_env),
        embedding=EmbeddingConfig(emb_provider, "m", emb_key_env),
    )


def test_factory_dispatches_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    with mock.patch("mindgraph.providers.anthropic.anthropic_lib.Anthropic"):
        from mindgraph.providers import get_llm_provider
        from mindgraph.providers.anthropic import AnthropicLLMProvider
        p = get_llm_provider(_cfg("anthropic", "ANTHROPIC_API_KEY"))
        assert isinstance(p, AnthropicLLMProvider)


def test_factory_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from mindgraph.providers import get_llm_provider
    with pytest.raises(EnvironmentError):
        get_llm_provider(_cfg("openai", "OPENAI_API_KEY"))


def test_factory_unsupported_provider(monkeypatch):
    monkeypatch.setenv("X_KEY", "v")
    from mindgraph.providers import get_llm_provider
    with pytest.raises(ValueError):
        get_llm_provider(_cfg("bogus", "X_KEY"))


def test_embedding_factory_rejects_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    from mindgraph.providers import get_embedding_provider
    with pytest.raises(ValueError):
        get_embedding_provider(_cfg("anthropic", "ANTHROPIC_API_KEY",
                                    emb_provider="anthropic", emb_key_env="ANTHROPIC_API_KEY"))
