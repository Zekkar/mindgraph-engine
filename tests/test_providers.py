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
    # patch the genai alias as bound in the gemini module
    with mock.patch("mindgraph.providers.gemini.genai.embed_content") as m:
        m.return_value = {"embedding": [0.1] * 768}
        from mindgraph.providers.gemini import GeminiEmbeddingProvider
        p = GeminiEmbeddingProvider(api_key="fake", model="models/gemini-embedding-001")
        result = p.embed("test text")
        assert len(result) == 768
        m.assert_called_once()


def test_gemini_llm_calls_api():
    # patch GenerativeModel as bound in the gemini module
    with mock.patch("mindgraph.providers.gemini.genai.GenerativeModel") as MockModel:
        instance = MockModel.return_value
        instance.generate_content.return_value.text = "generated text"
        from mindgraph.providers.gemini import GeminiLLMProvider
        p = GeminiLLMProvider(api_key="fake", model="gemini-pro")
        result = p.complete("prompt")
        assert result == "generated text"


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
