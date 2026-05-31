import pytest
from mindgraph.config import MindGraphConfig


def test_load_config(sample_config_yaml):
    cfg = MindGraphConfig.from_yaml(sample_config_yaml)
    assert cfg.domain == "test"
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.embedding.provider == "openai"
    assert cfg.server.rest_port == 8401
    assert cfg.language == "en"


def test_config_defaults(tmp_path):
    cfg_file = tmp_path / "mindgraph.yml"
    cfg_file.write_text(
        "domain: myapp\n"
        "llm:\n  provider: gemini\n  model: gemini-pro\n  api_key_env: GEMINI_API_KEY\n"
        "embedding:\n  provider: gemini\n  model: gemini-embedding-001\n  api_key_env: GEMINI_API_KEY\n",
        encoding="utf-8"
    )
    cfg = MindGraphConfig.from_yaml(cfg_file)
    assert cfg.server.rest_port == 8401  # default
    assert cfg.server.cors_origins == []  # secure default: no CORS unless explicitly opted in
    assert cfg.language == "en"  # default


def test_config_intent_patterns(tmp_path):
    cfg_file = tmp_path / "mindgraph.yml"
    cfg_file.write_text(
        "domain: test\n"
        "llm:\n  provider: openai\n  model: gpt-4o-mini\n  api_key_env: OPENAI_API_KEY\n"
        "embedding:\n  provider: openai\n  model: text-embedding-3-small\n  api_key_env: OPENAI_API_KEY\n"
        "intent_patterns:\n  failure: ['crash', 'error']\n",
        encoding="utf-8"
    )
    cfg = MindGraphConfig.from_yaml(cfg_file)
    assert "crash" in cfg.intent_patterns.get("failure", [])
