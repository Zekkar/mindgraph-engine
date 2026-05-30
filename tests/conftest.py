import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def tmp_raw_dir(tmp_path):
    (tmp_path / "raw" / "notes").mkdir(parents=True)
    (tmp_path / "wiki").mkdir()
    return tmp_path


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.complete.return_value = "# Test Page\n\n## Core Conclusions\n\nTest content."
    return m


@pytest.fixture
def mock_embedding():
    m = MagicMock()
    m.embed.return_value = [0.1] * 768
    m.dimension = 768
    return m


@pytest.fixture
def sample_config_yaml(tmp_path):
    cfg = tmp_path / "mindgraph.yml"
    cfg.write_text("""
domain: test
language: en
llm:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
embedding:
  provider: openai
  model: text-embedding-3-small
  api_key_env: OPENAI_API_KEY
database:
  host: localhost
  port: 5432
  name: mindgraph_test
server:
  rest_port: 8401
  mcp_port: 8400
  cors_origins: ["*"]
""", encoding="utf-8")
    return cfg
