import asyncio

import pytest
from fastapi.testclient import TestClient

from mindgraph.graph import WikiGraphEngine
from mindgraph.search import SmartSearchService
from mindgraph.server.rest import create_app
from mindgraph.server.mcp_server import create_mcp


def _make_wiki(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "alpha.md").write_text(
        "---\ntags: [topic]\nrelated: [beta]\n---\n# Alpha\n\n## Core\n\n"
        "Alpha connects to beta and explains the core idea.\n",
        encoding="utf-8",
    )
    (wiki / "beta.md").write_text(
        "---\ntags: [topic]\n---\n# Beta\n\nBeta is the second concept.\n",
        encoding="utf-8",
    )
    return wiki


@pytest.fixture
def client(tmp_path):
    _make_wiki(tmp_path)
    engine = WikiGraphEngine(str(tmp_path / "wiki"))
    engine.build()
    smart = SmartSearchService(graph_engine=engine, vector_retriever=None, llm_provider=None)
    return TestClient(create_app(engine, smart))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["pages"] == 2


def test_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json()["total_pages"] == 2


def test_keyword_search(client):
    r = client.get("/api/search", params={"q": "alpha", "limit": 5})
    assert r.status_code == 200
    assert "alpha" in [x["name"] for x in r.json()["results"]]


def test_smart_search(client):
    r = client.get("/api/smart_search", params={"q": "alpha core 是什麼"})
    assert r.status_code == 200
    body = r.json()
    assert "intent" in body
    # vector unavailable in test → degrades to keyword+graph
    assert body["degradation"] == "vector_unavailable_fallback_to_keyword"


def test_concept_found_and_404(client):
    assert client.get("/api/concept/alpha").status_code == 200
    assert client.get("/api/concept/does-not-exist").status_code == 404


def test_related(client):
    r = client.get("/api/related/alpha", params={"depth": 1})
    assert r.status_code == 200
    names = [n["name"] for n in r.json()["nodes"]]
    assert "beta" in names  # EXTRACTED edge via related: [beta]


def test_smart_search_degrades_without_service(tmp_path):
    _make_wiki(tmp_path)
    engine = WikiGraphEngine(str(tmp_path / "wiki"))
    engine.build()
    client = TestClient(create_app(engine, smart_service=None))
    r = client.get("/api/smart_search", params={"q": "alpha"})
    assert r.status_code == 200
    assert r.json()["degradation"] == "smart_search_unavailable"


def test_mcp_registers_expected_tools(tmp_path):
    _make_wiki(tmp_path)
    engine = WikiGraphEngine(str(tmp_path / "wiki"))
    engine.build()
    server = create_mcp(engine)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"search", "smart_search", "get_concept", "related",
            "communities", "stats", "god_nodes"} <= names
