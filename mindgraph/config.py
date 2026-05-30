from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key_env: str


@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    api_key_env: str


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "mindgraph"
    user: str = "mindgraph"
    password_env: str = "MINDGRAPH_DB_PASSWORD"


@dataclass
class ServerConfig:
    rest_port: int = 8401
    mcp_port: int = 8400
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class AdapterConfig:
    type: str
    source: str = ""
    target: str = "raw/notes"
    module: str = ""


@dataclass
class MindGraphConfig:
    domain: str
    llm: LLMConfig
    embedding: EmbeddingConfig
    language: str = "en"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    adapters: list[AdapterConfig] = field(default_factory=list)
    intent_patterns: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path) -> "MindGraphConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        llm = LLMConfig(**data["llm"])
        emb = EmbeddingConfig(**data["embedding"])

        db_data = data.get("database", {})
        db_fields = set(DatabaseConfig.__dataclass_fields__)
        db = DatabaseConfig(**{k: v for k, v in db_data.items() if k in db_fields}) if db_data else DatabaseConfig()

        srv_data = data.get("server", {})
        srv_fields = set(ServerConfig.__dataclass_fields__)
        srv = ServerConfig(**{k: v for k, v in srv_data.items() if k in srv_fields}) if srv_data else ServerConfig()

        adapters = [AdapterConfig(**a) for a in data.get("adapters", [])]

        return cls(
            domain=data["domain"],
            language=data.get("language", "en"),
            llm=llm,
            embedding=emb,
            database=db,
            server=srv,
            adapters=adapters,
            intent_patterns=data.get("intent_patterns", {}),
        )
