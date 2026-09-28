from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from mindgraph.recency import DEFAULT_BASE_LAMBDA, DEFAULT_FLOOR, FAILURE_CATEGORIES


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
    # Default to no CORS: a wildcard "*" would let any visited website read the
    # whole local knowledge base via the browser. Operators opt in explicitly.
    cors_origins: list[str] = field(default_factory=list)


@dataclass
class RecencyConfig:
    """時間遞減設定：同主題較新的知識排前面。

    category_ratio 的 key 是 wiki 資料夾名稱（＝知識種類），值是相對 base_lambda 的倍率；
    逐 key 覆蓋內建預設（失敗模式/架構決策類不衰減）。evergreen 列出永不衰減的概念名。
    """
    enabled: bool = True
    base_lambda: float = DEFAULT_BASE_LAMBDA
    floor: float = DEFAULT_FLOOR
    category_ratio: dict[str, float] | None = None
    evergreen: list[str] = field(default_factory=list)


@dataclass
class RecallConfig:
    """決策點主動召回（Claude Code hook）設定。

    api_url 指向 `mindgraph serve` 的 REST 位址；failure_categories 是哪些 wiki 資料夾
    算「失敗模式／教訓」頁——決策點只帶入這些頁，避免注入洗版（不變式 I-14）。
    """
    api_url: str = "http://127.0.0.1:8401"
    failure_categories: list[str] = field(default_factory=lambda: list(FAILURE_CATEGORIES))
    max_inject: int = 6
    min_score: float = 0.55
    decision_min_score: float = 0.5
    gate_cooldown_s: int = 1800
    timeout_s: float = 3.0


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
    recency: RecencyConfig = field(default_factory=RecencyConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)

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
        rec_fields = set(RecencyConfig.__dataclass_fields__)
        recency = RecencyConfig(**{k: v for k, v in (data.get("recency") or {}).items()
                                   if k in rec_fields})
        rc_fields = set(RecallConfig.__dataclass_fields__)
        recall = RecallConfig(**{k: v for k, v in (data.get("recall") or {}).items()
                                 if k in rc_fields})

        return cls(
            domain=data["domain"],
            language=data.get("language", "en"),
            llm=llm,
            embedding=emb,
            database=db,
            server=srv,
            adapters=adapters,
            intent_patterns=data.get("intent_patterns", {}),
            recency=recency,
            recall=recall,
        )
