from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RawDocument:
    content: str
    filename: str
    metadata: dict = field(default_factory=dict)


class DataSourceAdapter(ABC):
    @abstractmethod
    def fetch(self) -> list[RawDocument]: ...

    @abstractmethod
    def get_target_dir(self) -> str: ...
