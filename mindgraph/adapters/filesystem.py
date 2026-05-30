from pathlib import Path
from mindgraph.adapters.base import DataSourceAdapter, RawDocument


class FileSystemAdapter(DataSourceAdapter):
    def __init__(self, source: str, target: str = "raw/notes"):
        self._source = Path(source)
        self._target = target

    def fetch(self) -> list[RawDocument]:
        if not self._source.exists():
            return []
        return [
            RawDocument(
                content=p.read_text(encoding="utf-8"),
                filename=p.name,
                metadata={"source_path": str(p)},
            )
            for p in sorted(self._source.rglob("*.md"))
        ]

    def get_target_dir(self) -> str:
        return self._target
