import importlib
from mindgraph.config import MindGraphConfig
from mindgraph.adapters.base import DataSourceAdapter


def get_adapters(config: MindGraphConfig) -> list[DataSourceAdapter]:
    result = []
    for a in config.adapters:
        if a.type == "filesystem":
            from mindgraph.adapters.filesystem import FileSystemAdapter
            result.append(FileSystemAdapter(source=a.source, target=a.target))
        elif a.type == "custom":
            parts = a.module.rsplit(".", 1)
            mod = importlib.import_module(parts[0])
            result.append(getattr(mod, parts[1])())
    return result
