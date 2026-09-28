"""程式碼索引器的純函式測試（不需資料庫）。"""

from pathlib import Path

import pytest

from mindgraph.code.indexer import (
    build_package_map,
    collect_python_files,
    extract_repository,
    read_git_head,
    resolve_relative_import,
)


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def sample_repo(tmp_path):
    _write(tmp_path, "pkg/__init__.py", '"""範例套件。"""\n')
    _write(tmp_path, "pkg/util.py", '''
def helper(x):
    """把輸入加一，供其他模組呼叫。"""
    return x + 1


def _private():
    return 0
''')
    _write(tmp_path, "pkg/sub/__init__.py", "")
    _write(tmp_path, "pkg/sub/service.py", '''
from typing import TYPE_CHECKING
from ..util import helper
from . import models
import json

if TYPE_CHECKING:
    from pkg.util import _private


class Service:
    """服務類別。"""

    @staticmethod
    def run(value):
        """執行服務。"""
        return helper(value)

    async def go(self):
        return self.run(1)


def main():
    s = Service()
    return models.build()
''')
    _write(tmp_path, "pkg/sub/models.py", "def build():\n    return 1\n")
    _write(tmp_path, "pkg/broken.py", "def oops(:\n")
    _write(tmp_path, "venv/lib.py", "def skipped(): pass\n")
    _write(tmp_path, "node_modules/x.py", "def skipped(): pass\n")
    _write(tmp_path, ".hidden/y.py", "def skipped(): pass\n")
    _write(tmp_path, "tests/test_a.py", "def test_x(): pass\n")
    return tmp_path


def test_collect_skips_default_and_extra_excludes(sample_repo):
    rels = {p.relative_to(sample_repo).as_posix() for p in collect_python_files(sample_repo)}
    assert "pkg/util.py" in rels
    assert "tests/test_a.py" in rels
    assert not any(r.startswith(("venv/", "node_modules/", ".hidden/")) for r in rels)

    rels2 = {p.relative_to(sample_repo).as_posix()
             for p in collect_python_files(sample_repo, exclude=["tests", "*/models.py"])}
    assert "tests/test_a.py" not in rels2
    assert "pkg/sub/models.py" not in rels2


def test_package_map_and_relative_import(sample_repo):
    files = collect_python_files(sample_repo)
    pm = build_package_map(sample_repo, files)
    svc = str(sample_repo / "pkg/sub/service.py")
    init = str(sample_repo / "pkg/sub/__init__.py")
    assert pm[svc] == "pkg.sub.service"
    assert pm[init] == "pkg.sub"
    assert resolve_relative_import(svc, 1, "", pm) == "pkg.sub"
    assert resolve_relative_import(svc, 2, "util", pm) == "pkg.util"
    assert resolve_relative_import(init, 1, "models", pm) == "pkg.sub.models"
    assert resolve_relative_import(svc, 5, "", pm) is None


def test_extract_symbols_imports_edges(sample_repo):
    res = extract_repository(sample_repo, "demo")
    by_q = {s.qualified_name: s for s in res.symbols}

    helper = by_q["demo::pkg.util.helper"]
    assert helper.symbol_type == "function"
    assert helper.docstring == "把輸入加一，供其他模組呼叫。"
    assert helper.file_path == "pkg/util.py"
    assert by_q["demo::pkg.sub.service.Service"].symbol_type == "class"
    run = by_q["demo::pkg.sub.service.Service.run"]
    assert run.symbol_type == "method" and run.decorators == ["staticmethod"]
    assert by_q["demo::pkg.sub.service.Service.go"].symbol_type == "method"
    assert by_q["demo::pkg.sub.service"].symbol_type == "module"
    assert by_q["demo::pkg"].docstring == "範例套件。"

    # 語法錯誤檔案只記錄錯誤，不中斷
    assert [e["file_path"] for e in res.errors] == ["pkg/broken.py"]

    imports = {(e.importer, e.imported_module, e.is_type_only) for e in res.import_edges}
    assert ("demo::pkg.sub.service", "pkg.util.helper", False) in imports
    assert ("demo::pkg.sub.service", "pkg.sub.models", False) in imports
    assert ("demo::pkg.sub.service", "json", False) in imports
    assert ("demo::pkg.sub.service", "pkg.util._private", True) in imports

    calls = {(e.caller, e.callee) for e in res.call_edges}
    assert ("demo::pkg.sub.service.Service.run", "demo::pkg.util.helper") in calls   # 相對 import
    assert ("demo::pkg.sub.service.Service.go", "demo::pkg.sub.service.Service.run") in calls  # self.
    assert ("demo::pkg.sub.service.main", "demo::pkg.sub.service.Service") in calls   # 同模組
    assert ("demo::pkg.sub.service.main", "demo::pkg.sub.models.build") in calls      # 模組屬性
    assert len(calls) == len(res.call_edges)  # 已去重


def test_read_git_head_variants(tmp_path):
    assert read_git_head(tmp_path) is None

    sha = "a" * 40
    git = tmp_path / ".git"
    (git / "refs/heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs/heads/main").write_text(sha + "\n", encoding="utf-8")
    assert read_git_head(tmp_path) == sha

    # packed-refs
    (git / "refs/heads/main").unlink()
    (git / "packed-refs").write_text(f"# pack-refs\n{'b' * 40} refs/heads/main\n", encoding="utf-8")
    assert read_git_head(tmp_path) == "b" * 40

    # detached HEAD
    (git / "HEAD").write_text("c" * 40, encoding="utf-8")
    assert read_git_head(tmp_path) == "c" * 40


def test_read_git_head_worktree_file(tmp_path):
    main_git = tmp_path / "main" / ".git"
    wt_git = main_git / "worktrees" / "wt"
    (main_git / "refs/heads").mkdir(parents=True)
    wt_git.mkdir(parents=True)
    (main_git / "refs/heads/feat").write_text("d" * 40, encoding="utf-8")
    (wt_git / "HEAD").write_text("ref: refs/heads/feat", encoding="utf-8")
    (wt_git / "commondir").write_text("../..", encoding="utf-8")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}", encoding="utf-8")
    assert read_git_head(wt) == "d" * 40
