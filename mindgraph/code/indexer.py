"""MindGraph 程式碼索引器：以 Python AST 兩階段掃描 repo，建立程式碼圖。

流程：
1. 收集 .py 檔並建立「檔案 → 點分模組路徑」對照表（package map）
2. 抽取 module / class / function / method symbol（含 docstring 與 decorator）
3. 解析 import（絕對 import 與相對 import）建立每個檔案的別名對照
4. 以別名對照解析呼叫，產生 symbol 間的呼叫邊
5. 單一交易寫入 PostgreSQL：刪除該 repo 既有資料 → 批次寫入，並記錄索引當下
   的 git HEAD 與時間（不變式 I-7：索引狀態必須可與 repo 目前 HEAD 比對以偵測過期）

純函式部分（extract_repository / read_git_head）不需資料庫，可獨立測試。
"""

from __future__ import annotations

import ast
import fnmatch
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from mindgraph.code.schema import atomic, ensure_code_schema

logger = logging.getLogger("mindgraph.code.indexer")

DEFAULT_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".venv", "venv", ".git", "node_modules", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache",
})


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SymbolInfo:
    """索引器抽出的單一程式碼 symbol，是 MindGraph 程式碼圖的節點。

    qualified_name 格式為 `{repo}::{點分模組路徑}.{類別}.{方法}`，在同一 repo 內唯一；
    file_path 一律存 repo 相對路徑（POSIX 斜線），避免把主機絕對路徑寫進資料庫。
    symbol_type 為 module / class / function / method 之一。
    """
    qualified_name: str
    short_name: str
    symbol_type: str
    file_path: str
    lineno: int
    docstring: Optional[str]
    decorators: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CallEdge:
    """程式碼圖中「caller 呼叫 callee」的有向邊，兩端皆為同 repo 內已索引的 symbol。"""
    caller: str
    callee: str
    call_type: str = "direct"


@dataclass(slots=True)
class ImportEdge:
    """模組層級的 import 紀錄：importer 為模組 symbol 的 qualified_name，
    imported_module 為解析後的點分路徑；TYPE_CHECKING 區塊內的 import 標記 is_type_only。"""
    importer: str
    imported_module: str
    is_type_only: bool = False


@dataclass(slots=True)
class IndexResult:
    """一次 repo 掃描的完整產出，交給 persist_index 寫入資料庫。

    errors 為無法解析的檔案清單（語法錯誤等），每筆含 file_path / lineno /
    category / message，不會中斷整體索引。
    """
    repo: str
    symbols: list[SymbolInfo]
    call_edges: list[CallEdge]
    import_edges: list[ImportEdge]
    errors: list[dict]


# ---------------------------------------------------------------------------
# 檔案收集與模組路徑
# ---------------------------------------------------------------------------

def _is_excluded(rel_path: str, name: str, extra: tuple[str, ...]) -> bool:
    """判斷目錄或檔案是否命中呼叫端額外排除清單：比對名稱或 repo 相對路徑（支援萬用字元）。"""
    return any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat) for pat in extra)


def collect_python_files(repo_path: Path, exclude: Iterable[str] = ()) -> list[Path]:
    """收集 repo 內所有要索引的 .py 檔，作為 MindGraph 程式碼索引的第一步。

    固定跳過 DEFAULT_SKIP_DIRS（__pycache__、.venv、venv、.git、node_modules、dist、
    build 等）與所有隱藏目錄；呼叫端可再傳入 exclude，每個樣式會同時比對目錄/檔案
    名稱與 repo 相對 POSIX 路徑（fnmatch 語法，例如 "tests"、"scripts/*"、"*_pb2.py"）。
    回傳依路徑排序的清單，確保索引結果可重現。
    """
    repo_path = Path(repo_path)
    extra = tuple(exclude)
    files: list[Path] = []
    for root, dirs, names in os.walk(repo_path):
        root_p = Path(root)
        kept = []
        for d in dirs:
            if d in DEFAULT_SKIP_DIRS or d.startswith("."):
                continue
            rel = (root_p / d).relative_to(repo_path).as_posix()
            if extra and _is_excluded(rel, d, extra):
                continue
            kept.append(d)
        dirs[:] = kept
        for name in names:
            if not name.endswith(".py"):
                continue
            p = root_p / name
            if extra and _is_excluded(p.relative_to(repo_path).as_posix(), name, extra):
                continue
            files.append(p)
    return sorted(files)


def build_package_map(repo_path: Path, py_files: list[Path]) -> dict[str, str]:
    """建立「檔案絕對路徑 → 點分模組路徑」對照，供 symbol 命名與 import 解析共用。

    從檔案所在目錄往上走，只要目錄含 __init__.py 就視為套件的一層；
    `__init__.py` 本身代表套件而非子模組。例：pkg/sub/mod.py（pkg、sub 皆有
    __init__.py）→ `pkg.sub.mod`；非套件內的獨立腳本 → 檔名本身。
    """
    package_dirs = {str(f.parent) for f in py_files if f.name == "__init__.py"}
    result: dict[str, str] = {}
    for f in py_files:
        parts: list[str] = [] if f.name == "__init__.py" else [f.stem]
        current = f.parent
        while str(current) in package_dirs:
            parts.append(current.name)
            current = current.parent
        parts.reverse()
        result[str(f)] = ".".join(parts) if parts else f.stem
    return result


def resolve_relative_import(
    file_path: str, level: int, module_suffix: str, package_map: dict[str, str]
) -> Optional[str]:
    """把相對 import（from . / from ..x import y）解析成絕對點分模組路徑。

    level 為前導點數；一般 .py 檔的 level=1 代表所在套件（去掉自身模組名），
    `__init__.py` 本身就是套件，level=1 保持原地。往上爬超過根目錄時回傳 None。
    例：檔案模組 `pkg.sub.mod`，level=2、module_suffix='util' → `pkg.util`。
    """
    current = package_map.get(str(file_path))
    if not current:
        return None
    parts = current.split(".")
    is_init = os.path.basename(file_path) == "__init__.py"
    remove = level - (1 if is_init else 0)
    if remove < 0 or remove > len(parts):
        return None
    base = parts[: len(parts) - remove]
    if module_suffix:
        base = base + module_suffix.split(".")
    return ".".join(base) if base else None


# ---------------------------------------------------------------------------
# git HEAD（不變式 I-7）
# ---------------------------------------------------------------------------

def _resolve_git_dir(repo_path: Path) -> Optional[Path]:
    """找出 repo 的 .git 目錄；支援 worktree / submodule 使用的 `.git` 檔（gitdir: 指標）。"""
    dot_git = repo_path / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            target = Path(text[len("gitdir:"):].strip())
            return target if target.is_absolute() else (repo_path / target).resolve()
    return None


def read_git_head(repo_path: Path) -> Optional[str]:
    """讀取 repo 目前 HEAD 指向的完整 commit SHA，供索引狀態記錄與過期偵測（不變式 I-7）。

    直接解析 .git/HEAD、loose ref 與 packed-refs，不呼叫 git 執行檔（容器內常見
    safe.directory 限制或根本沒裝 git）。worktree 的 ref 若不在自身 gitdir，會退回
    commondir 查找。非 git repo 或無法判讀時回傳 None，呼叫端應視為「未知」而非錯誤。
    """
    try:
        git_dir = _resolve_git_dir(Path(repo_path))
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None  # detached HEAD
        ref = head[4:].strip()
        search_dirs = [git_dir]
        common = git_dir / "commondir"
        if common.is_file():
            c = Path(common.read_text(encoding="utf-8").strip())
            search_dirs.append(c if c.is_absolute() else (git_dir / c).resolve())
        for d in search_dirs:
            ref_file = d / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip() or None
            packed = d / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith(("#", "^")):
                        sha, _, name = line.partition(" ")
                        if name.strip() == ref:
                            return sha
        return None
    except OSError:
        logger.warning("讀取 git HEAD 失敗：%s", repo_path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# AST 走訪
# ---------------------------------------------------------------------------

def _decorators(node: ast.AST) -> list[str]:
    out = []
    for d in getattr(node, "decorator_list", []):
        try:
            out.append(ast.unparse(d))
        except Exception:  # noqa: BLE001 — decorator 還原失敗不影響索引
            pass
    return out


class _SymbolVisitor(ast.NodeVisitor):
    """走訪單一檔案 AST，抽出 module / class / function / method symbol。"""

    def __init__(self, repo: str, module: str, rel_path: str, tree: ast.Module) -> None:
        self._repo, self._mod, self._rel = repo, module, rel_path
        self._class_stack: list[str] = []
        self.symbols: list[SymbolInfo] = [SymbolInfo(
            qualified_name=f"{repo}::{module}",
            short_name=module.rsplit(".", 1)[-1],
            symbol_type="module",
            file_path=rel_path,
            lineno=0,
            docstring=ast.get_docstring(tree),
        )]

    def _qname(self, name: str) -> str:
        return f"{self._repo}::{'.'.join([self._mod, *self._class_stack, name])}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(SymbolInfo(
            self._qname(node.name), node.name, "class", self._rel,
            node.lineno, ast.get_docstring(node), _decorators(node),
        ))
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node) -> None:
        self.symbols.append(SymbolInfo(
            self._qname(node.name), node.name,
            "method" if self._class_stack else "function", self._rel,
            node.lineno, ast.get_docstring(node), _decorators(node),
        ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


class _CallVisitor(ast.NodeVisitor):
    """追蹤目前所在函式，將每個 ast.Call 透過 import 別名對照解析成 callee。"""

    def __init__(self, repo: str, module: str, imports: dict[str, str], symbols: set[str]) -> None:
        self._repo, self._mod = repo, module
        self._imports, self._symbols = imports, symbols
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []
        self.edges: list[CallEdge] = []

    def _qname(self, name: str) -> str:
        return f"{self._repo}::{'.'.join([self._mod, *self._class_stack, name])}"

    def _full(self, raw: str) -> str:
        return raw if "::" in raw else f"{self._repo}::{raw}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node) -> None:
        self._func_stack.append(self._qname(node.name))
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self._func_stack:
            callee = self._resolve(node.func)
            if callee and callee in self._symbols:
                self.edges.append(CallEdge(self._func_stack[-1], callee))
        self.generic_visit(node)

    def _resolve(self, fn: ast.expr) -> Optional[str]:
        if isinstance(fn, ast.Name):
            if fn.id in self._imports:
                return self._full(self._imports[fn.id])
            # 同模組內定義的頂層函式/類別
            return f"{self._repo}::{self._mod}.{fn.id}"
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            base = fn.value.id
            if base in ("self", "cls") and self._class_stack:
                return self._qname(fn.attr)
            if base in self._imports:
                return f"{self._full(self._imports[base])}.{fn.attr}"
        return None


def _type_checking_nodes(tree: ast.Module) -> set[int]:
    """回傳模組層級 `if TYPE_CHECKING:` 區塊內所有節點的 id，用於標記型別專用 import。"""
    ids: set[int] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.If) and "TYPE_CHECKING" in ast.unparse(stmt.test):
            ids.update(id(sub) for sub in ast.walk(stmt))
    return ids


def _resolve_imports(
    repo: str, file_path: str, module: str, tree: ast.Module,
    package_map: dict[str, str], symbols: set[str],
) -> tuple[dict[str, str], list[ImportEdge]]:
    """解析單一檔案的 import，回傳（別名 → 目標名稱對照, import 邊）。

    目標名稱若命中已索引 symbol 則為完整 qualified_name（含 `repo::`），否則為原始
    點分路徑（外部套件）；呼叫邊解析時只會保留命中 symbol 表的目標。
    """
    importer = f"{repo}::{module}"
    tc_nodes = _type_checking_nodes(tree)
    alias_map: dict[str, str] = {}
    edges: list[ImportEdge] = []

    for node in ast.walk(tree):
        type_only = id(node) in tc_nodes
        if isinstance(node, ast.Import):
            for a in node.names:
                alias_map[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0]
                )
                edges.append(ImportEdge(importer, a.name, type_only))
        elif isinstance(node, ast.ImportFrom):
            if not node.names or node.names[0].name == "*":
                continue
            if node.level:
                base = resolve_relative_import(file_path, node.level, node.module or "", package_map)
                if base is None:
                    continue
            else:
                base = node.module or ""
            for a in node.names:
                target = f"{base}.{a.name}" if base else a.name
                qualified = f"{repo}::{target}"
                alias_map[a.asname or a.name] = qualified if qualified in symbols else target
                edges.append(ImportEdge(importer, target, type_only))
    return alias_map, edges


# ---------------------------------------------------------------------------
# 對外：掃描與寫入
# ---------------------------------------------------------------------------

def extract_repository(repo_path: Path, repo: str, exclude: Iterable[str] = ()) -> IndexResult:
    """掃描整個 repo 並產出 MindGraph 程式碼圖（純函式，不觸碰資料庫）。

    依序執行：收集 .py 檔 → 建立模組路徑對照 → 抽 symbol → 解析 import（含相對
    import）→ 解析呼叫邊。單一檔案語法錯誤只記入 errors，不中斷其他檔案。
    同一檔案只 parse 一次，AST 在三個階段間共用。
    """
    repo_path = Path(repo_path)
    files = collect_python_files(repo_path, exclude)
    package_map = build_package_map(repo_path, files)
    errors: list[dict] = []
    trees: dict[str, tuple[ast.Module, str]] = {}
    symbols: dict[str, SymbolInfo] = {}

    for f in files:
        module = package_map.get(str(f), f.stem)
        rel = f.relative_to(repo_path).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
        except (SyntaxError, ValueError) as exc:
            errors.append({
                "file_path": rel, "lineno": getattr(exc, "lineno", None),
                "category": "syntax", "message": str(exc),
            })
            continue
        trees[str(f)] = (tree, module)
        visitor = _SymbolVisitor(repo, module, rel, tree)
        visitor.visit(tree)
        for s in visitor.symbols:
            symbols[s.qualified_name] = s

    symbol_names = set(symbols)
    import_edges: list[ImportEdge] = []
    call_edges: list[CallEdge] = []
    for fpath, (tree, module) in trees.items():
        try:
            alias_map, imp = _resolve_imports(repo, fpath, module, tree, package_map, symbol_names)
            import_edges.extend(imp)
            cv = _CallVisitor(repo, module, alias_map, symbol_names)
            cv.visit(tree)
            call_edges.extend(cv.edges)
        except Exception as exc:  # noqa: BLE001 — 單檔失敗記錄後繼續
            errors.append({
                "file_path": Path(fpath).relative_to(repo_path).as_posix(),
                "lineno": getattr(exc, "lineno", None), "category": "link",
                "message": f"{exc}\n{traceback.format_exc(limit=3)}",
            })

    # 去重：同一 caller→callee 只保留一條
    seen: set[tuple[str, str, str]] = set()
    unique_calls = []
    for e in call_edges:
        key = (e.caller, e.callee, e.call_type)
        if key not in seen:
            seen.add(key)
            unique_calls.append(e)

    logger.info(
        "extract done repo=%s files=%d symbols=%d calls=%d imports=%d errors=%d",
        repo, len(files), len(symbols), len(unique_calls), len(import_edges), len(errors),
    )
    return IndexResult(repo, list(symbols.values()), unique_calls, import_edges, errors)


def persist_index(conn, result: IndexResult, git_head: Optional[str], repo_path: Optional[str] = None) -> dict:
    """把一次掃描結果寫入 MindGraph 程式碼圖資料表，並更新索引狀態。

    單一交易內執行「刪除該 repo 全部 symbol（邊與 wiki 連結經 ON DELETE CASCADE
    一併刪除）→ 批次寫入 symbol / 呼叫邊 / import 邊 → upsert code.index_state」，
    讀取端不會看到半套索引。注意：因 symbol id 重新產生，既有 wiki 連結會被清空，
    索引後必須重跑 link_wiki。回傳實際寫入筆數統計。
    """
    repo = result.repo
    with atomic(conn):
        conn.execute("DELETE FROM code.symbols WHERE repo = %s", (repo,))
        qname_to_id: dict[str, int] = {}
        with conn.cursor() as cur:
            if result.symbols:
                cur.executemany(
                    """INSERT INTO code.symbols
                       (repo, qualified_name, short_name, symbol_type, file_path,
                        lineno, docstring, decorators, git_head)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)""",
                    [(repo, s.qualified_name, s.short_name, s.symbol_type, s.file_path,
                      s.lineno, s.docstring, json.dumps(s.decorators, ensure_ascii=False), git_head)
                     for s in result.symbols],
                )
                cur.execute("SELECT id, qualified_name FROM code.symbols WHERE repo = %s", (repo,))
                qname_to_id = {qn: sid for sid, qn in cur.fetchall()}

            calls = [(qname_to_id[e.caller], qname_to_id[e.callee], e.call_type)
                     for e in result.call_edges
                     if e.caller in qname_to_id and e.callee in qname_to_id]
            if calls:
                cur.executemany(
                    "INSERT INTO code.call_edges (caller_id, callee_id, call_type) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    calls,
                )
            imports = [(qname_to_id[e.importer], e.imported_module, e.is_type_only)
                       for e in result.import_edges if e.importer in qname_to_id]
            if imports:
                cur.executemany(
                    "INSERT INTO code.import_edges (importer_id, imported_module, is_type_only) "
                    "VALUES (%s, %s, %s)",
                    imports,
                )
            cur.execute(
                """INSERT INTO code.index_state
                   (repo, repo_path, git_head, indexed_at, symbol_count,
                    call_edge_count, import_edge_count, error_count)
                   VALUES (%s, %s, %s, now(), %s, %s, %s, %s)
                   ON CONFLICT (repo) DO UPDATE SET
                     repo_path = EXCLUDED.repo_path,
                     git_head = EXCLUDED.git_head,
                     indexed_at = EXCLUDED.indexed_at,
                     symbol_count = EXCLUDED.symbol_count,
                     call_edge_count = EXCLUDED.call_edge_count,
                     import_edge_count = EXCLUDED.import_edge_count,
                     error_count = EXCLUDED.error_count""",
                (repo, repo_path, git_head, len(qname_to_id), len(calls), len(imports),
                 len(result.errors)),
            )
    return {
        "symbols": len(qname_to_id),
        "call_edges": len(calls),
        "import_edges": len(imports),
    }


def index_repository(conn, repo_path, repo: str, exclude: Iterable[str] = ()) -> dict:
    """MindGraph 程式碼索引的對外入口：掃描 repo → 寫入資料庫 → 記錄 git HEAD。

    會先確保 `code` schema 存在（冪等），再呼叫 extract_repository 與 persist_index。
    回傳 {repo, git_head, symbols, call_edges, import_edges, errors, error_samples}；
    error_samples 最多列 10 筆無法解析的檔案，方便呼叫端直接呈現。
    """
    repo_path = Path(repo_path).resolve()
    ensure_code_schema(conn)
    git_head = read_git_head(repo_path)
    result = extract_repository(repo_path, repo, exclude)
    stats = persist_index(conn, result, git_head, str(repo_path))
    logger.info("index_repository done repo=%s head=%s stats=%s", repo, git_head, stats)
    return {
        "repo": repo,
        "git_head": git_head,
        **stats,
        "errors": len(result.errors),
        "error_samples": result.errors[:10],
    }
