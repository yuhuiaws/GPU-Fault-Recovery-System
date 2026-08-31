from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def scanned_modules() -> list[Path]:
    """Every file that carries test code, not just the collected wrappers.

    pytest 只收 ``test_*.py``，但拆分之后用例主体都在 ``_<name>_cases_N.py``
    分片里，wrapper 通常只剩一串 import。下面两条正确性扫描原来只
    ``rglob("test_*.py")``，等于随着用例主体搬走一起失效：在分片里写
    ``spec.loader.exec_module(...)`` 或 ``from x import *`` 都扫不到。
    函数名故意不以 ``test_`` 开头，否则 pytest 会把它当用例收走。
    """
    return sorted(path for path in TESTS.rglob("*.py") if path.name != "__init__.py")


class ModuleScopeCallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.exec_module_lines: list[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "exec_module":
            self.exec_module_lines.append(node.lineno)
        self.generic_visit(node)


def test_correctness_scans_cover_split_shards() -> None:
    # 防止上面两条扫描被改回只看 ``test_*.py``：分片文件必须在扫描面里，
    # 而且分片数量不为零（否则这条断言本身就是空的）。
    scanned = scanned_modules()
    shards = [path for path in scanned if "_cases_" in path.name]
    wrappers = [path for path in scanned if path.name.startswith("test_")]

    assert shards, "tests/ 下已经没有分片了，这两条扫描的扩面可以收回"
    assert len(scanned) == len(set(scanned))
    assert len(scanned) > len(wrappers)


def test_no_test_module_executes_path_modules_during_collection() -> None:
    offenders = []
    for path in scanned_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        collector = ModuleScopeCallCollector()
        for node in tree.body:
            collector.visit(node)
        offenders.extend(
            f"{path.relative_to(ROOT)}:{line}" for line in collector.exec_module_lines
        )

    assert offenders == []


def test_test_packages_and_imports_are_explicit() -> None:
    for directory in TESTS.iterdir():
        if directory.is_dir() and any(
            list(directory.glob("test_*.py")) + list(directory.glob("*_cases_*.py"))
        ):
            assert (directory / "__init__.py").is_file(), directory

    star_imports = []
    for path in scanned_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                star_imports.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert star_imports == []


def test_split_test_wrappers_export_collection_contracts() -> None:
    # 分片里的 ``test_*`` 函数也算契约，而且是最贵的那一类：pytest 只收
    # ``test_*.py``，分片文件名以 ``_`` 开头不会被收集，用例能跑全靠 wrapper
    # 把每个名字显式 import 进来。少写一行 import 就是静默少跑一条用例——
    # 套件仍然全绿，条数变少没人看。原来只要求 pytestmark 和 fixture，
    # 实测从 test_fleet.py 的 import 列表里删掉一个用例名仍然通过。
    required_by_wrapper: dict[Path, set[str]] = {}
    for path in TESTS.rglob("*_cases_*.py"):
        base = path.stem[1:].rsplit("_cases_", 1)[0]
        wrapper = path.with_name(f"test_{base}.py")
        required = required_by_wrapper.setdefault(wrapper, set())
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            ):
                required.add("pytestmark")
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("test_"):
                required.add(node.name)
            for decorator in node.decorator_list:
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if isinstance(target, ast.Attribute) and target.attr == "fixture":
                    required.add(node.name)

    assert required_by_wrapper
    for wrapper, required in required_by_wrapper.items():
        assert wrapper.is_file(), wrapper
        tree = ast.parse(wrapper.read_text(encoding="utf-8"))
        imported = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert required <= imported, (
            f"{wrapper.relative_to(ROOT)} must export collection contracts "
            f"{sorted(required - imported)}"
        )


def test_tests_are_covered_by_architecture_and_quality_gates() -> None:
    architecture = (ROOT / "scripts/check-python-architecture.py").read_text(
        encoding="utf-8"
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "SOURCE_ROOTS = (SOURCE, DEPLOY, SCRIPTS, TOOLS, TESTS)" in architecture
    assert "GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1" in makefile
    assert "artifacts-local-safety-check:" in makefile
    assert "coverage:" in makefile
    assert "test-parallel:" in makefile
    parallel = makefile.split("test-parallel:\n", 1)[1].split("\ncoverage:", 1)[0]
    check = makefile.split("check:\n", 1)[1].split("\narchitecture-check:", 1)[0]
    assert "GPU_FAULT_TEST_POSTGRES_URL=" in parallel
    assert "-n $(PYTEST_XDIST_WORKERS)" in parallel
    assert "$(MAKE) test-parallel" in check
    assert check.index("$(MAKE) test-parallel") < check.rindex(
        "$(MAKE) python-cache-clean"
    )
    assert "pytest-cov" in project
    assert "pytest-xdist" in project
    assert "private-test-coupling-check:" in makefile


def test_make_python_prefers_local_venv_and_preserves_overrides(tmp_path: Path) -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assignment = next(
        line for line in makefile.splitlines() if line.startswith("PYTHON ?=")
    )
    probe = assignment + "\nprint-python:\n\t@printf '%s\\n' '$(PYTHON)'\n"
    default_environment = os.environ.copy()
    for name in ("PYTHON", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES"):
        default_environment.pop(name, None)

    def resolve(cwd: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["make", "--no-print-directory", "-s", "-f", "-", *arguments],
            cwd=cwd,
            env=default_environment,
            input=probe,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout.strip()

    with_venv = tmp_path / "with-venv"
    with_venv.joinpath(".venv/bin").mkdir(parents=True)
    with_venv.joinpath(".venv/bin/python").touch()
    without_venv = tmp_path / "without-venv"
    without_venv.mkdir()

    assert resolve(with_venv, "print-python") == ".venv/bin/python"
    assert resolve(without_venv, "print-python") == "python3"
    assert resolve(with_venv, "print-python", "PYTHON=python-custom") == "python-custom"


def test_tests_do_not_need_architecture_size_exceptions() -> None:
    baseline = json.loads(
        (ROOT / "architecture-baseline.json").read_text(encoding="utf-8")
    )
    test_exceptions = {
        kind: sorted(key for key in baseline.get(kind, {}) if key.startswith("tests/"))
        for kind in ("files", "functions", "classes")
    }

    assert test_exceptions == {"files": [], "functions": [], "classes": []}


def test_test_suite_does_not_lose_assertion_coverage() -> None:
    paths = sorted(TESTS.rglob("*.py"))
    assert_count = sum(
        isinstance(node, ast.Assert)
        for path in paths
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )

    assert assert_count >= 6_263
