from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest

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
    gate_runner = (ROOT / "scripts/run_release_gates.py").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "SOURCE_ROOTS = (SOURCE, DEPLOY, SCRIPTS, TOOLS, TESTS)" in architecture
    assert "GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1" in makefile
    assert "artifacts-local-safety-check:" in makefile
    assert "coverage:" in makefile
    assert "test-parallel:" in makefile
    assert "test-parallel-release:" in makefile
    parallel = makefile.split("test-parallel-release:\n", 1)[1].split(
        "\ntest-impact:", 1
    )[0]
    check = makefile.split("check:\n", 1)[1].split("\narchitecture-check:", 1)[0]
    assert "GPU_FAULT_TEST_POSTGRES_URL=" in parallel
    assert "-n $(PYTEST_XDIST_WORKERS)" in parallel
    assert "--ignore=tests/test_artifact_consistency.py" in parallel
    assert "scripts/run_release_gates.py" in check
    assert "--mode check" in check
    assert '"pytest": ["make", "test-parallel-release"' in gate_runner
    assert '"artifact": ["make", "artifact-check"' in gate_runner
    assert "max_workers=2" in gate_runner
    assert gate_runner.index("parallel check tail failed") < gate_runner.rindex(
        '"python-cache-clean"'
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


# The test-tree size exceptions ``architecture-baseline.json`` carried on
# 2026-09-09, when the rule below was turned from "none" into "no new ones":
# the two files grew past the 1500-line limit across the 09-07/09-08 review
# batches and were baselined with ``--write-baseline`` instead of being split.
# Splitting them is owed; until then a new entry -- a test file that outgrows
# the limit and is baselined rather than split -- is what this contract refuses.
RECORDED_TEST_SIZE_EXCEPTIONS = {
    "files": [
        "tests/processor/test_telemetry_spool.py",
        "tests/regional/test_regional_admin_checks.py",
    ],
    "functions": [],
    "classes": [],
}


def test_tests_do_not_need_new_architecture_size_exceptions() -> None:
    baseline = json.loads(
        (ROOT / "architecture-baseline.json").read_text(encoding="utf-8")
    )
    test_exceptions = {
        kind: sorted(key for key in baseline.get(kind, {}) if key.startswith("tests/"))
        for kind in ("files", "functions", "classes")
    }

    unexpected = {
        kind: sorted(
            set(test_exceptions[kind]) - set(RECORDED_TEST_SIZE_EXCEPTIONS[kind])
        )
        for kind in test_exceptions
    }
    assert unexpected == {"files": [], "functions": [], "classes": []}, (
        "split the test module instead of baselining its size: "
        + json.dumps(unexpected)
    )
    retired = {
        kind: sorted(
            set(RECORDED_TEST_SIZE_EXCEPTIONS[kind]) - set(test_exceptions[kind])
        )
        for kind in test_exceptions
    }
    assert retired == {"files": [], "functions": [], "classes": []}, (
        "an exception was paid down; remove it from RECORDED_TEST_SIZE_EXCEPTIONS: "
        + json.dumps(retired)
    )


def test_shuffled_order_is_seeded_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """打乱只由种子决定：不设种子一动不动，设了同一个种子结果必须一样。

    最后那条同一种子、不同入参顺序得到同一结果的断言不是锦上添花：xdist 的每个
    worker 都会各自跑一遍这个钩子，一旦它们的收集顺序对不上，xdist 会直接以
    "Different tests were collected" 中止整轮，而不是报某条用例失败。
    """
    original = [
        SimpleNamespace(nodeid=f"tests/test_module_{index}.py::test_case")
        for index in range(64)
    ]

    monkeypatch.delenv(conftest.SHUFFLE_SEED_VARIABLE, raising=False)
    untouched = list(original)
    conftest.pytest_collection_modifyitems(untouched)

    monkeypatch.setenv(conftest.SHUFFLE_SEED_VARIABLE, "11")
    seeded = list(original)
    conftest.pytest_collection_modifyitems(seeded)
    from_reversed_input = list(reversed(original))
    conftest.pytest_collection_modifyitems(from_reversed_input)

    monkeypatch.setenv(conftest.SHUFFLE_SEED_VARIABLE, "12")
    other_seed = list(original)
    conftest.pytest_collection_modifyitems(other_seed)

    assert untouched == original, "没设种子时收集顺序必须原样保留"
    assert seeded != original, "设了种子却没有打乱"
    assert sorted(item.nodeid for item in seeded) == sorted(
        item.nodeid for item in original
    ), "打乱不能增减用例"
    assert from_reversed_input == seeded, "同一个种子必须无视入参顺序给出同一结果"
    assert other_seed != seeded, "不同种子应当给出不同顺序"


def test_shuffle_seed_rejects_a_non_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(conftest.SHUFFLE_SEED_VARIABLE, "not-a-number")

    with pytest.raises(pytest.UsageError):
        conftest.pytest_collection_modifyitems([])


def test_shuffled_order_is_a_required_ci_gate() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    shuffled = makefile.split("test-shuffled:\n", 1)[1].split("\ntest-parallel:", 1)[0]
    required = workflow.split("Require all CI gates", 1)[1].split("\n      - uses:", 1)[
        0
    ]

    assert conftest.SHUFFLE_SEED_VARIABLE in shuffled
    assert "-n $(PYTEST_XDIST_WORKERS)" in shuffled
    assert "make test-shuffled PYTHON=python" in workflow
    # 只有作为必需门禁才算数：不进 needs 就只是个装饰。
    assert 'needs.shuffle.result }}" = "success"' in required


def test_pydantic_mypy_plugin_checks_models(tmp_path: Path) -> None:
    """插件真的装上了，而且 ``init_typed``/``init_forbid_extra`` 没被翻掉。

    这条用例直接拿仓库自己的 ``pyproject.toml`` 跑一遍 mypy，所以
    ``plugins = ["pydantic.mypy"]`` 或 ``[tool.pydantic-mypy]`` 被摘掉它就红；
    只断言配置文件里有那几行，是断言不出插件有没有真的加载成功的。

    四个错误码分成两半，缺一半都说明配置退化了：

    * ``pydantic-field`` / ``pydantic-alias`` 只有插件会报。前者是真正的坑——
      模型体里少写注解的属性根本不是字段，它永远不参与校验，
      ``extra="forbid"`` 也拦不住，因为压根没人给它传值。
    * ``arg-type`` / ``call-arg`` 是 pydantic 自己的 ``dataclass_transform``
      给的，不装插件也有；但插件一旦加载而 ``init_typed`` 或
      ``init_forbid_extra`` 为假，合成出来的 ``__init__`` 就退化成
      ``**kwargs: Any``，这两个码会一起消失。所以它们在这里盯的是那两个开关。
    """
    module = tmp_path / "construct_models.py"
    module.write_text(
        "from pydantic import Field\n"
        "\n"
        "from gpu_fault.models import StrictModel\n"
        "\n"
        "\n"
        "class Probe(StrictModel):\n"
        "    count: int\n"
        "\n"
        "\n"
        "class MissingAnnotation(StrictModel):\n"
        "    forgot_the_annotation = 3\n"
        "\n"
        "\n"
        "class DynamicAlias(StrictModel):\n"
        "    aliased: int = Field(alias=str(1))\n"
        "\n"
        "\n"
        "def build() -> Probe:\n"
        '    return Probe(count="not-an-int", typo=1)\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "--config-file",
            str(ROOT / "pyproject.toml"),
            str(module),
        ],
        cwd=ROOT,
        env={**os.environ, "MYPYPATH": str(ROOT / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    codes = {
        line.rsplit("[", 1)[-1].rstrip("]")
        for line in completed.stdout.splitlines()
        if ": error:" in line
    }

    assert "pydantic-field" in codes, (
        f"没注解的模型属性没被拦住，pydantic.mypy 大概没加载：\n{completed.stdout}"
    )
    assert "pydantic-alias" in codes, (
        f"必填字段的动态别名没被拦住，warn_required_dynamic_aliases 没生效：\n"
        f"{completed.stdout}"
    )
    assert "arg-type" in codes, (
        f"构造点的字段类型没被检查，init_typed 大概被翻成了假：\n{completed.stdout}"
    )
    assert "call-arg" in codes, (
        f"构造点的多余关键字没被检查，init_forbid_extra 大概被翻成了假：\n"
        f"{completed.stdout}"
    )


def test_test_suite_does_not_lose_assertion_coverage() -> None:
    paths = sorted(TESTS.rglob("*.py"))
    assert_count = sum(
        isinstance(node, ast.Assert)
        for path in paths
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )

    assert assert_count >= 6_263
