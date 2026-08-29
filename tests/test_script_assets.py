from __future__ import annotations

import json
import re
import stat
from collections import Counter
from pathlib import Path

import yaml

from scripts.e2e.render_manifest import (
    RUNTIME_PROFILE_PLACEHOLDER,
    WHEEL_CONFIGMAP_PLACEHOLDER,
    render_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TOOLS = ROOT / "tools"
E2E_MANIFESTS = SCRIPTS / "e2e/regional/manifests"


def _walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


PYTHON_QUALITY_ROOTS = {"src", "tests", "deploy", "scripts", "tools"}
SHELL_QUALITY_ROOTS = {"deploy", "scripts", "tools"}
YAML_QUALITY_ROOTS = {
    "deploy",
    "examples",
    "testcases",
    "config",
    "scripts/e2e",
    "scripts/perf",
}
# 每个门禁在两份文件里各出现几次。substring 断言在这里是无效守卫：
# ``src tests deploy $(QUALITY_SCRIPTS)`` 在 Makefile 里出现 3 次
# （ruff format、ruff check、compileall），``find deploy scripts tools``
# 在 ci.yml 里出现 2 次（bash -n、shellcheck），删掉任意一处、或者从
# 其中一处删掉一个根，剩下的那一处仍让断言全绿。实测：把 compileall
# 的 tests 拿掉，旧断言不动声色地通过。
GATES = {
    "ruff format --check": {
        "Makefile": [PYTHON_QUALITY_ROOTS - {"tests"}, {"tests"}],
        "ci.yml": [PYTHON_QUALITY_ROOTS - {"tests"}, {"tests"}],
    },
    "ruff check": {
        "Makefile": [PYTHON_QUALITY_ROOTS, {"tests"}, {"tests"}],
        "ci.yml": [PYTHON_QUALITY_ROOTS, {"tests"}],
    },
    "compileall": {
        "Makefile": [PYTHON_QUALITY_ROOTS],
        "ci.yml": [PYTHON_QUALITY_ROOTS],
    },
    "-name '*.sh'": {
        "Makefile": [SHELL_QUALITY_ROOTS, SHELL_QUALITY_ROOTS],
        "ci.yml": [SHELL_QUALITY_ROOTS, SHELL_QUALITY_ROOTS],
    },
    "yamllint": {"Makefile": [YAML_QUALITY_ROOTS], "ci.yml": [YAML_QUALITY_ROOTS]},
}
_FLAG_WITH_VALUE = {"-c", "-o", "--cov", "-n", "--severity"}


def _makefile_variables(text: str) -> dict[str, str]:
    variables: dict[str, str] = {}
    for logical in _logical_lines(text):
        if logical.startswith("\t"):
            continue
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*\??=\s*(.*)$", logical)
        if match is not None:
            variables[match.group(1)] = match.group(2)
    return variables


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            pending += stripped[:-1].rstrip() + " "
            continue
        lines.append(pending + stripped)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _expand(value: str, variables: dict[str, str]) -> str:
    for _ in range(8):
        expanded = re.sub(
            r"\$\(([A-Z][A-Z0-9_]*)\)",
            lambda match: variables.get(match.group(1), match.group(0)),
            value,
        )
        if expanded == value:
            return expanded
        value = expanded
    return value


def _path_arguments(command: str) -> set[str]:
    """Collect the path operands a gate command actually walks."""
    tokens = command.split()
    paths: set[str] = set()
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        candidate = token.strip("'\"")
        if candidate in _FLAG_WITH_VALUE:
            skip_next = True
            continue
        if candidate.startswith("-") or "=" in candidate or candidate.startswith("$"):
            continue
        if index and tokens[index - 1] in {"-m", "-name", "-type"}:
            continue
        if candidate.endswith((".py", ".yaml", ".yml", ".txt")):
            continue
        if (ROOT / candidate).is_dir():
            paths.add(candidate)
    return paths


def _gate_roots(commands: list[str], marker: str) -> list[set[str]]:
    return [_path_arguments(command) for command in commands if marker in command]


def test_quality_gates_cover_scripts_and_tools() -> None:
    variables = _makefile_variables((ROOT / "Makefile").read_text(encoding="utf-8"))
    makefile_commands = [
        _expand(logical.lstrip("\t"), variables)
        for logical in _logical_lines((ROOT / "Makefile").read_text(encoding="utf-8"))
        if logical.startswith("\t")
    ]
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    workflow_commands = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if isinstance(step.get("run"), str)
    ]

    for source, commands in (
        ("Makefile", makefile_commands),
        ("ci.yml", workflow_commands),
    ):
        for marker, expected_by_source in GATES.items():
            found = _gate_roots(commands, marker)
            actual = Counter(frozenset(roots) for roots in found)
            expected = Counter(frozenset(roots) for roots in expected_by_source[source])
            assert actual == expected, f"{source}: {marker} -> {found}"

    architecture = (SCRIPTS / "check-python-architecture.py").read_text(
        encoding="utf-8"
    )
    assert "SOURCE_ROOTS = (SOURCE, DEPLOY, SCRIPTS, TOOLS, TESTS)" in architecture
    assert "$(MAKE) assert-message-check" in (ROOT / "Makefile").read_text(
        encoding="utf-8"
    )
    assert "scripts/check-assert-messages.py" in (
        ROOT / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")


def test_make_redirects_python_caches_outside_the_checkout() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "PYTHONPYCACHEPREFIX ?= /tmp/gpu-fault-pycache" in makefile
    assert "export PYTHONPYCACHEPREFIX" in makefile
    assert "python-cache-clean:" in makefile
    assert "$(MAKE) python-cache-clean" in makefile
    assert "deployment-contracts-update: python-cache-clean" in makefile
    assert "deployment-contracts-check: python-cache-clean" in makefile
    assert "deploy-check: python-cache-clean" in makefile
    assert "git ls-files" in makefile


def test_yamllint_configuration_is_load_bearing() -> None:
    # yamllint 默认把 .yamllint 里的规则报成 warning 然后 exit 0：
    # 不带 --strict 的话缩进漂移只是打印一行，门禁照样绿。
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "yamllint --strict" in makefile
    assert "yamllint --strict" in workflow
    assert "$(MAKE) yaml-check" in makefile


def test_ci_runs_and_uploads_fault_scenario_report() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["test"]["steps"]
    coverage_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run full suite with coverage floor"
    )
    runner_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run unit and component fault scenario catalog runner"
    )
    upload_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Upload fault test report"
    )
    upload = steps[upload_index]

    assert coverage_index < runner_index < upload_index
    assert steps[coverage_index]["run"] == "make coverage PYTHON=python"
    assert steps[runner_index]["run"] == "make fault-test-cases-ci"
    assert upload["if"] == "always()"
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"] == {
        "name": "fault-test-report",
        "path": "artifacts/fault/fault-tests-*.json",
        "if-no-files-found": "error",
        "retention-days": 30,
    }


def test_coverage_floor_is_wired_into_make_and_ci() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "COVERAGE_FLOOR ?= 78" in makefile
    assert "GPU_FAULT_TEST_POSTGRES_URL=" in makefile
    assert "--cov-append" in makefile
    assert "--cov-fail-under=$(COVERAGE_FLOOR)" in makefile
    assert "Run full suite with coverage floor" in workflow
    assert "make coverage PYTHON=python" in workflow
    assert "make coverage" in (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")


def test_scripts_and_tools_need_no_architecture_size_exceptions() -> None:
    baseline = json.loads(
        (ROOT / "architecture-baseline.json").read_text(encoding="utf-8")
    )
    exceptions = {
        kind: sorted(
            key
            for key in baseline.get(kind, {})
            if key.startswith(("scripts/", "tools/"))
        )
        for kind in ("files", "functions", "classes")
    }

    assert exceptions == {"files": [], "functions": [], "classes": []}


def test_script_manifests_do_not_pin_concrete_nodes() -> None:
    for path in SCRIPTS.rglob("*.yaml"):
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            for mapping in _walk(document):
                assert "nodeName" not in mapping, path


def test_deploy_manifests_do_not_pin_a_concrete_node(tmp_path: Path) -> None:
    # 同一条规则原来只扫 scripts/，而写死实例 id 的是
    # deploy/migrations/sqlite-to-postgres-migration.yaml（nodeName:
    # A concrete HyperPod node id was once pinned here; the manual already
    # states that environment-specific node identities are not accepted.
    # nodeName」，却没有任何门禁对 deploy/ 生效。deploy/ 里的 Job 确实可能
    # 需要定点（hostPath），所以这里不禁 nodeName 本身，只要求它是占位符。
    pinned = []
    for path in (ROOT / "deploy").rglob("*.yaml"):
        # BaseLoader，不是 safe_load_all：deploy/aws/lambda/ 下的
        # CloudFormation 模板带 !GetAtt，SafeLoader 会直接抛。
        for document in yaml.load_all(
            path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        ):
            for mapping in _walk(document):
                value = mapping.get("nodeName")
                if value is None:
                    continue
                text = str(value)
                if text.startswith("REPLACE_WITH") or "${" in text:
                    continue
                pinned.append(f"{path.relative_to(ROOT)}: {text}")

    assert pinned == []


def test_deploy_tree_names_no_concrete_instance_id() -> None:
    offenders = []
    for path in (ROOT / "deploy").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"hyperpod-i-[0-9a-f]{8,}", text):
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")

    assert offenders == []


def test_e2e_wheel_manifests_require_rendering(tmp_path: Path) -> None:
    candidates = []
    for path in E2E_MANIFESTS.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if WHEEL_CONFIGMAP_PLACEHOLDER not in text:
            continue
        candidates.append(path)
        assert "render_manifest.py" in "\n".join(text.splitlines()[:5])
        rendered = render_manifest(
            path, "gpu-fault-control-plane-wheel-0100-deadbeefcafe"
        )
        assert WHEEL_CONFIGMAP_PLACEHOLDER not in rendered
        assert list(yaml.safe_load_all(rendered)), (
            "expected list(yaml.safe_load_all(rendered)) to be truthy"
        )

    assert candidates, "expected candidates to be truthy"
    for path in E2E_MANIFESTS.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"name:\s+gpu-fault-control-plane-wheel", text) is None, path
    assert "render_manifest.py" in (E2E_MANIFESTS / "README.md").read_text(
        encoding="utf-8"
    )


def test_e2e_runtime_profile_manifests_require_rendering() -> None:
    candidates = []
    for path in E2E_MANIFESTS.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if RUNTIME_PROFILE_PLACEHOLDER not in text:
            continue
        candidates.append(path)
        rendered = render_manifest(
            path, runtime_profile="regional-hyperpod-deadbeefcafe"
        )
        assert RUNTIME_PROFILE_PLACEHOLDER not in rendered
        assert "regional-hyperpod-deadbeefcafe" in rendered
        assert list(yaml.safe_load_all(rendered)), path

    assert candidates, "expected runtime-profile manifest candidates"


def test_scripts_have_no_environment_specific_aws_or_node_defaults() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for root in (SCRIPTS, TOOLS)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )

    assert "51" + "4385905925" not in text
    assert re.search(r"\beks-cluster-hypd-[A-Za-z0-9-]+\b", text) is None
    assert "liang" + "aws" not in text
    assert re.search(r"hyperpod-i-[0-9a-f]{8,}", text) is None


def test_perf_tree_contains_only_supported_public_entries() -> None:
    perf = SCRIPTS / "perf"
    readme = (perf / "README.md").read_text(encoding="utf-8")

    assert "regional_capacity_suite.py" in readme
    assert "regional_action_capacity_suite.py" in readme
    assert not (perf / "legacy").exists(), (
        'expected (perf / "legacy").exists() to be falsy'
    )
    assert not list(perf.glob("*-job.yaml")), (
        'expected list(perf.glob("*-job.yaml")) to be falsy'
    )
    assert "private archive" in " ".join(readme.split())


def test_release_deploy_is_the_supported_developer_entrypoint() -> None:
    readme = (SCRIPTS / "README.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert (SCRIPTS / "release_deploy.py").is_file(), (
        "unified developer release entrypoint is missing"
    )
    assert "release_deploy.py" in readme
    assert "release-deploy:" in makefile


def test_importable_script_modules_do_not_mutate_sys_path() -> None:
    assert (SCRIPTS / "__init__.py").is_file(), (
        'expected (SCRIPTS / "__init__.py").is_file() to be truthy'
    )
    assert (SCRIPTS / "e2e/__init__.py").is_file(), (
        'expected (SCRIPTS / "e2e/__init__.py").is_file() to be truthy'
    )
    assert (SCRIPTS / "e2e/regional/__init__.py").is_file(), (
        "regional E2E package marker is missing"
    )
    assert (SCRIPTS / "e2e/hyperpod/__init__.py").is_file(), (
        "HyperPod E2E package marker is missing"
    )
    assert (SCRIPTS / "perf/__init__.py").is_file(), (
        'expected (SCRIPTS / "perf/__init__.py").is_file() to be truthy'
    )

    for relative in (
        "tests/regional/test_regional_capacity_suite.py",
        "tests/notifications/test_notifications.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text

    assert not list(TOOLS.glob("run_hyperpod_*")), (
        'expected list(TOOLS.glob("run_hyperpod_*")) to be falsy'
    )
    assert (SCRIPTS / "perf/control_plane_capacity_probe.py").is_file(), (
        'expected (SCRIPTS / "perf/control_plane_capacity_probe.py").is_file() to be truthy'
    )


def test_e2e_driver_manifest_pairs_are_explicit() -> None:
    pairs = {
        "hyperpod-dcgm-metrics-e2e.yaml": ("run_hyperpod_dcgm_metrics_e2e.py"),
        "hyperpod-efa-traffic-e2e.yaml": ("run_hyperpod_efa_traffic_e2e.py"),
        "hyperpod-three-source-fault-e2e.yaml": (
            "run_hyperpod_three_source_fault_e2e.py"
        ),
    }
    readme = (E2E_MANIFESTS / "README.md").read_text(encoding="utf-8")

    for manifest_name, driver_name in pairs.items():
        assert (E2E_MANIFESTS / manifest_name).is_file(), (
            "expected (E2E_MANIFESTS / manifest_name).is_file() to be truthy"
        )
        assert (SCRIPTS / "e2e/hyperpod" / driver_name).is_file(), (
            'expected (SCRIPTS / "e2e/hyperpod" / driver_name).is_file() to be truthy'
        )
        manifest = (E2E_MANIFESTS / manifest_name).read_text(encoding="utf-8")
        assert driver_name in manifest
        assert manifest_name in readme
        assert driver_name in readme
    assert "isolated_api.py" in readme


def test_regional_manifests_do_not_hardcode_runtime_profile() -> None:
    manifests = list(E2E_MANIFESTS.rglob("*.yaml"))
    assert manifests, "expected regional E2E manifests"
    for path in manifests:
        text = path.read_text(encoding="utf-8")
        assert "gpu-fault.io/runtime-profile-version: hyperpod-v1" not in text
        if "gpu-fault.io/runtime-profile-version:" in text:
            assert "REPLACE_WITH_RUNTIME_PROFILE_VERSION" in text


def test_python_shebang_matches_executable_mode() -> None:
    for root in (SCRIPTS, TOOLS):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            executable = bool(
                path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            )
            has_shebang = path.read_text(encoding="utf-8").startswith("#!")
            assert has_shebang == executable, path


def test_code_size_audit_has_a_reproducible_entrypoint() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    source = (SCRIPTS / "audit-code-size.py").read_text(encoding="utf-8")

    assert "code-size-audit:" in makefile
    assert '"generator": "scripts/audit-code-size.py"' in source
    assert '"schema_version": 1' in source


def test_e2e_yaml_files_live_under_manifests() -> None:
    assert not list((SCRIPTS / "e2e").glob("*.yaml")), (
        'expected list((SCRIPTS / "e2e").glob("*.yaml")) to be falsy'
    )
