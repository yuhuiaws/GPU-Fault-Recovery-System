from __future__ import annotations

import ast
import json
import re
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_PATTERN = re.compile(r"\bGPU_FAULT_[A-Z0-9_]+\b")
ADMIN_SPEC = ROOT / "scripts" / "admin-env-reference.yaml"
ADMIN_REFERENCE = ROOT / "docs" / "管理员环境变量参考.md"
REGIONAL_GENERATED = ROOT / "deploy" / "control-plane" / "regional" / "generated"
DEPLOY_ENV_SOURCES = (
    ROOT / "deploy" / "node" / "install-gpu-fault-collector.sh",
    ROOT / "deploy" / "node" / "verify-gpu-fault-collector.sh",
    *(ROOT / "deploy" / "systemd").glob("*"),
)


def test_environment_expression_fstrings_are_patch_version_stable() -> None:
    generator = runpy.run_path(str(ROOT / "scripts/generate-env-reference.py"))
    value = ast.parse(
        "f\"{os.environ['GPU_FAULT_CLUSTER_ID']}/{socket.gethostname()}\"", mode="eval"
    ).body

    assert generator["expression"](value) == (
        "f\"{os.environ['GPU_FAULT_CLUSTER_ID']}/{socket.gethostname()}\""
    )


def test_test_and_performance_variables_are_not_runtime_configuration() -> None:
    generator = runpy.run_path(str(ROOT / "scripts/generate-env-reference.py"))

    assert generator["concrete_name"]("GPU_FAULT_TEST_POSTGRES_URL") is False, (
        "test-only variables entered the production runtime inventory"
    )
    assert generator["concrete_name"]("GPU_FAULT_PERF_RUN_ID") is False, (
        "performance-only variables entered the production runtime inventory"
    )
    assert generator["concrete_name"]("GPU_FAULT_STORE_URL") is True, (
        "production runtime variables were excluded from the inventory"
    )


def test_value_kinds_follow_the_coercion_at_the_read_site() -> None:
    """The schema is derived from the code, so it cannot drift from it."""

    generator = runpy.run_path(str(ROOT / "scripts/generate-env-reference.py"))
    tree = ast.parse(
        textwrap.dedent(
            """
            import os

            def straight():
                return int(os.getenv("GPU_FAULT_STRAIGHT", "1"))

            def normalised():
                raw = os.environ["GPU_FAULT_NORMALISED"]
                return float(raw.strip())

            def tolerant():
                try:
                    return int(os.getenv("GPU_FAULT_TOLERANT", "1"))
                except ValueError:
                    return 1
            """
        )
    )

    by_name, _ = generator["_module_observations"](tree)

    assert by_name["GPU_FAULT_STRAIGHT"] == {("integer", frozenset())}
    assert by_name["GPU_FAULT_NORMALISED"] == {("number", frozenset())}
    assert "GPU_FAULT_TOLERANT" not in by_name, (
        "a coercion the code deliberately guards must stay unvalidated"
    )


def test_value_bounds_follow_the_guard_at_the_read_site() -> None:
    """A range is only recorded when the guarded value is the configured one.

    The refusal these bounds move to start-up is the reading code's own, so the
    inference has to stay narrower than the guards it reads. The cases below are
    the four ways it can be handed something that looks like a bound but is not:
    a value the code rescaled first, a comparison against another setting, a
    guard that only applies inside another branch, and a variable that is read
    twice with different limits.
    """

    generator = runpy.run_path(str(ROOT / "scripts/generate-env-reference.py"))
    tree = ast.parse(
        textwrap.dedent(
            """
            import os

            def bounded():
                limit = int(os.getenv("GPU_FAULT_BOUNDED", "10"))
                if not 1 <= limit <= 100:
                    raise ValueError("out of range")

            def rescaled():
                window = int(os.getenv("GPU_FAULT_RESCALED", "5")) * 1000
                if window < 5000:
                    raise ValueError("too small")

            def relative():
                ceiling = int(os.getenv("GPU_FAULT_RELATIVE", "5"))
                if ceiling < other_setting:
                    raise ValueError("inverted")

            def conditional(enabled):
                spread = int(os.getenv("GPU_FAULT_CONDITIONAL", "5"))
                if enabled:
                    if spread <= 0:
                        raise ValueError("must be positive")
            """
        )
    )

    bounds = generator["_module_bounds"](tree)
    merge = generator["_merge_bounds"]

    assert merge(bounds["GPU_FAULT_BOUNDED"], "integer") == {
        "minimum": 1,
        "maximum": 100,
    }
    for name in ("GPU_FAULT_RESCALED", "GPU_FAULT_RELATIVE", "GPU_FAULT_CONDITIONAL"):
        assert name not in bounds, f"{name} took a bound the read site does not state"

    # Two roles, two limits: refusing the stricter one at start-up would break
    # the role that can use it, so only what neither accepts is refused.
    assert merge([("minimum", 30, False), ("minimum", 1, False)], "integer") == {
        "minimum": 1
    }
    # "greater than 0" is exactly "at least 1" for an integer and has no
    # inclusive form for a real number.
    assert merge([("minimum", 0, True)], "integer") == {"minimum": 1}
    assert merge([("minimum", 0, True)], "number") == {}


def test_flag_helper_parameters_type_their_call_sites() -> None:
    """Most switches are read through a helper, not at the call site."""

    generator = runpy.run_path(str(ROOT / "scripts/generate-env-reference.py"))
    tree = ast.parse(
        textwrap.dedent(
            """
            import os

            def _enabled(name):
                return os.getenv(name, "0").strip().lower() in {"1", "true"}

            SPOOL = _enabled("GPU_FAULT_SPOOL")
            """
        )
    )

    _, by_parameter = generator["_module_observations"](tree)

    assert by_parameter[("_enabled", 0)] == {("boolean", frozenset({"1", "true"}))}
    assert generator["_named_arguments"](tree)["_enabled"] == [{0: "GPU_FAULT_SPOOL"}]


def test_conflicting_read_sites_leave_a_variable_unvalidated() -> None:
    """A guessed kind would refuse a legal production value."""

    generator = runpy.run_path(str(ROOT / "scripts/generate-env-reference.py"))
    merge = generator["_merge_observations"]

    assert merge({("integer", frozenset()), ("number", frozenset())}) == (
        "number",
        frozenset(),
    )
    assert merge({("boolean", frozenset({"true"})), ("integer", frozenset())}) is None


def test_environment_reference_matches_source() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate-env-reference.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_deployment_environment_is_known_to_python_processes() -> None:
    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    known = set(inventory["variables"])
    deployed = {
        name
        for path in DEPLOY_ENV_SOURCES
        if path.is_file()
        for name in NAME_PATTERN.findall(path.read_text(encoding="utf-8"))
    }
    assert deployed <= known, (
        "deployment code uses GPU_FAULT_* variables absent from "
        f"the runtime inventory: {sorted(deployed - known)}"
    )


def test_environment_inventory_contains_only_concrete_names() -> None:
    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    variables = inventory["variables"]

    assert all(NAME_PATTERN.fullmatch(name) for name in variables), (
        "runtime inventory contains a malformed GPU_FAULT_* name"
    )
    assert not [name for name in variables if name.endswith("_")]
    assert not set(variables).intersection(inventory["dynamic_prefixes"]), (
        "dynamic environment prefixes must not be listed as concrete variables"
    )


def test_regional_generated_environment_is_in_runtime_inventory() -> None:
    import yaml

    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    known = set(inventory["variables"])
    deployed: set[str] = set()
    for path in REGIONAL_GENERATED.glob("*.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        deployed.update(
            name
            for name in (document.get("data") or {})
            if name.startswith("GPU_FAULT_")
        )
        pod_spec = None
        kind = document.get("kind")
        if kind in {"Deployment", "DaemonSet", "Job"}:
            pod_spec = document["spec"]["template"]["spec"]
        elif kind == "CronJob":
            pod_spec = document["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        if pod_spec is None:
            continue
        for container in [
            *(pod_spec.get("initContainers") or []),
            *(pod_spec.get("containers") or []),
        ]:
            deployed.update(
                item["name"]
                for item in container.get("env", [])
                if item.get("name", "").startswith("GPU_FAULT_")
            )

    assert deployed <= known, (
        "regional generated manifests use GPU_FAULT_* variables absent "
        f"from the runtime inventory: {sorted(deployed - known)}"
    )


def test_admin_environment_reference_is_curated_from_inventory() -> None:
    import yaml

    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    spec = yaml.safe_load(ADMIN_SPEC.read_text(encoding="utf-8"))
    selected = [
        item["name"]
        for category in spec["categories"]
        for item in category["variables"]
    ]
    reference = ADMIN_REFERENCE.read_text(encoding="utf-8")

    assert len(selected) == len(set(selected))
    assert set(selected) <= set(inventory["variables"])
    assert f"精选 **{len(selected)}** 个" in reference
    assert "区域清单值" in reference
    assert "应用默认/要求" in reference
    for name in selected:
        assert f"`{name}`" in reference
