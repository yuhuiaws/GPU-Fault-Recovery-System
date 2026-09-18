from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = lazy_script_module(ROOT / "scripts" / "select-affected-tests.py")


def settings():
    return MODULE.load_settings()


def test_default_change_impact_matrix_is_valid() -> None:
    value = settings()

    assert len(value.rules) >= 15
    assert value.max_domains_before_full == 3


def test_release_change_selects_only_release_domain_and_cases() -> None:
    plan = MODULE.build_plan(
        ["src/gpu_fault_release/regional_release_diff.py"], settings()
    )

    assert plan.full is False
    assert plan.domains == ("release-rollout",)
    assert "tests/regional/test_release_diff.py" in plan.pytest_targets
    assert "GF-REGIONAL-BOOT-018" in plan.safe_cases
    assert "GF-REGIONAL-BOOT-020" in plan.approval_cases
    assert "GF-REGIONAL-HA-009" in plan.approval_cases
    assert "PREEMPT" in plan.not_selected_families


def test_deploy_host_component_change_stays_in_lifecycle_domain() -> None:
    plan = MODULE.build_plan(["scripts/deploy_host_component.py"], settings())

    assert plan.full is False
    assert plan.domains == ("lifecycle-registry",)
    assert "tests/test_deploy_host_setup.py" in plan.pytest_targets
    assert plan.postgres is False
    # The admin lifecycle chain runs on a live site, so it is approval-only.
    for number in range(24, 30):
        case_id = f"GF-REGIONAL-BOOT-{number:03d}"
        assert case_id in plan.approval_cases, case_id
        assert case_id not in plan.safe_cases, case_id


def test_remote_command_change_does_not_select_boot_or_collect() -> None:
    plan = MODULE.build_plan(["src/gpu_fault/cluster_executor/executor.py"], settings())

    assert plan.full is False
    assert plan.domains == ("remote-command",)
    assert "GF-REGIONAL-CMD-001" in plan.safe_cases
    assert "GF-REGIONAL-HA-004" in plan.approval_cases
    assert "BOOT" in plan.not_selected_families
    assert "COLLECT" in plan.not_selected_families


def test_workflow_reconcile_change_selects_its_own_domain_not_full() -> None:
    """The four reconcile modules fell through to the fail-closed fallback,
    so a one-line admin tooling change scheduled the whole acceptance run."""

    for path in (
        "src/gpu_fault/workflow_reconcile.py",
        "src/gpu_fault/workflow_resolution.py",
        "src/gpu_fault/retired_generation.py",
        "src/gpu_fault/admin/workflow_reconcile.py",
    ):
        plan = MODULE.build_plan([path], settings())

        assert plan.full is False, path
        assert plan.domains == ("workflow-reconcile",), (path, plan.domains)
        assert "tests/test_workflow_reconcile.py" in plan.pytest_targets, path
        assert "tests/admin/test_admin_workflow_reconcile.py" in plan.pytest_targets
        # The only acceptance case that drives the reconcile tooling is the
        # isolated, self-provisioning PREEMPT-036; nothing here needs approval.
        assert plan.safe_cases == ("GF-REGIONAL-PREEMPT-036",), (path, plan.safe_cases)
        assert plan.approval_cases == (), (path, plan.approval_cases)


def test_unmatched_file_escalates_to_full_acceptance() -> None:
    plan = MODULE.build_plan(["unknown/new-surface.xyz"], settings())
    ordered, do_not_run = MODULE.ordered_regional_cases()

    assert plan.full is True
    assert len(plan.safe_cases) + len(plan.approval_cases) == len(ordered)
    assert not set(ordered).intersection(do_not_run), (
        "full acceptance selected a DO_NOT_RUN case"
    )
    assert any("unmatched files" in reason for reason in plan.reasons), (
        "unmatched change did not record its fail-closed reason"
    )


def test_more_than_three_domains_escalates_to_full_acceptance() -> None:
    plan = MODULE.build_plan(
        [
            "src/gpu_fault_release/regional_release_diff.py",
            "src/gpu_fault/cluster_executor/executor.py",
            "src/gpu_fault/notifications/ses.py",
            "src/gpu_fault/transport/http_client.py",
        ],
        settings(),
    )

    assert plan.full is True
    assert any("exceed limit 3" in reason for reason in plan.reasons), (
        "cross-domain escalation reason is missing"
    )


def test_changed_test_is_added_to_targeted_pytest() -> None:
    path = "tests/notifications/test_notifications.py"

    plan = MODULE.build_plan([path], settings())

    assert plan.full is False
    assert "notifications" in plan.domains
    assert path in plan.pytest_targets


def test_general_api_change_uses_targeted_fallback_before_full_fallback() -> None:
    plan = MODULE.build_plan(["src/gpu_fault/app/factory.py"], settings())

    assert plan.full is False
    assert plan.domains == ("control-plane-api",)
    assert "tests/regional/test_api.py" in plan.pytest_targets
    assert "GF-REGIONAL-BOOT-001" in plan.safe_cases


def test_impact_matrix_change_is_always_full() -> None:
    plan = MODULE.build_plan(["testcases/change-impact.yaml"], settings())

    assert plan.full is True
    assert "impact-matrix" in plan.domains
    assert any("impact-matrix" in reason for reason in plan.reasons), (
        "impact matrix changes must explain the full-test escalation"
    )


def test_fleet_contract_changes_escalate_to_full() -> None:
    expected_domains = {
        "src/gpu_fault/fleet.py": ("shared-contracts", "workflow"),
        "src/gpu_fault/fleet_deployment.py": ("release-rollout", "shared-contracts"),
    }

    for path, domains in expected_domains.items():
        plan = MODULE.build_plan([path], settings())

        assert plan.full is True, path
        assert plan.domains == domains, path
        assert plan.reasons == ("full-test domains: shared-contracts",), path


def test_case_range_expansion_preserves_padding() -> None:
    assert MODULE.expand_case_expressions(["BOOT-018..020", "GF-REGIONAL-HA-007"]) == (
        "GF-REGIONAL-BOOT-018",
        "GF-REGIONAL-BOOT-019",
        "GF-REGIONAL-BOOT-020",
        "GF-REGIONAL-HA-007",
    )


def test_git_change_discovery_unions_committed_dirty_staged_and_untracked(
    tmp_path: Path,
) -> None:
    del tmp_path
    outputs = iter(
        [
            "base\n",
            "docs/变更影响与测试选择.md\n",
            "src/b.py\n",
            "src/c.py\n",
            "src/d.py\n",
        ]
    )
    calls: list[list[str]] = []

    def runner(arguments, **kwargs):
        del kwargs
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, next(outputs), "")

    changed = MODULE.changed_files_from_git("origin/main", runner=runner)

    assert changed == ("docs/变更影响与测试选择.md", "src/b.py", "src/c.py", "src/d.py")
    assert all(call[1:3] == ["-c", "core.quotePath=false"] for call in calls), (
        "git change discovery must disable C-style path quoting"
    )


def test_regional_only_output_never_contains_execution_command() -> None:
    plan = MODULE.build_plan(["src/gpu_fault/cluster_executor/executor.py"], settings())

    text = MODULE.render_text(plan, regional_only=True)

    assert "Run pytest:" not in text
    assert "Regional safe cases:" in text


def test_impact_plan_file_is_digest_and_base_bound(tmp_path: Path) -> None:
    plan = MODULE.build_plan(["src/gpu_fault/cluster_executor/executor.py"], settings())
    path = tmp_path / "impact-plan.json"

    MODULE.write_plan_file(path, base="origin/main", plan=plan)

    assert MODULE.load_plan_file(path, expected_base="origin/main") == plan
    value = json.loads(path.read_text(encoding="utf-8"))
    value["plan"]["postgres"] = not value["plan"]["postgres"]
    path.write_text(json.dumps(value), encoding="utf-8")

    try:
        MODULE.load_plan_file(path, expected_base="origin/main")
    except MODULE.ImpactError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("tampered impact plan was accepted")


def test_make_targets_execute_pytest_but_never_execute_regional_cases() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    test_impact = makefile.split("test-impact:", 1)[1].split(
        "regional-impact-plan:", 1
    )[0]
    regional_plan = makefile.split("regional-impact-plan:", 1)[1].split(
        "impact-check:", 1
    )[0]

    assert "--execute" in test_impact
    assert "--regional-only" in regional_plan
    assert "--execute" not in regional_plan


def test_test_asset_without_a_guard_test_escalates_to_full() -> None:
    """A path that only matches the ``changed-tests`` fallback but is not an
    existing ``tests/*.py`` file (so nothing gets added to pytest and no
    regional case runs) must fail safe to the full superset instead of
    deploying with an empty, narrow selection that skips its guard tests."""

    plan = MODULE.build_plan(
        ["tests/regional/fixtures/injected-manifest.json"], settings()
    )

    assert plan.full is True
    assert any("no guard tests mapped" in reason for reason in plan.reasons), (
        "an unmapped test asset did not record its fail-safe escalation"
    )


def test_existing_test_module_still_avoids_full_escalation() -> None:
    """The fail-safe must not fire for a real ``tests/*.py`` module: it is
    added to pytest directly, so a narrow selection remains correct."""

    plan = MODULE.build_plan(["tests/notifications/test_notifications.py"], settings())

    assert plan.full is False
    assert "tests/notifications/test_notifications.py" in plan.pytest_targets
    assert not any("no guard tests mapped" in reason for reason in plan.reasons), (
        plan.reasons
    )


def execution_plan(
    targets: tuple[str, ...], *, full: bool = False, postgres: bool = False
):
    return MODULE.Plan(
        changed_files=("inputs/changed.txt",),
        domains=("metrics",),
        pytest_targets=targets,
        checks=("config-check",),
        safe_cases=(),
        approval_cases=("GF-REGIONAL-DESTR-001",),
        not_selected_families=("BOOT",),
        full=full,
        postgres=postgres,
        reasons=(),
    )


@pytest.mark.parametrize(
    "target",
    [
        "tests/native/test_future_rules.py",
        "tests/native/test_future_rules.py::test_query[one]",
        "tests/native",
        "./tests/native",
        "tests",
    ],
)
def test_selected_promql_uses_make_metadata_and_binds_only_the_pytest_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    plan = execution_plan((target,), postgres=True)
    original_plan = plan.as_dict()
    monkeypatch.delenv("PROMTOOL", raising=False)
    monkeypatch.setenv("GPU_FAULT_TEST_POSTGRES_URL", "explicit-test-reference")
    monkeypatch.delenv("PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR", raising=False)
    monkeypatch.setenv("HOME", "/original/build-home")
    monkeypatch.setenv("PGPASSFILE", "/original/pgpass")
    monkeypatch.setenv("COSIGN_PASSWORD", "example-test-only-signing-password")
    parent = dict(os.environ)
    calls: list[tuple[list[str], dict]] = []

    def run(command, **options):
        calls.append((list(command), options))
        if "promql-test-files" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(["tests/native/test_future_rules.py"])
            )
        if "promtool-preflight" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {"status": "ready", "promtool": sys.executable, "version": "3.14.0"}
                ),
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        MODULE, "subprocess", SimpleNamespace(**{**vars(subprocess), "run": run})
    )
    MODULE.execute_plan(plan, settings(), root=tmp_path)
    assert [command[3] for command, _ in calls[:2]] == [
        "promql-test-files",
        "promtool-preflight",
    ], "Make metadata and the tool-only probe must precede every selected gate"
    assert calls[2][0] == ["make", "config-check", f"PYTHON={sys.executable}"], (
        "the original static selection must remain intact"
    )
    assert "env" not in calls[2][1], (
        "static gates must retain their existing environment"
    )
    command, options = calls[3]
    assert command == [sys.executable, "-m", "pytest", "-q", target], (
        "the guard must preserve the exact pytest node/directory selection"
    )
    child = options["env"]
    assert child["PROMTOOL"] == sys.executable, (
        "pytest must receive the absolute path returned by the tool-only probe"
    )
    assert (
        child["HOME"] == parent["HOME"] and child["PGPASSFILE"] == parent["PGPASSFILE"]
    ), "binding promtool must not load the native PostgreSQL allocation"
    assert "COSIGN_PASSWORD" not in child, "selected tests cannot consume signing data"
    native, native_options = calls[4]
    assert native == ["make", "test-postgres-stress", f"PYTHON={sys.executable}"], (
        "PromQL binding cannot alter native test permissions or selection"
    )
    assert native_options["env"] == parent, (
        "only the original native environment applies"
    )
    assert dict(os.environ) == parent, (
        "a Make child's tool binding cannot mutate the parent"
    )
    assert plan.as_dict() == original_plan, (
        "the dependency guard changed acceptance risks"
    )


@pytest.mark.parametrize(
    "targets,full",
    [
        ((), False),
        (("tests/plain.py",), False),
        (("tests/native-other",), False),
        ((), True),
    ],
)
def test_unrelated_or_full_selections_do_not_gain_a_selective_tool_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targets: tuple[str, ...],
    full: bool,
) -> None:
    monkeypatch.setenv("PROMTOOL", str(tmp_path / "unavailable"))
    calls: list[tuple[list[str], dict]] = []

    def run(command, **options):
        calls.append((list(command), options))
        assert "promtool-preflight" not in command, (
            "unrelated selections must not depend on promtool; full Make owns its preflight"
        )
        return subprocess.CompletedProcess(
            command, 0, json.dumps(["tests/native/test_future_rules.py"])
        )

    monkeypatch.setattr(
        MODULE, "subprocess", SimpleNamespace(**{**vars(subprocess), "run": run})
    )
    MODULE.execute_plan(execution_plan(targets, full=full), settings(), root=tmp_path)
    execution = [
        (command, options)
        for command, options in calls
        if "promql-test-files" not in command
    ]
    expected = [
        ["make", "check" if full else "config-check", f"PYTHON={sys.executable}"]
    ]
    if targets and not full:
        expected.append([sys.executable, "-m", "pytest", "-q", *targets])
    assert [command for command, _ in execution] == expected, (
        "dependency protection must not expand test or regional execution"
    )
    assert all("env" not in options for _, options in execution), (
        "non-PromQL execution must keep its original environment contract"
    )
    assert len(calls) == len(expected) + bool(targets and not full), (
        "only nonempty selective test plans need read-only Make metadata"
    )


@pytest.mark.parametrize(
    "metadata",
    ["not-json", "null", "[]", '[""]', '{"files": ["tests/native/test_rules.py"]}'],
)
def test_invalid_make_metadata_refuses_before_selective_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: str
) -> None:
    calls: list[list[str]] = []

    def run(command, **_options):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, metadata)

    monkeypatch.setattr(
        MODULE, "subprocess", SimpleNamespace(**{**vars(subprocess), "run": run})
    )
    with pytest.raises(MODULE.ImpactError):
        MODULE.execute_plan(
            execution_plan(("tests/native",)), settings(), root=tmp_path
        )
    assert len(calls) == 1 and "promql-test-files" in calls[0], (
        "unknown dependency metadata cannot authorize static, pytest or native work"
    )


@pytest.mark.parametrize("defect", ["json", "status", "path", "version", "extra"])
def test_invalid_tool_binding_refuses_before_selective_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    binding = {"status": "ready", "promtool": sys.executable, "version": "3.14.0"}
    if defect == "status":
        binding["status"] = "checked"
    elif defect == "path":
        binding["promtool"] = "relative/promtool"
    elif defect == "version":
        binding["version"] = ""
    elif defect == "extra":
        binding["unexpected"] = "value"
    calls: list[list[str]] = []

    def run(command, **_options):
        calls.append(command)
        output = (
            json.dumps(["tests/native/test_rules.py"])
            if "promql-test-files" in command
            else ("not-json" if defect == "json" else json.dumps(binding))
        )
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(
        MODULE, "subprocess", SimpleNamespace(**{**vars(subprocess), "run": run})
    )
    with pytest.raises(MODULE.ImpactError):
        MODULE.execute_plan(
            execution_plan(("tests/native",)), settings(), root=tmp_path
        )
    assert len(calls) == 2 and "promtool-preflight" in calls[1], (
        "a zero-exit preflight without a validated binding must not launch any tests"
    )
