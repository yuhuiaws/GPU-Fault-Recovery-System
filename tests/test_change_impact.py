from __future__ import annotations

import json
import subprocess
from pathlib import Path

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


def test_remote_command_change_does_not_select_boot_or_collect() -> None:
    plan = MODULE.build_plan(["src/gpu_fault/cluster_executor.py"], settings())

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
            "src/gpu_fault/cluster_executor.py",
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
    plan = MODULE.build_plan(["src/gpu_fault/cluster_executor.py"], settings())

    text = MODULE.render_text(plan, regional_only=True)

    assert "Run pytest:" not in text
    assert "Regional safe cases:" in text


def test_impact_plan_file_is_digest_and_base_bound(tmp_path: Path) -> None:
    plan = MODULE.build_plan(["src/gpu_fault/cluster_executor.py"], settings())
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
