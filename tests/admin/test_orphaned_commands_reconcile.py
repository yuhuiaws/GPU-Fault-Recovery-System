"""``gpu-fault-admin workflow-reconcile --mode orphaned-commands``: shipped as
source, plan bound to the site, apply bound to the reviewed digest."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import orphaned_commands
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError

WORKFLOW = "workflow-f48baa91-63e4-432b-9648-1469d9e7eb39"


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": ("arn:aws:eks:us-west-2:123456789012:cluster/cpu-control"),
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def _item(**overrides) -> dict:
    item = {
        "request_id": WORKFLOW,
        "incident_id": "incident-a",
        "workflow_status": "FAILED",
        "official_action": "CHECK_MECHANICALS",
        "fencing_token": 1,
        "open_commands": [
            {
                "command_id": "remote-2cbdab99",
                "status": "WAITING",
                "operation": "CHECK_MECHANICALS",
                "step_index": 1,
                "lease_owner": None,
            }
        ],
        "eligible": True,
        "reasons": [],
    }
    item.update(overrides)
    return item


def _fake_runner(monkeypatch: pytest.MonkeyPatch, *items: dict) -> list[dict]:
    calls: list[dict] = []

    def run(_site_value, payload, *, script=None):
        calls.append(payload)
        if payload["mode"] == "plan":
            return {
                "schema_version": 1,
                "mode": orphaned_commands.PLAN_MODE,
                "evaluated_at": "2026-09-06T14:50:00+00:00",
                "plan_sha256": "b" * 64,
                "items": list(items) or [_item()],
            }
        return {
            "mode": orphaned_commands.APPLY_MODE,
            "records_deleted": 0,
            "applied_workflow_ids": [WORKFLOW],
            "cancelled_remote_commands": {WORKFLOW: {"WAITING": 1}},
            "failed_workflow_ids": [],
            "failures": {},
            "settled_plan_sha256": "b" * 64,
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    return calls


def test_the_shipped_script_compiles_calls_both_entry_points_and_imports_safely() -> (
    None
):
    script = reconcile.orphaned_commands_script()
    assert script.startswith(
        Path(orphaned_commands.__file__).read_text(encoding="utf-8")
    ), "the shipped script must be the module's own source"
    tree = ast.parse(script)
    compile(script, "<orphaned-commands>", "exec")
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for entry_point in ("build_orphaned_commands_plan", "apply_orphaned_commands_plan"):
        assert entry_point in called, f"the driver no longer calls {entry_point}"
        assert entry_point in defined, (
            f"{entry_point} is not defined in the shipped source"
        )
    allowed = {
        "__future__",
        "hashlib",
        "json",
        "sys",
        "datetime",
        "typing",
        "gpu_fault.models",
        "gpu_fault.remote_command_models",
        "gpu_fault.store",
        "gpu_fault.app",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= allowed, sorted(imported - allowed)


def test_plan_and_apply_are_bound_to_site_digest_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_orphaned_commands_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    assert plan["mode"] == orphaned_commands.PLAN_MODE, plan
    assert plan["site_identity"]["site_sha256"] == "a" * 64, plan
    assert (
        json.loads((tmp_path / reconcile.ORPHANED_COMMANDS_PLAN_PATH).read_text())
        == plan
    ), "the plan on disk differs from the returned plan"

    result = reconcile.apply_orphaned_commands_reconcile(
        site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-orphan"
    )
    assert calls[-1] == {
        "mode": "apply",
        "workflow_ids": [WORKFLOW],
        "plan_sha256": "b" * 64,
        "reference": "CHG-orphan",
    }, calls[-1]
    assert result["admin_plan_sha256"] == plan["plan_sha256"], result
    archive = tmp_path / reconcile.ORPHANED_COMMANDS_HISTORY_PATH / plan["plan_sha256"]
    assert (archive / "applied.json").is_file(), "the apply result was not archived"
    assert not (tmp_path / reconcile.ORPHANED_COMMANDS_PLAN_PATH).exists(), (
        "a consumed plan must not stay on disk"
    )


def test_apply_refuses_ineligible_items_and_foreign_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(
        monkeypatch,
        _item(
            workflow_status="RUNNING",
            eligible=False,
            reasons=["workflow is RUNNING; its open commands are live work"],
        ),
    )
    site = _site(tmp_path)
    plan = reconcile.plan_orphaned_commands_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    with pytest.raises(BootstrapError, match="ineligible records"):
        reconcile.apply_orphaned_commands_reconcile(
            site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-1"
        )
    with pytest.raises(BootstrapError, match="SHA-256 does not match"):
        reconcile.apply_orphaned_commands_reconcile(
            site, tmp_path, expected_plan_sha256="c" * 64, reference="CHG-1"
        )
    with pytest.raises(BootstrapError, match="requires at least one --workflow-id"):
        reconcile.plan_orphaned_commands_reconcile(site, tmp_path, workflow_ids=())
