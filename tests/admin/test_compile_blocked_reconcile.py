"""``gpu-fault-admin workflow-reconcile --mode compile-blocked``: the shipped
script runs against the deployed image, the plan binds the site and the node
evidence, and the apply is bound to the reviewed digest."""

from __future__ import annotations

import ast
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import compile_blocked
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError

WORKFLOW = "workflow-924abfb5-be1d-481d-a0f6-72801da4007c"


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
        "cluster_id": "gpu-a",
        "node_ids": ["node-a"],
        "incident_state": "ESCALATED",
        "incident_workflow_id": "workflow-validated-restore-failed",
        "status": "BLOCKED",
        "official_action": "REMEDIATE_EFA_DRIVER",
        "blocked_reasons": ["no executable owner for efaDriverRemediation"],
        "fencing_token": 1,
        "execution_epoch": 0,
        "workflow_updated_at": "2026-09-06T09:52:44+00:00",
        "source_plan_id": None,
        "open_remote_commands": [],
        "step_execution_count": 0,
        "already_closed": False,
        "eligible": True,
        "reasons": [],
    }
    item.update(overrides)
    return item


def _runtime_plan(*items: dict) -> dict:
    return {
        "schema_version": 1,
        "mode": compile_blocked.PLAN_MODE,
        "evaluated_at": "2026-09-06T14:00:00+00:00",
        "plan_sha256": "b" * 64,
        "items": list(items) or [_item()],
    }


def _fake_runner(monkeypatch: pytest.MonkeyPatch, *items: dict, restored: bool = True):
    calls: list[dict] = []

    def run(_site_value, payload, *, script=None):
        calls.append(payload)
        if payload["mode"] == "plan":
            return _runtime_plan(*items)
        return {
            "mode": compile_blocked.APPLY_MODE,
            "records_deleted": 0,
            "applied_workflow_ids": [WORKFLOW],
            "already_closed_workflow_ids": [],
            "failed_workflow_ids": [],
            "failures": {},
            "settled_plan_sha256": "b" * 64,
        }

    def evidence(_site_value, plan_items):
        return {
            str(item["request_id"]): {
                "cluster_id": "gpu-a",
                "nodes": [],
                "restored": restored,
                "blockers": [] if restored else ["node-a: quarantine taint present"],
            }
            for item in plan_items
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    monkeypatch.setattr(reconcile, "_scheduling_evidence", evidence)
    return calls


def test_the_shipped_script_is_the_module_itself_plus_a_driver() -> None:
    script = reconcile.compile_blocked_script()
    source = Path(compile_blocked.__file__).read_text(encoding="utf-8")
    assert script.startswith(source), "script.startswith(source)"
    assert script.endswith(reconcile.COMPILE_BLOCKED_DRIVER), (
        "script.endswith(reconcile.COMPILE_BLOCKED_DRIVER)"
    )


def test_the_shipped_script_compiles_and_calls_both_entry_points() -> None:
    script = reconcile.compile_blocked_script()
    tree = ast.parse(script)
    compile(script, "<compile-blocked>", "exec")
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for entry_point in ("build_compile_blocked_plan", "apply_compile_blocked_plan"):
        assert entry_point in called, f"the driver no longer calls {entry_point}"
        assert entry_point in defined, (
            f"{entry_point} is not defined in the shipped source"
        )


def test_the_shipped_source_imports_nothing_the_old_image_may_lack() -> None:
    """This runs against the image that is *already* deployed, so it may only
    import long-standing modules; ``gpu_fault.workflow_resolution`` and
    ``gpu_fault.execution`` in particular are off limits (cycles, and shapes the
    deployed image need not have)."""

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
    tree = ast.parse(reconcile.compile_blocked_script())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= allowed, sorted(imported - allowed)


def test_plan_binds_the_site_the_runtime_digest_and_the_node_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch)
    plan = reconcile.plan_compile_blocked_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(WORKFLOW,)
    )
    assert plan["mode"] == compile_blocked.PLAN_MODE
    assert plan["runtime_plan_sha256"] == "b" * 64
    assert plan["site_identity"]["site_sha256"] == "a" * 64
    assert plan["items"][0]["eligible"] is True
    assert plan["items"][0]["scheduling_evidence"]["restored"] is True
    path = tmp_path / reconcile.COMPILE_BLOCKED_PLAN_PATH
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == plan


def test_a_node_that_still_carries_isolation_makes_the_record_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compile-time BLOCKED workflow never isolated anything, so isolation
    on the node belongs to someone else and closing the record would leave it
    unexplained."""

    _fake_runner(monkeypatch, restored=False)
    site = _site(tmp_path)
    plan = reconcile.plan_compile_blocked_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    assert plan["items"][0]["eligible"] is False
    assert (
        "GPU node scheduling state has not been restored" in plan["items"][0]["reasons"]
    )
    with pytest.raises(BootstrapError, match="ineligible records"):
        reconcile.apply_compile_blocked_reconcile(
            site,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-efa-owner",
        )


def test_apply_is_bound_to_the_reviewed_plan_and_archives_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_compile_blocked_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )

    result = reconcile.apply_compile_blocked_reconcile(
        site,
        tmp_path,
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-efa-owner",
    )

    assert calls[-1]["mode"] == "apply"
    assert calls[-1]["reference"] == "pre-deploy-efa-owner"
    assert calls[-1]["plan_sha256"] == "b" * 64
    assert calls[-1]["workflow_ids"] == [WORKFLOW]
    assert result["records_deleted"] == 0
    assert result["admin_plan_sha256"] == plan["plan_sha256"]
    archive = tmp_path / reconcile.COMPILE_BLOCKED_HISTORY_PATH / plan["plan_sha256"]
    assert (archive / "plan.json").is_file(), "(archive / 'plan.json').is_file()"
    assert (archive / "applied.json").is_file(), "(archive / 'applied.json').is_file()"
    assert not (tmp_path / reconcile.COMPILE_BLOCKED_PLAN_PATH).exists(), (
        "not (tmp_path / reconcile.COMPILE_BLOCKED_PLAN_PATH)."
    )


def test_apply_refuses_a_digest_that_matches_no_plan_and_a_bad_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch)
    site = _site(tmp_path)
    reconcile.plan_compile_blocked_reconcile(site, tmp_path, workflow_ids=(WORKFLOW,))
    with pytest.raises(BootstrapError, match="SHA-256 does not match"):
        reconcile.apply_compile_blocked_reconcile(
            site, tmp_path, expected_plan_sha256="c" * 64, reference="CHG-1"
        )
    with pytest.raises(BootstrapError, match="reference is invalid"):
        reconcile.apply_compile_blocked_reconcile(
            site, tmp_path, expected_plan_sha256="c" * 64, reference="!"
        )


def test_apply_refuses_a_plan_from_another_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch)
    site = _site(tmp_path)
    plan = reconcile.plan_compile_blocked_reconcile(
        site, tmp_path, workflow_ids=(WORKFLOW,)
    )
    path = tmp_path / reconcile.COMPILE_BLOCKED_PLAN_PATH
    tampered = dict(plan)
    tampered["mode"] = "retired-generation-plan"
    path.write_text(json.dumps(tampered))
    with pytest.raises(BootstrapError, match="not a compile-blocked plan"):
        reconcile.apply_compile_blocked_reconcile(
            site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-1"
        )


def test_plan_requires_a_workflow_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch)
    with pytest.raises(BootstrapError, match="requires at least one --workflow-id"):
        reconcile.plan_compile_blocked_reconcile(
            _site(tmp_path), tmp_path, workflow_ids=()
        )
