from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError


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


def _runtime_plan() -> dict:
    return {
        "schema_version": 1,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": "2026-09-03T07:00:00+00:00",
        "plan_sha256": "b" * 64,
        "items": [
            {
                "request_id": "workflow-blocked",
                "incident_id": "incident-a",
                "cluster_id": "gpu-a",
                "node_ids": ["node-a"],
                "fencing_token": 7,
                "workflow_updated_at": "2026-09-03T06:00:00+00:00",
                "successor_workflow_id": "workflow-restored",
                "source_plan_id": "plan-a",
                "source_plan_status": "FAILED",
                "open_remote_commands": [],
                "waiting_step_indexes": [],
                "eligible": True,
                "reasons": [],
            }
        ],
    }


def _node_result(*, unschedulable: bool = False, quarantine: bool = False):
    taints = (
        [
            {
                "key": reconcile.QUARANTINE_TAINT,
                "value": "incident-a",
                "effect": "NoSchedule",
            }
        ]
        if quarantine
        else []
    )
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "items": [
                    {
                        "metadata": {"name": "node-a", "annotations": {}},
                        "spec": {"unschedulable": unschedulable, "taints": taints},
                    }
                ]
            }
        ),
        stderr="",
    )


def test_plan_binds_site_runtime_and_restored_node_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reconcile, "_run_reconcile", lambda *_args, **_kwargs: _runtime_plan()
    )
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )

    plan = reconcile.plan_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=("workflow-blocked",)
    )

    assert plan["schema_version"] == 2
    assert plan["runtime_plan_sha256"] == "b" * 64
    assert plan["site_identity"]["site_sha256"] == "a" * 64
    assert plan["items"][0]["scheduling_evidence"]["restored"] is True
    path = tmp_path / reconcile.PLAN_PATH
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == plan


def test_plan_rejects_residual_cordon_or_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reconcile, "_run_reconcile", lambda *_args, **_kwargs: _runtime_plan()
    )
    monkeypatch.setattr(
        reconcile.subprocess,
        "run",
        lambda *_args, **_kwargs: _node_result(unschedulable=True, quarantine=True),
    )

    plan = reconcile.plan_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=("workflow-blocked",)
    )

    item = plan["items"][0]
    assert item["eligible"] is False
    assert item["scheduling_evidence"]["restored"] is False
    assert "node remains unschedulable" in item["scheduling_evidence"]["blockers"][0]


def test_apply_revalidates_and_archives_without_deleting_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(_site_value, payload):
        calls.append(payload)
        if payload["mode"] == "plan":
            return _runtime_plan()
        return {
            "mode": "workflow-reconcile-apply",
            "records_deleted": 0,
            "applied_workflow_ids": ["workflow-blocked"],
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )
    site = _site(tmp_path)
    plan = reconcile.plan_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",)
    )

    result = reconcile.apply_workflow_reconcile(
        site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-12345"
    )

    assert calls[-1]["mode"] == "apply"
    assert calls[-1]["plan_sha256"] == "b" * 64
    assert result["records_deleted"] == 0
    assert result["admin_plan_sha256"] == plan["plan_sha256"]
    archive = tmp_path / reconcile.HISTORY_PATH / plan["plan_sha256"]
    assert (archive / "plan.json").is_file(), "reconcile plan was not archived"
    assert (archive / "applied.json").is_file(), "reconcile result was not archived"
    assert not (tmp_path / reconcile.PLAN_PATH).exists(), (
        "applied reconcile plan remained active"
    )


def test_apply_fails_closed_when_node_state_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reconcile, "_run_reconcile", lambda *_args, **_kwargs: _runtime_plan()
    )
    node_results = iter([_node_result(), _node_result(unschedulable=True)])
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: next(node_results)
    )
    site = _site(tmp_path)
    plan = reconcile.plan_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",)
    )

    with pytest.raises(BootstrapError, match="plan changed"):
        reconcile.apply_workflow_reconcile(
            site,
            tmp_path,
            expected_plan_sha256=plan["plan_sha256"],
            reference="CHG-12345",
        )


def test_a_restamped_updated_at_does_not_invalidate_the_reviewed_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admin digest re-hashes the runtime items, so it needs the same rule.

    A merge into the BLOCKED record restamps ``workflow_updated_at`` and changes
    nothing the verdict reads. Hashing it here left the apply unwinnable for the
    records the tool exists to close (P0-72A), whatever the runtime digest did.
    """

    runtime_plans = iter(
        [
            _runtime_plan(),
            dict(
                _runtime_plan(),
                items=[
                    dict(
                        _runtime_plan()["items"][0],
                        workflow_updated_at="2026-09-03T06:59:00+00:00",
                    )
                ],
            ),
        ]
    )

    def run(_site_value, payload):
        if payload["mode"] == "plan":
            return next(runtime_plans)
        return {"mode": "workflow-reconcile-apply", "applied_workflow_ids": []}

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )
    site = _site(tmp_path)
    plan = reconcile.plan_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",)
    )

    result = reconcile.apply_workflow_reconcile(
        site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-1"
    )

    assert result["admin_plan_sha256"] == plan["plan_sha256"]


def test_a_changed_plan_names_the_field_that_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_plans = iter(
        [
            _runtime_plan(),
            dict(
                _runtime_plan(),
                items=[dict(_runtime_plan()["items"][0], fencing_token=8)],
            ),
        ]
    )
    monkeypatch.setattr(
        reconcile, "_run_reconcile", lambda *_args, **_kwargs: next(runtime_plans)
    )
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )
    site = _site(tmp_path)
    plan = reconcile.plan_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",)
    )

    with pytest.raises(BootstrapError, match=r"workflow-blocked.*fencing_token.*7.*8"):
        reconcile.apply_workflow_reconcile(
            site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-1"
        )


def test_plan_forwards_incident_and_blocked_kind_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads: list[dict] = []

    def run(_site_value, payload):
        payloads.append(payload)
        return _runtime_plan()

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )

    reconcile.plan_workflow_reconcile(
        _site(tmp_path),
        tmp_path,
        workflow_ids=(),
        incident_ids=("incident-a",),
        blocked_kinds=("INTERNAL_ERROR",),
        max_items=50,
    )

    assert payloads == [
        {
            "mode": "plan",
            "workflow_ids": [],
            "incident_ids": ["incident-a"],
            "blocked_kinds": ["INTERNAL_ERROR"],
            "max_items": 50,
        }
    ]


def _apply_runner(calls: list[dict]):
    def run(_site_value, payload, **_kwargs):
        calls.append(payload)
        if payload["mode"] == "plan":
            return _runtime_plan()
        return {
            "mode": "workflow-reconcile-apply",
            "records_deleted": 0,
            "applied_workflow_ids": ["workflow-blocked"],
        }

    return run


def test_apply_attributes_the_write_to_the_resolved_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference is free text; the write must also say *who* (I1).

    The apply payload carries the STS caller identity and the admin-side plan
    digest so the Pod records both on the workflow's event, and the archived
    ``applied.json`` names the same identity.
    """

    from tests.admin.conftest import TEST_OPERATOR_ARN

    calls: list[dict] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _apply_runner(calls))
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )
    site = _site(tmp_path)
    plan = reconcile.plan_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",)
    )

    result = reconcile.apply_workflow_reconcile(
        site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-12345"
    )

    apply_payload = calls[-1]
    assert apply_payload["mode"] == "apply"
    assert apply_payload["actor"] == TEST_OPERATOR_ARN, (
        "the apply payload does not carry the operator identity"
    )
    assert apply_payload["admin_plan_sha256"] == plan["plan_sha256"], (
        "the apply payload does not carry the approved admin plan digest"
    )
    assert result["actor"] == TEST_OPERATOR_ARN
    archive = tmp_path / reconcile.HISTORY_PATH / plan["plan_sha256"]
    applied = json.loads((archive / "applied.json").read_text(encoding="utf-8"))
    assert applied["actor"] == TEST_OPERATOR_ARN, (
        "the archived result must name who applied it"
    )


def test_a_failed_identity_lookup_never_blocks_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_fault.admin import operator_identity

    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: None
    )
    calls: list[dict] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _apply_runner(calls))
    monkeypatch.setattr(
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
    )
    site = _site(tmp_path)
    plan = reconcile.plan_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",)
    )

    result = reconcile.apply_workflow_reconcile(
        site, tmp_path, expected_plan_sha256=plan["plan_sha256"], reference="CHG-12345"
    )

    assert calls[-1]["actor"] == operator_identity.UNKNOWN_IDENTITY
    assert result["actor"] == operator_identity.UNKNOWN_IDENTITY, (
        "an unresolved identity must be recorded as unknown, not block the write"
    )


def test_the_pod_script_only_forwards_what_the_deployed_apply_accepts() -> None:
    """The restore script runs the *deployed* image's apply function.

    A payload key the deployed signature does not know would make the whole
    apply fail with a TypeError at the one moment it is needed, so the script
    checks the signature before forwarding ``actor`` and ``admin_plan_sha256``,
    the same way it already does for ``waiting_ttl``.
    """

    import ast

    tree = ast.parse(reconcile.RECONCILE_SCRIPT)
    compile(reconcile.RECONCILE_SCRIPT, "<workflow-reconcile>", "exec")
    guarded_keys = {
        node.left.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
    }
    assert {"waiting_ttl", "actor", "admin_plan_sha256"} <= guarded_keys, (
        f"the script forwards a key without checking the deployed signature: "
        f"{guarded_keys}"
    )
