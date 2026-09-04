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
