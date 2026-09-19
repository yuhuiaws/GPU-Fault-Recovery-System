"""``workflow-reconcile``: one command that plans and applies in one invocation.

The Pod builds the plan; the admin layer adds the site identity and the node
evidence only it can read, re-plans immediately before the apply and compares
field by field, and archives the plan and the result under the digest it
applied. ``--dry-run`` prints the plan and writes nothing. Every refusal below
used to be a flag combination the old ``--plan``/``--apply``/``--mode`` surface
silently ignored.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import operator_identity
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.execution import deployment_deadline, run_command
from tests.admin.conftest import TEST_OPERATOR_ARN


def _site(tmp_path: Path, **release_config) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": ("arn:aws:eks:us-west-2:123456789012:cluster/cpu-control"),
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
            **release_config,
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def _item(request_id: str = "workflow-blocked", **overrides) -> dict:
    item = {
        "request_id": request_id,
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
    item.update(overrides)
    return item


def _runtime_plan(*items: dict) -> dict:
    return {
        "schema_version": 1,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": "2026-09-03T07:00:00+00:00",
        "plan_sha256": "b" * 64,
        "items": list(items) or [_item()],
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


def _pod(calls: list[dict], plans=None):
    """A Pod double: answers plan requests from ``plans`` and applies with success."""

    queue = list(plans or [])

    def run(_site_value, payload, **_kwargs):
        calls.append(payload)
        if payload["mode"] == "plan":
            return queue.pop(0) if queue else _runtime_plan()
        return {
            "mode": "workflow-reconcile-apply",
            "records_deleted": 0,
            "applied_workflow_ids": list(payload["workflow_ids"]),
            "failed_workflow_ids": [],
            "failures": {},
        }

    return run


def _no_pod(*_args, **_kwargs):
    raise AssertionError("the Pod must not be reached before the request is validated")


def _archive_files(tmp_path: Path) -> list[Path]:
    root = tmp_path / reconcile.HISTORY_PATH
    return sorted(root.rglob("*.json")) if root.exists() else []


@pytest.fixture
def pod(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod(calls))
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )
    return calls


def test_dry_run_prints_the_plan_with_node_evidence_and_writes_nothing(
    tmp_path: Path, pod: list[dict]
) -> None:
    plan = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=("workflow-blocked",), dry_run=True
    )

    assert plan["dry_run"] is True
    assert plan["schema_version"] == 2
    assert plan["runtime_plan_sha256"] == "b" * 64
    assert plan["site_identity"]["site_sha256"] == "a" * 64
    assert plan["items"][0]["scheduling_evidence"]["restored"] is True
    assert [call["mode"] for call in pod] == ["plan"], "a dry run never applies"
    assert not (tmp_path / "workflow-reconcile").exists(), "a dry run writes nothing"


def test_one_invocation_plans_replans_applies_and_archives(
    tmp_path: Path, pod: list[dict]
) -> None:
    site = _site(tmp_path)

    result = reconcile.run_workflow_reconcile(
        site, tmp_path, workflow_ids=("workflow-blocked",), reference="CHG-12345"
    )

    assert [call["mode"] for call in pod] == ["plan", "plan", "apply"], (
        "the record is re-planned immediately before the apply"
    )
    apply_payload = pod[-1]
    assert apply_payload["workflow_ids"] == ["workflow-blocked"]
    assert apply_payload["plan_sha256"] == "b" * 64, "bound to the runtime digest"
    assert apply_payload["reference"] == "CHG-12345"
    assert apply_payload["actor"] == TEST_OPERATOR_ARN
    assert apply_payload["admin_plan_sha256"] == result["plan_sha256"]
    assert result["records_deleted"] == 0
    assert result["actor"] == TEST_OPERATOR_ARN
    assert result["reference"] == "CHG-12345"
    assert result["dry_run"] is False
    archive = tmp_path / reconcile.HISTORY_PATH / result["plan_sha256"]
    plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
    applied = json.loads((archive / "applied.json").read_text(encoding="utf-8"))
    assert plan["plan_sha256"] == result["plan_sha256"]
    assert applied["actor"] == TEST_OPERATOR_ARN, "the archive names who applied"
    assert stat.S_IMODE((archive / "plan.json").stat().st_mode) == 0o600
    assert not (tmp_path / "workflow-reconcile/plan.json").exists(), (
        "there is no pending plan file any more; the plan lives in the archive"
    )


def test_a_record_that_moved_between_plan_and_apply_is_named_by_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod(calls, [_runtime_plan(), _runtime_plan(_item(fencing_token=8))]),
    )
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    with pytest.raises(BootstrapError, match=r"workflow-blocked.*fencing_token.*7.*8"):
        reconcile.run_workflow_reconcile(
            _site(tmp_path),
            tmp_path,
            workflow_ids=("workflow-blocked",),
            reference="C-1",
        )

    assert "apply" not in {call["mode"] for call in calls}
    assert _archive_files(tmp_path) == [], "a refused apply archives nothing"


def test_node_state_that_drifts_between_plan_and_apply_refuses_the_apply(
    tmp_path: Path, pod: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    node_results = iter([_node_result(), _node_result(unschedulable=True)])
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: next(node_results)
    )

    with pytest.raises(BootstrapError, match="plan changed before apply"):
        reconcile.run_workflow_reconcile(
            _site(tmp_path),
            tmp_path,
            workflow_ids=("workflow-blocked",),
            reference="C-1",
        )

    assert "apply" not in {call["mode"] for call in pod}


def test_a_restamped_updated_at_is_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merge into the record restamps ``workflow_updated_at`` and changes
    nothing the verdict reads; hashing it made the apply unwinnable (P0-72A)."""

    calls: list[dict] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod(
            calls,
            [
                _runtime_plan(),
                _runtime_plan(_item(workflow_updated_at="2026-09-03T06:59:00+00:00")),
            ],
        ),
    )
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=("workflow-blocked",), reference="C-1"
    )

    assert result["applied_workflow_ids"] == ["workflow-blocked"]


def test_an_explicitly_named_ineligible_record_refuses_the_whole_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod(
            calls,
            [
                _runtime_plan(
                    _item(),
                    _item(
                        "workflow-live",
                        eligible=False,
                        reasons=["workflow has no source recovery plan"],
                    ),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    with pytest.raises(
        BootstrapError, match=r"refuses ineligible records.*workflow-live.*source"
    ):
        reconcile.run_workflow_reconcile(
            _site(tmp_path),
            tmp_path,
            workflow_ids=("workflow-blocked", "workflow-live"),
            reference="C-1",
        )

    assert [call["mode"] for call in calls] == ["plan"], "nothing was applied"


def test_discovery_applies_the_eligible_records_and_reports_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod(
            calls,
            [
                _runtime_plan(
                    _item(),
                    _item(
                        "workflow-live",
                        eligible=False,
                        reasons=["workflow has no source recovery plan"],
                    ),
                ),
                _runtime_plan(_item()),
            ],
        ),
    )
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path),
        tmp_path,
        incident_ids=("incident-a",),
        max_items=50,
        reference="C-1",
    )

    assert calls[0] == {
        "mode": "plan",
        "workflow_ids": [],
        "incident_ids": ["incident-a"],
        "max_items": 50,
    }
    assert calls[-1]["workflow_ids"] == ["workflow-blocked"]
    assert result["applied_workflow_ids"] == ["workflow-blocked"]
    assert result["ineligible"] == {
        "workflow-live": ["workflow has no source recovery plan"]
    }


def test_discovery_that_finds_nothing_eligible_applies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod(calls, [_runtime_plan(_item(eligible=False, reasons=["still live"]))]),
    )
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, reference="C-1"
    )

    assert [call["mode"] for call in calls] == ["plan"]
    assert result["applied_workflow_ids"] == []
    assert result["failed_workflow_ids"] == []
    assert result["records_deleted"] == 0
    assert result["ineligible"] == {"workflow-blocked": ["still live"]}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workflow_ids": ("w",)}, "requires --reference"),
        ({"workflow_ids": ("w",), "reference": "x"}, "reference is invalid"),
        (
            {"workflow_ids": ("w",), "incident_ids": ("i",), "reference": "CHG-1"},
            "do not combine them with --workflow-id",
        ),
        (
            {"workflow_ids": ("w",), "max_items": 5, "reference": "CHG-1"},
            "do not combine them with --workflow-id",
        ),
        ({"max_items": 0, "reference": "CHG-1"}, "at least 1"),
        ({"max_items": 0, "dry_run": True}, "at least 1"),
        ({"workflow_ids": (" ",), "reference": "CHG-1"}, "must not be blank"),
        ({"incident_ids": ("",), "reference": "CHG-1"}, "must not be blank"),
    ],
)
def test_flag_combinations_that_used_to_be_ignored_are_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict, message: str
) -> None:
    monkeypatch.setattr(reconcile, "_run_reconcile", _no_pod)
    monkeypatch.setattr(reconcile, "run_command", _no_pod)

    with pytest.raises(BootstrapError, match=message):
        reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, **kwargs)


def test_dry_run_needs_no_reference(tmp_path: Path, pod: list[dict]) -> None:
    plan = reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, dry_run=True)

    assert plan["dry_run"] is True


def test_control_plane_script_uses_explicit_supervised_command_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    payload = {"mode": "plan", "workflow_ids": ["workflow-a"]}

    def command(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(
            arguments, 0, "cpu-pod" if "get" in arguments else '{"items":[]}', ""
        )

    monkeypatch.setattr(reconcile, "run_command", command)
    result = reconcile.run_control_plane_script(
        _site(tmp_path, cpu_kubeconfig="/cpu/config", namespace="gpu-fault-system"),
        payload,
        script="print('test')",
    )
    assert result == {"items": []}
    assert calls[0][1] == {"timeout_seconds": 120}
    assert calls[1][1]["timeout_seconds"] == 900
    assert json.loads(calls[1][1]["input_text"]) == payload
    assert calls[1][0][-3:] == ["python", "-c", "print('test')"]
    assert not any(json.dumps(payload) in part for part in calls[1][0]), (
        "reconcile payload must be sent on stdin, not command arguments"
    )


def test_control_plane_exec_is_stopped_by_the_parent_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def command(arguments, **kwargs):
        calls.append(arguments)
        if "get" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "cpu-pod", "")
        return run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )

    monkeypatch.setattr(reconcile, "run_command", command)
    with deployment_deadline("admin exec regression", 0.2, recovery_seconds=0):
        with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
            reconcile.run_control_plane_script(
                _site(
                    tmp_path, cpu_kubeconfig="/cpu/config", namespace="gpu-fault-system"
                ),
                {"mode": "plan"},
            )
    assert len(calls) == 2


@pytest.mark.parametrize("operation", ["script", "nodes"])
def test_kubectl_failures_do_not_expose_credential_helper_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    marker = "synthetic-unstructured-auth-value-13579"

    def command(arguments, **kwargs):
        if "pod" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "cpu-pod", "")
        return subprocess.CompletedProcess(
            arguments, 1, "", f"Forbidden: exec helper failed\n{marker}"
        )

    monkeypatch.setattr(reconcile, "run_command", command)
    site = _site(tmp_path, cpu_kubeconfig="/cpu/config", namespace="gpu-fault-system")
    with pytest.raises(BootstrapError) as failure:
        if operation == "script":
            reconcile.run_control_plane_script(site, {"mode": "plan"})
        else:
            reconcile.cluster_nodes(site, "gpu-a")
    message = str(failure.value)
    assert marker not in message
    assert "Forbidden" in message
    assert "redacted" in message


def test_a_failed_sts_lookup_falls_back_to_the_local_operator(
    tmp_path: Path, pod: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: None
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=("workflow-blocked",), reference="C-1"
    )

    assert result["actor"] == operator_identity.local_operator_identity(), (
        "an unresolved STS identity is recorded as user@host, never anonymous"
    )
    assert pod[-1]["actor"] == result["actor"]


def test_node_evidence_is_read_through_the_sites_gpu_kubeconfig_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(list(command))
        return _node_result()

    monkeypatch.setattr(reconcile, "run_command", run)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "somebody-elses.kubeconfig"))

    reconcile.cluster_nodes(_site(tmp_path), "gpu-a")
    reconcile.cluster_nodes(
        _site(tmp_path, gpu_kubeconfig=str(tmp_path / "rendered.kubeconfig")), "gpu-a"
    )

    assert commands[0][:5] == [
        "kubectl",
        "--kubeconfig",
        str(tmp_path / "gpu.kubeconfig"),
        "--context",
        "gpu-a-context",
    ]
    assert commands[1][2] == str(tmp_path / "rendered.kubeconfig"), (
        "the rendered release config's GPU kubeconfig wins over the environment"
    )


def test_a_site_without_a_gpu_kubeconfig_fails_closed_instead_of_using_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconcile, "run_command", _no_pod)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "shell.kubeconfig"))
    (Path.home() / ".kube").mkdir(exist_ok=True)
    site = _site(tmp_path)
    site.environment = {}

    with pytest.raises(BootstrapError, match="no GPU kubeconfig for cluster gpu-a"):
        reconcile.cluster_nodes(site, "gpu-a")

    assert os.environ["KUBECONFIG"].endswith("shell.kubeconfig"), (
        "the refusal must not have touched the shell's environment"
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
    assert "blocked_kinds" not in reconcile.RECONCILE_SCRIPT, (
        "--blocked-kind was removed from the command; the script must not send it"
    )


# ------------------------------------------------ verified-restore-non-plan bridge


def _non_plan_item(request_id: str = "workflow-2a1858b1", **overrides) -> dict:
    """What the deployed planner says about a job DAG's parked node branch whose
    incident a later validated restore RECOVERED: eligible for nothing but the
    plan it never had."""

    values = {
        "incident_id": "inc-xid-79",
        # The node-evidence fake answers for ``node-a``; the record's nodes
        # must be read as restored for the promotion to survive the admin's
        # scheduling check, which is the point under test elsewhere.
        "node_ids": ["node-a"],
        "fencing_token": 4,
        "execution_epoch": 3,
        "successor_workflow_id": "workflow-validated-restore-1",
        "source_plan_id": None,
        "source_plan_status": None,
        "terminalization": "verified-restore",
        "eligible": False,
        "reasons": ["workflow has no source recovery plan"],
    }
    values.update(overrides)
    return _item(request_id, **values)


def test_a_non_plan_record_with_a_verified_successor_is_promoted_in_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod(calls, plans=[_runtime_plan(_non_plan_item())]),
    )
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    plan = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=("workflow-2a1858b1",), dry_run=True
    )

    [item] = plan["items"]
    assert item["eligible"] is True, item["reasons"]
    assert item["terminalization"] == reconcile.VERIFIED_RESTORE_NON_PLAN
    assert item["reasons"] == []


@pytest.mark.parametrize(
    "changes",
    [
        {
            "reasons": [
                "workflow has no source recovery plan",
                "incident is not RECOVERED",
            ]
        },
        {"terminalization": "never-changed"},
        {"successor_workflow_id": None},
        {"source_plan_id": "plan-a"},
    ],
)
def test_only_the_exact_non_plan_shape_is_promoted(changes: dict) -> None:
    item = _non_plan_item(**changes)

    reconcile.promote_non_plan_successor(item)

    assert item["eligible"] is False
    assert item["terminalization"] != reconcile.VERIFIED_RESTORE_NON_PLAN


def test_the_bridge_applies_promoted_records_with_the_bound_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def run(_site_value, payload, **_kwargs):
        calls.append(payload)
        if payload["mode"] == "plan":
            return _runtime_plan(_non_plan_item())
        assert payload["mode"] == "apply-verified-restore", payload["mode"]
        return {
            "mode": "workflow-reconcile-apply",
            "applied_workflow_ids": [item["request_id"] for item in payload["items"]],
            "failed_workflow_ids": [],
            "failures": {},
            "resolved_plan_ids": [],
            "archive_eligible_incident_ids": ["inc-xid-79"],
            "records_deleted": 0,
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path),
        tmp_path,
        workflow_ids=("workflow-2a1858b1",),
        reference="CHG-2026-0919-05",
    )

    assert [call["mode"] for call in calls] == [
        "plan",
        "plan",
        "apply-verified-restore",
    ]
    bridge = calls[-1]
    assert bridge["items"] == [
        {
            "request_id": "workflow-2a1858b1",
            "successor_workflow_id": "workflow-validated-restore-1",
            "fencing_token": 4,
            "execution_epoch": 3,
        }
    ]
    assert bridge["reference"] == "CHG-2026-0919-05"
    assert bridge["actor"] == TEST_OPERATOR_ARN
    assert bridge["admin_plan_sha256"] == result["plan_sha256"]
    assert result["applied_workflow_ids"] == ["workflow-2a1858b1"]
    assert result["resolved_plan_ids"] == []
    assert result["archive_eligible_incident_ids"] == ["inc-xid-79"]


def test_runtime_and_bridged_records_are_applied_apart_and_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deployed apply digests exactly the ids it is handed, so the runtime
    records are re-planned on their own before their digest is sent."""

    calls: list[dict] = []
    both = _runtime_plan(_item(), _non_plan_item())
    runtime_only = _runtime_plan(_item())
    runtime_only["plan_sha256"] = "c" * 64

    def run(_site_value, payload, **_kwargs):
        calls.append(payload)
        if payload["mode"] == "plan":
            return (
                runtime_only
                if payload["workflow_ids"] == ["workflow-blocked"]
                else both
            )
        if payload["mode"] == "apply":
            return {
                "mode": "workflow-reconcile-apply",
                "applied_workflow_ids": list(payload["workflow_ids"]),
                "failed_workflow_ids": [],
                "failures": {},
                "resolved_plan_ids": ["plan-a"],
                "archive_eligible_incident_ids": ["incident-a"],
                "records_deleted": 0,
            }
        return {
            "mode": "workflow-reconcile-apply",
            "applied_workflow_ids": [],
            "failed_workflow_ids": ["workflow-2a1858b1"],
            "failures": {
                "workflow-2a1858b1": "ValueError: workflow has open remote commands"
            },
            "resolved_plan_ids": [],
            "archive_eligible_incident_ids": [],
            "records_deleted": 0,
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", run)
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_args, **_kwargs: _node_result()
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path),
        tmp_path,
        workflow_ids=("workflow-blocked", "workflow-2a1858b1"),
        reference="CHG-2026-0919-05",
    )

    assert [call["mode"] for call in calls] == [
        "plan",
        "plan",
        "plan",
        "apply",
        "apply-verified-restore",
    ]
    runtime_apply = calls[3]
    assert runtime_apply["workflow_ids"] == ["workflow-blocked"]
    assert runtime_apply["plan_sha256"] == "c" * 64, (
        "bound to the digest the deployed apply will recompute over these ids"
    )
    assert result["applied_workflow_ids"] == ["workflow-blocked"]
    assert result["failed_workflow_ids"] == ["workflow-2a1858b1"]
    assert "open remote commands" in result["failures"]["workflow-2a1858b1"]
    assert result["resolved_plan_ids"] == ["plan-a"]


def test_the_bridge_mode_supersedes_the_record_with_deployed_primitives(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``apply-verified-restore`` against an in-memory control plane: the
    resolver's eligibility minus the plan gate, ``amend_workflow`` with the
    audit event, a compare-and-set ``save_incident``; a record the resolver
    refuses for any other reason is reported, not written."""

    import io
    from datetime import datetime, timedelta, timezone

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import (
        BlockedKind,
        IncidentState,
        WorkflowEventCode,
        WorkflowEventKind,
        WorkflowOperation,
        WorkflowStatus,
    )
    from tests._builders import (
        build_store,
        fault_incident,
        workflow_request,
        workflow_step,
    )

    now = datetime.now(timezone.utc)
    store = build_store()
    parked = workflow_request(
        "workflow-2a1858b1",
        "inc-xid-79",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=4,
        execution_epoch=3,
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
        remediation_budget_claims=["claim-1"],
        updated_at=now - timedelta(hours=1),
    )
    restore = workflow_request(
        "workflow-validated-restore-1",
        "inc-xid-79",
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=4,
        completed_operations=[
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
    )
    incident = fault_incident(
        "inc-xid-79",
        "event-79",
        state=IncidentState.RECOVERED,
        workflow_request_id=restore.request_id,
        fencing_token=4,
    )
    store.save_workflow(parked)
    store.save_workflow(restore)
    store.save_incident(incident)
    owned = workflow_request(
        "workflow-owned",
        "inc-owned",
        status=WorkflowStatus.BLOCKED,
        fencing_token=1,
        execution_owner_id="executor-1",
    )
    owned_restore = workflow_request(
        "workflow-validated-restore-owned",
        "inc-owned",
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=1,
        completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
    )
    store.save_workflow(owned)
    store.save_workflow(owned_restore)
    store.save_incident(
        fault_incident(
            "inc-owned",
            "event-owned",
            state=IncidentState.RECOVERED,
            workflow_request_id=owned_restore.request_id,
            fencing_token=1,
        )
    )
    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    payload = {
        "mode": "apply-verified-restore",
        "items": [
            {
                "request_id": "workflow-2a1858b1",
                "successor_workflow_id": "workflow-validated-restore-1",
                "fencing_token": 4,
                "execution_epoch": 3,
            },
            {
                "request_id": "workflow-owned",
                "successor_workflow_id": "workflow-validated-restore-owned",
                "fencing_token": 1,
                "execution_epoch": 0,
            },
        ],
        "reference": "CHG-2026-0919-05",
        "actor": TEST_OPERATOR_ARN,
        "admin_plan_sha256": "d" * 64,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    exec(compile(reconcile.RECONCILE_SCRIPT, "<workflow-reconcile>", "exec"), {})
    result = json.loads(capsys.readouterr().out)

    assert result["applied_workflow_ids"] == ["workflow-2a1858b1"]
    assert result["failed_workflow_ids"] == ["workflow-owned"]
    assert "execution owner" in result["failures"]["workflow-owned"]
    assert result["resolved_plan_ids"] == []
    assert result["archive_eligible_incident_ids"] == ["inc-xid-79"]
    closed = store.get_workflow("workflow-2a1858b1")
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id == "workflow-validated-restore-1"
    assert closed.remediation_budget_claims == []
    assert "after verified restore workflow-validated-restore-1" in (
        closed.preemption_reason or ""
    )
    event = closed.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.code == WorkflowEventCode.OPERATOR_RECONCILED.value
    assert event.actor == TEST_OPERATOR_ARN
    assert event.details["terminalization"] == reconcile.VERIFIED_RESTORE_NON_PLAN
    assert event.details["admin_plan_sha256"] == "d" * 64
    assert event.details["previous_status"] == "BLOCKED"
    assert any(
        "CHG-2026-0919-05" in reason
        for reason in store.get_incident("inc-xid-79").reasons
    ), "the incident carries the reconciliation audit"
    assert store.get_workflow("workflow-owned").status is WorkflowStatus.BLOCKED
    assert (
        store.list_active_workflow_incidents("cluster-a", node_ids={"node-a"}) == []
    ), "the superseded record no longer holds the node"
