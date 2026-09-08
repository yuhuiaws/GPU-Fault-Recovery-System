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
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import operator_identity
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: next(node_results)
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
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
        reconcile.subprocess, "run", lambda *_args, **_kwargs: _node_result()
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
    ],
)
def test_flag_combinations_that_used_to_be_ignored_are_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict, message: str
) -> None:
    monkeypatch.setattr(reconcile, "_run_reconcile", _no_pod)
    monkeypatch.setattr(reconcile.subprocess, "run", _no_pod)

    with pytest.raises(BootstrapError, match=message):
        reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, **kwargs)


def test_dry_run_needs_no_reference(tmp_path: Path, pod: list[dict]) -> None:
    plan = reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, dry_run=True)

    assert plan["dry_run"] is True


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

    monkeypatch.setattr(reconcile.subprocess, "run", run)
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
    monkeypatch.setattr(reconcile.subprocess, "run", _no_pod)
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
