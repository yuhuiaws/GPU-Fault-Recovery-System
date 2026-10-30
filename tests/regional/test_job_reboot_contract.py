"""The faulted job node's reboot contract shared by DESTR-028 and DESTR-029.

The passing shapes are the ones the product recorded live on HyperPod (DESTR-029
attempt 39): the job workflow FREEZE_EVIDENCE, MARK_UNSCHEDULABLE,
STOP_WORKLOADS, RESTART_NODE, VALIDATE_GPU/HOST/FABRIC, RESTORE_SCHEDULING,
RESTART_WORKLOAD -- every step SUCCEEDED, no provider submission in the state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import job_reboot_contract as contract
from scripts.e2e.regional.managed_workload_fixture import managed_workload_name

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "scripts/e2e/regional/manifests/training/two-node-replace-pytorchjob.yaml"
)
LIVE_JOB_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
)


def job_state(
    steps: tuple[str, ...] = LIVE_JOB_STEPS,
    *,
    failed: tuple[str, ...] = (),
    xid: int = 79,
    action: str = "REBOOT_NODE",
    status: str = "SUCCEEDED",
    agent_boot: str = "boot-new",
) -> dict[str, Any]:
    return {
        "event": {"xid": xid},
        "decision": {"action": action},
        "workflow": {
            "status": status,
            "official_steps": [{"operation": step} for step in steps],
            "step_executions": [
                {
                    "operation": step,
                    "status": "FAILED" if step in failed else "SUCCEEDED",
                }
                for step in steps
            ],
        },
        "agent": {
            "lifecycle_state": "ACTIVE",
            "boot_id": agent_boot,
            "artifact_sha256": "artifact-1",
        },
        # What the case's store read returns: no HyperPod cluster is named, so
        # no provider submission is carried (DESTR-002 judges that half).
        "submission": {},
    }


def node(
    boot: str, *, uid: str = "uid-1", gpus: str = "8", **extra: Any
) -> dict[str, Any]:
    return {
        "boot_id": boot,
        "uid": uid,
        "gpu_allocatable": gpus,
        "ready": "True",
        "unschedulable": False,
        "ownership_annotations": {},
        **extra,
    }


STORE_BEFORE = {"agent": {"boot_id": "agent-boot-old", "artifact_sha256": "artifact-1"}}


def test_the_live_job_reboot_workflow_passes() -> None:
    errors = contract.faulted_node_errors(
        job_state(agent_boot="agent-boot-new"),
        node_before=node("boot-old"),
        node_after=node("boot-new"),
        store_before=STORE_BEFORE,
    )

    assert errors == []


def test_no_provider_submission_is_required() -> None:
    state = job_state()
    state.pop("submission")

    assert (
        contract.job_workflow_errors(
            state, expected_artifact=None, expected_boot_id=None
        )
        == []
    )


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (
            tuple(step for step in LIVE_JOB_STEPS if step != "STOP_WORKLOADS"),
            "plans STOP_WORKLOADS 0 times",
        ),
        (
            tuple(step for step in LIVE_JOB_STEPS if step != "RESTART_WORKLOAD"),
            "plans RESTART_WORKLOAD 0 times",
        ),
        (
            (
                "FREEZE_EVIDENCE",
                "MARK_UNSCHEDULABLE",
                "RESTART_WORKLOAD",
                "RESTART_NODE",
                "VALIDATE_GPU",
                "VALIDATE_HOST",
                "VALIDATE_FABRIC",
                "RESTORE_SCHEDULING",
                "STOP_WORKLOADS",
            ),
            "in that order",
        ),
        (LIVE_JOB_STEPS + ("RESET_GPU",), "physical operations ['RESET_GPU']"),
    ],
)
def test_a_wrong_job_workflow_shape_fails(
    steps: tuple[str, ...], expected: str
) -> None:
    errors = contract.job_workflow_errors(
        job_state(steps), expected_artifact=None, expected_boot_id=None
    )

    assert any(expected in error for error in errors), errors


def test_a_failed_step_or_workflow_fails() -> None:
    errors = contract.job_workflow_errors(
        job_state(failed=("RESTART_WORKLOAD",), status="FAILED"),
        expected_artifact=None,
        expected_boot_id=None,
    )

    assert "job reboot workflow is not SUCCEEDED" in errors
    assert (
        "job reboot workflow lacks one successful RESTART_WORKLOAD execution" in errors
    )


def test_the_wrong_event_or_action_fails() -> None:
    errors = contract.job_workflow_errors(
        job_state(xid=11, action="RESTART_APP"),
        expected_artifact=None,
        expected_boot_id=None,
    )

    assert "matched event is not XID 79" in errors
    assert "policy did not resolve XID 79 to REBOOT_NODE" in errors


def test_an_agent_that_kept_its_boot_or_changed_its_artifact_fails() -> None:
    errors = contract.job_workflow_errors(
        job_state(agent_boot="agent-boot-old"),
        expected_artifact="artifact-0",
        expected_boot_id="agent-boot-old",
    )

    assert "fleet Agent boot ID did not provably change" in errors
    assert "Node Agent artifact changed across reboot" in errors


def test_a_node_that_did_not_reboot_or_stays_cordoned_fails() -> None:
    errors = contract.faulted_node_errors(
        job_state(agent_boot="agent-boot-new"),
        node_before=node("boot-old"),
        node_after=node("boot-old", unschedulable=True),
        store_before=STORE_BEFORE,
    )

    assert "Kubernetes Node boot ID did not change" in errors
    assert (
        "the faulted node is still cordoned or owned after the job workflow" in errors
    )


def test_a_healthy_peer_is_never_rebooted() -> None:
    before = {"n1": node("peer-boot")}

    assert contract.healthy_peer_errors(before, {"n1": node("peer-boot")}) == []
    rebooted = contract.healthy_peer_errors(before, {"n1": node("peer-boot-2")})
    assert len(rebooted) == 1 and "healthy peer rebooted" in rebooted[0]
    unreadable = contract.healthy_peer_errors(before, {})
    assert any("no readable boot id" in item for item in unreadable), unreadable


def test_the_shipped_manifest_name_is_what_the_cases_rename() -> None:
    assert contract.manifest_workload_name(MANIFEST) == "gpu-fault-two-node-replace"


def test_a_rendered_job_is_renamed_after_its_job_id(tmp_path: Path) -> None:
    rendered = tmp_path / "job.yaml"
    rendered.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")

    name = contract.name_rendered_job(rendered, "destr029-hpb13-a40-job")

    assert name == managed_workload_name(
        "gpu-fault-two-node-replace", "destr029-hpb13-a40-job"
    )
    document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert document["metadata"]["name"] == name
    assert rendered.stat().st_mode & 0o777 == 0o600


def test_two_job_ids_never_share_a_workload_name() -> None:
    first = managed_workload_name("gpu-fault-two-node-replace", "destr028-run-a")
    second = managed_workload_name("gpu-fault-two-node-replace", "destr028-run-b")

    assert first != second
