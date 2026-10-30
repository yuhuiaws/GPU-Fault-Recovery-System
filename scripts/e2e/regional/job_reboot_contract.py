"""A running job's faulted node, rebooted by its job workflow (DESTR-028, DESTR-029).

Both cases hit ONE node of a running PyTorchJob with a reboot-class fault. The
product recovers it with one merged job workflow -- freeze, cordon,
STOP_WORKLOADS for the job, RESTART_NODE on the faulted node, DESTR-002's three
validations, RESTORE_SCHEDULING, then RESTART_WORKLOAD for the job -- while the
job's healthy peers are only stopped and restarted with it, never rebooted.

The faulted node is judged with DESTR-002's reboot proof minus its provider
half: the HyperPod / EC2 submission record is DESTR-002's own subject, and these
cases read the store without naming a HyperPod cluster, so their state carries
none. Everything else -- the event, the policy action, the exact step order, one
successful execution per step, no physical GPU operation, the Node Agent back
ACTIVE with a provably new boot and the same artifact, the Kubernetes Node's new
boot id, unchanged UID and restored GPU capacity -- is the same proof.

The managed workload is named after its job id (``name_managed_workload``): the
product's job token is ``job:<cluster>:<namespace>/pytorchjob/<name>``, so a
case can only expect the token of the name it really submitted, and two jobs
rendered from one manifest no longer collide on its fixed name.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot_case
from scripts.e2e.regional.managed_workload_fixture import name_managed_workload

#: The job workflow's official steps the proof names, in order. FREEZE_EVIDENCE
#: leads every workflow and is not part of the reboot proof.
JOB_REBOOT_OPERATIONS = (
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
)
PHYSICAL_OPERATIONS = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "REPLACE_NODE"}
)
REBOOT_XID = 79


def manifest_workload_name(manifest: Path) -> str:
    """The fixed ``metadata.name`` a job manifest ships with."""

    document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return str((document.get("metadata") or {}).get("name") or "")


def name_rendered_job(path: Path, job_id: str) -> str:
    """Rename a rendered job manifest after ``job_id`` in place; return the name."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = name_managed_workload(document, job_id)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return name


def job_workflow_errors(
    state: Mapping[str, Any],
    *,
    expected_artifact: str | None,
    expected_boot_id: str | None,
) -> list[str]:
    """The job workflow's reboot proof on its faulted node (no provider half)."""

    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    agent = state.get("agent") or {}
    errors: list[str] = []
    if event.get("xid") != REBOOT_XID:
        errors.append(f"matched event is not XID {REBOOT_XID}")
    if decision.get("action") != reboot_case.EXPECTED_ACTION:
        errors.append(
            f"policy did not resolve XID {REBOOT_XID} to {reboot_case.EXPECTED_ACTION}"
        )
    if workflow.get("status") != "SUCCEEDED":
        errors.append("job reboot workflow is not SUCCEEDED")
    operations = [
        str(item.get("operation")) for item in workflow.get("official_steps") or []
    ]
    executions = workflow.get("step_executions") or []
    for operation in JOB_REBOOT_OPERATIONS:
        planned = operations.count(operation)
        if planned != 1:
            errors.append(
                f"job reboot workflow plans {operation} {planned} times, not once"
            )
        succeeded = [
            item
            for item in executions
            if item.get("operation") == operation and item.get("status") == "SUCCEEDED"
        ]
        if len(succeeded) != 1:
            errors.append(
                f"job reboot workflow lacks one successful {operation} execution"
            )
    if [item for item in operations if item in JOB_REBOOT_OPERATIONS] != list(
        JOB_REBOOT_OPERATIONS
    ):
        errors.append(
            "job reboot workflow does not stop the job, reboot, validate, restore "
            "scheduling and restart the job in that order"
        )
    unexpected = sorted(PHYSICAL_OPERATIONS & set(operations))
    if unexpected:
        errors.append(f"job reboot workflow contains physical operations {unexpected}")
    if expected_artifact and agent.get("artifact_sha256") != expected_artifact:
        errors.append("Node Agent artifact changed across reboot")
    if expected_boot_id and (
        not agent.get("boot_id") or agent.get("boot_id") == expected_boot_id
    ):
        errors.append("fleet Agent boot ID did not provably change")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("Node Agent did not return ACTIVE")
    return errors


def faulted_node_errors(
    state: Mapping[str, Any],
    *,
    node_before: Mapping[str, Any],
    node_after: Mapping[str, Any],
    store_before: Mapping[str, Any] | None,
) -> list[str]:
    """The faulted node: the job workflow's proof and the Kubernetes Node after it."""

    agent = (store_before or {}).get("agent") or {}
    errors = job_workflow_errors(
        state,
        expected_artifact=agent.get("artifact_sha256"),
        expected_boot_id=agent.get("boot_id"),
    )
    errors.extend(
        reboot_case.node_recovery_errors(
            dict(node_before), dict(node_after), dict(node_after)
        )
    )
    if node_after.get("ready") != "True":
        errors.append("the faulted node is not Ready after the job workflow")
    if node_after.get("unschedulable") or node_after.get("ownership_annotations"):
        errors.append(
            "the faulted node is still cordoned or owned after the job workflow"
        )
    return errors


def healthy_peer_errors(
    peers_before: Mapping[str, Mapping[str, Any]],
    peers_after: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """A job's healthy peer is stopped and restarted with the job, never rebooted."""

    errors: list[str] = []
    for node in sorted(peers_before):
        before, after = peers_before[node], peers_after.get(node) or {}
        old, new = before.get("boot_id"), after.get("boot_id")
        if not new:
            errors.append(f"{node}: the job's healthy peer has no readable boot id")
        elif old and new != old:
            errors.append(
                f"{node}: the job's healthy peer rebooted (boot id {old} -> {new}); "
                "it is only stopped and restarted with the job"
            )
        if after.get("ready") != "True":
            errors.append(f"{node}: the job's healthy peer is not Ready")
    return errors
