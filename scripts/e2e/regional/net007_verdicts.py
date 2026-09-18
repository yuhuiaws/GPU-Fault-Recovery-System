"""Pure verdict functions and constants of GF-REGIONAL-NET-007.

The case proves ARCH-B1's symmetry on a real regional deployment: a transient
Kubernetes API failure met by the GPU-plane ``gpu-fault-cluster-executor``
while it isolates a node leaves the remote command WAITING with a retryable
marker and the control-plane step WAITING, never FAILED; once the failure
clears the same step SUCCEEDs and the workflow closes as it would have
without the outage.

The outage is a ``ValidatingWebhookConfiguration`` scoped three ways -- node
``UPDATE`` only, ``objectSelector`` on the one target node, ``matchConditions``
on the executor's ServiceAccount username -- with ``failurePolicy: Fail``,
``timeoutSeconds: 1`` and a ``clientConfig.service`` that does not exist, so
the executor's node patch gets HTTP 500 "failed calling webhook". The fault
that opens the workflow is COLLECT-017 A's EFA function unbind (the only
proven workflow that both patches the node and never resets or reboots).

Every function here judges documents the runner wrote and touches no cluster.
"""

from __future__ import annotations

from typing import Any

CASE_ID = "GF-REGIONAL-NET-007"
CONFIRMATION = "NET007_EXECUTE"
# COLLECT-017 A's seven-step remediation, which the outage must not change.
EFA_REMEDIATION_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "REMEDIATE_EFA_DRIVER",
    "RESTART_EFA_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
# The steps that patch the node from the GPU-plane executor and can meet the 500.
ISOLATION_OPERATIONS = ("MARK_UNSCHEDULABLE", "RESTORE_SCHEDULING")
# What a transient outage must never turn into.
FORBIDDEN_OPERATIONS = (
    "ESCALATE_SUPPORT",
    "QUARANTINE",
    "RESTART_NODE",
    "REPLACE_NODE",
)
RETRYABLE_MARKERS = ("retryable_adapter_error", "retryable_transport_error")
WEBHOOK_HOOK_NAME = "net007.acceptance.gpu-fault.io"
ABSENT_SERVICE = "gpu-fault-net007-absent-webhook"
# The outage is held this long after the first retryable WAITING is observed,
# so the executor retries into the 500 several times before it clears.
OUTAGE_SECONDS = 60
# How long the runner waits for the isolation step to reach the webhook at all
# (the EFA finding rides the host collector's summary cadence).
EVIDENCE_TIMEOUT_SECONDS = 600
# The deadman removes the webhook on its own this long after the runner would
# have; it only ever acts for a runner that died.
DEADMAN_GRACE_SECONDS = 180
WORKFLOW_TIMEOUT_SECONDS = 600
INCIDENT_TIMEOUT_SECONDS = 180
EFA_RESTORE_SECONDS = 900
MIN_OUTAGE_SECONDS = 30
MAX_OUTAGE_SECONDS = 300


def executor_username(namespace: str, service_account: str) -> str:
    return f"system:serviceaccount:{namespace}:{service_account}"


def webhook_manifest(
    *,
    name: str,
    node: str,
    namespace: str,
    username: str,
    run_id: str,
) -> dict[str, Any]:
    """The outage, scoped to one node, one client and one verb."""

    return {
        "apiVersion": "admissionregistration.k8s.io/v1",
        "kind": "ValidatingWebhookConfiguration",
        "metadata": {
            "name": name,
            "labels": {
                "gpu-fault.io/acceptance-case": CASE_ID,
                "gpu-fault.io/acceptance-run": run_id,
            },
        },
        "webhooks": [
            {
                "name": WEBHOOK_HOOK_NAME,
                "admissionReviewVersions": ["v1"],
                "sideEffects": "None",
                "failurePolicy": "Fail",
                "timeoutSeconds": 1,
                "matchPolicy": "Equivalent",
                "rules": [
                    {
                        "apiGroups": [""],
                        "apiVersions": ["v1"],
                        "operations": ["UPDATE"],
                        "resources": ["nodes"],
                        "scope": "Cluster",
                    }
                ],
                "objectSelector": {"matchLabels": {"kubernetes.io/hostname": node}},
                "matchConditions": [
                    {
                        "name": "executor-only",
                        "expression": f"request.userInfo.username == '{username}'",
                    }
                ],
                "clientConfig": {
                    "service": {
                        "namespace": namespace,
                        "name": ABSENT_SERVICE,
                        "path": "/validate",
                        "port": 443,
                    }
                },
            }
        ],
    }


def webhook_errors(applied: dict[str, Any], *, node: str, username: str) -> list[str]:
    """The server kept every scoping the manifest asked for.

    A server that dropped ``matchConditions`` would make the webhook match
    every client's node update on that node -- kubelet included -- which is a
    different and unsafe experiment, so the case refuses to continue.
    """

    errors: list[str] = []
    hooks = applied.get("webhooks") or []
    if len(hooks) != 1:
        return [f"webhook configuration carries {len(hooks)} hooks, expected 1"]
    hook = hooks[0]
    conditions = hook.get("matchConditions") or []
    expected_condition = {
        "name": "executor-only",
        "expression": f"request.userInfo.username == '{username}'",
    }
    if conditions != [expected_condition]:
        errors.append(
            "the server did not keep the matchConditions restricted to the executor "
            "ServiceAccount; the outage would hit every client's node updates"
        )
    selector = hook.get("objectSelector")
    if selector != {"matchLabels": {"kubernetes.io/hostname": node}}:
        errors.append(f"objectSelector does not pin the node: {selector}")
    rules = hook.get("rules") or []
    if rules != [
        {
            "apiGroups": [""],
            "apiVersions": ["v1"],
            "operations": ["UPDATE"],
            "resources": ["nodes"],
            "scope": "Cluster",
        }
    ]:
        errors.append("rules are not nodes/UPDATE only in the core v1 API")
    if hook.get("failurePolicy") != "Fail":
        errors.append(f"failurePolicy {hook.get('failurePolicy')!r} != Fail")
    if int(hook.get("timeoutSeconds") or 0) != 1:
        errors.append(f"timeoutSeconds {hook.get('timeoutSeconds')!r} != 1")
    service = (hook.get("clientConfig") or {}).get("service") or {}
    identity = username.split(":")
    namespace = identity[2] if len(identity) == 4 else None
    if service != {
        "name": ABSENT_SERVICE,
        "namespace": namespace,
        "path": "/validate",
        "port": 443,
    } or (hook.get("clientConfig") or {}).get("url"):
        errors.append("clientConfig does not point at the absent Service")
    return errors


def retryable_marker(details: dict[str, Any]) -> str | None:
    for marker in RETRYABLE_MARKERS:
        if details.get(marker) is True:
            return marker
    return None


def outage_evidence(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first sample in which an isolation step is riding out the 500.

    A sample is what the runner read from the control plane at one instant:
    the workflow's ``step_executions`` and the workflow's remote commands. The
    proof needs both halves at once -- the remote command WAITING with a
    retryable marker and the step WAITING rather than FAILED.
    """

    for sample in samples:
        workflow_id = sample.get("workflow_request_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            continue
        steps = {
            item.get("step_index"): item
            for item in sample.get("step_executions") or []
            if type(item.get("step_index")) is int
        }
        for command in sample.get("remote_commands") or []:
            operation = str(command.get("operation"))
            if operation not in ISOLATION_OPERATIONS:
                continue
            if command.get("status") != "WAITING":
                continue
            marker = retryable_marker(command.get("result_details") or {})
            if marker is None:
                continue
            step = steps.get(command.get("step_index")) or {}
            command_id = command.get("command_id")
            if (
                isinstance(command_id, str)
                and command_id
                and step.get("operation") == operation
                and step.get("status") == "WAITING"
                and (step.get("details") or {}).get("remote_command_id") == command_id
            ):
                return {
                    "observed_at": sample.get("observed_at"),
                    "operation": operation,
                    "marker": marker,
                    "command_id": command_id,
                    "workflow_request_id": workflow_id,
                    "step_index": command["step_index"],
                    "status_source": (command.get("result_details") or {}).get(
                        "status_source"
                    ),
                }
    return None


def outage_errors(samples: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if outage_evidence(samples) is None:
        errors.append(
            "no sample showed an isolation step WAITING on a retryable marker "
            "while the webhook was in place; the outage was never met or the "
            "executor did not classify it as retryable"
        )
    for sample in samples:
        for item in sample.get("step_executions") or []:
            if item.get("status") == "FAILED":
                errors.append(
                    f"step {item.get('operation')} FAILED during the outage "
                    "(a transient 500 must wait, not fail)"
                )
                break
        else:
            continue
        break
    return errors


def recovery_errors(
    bundle: dict[str, Any],
    *,
    waited_operation: str | None,
    waited_evidence: dict[str, Any] | None = None,
) -> list[str]:
    """After the webhook is gone the same step succeeds and the workflow
    closes exactly as COLLECT-017 A does."""

    errors: list[str] = []
    workflow = bundle.get("workflow") or {}
    incident = bundle.get("incident") or {}
    operations = [str(s.get("operation")) for s in workflow.get("official_steps") or []]
    if operations != list(EFA_REMEDIATION_STEPS):
        errors.append(f"official steps {operations} != {list(EFA_REMEDIATION_STEPS)}")
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow status {workflow.get('status')!r} != SUCCEEDED")
    executions = {
        str(item.get("operation")): item
        for item in workflow.get("step_executions") or []
    }
    if waited_operation:
        waited = executions.get(waited_operation) or {}
        if waited.get("status") != "SUCCEEDED":
            errors.append(
                f"{waited_operation} waited on the 500 but ended "
                f"{waited.get('status')!r}, not SUCCEEDED"
            )
    if waited_evidence is not None:
        matching = [
            item
            for item in bundle.get("remote_commands") or []
            if item.get("command_id") == waited_evidence.get("command_id")
        ]
        matching_steps = [
            item
            for item in workflow.get("step_executions") or []
            if item.get("step_index") == waited_evidence.get("step_index")
        ]
        if (
            not waited_evidence.get("workflow_request_id")
            or workflow.get("request_id") != waited_evidence["workflow_request_id"]
            or len(matching) != 1
            or matching[0].get("status") != "SUCCEEDED"
            or matching[0].get("operation") != waited_operation
            or matching[0].get("step_index") != waited_evidence.get("step_index")
            or len(matching_steps) != 1
            or matching_steps[0].get("operation") != waited_operation
            or matching_steps[0].get("status") != "SUCCEEDED"
            or matching_steps[0].get("adapter_operation_id")
            != f"remote/{waited_evidence.get('command_id')}"
        ):
            errors.append(
                "recovery did not confirm the same waiting workflow, step and command"
            )
    if incident.get("state") != "RECOVERED":
        errors.append(f"incident state {incident.get('state')!r} != RECOVERED")
    matches = bundle.get("matches") or [bundle]
    if len(matches) != 1:
        errors.append(
            f"the node grew {len(matches)} workflows after the injection; the "
            "outage must not spawn an escalation chain"
        )
    for match in matches:
        chain = match.get("workflow") or {}
        seen = {str(s.get("operation")) for s in chain.get("official_steps") or []}
        seen |= {str(e.get("operation")) for e in chain.get("step_executions") or []}
        forbidden = sorted(seen & set(FORBIDDEN_OPERATIONS))
        if forbidden:
            errors.append(f"forbidden operations appeared: {forbidden}")
    return errors


def residual_errors(residuals: dict[str, bool]) -> list[str]:
    left = sorted(name for name, present in residuals.items() if present)
    return [f"outage resources remain: {left}"] if left else []


def provider_errors(events: list[dict[str, Any]]) -> list[str]:
    verbs = sorted(
        {str(item.get("event_name") or item.get("EventName")) for item in events}
    )
    return [f"provider mutation verbs appeared: {verbs}"] if events else []


def node_final_errors(snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if snapshot.get("ready") != "True":
        errors.append("target node is not Ready after the case")
    if snapshot.get("unschedulable"):
        errors.append("target node was left cordoned")
    if snapshot.get("ownership_annotations"):
        errors.append("target node was left with workflow ownership")
    if any(
        str(taint.get("key") or "").startswith("gpu-fault.io/")
        for taint in snapshot.get("taints") or []
    ):
        errors.append("target node was left with a gpu-fault taint")
    return errors


def stop_conditions() -> list[str]:
    return [
        "target node is not idle, Ready and schedulable",
        "the deployed cluster executor lacks the retryable-adapter-error classifier",
        "the server drops matchConditions (the outage would hit every client)",
        "the deadman Job cannot be created before the webhook",
        "no isolation step reaches the webhook within the evidence timeout",
        "the workflow does not close SUCCEEDED / incident RECOVERED after the outage",
        "any outage resource, EFA unbind or node isolation remains after cleanup",
    ]
