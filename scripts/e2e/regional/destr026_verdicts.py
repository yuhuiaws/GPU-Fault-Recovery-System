"""Pure verdict functions of GF-REGIONAL-DESTR-026.

Two GPUs of ONE node fault inside the aggregation window while that node runs
the Master of the case's own two-node training job. Both faults reach the
control plane before the job workflow's first step runs, so the attempt-scoped
merge (``orchestration/families/grouped_faults.py``) rewrites the first
fault's still-unclaimed record in place and widens its node-action scope to
both GPUs: one incident, one linear job workflow, one RESET_GPU step naming
both GPUs with ``parameters.gpu_uuids_by_node``, one quiesce window around it,
both GPUs reset one after another by one Node Agent command
(``node_agent/operations/reset.py``: ``_reset_gpu`` loops, the display-stack
release list is shared), the job stopped once and restarted once, the second
node of the job never cordoned, quiesced or reset.

Split out of ``run_destr026_same_node_two_gpu_reset.py`` so the runner stays a
driver; everything here reads synthetic or recorded store, node and host
snapshots and touches no cluster. The checks DESTR-015 and DESTR-001 already
own (event spread, host baseline, schedulability, workload Pods, CloudTrail,
the reset contract chain) are reused, not restated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional import run_destr009_workload_restart as workload_case
from scripts.e2e.regional.acceptance_runner_common import processor_queue_backlog
from scripts.e2e.regional.destr015_verdicts import (
    AGENT_OPERATIONS,
    FORBIDDEN_OPERATIONS,
    estimated_duration_seconds,
    lifetime_errors,
)
from scripts.e2e.regional.gpu_identity_preflight import gpu_identity_errors

EXPECTED_XID = 46
TARGET_GPU_COUNT = 2
GPUS_PER_NODE = 8
# Two concurrent kmsg writes on one node land well inside the shipped 5 s
# window; a narrower one could split them before the first step is claimed.
MIN_AGGREGATION_WINDOW_SECONDS = 5
# The linear job workflow one node's reset compiles to when the job also runs
# on another node (COLLECT-016 D's shape): the reset wrapped in the job-level
# stop and restart. The second fault adds no step, it widens these.
WORKLOAD_RESET_STEPS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
]
# Node actions the agent accepts a wider target for only through the mapping.
MAPPED_NODE_ACTIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
)
# Every step that must name both faulted GPUs.
GPU_SCOPED_STEPS = (*MAPPED_NODE_ACTIONS, "VALIDATE_GPU")
# Steps that act on the fault node alone, each exactly once.
NODE_STEPS = ("MARK_UNSCHEDULABLE", *GPU_SCOPED_STEPS, "RESTORE_SCHEDULING")
# Job-level steps: once for the whole job, over both of its nodes.
JOB_STEPS = ("STOP_WORKLOADS", "RESTART_WORKLOAD")
FABRIC_MANAGER_UNIT = "nvidia-fabricmanager"
FABRIC_MANAGER_SERVICE = f"{FABRIC_MANAGER_UNIT}.service"
RESET_COMMAND = "nvidia-smi --gpu-reset -i "
RESET_OUTCOMES = frozenset({"reset", "busy"})


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def _per_node(details: Mapping[str, Any] | None, key: str, node: str) -> Any:
    value = (details or {}).get(key)
    return value.get(node) if isinstance(value, Mapping) else None


def _service_names(value: Any) -> list[str]:
    """``kept_services`` as names: the agent's dicts or the ledger's strings."""

    names = []
    for item in value if isinstance(value, list) else []:
        name = item.get("service") if isinstance(item, Mapping) else item
        if isinstance(name, str):
            names.append(name)
    return names


def succeeded_execution(
    workflow: Mapping[str, Any], operation: str
) -> dict[str, Any] | None:
    """The one SUCCEEDED execution of ``operation``, or None."""

    matches = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == operation and item.get("status") == "SUCCEEDED"
    ]
    return dict(matches[0]) if len(matches) == 1 else None


def node_receipt(
    workflow: Mapping[str, Any], operation: str, node: str
) -> dict[str, Any]:
    """The Node Agent's result for ``operation`` on ``node``, as the control
    plane recorded it (``details.node_results[node]``); empty when absent."""

    execution = succeeded_execution(workflow, operation) or {}
    receipt = _per_node(execution.get("details"), "node_results", node)
    return dict(receipt) if isinstance(receipt, Mapping) else {}


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    nodes: tuple[str, str],
    node_snapshots: Mapping[str, Mapping[str, Any]],
    agent: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    queue: Mapping[str, Any],
    remote_commands: Mapping[str, Any],
    gpu_workloads: Sequence[Mapping[str, Any]],
    business_workloads: Mapping[str, Sequence[Any]],
    recent_event: Mapping[str, Any] | None,
    predecessor: Mapping[str, Any],
    tests: Mapping[str, Any],
    control_env: Mapping[str, Any],
    runtime_identity_errors: Sequence[str],
    gpu_identity: Mapping[str, Any] | None,
    arbiter_pods: Mapping[str, Sequence[str]],
    dns_nodes: Sequence[str],
) -> list[str]:
    """``nodes`` is ``(fault node, job's second node)``; only the first is
    faulted, quiesced and reset, so only it must be free of arbiter Pods and
    must not hold every kube-dns endpoint."""

    fault, partner = nodes
    errors: list[str] = list(runtime_identity_errors)
    if fault == partner:
        errors.append("the fault node and the job's second node are identical")
    if not predecessor.get("valid"):
        errors.append("DESTR-015 predecessor evidence is not PASS")
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    for node in nodes:
        snapshot = node_snapshots.get(node) or {}
        if (
            snapshot.get("ready") != "True"
            or snapshot.get("unschedulable")
            or snapshot.get("taints")
            or snapshot.get("ownership_annotations")
        ):
            errors.append(f"{node} is not Ready, schedulable and untainted")
        if int(snapshot.get("gpu_allocatable") or 0) != GPUS_PER_NODE:
            errors.append(f"{node} does not allocate {GPUS_PER_NODE} GPUs")
        if business_workloads.get(node):
            errors.append(
                f"{node} carries a critical system workload: {business_workloads[node]}"
            )
    if arbiter_pods.get(fault):
        errors.append(
            f"{fault} hosts controller/executor arbiter Pods: {list(arbiter_pods[fault])}"
        )
    if dns_nodes and set(dns_nodes) <= {fault}:
        errors.append(
            f"{fault} holds every kube-dns endpoint; quiescing it would take "
            "cluster DNS down"
        )
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(f"{fault} Node Agent is not ACTIVE")
    missing = sorted(set(AGENT_OPERATIONS) - set(agent.get("allowed_operations") or []))
    if missing:
        errors.append(f"{fault} Node Agent does not allow {missing}")
    if recent_event is not None:
        errors.append(
            f"{fault} has a recent XID event; wait out the quiet window before "
            "writing two more"
        )
    errors.extend(
        gpu_identity_errors(
            fault,
            dict(gpu_identity) if gpu_identity is not None else None,
            boot_id=(node_snapshots.get(fault) or {}).get("boot_id"),
        )
    )
    errors.extend(workload_case.managed_owner_errors(dict(profile or {})))
    reset = workload_case.capability(dict(profile or {}), "gpuReset")
    if reset is None:
        errors.append("runtime profile has no gpuReset capability")
    elif reset.get("mode") != "OWN" or reset.get("owner") != "gpu-fault-node-agent":
        errors.append("gpuReset is not OWN by the Node Agent")
    if (profile or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if processor_queue_backlog(dict(queue)):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if gpu_workloads:
        errors.append(f"GPU cluster already has a GPU workload: {list(gpu_workloads)}")
    window = int(control_env.get("aggregation_window_seconds") or 0)
    if window < MIN_AGGREGATION_WINDOW_SECONDS:
        errors.append(
            f"aggregation window {window}s is below {MIN_AGGREGATION_WINDOW_SECONDS}s; "
            "the two faults could be split before the first step is claimed"
        )
    errors.extend(
        lifetime_errors(
            estimated_seconds=estimated_duration_seconds(),
            lifetime_seconds=control_env.get("job_lifetime_seconds"),
        )
    )
    return errors


def target_gpus(
    inventory: Sequence[Mapping[str, Any]],
    *,
    pci_bdfs: Sequence[str],
    allocation: set[str],
) -> dict[str, str]:
    """``{pci_bdf: gpu uuid}`` for the two requested BDFs; empty BDFs take the
    first two inventory entries. Refuses unless both are distinct GPUs of this
    node that the job's Master holds."""

    def normal(value: Any) -> str:
        return str(value or "").strip().lower().removesuffix(".0")

    rows = {normal(row.get("pci_bdf")): str(row.get("uuid") or "") for row in inventory}
    requested = [normal(item) for item in pci_bdfs]
    spare = iter(bdf for bdf in rows if bdf not in requested)
    chosen = [item or next(spare, "") for item in requested]
    result = {bdf: rows.get(bdf, "") for bdf in chosen}
    if len(chosen) != TARGET_GPU_COUNT or len(set(chosen)) != TARGET_GPU_COUNT:
        raise ValueError(f"the two target PCI addresses are not distinct: {chosen}")
    unknown = [bdf for bdf, uuid in result.items() if not uuid]
    if unknown:
        raise ValueError(f"no GPU of the node answers at {unknown}")
    outside = sorted(uuid for uuid in result.values() if uuid not in allocation)
    if outside:
        raise ValueError(f"the job does not hold the target GPUs {outside}")
    return result


# --------------------------------------------------------------------------- #
# Injection
# --------------------------------------------------------------------------- #
def injection_errors(
    states: Mapping[str, Mapping[str, Any]],
    *,
    node: str,
    targets: Mapping[str, str],
) -> list[str]:
    """``states[label]`` is the store read for the fault written on GPU
    ``targets[label]``; both must be XID 46 kmsg events of that GPU, finalized
    as RESET_GPU, and land in one incident and one workflow."""

    errors: list[str] = []
    requests: set[str] = set()
    incidents: set[str] = set()
    events: set[str] = set()
    decisions: dict[str, Any] = {}
    for label, gpu in targets.items():
        state = states.get(label) or {}
        event = state.get("event") or {}
        decision = state.get("decision") or {}
        incident = state.get("incident") or {}
        workflow = state.get("workflow") or {}
        decisions[label] = {
            "disposition": decision.get("disposition"),
            "official_action": decision.get("official_action"),
            "reasons": decision.get("reasons"),
        }
        if event.get("xid") != EXPECTED_XID:
            errors.append(
                f"GPU {label} event is not XID {EXPECTED_XID}: {event.get('xid')}"
            )
        if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
            errors.append(f"GPU {label} XID evidence is not backed by kmsg://")
        if event.get("node_id") != node:
            errors.append(f"GPU {label} event names node {event.get('node_id')}")
        if event.get("gpu_uuid") != gpu:
            errors.append(
                f"GPU {label} event resolved to {event.get('gpu_uuid')}, not {gpu}"
            )
        if decision.get("official_action") != "RESET_GPU":
            errors.append(
                f"GPU {label} policy did not resolve to RESET_GPU: {decisions[label]}"
            )
        request = workflow.get("request_id") or incident.get("workflow_request_id")
        if not request:
            errors.append(f"GPU {label} event has no workflow")
        requests.add(str(request))
        incidents.add(str(incident.get("incident_id")))
        events.add(str(event.get("event_id")))
    if len(events) != len(targets):
        errors.append("the two faults did not produce two distinct events")
    if len(incidents) != 1:
        errors.append(
            f"the two faults opened {len(incidents)} incidents: {sorted(incidents)}"
        )
    if len(requests) != 1:
        errors.append(
            f"the two faults landed in {len(requests)} workflows: {sorted(requests)}; "
            f"decisions: {decisions}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Control plane
# --------------------------------------------------------------------------- #
def _status_errors(
    workflow: Mapping[str, Any],
    incident: Mapping[str, Any],
    node: str,
    targets: set[str],
) -> list[str]:
    errors = []
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow status is not SUCCEEDED: {workflow.get('status')}")
    if incident.get("state") != "RECOVERED":
        errors.append(f"incident state is not RECOVERED: {incident.get('state')}")
    if set(incident.get("gpu_uuids") or []) != targets:
        errors.append(
            f"incident does not name exactly the two faulted GPUs: {incident.get('gpu_uuids')}"
        )
    if list(incident.get("node_ids") or []) != [node]:
        errors.append(
            f"incident does not name the fault node alone: {incident.get('node_ids')}"
        )
    if workflow.get("superseded_step_indexes"):
        errors.append(
            f"a single-window reset has superseded steps: {workflow['superseded_step_indexes']}"
        )
    if workflow.get("branch_escalation_counts") or workflow.get("exhausted_branch_ids"):
        errors.append(
            "the reset escalated or exhausted a branch: "
            f"{workflow.get('branch_escalation_counts')} {workflow.get('exhausted_branch_ids')}"
        )
    if workflow.get("workload_withdrawn_at") is not None:
        errors.append("the workflow was withdrawn; the job was stopped externally")
    if workflow.get("terminal_failure_reason"):
        errors.append(
            f"workflow carries a failure reason: {workflow.get('terminal_failure_reason')}"
        )
    return errors


def _scope_errors(
    steps: Sequence[Mapping[str, Any]], *, node: str, partner: str, targets: set[str]
) -> list[str]:
    errors = []
    operations = [str(step.get("operation")) for step in steps]
    for operation in (*NODE_STEPS, *JOB_STEPS):
        if operations.count(operation) != 1:
            errors.append(
                f"workflow does not contain exactly one {operation}: "
                f"{operations.count(operation)}"
            )
    forbidden = sorted(FORBIDDEN_OPERATIONS.intersection(operations))
    if forbidden:
        errors.append(f"workflow contains a node mutation beyond reset: {forbidden}")
    for step in steps:
        operation = str(step.get("operation"))
        nodes = list(step.get("node_ids") or [])
        if operation in JOB_STEPS:
            if set(nodes) != {node, partner}:
                errors.append(
                    f"{operation} does not cover exactly the job's nodes: {nodes}"
                )
            continue
        if nodes != [node]:
            errors.append(f"{operation} acts on {nodes}, not on the fault node alone")
        if operation not in GPU_SCOPED_STEPS:
            continue
        gpus = list(step.get("gpu_uuids") or [])
        if len(gpus) != len(targets) or set(gpus) != targets:
            errors.append(f"{operation} does not name both faulted GPUs: {gpus}")
        if operation not in MAPPED_NODE_ACTIONS:
            continue
        mapping = (step.get("parameters") or {}).get("gpu_uuids_by_node")
        if (
            not isinstance(mapping, Mapping)
            or set(mapping) != {node}
            or set(mapping.get(node) or []) != targets
        ):
            errors.append(
                f"{operation} carries no gpu_uuids_by_node naming both GPUs on "
                f"{node}: {mapping}"
            )
    return errors


def _execution_errors(
    steps: Sequence[Mapping[str, Any]], executions: Sequence[Mapping[str, Any]]
) -> list[str]:
    errors = []
    for index, step in enumerate(steps):
        operation = step.get("operation")
        mine = [
            item
            for item in executions
            if item.get("step_index") == index and item.get("operation") == operation
        ]
        succeeded = [item for item in mine if item.get("status") == "SUCCEEDED"]
        failed = [item for item in mine if item.get("status") == "FAILED"]
        if len(succeeded) != 1 or failed:
            errors.append(
                f"{operation} (step {index}) did not succeed exactly once: "
                f"{[item.get('status') for item in mine]}"
            )
            continue
        started = _parse(succeeded[0].get("started_at"))
        ended = _parse(succeeded[0].get("updated_at"))
        if started is None or ended is None or ended < started:
            errors.append(
                f"{operation} execution has incomplete or unordered timestamps"
            )
    return errors


def _restart_errors(
    workflow: Mapping[str, Any],
    *,
    expected_gpu_count: int,
    restart_budget: Mapping[str, Any],
) -> list[str]:
    errors = []
    restarts = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == "RESTART_WORKLOAD"
        and item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    if len(restarts) != 1:
        errors.append(
            f"RESTART_WORKLOAD does not have exactly one execution: {len(restarts)}"
        )
    else:
        details = restarts[0].get("details") or {}
        context = details.get("notification_context") or {}
        for key in ("source_gpu_count", "target_gpu_count"):
            value = details.get(key, context.get(key))
            if value != expected_gpu_count:
                errors.append(
                    f"RESTART_WORKLOAD {key} is not {expected_gpu_count}: {value}"
                )
        release = succeeded_execution(workflow, "RESTORE_SCHEDULING") or {}
        released = _parse(release.get("updated_at"))
        started = _parse(restarts[0].get("started_at"))
        if released is None or started is None or started < released:
            errors.append("RESTART_WORKLOAD started before RESTORE_SCHEDULING finished")
    count, budget = restart_budget.get("restart_count"), restart_budget.get("budget")
    if count != 1 or budget != 1:
        errors.append(
            f"restart budget did not advance exactly once: count={count} budget={budget}"
        )
    return errors


def workflow_errors(
    workflow: Mapping[str, Any],
    incident: Mapping[str, Any],
    *,
    node: str,
    partner: str,
    targets: Sequence[str],
    expected_gpu_count: int,
    restart_budget: Mapping[str, Any],
) -> list[str]:
    """One window, two GPUs, one join: the terminal record's verdict.

    The order and dependency chain of the ten steps is DESTR-001's reset
    contract (``run_destr001_gpu_reset.workflow_errors`` with
    ``WORKLOAD_RESET_STEPS``), applied by the runner; this function judges what
    makes the case: the scope of every step, one execution each, one restart.
    """

    wanted = set(targets)
    steps = list(workflow.get("official_steps") or [])
    return [
        *_status_errors(workflow, incident, node, wanted),
        *_scope_errors(steps, node=node, partner=partner, targets=wanted),
        *_execution_errors(steps, workflow.get("step_executions") or []),
        *_restart_errors(
            workflow,
            expected_gpu_count=expected_gpu_count,
            restart_budget=restart_budget,
        ),
    ]


# --------------------------------------------------------------------------- #
# Node Agent: the reset, its window, Fabric Manager
# --------------------------------------------------------------------------- #
def reset_receipt_errors(
    receipt: Mapping[str, Any] | None,
    *,
    targets: Sequence[str],
    source: str,
) -> list[str]:
    """One RESET_GPU command reset both GPUs, one after another, and released
    and reloaded the display stack at most once.

    ``receipt`` is the agent's result details: the control plane's
    ``node_results[node]`` or the ledger row's bounded ``evidence``. A GPU is
    reset when its last ``nvidia-smi --gpu-reset`` call answered ``reset``;
    ``busy`` attempts before it are the documented retry.
    """

    if not isinstance(receipt, Mapping) or not receipt:
        return [f"{source} carries no RESET_GPU result"]
    errors: list[str] = []
    reset = [str(item) for item in receipt.get("reset_gpu_uuids") or []]
    if sorted(reset) != sorted(targets) or len(set(reset)) != len(reset):
        errors.append(
            f"{source} RESET_GPU did not target exactly the two GPUs: {reset}"
        )
    if receipt.get("reset_successes") != len(targets):
        errors.append(
            f"{source} RESET_GPU reports {receipt.get('reset_successes')} successful "
            f"resets, not {len(targets)}"
        )
    calls: list[tuple[str, str]] = []
    released: list[str] = []
    for call in receipt.get("physical_calls") or []:
        command = str((call or {}).get("command") or "")
        if command.startswith(RESET_COMMAND):
            calls.append(
                (command[len(RESET_COMMAND) :].strip(), str(call.get("outcome")))
            )
        elif command.startswith("rmmod "):
            released.append(command.removeprefix("rmmod ").strip())
    foreign = sorted({gpu for gpu, _ in calls} - set(targets))
    if foreign:
        errors.append(f"{source} reset a GPU the case did not fault: {foreign}")
    for gpu in targets:
        outcomes = [outcome for target, outcome in calls if target == gpu]
        if outcomes.count("reset") != 1 or not outcomes or outcomes[-1] != "reset":
            errors.append(
                f"{source} does not show {gpu} reset exactly once: {outcomes}"
            )
    odd = sorted({outcome for _, outcome in calls} - RESET_OUTCOMES)
    if odd:
        errors.append(f"{source} carries reset outcomes other than reset/busy: {odd}")
    sequence = [
        gpu
        for index, (gpu, _) in enumerate(calls)
        if not index or calls[index - 1][0] != gpu
    ]
    if calls and sequence != reset:
        errors.append(
            f"{source} did not reset the GPUs one after another in command order: {sequence}"
        )
    if len(released) != len(set(released)):
        errors.append(f"{source} released a display module more than once: {released}")
    for key in ("released_kernel_modules", "reloaded_kernel_modules"):
        values = [str(item) for item in receipt.get(key) or []]
        if len(values) != len(set(values)):
            errors.append(f"{source} {key} names a module twice: {values}")
    return errors


def relay_errors(control: Mapping[str, Any], ledger: Mapping[str, Any]) -> list[str]:
    """The control plane relayed the reset the node's own ledger recorded."""

    def calls(receipt: Mapping[str, Any]) -> list[tuple[Any, Any, Any]]:
        return [
            (item.get("command"), item.get("attempt"), item.get("outcome"))
            for item in receipt.get("physical_calls") or []
            if isinstance(item, Mapping)
        ]

    if calls(control) != calls(ledger) or list(
        control.get("reset_gpu_uuids") or []
    ) != list(ledger.get("reset_gpu_uuids") or []):
        return [
            "the Node Agent ledger and the control plane disagree on the reset's "
            "physical calls"
        ]
    return []


def ledger_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    targets: Sequence[str],
    incident_id: str,
    workflow_request_id: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The fault node's ledger: one window for this workflow, nothing else.

    Returns the errors and this workflow's QUIESCE/RESET/RESTORE rows by
    operation, so the caller grades the reset row's ``evidence`` too.
    """

    errors: list[str] = []
    baseline = {
        (row.get("command_id"), row.get("attempt"))
        for row in before.get("ledger") or []
    }
    added = [
        row
        for row in after.get("ledger") or []
        if (row.get("command_id"), row.get("attempt")) not in baseline
    ]
    foreign = sorted(
        {str(row.get("workflow_request_id")) for row in added} - {workflow_request_id}
    )
    if foreign:
        errors.append(f"another workflow acted on the fault node: {foreign}")
    rows: dict[str, dict[str, Any]] = {}
    for operation in ("QUIESCE_GPU_SERVICES", "RESET_GPU", "RESTORE_GPU_SERVICES"):
        matches = [
            row
            for row in added
            if row.get("operation") == operation
            and row.get("workflow_request_id") == workflow_request_id
        ]
        if len(matches) != 1 or matches[0].get("state") != "SUCCEEDED":
            errors.append(
                f"Node Agent ledger does not add exactly one successful {operation}: "
                f"{[row.get('state') for row in matches]}"
            )
            continue
        rows[operation] = dict(matches[0])
    reset = rows.get("RESET_GPU")
    if reset is None:
        return errors, rows
    if reset.get("attempt") != 1:
        errors.append(f"RESET_GPU ledger attempt is not one: {reset.get('attempt')}")
    if sorted(reset.get("gpu_uuids") or []) != sorted(targets):
        errors.append(
            f"RESET_GPU ledger row does not name both GPUs: {reset.get('gpu_uuids')}"
        )
    if reset.get("incident_id") != incident_id:
        errors.append("RESET_GPU ledger incident_id does not match this workflow")
    if len(rows) == 3:
        stamps = [
            _parse(rows["QUIESCE_GPU_SERVICES"].get("completed_at")),
            _parse(reset.get("started_at")),
            _parse(reset.get("completed_at")),
            _parse(rows["RESTORE_GPU_SERVICES"].get("started_at")),
        ]
        ordered = all(
            earlier is not None and later is not None and earlier <= later
            for earlier, later in zip(stamps, stamps[1:], strict=False)
        )
        if not ordered:
            errors.append(
                "the reset did not run inside the one quiesce window on the node"
            )
    return errors, rows


def fabric_manager_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    workflow: Mapping[str, Any],
    ledger_rows: Mapping[str, Mapping[str, Any]],
    node: str,
) -> list[str]:
    """A window whose only GPU-state change is the per-GPU reset keeps Fabric
    Manager running: neither the quiesce nor the restore touches it, the
    node's ledger says so, and the unit is the same process afterwards. A node
    that runs no Fabric Manager has nothing to keep."""

    baseline = (before.get("services") or {}).get(FABRIC_MANAGER_SERVICE) or {}
    if baseline.get("ActiveState") != "active":
        return []
    errors = []
    quiesce = (succeeded_execution(workflow, "QUIESCE_GPU_SERVICES") or {}).get(
        "details"
    )
    restore = (succeeded_execution(workflow, "RESTORE_GPU_SERVICES") or {}).get(
        "details"
    )
    if FABRIC_MANAGER_UNIT not in _service_names(
        _per_node(quiesce, "kept_services", node)
    ):
        errors.append("QUIESCE_GPU_SERVICES did not keep Fabric Manager running")
    if FABRIC_MANAGER_UNIT in (_per_node(quiesce, "stopped_services", node) or []):
        errors.append("QUIESCE_GPU_SERVICES stopped Fabric Manager")
    if FABRIC_MANAGER_UNIT in (_per_node(restore, "restored_services", node) or []):
        errors.append("RESTORE_GPU_SERVICES started Fabric Manager again")
    evidence = (ledger_rows.get("QUIESCE_GPU_SERVICES") or {}).get("evidence") or {}
    if FABRIC_MANAGER_UNIT not in _service_names(evidence.get("kept_services")):
        errors.append(
            "the Node Agent ledger's quiesce row does not name Fabric Manager kept"
        )
    current = (after.get("services") or {}).get(FABRIC_MANAGER_SERVICE) or {}
    if current.get("ActiveState") != "active" or any(
        current.get(key) != baseline.get(key) for key in ("MainPID", "InvocationID")
    ):
        errors.append("Fabric Manager was stopped or restarted around the reset")
    return errors


def sampler_errors(
    inventory: Sequence[Mapping[str, Any]],
    sampler: Mapping[str, Any] | None,
    *,
    targets: Sequence[str],
) -> list[str]:
    """The detached host sampler saw each target GPU leave and the node end
    with every GPU back; no other GPU ever left (DESTR-001's physical proof,
    for two targets)."""

    uuids = {str(row.get("uuid")) for row in inventory}
    sampler = sampler or {}
    if len(uuids) != GPUS_PER_NODE or not set(targets) <= uuids:
        return ["the baseline GPU inventory does not contain both target GPUs"]
    if int(sampler.get("sample_count") or 0) < 2:
        return ["detached host GPU sampler has insufficient samples"]
    errors = []
    sets = [set(values) for values in sampler.get("observed_gpu_uuid_sets") or []]
    others = uuids - set(targets)
    if any(not others <= values for values in sets):
        errors.append("the sampler saw a GPU other than the two targets leave")
    for gpu in targets:
        if not any(gpu not in values for values in sets):
            errors.append(f"the sampler never saw {gpu} leave and return")
    last = sampler.get("last") or {}
    if set(last.get("gpu_uuids") or []) != uuids or last.get("gpu_count") != len(uuids):
        errors.append("the sampler's final GPU set differs from the baseline")
    return errors


# --------------------------------------------------------------------------- #
# Nothing else: the job's second node, every other GPU node, the job
# --------------------------------------------------------------------------- #
def partner_host_errors(
    before: Mapping[str, Any], after: Mapping[str, Any], *, node: str
) -> list[str]:
    """The job's second node: stopped and restarted with the job, never acted on."""

    errors = []
    if before.get("boot_id") != after.get("boot_id"):
        errors.append(f"{node} rebooted")
    baseline = {
        (row.get("command_id"), row.get("attempt"))
        for row in before.get("ledger") or []
    }
    added = [
        row.get("operation")
        for row in after.get("ledger") or []
        if (row.get("command_id"), row.get("attempt")) not in baseline
    ]
    if added:
        errors.append(f"{node} Node Agent ledger gained rows: {added}")
    if after.get("quiesce_states"):
        errors.append(f"{node} carries a GPU quiesce state")
    identity = {
        (row.get("uuid"), row.get("pci_bdf"))
        for row in before.get("gpu_inventory") or []
    }
    if identity != {
        (row.get("uuid"), row.get("pci_bdf"))
        for row in after.get("gpu_inventory") or []
    }:
        errors.append(f"{node} GPU inventory changed")
    for unit, state in (before.get("services") or {}).items():
        current = (after.get("services") or {}).get(unit) or {}
        if (
            state.get("ActiveState") == "active"
            and current.get("ActiveState") != "active"
        ):
            errors.append(f"{node} service did not stay active: {unit}")
    fabric = [
        (services.get(FABRIC_MANAGER_SERVICE) or {}).get("InvocationID")
        for services in (before.get("services") or {}, after.get("services") or {})
    ]
    if fabric[0] != fabric[1]:
        errors.append(f"{node} Fabric Manager was restarted")
    return errors


def other_node_errors(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    fault_node: str,
) -> list[str]:
    """Every GPU node but the fault node keeps its scheduling state."""

    current = {str(item.get("name")): item for item in after}
    errors = []
    for item in before:
        name = str(item.get("name"))
        if name == fault_node:
            continue
        now = current.get(name)
        if now is None:
            errors.append(f"{name} disappeared from the GPU node list")
            continue
        keys = [
            {str(taint.get("key")) for taint in snapshot.get("taints") or []}
            for snapshot in (item, now)
        ]
        if (
            bool(item.get("unschedulable")) != bool(now.get("unschedulable"))
            or keys[0] != keys[1]
        ):
            errors.append(f"{name} scheduling state changed: {sorted(keys[1])}")
    return errors


def attempt_errors(
    *,
    source_attempt_id: str,
    pods: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The job's attempt id changed exactly once: every restarted Pod carries
    one new attempt, and the store knows no attempt besides the two."""

    targets = {str(pod.get("attempt_id") or "") for pod in pods}
    if len(targets) != 1 or "" in targets or source_attempt_id in targets:
        return [
            f"the restarted Pods do not carry one new attempt id: {sorted(targets)}"
        ]
    known = {str(item.get("attempt_id")) for item in observations}
    if known != {source_attempt_id, *targets}:
        return [
            f"the job's attempts are not exactly the source and one restart: {sorted(known)}"
        ]
    return []
