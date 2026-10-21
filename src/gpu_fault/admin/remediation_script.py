"""The script ``gpu-fault-admin submit-remediation`` execs in the CPU ingress Pod.

One script, four modes, chosen by ``payload["mode"]`` on stdin:

``inspect``
    the incident, its workflow, the open and the BLOCKED workflows on its
    nodes, active attempt observations, existing operator markers, the fleet
    agent records of the incident's nodes, the remote commands of those
    workflows reduced to their evidence keys, and the GPU inventory of the
    nodes (for an escalation successor also of its lineage's nodes). Every
    disposition starts here; nothing is written.
``restore``
    the validated restore of a QUARANTINED incident -- or of an ESCALATED one
    whose workflows have all ended, judged on the admin side -- for the nodes
    the admin side found still isolated by it (``node_ids``; default: every
    node), releasing ``release_only_node_ids`` (quiesce taint only) without
    validating them, exactly as the plan's preview built it. The script binds
    the state it inspected (``expected``) and refuses open workflows; it does
    not re-judge the state, so the deployed image needs no change for the
    ESCALATED exit.
``submit``
    a CHECK_MECHANICALS hardware disposition: the operator marker and the
    terminal trigger, through the completion service.
``confirm-node-action``
    the operator's confirmation of a node action whose outcome the executor
    never observed. The verdict is computed on the admin side
    (``gpu_fault.execution.node_action_confirmation``, the checkout's code)
    from this script's ``inspect`` output and the node read through the GPU
    kubeconfig; the script binds the write to the exact step record the admin
    judged, re-checks the live facts it can read itself (the fleet agent
    record against the kubelet's boot id and the recorded pre-reboot boot id;
    for a ``node_departed`` confirmation, the departure proof the admin side
    read and an Agent that is not alive), answers the step's uncertainty
    flags, records the confirmation verbatim, appends the
    ``OPERATOR_RECONCILED`` / ``NODE_ACTION_CONFIRMED`` event and writes with a
    compare-and-set.

Every write mode compares the record with what the admin side inspected
(``expected``) and exits with the drift named, so a record that moved between
plan and apply is refused rather than overwritten.

The script runs on the *deployed* image, which may be older than the checkout
that sends it. It therefore uses only primitives the running package has had
since before ``confirm-node-action`` existed: ``store.get_incident`` /
``get_workflow`` / ``get_agent`` / ``list_remote_commands`` /
``list_active_workflow_incidents`` / ``save_workflow(expected=)`` /
``save_incident_and_workflow``, ``models.record_workflow_event`` and
``models.bounded_event_details``, ``validated_restore.build_validated_restore_workflow``
(with ``node_ids``), ``node_rebinding.inventory_gpu_uuids`` and
``escalation.escalation_origin``. Nothing here
imports a module added together with the disposition, so the lever works on
the release whose parked records block the very deploy that would ship it.
The builder's ``release_only_node_ids`` is newer: it is passed only when the
plan has such nodes, and an image whose builder lacks it refuses that restore
(nothing written) rather than build one the plan did not show.
"""

from __future__ import annotations

REMEDIATION_SCRIPT = """
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    NodeMarker,
    TerminalEvent,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    bounded_event_details,
    record_workflow_event,
)

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
store = context.store
OPEN = {WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING}
EVIDENCE_KEYS = ("agent_baselines", "node_results", "stop_reboot_authorization_v1")
UNCERTAINTY_FLAGS = (
    "outcome_unknown",
    "manual_confirmation_required",
    "node_action_interrupted",
    "node_action_response_unknown",
    "ownership_permit_delivery_unknown",
)


def dump(model):
    return None if model is None else json.loads(model.model_dump_json())


def load(incident_id):
    incident = store.get_incident(incident_id)
    workflow = (
        store.get_workflow(incident.workflow_request_id)
        if incident.workflow_request_id
        else None
    )
    return incident, workflow


def workflows_of(incident, wanted):
    # The incident's workflows on its nodes that ``wanted`` accepts, the
    # pointer included.
    rows = {}
    if incident.workflow_request_id:
        current = store.get_workflow(incident.workflow_request_id)
        if wanted(current):
            rows[current.request_id] = current
    if incident.node_ids:
        for owner, workflow in store.list_active_workflow_incidents(
            incident.cluster_id, node_ids=set(incident.node_ids)
        ):
            if owner.incident_id == incident.incident_id and wanted(workflow):
                rows[workflow.request_id] = workflow
    return [rows[key] for key in sorted(rows)]


def open_workflows(incident):
    return workflows_of(incident, lambda item: item.status in OPEN)


def blocked_workflows(incident):
    # The parked records ``confirm-node-action`` answers for.
    return workflows_of(incident, lambda item: item.status is WorkflowStatus.BLOCKED)


def agent_record(cluster_id, node_id):
    try:
        return store.get_agent(cluster_id, node_id)
    except KeyError:
        return None


def evidence_of(details):
    kept = {key: details[key] for key in EVIDENCE_KEYS if key in details}
    batched = details.get("batched_results")
    if isinstance(batched, dict):
        kept["batched_results"] = {
            str(index): {
                "status": entry.get("status"),
                "details": {
                    key: entry["details"][key]
                    for key in EVIDENCE_KEYS
                    if isinstance(entry.get("details"), dict) and key in entry["details"]
                },
            }
            for index, entry in batched.items()
            if isinstance(entry, dict)
        }
    return kept


def command_evidence(workflow_ids):
    # The remote commands reduced to what the confirmation verdict reads; the
    # rows carry whole workflow and incident copies the admin side has already.
    return {
        workflow_id: [
            {
                "command_id": command.command_id,
                "step_index": command.step_index,
                "batched_step_indexes": [
                    item.step_index for item in getattr(command, "batched_steps", [])
                ],
                "status": command.status.value,
                "status_source": command.status_source,
                "fencing_token": command.fencing_token,
                "result_details": evidence_of(dict(command.result_details or {})),
            }
            for command in store.list_remote_commands(workflow_request_ids=[workflow_id])
        ]
        for workflow_id in workflow_ids
    }


def node_validations(incident):
    # SUCCEEDED workflows of the incident's cluster whose VALIDATE_GPU step names
    # one of its nodes: a later validated restore of the node under *any*
    # incident is evidence for a node action whose answer was lost (live: the
    # heal of a DCGM incident validated the node seconds after a refused reset
    # of an XID 46 incident, and the confirmation refused for want of its own
    # incident's restore). Newest first, bounded; the admin side judges them.
    nodes = set(incident.node_ids)
    found = []
    for item in store.list_workflows(statuses={WorkflowStatus.SUCCEEDED}, limit=10_000):
        if WorkflowOperation.VALIDATE_GPU not in item.completed_operations or not any(
            step.operation is WorkflowOperation.VALIDATE_GPU and nodes & set(step.node_ids)
            for step in item.official_steps
        ):
            continue
        if item.incident_id != incident.incident_id:
            try:
                other = store.get_incident(item.incident_id)
            except Exception:  # noqa: BLE001 - an unreadable incident is no evidence
                continue
            if other.cluster_id != incident.cluster_id:
                continue
        found.append(item)
    found.sort(key=lambda item: item.updated_at, reverse=True)
    return [dump(item) for item in found[:10]]


def lineage_nodes(incident):
    # An escalation successor's GPU list was gathered over its source's nodes:
    # the incidents and workflows its "<rung>-after-<workflow>" event ids lead
    # back to (bounded; an unreadable link ends the walk). Their inventories
    # tell the restore which of those GPUs sit on another node.
    from gpu_fault.orchestration.escalation import escalation_origin

    nodes = set()
    current = incident
    for _ in range(8):
        origin = escalation_origin(current.event_id)
        if origin is None:
            break
        try:
            source = store.get_workflow(origin[1])
            current = store.get_incident(source.incident_id)
        except Exception:  # noqa: BLE001 - an unreadable link ends the walk
            break
        nodes.update(current.node_ids)
        nodes.update(node for step in source.official_steps for node in step.node_ids)
    return sorted(nodes - set(incident.node_ids))


def node_inventory(incident):
    from gpu_fault.execution.node_rebinding import inventory_gpu_uuids

    return {
        node_id: inventory_gpu_uuids(store, incident.cluster_id, node_id)
        for node_id in [*incident.node_ids, *lineage_nodes(incident)]
    }


def refuse_drift(expected, observed):
    drift = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if drift:
        refuse("incident moved since inspection: " + json.dumps(drift, sort_keys=True))


def refuse(reason):
    # The refusal travels on stdout as JSON, where the admin side shows it after
    # its ordinary redaction; stderr is blacked out wholesale ("<sensitive
    # output redacted>") and left the operator guessing which rule had refused
    # (live: a reboot-confirmed RESET_GPU, refused by the rule below for want
    # of a terminal result, reported as nothing at all).
    message = "submit-remediation: " + reason
    print(json.dumps({"error": message, "error_kind": "refused"}))
    raise SystemExit(message)


def confirmed_step(workflow, executions, item, node_id, operator, reference, now):
    # Bind to the exact record the admin judged: identity, status, adapter
    # operation and details all unchanged since the inspection.
    target = item["execution"]
    positions = [
        position
        for position, execution in enumerate(executions)
        if execution.phase == target["phase"]
        and execution.step_index == target["step_index"]
        and execution.operation.value == target["operation"]
    ]
    if not positions:
        refuse(
            f"step {target['step_index']} {target['operation']} is no longer on "
            f"workflow {workflow.request_id}"
        )
    current = executions[positions[-1]]
    if (
        current.status.value != target["status"]
        or current.adapter_operation_id != target.get("adapter_operation_id")
        or dump(current)["details"] != target["details"]
    ):
        refuse(
            f"step {target['step_index']} {target['operation']} of workflow "
            f"{workflow.request_id} changed since inspection; plan again"
        )
    confirmation = dict(item["confirmation"])
    for key in ("actor", "reference", "confirmed_at", "node_id", "operation"):
        if not isinstance(confirmation.get(key), str) or not confirmation[key]:
            refuse(f"confirmation for step {target['step_index']} lacks {key}")
    if confirmation["node_id"] != node_id or confirmation["actor"] != operator:
        refuse("confirmation does not name this node and operator")
    if confirmation["reference"] != reference:
        refuse("confirmation reference differs from --reference")
    details = dict(current.details)
    for key in UNCERTAINTY_FLAGS:
        if key in details:
            details[key] = False
    if details.get("node_action_state") == "PENDING":
        details["node_action_state"] = "OPERATOR_CONFIRMED"
    details["operator_confirmed"] = confirmation
    executions[positions[-1]] = current.model_copy(
        update={"details": details, "updated_at": now}
    )
    return record_workflow_event(
        workflow,
        WorkflowEventKind.OPERATOR_RECONCILED,
        code="NODE_ACTION_CONFIRMED",
        actor=operator,
        step_index=current.step_index,
        operation=current.operation,
        phase=current.phase,
        status=workflow.status.value,
        reason=(
            f"operator confirmed {current.operation.value} outcome on {node_id} "
            f"({reference})"
        ),
        details=bounded_event_details(
            {
                key: value
                for key, value in confirmation.items()
                if key not in ("kubernetes_node", "agent", "superseded_flags")
            }
        ),
        at=now,
    )


def recheck_live_facts(incident, node_id, items, node_evidence, now):
    # What the Pod can verify itself, from the fleet record it reads here: the
    # agent is ACTIVE on a live lease and reports the boot the kubelet
    # reports, and for a reboot that boot differs from the recorded one.
    agent = agent_record(incident.cluster_id, node_id)
    if agent is None:
        refuse(f"node {node_id} has no Node Agent record in the fleet registry")
    if agent.lifecycle_state.value != "ACTIVE":
        refuse(f"node {node_id} Node Agent lifecycle state is {agent.lifecycle_state.value}")
    if agent.lease_expires_at is None or agent.lease_expires_at <= now:
        refuse(f"node {node_id} Node Agent lease expired; a fresh heartbeat is required")
    kubelet_boot = node_evidence.get("boot_id")
    if not agent.boot_id or not kubelet_boot or agent.boot_id != kubelet_boot:
        refuse(
            f"node {node_id} kubelet reports boot {kubelet_boot} but the Node Agent "
            f"reports {agent.boot_id}; the two must agree on the running boot"
        )
    for item in items:
        confirmation = item["confirmation"]
        operation = confirmation.get("operation")
        # A reboot is the proof for RESTART_NODE and, since the admin side
        # accepts it, for any node action settled by a reboot since the action
        # (``settled_by`` = reboot_since_action): no GPU reset survives a boot.
        if operation == "RESTART_NODE" or confirmation.get("settled_by") in (
            "boot_transition",
            "reboot_since_action",
        ):
            previous = confirmation.get("previous_boot_id")
            if not previous or previous == agent.boot_id:
                refuse(
                    f"node {node_id} still runs boot {agent.boot_id} recorded before "
                    f"{operation}; the reboot did not happen"
                )
        elif not confirmation.get("terminal_node_result") and not confirmation.get(
            "validated_by_workflow"
        ):
            refuse(
                f"confirmation for {operation} on {node_id} cites neither a terminal "
                "Node Agent result, nor a validating restore, nor a reboot since the action"
            )


def recheck_departure(incident, node_id, items, node_evidence, now):
    # A node_departed confirmation rests on the admin side's read of Kubernetes
    # and the provider (the departure in node_evidence); the Pod re-checks that
    # record and what it can read itself: the node's Agent is not alive -- no
    # record, REVOKED, or an expired lease with a heartbeat past the 10-minute
    # departure guard.
    departure = node_evidence.get("departure")
    if (
        node_evidence.get("exists") is not False
        or not isinstance(departure, dict)
        or departure.get("node_id") != node_id
        or departure.get("absent_from_kubernetes") is not True
        or departure.get("absent_from_provider") is not True
    ):
        refuse(f"node {node_id}: the departure proof does not show it gone")
    for item in items:
        if set(item["confirmation"].get("step_node_ids") or [node_id]) != {node_id}:
            refuse(f"a node_departed confirmation on {node_id} names other nodes")
    agent = agent_record(incident.cluster_id, node_id)
    if agent is not None and agent.lifecycle_state.value != "REVOKED":
        if agent.lease_expires_at is not None and agent.lease_expires_at > now:
            refuse(f"node {node_id} Node Agent lease has not expired; it is not departed")
        if agent.last_seen_at > now - timedelta(minutes=10):
            refuse(f"node {node_id} Node Agent was seen in the last 10 minutes")


if payload["mode"] == "inspect":
    incident, workflow = load(payload["incident_id"])
    nodes = set(incident.node_ids)
    observations = [
        dump(state)
        for state in store.list_attempt_observation_states(incident.cluster_id)
        if str(getattr(state.observation.workload_phase, "value",
                       state.observation.workload_phase)) in ("PENDING", "RUNNING")
        and any(item.node_id in nodes for item in state.observation.containers)
    ]
    blocked = blocked_workflows(incident)
    result = {
        "incident": dump(incident),
        "workflow": dump(workflow),
        "open_workflows": [dump(item) for item in open_workflows(incident)],
        "blocked_workflows": [dump(item) for item in blocked],
        "active_observations": observations,
        "existing_markers": [
            dump(item)
            for item in store.list_markers_for_incident(payload["action_incident_id"])
        ],
        "acknowledgement_timeout_seconds": (
            context.production_executor_config.step_waiting_limit(
                WorkflowOperation.CHECK_MECHANICALS
            )
        ),
        "remote_commands": command_evidence(
            sorted(
                {item.request_id for item in blocked}
                | ({workflow.request_id} if workflow is not None else set())
            )
        ),
        "agents": {
            node_id: dump(agent_record(incident.cluster_id, node_id))
            for node_id in incident.node_ids
        },
        "node_gpu_uuids": node_inventory(incident),
        "node_validations": node_validations(incident),
    }
elif payload["mode"] == "restore":
    from gpu_fault.orchestration.validated_restore import (
        build_validated_restore_workflow,
        is_validated_restore_workflow,
    )

    incident, workflow = load(payload["incident_id"])
    refuse_drift(
        payload["expected"],
        {
            "fencing_token": incident.fencing_token,
            "workflow_request_id": incident.workflow_request_id,
            "state": incident.state.value,
            "node_ids": sorted(incident.node_ids),
        },
    )
    restore_nodes = payload.get("node_ids")
    if restore_nodes is not None:
        unknown = sorted(set(restore_nodes) - set(incident.node_ids))
        if unknown or not restore_nodes:
            refuse(
                "restore targets "
                + (", ".join(unknown) if unknown else "nothing")
                + f" -- not nodes of incident {incident.incident_id}"
            )
    # The restored nodes the plan releases without validating them (only the
    # incident's quiesce taint holds them): the builder takes them as the
    # plan's preview did, so the workflow written is the one the operator saw.
    release_only = sorted(payload.get("release_only_node_ids") or [])
    targets = sorted(restore_nodes or incident.node_ids)
    stray = sorted(set(release_only) - set(targets))
    if stray:
        refuse(
            "release-only node " + ", ".join(stray) + " is not restored by this "
            f"request (restore targets: {', '.join(targets)})"
        )
    still_open = open_workflows(incident)
    existing = [
        item for item in still_open if is_validated_restore_workflow(item.request_id)
    ]
    if existing:
        result = {"no_op": True, "workflow": dump(existing[0]), "incident": dump(incident)}
    elif still_open:
        refuse(
            "incident still has an open workflow "
            + ", ".join(f"{item.request_id} ({item.status.value})" for item in still_open)
        )
    else:
        options = {}
        if release_only:
            from inspect import signature

            # An image whose builder predates the keyword validates every node it
            # restores, and its RESTORE_SCHEDULING cannot strip a quiesce taint
            # under another incident's cordon (both came in the same release).
            if "release_only_node_ids" not in signature(
                build_validated_restore_workflow
            ).parameters:
                refuse(
                    "the deployed image's validated restore validates every node it "
                    f"restores and cannot release {', '.join(release_only)} "
                    "unvalidated, as the plan does (only this incident's quiesce "
                    "taint holds it there); deploy a release that builds release-only "
                    "restores, then plan again"
                )
            options["release_only_node_ids"] = release_only
        updated, created = build_validated_restore_workflow(
            incident,
            operator=payload["operator"],
            reference=payload.get("reference"),
            now=datetime.now(timezone.utc),
            node_ids=restore_nodes,
            runtime_profile_version=payload.get("runtime_profile_version"),
            node_gpu_uuids=node_inventory(incident),
            **options,
        )
        store.save_incident_and_workflow(updated, created)
        context.dispatcher.wake()
        result = {"no_op": False, "workflow": dump(created), "incident": dump(updated)}
elif payload["mode"] == "confirm-node-action":
    incident, _pointer = load(payload["incident_id"])
    workflow = store.get_workflow(payload["workflow_request_id"])
    refuse_drift(
        payload["expected"],
        {
            "fencing_token": incident.fencing_token,
            "workflow_request_id": workflow.request_id,
            "workflow_status": workflow.status.value,
            "execution_epoch": workflow.execution_epoch,
            "merge_revision": workflow.merge_revision,
            "node_ids": sorted(incident.node_ids),
        },
    )
    if workflow.incident_id != incident.incident_id:
        refuse(f"workflow {workflow.request_id} does not belong to incident {incident.incident_id}")
    if workflow.execution_owner_id is not None:
        refuse(f"workflow {workflow.request_id} still has an execution owner")
    node_id = payload["node_id"]
    now = datetime.now(timezone.utc)
    items = list(payload.get("confirmations") or [])
    result = {"no_op": not items, "confirmations": [], "workflow": dump(workflow)}
    if items:
        if workflow.execution_lease_expires_at is not None and (
            workflow.execution_lease_expires_at > now
        ):
            refuse(f"workflow {workflow.request_id} execution lease has not expired")
        if all(
            item["confirmation"].get("settled_by") == "node_departed" for item in items
        ):
            recheck_departure(incident, node_id, items, payload["node_evidence"], now)
        else:
            recheck_live_facts(incident, node_id, items, payload["node_evidence"], now)
        executions = list(workflow.step_executions)
        updated = workflow
        for item in items:
            updated = confirmed_step(
                updated,
                executions,
                item,
                node_id,
                payload["operator"],
                payload["reference"],
                now,
            )
        updated = updated.model_copy(update={"step_executions": executions})
        store.save_workflow(updated, expected=workflow)
        result = {
            "no_op": False,
            "confirmations": [dict(item["confirmation"]) for item in items],
            "workflow": dump(updated),
        }
elif payload["mode"] == "submit":
    incident, workflow = load(payload["incident_id"])
    refuse_drift(
        payload["expected"],
        {
            "fencing_token": incident.fencing_token,
            "workflow_request_id": incident.workflow_request_id,
            "node_ids": sorted(incident.node_ids),
        },
    )
    terminal = TerminalEvent.model_validate(payload["terminal"])
    existing = store.get_decision_by_event(terminal.event_key)
    if existing is not None:
        result = {"duplicate": True, "decision": dump(existing)}
        plan = (
            store.get_plan(existing.recovery_plan_id)
            if existing.recovery_plan_id
            else None
        )
    else:
        context.completion.add_marker(NodeMarker.model_validate(payload["marker"]))
        decision = context.completion.handle_terminal(terminal)
        result = {"duplicate": bool(decision.duplicate), "decision": dump(decision)}
        plan = (
            store.get_plan(decision.recovery_plan_id)
            if decision.recovery_plan_id
            else None
        )
    result["plan"] = dump(plan)
    result["workflow"] = (
        dump(store.get_workflow(plan.workflow_request_id))
        if plan is not None and plan.workflow_request_id
        else None
    )
else:
    raise ValueError("unsupported submit-remediation mode")
print(json.dumps(result, sort_keys=True))
"""
