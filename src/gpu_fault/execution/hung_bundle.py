"""Bound the second-stage diagnostic and bind it to observed process identities."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from gpu_fault.models import WorkflowOperation, WorkflowStepSpec

#: The part of a triage decision a node never reads: every sampled process's
#: thread stacks and wait observations.
WIRE_OMITTED_PATH = "evidence_branches.process_observations.processes"
WIRE_RECORDED_IN = "workflow step parameters and COLLECT_HUNG_TRIAGE details"


def wire_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """The triage decision a bundle command carries to a node.

    The verdict, the candidate processes, the coverage and the flight-recorder
    branches stay. The per-process observations -- one stack and one wait row
    per thread of every sampled process, ~15 KB a process, bounded only at 256
    processes -- become their count, and ``wire_form`` records what was left
    out with the size and SHA-256 of the full decision, which the workflow
    record keeps in the bundle step's parameters and the triage step's
    execution details. The node only copies the decision into the bundle
    manifest's ``fault_context``; live, sixteen processes made it 259 KB and
    the node's 256 KiB observation request budget refused the whole bundle.
    A decision without process observations is returned as it is, so the
    wire form is its own fixed point.
    """

    branches = decision.get("evidence_branches")
    if not isinstance(branches, Mapping):
        return dict(decision)
    observations = branches.get("process_observations")
    if not isinstance(observations, Mapping) or "processes" not in observations:
        return dict(decision)
    encoded = json.dumps(
        decision, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    processes = observations["processes"]
    return {
        **decision,
        "evidence_branches": {
            **branches,
            "process_observations": {
                **{
                    key: value
                    for key, value in observations.items()
                    if key != "processes"
                },
                "processes_omitted": len(processes)
                if isinstance(processes, (list, tuple))
                else 1,
            },
        },
        "wire_form": {
            "omitted": [WIRE_OMITTED_PATH],
            "full_bytes": len(encoded),
            "full_sha256": hashlib.sha256(encoded).hexdigest(),
            "recorded_in": WIRE_RECORDED_IN,
        },
    }


def wire_parameters(
    operation: WorkflowOperation, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """The parameters a node command for ``operation`` carries: the step's own,
    with a hung bundle's ``hung_triage_decision`` in its wire form. Any other
    operation's parameters are returned unchanged."""

    result = dict(parameters)
    decision = result.get("hung_triage_decision")
    if operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE and isinstance(
        decision, Mapping
    ):
        result["hung_triage_decision"] = wire_decision(decision)
    return result


def bind_hung_bundle(
    bundle: WorkflowStepSpec,
    decision: dict[str, Any],
    signals: list[dict[str, Any]],
    triage_nodes: list[str],
) -> WorkflowStepSpec:
    ranks = set(decision.get("culprit_ranks", [])) | set(
        decision.get("control_ranks", [])
    )
    selected = [signal for signal in signals if signal.get("rank") in ranks]
    actionable = bool(
        decision.get("targeted_capture_eligible")
        and selected
        and len(selected) == len(ranks)
        and all(
            type(signal.get("pid")) is int
            and signal["pid"] > 0
            and signal.get("node_id")
            and (signal.get("process_identity") or {}).get("verified") is True
            and type((signal.get("process_identity") or {}).get("start_time_ticks"))
            is int
            for signal in selected
        )
    )
    pids: dict[str, list[int]] = {}
    gpu_by_pid: dict[str, dict[str, str]] = {}
    starts: dict[str, dict[str, int]] = {}
    if actionable:
        for signal in selected:
            node, pid = signal["node_id"], signal["pid"]
            pids.setdefault(node, []).append(pid)
            starts.setdefault(node, {})[str(pid)] = signal["process_identity"][
                "start_time_ticks"
            ]
            if signal.get("gpu_uuid"):
                gpu_by_pid.setdefault(node, {})[str(pid)] = signal["gpu_uuid"]
    nodes = sorted(pids) if actionable else sorted(set(triage_nodes))[:32]
    raw_flights: dict[str, list[dict[str, Any]]] = {}
    for signal in signals:
        node = signal.get("node_id")
        if node not in nodes:
            continue
        for sample in ("first", "second"):
            evidence = (signal.get("flight_recorder_series") or {}).get(sample, {}).get(
                "raw_evidence"
            ) or {}
            if evidence.get("status") == "PRESERVED":
                raw_flights.setdefault(node, []).append(
                    {
                        **evidence,
                        "sample": sample,
                        "rank": signal.get("rank"),
                        "pid": signal.get("pid"),
                        "process_start_ticks": (
                            signal.get("process_identity") or {}
                        ).get("start_time_ticks"),
                    }
                )
    parameters = {
        **bundle.parameters,
        "hung_triage_decision": decision,
        "raw_flights_by_node": raw_flights,
        "hung_triage_target_pending": False,
        "capture_process_state": actionable,
        "target_pids_by_node": pids,
        "target_gpu_uuids_by_pid_by_node": gpu_by_pid,
        "target_start_ticks_by_pid_by_node": starts,
        "require_process_start_identity": True,
        "max_processes": len(selected) if actionable else 0,
        "expand_python_cgroup_processes": False,
        "strace_sample_count": 0,
        "disable_strace": True,
        "operator_escalation_required": not actionable,
        "bundle_scope": "identity_bound_candidates"
        if actionable
        else "bounded_triage_nodes",
        "bundle_not_collected_nodes": sorted(
            (set(bundle.node_ids) - set(nodes))
            | set(bundle.parameters.get("bundle_not_collected_nodes", []))
        ),
    }
    return bundle.model_copy(
        update={
            "node_ids": nodes,
            "gpu_uuids": sorted(
                {gpu for values in gpu_by_pid.values() for gpu in values.values()}
            )
            if actionable
            else bundle.gpu_uuids,
            "parameters": parameters,
        }
    )
