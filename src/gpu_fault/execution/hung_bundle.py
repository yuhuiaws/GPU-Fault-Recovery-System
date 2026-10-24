"""Bound the second-stage diagnostic and bind it to observed process identities."""

from typing import Any

from gpu_fault.models import WorkflowStepSpec


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
