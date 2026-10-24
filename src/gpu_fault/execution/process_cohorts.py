"""Only declared equivalent roles, hardware and intersecting windows are peers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from gpu_fault.execution.process_observations import observation_window


def cohort_key(row: dict[str, Any]) -> str | None:
    context = row.get("comparison_context") or {}
    fields = (
        "job_id",
        "attempt_id",
        "group_name",
        "role",
        "pipeline_stage",
        "expert_role",
        "hardware_class",
        "code_version",
    )
    if context.get("verified") is not True or any(
        not isinstance(context.get(field), str)
        or context[field].strip().lower() in {"", "unknown"}
        for field in fields
    ):
        return None
    members = context.get("members")
    if isinstance(members, list) and members:
        if (
            any(type(rank) is not int or rank < 0 for rank in members)
            or context.get("declared_rank") not in members
            or len(set(members)) != len(members)
        ):
            return None
        digest = hashlib.sha256(
            json.dumps(sorted(members), separators=(",", ":")).encode()
        ).hexdigest()
        count = len(members)
    else:
        raw_digest, raw_count = (
            context.get("membership_sha256"),
            context.get("member_count"),
        )
        if (
            context.get("membership_verified") is not True
            or not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or type(raw_count) is not int
            or raw_count <= 0
        ):
            return None
        digest, count = raw_digest, raw_count
    gpu = row.get("gpu_series") or {}
    if (
        gpu.get("mapping_verified") is not True
        or gpu.get("shared_with_other_processes") is not False
    ):
        return None
    models = {
        device.get("device_model")
        for sample in gpu.get("samples", [])
        for device in sample.get("devices", {}).values()
    }
    if not models or None in models or "" in models:
        return None
    return hashlib.sha256(
        json.dumps(
            [
                *[context[field] for field in fields],
                digest,
                count,
                sorted(models),
            ]
        ).encode()
    ).hexdigest()


def compare_cohorts(
    rows: list[dict[str, Any]], reports: dict[tuple[str, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["_process_key"]
        cohort = cohort_key(row)
        report = reports[key]
        if cohort is None:
            report["peer_comparison"] = "GROUP_ROLE_OR_HARDWARE_UNKNOWN"
            continue
        span = observation_window(
            (row.get("process_series") or {}).get("samples") or []
        )
        if span is not None:
            groups[cohort].append({"row": row, "report": report, "span": span})
    comparisons = []
    for key, peers in groups.items():
        if len(
            {peer["row"]["comparison_context"].get("declared_rank") for peer in peers}
        ) != len(peers):
            for peer in peers:
                peer["report"]["peer_comparison"] = "AMBIGUOUS_COHORT_MEMBERS"
            continue
        overlap = (
            min(peer["span"][1] for peer in peers)
            - max(peer["span"][0] for peer in peers)
        ).total_seconds()
        if len(peers) < 3 or overlap <= 0:
            for peer in peers:
                peer["report"]["peer_comparison"] = (
                    "INSUFFICIENT_OR_NONOVERLAPPING_PEERS"
                )
            continue
        training = []
        for peer in peers:
            threads = [
                item
                for item in peer["report"]["stack"]["stable_threads"]
                if item["role"] == "TRAINING"
            ]
            if len(threads) == 1:
                training.append((peer, threads[0]))
            peer["report"]["peer_comparison"] = "ROLE_COHORT"
        signatures = Counter(thread["signature"] for _, thread in training)
        mode, count = signatures.most_common(1)[0] if signatures else (None, 0)
        stack_spans = [
            observation_window(peer["row"]["stack_series"]["samples"])
            for peer, _ in training
        ]
        stack_overlap = (
            (
                min(span[1] for span in stack_spans if span is not None)
                - max(span[0] for span in stack_spans if span is not None)
            ).total_seconds()
            if stack_spans and all(span is not None for span in stack_spans)
            else 0
        )
        if len(training) >= 3 and stack_overlap > 0 and count >= 0.8 * len(peers):
            for peer, thread in training:
                if thread["signature"] != mode:
                    peer["report"]["observations"].append(
                        {
                            "code": "ROLE_COHORT_STACK_DIFFERENCE",
                            "candidate": True,
                            "cohort": key,
                            "observed_peer_count": len(peers),
                            "common_window_seconds": stack_overlap,
                            "qualification": "Diagnostic difference, not causal attribution",
                        }
                    )
        cpu = [
            peer["report"]["behavior"]["cpu_cores"]
            for peer in peers
            if peer["report"]["behavior"]["cpu_valid_windows"] == 2
        ]
        typical = median(cpu) if len(cpu) == len(peers) else None
        if typical is not None and typical > 0:
            for peer in peers:
                actual = peer["report"]["behavior"]["cpu_cores"]
                if actual < typical * 0.2:
                    peer["report"]["observations"].append(
                        {
                            "code": "ROLE_COHORT_CPU_DIFFERENCE",
                            "candidate": True,
                            "cpu_cores": actual,
                            "reference_cpu_cores": typical,
                            "cohort": key,
                            "observed_peer_count": len(peers),
                            "common_window_seconds": overlap,
                            "qualification": "Utilization difference, not fault probability",
                        }
                    )
        comparisons.append(
            {
                "cohort": key,
                "definition": {
                    field: peers[0]["row"]["comparison_context"][field]
                    for field in (
                        "job_id",
                        "attempt_id",
                        "group_name",
                        "role",
                        "pipeline_stage",
                        "expert_role",
                        "hardware_class",
                        "code_version",
                    )
                },
                "process_count": len(peers),
                "common_window_seconds": overlap,
                "stack_common_window_seconds": max(0, stack_overlap),
                "processes": [list(peer["row"]["_process_key"]) for peer in peers],
            }
        )
    return comparisons
