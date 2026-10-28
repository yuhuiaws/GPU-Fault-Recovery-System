"""Positive coverage and cleanup gates for low-utilization notification phases."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Collection, Sequence
from typing import Any

#: GPUs of a selected node, every one of which the fixture requests: the
#: product grades low utilisation only on the GPUs a managed attempt holds
#: (``gpu_fault/host_low_utilization.py``), so one requested GPU of eight
#: would give one graded GPU and nothing to aggregate.
NODE_GPU_COUNT = 8

#: The mails one node's low-utilisation episode sends, told apart by the
#: product's own category and deduplication key. The host-resource mail
#: (``notification_preview.py``) is the one that aggregates the node's GPUs.
#: The finding's incident -- one per node, the other GPUs' findings absorbed
#: into it -- runs the diagnostic-only chain FREEZE_EVIDENCE -> VALIDATE_GPU,
#: whose ``host_diagnostic_probe`` re-reads the held GPU and fails while the
#: fixture stays idle; the incident closes RECOVERED "diagnostic inconclusive"
#: with one advisory (``execution/workflow_notifications.py``) whose trigger
#: reasons list the absorbed per-GPU findings, so it names
#: LOW_GPU_UTILIZATION as well. Each kind is one per node. Counted together
#: they failed a correct aggregation (live 2026-10-28 HyperPod, one node: the
#: mail and, 13 s later, its advisory -- two records for one node).
LOW_UTILIZATION_MAIL = "LOW_UTILIZATION_MAIL"
DIAGNOSTIC_INCONCLUSIVE = "DIAGNOSTIC_INCONCLUSIVE"
KINDS = {
    LOW_UTILIZATION_MAIL: ("HEALTH_TREND", "host-resource/"),
    DIAGNOSTIC_INCONCLUSIVE: ("ADVISORY", "/diagnostic-inconclusive/"),
}
#: ``gpu_fault.models.EXECUTABLE_WORKFLOW_STATUSES``: a chain in one of these
#: still runs, and may still mail.
RUNNING_WORKFLOW_STATUSES = frozenset({"PENDING", "SAFETY_PENDING", "RUNNING"})
#: The low-utilisation policy a control worker runs with; the phase and
#: quiesce deadlines are derived from its window.
LOW_UTILIZATION_CONFIG_PROBE = r"""
import json
import os
print(json.dumps({
    "duration_seconds": int(
        os.getenv("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "300")
    ),
    "threshold_percent": float(
        os.getenv("GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT", "10")
    ),
}, sort_keys=True))
"""
LIMITATIONS = (
    "The workloads reserve every GPU of each node while doing CPU work only; "
    "they validate notification aggregation without damaging hardware.",
    "The sustained low-utilization signal latches per GPU and attempt, and "
    "its unscoped window resets only after the node reports no active "
    "workload, so the case refuses nodes that already hold a GPU for an "
    "unrelated workload and waits for each phase's nodes to be re-armed; "
    "quiesce_polls records that wait and pre_existing_notifications records "
    "the episodes that preceded the run.",
    "Each node's incident runs the diagnostic-only chain, which fails while "
    "the fixture stays idle and sends one inconclusive-diagnostic advisory; "
    "the count is bounded per mail kind, and a phase is judged only once "
    "every counted incident's chain has finished.",
)


def low_utilization_manifest(
    *,
    name: str,
    nodes: tuple[str, ...],
    image: str,
) -> dict[str, Any]:
    """Eight CPU burners per node, holding every GPU and touching none.

    No burner creates a CUDA context, so nvidia-smi lists no compute process:
    ``LOW_GPU_UTILIZATION`` fires on each held GPU and ``LOW_CPU_UTILIZATION``
    (the CPU of the processes driving a GPU) has nothing to grade.
    """

    gpus = str(NODE_GPU_COUNT)
    command = [
        "/bin/bash",
        "-ceu",
        (
            "python - <<'PY'\n"
            "import multiprocessing\n"
            "import time\n"
            "def burn():\n"
            "    value = 1\n"
            "    while True:\n"
            "        value = (value * 1103515245 + 12345) & 0x7fffffff\n"
            "workers = [multiprocessing.Process(target=burn, daemon=True) "
            "for _ in range(8)]\n"
            "for worker in workers: worker.start()\n"
            "time.sleep(1800)\n"
            "PY"
        ),
    ]
    container = {
        "name": "pytorch",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "resources": {
            "requests": {"cpu": "8", "memory": "2Gi", "nvidia.com/gpu": gpus},
            "limits": {"cpu": "16", "memory": "4Gi", "nvidia.com/gpu": gpus},
        },
    }
    if len(nodes) == 1:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeName": nodes[0],
                        "containers": [container],
                    },
                }
            },
        }
    affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "In",
                                "values": list(nodes),
                            }
                        ]
                    }
                ],
            }
        },
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {"matchLabels": {"app": name}},
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        },
    }
    template = {
        "metadata": {"labels": {"app": name}},
        "spec": {
            "restartPolicy": "Never",
            "affinity": affinity,
            "containers": [container],
        },
    }
    # Independent templates prevent YAML aliases and double metadata injection.
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": name},
        "spec": {
            "runPolicy": {"cleanPodPolicy": "All"},
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "Never",
                    "template": copy.deepcopy(template),
                },
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "Never",
                    "template": copy.deepcopy(template),
                },
            },
        },
    }


def node_errors(node: dict[str, Any]) -> list[str]:
    errors = []
    if not node.get("uid") or node.get("ready") != "True":
        errors.append("node identity or Ready state is unknown")
    if node.get("unschedulable") is not False or node.get("ownership_annotations"):
        errors.append("node is cordoned or owned by another workflow")
    if str(node.get("gpu_allocatable")) != str(NODE_GPU_COUNT):
        errors.append("the aggregation fixture requires an eight-GPU node")
    return errors


def record_kind(record: dict[str, Any]) -> str:
    """The mail kind (``KINDS``) of one query record, ``UNEXPECTED:<category>``
    for anything else."""

    key = str(record.get("deduplication_key") or "")
    for kind, (category, marker) in KINDS.items():
        if record.get("category") == category and marker in key:
            return kind
    return f"UNEXPECTED:{record.get('category')}"


def phase_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The LOW_GPU_UTILIZATION records of one query, each tagged with its kind."""

    return [
        {**item, "kind": record_kind(item)}
        for item in records
        if item["low_gpu_utilization"]
    ]


def mailed_nodes(records: Sequence[dict[str, Any]]) -> set[str]:
    """The nodes a low-utilisation mail reported (an advisory reports none)."""

    return {
        node
        for item in records
        if record_kind(item) == LOW_UTILIZATION_MAIL
        for node in item["matched_nodes"]
    }


def delivered(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "SENT"
        and record.get("provider_message_id_present") is True
    )


def chain_finished(record: dict[str, Any]) -> bool:
    """The record's incident ran its workflow to the end and mails no more.

    The inconclusive-diagnostic advisory follows the chain's terminal write:
    judged while the chain still runs, a phase counts part of what its
    incidents send, and the next phase finds the rest under an incident it
    never mailed. An incident without a workflow has finished; a record read
    without the field has not.
    """

    return (
        "workflow_status" in record
        and record["workflow_status"] not in RUNNING_WORKFLOW_STATUSES
    )


def phase_progress(
    records: Sequence[dict[str, Any]], nodes: Collection[str]
) -> dict[str, Any]:
    """One poll of a phase: the count per kind, the nodes their mail reported,
    whether the phase is complete -- every node mailed, every record delivered
    and its incident's chain finished -- and whether a kind already exceeds
    the node count, a fan-out no further wait can undo."""

    kinds = Counter(record_kind(item) for item in records)
    reported = mailed_nodes(records)
    return {
        "kinds": dict(sorted(kinds.items())),
        "reported_nodes": sorted(reported),
        "complete": set(nodes) == reported
        and all(delivered(item) and chain_finished(item) for item in records),
        "overflow": any(count > len(nodes) for count in kinds.values()),
    }


def phase_checks(observation: dict[str, Any]) -> dict[str, bool]:
    nodes = set(observation["nodes"])
    records = observation["notifications"]
    progress = phase_progress(records, nodes)
    mails = [item for item in records if record_kind(item) == LOW_UTILIZATION_MAIL]
    mailed_incidents = {
        item["incident_id"] for item in mails if item.get("incident_id")
    }
    return {
        "replica_metadata_injected_per_role": not observation["metadata_errors"],
        "workload_on_selected_nodes": {pod["node"] for pod in observation["pods"]}
        == nodes,
        # Per kind: without the aggregation a node's eight GPUs would be eight
        # of each, three nodes 24.
        "count_at_most_node_count": not progress["overflow"],
        "every_notification_kind_expected": set(progress["kinds"]) <= set(KINDS),
        "every_selected_node_reported_in_its_phase": set(progress["reported_nodes"])
        == nodes,
        "every_notification_is_node_scoped": all(
            len(item["matched_nodes"]) == 1 for item in records
        ),
        "notifications_aggregate_multiple_gpu_devices": bool(mails)
        and all(1 < len(item["gpu_devices"]) <= NODE_GPU_COUNT for item in mails),
        # An advisory closes the incident a mail of this phase opened: not one
        # incident per GPU, and not an earlier phase's late chain.
        "diagnostic_notices_follow_mailed_incidents": all(
            item.get("incident_id") in mailed_incidents
            for item in records
            if record_kind(item) == DIAGNOSTIC_INCONCLUSIVE
        ),
        "every_incident_chain_finished": all(chain_finished(item) for item in records),
        # The burners hold no compute context: a LOW_CPU_UTILIZATION mail for
        # these nodes would be the host-wide CPU reading the rule dropped.
        "cpu_rule_silent": not any(
            poll.get("low_cpu_notification_count")
            for poll in observation.get("polls", [])
        ),
        "every_notification_sent": bool(records)
        and all(delivered(item) for item in records),
    }
