"""What a destructive case's directory holds after it recorded its own NOT_RUN.

The live shape (round 14, a plain EKS run dir whose only GPU node group has
two nodes): DESTR-027/028/029 need more GPU nodes in one node group than that.
Their preflight records the node-group gate's ``not_run`` and no error, so the
plan says the preflight passed and plans that ``not_run`` in its details;
``--execute`` writes the verdict as the case's own record and touches nothing.
The directory holds exactly those three files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DETAIL = (
    "needs >=3 GPU nodes in one node group; cluster gpu-eks-cluster group "
    "gpu-p5en-efa-spot-node-group (eks.amazonaws.com/nodegroup) has 2"
)
IDENTITY = {"release_id": "98422873089d", "cluster_id": "gpu-eks-cluster"}


def not_run_block(
    reason: str = "CAPABILITY_ABSENT", *, verdict: str = "NOT_RUN", detail: str = DETAIL
) -> dict[str, str]:
    """The ``not_run`` block ``workload_domain_oracle.capability_absent`` writes."""

    return {"verdict": verdict, "reason": reason, "detail": detail}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_not_run_case(
    root: Path,
    case_id: str,
    *,
    attempt: int = 7,
    not_run: dict[str, str] | None = None,
    record: dict[str, Any] | None = None,
) -> Path:
    """Write the plan, the preflight and the case's own record; return the dir.

    ``not_run`` is the block all three carry (default: the live
    NOT_RUN(CAPABILITY_ABSENT)); ``record`` replaces the case's own record
    (default: what the runners' ``not_run_result`` writes for that block).
    """

    block = not_run if not_run is not None else not_run_block()
    nodes = ["ip-10-0-32-162", "ip-10-0-43-5"]
    group = {
        "label": "eks.amazonaws.com/nodegroup",
        "name": "gpu-p5en-efa-spot-node-group",
        "gpu_nodes": len(nodes),
        "nodes": nodes,
    }
    gate = {
        "minimum_gpu_nodes_in_one_node_group": 3,
        "group": group,
        "groups": [group],
        "not_run": block,
    }
    case = root / "cases" / case_id
    write_json(
        case / "preflight.json",
        {
            "release_id": IDENTITY["release_id"],
            "evidence_identity": IDENTITY,
            "gpu_nodes": [{"name": name, "gpu_allocatable": 8} for name in nodes],
            "node_group_gate": gate,
            "not_run": block,
            "errors": [],
        },
    )
    write_json(
        case / "plan.json",
        {
            "schema_version": 3,
            "case_id": case_id,
            "attempt": attempt,
            "execution_scope": "selective",
            "details": {
                "risk": "destructive",
                "not_run": block,
                "node_group_gate": gate,
                "mutation": "none: the cluster cannot carry the case "
                "(NOT_RUN(CAPABILITY_ABSENT)); --execute records the verdict only",
                "preflight_errors": [],
            },
            "preflight_passed": True,
            "mutation_performed": False,
        },
    )
    write_json(
        case / f"{case_id}.json",
        record
        if record is not None
        else {
            "case_id": case_id,
            "attempt": attempt,
            **IDENTITY,
            **block,
            "node_group_gate": gate,
            "mutation_performed": False,
            "ended_at": "2026-09-07T09:30:00+00:00",
            "execution_scope": "selective",
            "formal_sequence_satisfied": False,
        },
    )
    return case
