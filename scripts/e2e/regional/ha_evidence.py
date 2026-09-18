"""Bind the legacy HA entrypoints to the shared formal evidence chain."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from scripts.e2e.regional.regional_case_contract import predecessor_path
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    predecessor_evidence,
    settings_from_arguments,
)


def chain_preflight(arguments: argparse.Namespace, case_id: str) -> dict[str, Any]:
    defaults = {
        "cpu_kubeconfig": "",
        "gpu_kubeconfig": "",
        "gpu_context": "",
        "namespace": os.getenv("GPU_FAULT_NAMESPACE", "gpu-fault-system"),
        "cluster_id": "",
        "region": "",
    }
    settings = settings_from_arguments(
        argparse.Namespace(**{**defaults, **vars(arguments)})
    )
    identity = RegionalLiveFixture(settings).evidence_identity()
    if any(
        not isinstance(identity.get(key), str) or not identity[key].strip()
        for key in ("release_id", "cluster_id")
    ):
        raise RegionalFixtureError("HA evidence requires release and cluster identity")
    previous, path = predecessor_path(
        Path(arguments.run_dir),
        case_id,
        getattr(arguments, "predecessor_evidence", ""),
    )
    predecessor = (
        predecessor_evidence(path, previous, **identity)
        if previous is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    return {
        "identity": identity,
        "predecessor": predecessor,
        "errors": []
        if predecessor.get("valid") is True
        else ["formal predecessor evidence is not PASS for this release and cluster"],
    }


def require_chain(planned: dict[str, Any], current: dict[str, Any]) -> None:
    if current.get("errors") != [] or current["predecessor"].get("valid") is not True:
        raise RegionalFixtureError("HA formal predecessor evidence is not PASS")
    if planned.get("identity") != current.get("identity"):
        raise RegionalFixtureError("HA release or cluster changed since planning")
    if planned.get("predecessor", {}).get("case_id") != current["predecessor"].get(
        "case_id"
    ):
        raise RegionalFixtureError("HA formal predecessor changed since planning")


def result_identity(chain: dict[str, Any] | None) -> dict[str, Any]:
    if chain is None:
        return {}
    return {**chain["identity"], "predecessor": chain["predecessor"]}


def isolated_chain(arguments: argparse.Namespace, case_id: str) -> dict[str, Any]:
    """Local probes never read live identity or imply they exercised a cluster."""
    identity = {"release_id": arguments.release_id, "cluster_id": arguments.cluster_id}
    if not all(isinstance(value, str) and value.strip() for value in identity.values()):
        return {
            "validation_scope": "isolated-source",
            "formal_sequence_satisfied": False,
            "identity_source": "not supplied",
        }
    previous, path = predecessor_path(arguments.run_dir, case_id, "")
    proof = (
        predecessor_evidence(path, previous, **identity)
        if previous is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    if proof.get("valid") is not True:
        raise RegionalFixtureError("isolated HA probe predecessor is not PASS")
    return {
        **identity,
        "predecessor": proof,
        "validation_scope": "isolated-source",
        "identity_source": "enclosing acceptance run",
    }
