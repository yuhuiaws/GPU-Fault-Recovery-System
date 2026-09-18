#!/usr/bin/env python3
"""GF-REGIONAL-HA-011: busy owner loss in a new, isolated production-image Pod."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin.atomic_json import (  # noqa: E402
    write_json_atomic as write_initial_json,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.acceptance_scope import current_acceptance_scope  # noqa: E402
from scripts.e2e.regional.ha011_contracts import (  # noqa: E402
    BOUNDARY,
    CASE_ID,
    CONFIRMATION,
    ProofError,
    Settings,
    digest,
    evidence_errors,
)
from scripts.e2e.regional.ha011_manifests import manifests, source_bundle  # noqa: E402
from scripts.e2e.regional.ha011_resources import (  # noqa: E402
    CpuKubernetes,
    IsolatedResources,
)
from scripts.e2e.regional.ha_cleanup import (  # noqa: E402
    record_supervision_loss,
)
from scripts.e2e.regional.ha_plan_preflight import require_window  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_case_contract import predecessor_path  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
    run_case_main,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", type=Path, required=True)
    value.add_argument("--cpu-context", required=True)
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", required=True)
    value.add_argument("--region", required=True)
    value.add_argument("--isolation-id", required=True)
    value.add_argument("--postgres-image", required=True)
    value.add_argument("--predecessor-evidence", default="")
    return value


def configure(arguments: argparse.Namespace) -> Settings:
    previous, path = predecessor_path(
        arguments.run_dir, CASE_ID, arguments.predecessor_evidence
    )
    if previous is None or path is None:
        raise ProofError("HA011 requires a registered predecessor in the formal order")
    return Settings(
        cpu_kubeconfig=arguments.cpu_kubeconfig.expanduser().resolve(),
        cpu_context=arguments.cpu_context,
        namespace=arguments.namespace,
        cluster_id=arguments.cluster_id,
        isolation_id=arguments.isolation_id,
        postgres_image=arguments.postgres_image,
        predecessor_case=previous,
        predecessor_path=path,
        region=arguments.region,
    )


def read_only_preflight(settings: Settings, _case_dir: Path) -> dict[str, Any]:
    kubernetes = CpuKubernetes(settings)
    identity = kubernetes.identity()
    proof = predecessor_evidence(
        settings.predecessor_path,
        settings.predecessor_case,
        release_id=identity["release_id"],
        cluster_id=settings.cluster_id,
    )
    errors = []
    if proof.get("valid") is not True:
        errors.append("formal predecessor is not valid for this release and cluster")
    if (
        kubernetes.read("Namespace", settings.isolated_namespace, optional=True)
        is not None
    ):
        errors.append(
            "isolated namespace already exists; existing resources are never adopted"
        )
    if (
        kubernetes.read("PriorityClass", settings.priority_class_name, optional=True)
        is not None
    ):
        errors.append("isolated PriorityClass already exists; it is never adopted")
    bundle = source_bundle(Path(__file__).parent)
    intended = manifests(
        settings,
        runtime_image=identity["runtime_image"],
        bundle=bundle,
        password="<generated-private-credential>",
        cpu_node=identity["cpu_node"],
    )
    return {
        "identity": identity,
        "predecessor": proof,
        "errors": errors,
        "intent_sha256": digest(intended),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_scope": BOUNDARY,
        "isolated_namespace": settings.isolated_namespace,
        "isolated_priority_class": settings.priority_class_name,
        "business_worker_targeted": False,
        "cpu_saturation_tested": False,
        "owned_process_crash_only": True,
        "preflight": preflight,
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    receipt_path = case_dir / f"{CASE_ID}.json"
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "status": "PREFLIGHT",
        "validation_scope": BOUNDARY,
    }
    history = case_dir / f"{CASE_ID}.history-{secrets.token_hex(16)}.json"
    try:
        receipt_path.rename(history)
    except FileNotFoundError:
        pass
    else:
        result["previous_receipt"] = history.name
    write_initial_json(receipt_path, result)
    resources = None
    supervision_lost = False
    try:
        result.update(current_acceptance_scope().result_fields())
        require_window(maintenance_window_end, required_seconds=480)
        plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
        current = read_only_preflight(settings, case_dir)
        if current["errors"] or plan.get("details") != plan_details(settings, current):
            raise ProofError(
                "approved deployment, predecessor, or resource intent drifted"
            )
        result.update(current["identity"])
        result["predecessor"] = current["predecessor"]
        kubernetes = CpuKubernetes(settings)
        resources = IsolatedResources(
            kubernetes,
            intent_sha256=current["intent_sha256"],
            deadline=maintenance_window_end,
            cpu_node_identity=current["identity"]["cpu_node"],
        )
        bundle = source_bundle(Path(__file__).parent)
        if (
            digest(
                manifests(
                    settings,
                    runtime_image=current["identity"]["runtime_image"],
                    bundle=bundle,
                    password="<generated-private-credential>",
                    cpu_node=current["identity"]["cpu_node"],
                )
            )
            != current["intent_sha256"]
        ):
            raise ProofError("probe source changed after execution preflight")
        resources.start(
            manifests(
                settings,
                runtime_image=current["identity"]["runtime_image"],
                bundle=bundle,
                password=secrets.token_urlsafe(40),
                cpu_node=current["identity"]["cpu_node"],
            )
        )
        resources.arm(current["identity"]["runtime_image_id"])
        proof, pod_uid = resources.collect(current["identity"]["runtime_image_id"])
        errors = evidence_errors(
            proof,
            isolation_id=settings.isolation_id,
            pod_uid=pod_uid,
            intent_sha256=current["intent_sha256"],
        )
        if errors:
            raise ProofError(
                "isolated runtime did not prove the complete busy takeover contract"
            )
        if kubernetes.identity() != current["identity"]:
            raise ProofError(
                "business deployment identity drifted during isolated acceptance"
            )
        result["probe"] = proof
        result["verdict"] = "PASS"
    except ProcessSupervisionLost:
        supervision_lost = True
        record_supervision_loss(result)
        result["error_type"] = "ProcessSupervisionLost"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        if resources is not None and resources.failure_details is not None:
            result["probe_failure"] = resources.failure_details
    finally:
        if resources is not None and not supervision_lost:
            try:
                result["cleanup"] = resources.cleanup()
            except ProcessSupervisionLost:
                record_supervision_loss(result)
            except Exception as exc:
                result["verdict"] = "FAIL"
                result["cleanup_error_type"] = type(exc).__name__
        elif resources is not None:
            result["cleanup"] = {
                "remote_cleanup_attempted": False,
                "namespace_absent": False,
            }
    result["status"] = "COMPLETED" if result["verdict"] == "PASS" else "FAILED"
    write_json_atomic(receipt_path, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    return run_standard_case(CASE)


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)

if __name__ == "__main__":
    sys.exit(run_case_main(main))
