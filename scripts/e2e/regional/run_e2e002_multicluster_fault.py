#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import run_destr009_workload_restart as workload_case  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    ClusterTarget,
    MultiClusterSettings,
    registration_snapshot,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    predecessor_evidence,
    required,
)


CASE_ID = "GF-REGIONAL-E2E-002"
PREDECESSOR_CASE_ID = "GF-REGIONAL-ISO-006"
CONFIRMATION = "E2E002_RESTART_SAME_JOB_IN_TWO_CLUSTERS"
MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "xid11-three-node-pytorchjob.yaml"
)


@dataclass(frozen=True)
class Settings:
    multi: MultiClusterSettings
    site_file: Path
    job_id: str
    attempt_id: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.multi.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_SHARED_JOB_ID": self.job_id,
            "GPU_FAULT_SHARED_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"e2e002-{suffix}"
    return job_id, f"{job_id}-a001"


def configure(arguments: argparse.Namespace) -> Settings:
    job_id, attempt_id = derived_identity(arguments.run_dir, arguments.attempt)
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    multi = MultiClusterSettings(
        cpu_kubeconfig=Path(
            required(
                arguments.cpu_kubeconfig
                or os.getenv("GPU_FAULT_CONTROL_KUBECONFIG", ""),
                "CPU kubeconfig",
            )
        )
        .expanduser()
        .resolve(),
        namespace=arguments.namespace,
        region=required(
            arguments.region
            or os.getenv("AWS_REGION", "")
            or os.getenv("AWS_DEFAULT_REGION", ""),
            "AWS Region",
        ),
        cluster_a=ClusterTarget(
            cluster_id=required(arguments.cluster_a, "cluster A ID"),
            gpu_kubeconfig=Path(
                required(arguments.gpu_a_kubeconfig, "cluster A kubeconfig")
            )
            .expanduser()
            .resolve(),
            gpu_context=required(arguments.gpu_a_context, "cluster A context"),
        ),
        cluster_b=ClusterTarget(
            cluster_id=required(arguments.cluster_b, "cluster B ID"),
            gpu_kubeconfig=Path(
                required(arguments.gpu_b_kubeconfig, "cluster B kubeconfig")
            )
            .expanduser()
            .resolve(),
            gpu_context=required(arguments.gpu_b_context, "cluster B context"),
        ),
    )
    configured_job = arguments.job_id.strip() or job_id
    return Settings(
        multi=multi,
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        job_id=configured_job,
        attempt_id=arguments.attempt_id.strip()
        or (f"{configured_job}-a001" if arguments.job_id else attempt_id),
        predecessor_path=predecessor,
    )


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    a = settings.multi.regional(settings.multi.cluster_a)
    b = settings.multi.regional(settings.multi.cluster_b)
    registrations = registration_snapshot(a, settings.multi)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    nodes_a = a.gpu_nodes()
    nodes_b = b.gpu_nodes()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/regional/test_regional_control_plane.py::"
        "test_attempt_index_is_cluster_scoped",
        "tests/regional/test_regional_control_plane.py::"
        "test_failed_remote_restart_releases_restart_budget",
    ]
    completed = a.run(command, cwd=ROOT, check=False, timeout=300)
    (case_dir / "focused-tests.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("ISO-006 predecessor evidence is not PASS")
    if len(registrations) != 2 or any(
        not item.get("enabled") or item.get("synthetic") for item in registrations
    ):
        errors.append("two enabled physical cluster registrations are required")
    elif not registrations_are_distinct_physical_clusters(registrations):
        errors.append("cluster registrations do not identify two physical clusters")
    if len(nodes_a) < 3 or len(nodes_b) < 3:
        errors.append("both clusters require at least three GPU nodes")
    if a.gpu_workloads() or b.gpu_workloads():
        errors.append("a target cluster already has GPU workloads")
    if not settings.site_file.is_file() or not MANIFEST.is_file():
        errors.append("site or training manifest is missing")
    if completed.returncode:
        errors.append("focused regression tests failed")
    result = {
        "release_id": a.release_id(),
        "registrations": registrations,
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "predecessor": predecessor,
        "cpu_pods": a.ready_pods("cpu", "gpu-fault-api-ha"),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def workload_settings(
    settings: Settings,
    target: ClusterTarget,
) -> workload_case.Settings:
    fixture = settings.multi.regional(target)
    return workload_case.Settings(
        regional=fixture.settings,
        site_file=settings.site_file,
        manifest=MANIFEST,
        job_id=settings.job_id,
        attempt_id=settings.attempt_id,
        predecessor_path=settings.predecessor_path,
    )


def managed_workload(
    settings: Settings,
    target: ClusterTarget,
) -> ManagedWorkloadFixture:
    fixture = settings.multi.regional(target)
    return ManagedWorkloadFixture(
        fixture,
        ManagedWorkloadSettings(
            manifest=MANIFEST,
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-workload-restart",
        "predecessor": preflight["predecessor"],
        "cluster_ids": [
            settings.multi.cluster_a.cluster_id,
            settings.multi.cluster_b.cluster_id,
        ],
        "shared_job_id": settings.job_id,
        "shared_attempt_id": settings.attempt_id,
        "mutation": (
            "submit the same managed job/attempt identity to two physical GPU "
            "clusters and inject XID11 into both within one 60-second window"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "cluster_a_node_uids": sorted(
                str(item["uid"]) for item in preflight["nodes_a"]
            ),
            "cluster_b_node_uids": sorted(
                str(item["uid"]) for item in preflight["nodes_b"]
            ),
        },
        "stop_conditions": [
            "ISO-006 predecessor evidence is not PASS",
            "fewer than two enabled physical clusters",
            "either workload or 24-GPU observation fails",
            "injections are more than 60 seconds apart",
            "incident, command, budget or notification crosses cluster scope",
            "cleanup leaves either workload behind",
        ],
        "rollback": {
            "delete_both_training_workloads": True,
            "delete_both_image_prewarm_sets": True,
            "no_node_or_provider_mutation": True,
        },
        "preflight": preflight,
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    targets = (settings.multi.cluster_a, settings.multi.cluster_b)
    fixtures = [settings.multi.regional(target) for target in targets]
    workloads = [managed_workload(settings, target) for target in targets]
    prewarms = [
        ImagePrewarmFixture(
            fixture,
            case_id=CASE_ID,
            run_id=f"e2e002-{index}-{attempt}",
        )
        for index, fixture in enumerate(fixtures)
    ]
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "errors": [],
    }
    try:
        sources = []
        observations = []
        for fixture, workload, prewarm, target in zip(
            fixtures,
            workloads,
            prewarms,
            targets,
            strict=True,
        ):
            nodes = [
                str(item["name"])
                for item in fixture.gpu_nodes()
                if item["ready"] == "True" and not item["unschedulable"]
            ]
            prewarm.create(nodes)
            workload.submit()
            source = workload.wait_running(timeout_seconds=900)
            sources.append(source)
            node = str(source["pods"][0]["node"])
            observations.append(
                workload_case.wait_observation(
                    fixture,
                    workload_settings(settings, target),
                    node=node,
                    expected_gpu_count=24,
                )
            )
        injected_at = datetime.now(timezone.utc)
        payloads = [
            workload_case.xid11_payload(
                workload_settings(settings, target),
                case_id=CASE_ID,
                marker=f"e2e002-{index}-{int(time.time())}",
                node=str(source["pods"][0]["node"]),
                product=workload_case.normalize_product(
                    fixture.node_metadata(str(source["pods"][0]["node"])).get("product")
                ),
                observation=observation,
                observed_at=injected_at,
            )
            for index, (fixture, source, observation, target) in enumerate(
                zip(fixtures, sources, observations, targets, strict=True)
            )
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            injections = list(
                pool.map(
                    lambda item: item[0].post_xid_event(item[1]),
                    zip(fixtures, payloads, strict=True),
                )
            )
        completed_at = datetime.now(timezone.utc)
        if (completed_at - injected_at).total_seconds() > 60:
            result["errors"].append("concurrent injections exceeded 60 seconds")
        states = []
        targets_after = []
        for fixture, workload, payload, source in zip(
            fixtures,
            workloads,
            payloads,
            sources,
            strict=True,
        ):
            state = fixture.wait_for_workflow(
                node=str(source["pods"][0]["node"]),
                marker=str(payload["record_id"]),
                observed_after=injected_at,
                case_dir=case_dir / str(fixture.settings.cluster_id),
                timeout_seconds=1200,
                job_id=settings.job_id,
                attempt_id=settings.attempt_id,
            )
            states.append(state)
            result["errors"].extend(
                workload_case.workflow_errors(state, expected_gpu_count=24)
            )
            targets_after.append(
                workload.wait_restarted(
                    {str(item["uid"]) for item in source["pods"]},
                    timeout_seconds=900,
                )
            )
        workflow_ids = {
            (state.get("workflow") or {}).get("request_id") for state in states
        }
        incident_ids = {
            (state.get("incident") or {}).get("incident_id") for state in states
        }
        if len(workflow_ids) != 2 or len(incident_ids) != 2:
            result["errors"].append("incident/workflow identities are not isolated")
        for fixture, state in zip(fixtures, states, strict=True):
            budget = state.get("restart_budget") or {}
            if budget.get("restart_count") != 1:
                result["errors"].append(
                    f"{fixture.settings.cluster_id} restart budget is not one"
                )
            commands = state.get("commands") or []
            if any(
                item.get("cluster_id") != fixture.settings.cluster_id
                for item in commands
            ):
                result["errors"].append("remote command crossed cluster scope")
        result.update(
            {
                "verdict": "PASS" if not result["errors"] else "FAIL",
                "injections": injections,
                "states": states,
                "target_workloads": targets_after,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for workload in workloads:
            try:
                workload.delete()
            except Exception as exc:
                result["errors"].append(
                    f"workload cleanup failed: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"
        for prewarm in prewarms:
            try:
                residuals = prewarm.cleanup()
                if any(residuals.values()):
                    result["errors"].append("prewarm Pods remain")
                    result["verdict"] = "FAIL"
            except Exception as exc:
                result["errors"].append(
                    f"prewarm cleanup failed: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def abort_on_signal(signum: int, _frame: object) -> None:
    raise RegionalFixtureError(f"received signal {signum}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run E2E-002 concurrent same-identity faults in two clusters."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--region", default="")
    value.add_argument("--cluster-a", required=True)
    value.add_argument("--gpu-a-kubeconfig", required=True)
    value.add_argument("--gpu-a-context", required=True)
    value.add_argument("--cluster-b", required=True)
    value.add_argument("--gpu-b-kubeconfig", required=True)
    value.add_argument("--gpu-b-context", required=True)
    value.add_argument("--site-file", default="")
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, abort_on_signal)
    signal.signal(signal.SIGINT, abort_on_signal)
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return execute_case(settings, arguments.run_dir, arguments.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(main())
