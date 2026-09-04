#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    ClusterTarget,
    MultiClusterSettings,
    registration_snapshot,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
)

CASE_ID = "GF-REGIONAL-ISO-006"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-013"
CONFIRMATION = "ISO006_BLOCK_CLUSTER_B_CONTROL_PLANE_443"
PROBE = Path(__file__).with_name("probes") / "cluster_network_probe.py"


@dataclass(frozen=True)
class Settings:
    multi: MultiClusterSettings
    host_probe_image: str
    control_plane_cidrs: tuple[str, ...]
    duration_seconds: int
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.multi.environment(),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_CONTROL_PLANE_CIDRS": ",".join(self.control_plane_cidrs),
            "GPU_FAULT_ISO006_DURATION_SECONDS": str(self.duration_seconds),
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
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
    if not 60 <= arguments.duration_seconds <= 1800:
        raise RegionalFixtureError("duration seconds is outside 60..1800")
    if not arguments.control_plane_cidr:
        raise RegionalFixtureError("at least one control-plane CIDR is required")
    return Settings(
        multi=multi,
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        control_plane_cidrs=tuple(arguments.control_plane_cidr),
        duration_seconds=arguments.duration_seconds,
        predecessor_path=predecessor,
    )


def cpu_pod_snapshot(
    fixture: RegionalLiveFixture,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for app in ("gpu-fault-api-ha", "gpu-fault-control-worker"):
        result.extend({"app": app, **item} for item in fixture.ready_pods("cpu", app))
    return result


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
        "test_regional_api_authenticates_cluster_and_rejects_spoofing",
    ]
    completed = a.run(command, cwd=ROOT, check=False, timeout=300)
    (case_dir / "focused-tests.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("DESTR-013 predecessor evidence is not PASS")
    if len(registrations) != 2 or any(
        not item.get("enabled") or item.get("synthetic") for item in registrations
    ):
        errors.append("two enabled non-synthetic cluster registrations are required")
    elif not registrations_are_distinct_physical_clusters(registrations):
        errors.append("cluster registrations do not identify two physical clusters")
    if not nodes_a or not nodes_b:
        errors.append("both clusters require GPU nodes")
    if any(item["ready"] != "True" for item in [*nodes_a, *nodes_b]):
        errors.append("a GPU node is not Ready")
    if completed.returncode:
        errors.append("focused regression tests failed")
    result = {
        "release_id": a.release_id(),
        "registrations": registrations,
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "cpu_pods": cpu_pod_snapshot(a),
        "cpu_blast": a.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "predecessor": preflight["predecessor"],
        "cluster_a": settings.multi.cluster_a.cluster_id,
        "cluster_b": settings.multi.cluster_b.cluster_id,
        "cluster_b_nodes": [item["name"] for item in preflight["nodes_b"]],
        "mutation": (
            "block cluster B node egress to explicit control-plane CIDRs on TCP 443 "
            f"for {settings.duration_seconds}s; cluster A remains an unmodified control"
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
            "DESTR-013 predecessor evidence is not PASS",
            "fewer than two enabled physical clusters",
            "network rollback timer cannot be armed on every B node",
            "cluster A readiness or control-plane Pods regress",
            "rules cannot be removed or B does not recover",
        ],
        "rollback": {
            "host_restore_seconds": min(3600, settings.duration_seconds + 300),
            "runner_finally_unblocks_every_B_node": True,
            "probe_resources_have_active_deadlines": True,
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
    a = settings.multi.regional(settings.multi.cluster_a)
    b = settings.multi.regional(settings.multi.cluster_b)
    run_id = f"iso006-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    probes = [
        HostProbeFixture(
            HostProbeSettings(
                kubeconfig=settings.multi.cluster_b.gpu_kubeconfig,
                context=settings.multi.cluster_b.gpu_context,
                namespace=settings.multi.namespace,
                node=str(item["name"]),
                image=settings.host_probe_image,
                case_id=CASE_ID,
                run_id=f"{run_id}-{index}",
                probe_script=PROBE,
                active_deadline_seconds=3600,
            )
        )
        for index, item in enumerate(preflight["nodes_b"])
    ]
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "samples": [],
    }
    try:
        for probe in probes:
            probe.create()
            probe.execute(
                "block",
                "--run-id",
                run_id,
                *[
                    value
                    for cidr in settings.control_plane_cidrs
                    for value in ("--control-plane-cidr", cidr)
                ],
                "--restore-seconds",
                str(min(3600, settings.duration_seconds + 300)),
            )
        deadline = time.monotonic() + settings.duration_seconds
        while time.monotonic() < deadline:
            result["samples"].append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "a_ready_executors": len(
                        a.ready_pods("gpu", "gpu-fault-cluster-executor")
                    ),
                    "b_ready_executors": len(
                        b.ready_pods("gpu", "gpu-fault-cluster-executor")
                    ),
                    "cpu_pods": cpu_pod_snapshot(a),
                }
            )
            if not result["samples"][-1]["a_ready_executors"]:
                raise RegionalFixtureError("cluster A executor lost readiness")
            time.sleep(min(30, settings.duration_seconds))
        errors = []
        if a.cpu_blast_snapshot() != preflight["cpu_blast"]:
            errors.append("control-plane EKS state changed")
        result["errors"] = errors
        result["verdict"] = "PASS" if not errors else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        residuals = {}
        for index, probe in enumerate(probes):
            try:
                probe.execute("unblock", "--run-id", run_id)
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(
                    f"unblock {index}: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"
            try:
                residuals[str(index)] = probe.cleanup()
            except Exception as exc:
                residuals[str(index)] = {"cleanup_error": True}
                result.setdefault("cleanup_errors", []).append(
                    f"probe {index}: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run ISO-006 whole-cluster control-plane isolation."
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
    value.add_argument("--control-plane-cidr", action="append", default=[])
    value.add_argument("--duration-seconds", type=int, default=1800)
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
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
    raise SystemExit(run_case_main(main))
