#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from scripts.e2e.regional.identity_acceptance_common import (  # noqa: E402
    claim_sample,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    ClusterTarget,
    MultiClusterSettings,
    container_status_errors,
    control_plane_container_statuses,
    control_plane_pressure,
    registration_snapshot,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    predecessor_evidence,
    required,
    run_case_main,
)

CASE_ID = "GF-REGIONAL-ISO-006"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-013"
CONFIRMATION = "ISO006_BLOCK_CLUSTER_B_CONTROL_PLANE_443"
PROBE = Path(__file__).with_name("probes") / "cluster_network_probe.py"
# The window was 1800 s while the case measured nothing during it; with real
# sampling every 30 s, 900 s gives 30 A-claim samples and is the default. The
# catalog's 30-minute window remains reachable through --duration-seconds.
DEFAULT_DURATION_SECONDS = 900
SAMPLE_INTERVAL_SECONDS = 30
BASELINE_SAMPLES = 5
BASELINE_INTERVAL_SECONDS = 10
# Margin between the observation window's end and the per-node restore timer:
# the runner must unblock (and measure B's recovery) before any timer fires,
# and the timers are armed from each node's own block time.
RESTORE_MARGIN_SECONDS = 300
MAX_BLOCK_SPAN_SECONDS = 120
RECOVERY_TIMEOUT_SECONDS = 300
LATENCY_P95_MAX_INCREASE = 0.20
VALIDATION_LIMITATIONS = [
    "store_io admission and the processor worker pool are shared across the "
    "whole region; there is no per-cluster quota, so a blocked cluster's "
    "backlog is bounded only by the absence of its requests.",
    "The block is an iptables REJECT on cluster B's nodes toward the "
    "control-plane CIDRs, not a security-group change; B's inbound requests "
    "never reach the control plane, so cluster_queue_depth for B is recorded "
    "rather than required to grow.",
    "A-cluster claim latency is sampled from one Ready executor Pod with the "
    "acceptance probe owner; it measures the authenticated claim path, not "
    "the executor's own poll loop.",
]


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

    @property
    def restore_seconds(self) -> int:
        return min(3600, self.duration_seconds + RESTORE_MARGIN_SECONDS)


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


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``."""

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
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
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=300)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    a = settings.multi.regional(settings.multi.cluster_a)
    b = settings.multi.regional(settings.multi.cluster_b)
    registrations = registration_snapshot(a, settings.multi)
    identity = a.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        release_id=identity["release_id"],
    )
    nodes_a = a.gpu_nodes()
    nodes_b = b.gpu_nodes()
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    # Both clusters must claim successfully before anything is blocked; the
    # "cut proven" step later compares against exactly this.
    claim_a = claim_sample(a)
    claim_b = claim_sample(b)
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
    if claim_a.get("status") != 200 or claim_b.get("status") != 200:
        errors.append("a cluster cannot claim before the block")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        **identity,
        "registrations": registrations,
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "cpu_pods": cpu_pod_snapshot(a),
        "cpu_containers": control_plane_container_statuses(a),
        "claim_a": claim_a,
        "claim_b": claim_b,
        "focused_tests": tests,
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = {
        "risk": "live-non-destructive",
        "predecessor": preflight["predecessor"],
        "cluster_a": settings.multi.cluster_a.cluster_id,
        "cluster_b": settings.multi.cluster_b.cluster_id,
        "cluster_b_nodes": [item["name"] for item in preflight["nodes_b"]],
        "mutation": (
            "block cluster B node egress (OUTPUT and FORWARD) to explicit "
            "control-plane CIDRs on TCP 443 for "
            f"{settings.duration_seconds}s; cluster A remains an unmodified control"
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
            "the cut cannot be proven from a B executor Pod",
            "cluster A readiness or control-plane Pods regress",
            "rules cannot be removed or B does not recover",
        ],
        "rollback": {
            "host_restore_seconds": settings.restore_seconds,
            "runner_finally_unblocks_every_B_node": True,
            "probe_resources_have_active_deadlines": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def latency_p95(samples: list[dict[str, Any]]) -> float | None:
    values = sorted(
        float(item["latency_seconds"])
        for item in samples
        if isinstance(item.get("latency_seconds"), (int, float))
        and item.get("status") is not None
    )
    if not values:
        return None
    index = max(0, min(len(values) - 1, round(0.95 * (len(values) - 1))))
    return values[index]


def latency_errors(
    baseline: list[dict[str, Any]],
    window: list[dict[str, Any]],
) -> list[str]:
    """The catalog's A-cluster judgments: p95 within +20 %, no 5xx."""

    errors = []
    base = latency_p95(baseline)
    during = latency_p95(window)
    if base is None or during is None:
        errors.append("claim latency could not be measured in both phases")
    elif during > base * (1 + LATENCY_P95_MAX_INCREASE):
        errors.append(
            f"A claim p95 rose from {base:.3f}s to {during:.3f}s "
            f"(> {LATENCY_P95_MAX_INCREASE:.0%})"
        )
    for item in window:
        status = item.get("status")
        if status is None:
            errors.append("A claim failed at transport level during the block")
            break
        if int(status) >= 500:
            errors.append(f"A claim returned {status} during the block")
            break
    return errors


def pressure_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    errors = []
    for name, value in after.get("rejections", {}).items():
        if float(value) > float(before.get("rejections", {}).get(name, 0.0)):
            errors.append(f"{name} increased during the block")
    return errors


def cut_proven(sample: dict[str, Any]) -> bool:
    """A blocked cluster's claim must fail at the transport, not with an HTTP status."""

    return sample.get("status") is None and bool(sample.get("transport_error"))


def block_arguments(settings: Settings, run_id: str) -> list[str]:
    return [
        "block",
        "--run-id",
        run_id,
        *[
            value
            for cidr in settings.control_plane_cidrs
            for value in ("--control-plane-cidr", cidr)
        ],
        "--restore-seconds",
        str(settings.restore_seconds),
    ]


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    a = settings.multi.regional(settings.multi.cluster_a)
    b = settings.multi.regional(settings.multi.cluster_b)
    cluster_b = settings.multi.cluster_b.cluster_id
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
        **a.evidence_identity(),
        "verdict": "FAIL",
        "duration_seconds": settings.duration_seconds,
        "baseline_samples": [],
        "samples": [],
        "errors": [],
        "validation_limitations": VALIDATION_LIMITATIONS,
    }
    blocked = False
    try:
        containers_before = preflight["cpu_containers"]
        pressure_before = control_plane_pressure(a, cluster_b)
        for _ in range(BASELINE_SAMPLES):
            result["baseline_samples"].append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    **claim_sample(a),
                }
            )
            time.sleep(BASELINE_INTERVAL_SECONDS)
        # Every probe Pod Ready first, then the blocks back to back: each
        # node's restore timer is armed from its own block time, so the
        # blocks must not be spread over the Pod scheduling of the others.
        with ThreadPoolExecutor(max_workers=max(1, len(probes))) as pool:
            list(pool.map(lambda probe: probe.create(), probes))
        first_block = time.monotonic()
        blocks = []
        for probe in probes:
            blocks.append(probe.execute(*block_arguments(settings, run_id)))
            blocked = True
        block_span = time.monotonic() - first_block
        result["blocks"] = blocks
        result["block_span_seconds"] = block_span
        if block_span > MAX_BLOCK_SPAN_SECONDS:
            raise RegionalFixtureError(
                f"blocking B nodes took {block_span:.0f}s; a restore timer could "
                "fire inside the observation window"
            )
        # Refuse to observe a block that is not a block.
        cut = claim_sample(b)
        result["cut_proof"] = cut
        if not cut_proven(cut):
            raise RegionalFixtureError(
                f"cluster B still reaches the control plane after the block: {cut}"
            )
        window_started = time.monotonic()
        deadline = window_started + settings.duration_seconds
        while time.monotonic() < deadline:
            sample = {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "a_claim": claim_sample(a),
                "a_ready_executors": len(
                    a.ready_pods("gpu", "gpu-fault-cluster-executor")
                ),
                "b_pressure": control_plane_pressure(a, cluster_b),
                "cpu_pods": cpu_pod_snapshot(a),
            }
            result["samples"].append(sample)
            if not sample["a_ready_executors"]:
                raise RegionalFixtureError("cluster A executor lost readiness")
            time.sleep(
                max(0.0, min(SAMPLE_INTERVAL_SECONDS, deadline - time.monotonic()))
            )
        containers_after = control_plane_container_statuses(a)
        pressure_after = (
            result["samples"][-1]["b_pressure"] if result["samples"] else {}
        )
        errors = latency_errors(
            result["baseline_samples"],
            [item["a_claim"] for item in result["samples"]],
        )
        errors.extend(pressure_errors(pressure_before, pressure_after))
        errors.extend(container_status_errors(containers_before, containers_after))
        result["pressure_before"] = pressure_before
        result["pressure_after"] = pressure_after
        result["b_queue_depth_delta"] = float(
            pressure_after.get("cluster_queue_depth", 0.0)
        ) - float(pressure_before.get("cluster_queue_depth", 0.0))
        result["cpu_containers_after"] = containers_after
        result["a_claim_p95_baseline_seconds"] = latency_p95(result["baseline_samples"])
        result["a_claim_p95_window_seconds"] = latency_p95(
            [item["a_claim"] for item in result["samples"]]
        )
        # Unblock inside the window's accounting so B's recovery is measured
        # from the moment the rules left, not from the end of cleanup.
        for probe in probes:
            probe.execute("unblock", "--run-id", run_id)
        blocked = False
        unblocked_at = time.monotonic()
        recovery: dict[str, Any] | None = None
        recovery_deadline = unblocked_at + RECOVERY_TIMEOUT_SECONDS
        while time.monotonic() < recovery_deadline:
            sample = claim_sample(b)
            if sample.get("status") == 200:
                recovery = sample
                break
            time.sleep(10)
        result["b_recovery_seconds"] = (
            time.monotonic() - unblocked_at if recovery is not None else None
        )
        result["b_recovery_claim"] = recovery
        if recovery is None:
            errors.append(
                f"cluster B did not claim within {RECOVERY_TIMEOUT_SECONDS}s of unblock"
            )
        result["errors"] = errors
        result["verdict"] = "PASS" if not errors else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        residuals = {}
        for index, probe in enumerate(probes):
            if blocked:
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
    value.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
