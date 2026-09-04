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
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)

CASE_IDS = (
    "GF-REGIONAL-COLLECT-001",
    "GF-REGIONAL-COLLECT-002",
    "GF-REGIONAL-COLLECT-003",
    "GF-REGIONAL-COLLECT-005",
    "GF-REGIONAL-COLLECT-009",
    "GF-REGIONAL-COLLECT-010",
    "GF-REGIONAL-COLLECT-011",
    "GF-REGIONAL-COLLECT-012",
)
PREDECESSORS = {
    "GF-REGIONAL-COLLECT-001": "GF-REGIONAL-E2E-002",
    "GF-REGIONAL-COLLECT-002": "GF-REGIONAL-COLLECT-001",
    "GF-REGIONAL-COLLECT-003": "GF-REGIONAL-COLLECT-002",
    "GF-REGIONAL-COLLECT-005": "GF-REGIONAL-COLLECT-004",
    "GF-REGIONAL-COLLECT-009": "GF-REGIONAL-COLLECT-007",
    "GF-REGIONAL-COLLECT-010": "GF-REGIONAL-COLLECT-009",
    "GF-REGIONAL-COLLECT-011": "GF-REGIONAL-COLLECT-010",
    "GF-REGIONAL-COLLECT-012": "GF-REGIONAL-COLLECT-011",
}
CONFIRMATIONS = {
    case_id: case_id.replace("GF-REGIONAL-", "").replace("-", "") + "_EXECUTE"
    for case_id in CASE_IDS
}
NODE_COUNTS = {
    **{case_id: 1 for case_id in CASE_IDS},
    "GF-REGIONAL-COLLECT-011": 2,
}


RECENT_EVIDENCE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, observed_after_text = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
records = [
    item.model_dump(mode="json")
    for item in store.list_raw_evidence(cluster_id, node_id=node_id, limit=5000)
    if item.observed_at >= observed_after
]
print(json.dumps({"records": records}, sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    case_id: str
    nodes: tuple[str, ...]
    host_probe_image: str
    predecessor_path: Path

    @property
    def confirmation(self) -> str:
        return CONFIRMATIONS[self.case_id]

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_COLLECT_CASE": self.case_id,
            "GPU_FAULT_COLLECT_NODES": ",".join(self.nodes),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    case_id = required(arguments.case, "case ID")
    nodes = tuple(arguments.node)
    expected = NODE_COUNTS[case_id]
    if len(nodes) != expected or len(set(nodes)) != expected:
        raise RegionalFixtureError(
            f"{case_id} requires exactly {expected} distinct --node values"
        )
    predecessor_id = PREDECESSORS[case_id]
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir / "cases" / predecessor_id / f"{predecessor_id}.json"
        ).resolve()
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        case_id=case_id,
        nodes=nodes,
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
    )


def focused_tests(case_id: str, case_dir: Path) -> dict[str, Any]:
    definitions = {
        "GF-REGIONAL-COLLECT-001": [
            "tests/collectors/test_gpu.py::"
            "test_dcgm_edge_filter_suppresses_unchanged_healthy_samples",
            "tests/collectors/test_host.py::"
            "test_host_edge_filter_suppresses_health_and_delivers_edges",
        ],
        "GF-REGIONAL-COLLECT-002": [
            "tests/collectors/test_gpu.py::"
            "test_dcgm_edge_filter_emits_counter_delta_and_candidate_recovery",
        ],
        "GF-REGIONAL-COLLECT-003": [
            "tests/collectors/test_host.py::"
            "test_efa_inventory_ignores_non_efa_rdma_devices",
            "tests/collectors/test_gpu.py::"
            "test_host_collector_reports_persistent_gpu_and_efa_card_loss",
        ],
        "GF-REGIONAL-COLLECT-005": [
            "tests/collectors/test_logs.py::"
            "test_fabric_manager_file_collector_persists_offsets",
            "tests/collectors/test_logs.py::"
            "test_fabric_manager_file_cursor_rolls_back_on_delivery_failure",
        ],
        "GF-REGIONAL-COLLECT-009": [
            "tests/execution/test_misc.py::"
            "test_mechanical_inspection_waits_for_incident_fenced_annotation",
        ],
        "GF-REGIONAL-COLLECT-010": [
            "tests/orchestration/test_misc.py::"
            "test_failed_firmware_remediation_escalates_directly_to_support",
        ],
        "GF-REGIONAL-COLLECT-011": [
            "tests/orchestration/test_sxid_topology.py::"
            "test_conflicting_or_untrusted_topology_fails_closed",
        ],
        "GF-REGIONAL-COLLECT-012": [
            "tests/orchestration/test_misc.py::"
            "test_restart_app_requires_workload_identity",
        ],
    }
    command = [sys.executable, "-m", "pytest", "-q", *definitions[case_id]]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
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
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    nodes = [regional.node_snapshot(node) for node in settings.nodes]
    states = [regional.store_snapshot(node=node) for node in settings.nodes]
    predecessor_id = PREDECESSORS[settings.case_id]
    predecessor = predecessor_evidence(settings.predecessor_path, predecessor_id)
    tests = focused_tests(settings.case_id, case_dir)
    errors = []
    if not predecessor["valid"]:
        errors.append(f"{predecessor_id} predecessor evidence is not PASS")
    for node, state in zip(nodes, states, strict=True):
        if node["ready"] != "True":
            errors.append(f"{node['name']} is not Ready")
        if node["ownership_annotations"]:
            errors.append(f"{node['name']} has pre-existing workflow ownership")
        if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
            errors.append(f"{node['name']} Node Agent is not ACTIVE")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": states[0].get("release_id") if states else None,
        "nodes": nodes,
        "stores": states,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def recent_evidence(
    regional: RegionalLiveFixture,
    *,
    node: str,
    observed_after: datetime,
) -> list[dict[str, Any]]:
    value = regional.cpu_python(
        RECENT_EVIDENCE,
        regional.settings.cluster_id,
        node,
        observed_after.isoformat(),
    )
    return cast(list[dict[str, Any]], value.get("records") or [])


def evidence_kinds(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        result.setdefault(str(item.get("kind")), []).append(item)
    return result


def run_collect001(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    env = baseline["collector_env"]
    interval = max(
        int(env["GPU_FAULT_DCGM_INTERVAL_SECONDS"]),
        int(env["GPU_FAULT_HOST_INTERVAL_SECONDS"]),
    )
    summary = max(
        int(env["GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS"]),
        int(env["GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS"]),
    )
    started_at = datetime.now(timezone.utc)
    duration = summary * 2 + interval * 2
    time.sleep(duration)
    records = recent_evidence(
        fixture.regional,
        node=fixture.node,
        observed_after=started_at,
    )
    by_kind = evidence_kinds(records)
    errors = []
    for kind in ("GPU_METRICS", "HOST_TELEMETRY"):
        items = sorted(
            by_kind.get(kind, []),
            key=lambda item: str(item.get("observed_at")),
        )
        if len(items) < 2:
            errors.append(f"{kind} has fewer than two delivered records")
            continue
        stamps = [
            datetime.fromisoformat(str(item["observed_at"]).replace("Z", "+00:00"))
            for item in items
        ]
        gaps = [
            (right - left).total_seconds()
            for left, right in zip(stamps, stamps[1:], strict=False)
        ]
        if any(abs(gap - summary) > interval * 2 for gap in gaps):
            errors.append(f"{kind} delivery gaps do not match summary cadence")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "collector_env": env,
        "observation_seconds": duration,
        "evidence_counts": {key: len(value) for key, value in sorted(by_kind.items())},
    }


def run_collect002(
    fixture: CollectorAcceptanceFixture,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    env = baseline["collector_env"]
    interval = int(env["GPU_FAULT_DCGM_INTERVAL_SECONDS"])
    persistence = baseline.get("persistence_mode")
    if persistence is None:
        raise RegionalFixtureError("cannot determine baseline persistence mode")
    started_at = datetime.now(timezone.utc)
    fixture.execute("set-persistence-mode", "--enabled", "false")
    try:
        time.sleep(interval * 2)
        records = recent_evidence(
            fixture.regional,
            node=fixture.node,
            observed_after=started_at,
        )
    finally:
        fixture.execute(
            "set-persistence-mode",
            "--enabled",
            "true" if persistence else "false",
        )
    gpu = evidence_kinds(records).get("GPU_METRICS", [])
    candidate = [
        item
        for item in gpu
        if any(
            "candidate" in str(reason).lower() or "threshold" in str(reason).lower()
            for reason in item.get("payload", {}).get("edge_filter_reasons", [])
        )
    ]
    errors = [] if candidate else ["DCGM anomaly did not bypass the edge filter"]
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "candidate_records": candidate,
    }


def run_collect003(
    fixture: CollectorAcceptanceFixture,
) -> dict[str, Any]:
    snapshot = fixture.snapshot()
    env = snapshot["collector_env"]
    gpu_actual = len(snapshot["gpu_inventory"])
    efa = snapshot["efa_inventory"]
    expected_gpu = int(env["GPU_FAULT_EXPECTED_GPU_COUNT"])
    expected_efa = int(env["GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT"])
    errors = []
    if gpu_actual != expected_gpu:
        errors.append("actual GPU count differs from collector configuration")
    if int(efa["active_count"]) != expected_efa:
        errors.append("active EFA count differs from collector configuration")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gpu_actual": gpu_actual,
        "gpu_expected": expected_gpu,
        "efa_inventory": efa,
        "efa_expected": expected_efa,
    }


def run_collect005(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    before = fixture.snapshot()
    gpu = before["gpu_inventory"][0]
    marker = f"c005-{int(time.time())}-a{attempt}"
    fixture.execute(
        "append-sxid",
        "--sxid",
        "99999",
        "--marker",
        marker,
        "--pci-bdf",
        str(gpu["pci_bdf"]),
        "--classification",
        "Non-fatal",
        "--message",
        "unknown collection-only event",
    )
    first = fixture.wait_marker(marker, case_dir=case_dir, timeout_seconds=120)
    count = len(first.get("evidence") or [])
    fixture.execute(
        "restart-service",
        "--service",
        "gpu-fault-fabric-manager-collector.service",
    )
    time.sleep(15)
    second = fixture.store_snapshot(marker)
    errors = []
    if count != 1 or len(second.get("evidence") or []) != 1:
        errors.append("Fabric Manager event was lost or replayed")
    if first.get("workflows") or second.get("workflows"):
        errors.append("unknown non-fatal SXID created a workflow")
    after = fixture.snapshot()
    before_size = (before.get("fabric_manager_log") or {}).get("size", 0)
    after_size = (after.get("fabric_manager_log") or {}).get("size", 0)
    if int(after_size) <= int(before_size):
        errors.append("Fabric Manager log cursor source did not advance")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "before": before,
        "after": after,
    }


def waiting_workflow(
    fixture: CollectorAcceptanceFixture,
    marker: str,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = fixture.store_snapshot(marker)
        if any(
            execution.get("status") == "WAITING"
            for workflow in last.get("workflows") or []
            for execution in workflow.get("step_executions", [])
        ):
            return last
        time.sleep(2)
    raise RegionalFixtureError(f"workflow did not reach WAITING: {last}")


def run_collect009(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    gpu = baseline["gpu_inventory"][0]
    marker = f"c009-{int(time.time())}-a{attempt}"
    fixture.execute(
        "write-xid",
        "--xid",
        "54",
        "--marker",
        marker,
        "--pci-bdf",
        str(gpu["pci_bdf"]),
        "--message",
        "Auxiliary power is not connected to the GPU board",
    )
    state = waiting_workflow(fixture, marker)
    workflow = (state.get("workflows") or [None])[0] or {}
    execution: dict[str, Any] = next(
        (
            item
            for item in workflow.get("step_executions", [])
            if item.get("operation") == "CHECK_MECHANICALS"
        ),
        {},
    )
    details = execution.get("details") or {}
    annotation = str(details.get("required_annotation") or "")
    expected = str(details.get("required_annotation_value") or "")
    if not annotation or not expected:
        raise RegionalFixtureError("CHECK_MECHANICALS omitted annotation details")
    fixture.regional.kubectl(
        "gpu",
        "annotate",
        "node",
        fixture.node,
        f"{annotation}=wrong:0",
        "--overwrite",
    )
    time.sleep(10)
    wrong = fixture.store_snapshot(marker)
    fixture.regional.kubectl(
        "gpu",
        "annotate",
        "node",
        fixture.node,
        f"{annotation}={expected}",
        "--overwrite",
    )
    final = fixture.wait_marker(
        marker,
        case_dir=case_dir,
        timeout_seconds=300,
        terminal_workflow=True,
    )
    fixture.regional.kubectl(
        "gpu",
        "annotate",
        "node",
        fixture.node,
        f"{annotation}-",
        check=False,
    )
    errors = []
    if any(
        workflow.get("status") == "SUCCEEDED"
        for workflow in wrong.get("workflows") or []
    ):
        errors.append("wrong mechanical acknowledgement was accepted")
    if not any(
        workflow.get("status") == "SUCCEEDED"
        for workflow in final.get("workflows") or []
    ):
        errors.append("correct mechanical acknowledgement did not complete")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "required_annotation": annotation,
        "required_value": expected,
    }


def inject_blocked_xid(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    *,
    xid: int,
    marker: str,
    message: str,
) -> dict[str, Any]:
    gpu = fixture.snapshot()["gpu_inventory"][0]
    fixture.execute(
        "write-xid",
        "--xid",
        str(xid),
        "--marker",
        marker,
        "--pci-bdf",
        str(gpu["pci_bdf"]),
        "--message",
        message,
    )
    return cast(
        dict[str, Any],
        fixture.wait_marker(
            marker,
            case_dir=case_dir,
            timeout_seconds=300,
            terminal_workflow=True,
        ),
    )


def run_collect010(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
) -> dict[str, Any]:
    baseline = fixture.snapshot()
    marker = f"c010-{int(time.time())}-a{attempt}"
    state = inject_blocked_xid(
        fixture,
        case_dir,
        xid=78,
        marker=marker,
        message="GPU firmware update required",
    )
    errors = []
    workflows = state.get("workflows") or []
    if not workflows or workflows[0].get("status") != "BLOCKED":
        errors.append("UPDATE_SWFW workflow is not BLOCKED")
    operations = {
        item.get("operation")
        for workflow in workflows
        for item in workflow.get("official_steps", [])
    }
    if "UPDATE_SOFTWARE_FIRMWARE" in operations:
        errors.append("blocked workflow contains firmware mutation")
    after = fixture.snapshot()
    if baseline["boot_id"] != after["boot_id"]:
        errors.append("blocked firmware case rebooted the node")
    restore = fixture.restore_incidents(
        state,
        profile_version=profile_version,
        reason="COLLECT-010 validated cleanup",
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "restore_workflows": restore,
    }


def run_collect011(
    fixtures: list[CollectorAcceptanceFixture],
    case_dir: Path,
    attempt: int,
    profile_version: str,
) -> dict[str, Any]:
    markers = [
        f"c011-access-{int(time.time())}-a{attempt}",
        f"c011-unknown-{int(time.time())}-a{attempt}",
    ]
    states = []
    for index, (fixture, marker) in enumerate(zip(fixtures, markers, strict=True)):
        gpu = fixture.snapshot()["gpu_inventory"][0]
        fixture.execute(
            "append-sxid",
            "--sxid",
            "11001",
            "--marker",
            marker,
            "--pci-bdf",
            str(gpu["pci_bdf"]),
            "--classification",
            "Fatal",
            "--message",
            "NVLINK_FATAL_ERROR",
            "--include-switch" if index == 0 else "--no-include-switch",
        )
        states.append(
            fixture.wait_marker(
                marker,
                case_dir=case_dir / f"direction-{index + 1}",
                timeout_seconds=300,
                terminal_workflow=True,
            )
        )
    errors = []
    for state in states:
        if (
            not state.get("workflows")
            or state["workflows"][0].get("status") != "BLOCKED"
        ):
            errors.append("scope-dependent SXID did not fail closed")
    restores = [
        fixture.restore_incidents(
            state,
            profile_version=profile_version,
            reason="COLLECT-011 validated cleanup",
        )
        for fixture, state in zip(fixtures, states, strict=True)
    ]
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "markers": markers,
        "restore_workflows": restores,
    }


def run_collect012(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
) -> dict[str, Any]:
    markers = []
    states = []
    for offset, xid in enumerate((13, 13, 31), start=1):
        marker = f"c012-{xid}-{offset}-{int(time.time())}-a{attempt}"
        markers.append(marker)
        states.append(
            inject_blocked_xid(
                fixture,
                case_dir / f"sample-{offset}",
                xid=xid,
                marker=marker,
                message="RESTART_APP collector path",
            )
        )
    errors = []
    record_ids = {
        item.get("record_id")
        for state in states
        for item in state.get("evidence") or []
    }
    if len(record_ids) < 3:
        errors.append("distinct kmsg sequences did not create distinct evidence")
    for state in states:
        workflows = state.get("workflows") or []
        if not workflows or workflows[0].get("status") != "BLOCKED":
            errors.append("RESTART_APP without workload did not fail closed")
    restores = [
        fixture.restore_incidents(
            state,
            profile_version=profile_version,
            reason="COLLECT-012 validated cleanup",
        )
        for state in states
    ]
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "markers": markers,
        "record_ids": sorted(str(item) for item in record_ids if item),
        "restore_workflows": restores,
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "case-defined",
        "case_id": settings.case_id,
        "predecessor": preflight["predecessor"],
        "nodes": list(settings.nodes),
        "mutation": {
            "GF-REGIONAL-COLLECT-001": "read evidence history for two summary cycles",
            "GF-REGIONAL-COLLECT-002": "temporarily toggle NVIDIA persistence mode",
            "GF-REGIONAL-COLLECT-003": "read live GPU/EFA inventory and env",
            "GF-REGIONAL-COLLECT-005": "append one unknown non-fatal SXID and restart collector",
            "GF-REGIONAL-COLLECT-009": "inject XID54 and write exact acknowledgement annotation",
            "GF-REGIONAL-COLLECT-010": "inject XID78 fail-closed quarantine only",
            "GF-REGIONAL-COLLECT-011": "append scope-dependent SXID on two nodes",
            "GF-REGIONAL-COLLECT-012": "inject XID13/31 via real kmsg and restore quarantine",
        }[settings.case_id],
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uids": [item["uid"] for item in preflight["nodes"]],
        },
        "stop_conditions": [
            "predecessor evidence is not PASS",
            "node, Agent or release baseline drifts",
            "host probe cannot read current collector configuration",
            "case-specific evidence or rollback assertion fails",
        ],
        "rollback": {
            "host_probe_has_active_deadline": True,
            "temporary_persistence_or_annotations_are_restored": True,
            "quarantine_is_restored_only_by_validation_workflow": True,
        },
        "preflight": preflight,
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    regional = RegionalLiveFixture(settings.regional)
    fixtures = [
        CollectorAcceptanceFixture(
            regional,
            node=node,
            image=settings.host_probe_image,
            case_id=settings.case_id,
            run_id=f"{settings.case_id.lower()}-{attempt}-{index}",
        )
        for index, node in enumerate(settings.nodes)
    ]
    result: dict[str, Any] = {
        "case_id": settings.case_id,
        "attempt": attempt,
        "verdict": "FAIL",
    }
    try:
        for fixture in fixtures:
            fixture.create()
        profile_version = str(
            (preflight["stores"][0].get("profile") or {}).get("profile_version") or ""
        )
        handlers = {
            "GF-REGIONAL-COLLECT-001": lambda: run_collect001(fixtures[0], case_dir),
            "GF-REGIONAL-COLLECT-002": lambda: run_collect002(fixtures[0]),
            "GF-REGIONAL-COLLECT-003": lambda: run_collect003(fixtures[0]),
            "GF-REGIONAL-COLLECT-005": lambda: run_collect005(
                fixtures[0], case_dir, attempt
            ),
            "GF-REGIONAL-COLLECT-009": lambda: run_collect009(
                fixtures[0], case_dir, attempt
            ),
            "GF-REGIONAL-COLLECT-010": lambda: run_collect010(
                fixtures[0], case_dir, attempt, profile_version
            ),
            "GF-REGIONAL-COLLECT-011": lambda: run_collect011(
                fixtures, case_dir, attempt, profile_version
            ),
            "GF-REGIONAL-COLLECT-012": lambda: run_collect012(
                fixtures[0], case_dir, attempt, profile_version
            ),
        }
        outcome = handlers[settings.case_id]()
        result.update(outcome)
        if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
            result.setdefault("errors", []).append(
                "control-plane EKS state differs from baseline"
            )
            result["verdict"] = "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        residuals = {}
        for fixture in fixtures:
            try:
                residuals[fixture.node] = fixture.cleanup()
            except Exception as exc:
                residuals[fixture.node] = {"cleanup_error": True}
                result.setdefault("cleanup_errors", []).append(
                    f"{fixture.node}: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
        if any(any(value.values()) for value in residuals.values()):
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{settings.case_id}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one guarded regional Collector acceptance case."
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", action="append", default=[])
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=settings.case_id,
            attempt=arguments.attempt,
            confirmation=settings.confirmation,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    if arguments.confirm != settings.confirmation:
        raise RegionalFixtureError(
            f"confirmation must be exactly {settings.confirmation}"
        )
    deadline = authorize_execution(
        arguments,
        case_id=settings.case_id,
        confirmation=settings.confirmation,
        environment=settings.environment(),
    )
    return execute_case(settings, arguments.run_dir, arguments.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
