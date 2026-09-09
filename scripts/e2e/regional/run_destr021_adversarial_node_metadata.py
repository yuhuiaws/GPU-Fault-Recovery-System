#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-021: EFA remediation under adversarial node metadata.

One idle GPU node. Before the injection the runner writes a *stale, foreign*
device-plugin-restart annotation set on the node (ARCH-A3's poison pill), and
from the injection until the workflow is terminal a bounded local writer keeps
patching a harmless tick annotation so every node patch the deployed executor
issues races a real ``409 Conflict`` (ARCH-A2). Then COLLECT-017 A's real EFA
unbind opens the seven-step remediation workflow the deployed control plane
already proved.

What has to hold: ``RESTART_EFA_DEVICE_PLUGIN`` takes the stale foreign
annotation over (``node_results[node].took_over_incident``) and leaves none of
the four keys behind; it was dispatched with a positive ``expected_count`` equal
to the node's EFA allocatable; ``MARK_UNSCHEDULABLE`` and ``RESTORE_SCHEDULING``
record observed ``before``/``after`` scheduling baselines on their remote
command (ARCH-A8); ``RESTORE_SCHEDULING`` ends SUCCEEDED under the race and the
node is never left cordoned. No reset, no reboot, no provider call: the only
mutations are the EFA function unbind (with the host-side rebind fail-safe
COLLECT-017 already uses) and node annotations the runner removes itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import destr021_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import run_collect017_efa_plugin as c017  # noqa: E402
from scripts.e2e.regional import run_collector_destructive as base  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.probes.destr021_annotation_writer import (  # noqa: E402
    AnnotationWriter,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
    waiting_step_executions,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    NodeMutationFixture,
    NodePatch,
    WarmSpareLiveFixture,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
MUTATED_ANNOTATIONS = (*verdicts.PLUGIN_RESTART_ANNOTATIONS, verdicts.TICK_ANNOTATION)
FOCUSED_TESTS = (
    "tests/collectors/test_host.py::"
    "test_efa_inventory_distinguishes_unbound_driver_from_missing_pci",
    "tests/orchestration/test_health.py::"
    "test_idle_node_resource_group_upgrades_plugin_to_driver_remediation",
    "tests/regional/test_destr021_adversarial_node_metadata.py",
)


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    site_file: Path
    host_probe_image: str
    predecessor_path: Path
    writer_interval_seconds: float
    writer_max_seconds: float

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
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
    interval = float(arguments.writer_interval_seconds)
    max_seconds = float(arguments.writer_max_seconds)
    if interval < 0.2:
        raise RegionalFixtureError("writer interval below 0.2 s would be an API flood")
    if not 60 <= max_seconds <= 3600:
        raise RegionalFixtureError("writer max seconds is outside 60..3600")
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
        writer_interval_seconds=interval,
        writer_max_seconds=max_seconds,
    )


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def node_document(regional: RegionalLiveFixture, node: str) -> dict[str, Any]:
    """Raw annotations and EFA allocatable, which ``node_snapshot`` filters."""

    value = json.loads(regional.kubectl("gpu", "get", "node", node, "-o", "json"))
    annotations = dict(value["metadata"].get("annotations") or {})
    allocatable = value.get("status", {}).get("allocatable", {})
    try:
        efa = int(allocatable.get(verdicts.EFA_RESOURCE, 0))
    except (TypeError, ValueError):
        efa = 0
    return {
        "annotations": annotations,
        "resource_version": value["metadata"].get("resourceVersion"),
        "efa_allocatable": efa,
    }


def bound_efa_functions(inventory: dict[str, Any]) -> list[str]:
    return [
        str(item["pci_bdf"])
        for item in inventory.get("devices") or []
        if item.get("pci_bdf") and item.get("driver") == "efa"
    ]


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=600)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def _collector(settings: Settings, run_id: str) -> CollectorAcceptanceFixture:
    return CollectorAcceptanceFixture(
        RegionalLiveFixture(settings.regional),
        node=settings.node,
        image=settings.host_probe_image,
        case_id=CASE_ID,
        run_id=run_id,
    )


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    document = node_document(regional, settings.node)
    workloads = regional.business_workloads(settings.node)
    state = regional.store_snapshot(node=settings.node)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    tests = focused_tests(case_dir)
    runtime_identity = regional.runtime_identity()
    collector = _collector(settings, "destr021-preflight")
    try:
        collector.create()
        inventory = collector.snapshot().get("efa_inventory") or {}
    finally:
        residual = collector.cleanup()
    bound = bound_efa_functions(inventory)
    errors = verdicts.preflight_errors(
        node=node,
        node_annotations=document["annotations"],
        workloads=workloads,
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        efa_allocatable=document["efa_allocatable"],
        bound_efa_functions=len(bound),
        predecessor_valid=bool(predecessor["valid"]),
        tests_passed=tests["passed"],
        site_file_exists=settings.site_file.is_file(),
    )
    errors.extend(runtime_identity_errors(runtime_identity))
    if any(residual.values()):
        errors.append(f"probe residue remains after the preflight: {residual}")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "efa_allocatable": document["efa_allocatable"],
        "bound_efa_functions": bound,
        "plugin_restart_annotations": {
            key: document["annotations"].get(key)
            for key in verdicts.PLUGIN_RESTART_ANNOTATIONS
        },
        "business_workloads": workloads,
        "store": {
            "queue": state.get("queue"),
            "remote_commands": state.get("remote_commands"),
            "agent": state.get("agent"),
        },
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "runtime_identity": runtime_identity,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "node_boot_id": preflight["node"].get("boot_id"),
        "efa_allocatable": preflight["efa_allocatable"],
        "bound_efa_functions": list(preflight["bound_efa_functions"]),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": verdicts.RISK,
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "writer": {
            "annotation": verdicts.TICK_ANNOTATION,
            "interval_seconds": settings.writer_interval_seconds,
            "max_seconds": settings.writer_max_seconds,
        },
        "mutation": (
            "pre-write a stale foreign efa-plugin-restart annotation set on one "
            "idle node; unbind one EFA function through the COLLECT-017 host probe "
            "(with its bounded host-side rebind fail-safe) so the deployed control "
            "plane runs the seven-step EFA remediation; keep patching a harmless "
            "tick annotation on the node until the workflow is terminal; remove "
            "every annotation this run wrote. No reset, no reboot, no provider call."
        ),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle, or already carries a "
            "gpu-fault taint, ownership annotation or plugin-restart annotation",
            "target node has no allocatable EFA device or no bound EFA function",
            "any remote command or processor work is open at injection time",
            "the pre-seeded annotation did not land or is not provably stale",
            "the workflow reaches any quarantine/quiesce/reset/reboot/replace step",
            "RESTORE_SCHEDULING fails or the node is left cordoned",
            "provider mutation appears",
            "cleanup leaves the EFA function unbound, an annotation behind, the "
            "writer running or a probe resource behind",
        ],
        "rollback": {
            "efa_rebind_fail_safe_is_armed_by_the_probe_before_the_unbind": True,
            "restore_efa_runs_in_finally": True,
            "annotation_writer_is_time_bounded_and_stopped_in_finally": True,
            "every_annotation_this_run_wrote_is_removed_in_finally": True,
            "node_isolation_is_only_ever_restored_by_the_workflow_itself": True,
            "no_reset_reboot_or_provider_action_is_authorized": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #
@dataclass
class _LiveRun:
    settings: Settings
    regional: RegionalLiveFixture
    case_dir: Path
    preflight: dict[str, Any]
    run_id: str
    collector: CollectorAcceptanceFixture
    writer: AnnotationWriter
    mutation: NodeMutationFixture | None = None
    foreign_incident: str = ""
    foreign_operation: str = ""
    efa_bdf: str = ""
    efa_restore_owed: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    baseline_node: dict[str, Any] = field(default_factory=dict)
    baseline_inventory: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    waiting_evidence: list[dict[str, Any]] = field(default_factory=list)


def _kubectl_prefix(settings: RegionalLiveSettings) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(settings.gpu_kubeconfig),
        "--context",
        settings.gpu_context,
    ]


def _prepare_live_run(settings: Settings, run_dir: Path, attempt: int) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    run_id = f"destr021-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    regional = RegionalLiveFixture(settings.regional)
    run = _LiveRun(
        settings=settings,
        regional=regional,
        case_dir=case_dir,
        preflight=preflight,
        run_id=run_id,
        collector=_collector(settings, run_id),
        writer=AnnotationWriter(
            _kubectl_prefix(settings.regional),
            settings.node,
            interval_seconds=settings.writer_interval_seconds,
            max_seconds=settings.writer_max_seconds,
        ),
    )
    suffix = f"{int(time.time())}-a{attempt}"
    run.foreign_incident = f"acceptance-foreign-{suffix}"
    run.foreign_operation = f"acceptance-foreign-op-{suffix}"
    run.baseline_node = dict(preflight["node"])
    return run


def _preseed(run: _LiveRun) -> None:
    warm = WarmSpareLiveFixture(run.regional, "")
    run.mutation = NodeMutationFixture(
        warm, run.settings.node, annotation_keys=MUTATED_ANNOTATIONS
    )
    annotations = verdicts.preseed_annotations(
        foreign_operation=run.foreign_operation,
        foreign_incident=run.foreign_incident,
        now=datetime.now(timezone.utc),
    )
    run.mutation.apply(NodePatch(labels={}, annotations=dict(annotations)))
    landed = node_document(run.regional, run.settings.node)["annotations"]
    document = {
        "written": annotations,
        "landed": {key: landed.get(key) for key in verdicts.PLUGIN_RESTART_ANNOTATIONS},
    }
    write_json_atomic(run.case_dir / "preseed.json", document)
    errors = verdicts.preseed_errors(landed, foreign_incident=run.foreign_incident)
    if errors:
        raise RegionalFixtureError("; ".join(errors))


def _quiet_control_plane(run: _LiveRun) -> None:
    state = run.regional.store_snapshot(node=run.settings.node, queue_attempts=1)
    write_json_atomic(run.case_dir / "store-before-injection.json", state)
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        raise RegionalFixtureError("remote commands are open; refusing to inject")
    if processor_queue_backlog(state.get("queue") or {}):
        raise RegionalFixtureError("the processor queue is not empty before injection")


def _wait_inventory(
    run: _LiveRun, *, discovered_count: int, timeout_seconds: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    snapshot = run.collector.snapshot()
    while (
        int(snapshot["efa_inventory"]["discovered_count"]) != discovered_count
        and time.monotonic() < deadline
    ):
        time.sleep(3)
        snapshot = run.collector.snapshot()
    return cast(dict[str, Any], snapshot["efa_inventory"])


def _latest_bundle(run: _LiveRun) -> dict[str, Any]:
    value = run.regional.cpu_python(
        base.LATEST_NODE_WORKFLOW,
        run.settings.regional.cluster_id,
        run.settings.node,
        run.started_at.isoformat(),
    )
    matches = value.get("matches") or []
    return cast(dict[str, Any], {**matches[0], "matches": matches}) if matches else {}


def _wait_workflow(run: _LiveRun) -> dict[str, Any]:
    """Poll to the terminal state, keeping every WAITING/FAILED sample seen.

    ``latest_node_workflow`` only returns the terminal document; the 409 race
    this case stages shows up in the samples on the way there, so the loop is
    inlined to record them.
    """

    deadline = time.monotonic() + verdicts.WORKFLOW_TIMEOUT_SECONDS
    bundle: dict[str, Any] = {}
    seen_waiting: dict[tuple[int, str], dict[str, Any]] = {}
    while time.monotonic() < deadline:
        bundle = _latest_bundle(run)
        workflow = bundle.get("workflow") or {}
        for execution in waiting_step_executions(workflow):
            index = execution.get("step_index")
            key = (index if isinstance(index, int) else -1, str(execution["operation"]))
            seen_waiting[key] = execution
        for execution in workflow.get("step_executions") or []:
            run.timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "operation": execution.get("operation"),
                    "status": execution.get("status"),
                }
            )
        run.waiting_evidence = [seen_waiting[key] for key in sorted(seen_waiting)]
        write_json_atomic(
            run.case_dir / "timeline.json",
            {"entries": run.timeline, "waiting_step_executions": run.waiting_evidence},
        )
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            break
        time.sleep(5)
    else:
        raise RegionalFixtureError(f"EFA remediation did not converge: {bundle}")
    incident_deadline = time.monotonic() + c017.EFA_INCIDENT_TIMEOUT_SECONDS
    while (bundle.get("incident") or {}).get(
        "state"
    ) != "RECOVERED" and time.monotonic() < incident_deadline:
        time.sleep(5)
        bundle = _latest_bundle(run)
    request_id = str((bundle.get("workflow") or {}).get("request_id") or "")
    if request_id:
        bundle = {**bundle, **run.regional.cpu_python(c017.REMOTE_COMMANDS, request_id)}
    return bundle


def _inject_and_observe(run: _LiveRun) -> dict[str, Any]:
    run.collector.create()
    baseline = run.collector.snapshot()["efa_inventory"]
    run.baseline_inventory = baseline
    bound = bound_efa_functions(baseline)
    if not bound:
        raise RegionalFixtureError("no bound EFA BDF was discovered")
    run.efa_bdf = bound[0]
    _quiet_control_plane(run)
    run.writer.start()
    run.started_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    run.efa_restore_owed = True
    injection = run.collector.execute(
        "unbind-efa",
        "--run-id",
        run.run_id,
        "--pci-bdf",
        run.efa_bdf,
        "--restore-seconds",
        str(c017.EFA_RESTORE_SECONDS),
    )
    write_json_atomic(run.case_dir / "injection.json", injection)
    unbound = _wait_inventory(
        run,
        discovered_count=int(baseline["discovered_count"]) - 1,
        timeout_seconds=60,
    )
    bundle = _wait_workflow(run)
    run.writer.stop()
    recovered = _wait_inventory(
        run, discovered_count=int(baseline["discovered_count"]), timeout_seconds=30
    )
    document = {
        "efa_bdf": run.efa_bdf,
        "baseline_inventory": baseline,
        "unbound_inventory": unbound,
        "recovered_inventory": recovered,
        **bundle,
    }
    write_json_atomic(run.case_dir / "workflow.json", document)
    return document


def _judge(run: _LiveRun, bundle: dict[str, Any]) -> list[str]:
    final_annotations = node_document(run.regional, run.settings.node)["annotations"]
    errors = verdicts.workflow_errors(bundle)
    errors.extend(
        verdicts.takeover_errors(
            bundle,
            node=run.settings.node,
            foreign_incident=run.foreign_incident,
            final_annotations=final_annotations,
        )
    )
    errors.extend(
        verdicts.expected_count_errors(
            bundle,
            node=run.settings.node,
            baseline_efa=int(run.preflight["efa_allocatable"]),
        )
    )
    errors.extend(verdicts.baseline_errors(bundle, node=run.settings.node))
    errors.extend(verdicts.restore_errors(bundle, timeline_statuses=run.timeline))
    if int(bundle["recovered_inventory"]["discovered_count"]) != int(
        bundle["baseline_inventory"]["discovered_count"]
    ):
        errors.append("the EFA function did not come back bound")
    return errors


def _provider_errors(run: _LiveRun) -> list[str]:
    events = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": events})
    return ["provider mutation appeared during DESTR-021"] if events else []


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are. Every cleanup failure downgrades the verdict to FAIL."""

    run = _prepare_live_run(settings, run_dir, attempt)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "node": settings.node,
        "foreign_incident": run.foreign_incident,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("approved maintenance window has ended")
        _preseed(run)
        bundle = _inject_and_observe(run)
        errors = _judge(run, bundle)
        errors.extend(_provider_errors(run))
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "efa_bdf": run.efa_bdf,
                "workflow_request_id": (bundle.get("workflow") or {}).get("request_id"),
                "incident_id": (bundle.get("incident") or {}).get("incident_id"),
                "conflict_retries_observed": verdicts.conflict_retries_observed(
                    run.waiting_evidence
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = _cleanup(run)
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    guard("writer", lambda: _stop_writer(run))
    if run.efa_restore_owed:
        guard(
            "restore_efa",
            lambda: run.collector.execute(
                "restore-efa", "--run-id", run.run_id, "--pci-bdf", run.efa_bdf
            ),
        )
    if run.mutation is not None:
        guard("annotations_restored", lambda: _restore_annotations(run))
    guard("final_node", lambda: _final_node(run))
    guard("probe_cleanup", lambda: _refuse_residual_map(run.collector.cleanup()))
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage=f"after {CASE_ID} cleanup",
        ),
    )
    return result


def _stop_writer(run: _LiveRun) -> dict[str, Any]:
    run.writer.stop()
    run.writer.clear()
    report = run.writer.report()
    report["achieved_interval_seconds"] = verdicts.writer_achieved_interval(report)
    write_json_atomic(run.case_dir / "writer.json", report)
    errors = verdicts.writer_errors(report)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return report


def _restore_annotations(run: _LiveRun) -> dict[str, Any]:
    if run.mutation is None:
        raise RegionalFixtureError("no annotation baseline to restore")
    run.mutation.restore()
    annotations = node_document(run.regional, run.settings.node)["annotations"]
    left = {key: annotations.get(key) for key in MUTATED_ANNOTATIONS}
    if any(value is not None for value in left.values()):
        raise RegionalFixtureError(f"annotations remain after restore: {left}")
    return left


def _refuse_residual_map(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _final_node(run: _LiveRun) -> dict[str, Any]:
    snapshot = run.regional.node_snapshot(run.settings.node)
    document = node_document(run.regional, run.settings.node)
    write_json_atomic(
        run.case_dir / "node-final.json",
        {"snapshot": snapshot, "efa_allocatable": document["efa_allocatable"]},
    )
    errors = verdicts.node_final_errors(
        run.baseline_node,
        snapshot,
        final_annotations=document["annotations"],
        baseline_efa=int(run.preflight["efa_allocatable"]),
        final_efa=document["efa_allocatable"],
    )
    if run.regional.business_workloads(run.settings.node):
        errors.append("target node acquired a non-system workload")
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return snapshot


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-021 acceptance: the EFA remediation workflow "
            "on one idle node whose metadata carries a stale foreign plugin-restart "
            "annotation and is patched concurrently throughout."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--node", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--writer-interval-seconds",
        type=float,
        default=verdicts.WRITER_INTERVAL_SECONDS,
        help="seconds between two tick-annotation patches (>= 0.2)",
    )
    value.add_argument(
        "--writer-max-seconds",
        type=float,
        default=verdicts.WRITER_MAX_SECONDS,
        help="hard bound on the writer even if the runner dies (60..3600)",
    )
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
