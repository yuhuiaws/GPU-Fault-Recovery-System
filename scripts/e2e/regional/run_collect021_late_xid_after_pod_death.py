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

from scripts.e2e.regional import run_collector_destructive as base  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.kmsg_clock import marker_observed_after  # noqa: E402
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)

CASE_ID = "GF-REGIONAL-COLLECT-021"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-016"
CONFIRMATION = "COLLECT021_EXECUTE"
# A RESTART_APP class XID whose process death precedes the kernel line. XID 13
# (Graphics Exception) is the same class COLLECT-012/016 drive; the race is
# what is new, not the code.
LATE_XID = 13
QUARANTINE_TAINT_PREFIX = "gpu-fault.io"


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    host_probe_image: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
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
    return Settings(
        regional=settings_from_arguments(arguments),
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
    )


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    candidates = [
        item
        for item in regional.gpu_nodes()
        if item["ready"] == "True" and not item["unschedulable"] and not item["taints"]
    ]
    identity = regional.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/completion/test_pod_dies_before_xid_ingest.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=300)
    (case_dir / "focused-tests.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("COLLECT-016 predecessor evidence is not PASS")
    if not settings.site_file.is_file():
        errors.append("regional site file does not exist")
    if len(candidates) < 3:
        errors.append("fewer than three idle Ready GPU nodes")
    if regional.gpu_workloads():
        errors.append("GPU cluster already has active GPU workloads")
    if completed.returncode:
        errors.append("focused regression test failed")
    result = {
        **identity,
        "candidate_nodes": candidates,
        "predecessor": predecessor,
        "focused_tests_passed": completed.returncode == 0,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def managed_fixture(
    settings: Settings,
    *,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> ManagedWorkloadFixture:
    return ManagedWorkloadFixture(
        RegionalLiveFixture(settings.regional),
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=settings.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )


def workload_settings(
    settings: Settings,
    *,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> workload_case.Settings:
    return workload_case.Settings(
        regional=settings.regional,
        site_file=settings.site_file,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
        predecessor_path=settings.predecessor_path,
    )


def node_reads_idle(observation: dict[str, Any], node: str) -> bool:
    """Whether ``resolve`` would see no live container for ``node``.

    ``resolve`` only counts an observation whose phase is PENDING or RUNNING
    and whose container on the node is not terminated. A FAILED/STOPPED phase,
    a terminated container, or no container at all is exactly the death the
    late XID lands after, and the node then resolves as IDLE.
    """

    if observation.get("workload_phase") not in {"PENDING", "RUNNING"}:
        return True
    return not any(
        container.get("node_id") == node and not container.get("terminated")
        for container in observation.get("containers") or []
    )


def target_bdf(
    observation: dict[str, Any],
    inventory: list[dict[str, Any]],
    *,
    pod_uid: str,
) -> str:
    """The BDF of the killed Pod's GPU, or the node's first GPU as a fallback."""

    uuids = {
        str(gpu)
        for container in observation.get("containers") or []
        if container.get("pod_uid") == pod_uid
        for gpu in container.get("gpu_uuids") or []
    }
    for item in inventory:
        if str(item.get("uuid")) in uuids:
            return str(item["pci_bdf"])
    return str(inventory[0]["pci_bdf"])


def node_untouched_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    """MONITOR_ONLY plans nothing: the node is schedulable, untainted, un-owned."""

    errors: list[str] = []
    if after.get("unschedulable"):
        errors.append(f"{label}: node became unschedulable")
    ours = [
        item
        for item in after.get("taints") or []
        if str(item.get("key") or "").startswith(QUARANTINE_TAINT_PREFIX)
    ]
    if ours:
        errors.append(f"{label}: node carries a gpu-fault.io taint: {ours}")
    if after.get("ownership_annotations"):
        errors.append(
            f"{label}: node carries ownership annotations: "
            f"{after.get('ownership_annotations')}"
        )
    if before.get("boot_id") != after.get("boot_id"):
        errors.append(f"{label}: node boot_id changed (a reboot happened)")
    return errors


def proactive_errors(
    state: dict[str, Any],
    node_before: dict[str, Any],
    node_after: dict[str, Any],
    *,
    xid: int,
) -> list[str]:
    """The proactive judgement: IDLE -> MONITOR_ONLY / NO_ACTION, no workflow.

    This is the whole point of 55272a0: before it, this same sequence compiled
    STOP -> RESTART with no workload id, fell to the fail-closed path and
    quarantined the node.
    """

    errors: list[str] = []
    event = state.get("event")
    decision = state.get("decision")
    incident = state.get("incident")
    if event is None:
        errors.append("no XID event for the late marker")
    else:
        if event.get("xid") != xid:
            errors.append(f"event xid={event.get('xid')!r}, expected {xid}")
        if event.get("workload_state") != "IDLE":
            errors.append(
                f"event workload_state={event.get('workload_state')!r}, expected IDLE"
            )
        if event.get("affected_workload_ids"):
            errors.append(
                "event names affected_workload_ids on an idle node: "
                f"{event.get('affected_workload_ids')}"
            )
    if decision is None:
        errors.append("no policy decision for the late XID")
    else:
        if decision.get("disposition") != "MONITOR_ONLY":
            errors.append(
                f"decision disposition={decision.get('disposition')!r}, "
                "expected MONITOR_ONLY"
            )
        if decision.get("action") != "NO_ACTION":
            errors.append(
                f"decision action={decision.get('action')!r}, expected NO_ACTION"
            )
        if decision.get("official_action") != "RESTART_APP":
            errors.append(
                f"decision official_action={decision.get('official_action')!r}, "
                "expected RESTART_APP"
            )
        if not any(
            "no managed application to restart" in str(item)
            for item in decision.get("reasons") or []
        ):
            errors.append(
                f"decision reasons do not name the idle-node rule: "
                f"{decision.get('reasons')}"
            )
        if decision.get("workflow_request_id") is not None:
            errors.append("decision points at a workflow on an idle node")
    if incident is None:
        errors.append("no incident for the late XID")
    else:
        if incident.get("state") != "RECOVERED":
            errors.append(
                f"incident state={incident.get('state')!r}, expected RECOVERED"
            )
        if incident.get("workflow_request_id") is not None:
            errors.append("incident opened a workflow on an idle node")
    if state.get("workflow") is not None:
        errors.append("a workflow exists for the MONITOR_ONLY decision")
    if state.get("commands"):
        errors.append("a remote command was dispatched for the MONITOR_ONLY decision")
    errors.extend(node_untouched_errors(node_before, node_after, label="proactive"))
    return errors


def passive_errors(
    *,
    kill: dict[str, Any],
    death: dict[str, Any],
    restarted: dict[str, Any],
    restart_state: dict[str, Any],
    source_uids: set[str],
    pod_uid: str,
    attempt_id: str,
) -> list[str]:
    """The passive judgement: the attempt's own failure still restarts the job.

    The XID incident is MONITOR_ONLY and inactive, so the terminal event goes
    to quick triage and its PASS restarts the attempt -- the restart the old
    quarantine made impossible.
    """

    errors: list[str] = []
    if not kill.get("killed"):
        errors.append("kill-workload reported no killed processes")
    phase = death.get("workload_phase")
    if phase not in {"FAILED", "STOPPED"}:
        errors.append(f"death observation phase={phase!r}, expected FAILED/STOPPED")
    for container in death.get("containers") or []:
        if container.get("pod_uid") != pod_uid:
            continue
        if not container.get("terminated"):
            errors.append("killed container is not terminated in the observation")
        elif container.get("exit_code") in (None, 0):
            errors.append(
                f"killed container exit_code={container.get('exit_code')!r}, "
                "expected non-zero"
            )
    pods = restarted.get("pods") or []
    if len(pods) != 3:
        errors.append(f"restart did not bring up three Running pods: {len(pods)}")
    if {str(item.get("uid")) for item in pods} & source_uids:
        errors.append("a source pod uid survived the restart")
    if any(item.get("attempt_id") == attempt_id for item in pods):
        errors.append("restarted pods kept the failed attempt id")
    budget = restart_state.get("restart_budget") or {}
    if budget.get("restart_count") != 1:
        errors.append(
            f"restart budget restart_count={budget.get('restart_count')!r}, expected 1"
        )
    if not any(
        item.get("attempt_id") != attempt_id
        and item.get("workload_phase") in {"RUNNING", "PENDING"}
        for item in restart_state.get("observations") or []
    ):
        errors.append("no new attempt observation appeared after the restart")
    return errors


def wait_death_observation(
    regional: RegionalLiveFixture,
    *,
    node: str,
    job_id: str,
    attempt_id: str,
    pod_uid: str,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    """Poll until the store shows the killed container terminated / phase FAILED."""

    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while True:
        state = regional.store_snapshot(
            node=node,
            job_id=job_id,
            attempt_id=attempt_id,
            queue_attempts=1,
        )
        for observation in state.get("observations") or []:
            if node_reads_idle(observation, node):
                return observation
            last = observation
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_seconds)
    raise RegionalFixtureError(
        f"the killed attempt observation never went FAILED: {last}"
    )


def wait_for_decision(
    regional: RegionalLiveFixture,
    *,
    node: str,
    marker: str,
    observed_after: datetime | None,
    case_dir: Path,
    timeout_seconds: int,
    job_id: str = "",
    attempt_id: str = "",
    poll_seconds: int = 5,
) -> dict[str, Any]:
    """Poll until the policy records a decision for the marker.

    ``wait_for_workflow`` cannot terminate here: MONITOR_ONLY produces no
    workflow, so its loop would spin until timeout. The decision (and the
    RECOVERED incident behind it) is the terminal signal instead.
    """

    deadline = time.monotonic() + timeout_seconds
    timeline: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    while True:
        last = regional.store_snapshot(
            node=node,
            marker=marker,
            observed_after=observed_after,
            job_id=job_id,
            attempt_id=attempt_id,
            queue_attempts=1,
        )
        decision = last.get("decision")
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "has_event": bool(last.get("event")),
                "disposition": (decision or {}).get("disposition"),
                "incident_state": (last.get("incident") or {}).get("state"),
                "has_workflow": last.get("workflow") is not None,
            }
        )
        write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
        if decision:
            return last
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_seconds)
    raise RegionalFixtureError(f"policy recorded no decision for the late XID: {last}")


def run_late_xid_section(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
    *,
    workloads: list[ManagedWorkloadFixture],
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture],
) -> dict[str, Any]:
    """Submit, kill the process before any XID, then judge both sides.

    Resources land in ``workloads``/``fixtures`` the moment they exist, so a
    timeout inside a wait never leaks a 24-GPU PyTorchJob or a privileged probe
    Pod (the COLLECT-016 ownership lesson).
    """

    job_id = f"c021-{suffix}"
    attempt_id = f"{job_id}-a001"
    manifest = base.render_named_training_manifest(
        case_dir / "workload.yaml",
        name=f"gpu-fault-c021-{suffix}",
    )
    workload = managed_fixture(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workloads.append(workload)
    workload.submit()
    source = workload.wait_running(timeout_seconds=900)
    source_uids = {str(item["uid"]) for item in source["pods"]}
    target_node = str(source["pods"][0]["node"])
    target_uid = str(source["pods"][0]["uid"])
    node_before = regional.node_snapshot(target_node)
    workload_case.wait_observation(
        regional,
        workload_settings(
            settings, manifest=manifest, job_id=job_id, attempt_id=attempt_id
        ),
        node=target_node,
        expected_gpu_count=24,
    )
    collector = CollectorAcceptanceFixture(
        regional,
        node=target_node,
        image=settings.host_probe_image,
        case_id=CASE_ID,
        run_id=f"c021-{suffix}",
    )
    fixtures.append(collector)
    collector.create()
    inventory = collector.snapshot()["gpu_inventory"]

    # Kill the process before any XID exists, so the container exits non-zero
    # and the Pod object survives (a deleted Pod reads as a user stop).
    kill = collector.execute("kill-workload", "--pod-uid", target_uid, timeout=120)
    death = wait_death_observation(
        regional,
        node=target_node,
        job_id=job_id,
        attempt_id=attempt_id,
        pod_uid=target_uid,
    )

    marker = f"c021-{int(time.time())}"
    injected_at = datetime.now(timezone.utc)
    bdf = target_bdf(death, inventory, pod_uid=target_uid)
    collector.execute(
        "write-xid",
        "--xid",
        str(LATE_XID),
        "--marker",
        marker,
        "--pci-bdf",
        bdf,
        "--message",
        "Graphics Exception after process death",
    )
    state = wait_for_decision(
        regional,
        node=target_node,
        marker=marker,
        observed_after=marker_observed_after(marker, injected_at),
        case_dir=case_dir / "proactive",
        timeout_seconds=600,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    node_after = regional.node_snapshot(target_node)
    errors = proactive_errors(state, node_before, node_after, xid=LATE_XID)

    restarted = workload.wait_restarted(source_uids, timeout_seconds=900)
    restart_state = regional.store_snapshot(
        node=target_node,
        job_id=job_id,
        queue_attempts=1,
    )
    errors.extend(
        passive_errors(
            kill=kill,
            death=death,
            restarted=restarted,
            restart_state=restart_state,
            source_uids=source_uids,
            pod_uid=target_uid,
            attempt_id=attempt_id,
        )
    )
    return {
        "errors": errors,
        "marker": marker,
        "job_id": job_id,
        "target_node": target_node,
        "killed_pod_uid": target_uid,
        "kill": kill,
        "source_pod_uids": sorted(source_uids),
        "restarted_pod_uids": sorted(str(item["uid"]) for item in restarted["pods"]),
        "event": state.get("event"),
        "decision": state.get("decision"),
        "incident": state.get("incident"),
        "node_before": node_before,
        "node_after": node_after,
        "restart_budget": restart_state.get("restart_budget"),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-workload-restart",
        "predecessor": preflight["predecessor"],
        "mutation": (
            "submit a managed 24-GPU workload, SIGKILL its training process "
            "before any XID, then inject a late RESTART_APP kmsg XID"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "candidate_node_uids": sorted(
                str(item["uid"]) for item in preflight["candidate_nodes"]
            ),
        },
        "stop_conditions": [
            "COLLECT-016 predecessor evidence is not PASS",
            "fewer than three idle GPU nodes or image prewarm fails",
            "the kill did not make the container exit non-zero",
            "the late XID quarantined the node or opened a workflow",
            "the attempt was not restarted by the passive path",
            "workload/probe cleanup leaves residuals",
        ],
        "rollback": {
            "delete_workload_and_restarted_copy": True,
            "delete_prewarm_pods": True,
            "delete_probe_pod": True,
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
    regional = RegionalLiveFixture(settings.regional)
    candidates = [str(item["name"]) for item in preflight["candidate_nodes"]]
    prewarm = ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=f"c021-{attempt}")
    workloads: list[ManagedWorkloadFixture] = []
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture] = []
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "errors": [],
        **regional.evidence_identity(),
    }
    try:
        prewarm.create(candidates)
        suffix = f"{int(time.time())}-{attempt}"
        section = run_late_xid_section(
            settings,
            regional,
            case_dir,
            suffix,
            workloads=workloads,
            fixtures=fixtures,
        )
        result["errors"].extend(section["errors"])
        result["late_xid"] = section
        result["verdict"] = "PASS" if not result["errors"] else "FAIL"
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
        for fixture in fixtures:
            try:
                residuals = fixture.cleanup()
                if isinstance(residuals, dict) and any(residuals.values()):
                    result["errors"].append("probe resources remain")
            except Exception as exc:
                result["errors"].append(
                    f"probe cleanup failed: {type(exc).__name__}: {exc}"
                )
        try:
            residuals = prewarm.cleanup()
            if any(residuals.values()):
                result["errors"].append("prewarm Pods remain")
        except Exception as exc:
            result["errors"].append(
                f"prewarm cleanup failed: {type(exc).__name__}: {exc}"
            )
        if result["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run COLLECT-021 late RESTART_APP XID after Pod death acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
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
