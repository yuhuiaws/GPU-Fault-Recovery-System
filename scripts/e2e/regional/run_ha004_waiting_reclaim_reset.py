#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import run_destr001_gpu_reset as reset_case  # noqa: E402
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
    install_site_profile,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    WaitingEvidence,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

CASE_ID = "GF-REGIONAL-HA-004"
PREDECESSOR_CASE_ID = "GF-REGIONAL-HA-003"
CONFIRMATION = "HA004_FORCE_WAITING_COMMAND_RECLAIM"
DEPLOYMENT = "gpu-fault-cluster-executor"
LEASE_ENV = "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS"
POLL_ENV = "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS"
TEST_LEASE_SECONDS = 10
TEST_POLL_SECONDS = 2
WATCHDOG_SECONDS = 900


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
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
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/node_agent/test_diagnostics.py::"
        "test_same_command_is_serialized_while_in_flight",
        "tests/node_agent/test_protocol.py::"
        "test_ledger_save_failure_releases_all_inflight_waiters",
        "tests/node_agent/test_remediation.py::"
        "test_gpu_reset_is_idempotent_and_rechecks_clients",
        "tests/hyperpod/test_cluster_executor.py::"
        "test_cluster_executor_renews_remote_command_lease",
        "tests/regional/test_regional_control_plane.py::"
        "test_remote_command_lease_and_result_advance_adapter",
    ]
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


def deployment_snapshot(regional: RegionalLiveFixture) -> dict[str, Any]:
    value = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "deployment",
            DEPLOYMENT,
            "-o",
            "json",
        )
    )
    container = value["spec"]["template"]["spec"]["containers"][0]
    env = {}
    for item in container.get("env", []):
        if item.get("name") in {LEASE_ENV, POLL_ENV}:
            if "valueFrom" in item:
                raise RegionalFixtureError(
                    f"{item['name']} is valueFrom and cannot be safely restored"
                )
            env[str(item["name"])] = item.get("value")
    return {
        "uid": value["metadata"]["uid"],
        "generation": value["metadata"].get("generation"),
        "replicas": int(value["spec"].get("replicas", 0)),
        "ready_replicas": int(value.get("status", {}).get("readyReplicas", 0)),
        "env": {
            LEASE_ENV: env.get(LEASE_ENV),
            POLL_ENV: env.get(POLL_ENV),
        },
    }


class ExecutorTimingFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        case_dir: Path,
    ) -> None:
        self.regional = regional
        self.case_dir = case_dir
        self.baseline = deployment_snapshot(regional)
        self.watchdog: subprocess.Popen[str] | None = None

    def _env_arguments(self, values: dict[str, str | None]) -> list[str]:
        return [
            f"{name}={value}" if value is not None else f"{name}-"
            for name, value in values.items()
        ]

    def _kubectl_base(self) -> list[str]:
        settings = self.regional.settings
        return [
            "kubectl",
            "--kubeconfig",
            str(settings.gpu_kubeconfig),
            "--context",
            settings.gpu_context,
            "-n",
            settings.namespace,
        ]

    def start_watchdog(self) -> None:
        log_path = self.case_dir / "executor-rollback-watchdog.log"
        command = [
            *self._kubectl_base(),
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            *self._env_arguments(self.baseline["env"]),
        ]
        scale = [
            *self._kubectl_base(),
            "scale",
            f"deployment/{DEPLOYMENT}",
            f"--replicas={self.baseline['replicas']}",
        ]
        script = (
            f"sleep {WATCHDOG_SECONDS}; "
            + shlex.join(command)
            + "; "
            + shlex.join(scale)
        )
        stream = log_path.open("w", encoding="utf-8")
        self.watchdog = subprocess.Popen(
            ["bash", "-ceu", script],
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        (self.case_dir / "executor-rollback-watchdog.pid").write_text(
            str(self.watchdog.pid),
            encoding="utf-8",
        )

    def apply(self) -> dict[str, Any]:
        self.start_watchdog()
        self.regional.kubectl(
            "gpu",
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            f"{LEASE_ENV}={TEST_LEASE_SECONDS}",
            f"{POLL_ENV}={TEST_POLL_SECONDS}",
        )
        self.regional.kubectl(
            "gpu",
            "scale",
            f"deployment/{DEPLOYMENT}",
            "--replicas=2",
        )
        self.regional.kubectl(
            "gpu",
            "rollout",
            "status",
            f"deployment/{DEPLOYMENT}",
            "--timeout=600s",
            timeout=630,
        )
        return deployment_snapshot(self.regional)

    def restore(self) -> dict[str, Any]:
        self.regional.kubectl(
            "gpu",
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            *self._env_arguments(self.baseline["env"]),
        )
        self.regional.kubectl(
            "gpu",
            "scale",
            f"deployment/{DEPLOYMENT}",
            f"--replicas={self.baseline['replicas']}",
        )
        self.regional.kubectl(
            "gpu",
            "rollout",
            "status",
            f"deployment/{DEPLOYMENT}",
            "--timeout=600s",
            timeout=630,
        )
        if self.watchdog is not None and self.watchdog.poll() is None:
            self.watchdog.terminate()
            self.watchdog.wait(timeout=10)
        return deployment_snapshot(self.regional)


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(node=settings.node)
    workloads = regional.business_workloads(settings.node)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    deployment = deployment_snapshot(regional)
    tests = focused_tests(case_dir)
    errors = []
    if not predecessor["valid"]:
        errors.append("HA-003 predecessor evidence is not PASS")
    if node["ready"] != "True" or not node["unschedulable"]:
        errors.append("target node is not Ready and pre-cordoned")
    if node["ownership_annotations"]:
        errors.append("target node has pre-existing gpu-fault ownership")
    if workloads:
        errors.append("target node has non-system running Pods")
    if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    reset = reset_case.capability(state.get("profile"), "gpuReset")
    if reset is None or (
        reset.get("mode") != "OWN"
        or reset.get("owner") != "gpu-fault-node-agent"
        or reset.get("adapter") != "node-action"
    ):
        errors.append("gpuReset is not OWN by the Node Agent")
    if deployment["replicas"] != 2 or deployment["ready_replicas"] != 2:
        errors.append("cluster executor does not have two Ready replicas")
    if int((state.get("queue") or {}).get("depth") or 0):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "store": state,
        "business_workloads": workloads,
        "deployment": deployment,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def command_timeline(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    marker: str,
    observed_after: datetime,
    timeout_seconds: int,
    kill_owner: Callable[[str], None],
    evidence: WaitingEvidence | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    first_owner = ""
    killed = False
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = regional.store_snapshot(
            node=settings.node,
            marker=marker,
            observed_after=observed_after,
            queue_attempts=1,
        )
        if evidence is not None:
            evidence.observe(state)
        commands = [
            item
            for item in state.get("commands") or []
            if item.get("step", {}).get("operation") == "RESET_GPU"
        ]
        command = commands[0] if len(commands) == 1 else {}
        sample = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "command_id": command.get("command_id"),
            "status": command.get("status"),
            "lease_owner": command.get("lease_owner"),
            "last_lease_owner": command.get("last_lease_owner"),
            "lease_token": command.get("lease_token"),
            "lease_expires_at": command.get("lease_expires_at"),
        }
        timeline.append(sample)
        # The reset is in flight from the first claim until the Node Agent's
        # result lands, but the remote command is LEASED only for each claim
        # round trip: the executor hands the action to the Node Agent, reports
        # WAITING, releases the lease and re-claims on its next cycle. A
        # sampler that needs seconds per store read therefore sees WAITING,
        # with the dispatching replica in `last_lease_owner`. That replica is
        # the one to remove: with it gone, only the other replica can claim
        # the same command ID, which is exactly the reclaim this case must
        # observe (2026-09-05: a 45s reset never showed LEASED to the sampler).
        if command.get("status") in {"LEASED", "WAITING"} and not killed:
            first_owner = str(
                command.get("lease_owner") or command.get("last_lease_owner") or ""
            )
            if not first_owner:
                raise RegionalFixtureError("in-flight command has no owner")
            kill_owner(first_owner)
            killed = True
        if killed and command.get("status") in {"SUCCEEDED", "FAILED"}:
            last = state
            break
        time.sleep(0.25)
    else:
        raise RegionalFixtureError("RESET_GPU command did not reach a terminal state")
    owners = {
        str(item.get("lease_owner") or item.get("last_lease_owner") or "")
        for item in timeline
        if item.get("lease_owner") or item.get("last_lease_owner")
    }
    last["ha004_first_owner"] = first_owner
    last["ha004_owners"] = sorted(owners)
    return last, timeline


def owner_pod(
    regional: RegionalLiveFixture,
    owner: str,
) -> dict[str, Any]:
    pods = regional.ready_pods("gpu", "gpu-fault-cluster-executor")
    match = next(
        (item for item in pods if str(item["name"]) in owner),
        None,
    )
    if match is None:
        raise RegionalFixtureError(f"cannot map executor owner to a Pod: {owner}")
    return cast(dict[str, Any], match)


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    return {
        "risk": "destructive",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": (
            "temporarily roll the executor Deployment to lease=10s/poll=2s, "
            "execute one real RESET_GPU, and force-delete the first lease owner"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "agent_generation": (state.get("agent") or {}).get("generation"),
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
            "deployment_uid": preflight["deployment"]["uid"],
            "deployment_env": preflight["deployment"]["env"],
        },
        "stop_conditions": [
            "HA-003 predecessor evidence is not PASS",
            "target node or executor Deployment baseline drifts",
            "rollback watchdog cannot be armed",
            "RESET_GPU command is not observed in flight (LEASED or WAITING) "
            "with an identifiable executor owner",
            "a second executor never reclaims the same command ID",
            "Node Agent ledger/journal show more than one physical reset",
            "Deployment, node ownership or services cannot be restored",
        ],
        "rollback": {
            "detached_executor_config_watchdog_seconds": WATCHDOG_SECONDS,
            "restore_original_lease_poll_and_replica_values": True,
            "restore_GPU_services_and_node_ownership": True,
            "delete_host_probe_resources": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "agent_generation": (preflight["store"].get("agent") or {}).get("generation"),
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
        "deployment_uid": preflight["deployment"]["uid"],
        "deployment_env": preflight["deployment"]["env"],
    }
    if current != planned:
        raise RegionalFixtureError(f"HA-004 plan drifted: {planned} != {current}")


def lease_reissue_observed(
    timeline: list[dict[str, Any]],
    *,
    first_owner: str,
) -> bool:
    """Whether the timeline proves a second lease was granted after the kill.

    A lease token is readable only while the command is LEASED; the store
    clears it the moment the executor reports WAITING or a terminal result. A
    sampler that sees the command through WAITING therefore usually sees no
    token at all, although every claim mints a fresh one. Two distinct tokens
    remain the direct proof. Failing that, the same command carrying an
    executor identity other than the one that was removed is the same fact
    seen from the owner side: a claim never keeps the previous token.
    """

    tokens = {
        str(item.get("lease_token")) for item in timeline if item.get("lease_token")
    }
    if len(tokens) >= 2:
        return True
    owners = {
        str(item.get("lease_owner") or item.get("last_lease_owner") or "")
        for item in timeline
        if item.get("lease_owner") or item.get("last_lease_owner")
    }
    return bool(first_owner) and any(owner != first_owner for owner in owners)


def evaluate_reclaim(
    *,
    settings: Settings,
    host: HostProbeFixture,
    state: dict[str, Any],
    timeline: list[dict[str, Any]],
    baseline_host: dict[str, Any],
    target_bdf: str,
    run_id: str,
    injected_at: datetime,
    case_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    errors = reset_case.workflow_errors(state)
    commands = [
        item
        for item in state.get("commands") or []
        if item.get("step", {}).get("operation") == "RESET_GPU"
    ]
    if len(commands) != 1:
        errors.append("RESET_GPU remote command is not unique")
        command: dict[str, Any] = {}
    else:
        command = commands[0]
    command_ids = {
        str(item.get("command_id")) for item in timeline if item.get("command_id")
    }
    owners = {
        str(item.get("lease_owner") or item.get("last_lease_owner") or "")
        for item in timeline
        if item.get("lease_owner") or item.get("last_lease_owner")
    }
    tokens = {
        str(item.get("lease_token")) for item in timeline if item.get("lease_token")
    }
    if len(command_ids) != 1:
        errors.append("remote command ID changed across reclaim")
    if len(owners) < 2:
        errors.append("a second executor did not reclaim the command")
    if not lease_reissue_observed(
        timeline,
        first_owner=str(state.get("ha004_first_owner") or ""),
    ):
        errors.append("no second lease was observed after the first owner was removed")
    after = host.execute(
        "snapshot",
        "--since-epoch",
        str(injected_at.timestamp()),
        "--pci-bdf",
        target_bdf,
        "--run-id",
        run_id,
        timeout=180,
    )
    write_json_atomic(case_dir / "host-after.json", after)
    errors.extend(
        reset_case.host_errors(
            baseline_host,
            after,
            expected_gpu_count=len(baseline_host["gpu_inventory"]),
            target_bdf=target_bdf,
        )
    )
    expected_node_command = ""
    if command:
        generation = int((state.get("agent") or {}).get("generation") or 0)
        expected_node_command = (
            f"{command.get('idempotency_key')}/{settings.node}/agent-{generation}"
        )
        added = [
            item
            for item in after["ledger"]
            if item.get("command_id") == expected_node_command
        ]
        if len(added) != 1:
            errors.append("deterministic node command ledger entry is not unique")
    return errors, {
        "command": command,
        "command_ids": sorted(command_ids),
        "owners": sorted(owners),
        "token_count": len(tokens),
        "expected_node_command": expected_node_command,
        "host_after": after,
    }


def cleanup_case(
    *,
    settings: Settings,
    regional: RegionalLiveFixture,
    timing: ExecutorTimingFixture,
    host: HostProbeFixture,
    preflight: dict[str, Any],
    incident_id: str,
    run_id: str,
    sampler_started: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    if incident_id:
        try:
            result["quiesce_recovery"] = host.execute(
                "restore-quiesce",
                "--incident-id",
                incident_id,
                timeout=300,
            )
        except Exception as exc:
            result["errors"].append(f"quiesce recovery: {type(exc).__name__}: {exc}")
    if sampler_started:
        try:
            result["sampler_final"] = host.execute(
                "stop-reset-sampler",
                "--run-id",
                run_id,
                timeout=60,
            )
        except Exception as exc:
            result["errors"].append(f"sampler cleanup: {type(exc).__name__}: {exc}")
    try:
        residuals = host.cleanup()
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["errors"].append("host probe resources remain")
    except Exception as exc:
        result["errors"].append(f"probe cleanup: {type(exc).__name__}: {exc}")
    try:
        restored = timing.restore()
        result["executor_restored"] = restored
        if (
            restored["replicas"] != timing.baseline["replicas"]
            or restored["env"] != timing.baseline["env"]
        ):
            result["errors"].append("executor Deployment did not return to baseline")
    except Exception as exc:
        result["errors"].append(f"executor restore: {type(exc).__name__}: {exc}")
    if incident_id:
        try:
            node = regional.node_snapshot(settings.node)
            if node["ownership_annotations"]:
                restore = WarmSpareLiveFixture(regional, "")
                created = restore.create_restore_workflow(
                    incident_id=incident_id,
                    node=settings.node,
                    profile_version=str(
                        (preflight["store"].get("profile") or {}).get("profile_version")
                        or ""
                    ),
                    reason="HA-004 validated cleanup",
                )
                result["node_restore"] = restore.wait_workflow_id(
                    str(created["workflow_request_id"])
                )
                if result["node_restore"].get("status") != "SUCCEEDED":
                    result["errors"].append("node restore workflow failed")
        except Exception as exc:
            result["errors"].append(f"node restore: {type(exc).__name__}: {exc}")
    return result


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
    verify_plan_identity(case_dir, preflight)
    regional = RegionalLiveFixture(settings.regional)
    timing = ExecutorTimingFixture(regional, case_dir)
    run_id = f"ha004-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"ha004-{int(time.time())}-a{attempt}"
    host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=reset_case.PROBE_SCRIPT,
            active_deadline_seconds=3600,
        )
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    incident_id = ""
    sampler_started = False
    killed_pod: dict[str, Any] | None = None
    try:
        write_json_atomic(case_dir / "executor-baseline.json", timing.baseline)
        applied = timing.apply()
        write_json_atomic(case_dir / "executor-test-config.json", applied)
        host.create()
        baseline_host = host.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if baseline_host["compute_clients"]:
            raise RegionalFixtureError("target node has active NVIDIA clients")
        target_bdf = str(baseline_host["gpu_inventory"][0]["pci_bdf"])
        host.execute(
            "start-reset-sampler",
            "--run-id",
            run_id,
            "--probe-script",
            host.host_script,
            timeout=60,
        )
        sampler_started = True
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before injection")
        injected_at = datetime.now(timezone.utc)
        injection = host.execute(
            "write-xid46",
            "--marker",
            marker,
            "--drill-id",
            run_id,
            "--pci-bdf",
            target_bdf,
        )
        write_json_atomic(case_dir / "injection.json", injection)

        def kill(owner: str) -> None:
            nonlocal killed_pod
            killed_pod = owner_pod(regional, owner)
            regional.kubectl(
                "gpu",
                "delete",
                "pod",
                str(killed_pod["name"]),
                "--grace-period=0",
                "--force",
                timeout=180,
            )

        # The command timeline samples the store through the whole reset, so
        # it also has to keep the WAITING records of the steps that finish
        # before `wait_for_workflow` starts polling.
        waiting_evidence = WaitingEvidence()
        claimed_state, timeline = command_timeline(
            regional,
            settings,
            marker=marker,
            observed_after=injected_at,
            timeout_seconds=900,
            kill_owner=kill,
            evidence=waiting_evidence,
        )
        write_json_atomic(case_dir / "command-timeline.json", {"entries": timeline})
        state = waiting_evidence.merged_into(
            regional.wait_for_workflow(
                node=settings.node,
                marker=marker,
                observed_after=injected_at,
                case_dir=case_dir,
                timeout_seconds=1200,
            )
        )
        # The workflow wait re-reads the store, so the facts only the command
        # timeline knew -- which replica was removed, which ones held the
        # command -- have to be carried across explicitly.
        state["ha004_first_owner"] = claimed_state.get("ha004_first_owner")
        state["ha004_owners"] = claimed_state.get("ha004_owners")
        write_json_atomic(case_dir / "workflow-state.json", state)
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        errors, evidence = evaluate_reclaim(
            settings=settings,
            host=host,
            state=state,
            timeline=timeline,
            baseline_host=baseline_host,
            target_bdf=target_bdf,
            run_id=run_id,
            injected_at=injected_at,
            case_dir=case_dir,
        )
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "incident_id": incident_id,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "remote_command_ids": evidence["command_ids"],
                "lease_owners": evidence["owners"],
                "lease_token_count": evidence["token_count"],
                "node_command_id": evidence["expected_node_command"],
                "killed_owner_pod": killed_pod,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = cleanup_case(
            settings=settings,
            regional=regional,
            timing=timing,
            host=host,
            preflight=preflight,
            incident_id=incident_id,
            run_id=run_id,
            sampler_started=sampler_started,
        )
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run HA-004 WAITING command reclaim during one GPU reset."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
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
    return execute_case(
        settings,
        arguments.run_dir,
        arguments.attempt,
        deadline,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
