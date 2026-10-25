"""GF-GENERAL-LIVE-028 ``api_unreachable`` end to end over fakes.

A fake job whose Pods vanish once the runner's trigger file is created, a fake
watcher Pod inventory, a probe whose block/cleanup flip the phase, and a store
that answers the attempt records and the watcher's coverage heartbeat per phase:
the whole sequence -- block, trigger, emergency stop, frozen heartbeat, restore,
single-action grading -- executes, and the INCONCLUSIVE and FAIL shapes with it.
"""

from __future__ import annotations

import base64
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_network_isolation as iso
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import APPROVALS
from scripts.e2e.regional.general_live_store import ATTEMPT_PROBE
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import FakeProbe
from tests.regional._general_live_isolation_fakes import (
    NODE,
    POD_IP,
    T0,
    WATCHER_NODE,
    Regional,
    attempt_entry,
    attempt_records,
    heartbeat,
    isolation_context,
    probe_docs,
    watcher_items,
)


class Phases:
    """What the fake cluster and store answer in each phase of the variant."""

    def __init__(self) -> None:
        self.phase = "before"
        self.triggered = False
        self.heartbeat_during = T0
        self.stops = 1
        self.decision = "PLAN_CREATED"
        self.emergency = True
        self.dup_notifications = False
        self.notifications = [
            {
                "incident_id": "inc-c",
                "category": "containment",
                "deduplication_key": "k1",
                "created_at": T0.isoformat(),
                "status": "SENT",
            }
        ]
        self.exec_calls: list[tuple[str, ...]] = []


class FakeFailJob:
    instances: list["FakeFailJob"] = []
    phases: Phases

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        mode: Any,
        restart_budget: int = 1,
        gpu_count: int = 8,
        ranks: int = 1,
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.mode = mode
        self.restart_budget = restart_budget
        job = self

        class _Fixture:
            def pods(self) -> list[dict[str, Any]]:
                if job.phases.triggered:
                    return []
                return [
                    {
                        "name": f"{job.name}-master-0",
                        "uid": "pod-1",
                        "node": NODE,
                        "phase": "Running",
                        "ready": True,
                        "attempt_id": job.attempt_id,
                    }
                ]

            def pods_healthy(self, pods: list[dict[str, Any]]) -> bool:
                return len(pods) == 1 and pods[0]["ready"]

            def workload(self) -> dict[str, Any]:
                annotations = {}
                if job.phases.triggered and job.phases.emergency:
                    annotations = {
                        iso.EMERGENCY_MODE_ANNOTATION: iso.EMERGENCY_MODE,
                        iso.TERMINATION_ANNOTATION: "inc-c",
                    }
                return {
                    "kind": "PyTorchJob",
                    "metadata": {
                        "name": job.name,
                        "uid": "wl-1",
                        "annotations": annotations,
                    },
                }

            def delete(self) -> None:
                pass

        self.fixture = _Fixture()
        self.prewarm = type("P", (), {"cleanup": staticmethod(lambda: None)})()
        FakeFailJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def delete(self) -> None:
        pass


def _attempt_entry(phases: Phases) -> dict[str, Any]:
    return attempt_entry(stops=phases.stops, decision=phases.decision)


def _store(phases: Phases, job_name: str) -> Any:
    def respond(script: str, *arguments: str) -> dict[str, Any]:
        if script == ATTEMPT_PROBE:
            if phases.phase != "restored":
                return {
                    "attempts": {
                        f"{job_name}-a001": {
                            "event": None,
                            "decision": None,
                            "containment_incident": None,
                            "recovery": [],
                            "containment_workflows": [],
                        }
                    },
                    "restart_budget": None,
                }
            entry = _attempt_entry(phases)
            return {**attempt_records(entry), "attempts": {f"{job_name}-a001": entry}}
        if script == iso.ISOLATION_RECORDS_PROBE:
            at = (
                T0 + timedelta(days=1)
                if phases.phase == "restored"
                else phases.heartbeat_during
            )
            notifications = phases.notifications * (
                2 if phases.dup_notifications else 1
            )
            return {
                "heartbeat": heartbeat(at),
                "observations": [
                    {
                        "attempt_id": f"{job_name}-a001",
                        "job_id": job_name,
                        "workload_phase": "RUNNING",
                        "observed_at": T0.isoformat(),
                        "containers": 1,
                    }
                ],
                "notifications": notifications if arguments[3] else [],
                "agent": None,
            }
        raise AssertionError("unexpected store script")

    return respond


def _variant_context(
    tmp_path: Path,
    phases: Phases,
    monkeypatch: pytest.MonkeyPatch,
    *,
    inventory: dict[str, Any] | None = None,
):
    FakeFailJob.instances = []
    FakeFailJob.phases = phases
    monkeypatch.setattr(att, "ManagedFailJob", FakeFailJob)
    inventory = watcher_items({}) if inventory is None else inventory

    def kubectl(plane: str, arguments: tuple[str, ...], _options: Any) -> str:
        if arguments[:2] == ("get", "pod"):
            # The job is every managed-job variant's first mutation
            # (test_general_live_managed_job_start); the watcher comes after.
            assert FakeFailJob.instances, "the watcher was looked up before the job"
            return json.dumps(inventory)
        if arguments[:2] == ("get", "secret"):
            return base64.b64encode(b"https://cp.example.internal/").decode()
        if arguments[:1] == ("exec",):
            phases.exec_calls.append(arguments)
            assert phases.phase == "blocked", "the exit is triggered under the block"
            phases.triggered = True
            return ""
        raise AssertionError(f"unexpected kubectl {arguments}")

    job_name_holder: dict[str, str] = {}

    def store(script: str, *arguments: str) -> dict[str, Any]:
        job_name_holder.setdefault("name", arguments[1])
        return _store(phases, job_name_holder["name"])(script, *arguments)

    regional = Regional(kubectl_responder=kubectl, cpu_python=store)

    def block(_arguments: tuple[str, ...]) -> dict[str, Any]:
        phases.phase = "blocked"
        return probe_docs("pod-egress")["block"]

    def cleanup(_arguments: tuple[str, ...]) -> dict[str, Any]:
        phases.phase = "restored"
        return probe_docs("pod-egress")["cleanup"]

    fake = FakeProbe({"block": block, "cleanup": cleanup})
    ctx = isolation_context(
        tmp_path, regional, probes={f"isolation@{WATCHER_NODE}": fake}
    )
    return ctx, fake


def test_a_missing_watcher_after_the_job_started_is_a_fixture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, fake = _variant_context(
        tmp_path, Phases(), monkeypatch, inventory={"items": []}
    )
    with pytest.raises(RegionalFixtureError, match="exactly one Ready"):
        ctx.spec.execute(ctx)
    (job,) = FakeFailJob.instances
    assert ctx.cleanup.pending() == [f"job-{job.name}"], "the job's undo is registered"
    assert fake.calls == [], "no block is attempted without a watcher"


def test_api_unreachable_passes_when_every_action_happens_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phases = Phases()
    ctx, fake = _variant_context(tmp_path, phases, monkeypatch)
    spec = ctx.spec
    assert spec.approvals == ("WATCHER_NETWORK_ISOLATION",) and spec.flags == "X"
    assert spec.risk == "H" and spec.lanes == ("eks", "hp")
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert set(result["stages"]) == {
        "emergency_stop",
        "terminal_once",
        "decision",
        "single_action",
        "notifications",
        "watcher_restored",
    }
    assert [call[0] for call in fake.calls] == ["block", "cleanup"]
    assert fake.calls[0][1:9] == (
        "--kind",
        "pod-egress",
        "--deadman-seconds",
        str(iso.WATCHER_BLOCK_SECONDS),
        "--pod-ip",
        POD_IP,
        "--port",
        "443",
    )
    assert phases.exec_calls == [
        (
            "exec",
            FakeFailJob.instances[-1].name + "-master-0",
            "-c",
            "pytorch",
            "--",
            "/bin/sh",
            "-c",
            f"touch {att.EXIT_TRIGGER_PATH}",
        )
    ]
    assert FakeFailJob.instances[-1].mode == "trigger:1"
    job = FakeFailJob.instances[-1]
    assert ctx.cleanup.pending() == [f"job-{job.name}"], (
        "the block's undo was done and discarded"
    )
    assert result["watcher"]["node"] == WATCHER_NODE and result["endpoint"] == {
        "host": "cp.example.internal",
        "port": 443,
    }
    assert result["block_seconds"] >= att.API_BLOCK_SETTLE_SECONDS
    assert (ctx.evidence.case_dir / "api-blocked-records.json").is_file(), (
        "the in-block oracle read is evidence"
    )


def test_api_unreachable_is_inconclusive_when_the_watcher_still_reaches_the_control_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phases = Phases()
    phases.heartbeat_during = T0 + timedelta(days=2)
    ctx, fake = _variant_context(tmp_path, phases, monkeypatch)
    with pytest.raises(
        VariantInconclusive, match="did not cut the control-plane access"
    ):
        ctx.spec.execute(ctx)
    assert [call[0] for call in fake.calls] == ["block", "cleanup"], (
        "restored before giving up"
    )


@pytest.mark.parametrize(
    ("change", "stage", "needle"),
    [
        (
            {"stops": 2, "dup_notifications": True},
            "single_action",
            "STOP_WORKLOADS succeeded 2 times",
        ),
        ({"decision": "NO_ACTION"}, "decision", "expected one budgeted restart"),
        ({"emergency": False}, "emergency_stop", "no emergency-fallback"),
    ],
    ids=["double-stop", "no-restart-decision", "no-emergency-stop"],
)
def test_api_unreachable_fails_on_a_duplicate_or_missing_product_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
    stage: str,
    needle: str,
) -> None:
    phases = Phases()
    for key, value in change.items():
        setattr(phases, key, value)
    ctx, _fake = _variant_context(tmp_path, phases, monkeypatch)
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(needle in item for item in result["stages"][stage]), result["stages"]
    if phases.dup_notifications:
        assert result["stages"]["notifications"] == [
            "notification k1 was recorded 2 times",
            "2 containment notifications for incident inc-c",
        ]
    else:
        assert result["stages"]["notifications"] == []


def test_trigger_mode_renders_a_file_gated_exit_with_its_code() -> None:
    document = {
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "template": {
                        "spec": {"containers": [{"name": "pytorch", "env": []}]}
                    }
                }
            }
        }
    }
    del document  # the rendering itself runs live (test_general_live_managed_job_start)
    command = att.MODE_COMMANDS["trigger"]
    assert command == (
        f"while [ ! -e {att.EXIT_TRIGGER_PATH} ]; do sleep 1; done; exit ${{CODE:-1}}"
    )
    assert att.EXIT_TRIGGER_PATH.startswith("/tmp/"), (
        "the trigger lives in the container's /tmp"
    )
    # Other lanes add modes of their own (029 exit_once); these four are ours.
    assert {"exit", "stop", "hang", "trigger"} <= set(att.MODE_COMMANDS)


def test_the_approval_tokens_describe_the_scoped_blocks() -> None:
    assert "Pod's IP only" in APPROVALS["WATCHER_NETWORK_ISOLATION"]
    assert "600 s" in APPROVALS["WATCHER_NETWORK_ISOLATION"]
    assert "Node Agent listening port" in APPROVALS["NODE_AGENT_NETWORK_ISOLATION"]
    assert (
        "heartbeats, collectors and kubelet untouched"
        in APPROVALS["NODE_AGENT_NETWORK_ISOLATION"]
    )
