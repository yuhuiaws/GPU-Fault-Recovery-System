"""COLLECT-021: the Pod died before its RESTART_APP XID was ingested.

The proactive side must read the node as IDLE and record MONITOR_ONLY without
a workflow or a cordon; the passive side must still restart the attempt. The
helpers under test are the pure readings the live runner applies to store and
node snapshots, plus the host probe's process selection for ``kill-workload``.
"""

from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_collect021_late_xid_after_pod_death as collect021
from scripts.e2e.regional.probes import collector_node_probe as probe

NODE = "node-a"
POD_UID = "3f1c2b6e-1111-4a2a-9b3c-0123456789ab"
JOB_ID = "c021-1"
ATTEMPT_ID = "c021-1-a001"


# --- probe: which processes belong to the Pod ------------------------------


def _proc(root: Path, pid: int, cgroup: str, comm: str) -> None:
    directory = root / str(pid)
    directory.mkdir()
    (directory / "cgroup").write_text(f"0::{cgroup}\n", encoding="utf-8")
    (directory / "comm").write_text(comm + "\n", encoding="utf-8")


def test_kill_workload_matches_both_cgroup_spellings_and_skips_itself(
    tmp_path: Path,
) -> None:
    underscored = POD_UID.replace("-", "_")
    _proc(
        tmp_path, 4242, f"/kubepods.slice/kubepods-pod{underscored}.slice/x", "torchrun"
    )
    _proc(tmp_path, 4243, f"/kubepods/burstable/pod{POD_UID}/abc", "python")
    _proc(tmp_path, 4244, f"/kubepods/burstable/pod{POD_UID}/sandbox", "pause")
    _proc(tmp_path, 4245, "/kubepods/burstable/podother-uid/abc", "python")
    _proc(tmp_path, 7777, f"/kubepods/burstable/pod{POD_UID}/abc", "python")
    (tmp_path / "self").mkdir()
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "cgroup").write_text(f"0::/pod{POD_UID}\n", encoding="utf-8")

    processes = probe.workload_processes(
        POD_UID, proc=tmp_path, excluded_pids=frozenset({7777})
    )

    assert [(item["pid"], item["comm"]) for item in processes] == [
        (4242, "torchrun"),
        (4243, "python"),
    ], "both kubelet spellings match; pause, PID 1, other Pods and self do not"


def test_kill_workload_refuses_a_pod_with_no_process(tmp_path: Path) -> None:
    _proc(tmp_path, 4245, "/kubepods/burstable/podother-uid/abc", "python")
    with pytest.raises(probe.ProbeError, match="no process"):
        probe.kill_workload(
            Namespace(pod_uid=POD_UID), proc=tmp_path, kill=lambda pid, sig: None
        )


def test_kill_workload_sends_sigkill_and_reports_what_it_hit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    import signal

    _proc(tmp_path, 4242, f"/kubepods/pod{POD_UID}/a", "torchrun")
    _proc(tmp_path, 4243, f"/kubepods/pod{POD_UID}/a", "python")
    sent: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        if pid == 4243:
            raise ProcessLookupError

    probe.kill_workload(Namespace(pod_uid=POD_UID), proc=tmp_path, kill=kill)

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert sent == [(4242, signal.SIGKILL), (4243, signal.SIGKILL)]
    assert payload["pod_uid"] == POD_UID
    assert [item["pid"] for item in payload["killed"]] == [4242]
    assert [item["pid"] for item in payload["already_exited"]] == [4243]
    assert payload["signal"] == "SIGKILL"


def test_kill_workload_rejects_an_unsafe_pod_uid(tmp_path: Path) -> None:
    with pytest.raises(probe.ProbeError, match="unsafe"):
        probe.workload_processes("../etc", proc=tmp_path)


def test_probe_parser_exposes_kill_workload() -> None:
    arguments = probe.parser().parse_args(["kill-workload", "--pod-uid", POD_UID])
    assert arguments.handler is probe.kill_workload
    assert arguments.pod_uid == POD_UID


# --- runner: when does the store read the node as idle ----------------------


def _observation(**overrides: Any) -> dict[str, Any]:
    observation = {
        "job_id": JOB_ID,
        "attempt_id": ATTEMPT_ID,
        "workload_phase": "RUNNING",
        "containers": [
            {
                "pod_uid": POD_UID,
                "node_id": NODE,
                "terminated": False,
                "exit_code": None,
                "gpu_uuids": ["GPU-a"],
            },
            {
                "pod_uid": "other",
                "node_id": "node-b",
                "terminated": False,
                "exit_code": None,
                "gpu_uuids": ["GPU-b"],
            },
        ],
    }
    observation.update(overrides)
    return observation


def test_node_reads_idle_only_once_the_store_saw_the_death() -> None:
    assert not collect021.node_reads_idle(_observation(), NODE), (
        "a live container on the node is ACTIVE"
    )
    dead = _observation()
    dead["containers"][0].update(terminated=True, exit_code=137)
    assert collect021.node_reads_idle(dead, NODE), "a terminated container is not live"
    assert collect021.node_reads_idle(_observation(workload_phase="FAILED"), NODE), (
        "resolve() skips every non-PENDING/RUNNING phase"
    )
    assert collect021.node_reads_idle(_observation(containers=[]), NODE), (
        "no container at all on the node is idle"
    )
    assert not collect021.node_reads_idle(_observation(), "node-b"), (
        "node-b still has a live container of its own"
    )


def test_target_bdf_prefers_the_killed_pods_gpu() -> None:
    inventory = [
        {"uuid": "GPU-x", "pci_bdf": "0000:19:00.0"},
        {"uuid": "GPU-a", "pci_bdf": "0000:59:00.0"},
    ]
    assert collect021.target_bdf(_observation(), inventory, pod_uid=POD_UID) == (
        "0000:59:00.0"
    )
    assert (
        collect021.target_bdf(_observation(), inventory, pod_uid="missing")
        == "0000:19:00.0"
    ), "falls back to the node's first GPU"


# --- runner: proactive side --------------------------------------------------


def _node(**overrides: Any) -> dict[str, Any]:
    node = {
        "name": NODE,
        "boot_id": "boot-1",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    node.update(overrides)
    return node


def _proactive_state() -> dict[str, Any]:
    return {
        "event": {
            "xid": 13,
            "workload_state": "IDLE",
            "affected_workload_ids": [],
            "job_id": None,
        },
        "decision": {
            "disposition": "MONITOR_ONLY",
            "action": "NO_ACTION",
            "official_action": "RESTART_APP",
            "reasons": [
                "catalog RESTART_APP",
                "no managed application to restart on an IDLE node; recorded",
            ],
            "workflow_request_id": None,
            "incident_id": "inc-1",
        },
        "incident": {
            "incident_id": "inc-1",
            "state": "RECOVERED",
            "workflow_request_id": None,
        },
        "workflow": None,
        "commands": [],
    }


def test_proactive_side_passes_on_the_documented_readings() -> None:
    assert (
        collect021.proactive_errors(_proactive_state(), _node(), _node(), xid=13) == []
    )


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda s: s["event"].__setitem__("xid", 31), "xid"),
        (lambda s: s["event"].__setitem__("workload_state", "ACTIVE"), "IDLE"),
        (lambda s: s["event"].__setitem__("workload_state", "UNKNOWN"), "IDLE"),
        (
            lambda s: s["event"].__setitem__("affected_workload_ids", ["w"]),
            "affected_workload_ids",
        ),
        (
            lambda s: s["decision"].__setitem__("disposition", "EXECUTABLE"),
            "disposition",
        ),
        (lambda s: s["decision"].__setitem__("action", "RESTART_WORKLOAD"), "action"),
        (
            lambda s: s["decision"].__setitem__("official_action", "RESET_GPU"),
            "official",
        ),
        (lambda s: s["decision"].__setitem__("reasons", ["catalog"]), "reason"),
        (
            lambda s: s["decision"].__setitem__("workflow_request_id", "wf-1"),
            "workflow",
        ),
        (lambda s: s["incident"].__setitem__("state", "QUARANTINED"), "RECOVERED"),
        (
            lambda s: s["incident"].__setitem__("workflow_request_id", "wf-1"),
            "workflow",
        ),
        (lambda s: s.__setitem__("workflow", {"status": "BLOCKED"}), "workflow"),
        (lambda s: s.__setitem__("commands", [{"command_id": "c"}]), "command"),
        (lambda s: s.__setitem__("decision", None), "decision"),
        (lambda s: s.__setitem__("incident", None), "incident"),
        (lambda s: s.__setitem__("event", None), "event"),
    ],
)
def test_each_proactive_reading_is_checked(mutate, fragment: str) -> None:
    state = _proactive_state()
    mutate(state)
    errors = collect021.proactive_errors(state, _node(), _node(), xid=13)
    assert any(fragment in item for item in errors), (fragment, errors)


@pytest.mark.parametrize(
    ("after", "fragment"),
    [
        (_node(unschedulable=True), "unschedulable"),
        (
            _node(taints=[{"key": "gpu-fault.io/quarantine", "effect": "NoSchedule"}]),
            "taint",
        ),
        (_node(ownership_annotations={"gpu-fault.io/owner": "inc"}), "annotation"),
        (_node(boot_id="boot-2"), "boot_id"),
    ],
)
def test_node_mutations_fail_the_proactive_side(
    after: dict[str, Any], fragment: str
) -> None:
    errors = collect021.node_untouched_errors(_node(), after, label="A")
    assert any(fragment in item for item in errors), (fragment, errors)
    assert collect021.node_untouched_errors(_node(), _node(), label="A") == []


def test_an_unrelated_taint_is_not_ours() -> None:
    after = _node(
        taints=[{"key": "node.kubernetes.io/unreachable", "effect": "NoExecute"}]
    )
    errors = collect021.node_untouched_errors(_node(), after, label="A")
    assert errors == [], "only gpu-fault.io taints are this case's side effect"


# --- runner: passive side ----------------------------------------------------


def _death() -> dict[str, Any]:
    dead = _observation(workload_phase="FAILED")
    dead["containers"][0].update(terminated=True, exit_code=137)
    return dead


def _restarted() -> dict[str, Any]:
    return {
        "pods": [
            {
                "uid": "n1",
                "node": NODE,
                "phase": "Running",
                "attempt_id": "c021-1-a002",
            },
            {
                "uid": "n2",
                "node": "node-b",
                "phase": "Running",
                "attempt_id": "c021-1-a002",
            },
            {
                "uid": "n3",
                "node": "node-c",
                "phase": "Running",
                "attempt_id": "c021-1-a002",
            },
        ]
    }


def _restart_state() -> dict[str, Any]:
    return {
        "restart_budget": {"budget": 1, "restart_count": 1},
        "observations": [
            _death(),
            _observation(attempt_id="c021-1-a002", workload_phase="RUNNING"),
        ],
    }


def test_passive_side_passes_on_the_documented_readings() -> None:
    assert (
        collect021.passive_errors(
            kill={"killed": [{"pid": 4242, "comm": "torchrun"}]},
            death=_death(),
            restarted=_restarted(),
            restart_state=_restart_state(),
            source_uids={"s1", "s2", "s3"},
            pod_uid=POD_UID,
            attempt_id=ATTEMPT_ID,
        )
        == []
    )


@pytest.mark.parametrize(
    ("field", "mutate", "fragment"),
    [
        ("kill", lambda k: k.__setitem__("killed", []), "killed"),
        (
            "death",
            lambda d: d["containers"][0].update(terminated=True, exit_code=0),
            "exit",
        ),
        ("death", lambda d: d.__setitem__("workload_phase", "RUNNING"), "phase"),
        ("restarted", lambda r: r["pods"].pop(), "three"),
        ("restarted", lambda r: r["pods"][0].__setitem__("uid", "s1"), "source"),
        (
            "restarted",
            lambda r: [p.__setitem__("attempt_id", ATTEMPT_ID) for p in r["pods"]],
            "attempt",
        ),
        (
            "restart_state",
            lambda s: s["restart_budget"].__setitem__("restart_count", 0),
            "restart_count",
        ),
        (
            "restart_state",
            lambda s: s.__setitem__("restart_budget", None),
            "restart_count",
        ),
        (
            "restart_state",
            lambda s: s.__setitem__("observations", [_death()]),
            "observation",
        ),
    ],
)
def test_each_passive_reading_is_checked(field: str, mutate, fragment: str) -> None:
    inputs: dict[str, Any] = {
        "kill": {"killed": [{"pid": 4242, "comm": "torchrun"}]},
        "death": _death(),
        "restarted": _restarted(),
        "restart_state": _restart_state(),
    }
    mutate(inputs[field])
    errors = collect021.passive_errors(
        source_uids={"s1", "s2", "s3"}, pod_uid=POD_UID, attempt_id=ATTEMPT_ID, **inputs
    )
    assert any(fragment in item for item in errors), (fragment, errors)


def test_death_without_the_killed_container_still_counts_by_phase() -> None:
    """After STOP_WORKLOADS the Pods are gone; the phase alone proves the death."""

    death = _observation(workload_phase="STOPPED", containers=[])
    errors = collect021.passive_errors(
        kill={"killed": [{"pid": 1, "comm": "python"}]},
        death=death,
        restarted=_restarted(),
        restart_state=_restart_state(),
        source_uids={"s1", "s2", "s3"},
        pod_uid=POD_UID,
        attempt_id=ATTEMPT_ID,
    )
    assert errors == []


# --- runner: resources are owned before their first wait ---------------------


def _settings(tmp_path: Path) -> Any:
    class Regional:
        gpu_kubeconfig = "kc"
        gpu_context = "ctx"
        namespace = "ns"

    return collect021.Settings(
        regional=Regional(),  # type: ignore[arg-type]
        site_file=tmp_path / "site.yaml",
        host_probe_image="img",
        predecessor_path=tmp_path / "pred.json",
    )


def test_workload_is_owned_before_its_first_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            raise TimeoutError("pods never ran")

    workload = Workload()
    monkeypatch.setattr(
        collect021.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect021, "managed_fixture", lambda *_, **__: workload)
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(TimeoutError):
        collect021.run_late_xid_section(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert workloads == [workload]


def test_probe_is_owned_before_it_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            return {"pods": [{"uid": "u1", "node": NODE}]}

    class Collector:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def create(self) -> None:
            raise RuntimeError("image pull failed")

    class Regional:
        def node_snapshot(self, node: str) -> dict[str, Any]:
            return _node()

    monkeypatch.setattr(
        collect021.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect021, "managed_fixture", lambda *_, **__: Workload())
    monkeypatch.setattr(
        collect021.workload_case, "wait_observation", lambda *_, **__: {}
    )
    monkeypatch.setattr(collect021, "CollectorAcceptanceFixture", Collector)
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(RuntimeError, match="image pull failed"):
        collect021.run_late_xid_section(
            _settings(tmp_path),
            Regional(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert len(workloads) == 1
    assert len(fixtures) == 1 and isinstance(fixtures[0], Collector)


def test_wait_for_decision_returns_once_the_policy_has_answered(tmp_path: Path) -> None:
    """No workflow is the expected outcome, so the wait keys on the decision."""

    snapshots = iter(
        [
            {"event": None, "decision": None, "incident": None, "workflow": None},
            {
                "event": {"xid": 13},
                "decision": None,
                "incident": None,
                "workflow": None,
            },
            {
                "event": {"xid": 13},
                "decision": {"disposition": "MONITOR_ONLY"},
                "incident": {"state": "RECOVERED"},
                "workflow": None,
            },
        ]
    )
    calls: list[dict[str, Any]] = []

    class Regional:
        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return copy.deepcopy(next(snapshots))

    state = collect021.wait_for_decision(
        Regional(),  # type: ignore[arg-type]
        node=NODE,
        marker="m",
        observed_after=None,
        case_dir=tmp_path,
        timeout_seconds=60,
        job_id=JOB_ID,
        attempt_id=ATTEMPT_ID,
        poll_seconds=0,
    )
    assert state["decision"]["disposition"] == "MONITOR_ONLY"
    assert len(calls) == 3
    assert all(call["queue_attempts"] == 1 for call in calls), "wait loops read light"
    assert (tmp_path / "timeline.json").is_file(), "the wait writes a timeline"


def test_wait_for_decision_times_out_without_a_decision(tmp_path: Path) -> None:
    class Regional:
        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            return {"event": {"xid": 13}, "decision": None, "incident": None}

    with pytest.raises(collect021.RegionalFixtureError, match="decision"):
        collect021.wait_for_decision(
            Regional(),  # type: ignore[arg-type]
            node=NODE,
            marker="m",
            observed_after=None,
            case_dir=tmp_path,
            timeout_seconds=0,
            poll_seconds=0,
        )
