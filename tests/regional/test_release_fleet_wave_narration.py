"""What a release says while it is rolling nodes, and how often it looks.

`data-plane-progress` is checkpointed once per cluster, so on the 2026-09-05
upgrade the ROLLING phase printed two phase lines seven and a half minutes apart
with four one-node waves in between and nothing to distinguish a working release
from a hung one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
CONVERGENCE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_agent_convergence.py"
)
FLEET = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_fleet_rollout.py"
)
NARRATION = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_narration.py"
)

CLUSTER = "hp-cluster"
ARTIFACT = "a" * 64
BUNDLE = "b" * 64
TEMPLATE = "e" * 64
CONFIG = "c" * 64
TARGET = SimpleNamespace(cluster_id="gpu-1", hyperpod_cluster_name=CLUSTER)


def _node(name: str, *, state: str, aligned: bool) -> dict[str, Any]:
    annotations = {"gpu-fault.io/installer-state": state}
    if aligned:
        annotations.update(
            {
                "gpu-fault.io/installer-artifact-sha256": ARTIFACT,
                "gpu-fault.io/installer-config-digest": CONFIG,
                "gpu-fault.io/installer-bundle-sha256": BUNDLE,
                "gpu-fault.io/installer-template-sha256": TEMPLATE,
                "gpu-fault.io/installer-node-uid": f"uid-{name}",
            }
        )
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {"sagemaker.amazonaws.com/cluster-name": CLUSTER},
            "annotations": annotations,
        }
    }


def _wait_agents(
    monkeypatch: pytest.MonkeyPatch, *polls: list[dict[str, Any]]
) -> list[float]:
    """Drive `wait_agents` over a scripted sequence of node listings.

    Returns the sleep durations it asked for, which is how the polling cadence is
    observed without spending the wall time.
    """

    listings = iter(polls)
    slept: list[float] = []
    monkeypatch.setattr(
        CONVERGENCE, "gpu_node_items", lambda *_args, **_kwargs: next(listings)
    )
    monkeypatch.setattr(
        CONVERGENCE.time, "sleep", lambda seconds: slept.append(seconds)
    )
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        bundle_sha=BUNDLE,
        node_template_sha=TEMPLATE,
        config=SimpleNamespace(
            agent_config_digest=CONFIG,
            runtime_profile_version="profile-v1",
            release_manifest_schema_version=3,
        ),
        _agent_heartbeats_converged=lambda *_args, **_kwargs: True,
    )
    CONVERGENCE.wait_agents(
        release,
        TARGET,
        ARTIFACT,
        bundle_sha=BUNDLE,
        template_sha=TEMPLATE,
        config_digest=CONFIG,
        node_names=("node-a", "node-b"),
        timeout_seconds=900,
    )
    return slept


def test_a_wait_names_the_node_it_is_waiting_for(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole silence inside `data-plane-progress` is this wait.

    Without a line from it, an operator watching a rollout sees `get nodes -o
    json` repeated for minutes and cannot tell whether the Reconciler has even
    created the installer Job.
    """

    _wait_agents(
        monkeypatch,
        [
            _node("node-a", state="Running", aligned=False),
            _node("node-b", state="Succeeded", aligned=True),
        ],
        [
            _node("node-a", state="Succeeded", aligned=True),
            _node("node-b", state="Succeeded", aligned=True),
        ],
    )

    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("installer-wait ")
    ]
    assert len(lines) == 1, lines
    assert "cluster=gpu-1" in lines[0]
    assert "nodes=2 aligned=1" in lines[0]
    assert "waiting=installers" in lines[0]
    assert "pending=node-a:Running" in lines[0], (
        "the line has to name the node and its installer state to be actionable"
    )
    assert re.search(r"elapsed=\d+\.\ds", lines[0]) is not None, lines[0]


def test_a_node_with_no_installer_job_yet_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing annotation is the Reconciler not having started, not a failure.

    It is the state that says the wait is on the Reconciler's own 15s reconcile
    pass rather than on the installer, so it must be distinguishable.
    """

    _wait_agents(
        monkeypatch,
        [_node("node-a", state="", aligned=False)],
        [_node("node-a", state="Succeeded", aligned=True)],
    )

    assert "pending=node-a:<none>" in capsys.readouterr().err


def test_a_wait_on_heartbeats_alone_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Installed but not yet reporting is a different problem from not installed.

    The nodes carry the new identity and the release is waiting on the Agent to
    heartbeat it to the control plane, so pointing an operator at the installer
    would send them to the wrong place.
    """

    heartbeats = iter((False, True))
    listings = iter(
        (
            [_node("node-a", state="Succeeded", aligned=True)],
            [_node("node-a", state="Succeeded", aligned=True)],
        )
    )
    monkeypatch.setattr(
        CONVERGENCE, "gpu_node_items", lambda *_args, **_kwargs: next(listings)
    )
    monkeypatch.setattr(CONVERGENCE.time, "sleep", lambda _seconds: None)
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        bundle_sha=BUNDLE,
        node_template_sha=TEMPLATE,
        config=SimpleNamespace(
            agent_config_digest=CONFIG,
            runtime_profile_version="profile-v1",
            release_manifest_schema_version=3,
        ),
        _agent_heartbeats_converged=lambda *_args, **_kwargs: next(heartbeats),
    )

    CONVERGENCE.wait_agents(
        release,
        TARGET,
        ARTIFACT,
        bundle_sha=BUNDLE,
        template_sha=TEMPLATE,
        config_digest=CONFIG,
        node_names=("node-a",),
        timeout_seconds=900,
    )

    line = next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("installer-wait ")
    )
    assert "waiting=heartbeats" in line
    assert "aligned=1" in line and "pending=-" in line


def test_an_unchanged_wait_is_not_narrated_twice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wait window is up to fifteen minutes of polling at 5 to 15 seconds.

    Repeating an identical line per poll would bury the command trace it was
    added to explain, so the line repeats only when the state moves -- or on the
    heartbeat interval, which no test here is long enough to reach.
    """

    pending = [_node("node-a", state="Running", aligned=False)]
    _wait_agents(
        monkeypatch,
        list(pending),
        list(pending),
        list(pending),
        [_node("node-a", state="Succeeded", aligned=True)],
    )

    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("installer-wait ")
    ]
    assert len(lines) == 1, lines


def test_polling_backs_off_while_nothing_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    pending = [_node("node-a", state="Running", aligned=False)]
    slept = _wait_agents(
        monkeypatch,
        list(pending),
        list(pending),
        list(pending),
        [_node("node-a", state="Succeeded", aligned=True)],
    )

    assert slept == [5.0, 7.5, 11.25], slept


def test_polling_speeds_back_up_when_a_node_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished node means the next one starts now, not in fifteen seconds.

    The backoff exists so a quiet fifteen-minute window does not re-list every
    node every five seconds, but held across a wave it also spends the tail of
    every install asleep -- on a fleet rolled one node at a time that is the
    difference between polling granularity and real convergence time.
    """

    both_pending = [
        _node("node-a", state="Running", aligned=False),
        _node("node-b", state="", aligned=False),
    ]
    one_pending = [
        _node("node-a", state="Succeeded", aligned=True),
        _node("node-b", state="Running", aligned=False),
    ]
    slept = _wait_agents(
        monkeypatch,
        list(both_pending),
        list(both_pending),
        list(one_pending),
        list(one_pending),
        [
            _node("node-a", state="Succeeded", aligned=True),
            _node("node-b", state="Succeeded", aligned=True),
        ],
    )

    assert slept == [5.0, 7.5, 5.0, 7.5], slept


def _fleet_release(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    stages: list[str] = []
    monkeypatch.setattr(
        FLEET,
        "ensure_rollout_wave_safe",
        lambda *_args, **_kwargs: stages.append("safety"),
    )
    monkeypatch.setattr(
        FLEET,
        "hand_wave_to_reconciler",
        lambda *_args, **_kwargs: stages.append("handoff") or ("b" * 64, "e" * 64),
    )
    reads = iter(({"status": "SUCCEEDED", **_DEPLOYMENT_DONE},))
    release = SimpleNamespace(
        _fleet_command=lambda operation, _payload: (
            {"node_ids": ["node-b"]} if operation == "next-wave" else next(reads)
        ),
        _wait_agents=lambda *_args, **_kwargs: stages.append("install"),
    )
    return release, stages


_DEPLOYMENT_IN_PROGRESS = {
    "status": "IN_PROGRESS",
    "waves": [["node-a"], ["node-b"]],
    "nodes": [
        {"node_id": "node-a", "status": "READY"},
        {"node_id": "node-b", "status": "PENDING"},
    ],
}
_DEPLOYMENT_DONE = {
    "waves": [["node-a"], ["node-b"]],
    "nodes": [
        {"node_id": "node-a", "status": "READY"},
        {"node_id": "node-b", "status": "READY"},
    ],
}


def test_a_wave_is_announced_before_it_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is checkpointed until the whole cluster is done.

    A wave is the loop's unit of work, so its start is where the log has to say
    which nodes are about to move and how much of the fleet is already on the new
    identity -- otherwise the release is silent from the first wave to the last.
    """

    release, _stages = _fleet_release(monkeypatch)

    FLEET.run_fleet_waves(
        release, TARGET, _wave_context(), dict(_DEPLOYMENT_IN_PROGRESS)
    )

    start = next(
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("fleet-wave ")
    )
    assert "cluster=gpu-1" in start
    assert "wave=2/2" in start, start
    assert "ready=1/2" in start, "the line has to place the wave in the fleet"
    assert "nodes=node-b" in start


def test_a_finished_wave_reports_where_its_time_went(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three stages with three different causes, so one total cannot be acted on.

    `safety` is fixed by Agent leases and open commands, `handoff` by the
    Reconciler rollout and the previous wave's Jobs, `install` by the node
    itself. Only the last is the cluster doing the work the release exists for,
    and telling them apart is what says whether a slow rollout is worth
    optimising or is simply installing.
    """

    release, stages = _fleet_release(monkeypatch)

    FLEET.run_fleet_waves(
        release, TARGET, _wave_context(), dict(_DEPLOYMENT_IN_PROGRESS)
    )

    assert stages == ["safety", "safety", "handoff", "install"]
    done = next(
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("fleet-wave-done ")
    )
    for field in ("safety", "handoff", "install", "elapsed"):
        assert re.search(rf"{field}=\d+\.\ds", done) is not None, done
    assert "ready=2/2" in done, "the done line reports the fleet after the wave"


def _wave_context() -> Any:
    return FLEET.FleetWaveContext(
        phase="upgrade",
        deployment_id="deployment",
        node_names=("node-a", "node-b"),
        paused_identity=("b" * 64, "e" * 64),
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        expected_profile="profile-v1",
        executor_wheel_filename=None,
        expected_compatibility="compatibility",
        desired_bundle=BUNDLE,
        desired_template=TEMPLATE,
        template_config_map=None,
        max_unavailable=1,
        runtime_image=None,
        node_installer_image=None,
        allow_legacy_identity=False,
        agent_identity=None,
    )


def test_a_handoff_reads_the_reconciler_deployment_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollout barrier already holds the object the identity check needs.

    Reading it again was a second identical `get deployment` on every wave, and
    worse than redundant: the identity would have been checked against whatever
    generation the later read returned rather than against the one the barrier
    passed.
    """

    reads: list[list[str]] = []
    environment = [
        {"name": "GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP", "value": "wave-cm"},
        {"name": "GPU_FAULT_INSTALLER_BUNDLE_SHA256", "value": BUNDLE},
        {"name": "GPU_FAULT_INSTALLER_TEMPLATE_SHA256", "value": TEMPLATE},
    ]
    rolled_out = {
        "spec": {
            "template": {
                "spec": {"containers": [{"name": "reconciler", "env": environment}]}
            }
        }
    }
    monkeypatch.setattr(
        FLEET,
        "wait_deployment_rollout",
        lambda *_args, **_kwargs: {"object": rolled_out},
    )
    patched: dict[str, str] = {}

    def _run(arguments: list[str], **_kwargs: Any) -> str:
        if "patch" in arguments:
            patched.update(json.loads(arguments[arguments.index("-p") + 1])["data"])
        return ""

    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False, run=_run),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *args: list(args),
        _get_json=lambda arguments: reads.append(list(arguments)) or {"data": patched},
        _settle_installer_jobs=lambda _target: None,
    )

    identity = FLEET.hand_wave_to_reconciler(
        release, TARGET, _wave_context(), ("node-a",)
    )

    assert identity == (BUNDLE, TEMPLATE)
    assert [read for read in reads if "deployment" in read] == [], reads


def test_a_step_line_does_not_consume_a_phase_split(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """These lines are printed inside a phase, not at its boundary.

    `elapsed` on a phase line is the cost of reaching that checkpoint, and the
    clock's split is what produces it. A step line taking a split would hand the
    next phase an elapsed measured from the middle of the previous one -- and a
    wave narrating every thirty seconds would shrink a seven-minute phase to the
    last thirty of it.
    """

    ticks = iter((100.0, 130.0, 160.0))
    monkeypatch.setattr(NARRATION.time, "monotonic", lambda: next(ticks))
    NARRATION.PHASE_CLOCK.start()
    release = SimpleNamespace(state={"cluster_ids": []})

    NARRATION.narrate_step("fleet-wave", cluster="gpu-1")
    NARRATION.narrate_phase(release, "data-plane-progress")

    step, phase = capsys.readouterr().err.strip().splitlines()
    assert "total=" in step and "elapsed=" not in step, step
    phase_elapsed = float(re.search(r"elapsed=(\d+\.\d)s", phase).group(1))
    phase_total = float(re.search(r"total=(\d+\.\d)s", phase).group(1))
    assert phase_elapsed == pytest.approx(phase_total, abs=0.2), (
        "the step line stole the phase's elapsed time"
    )
