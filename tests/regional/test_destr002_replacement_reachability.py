"""DESTR-002: the replay reaches the replacement executor wherever it was placed.

The arbiter Deployments tolerate the cordon and ``gpu-fault.io/quarantined`` by
design, so the replacement for the deleted submitting executor may be bound to
the very node the case is rebooting (live round 8 and round 9 on HyperPod).
Live 2026-10-16 (round 9 attempt 1): the replacement became Ready on the
rebooted node 258 s after the delete, the runner exec'ed at once, HyperPod
restarted kubelet with its final configuration at that moment and the single
exec failed "unable to upgrade connection" -- the case FAILed although the
product had submitted exactly one reboot and finished the workflow.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional.regional_commands import RegionalCommandFailed
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_actions import ActionHarness
from tests.regional._cov95_destr_warm import NOW

TARGET = "node-a"
COMMAND = {
    "step": {"operation": "RESTART_NODE", "node_ids": [TARGET]},
    "result_details": {
        "executor_id": "executor-a/7",
        "submission_idempotency_key": "key-a",
    },
}
BEFORE = [
    {"name": "executor-a", "uid": "uid-a", "node": "gpu-1"},
    {"name": "executor-b", "uid": "uid-b", "node": "gpu-2"},
]
UPGRADE_REFUSED = "error: unable to upgrade connection: pod does not exist"


def _restart(replacement_node: str) -> dict[str, Any]:
    return {
        "before": BEFORE,
        "after": [
            BEFORE[1],
            {"name": "executor-c", "uid": "uid-c", "node": replacement_node},
        ],
    }


def _execs(h: ActionHarness) -> list[tuple[Any, ...]]:
    return [
        args
        for name, args, _ in h.calls
        if name == "regional.kubectl" and "exec" in args
    ]


def _refuse_execs(h: ActionHarness, stderr: str, *, times: int) -> list[float]:
    """Refuse the first ``times`` replay execs with ``stderr``; when each ran."""

    moments: list[float] = []

    def kubectl(value: Any, args: Any, _kwargs: Any) -> Any:
        if args[1] == "exec":
            moments.append(h.clock.elapsed)
            if len(moments) <= times:
                raise RegionalCommandFailed(
                    1, stderr, ["kubectl", "exec", "-i", args[3]]
                )
        return value

    h.transforms["regional.kubectl"] = kubectl
    return moments


def _node_ready_for(h: ActionHarness, seconds: float) -> None:
    """The target flipped Ready ``seconds`` before the harness clock started."""

    since = (NOW - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")

    def snapshot(value: dict[str, Any], args: Any, _kwargs: Any) -> dict[str, Any]:
        return {**value, "ready_since": since} if args[0] == TARGET else value

    h.transforms["regional.node_snapshot"] = snapshot


def test_a_replacement_bound_to_the_rebooting_node_is_waited_for_past_300_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)

    def pods(value: list[dict[str, Any]], _args: Any, _kwargs: Any) -> Any:
        if not any(item["name"] == "executor-c" for item in value):
            return value
        if h.clock.elapsed < 400:
            # The replacement is bound to the rebooting node and not Ready yet.
            return [item for item in value if item["name"] != "executor-c"]
        return [
            {**item, "node": TARGET} if item["name"] == "executor-c" else item
            for item in value
        ]

    inventory = {
        "items": [
            {"metadata": {"uid": "uid-b"}, "spec": {"nodeName": "gpu-2"}},
            {
                "metadata": {"uid": "uid-a", "deletionTimestamp": "t"},
                "spec": {"nodeName": TARGET},
            },
            {"metadata": {"uid": "uid-c"}, "spec": {"nodeName": TARGET}},
        ]
    }

    def kubectl(value: Any, args: Any, _kwargs: Any) -> Any:
        return json.dumps(inventory) if "-l" in args else value

    h.transforms["regional.ready_pods"] = pods
    h.transforms["regional.kubectl"] = kubectl

    result = reboot.restart_executor(h.regional, COMMAND)

    assert result["bound_to_rebooting_node"] == TARGET, result
    assert {item["uid"] for item in result["after"]} == {"uid-b", "uid-c"}, result
    assert 400 <= h.clock.elapsed < reboot.REPLACEMENT_ON_REBOOTING_NODE_SECONDS, (
        h.clock.elapsed
    )
    reads = [
        args for name, args, _ in h.calls if name == "regional.kubectl" and "-l" in args
    ]
    assert len(reads) == 1, "the inventory is read once, when 300 s are spent"


@pytest.mark.parametrize("inventory", ["elsewhere", "unreadable"])
def test_a_replacement_not_bound_to_the_rebooting_node_keeps_the_300_s_budget(
    inventory: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    h.transforms["regional.ready_pods"] = lambda value, *_a: [
        item for item in value if item["name"] != "executor-c"
    ]
    pending = {"items": [{"metadata": {"uid": "uid-c"}, "spec": {"nodeName": "gpu-3"}}]}
    h.transforms["regional.kubectl"] = lambda value, args, _k: (
        (json.dumps(pending) if inventory == "elsewhere" else "{")
        if "-l" in args
        else value
    )

    with pytest.raises(RegionalFixtureError, match="did not replace deleted Pod"):
        reboot.restart_executor(h.regional, COMMAND)

    assert h.clock.elapsed == reboot.REPLACEMENT_READY_SECONDS, h.clock.elapsed


def test_the_replay_waits_until_the_rebooted_node_has_been_ready_long_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    _node_ready_for(h, 10)
    moments = _refuse_execs(h, UPGRADE_REFUSED, times=0)

    value = reboot.replay_from_replacement(h.regional, COMMAND, _restart(TARGET))

    assert moments and moments[0] >= reboot.REPLAY_NODE_READY_SECONDS - 10, moments
    assert value["duplicate"] is True, value
    assert value["executor_node"] == TARGET, value
    assert value["replacement_on_rebooting_node"] is True, value
    assert value["node_ready_seconds"] >= reboot.REPLAY_NODE_READY_SECONDS, value
    assert value["refused_execs"] == [], value


def test_an_exec_refused_before_the_probe_ran_is_retried_on_the_same_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live round-9 shape: kubelet restarted under the first exec."""

    h = ActionHarness(reboot, tmp_path, monkeypatch)
    _node_ready_for(h, 3600)
    _refuse_execs(h, UPGRADE_REFUSED, times=1)

    value = reboot.replay_from_replacement(h.regional, COMMAND, _restart(TARGET))

    assert [args[3] for args in _execs(h)] == ["executor-c", "executor-c"], h.calls
    assert value["duplicate"] is True, value
    assert value["executor_pod_uid"] == "uid-c", value
    assert value["refused_execs"] == ["kubectl: unable to upgrade connection"], value


def test_a_failure_after_the_probe_started_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    _refuse_execs(h, "Traceback (most recent call last):\nStopIteration\n", times=5)

    with pytest.raises(
        RegionalFixtureError, match="could not run in replacement"
    ) as raised:
        reboot.replay_from_replacement(h.regional, COMMAND, _restart("gpu-1"))

    assert len(_execs(h)) == 1, h.calls
    assert "[StopIteration]" in str(raised.value), raised.value
    assert "refused before" not in str(raised.value), raised.value


def test_refusals_on_a_healthy_node_stop_at_the_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    _refuse_execs(h, UPGRADE_REFUSED, times=10_000)

    with pytest.raises(RegionalFixtureError, match="refused before") as raised:
        reboot.replay_from_replacement(h.regional, COMMAND, _restart("gpu-1"))

    expected = reboot.REPLAY_EXEC_RETRY_SECONDS // reboot.REPLAY_POLL_SECONDS + 1
    assert len(_execs(h)) == expected, len(_execs(h))
    assert h.clock.elapsed == reboot.REPLAY_EXEC_RETRY_SECONDS, h.clock.elapsed
    assert "pod does not exist" not in str(raised.value), "the output stays redacted"


@pytest.mark.parametrize("defect", ["recreated", "never-ready"])
def test_the_replay_refuses_a_recreated_pod_or_a_node_that_never_settles(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    if defect == "recreated":
        _refuse_execs(h, UPGRADE_REFUSED, times=1)
        original = h.transforms["regional.kubectl"]

        def kubectl(value: Any, args: Any, kwargs: Any) -> Any:
            if args[1:3] == ("get", "pod") and _execs(h):
                return json.dumps({"metadata": {"uid": "uid-d"}})
            return original(value, args, kwargs)

        h.transforms["regional.kubectl"] = kubectl
        expected = "UID changed before replay"
    else:
        h.transforms["regional.node_snapshot"] = lambda value, *_a: {
            **value,
            "ready": "False",
        }
        expected = "did not settle before the replay"

    with pytest.raises(RegionalFixtureError, match=expected):
        reboot.replay_from_replacement(h.regional, COMMAND, _restart("gpu-1"))

    assert len(_execs(h)) == (1 if defect == "recreated" else 0), h.calls


def test_the_case_passes_when_the_replacement_lands_on_the_rebooting_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end on the live round-9 sequence: the replacement is bound to the
    rebooted node, that node has just flipped Ready, the first exec is refused."""

    h = ActionHarness(reboot, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.transforms["regional.ready_pods"] = lambda value, *_a: [
        {**item, "node": TARGET} if item["name"] == "executor-c" else item
        for item in value
    ]
    _node_ready_for(h, 5)
    _refuse_execs(h, UPGRADE_REFUSED, times=1)

    code, report = h.execute(tmp_path)

    assert code == 0 and report["verdict"] == "PASS", report
    replay = report["duplicate_replay"]
    assert replay["duplicate"] is True, replay
    assert replay["replacement_on_rebooting_node"] is True, replay
    assert replay["refused_execs"] == ["kubectl: unable to upgrade connection"], replay
    assert report["executor_restart"]["bound_to_rebooting_node"] is None, report
    evidence = json.loads(
        (tmp_path / "cases" / reboot.CASE_ID / "duplicate-replay.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence == replay, evidence


def test_a_transient_node_read_failure_is_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    reads: list[float] = []

    def snapshot(value: dict[str, Any], _args: Any, _kwargs: Any) -> dict[str, Any]:
        reads.append(h.clock.elapsed)
        if len(reads) == 1:
            raise RegionalFixtureError("node read refused")
        return value

    h.transforms["regional.node_snapshot"] = snapshot

    value = reboot.replay_from_replacement(h.regional, COMMAND, _restart("gpu-1"))

    assert reads == [0, reboot.REPLAY_POLL_SECONDS], reads
    assert len(_execs(h)) == 1 and value["duplicate"] is True, value
    assert value["replacement_on_rebooting_node"] is False, value
