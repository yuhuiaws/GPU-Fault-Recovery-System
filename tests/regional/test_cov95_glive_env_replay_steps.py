"""GF-ENV-LIVE host controller: the steps that change the node.

``install-source``, ``deploy-proxy``, ``repoint-collector``, ``arm``, the 009
pause (``pause-collector`` / ``pause-wait`` / ``resume-collector``),
``restart-collector``, ``restore-collector`` and ``stop-proxy`` run through
the probe's ``main`` over ``EnvNode`` (``_cov95_glive_env_replay``): what each
refuses before it touches anything, what it rolls back on the node, and what
it leaves for its deadman.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import env_replay_proxy as probe
from tests.regional._cov95_glive_env_replay import (
    ENDPOINT,
    METRICS,
    PROXY,
    REAL,
    RUN,
    EnvNode,
    env_node,
    installed,
    proxy_up,
)

UUID = "GPU-11111111-2222-3333-4444-555555555555"
DEPLOY = ["deploy-proxy", "--run-id", RUN, "--gpu-uuid", UUID]
PAUSE = ["pause-collector", "--run-id", RUN, "--after-reads", "3", "--seconds", "120"]
STATUS_URL = "http://127.0.0.1:9411/status"
#: What a restarted collector that never got its endpoint holds.
NO_ENDPOINT = {"environment": {"GPU_FAULT_METRICS_MODE": "dcgm"}}


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EnvNode:
    return env_node(monkeypatch, tmp_path)


def main(*argv: str) -> int:
    return probe.main(list(argv))


def failure(node: EnvNode) -> str:
    return str(node.emitted[-1]["error_message"])


def repointed(node: EnvNode) -> None:
    """The node as a repoint left it: drop-in, override file and record."""

    root = installed(probe.REPLAY_TOOL_NAME, probe.PLAN_NAME, "probe.py")
    override = probe.override_env_path(RUN)
    override.write_text(f"{ENDPOINT}={PROXY}\n")
    (root / "repoint.json").write_text('{"original_endpoint": "' + REAL + '"}')
    probe.dropin_path(RUN).parent.mkdir(parents=True, exist_ok=True)
    probe.dropin_path(RUN).write_text(f"[Service]\nEnvironmentFile=-{override}\n")
    node.collector("200", {ENDPOINT: PROXY})


def restarts(node: EnvNode) -> list[list[str]]:
    return [c for c in node.commands if c[:3] == ["systemctl", "restart", METRICS]]


# --------------------------------------------------------------------------- #
# install-source / deploy-proxy
# --------------------------------------------------------------------------- #
def test_install_source_refuses_a_missing_sha256_and_invalid_base64(
    node: EnvNode,
) -> None:
    argv = ["install-source", "--run-id", RUN, "--name", probe.PLAN_NAME]
    assert main(*argv, "--sha256", "abc", "--b64", "AA==") == 1
    assert failure(node) == "a sha256 is required for every installed source"
    assert main(*argv, "--sha256", "0" * 64, "--b64", "not base64!") == 1
    assert failure(node).startswith("source is not valid base64: "), failure(node)
    assert not probe.run_dir(RUN).exists(), "a refused source writes nothing"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--upstream", "http://10.0.0.1:9400/metrics"], "upstream must be"),
        (["--upstream", "http://127.0.0.1:9400/other"], "upstream must be"),
        (["--max-seconds", "901"], "max-seconds must be 1..900"),
        (["--max-seconds", "0"], "max-seconds must be 1..900"),
        (["--gpu-stop-c", "81"], "safety limits may only be lowered"),
        (["--memory-stop-c", "0"], "safety limits may only be lowered"),
    ],
)
def test_deploy_proxy_refuses_unsafe_arguments_before_starting_anything(
    node: EnvNode, extra: list[str], message: str
) -> None:
    installed(probe.REPLAY_TOOL_NAME, probe.PLAN_NAME)
    assert main(*DEPLOY, *extra) == 1
    assert message in failure(node), failure(node)
    assert node.commands == [], "nothing is stopped or started"


def test_deploy_proxy_refuses_an_audit_file_of_an_earlier_start(node: EnvNode) -> None:
    root = installed(probe.REPLAY_TOOL_NAME, probe.PLAN_NAME)
    (root / "replay.jsonl").write_text("")
    assert main(*DEPLOY) == 1
    assert failure(node) == "an audit file already exists; start a new audited run"
    assert node.commands == [], "nothing is started"


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("activating", "replay proxy did not answer /status"),
        ("active", "replay proxy did not answer /status"),
        ("failed", "replay proxy unit did not stay up: "),
    ],
)
def test_deploy_proxy_gives_up_on_a_unit_that_never_serves(
    node: EnvNode, state: str, message: str
) -> None:
    installed(probe.REPLAY_TOOL_NAME, probe.PLAN_NAME)
    node.service_state = state
    assert main(*DEPLOY) == 1
    assert failure(node).startswith(message), failure(node)
    started = [c for c in node.commands if c[0] == "systemd-run"]
    assert len(started) == 1 and "--serve" in started[0], started
    waited = node.clock.now - 1000.0
    if state == "failed":
        assert waited == 0.0, "a failed unit is refused at once"
    else:
        assert waited >= 20, f"the whole start window was waited out: {waited}"


# --------------------------------------------------------------------------- #
# repoint-collector / arm
# --------------------------------------------------------------------------- #
def test_repoint_refuses_an_inactive_collector_and_an_existing_override(
    node: EnvNode,
) -> None:
    proxy_up(node)
    node.unit(METRICS, ActiveState="inactive", MainPID="0")
    assert main("repoint-collector", "--run-id", RUN) == 1
    assert failure(node) == f"{METRICS} is not active at baseline"
    node.collector("100", {ENDPOINT: REAL})
    installed(probe.OVERRIDE_ENV_NAME)
    assert main("repoint-collector", "--run-id", RUN) == 1
    assert failure(node) == "a repoint override for this run already exists"
    assert not any(c[0] == "systemd-run" for c in node.commands), "no deadman"


def test_repoint_reuses_the_probe_copy_and_refuses_an_unarmed_deadman(
    node: EnvNode,
) -> None:
    proxy_up(node)
    copy = installed("probe.py") / "probe.py"
    node.timers_fail = True
    assert main("repoint-collector", "--run-id", RUN) == 1
    assert failure(node) == "collector restore deadman timer is not active"
    assert copy.read_text() == "# reviewed probe.py\n", "an installed copy is kept"
    assert not probe.dropin_path(RUN).exists(), "no drop-in without a deadman"
    assert restarts(node) == [], "the collector is never restarted"


@pytest.mark.parametrize(
    ("restart", "message"),
    [
        ({"state": "failed"}, f"{METRICS} is not active after the repoint; restored"),
        (NO_ENDPOINT, "did not exec within the settle window; restored"),
    ],
    ids=["down", "no-endpoint"],
)
def test_repoint_rolls_back_a_collector_that_is_down_or_lacks_its_endpoint(
    node: EnvNode, restart: dict[str, Any], message: str
) -> None:
    installed(probe.REPLAY_TOOL_NAME, probe.PLAN_NAME)
    proxy_up(node)
    node.restarts.append(restart)
    assert main("repoint-collector", "--run-id", RUN) == 1
    assert failure(node).endswith(message), failure(node)
    assert not probe.dropin_path(RUN).exists(), "the drop-in is rolled back"
    assert not probe.override_env_path(RUN).exists(), "and its override file"
    assert len(restarts(node)) == 2, "the repoint's restart and the rollback's"
    assert probe.effective_endpoint(METRICS) == REAL, "the real exporter again"


def test_arm_reports_a_proxy_that_refuses_the_arm(node: EnvNode) -> None:
    proxy_up(node)
    node.collector("200", {ENDPOINT: PROXY})
    node.http["http://127.0.0.1:9411/arm"] = (409, "already armed\n")
    assert main("arm", "--run-id", RUN) == 1
    assert failure(node) == "arm returned 409: already armed"
    assert node.requests[-1] == ("POST", "http://127.0.0.1:9411/arm")


# --------------------------------------------------------------------------- #
# the 009 pause
# --------------------------------------------------------------------------- #
def test_pause_collector_refuses_bad_bounds_a_stopped_unit_and_a_dead_waiter(
    node: EnvNode,
) -> None:
    zero = ["pause-collector", "--run-id", RUN, "--after-reads", "0"]
    assert main(*zero, "--seconds", "120") == 1
    assert failure(node) == "after-reads must be at least 1"
    assert main(*PAUSE, "--wait-seconds", "601") == 1
    assert failure(node) == "wait seconds is outside 1..600"
    installed("probe.py")
    node.unit(METRICS, ActiveState="inactive", MainPID="0")
    assert main(*PAUSE) == 1
    assert failure(node) == f"{METRICS} is not active at baseline"
    assert probe.read_pause(RUN) == {}, "nothing is scheduled"
    node.collector("100", {ENDPOINT: PROXY})
    proxy_up(node, reads=1)
    node.service_state = "failed"
    assert main(*PAUSE) == 1
    assert failure(node).startswith("pause waiter unit did not start: "), failure(node)
    assert probe.read_pause(RUN)["target_reads"] == 3, "the record names the pause"


def test_pause_wait_rides_out_status_errors_and_needs_its_resume_deadman(
    node: EnvNode,
) -> None:
    installed("probe.py")
    answers = iter(
        [
            ConnectionRefusedError(111, "Connection refused"),
            (200, '{"armed": true, "reads": 3}'),
            ConnectionResetError(104, "Connection reset by peer"),
        ]
    )
    node.http[STATUS_URL] = lambda: next(answers)
    node.timers_fail = True
    waiter = ["pause-wait", "--run-id", RUN, "--after-reads", "3", "--seconds", "120"]
    assert main(*waiter) == 1
    assert failure(node) == "collector resume deadman timer is not active"
    assert node.clock.sleeps == [probe.WAIT_POLL_SECONDS, probe.PAUSE_SETTLE_SECONDS]
    assert not any("--signal=SIGSTOP" in c for c in node.commands), (
        "never stopped without a resume deadman"
    )


def test_resume_collector_leaves_the_resume_deadman_that_runs_it(node: EnvNode) -> None:
    deadman = probe.resume_deadman_unit(RUN)
    node.unit(f"{deadman}.service", MainPID=str(os.getpid()))
    node.unit(f"{deadman}.timer")
    node.signal("100", "T")
    installed()
    probe.write_pause(RUN, {"run_id": RUN, "target_reads": 3})
    assert main("resume-collector", "--run-id", RUN) == 0
    receipt = node.emitted[-1]
    assert receipt["stopped"] is False and "resumed_at" in receipt["pause"]
    stops = [c for c in node.commands if c[:2] == ["systemctl", "stop"]]
    assert ["systemctl", "stop", f"{deadman}.timer"] in stops
    assert ["systemctl", "stop", f"{deadman}.service"] not in stops, (
        "the deadman running this resume is not stopped under itself"
    )


# --------------------------------------------------------------------------- #
# restart / restore / stop
# --------------------------------------------------------------------------- #
def test_restart_collector_refuses_a_collector_that_stays_down(node: EnvNode) -> None:
    repointed(node)
    node.restarts.append({"state": "failed"})
    assert main("restart-collector", "--run-id", RUN) == 1
    assert failure(node) == f"{METRICS} is not active after the restart"


@pytest.mark.parametrize("record", ["{broken", "{}"], ids=["not-json", "no-endpoint"])
def test_restore_collector_falls_back_to_the_default_upstream(
    node: EnvNode, record: str
) -> None:
    repointed(node)
    (probe.run_dir(RUN) / "repoint.json").write_text(record)
    assert main("restore-collector", "--run-id", RUN) == 0
    receipt = node.emitted[-1]
    assert receipt["original_endpoint"] == probe.DEFAULT_UPSTREAM
    assert receipt["restored_endpoint"] == REAL and receipt["restarted"] is True
    assert receipt["dropin_removed"] is True and receipt["override_removed"] is True


def test_restore_collector_on_a_clean_node_run_by_its_deadman(node: EnvNode) -> None:
    deadman = probe.deadman_unit(RUN)
    node.unit(f"{deadman}.service", MainPID=str(os.getpid()))
    assert main("restore-collector", "--run-id", RUN) == 0
    receipt = node.emitted[-1]
    assert receipt["restarted"] is False, "a clean unit is never restarted"
    assert receipt["original_endpoint"] == probe.DEFAULT_UPSTREAM
    assert (receipt["dropin_removed"], receipt["override_removed"]) == (False, False)
    assert restarts(node) == []
    stops = [c for c in node.commands if c[:2] == ["systemctl", "stop"]]
    assert stops == [], "the deadman restoring is never stopped under itself"
    assert node.commands[-1] == [
        "systemctl",
        "reset-failed",
        f"{deadman}.service",
        f"{deadman}.timer",
    ]


@pytest.mark.parametrize(
    ("restart", "message"),
    [
        ({"state": "failed"}, f"{METRICS} is not active after the restore"),
        (NO_ENDPOINT, "the restored collector did not exec within the settle window"),
    ],
    ids=["down", "no-endpoint"],
)
def test_restore_collector_refuses_a_collector_it_cannot_prove_restored(
    node: EnvNode, restart: dict[str, Any], message: str
) -> None:
    repointed(node)
    node.unit(f"{probe.deadman_unit(RUN)}.timer")
    node.restarts.append(restart)
    assert main("restore-collector", "--run-id", RUN) == 1
    assert failure(node) == message
    assert f"{probe.deadman_unit(RUN)}.timer" in node.units, "the deadman stays armed"


def test_stop_proxy_stops_the_tool_and_removes_the_run_directory(node: EnvNode) -> None:
    proxy_up(node)
    root = installed("replay.jsonl")
    unit = f"{probe.proxy_unit(RUN)}.service"
    assert main("stop-proxy", "--run-id", RUN) == 0
    assert node.emitted[-1] == {"run_id": RUN, "unit": unit, "audit_was_present": True}
    assert not root.exists(), "the run directory is removed"
    assert unit not in node.units, "the proxy unit is stopped"
    assert main("stop-proxy", "--run-id", RUN) == 0
    assert node.emitted[-1]["audit_was_present"] is False, "nothing left to remove"
