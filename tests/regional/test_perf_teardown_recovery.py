"""Every capacity suite cleans up through the bounded, retried teardown.

Live 2026-10-13: the action suite ran one purge outside the retried teardown;
it raised once ("capacity cleanup left run-owned data": a writer still touched
the run's rows, the transaction deleted nothing), the teardown never ran and
perf-cap-000..031 stayed in the live registry until an operator re-ran
``regional_capacity_suite.py teardown`` from the run's receipts an hour later.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_recovery as recovery
from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as capacity
from tests.regional._perf_caller_support import (
    TEARDOWN_TAIL,
    CapacityWire,
    action_args,
    capacity_args,
)

SUITES: dict[str, Callable[[Path], int]] = {
    "action": lambda artifacts: action.main(action_args(artifacts)),
    "capacity": lambda artifacts: capacity.main(capacity_args("all", artifacts)),
}
RECEIPTS = (
    "run.json",
    "registry-registration-intent.json",
    "registry-token-proof.json",
    "capacity-resources.json",
)
SUITE_SCRIPT = str(Path(capacity.__file__).resolve())


@pytest.mark.parametrize("suite", sorted(SUITES))
def test_a_purge_that_meets_a_writer_once_is_retried_and_deregisters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.cleanup_races = 1

    assert SUITES[suite](tmp_path) == 0, "a purge that succeeds on retry is a clean run"

    assert wire.events.count("data-cleanup") == 2, "the refused purge is retried"
    assert wire.sleeps == [recovery.TEARDOWN_RETRY_DELAYS_SECONDS[0]], (
        "the retry waits the first backoff"
    )
    assert wire.events[-4:] == TEARDOWN_TAIL, (
        "deregistration still follows the successful data proof"
    )
    assert not wire.registered() and wire.objects == {}, (
        "the retried teardown removes the registration and the token"
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "ok" and set(status) == {"status", "updated_at"}, (
        "a teardown that finished on retry leaves no recovery note"
    )


@pytest.mark.parametrize("suite", sorted(SUITES))
def test_a_clean_teardown_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    assert SUITES[suite](tmp_path) == 0, "the fake complete run should finish"

    assert wire.events.count("data-cleanup") == 1, "one purge, inside the teardown"
    assert wire.sleeps == [], "a clean teardown never waits"
    assert wire.events[-4:] == TEARDOWN_TAIL, "data, registry, then token"
    status = json.loads((tmp_path / "status.json").read_text())
    assert set(status) == {"status", "updated_at"} and status["status"] == "ok", (
        "the happy path status carries no cleanup fields"
    )


@pytest.mark.parametrize("suite", sorted(SUITES))
def test_a_purge_that_keeps_failing_keeps_the_receipts_and_logs_the_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    suite: str,
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.cleanup_races = len(recovery.TEARDOWN_RETRY_DELAYS_SECONDS) + 1

    with pytest.raises(RuntimeError, match="failed after 4 attempts: capacity cleanup"):
        SUITES[suite](tmp_path)

    assert wire.sleeps == list(recovery.TEARDOWN_RETRY_DELAYS_SECONDS), (
        "every bounded retry waits its backoff"
    )
    assert wire.registered() and "revoke" not in wire.events, (
        "no attempt may deregister while the run's data is present"
    )
    assert set(wire.objects) == {f"secret/{registry.TOKEN_SECRET}"}, (
        "the token stays with the registration"
    )
    assert all((tmp_path / name).is_file() for name in RECEIPTS), (
        "the receipts stay for the recovery"
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "aborted", "a failed teardown aborts the run"
    assert status["cleanup_error"].startswith(
        "RuntimeError: capacity teardown failed after 4 attempts"
    ), "status.json names the cleanup error"
    command = status["recovery_command"]
    assert f"recovery command: {command}\n" in capsys.readouterr().out, (
        "the one-line recovery command is logged"
    )
    words = shlex.split(command)
    start = words.index("teardown")
    assert words[start - 2 :] == [
        sys.executable,
        SUITE_SCRIPT,
        "teardown",
        "--suite-id",
        "run-a",
        "--run-dir",
        str(tmp_path),
    ], "the command adopts exactly this run's receipts"

    assert capacity.main(words[start:]) == 0, (
        "once the writer is gone, the logged command finishes the teardown"
    )
    assert not wire.registered() and wire.objects == {}, (
        "the recovery removes the registration and the token"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "ok", (
        "the recovered teardown completes the run's cleanup"
    )


@pytest.mark.parametrize("scope", ["isolated", "live"])
def test_recovery_command_pins_the_runs_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    monkeypatch.setattr(registry, "DATAPLANE_CONTEXT", "perf context")
    run_dir = tmp_path / "run dir"

    command = recovery.teardown_recovery_command(
        run_id="run-a", artifacts=run_dir, scope=scope
    )

    assert "\n" not in command, "the recovery is one line"
    words = shlex.split(command)
    assert words[:6] == [
        f"GPU_FAULT_CONTROL_KUBECONFIG={registry.CONTROL_KUBECONFIG}",
        "GPU_FAULT_DATAPLANE_CONTEXT=perf context",
        f"GPU_FAULT_PERF_CONTROL_NAMESPACE={registry.CONTROL_NAMESPACE}",
        f"GPU_FAULT_PERF_DATAPLANE_NAMESPACE={registry.NAMESPACE}",
        f"GPU_FAULT_PERF_REGISTRY_SECRET={registry.REGISTRY_SECRET}",
        f"GPU_FAULT_PERF_CONNECTION_SECRET={registry.CONNECTION_SECRET}",
    ], "the recovery must reach the run's own control plane, context and Secrets"
    consent = [
        "--allow-live-registry",
        "--confirm-live-registry",
        registry.LIVE_REGISTRY_CONFIRMATION,
    ]
    assert words[6:] == [
        sys.executable,
        SUITE_SCRIPT,
        "teardown",
        "--suite-id",
        "run-a",
        "--run-dir",
        str(run_dir),
        *(consent if scope == "live" else []),
    ], "only a live-registry run's recovery carries the live-registry consent"


def test_teardown_backs_off_between_its_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sleeps: list[float] = []
    attempts: list[dict[str, object]] = []
    error = RuntimeError(
        "command failed (1): kubectl exec\n"
        "RuntimeError: capacity cleanup left run-owned data\n"
    )

    def teardown_once(**kwargs: object) -> None:
        attempts.append(kwargs)
        raise error

    monkeypatch.setattr(capacity, "validate_registry_target", lambda **_kw: "live")
    monkeypatch.setattr(capacity, "_teardown_once", teardown_once)
    monkeypatch.setattr(capacity.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="failed after 4 attempts") as caught:
        capacity.teardown(purge=True, deregister_clusters=True, run_id="run-a")

    assert caught.value.__cause__ is error, "the last attempt's error is the cause"
    assert len(attempts) == 4 and all(item["scope"] == "live" for item in attempts), (
        "every attempt repeats the whole scoped teardown"
    )
    assert sleeps == [15, 45, 90], "the waits grow between the attempts"
    assert (
        "teardown attempt 1/4 failed (RuntimeError: command failed (1): kubectl "
        "exec RuntimeError: capacity cleanup left run-owned data); retrying in 15 s"
    ) in capsys.readouterr().out, "each retry logs its reason on one line"
