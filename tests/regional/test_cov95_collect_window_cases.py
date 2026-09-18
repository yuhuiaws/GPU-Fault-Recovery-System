"""Run full collector window cases with real verdicts and fake observations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collect018_rejected_event as c018
from scripts.e2e.regional import run_collect019_nvidia_smi_hang as c019
from scripts.e2e.regional import run_collect020_gpu_identity as c020
from tests.regional import _cov95_collect_windows as window_support
from tests.regional._cov95_collect_net import (  # noqa: F401
    StopLoop,
    no_external_effects,
)
from tests.regional._cov95_collect_windows import WindowHost

window_case = window_support.window_case


def execute(host: WindowHost, case_dir: Path) -> Any:
    return host.module.execute(
        host.settings,
        host,
        case_dir,
        1,
        datetime.now(timezone.utc) + timedelta(hours=1),
    )


@pytest.mark.parametrize(
    "fallback,pending", [(False, False), (True, False), (False, True)]
)
def test_window_case_completes_and_restores_after_real_predicate_convergence(
    window_case: Any, tmp_path: Path, fallback: bool, pending: bool
) -> None:
    host = window_case
    host.fallback, host.pending = fallback, pending
    result = execute(host, tmp_path)
    assert result["verdict"] == "PASS", result.get("errors", result)
    if host.module is c018:
        assert host.waits == ["rejection", "unparsed-finding", "recovery"]
        assert host.recovered is True
        assert result["unparsed_evidence_record_ids"] == ["raw-a"]
    elif host.module is c019:
        assert host.waits == ["erroring", "recovery"]
        assert not host.window, "hang window must close before recovery sampling"
        assert result["errors_seen_during_window"] == sorted(
            ["nvidia-smi timed out", c019.verdicts.BREAKER_TEXT]
        )
    else:
        assert host.waits == ["finding", "workflow"]
        assert host.restored is True
        assert not host.window, "identity shadow must close before diagnostics"
        assert result["inventory_evidence_record_ids"] == ["inventory-a"]


@pytest.mark.parametrize("failure", ["exception", "abort"])
def test_open_or_post_failure_does_not_continue_later_phases(
    window_case: Any, tmp_path: Path, failure: str
) -> None:
    host = window_case
    command = "post-rejected-event" if host.module is c018 else "open-window"
    error = RuntimeError("fixture failure") if failure == "exception" else StopLoop()
    host.failures[command] = error
    with pytest.raises(type(error)):
        execute(host, tmp_path)
    assert host.waits == [], "failed post/open must not start later waits"
    if host.module is not c018:
        assert any(call[0] == "close-window" for call in host.calls), (
            "opening ACK failure needs rollback"
        )


@pytest.mark.parametrize(
    "module,problem",
    [
        (c018, "record-id"),
        (c018, "rejection"),
        (c018, "inventory"),
        (c019, "timing"),
        (c020, "precondition"),
    ],
)
def test_invalid_case_prerequisites_stop_before_unsafe_next_phase(
    module: Any, problem: str, monkeypatch: Any, tmp_path: Path
) -> None:
    host = WindowHost(module)
    monkeypatch.setattr(module, "time", host.clock)
    host.problem = problem
    if problem == "inventory":
        host.inventory = []
    elif problem == "timing":
        host.env["GPU_FAULT_HOST_INTERVAL_SECONDS"] = "600"
    elif problem == "precondition":
        host.env["GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES"] = "1"
    with pytest.raises(RuntimeError):
        execute(host, tmp_path)
    assert not any(call[0] == "write-kmsg" for call in host.calls), (
        "invalid rejection or baseline cannot authorize kmsg"
    )
    if module is not c018:
        assert not any(call[0] == "open-window" for call in host.calls), (
            "invalid timing or inventory must not open a host window"
        )


@pytest.mark.parametrize("problem", ["window", "service", "status", "silence", "close"])
def test_host_hang_verdict_fails_unproven_observations_but_closes_window(
    monkeypatch: Any, tmp_path: Path, problem: str
) -> None:
    host = WindowHost(c019)
    host.problem = problem
    monkeypatch.setattr(c019, "time", host.clock)
    result = execute(host, tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["errors"], "unproven host observation needs a failing verdict"
    assert not host.window, "failed observation must still close the hang window"


@pytest.mark.parametrize(
    "problem", ["finding", "close", "reboot", "reboot-restore", "already-restored"]
)
@pytest.mark.parametrize("window_case", [c020], indirect=True, ids=["collect020"])
def test_identity_case_hard_stop_and_validated_restore(
    window_case: Any, tmp_path: Path, problem: str
) -> None:
    host = window_case
    host.problem = problem
    result = execute(host, tmp_path)
    assert result["verdict"] == ("PASS" if problem == "already-restored" else "FAIL")
    assert not host.window, "diagnostic or hard stop must follow shadow removal"
    if problem in {"reboot", "reboot-restore"}:
        assert "workflow" not in host.waits, (
            "reset/reboot escalation must stop the normal wait"
        )
    if problem == "reboot-restore":
        assert result["stages"]["restore"] == ["RuntimeError: restore unavailable"]
    if problem in {"finding", "already-restored"}:
        assert not any(call[0] == "restore-workflow" for call in host.calls), (
            "no restore without current ownership"
        )


def test_expected_count_lost_between_precondition_and_execution_is_refused(
    tmp_path: Path, monkeypatch: Any
) -> None:
    host = WindowHost(c020)
    values = iter([8, None])
    monkeypatch.setattr(c020.verdicts, "expected_count_for", lambda _: next(values))
    with pytest.raises(RuntimeError, match="no expected GPU count"):
        execute(host, tmp_path)
    assert host.calls == []


@pytest.mark.parametrize("aligned", [False, True])
def test_rejection_waits_for_new_heartbeat_before_posting(
    tmp_path: Path, aligned: bool
) -> None:
    now = datetime.now(timezone.utc)
    before = [
        {
            "collector": "NVIDIA_KERNEL",
            "batch_id": "kernel-health-before",
            "observed_at": (now - timedelta(seconds=240)).isoformat(),
        }
    ]
    reads = iter([[], before, [{**before[0], "observed_at": now.isoformat()}]])
    observed = []

    def wait(accept: Any, **kwargs: Any) -> Any:
        observed.append(kwargs["name"])
        assert accept() is None
        assert accept() is None
        result = accept()
        return result if aligned else None

    fixture = SimpleNamespace(collector_statuses=lambda: next(reads), wait_until=wait)
    if aligned:
        assert (
            c018.align_kernel_heartbeat(fixture, before, case_dir=tmp_path)[
                "heartbeat_aligned"
            ]
            is True
        )
    else:
        with pytest.raises(RuntimeError, match="unproven"):
            c018.align_kernel_heartbeat(fixture, before, case_dir=tmp_path)
    assert observed == ["heartbeat-align"]


@pytest.mark.parametrize("module", [c018, c019, c020])
def test_main_delegates_unchanged_case_callbacks(module: Any, monkeypatch: Any) -> None:
    calls = []
    monkeypatch.setattr(
        module, "run_window_case", lambda **kwargs: calls.append(kwargs) or 9
    )
    assert module.main() == 9
    assert calls[0]["execute"] is module.execute
    assert (
        calls[0]["plan_details"](
            SimpleNamespace(node="node-a"), {"predecessor": {"valid": True}}
        )["target_node"]
        == "node-a"
    )
