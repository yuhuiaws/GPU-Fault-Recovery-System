"""The cluster executor says at INFO what it claims, finishes and that it polls.

Live (ISO-001 on a mixed site): both clusters' executors had logged exactly one
line since start -- the metrics listener -- and executed every workflow in
silence; the acceptance runner refused the empty log window as missing
evidence, and an operator reading ``kubectl logs`` saw nothing either. The
loop now logs one line per claimed command (never the lease token), one per
finished command, and a heartbeat at most every ``HEARTBEAT_LOG_SECONDS`` so an
idle executor still shows it is alive.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gpu_fault.cluster_executor import executor as executor_module
from tests.execution.test_cluster_executor_batching import (
    FakeClient,
    FakeNodeAdapter,
    all_succeed,
    compound_command,
)
from tests.regional.test_cov95_runtime_executor_guards import make_executor


def _info_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "gpu_fault.cluster_executor"
        and record.levelno == logging.INFO
    ]


def test_claims_and_outcomes_are_logged_at_info_without_the_lease_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    command = compound_command()
    client = FakeClient(command)
    runtime = make_executor(tmp_path, client, [FakeNodeAdapter(all_succeed())])
    caplog.set_level(logging.INFO, logger="gpu_fault.cluster_executor")

    assert runtime.run_once() == 1

    lines = _info_lines(caplog)
    claimed = [line for line in lines if "claimed command=" in line]
    finished = [line for line in lines if "finished command=" in line]
    assert len(claimed) == 1 and command.command_id in claimed[0], lines
    assert f"cluster={command.cluster_id}" in claimed[0]
    assert f"operation={command.step.operation.value}" in claimed[0]
    assert f"workflow={command.workflow.request_id}" in claimed[0]
    assert "generation=3" in claimed[0]
    assert len(finished) == 1 and command.command_id in finished[0]
    assert "status=" in finished[0]
    reported = client.reported()
    assert reported.lease_token is not None
    assert reported.lease_token not in caplog.text, (
        "the lease token never reaches a log"
    )


def test_the_heartbeat_is_rate_limited_and_survives_idle_cycles(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = {"now": 1000.0}
    runtime = make_executor(
        tmp_path,
        FakeClient(),
        [FakeNodeAdapter(all_succeed())],
        clock=lambda: clock["now"],
    )
    caplog.set_level(logging.INFO, logger="gpu_fault.cluster_executor")

    assert runtime.run_once() == 0
    clock["now"] += 5
    assert runtime.run_once() == 0
    heartbeats = [line for line in _info_lines(caplog) if "heartbeat" in line]
    assert len(heartbeats) == 1, "two polls inside the interval: one heartbeat"
    assert "executor=unit-executor" in heartbeats[0]
    assert "claimed_now=0" in heartbeats[0]

    clock["now"] += executor_module.HEARTBEAT_LOG_SECONDS
    assert runtime.run_once() == 0
    heartbeats = [line for line in _info_lines(caplog) if "heartbeat" in line]
    assert len(heartbeats) == 2, "the interval passed: the idle executor speaks again"
