"""One dropped connection costs a fake executor a retry, not its Pod.

A formal integrated round (50 clusters) failed on one transient drop: three
executor Pods each had a claim and a result post closed by the control plane
without a response; the lost results failed the Pods, ``backoffLimit: 0``
stopped twenty healthy siblings, and the runner waited out its timeout. The
fake executors now retry the way the product executor does
(``benchmark_transport_retry``), only fatal conditions fail a Pod, and the
runners stop waiting once the executor Job has failed.
"""

from __future__ import annotations

import inspect
import json
import socket
import subprocess
import sys
import threading
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib import error

import pytest

from gpu_fault.cluster_executor import regional_client
from gpu_fault.cluster_executor.executor import ClusterActionExecutor
from scripts.perf import benchmark_correlated_action_scenario as scenario
from scripts.perf import benchmark_regional_action_executor as executor
from scripts.perf import benchmark_transport_retry as transport
from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_executor_support as support
from scripts.perf import regional_integrated_workflow_capacity_suite as integrated
from tests.regional._perf_caller_support import CapacityWire, action_args


def dropped() -> RemoteDisconnected:
    return RemoteDisconnected("Remote end closed connection without response")


def refused(code: int) -> error.HTTPError:
    return error.HTTPError("https://control.example", code, "refused", {}, None)  # type: ignore[arg-type]


def test_the_retry_policy_is_the_product_executors() -> None:
    defaults = inspect.signature(ClusterActionExecutor.__init__).parameters

    assert (
        transport.CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS
        == regional_client.CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS
    ), "the client's one repeat of a claim closed without a response"
    assert transport.POLL_SECONDS == defaults["poll_seconds"].default
    assert (
        transport.CLAIM_BACKOFF_MAX_SECONDS
        == defaults["claim_backoff_max_seconds"].default
    )
    assert [transport.claim_retry_seconds(n) for n in range(1, 9)] == [
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        60.0,
        60.0,
    ], "the client's repeat, then the loop's doubling backoff up to its cap"
    assert [transport.report_backoff_seconds(n, lambda: 0.0) for n in (1, 2, 3)] == [
        0.5,
        1.0,
        1.0,
    ]
    assert transport.report_backoff_seconds(1, lambda: 1.0) == 0.75, "50 % jitter"


@pytest.mark.parametrize(
    ("failure", "repeat"),
    [
        (dropped(), True),
        (ConnectionResetError(104, "reset"), True),
        (error.URLError(ConnectionRefusedError(111, "refused")), True),
        (TimeoutError("read timed out"), True),
        (socket.timeout("timed out"), True),
        (IncompleteRead(b""), True),
        (refused(503), True),
        (refused(429), True),
        (refused(409), False),
        (refused(403), False),
        (json.JSONDecodeError("bad", "{", 0), False),
    ],
    ids=lambda value: type(value).__name__ if not isinstance(value, bool) else "",
)
def test_only_a_failure_without_a_verdict_is_repeated(
    failure: BaseException, repeat: bool
) -> None:
    assert transport.retryable(failure) is repeat, transport.category(failure)


def test_a_result_is_repeated_with_its_body_until_it_lands_or_runs_out() -> None:
    sleeps: list[float] = []
    attempts: list[int] = []

    def flaky() -> dict[str, str]:
        attempts.append(1)
        if len(attempts) == 1:
            raise dropped()
        return {"status": "SUCCEEDED"}

    answer, retried = transport.deliver(flaky, sleep=sleeps.append, jitter=lambda: 0)

    assert (answer, retried, sleeps) == (
        {"status": "SUCCEEDED"},
        ["RemoteDisconnected"],
        [0.5],
    )

    def lost() -> None:
        raise dropped()

    with pytest.raises(transport.Undelivered) as undelivered:
        transport.deliver(lost, sleep=sleeps.append, jitter=lambda: 0)
    assert undelivered.value.retried == ["RemoteDisconnected"] * 2, (
        "three attempts are two retries"
    )
    assert undelivered.value.category == "RemoteDisconnected"

    def stale_lease() -> None:
        raise refused(409)

    sleeps.clear()
    with pytest.raises(transport.Undelivered) as rejected:
        transport.deliver(stale_lease, sleep=sleeps.append)
    assert (rejected.value.retried, sleeps) == ([], []), "a refusal is final"


def test_claims_are_retried_until_refused_or_failing_for_a_whole_lease() -> None:
    now = [0.0]
    claims = transport.ClaimRetries(persistent_after_seconds=120, clock=lambda: now[0])

    assert claims.failed(dropped(), elapsed=1.0) == 1.0
    now[0] = 1.0
    assert claims.failed(dropped(), elapsed=2.0) == 2.0
    claims.succeeded()
    now[0] = 50.0
    assert claims.failed(dropped(), elapsed=50.0) == 1.0, "a success resets it"
    now[0] = 170.0
    assert claims.failed(dropped(), elapsed=170.0) is None, "persistent: fatal"
    assert claims.failed(refused(403), elapsed=171.0) is None, "refused: fatal"
    assert claims.retries == 3
    assert claims.retry_counts == {"RemoteDisconnected": 3}
    assert [item["elapsed_seconds"] for item in claims.retry_samples] == [1, 2, 50]


class FlakyControlPlane:
    """Claims, renewals and results; the drops are posts closed without a
    response -- a dropped result was applied before its answer was lost."""

    def __init__(
        self,
        commands: list[dict[str, Any]],
        *,
        claim_drops: int = 0,
        result_drops: int = 0,
        claim_refusal: int | None = None,
    ) -> None:
        self.commands = commands
        self.claim_drops = claim_drops
        self.result_drops = result_drops
        self.claim_refusal = claim_refusal
        self.claims = 0
        self.results: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if path.endswith("/claim"):
                self.claims += 1
                if self.claim_refusal is not None:
                    raise refused(self.claim_refusal)
                if self.claim_drops:
                    self.claim_drops -= 1
                    raise dropped()
                batch = self.commands[: payload["max_commands"]]
                del self.commands[: len(batch)]
                return {"commands": batch}
            if path.endswith("/renew"):
                return {}
            self.results.append((path, payload))
            if self.result_drops:
                self.result_drops -= 1
                raise dropped()
            return {"status": "SUCCEEDED"}


def command(index: int) -> dict[str, Any]:
    return {
        "command_id": f"command-{index}",
        "lease_token": f"lease-{index}",
        "step": {"operation": "MARK_UNSCHEDULABLE", "node_ids": ["node-a"]},
    }


def claim_loop(
    control: FlakyControlPlane,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected: int,
    max_seconds: float = 30.0,
) -> tuple[executor.ClaimTally, list[float]]:
    monkeypatch.setenv("EXECUTOR_PROTOCOL_VERSION", "2")
    monkeypatch.setenv("EXECUTOR_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("EXECUTOR_COMPATIBILITY_DIGEST", "b" * 64)
    monkeypatch.setattr(transport, "CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(transport, "POLL_SECONDS", 0.01)
    retry_sleeps: list[float] = []
    state = executor.SimulationState(
        client=control,  # type: ignore[arg-type]
        executor_id="action-executor-000",
        operation_delays=dict(executor.DELAYS),
        delay_scale=0.0,
        lease_seconds=120,
        renewal_interval=3600.0,
        inject_renew_failure_pending=False,
        retry_sleep=retry_sleeps.append,
    )
    tally = executor.run_claim_loop(
        state,
        workers=2,
        expected=expected,
        max_seconds=max_seconds,
        idle_exit_seconds=0,
    )
    return tally, retry_sleeps


def fatal(tally: executor.ClaimTally, *, expected: int) -> list[str]:
    return executor.fatal_conditions(
        {
            "expected_commands": expected,
            "completed_commands": len(tally.completed_ids),
            "duplicate_claims": tally.duplicate_claims,
            "claim_errors": tally.claim_errors,
            "result_errors": tally.result_errors,
            "max_concurrent_commands": 2,
            "long_commands": 0,
            "renewals": 0,
            "injected_renewal_failures": 0,
            "renewal_errors": 0,
        },
        min_concurrent_commands=1,
        inject_renew_failure=False,
    )


def test_a_dropped_claim_and_a_dropped_result_are_retried_and_the_pod_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = FlakyControlPlane([command(0), command(1)], claim_drops=1, result_drops=1)

    tally, retry_sleeps = claim_loop(control, monkeypatch, expected=2)

    assert tally.completed_ids == {"command-0", "command-1"}
    assert (tally.claim_errors, tally.result_errors) == (0, 0), "nothing is lost"
    assert tally.claims.retries == 1
    assert tally.claims.retry_counts == {"RemoteDisconnected": 1}
    assert tally.result_retries == 1
    assert tally.result_retry_counts == {"RemoteDisconnected": 1}
    assert len(retry_sleeps) == 1, "the product's report backoff before the repeat"
    posted = [payload for _path, payload in control.results]
    assert len(posted) == 3, "two results, one of them twice"
    repeated = [item for item in posted if posted.count(item) == 2]
    assert repeated and repeated[0]["lease_token"].startswith("lease-"), (
        "the repeat carries the same body: its lease token is the idempotency key"
    )
    assert fatal(tally, expected=2) == [], "retries alone never fail the Pod"


def test_a_result_still_undelivered_after_its_attempts_is_lost_and_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = FlakyControlPlane([command(0)], result_drops=3)

    tally, _sleeps = claim_loop(control, monkeypatch, expected=1, max_seconds=1.0)

    assert tally.result_errors == 1
    assert tally.result_error_counts == {"RemoteDisconnected": 1}
    assert tally.result_retries == 2
    assert fatal(tally, expected=1) == [
        "completed 0 of 1 commands",
        "results lost: undelivered after their attempts or refused",
    ]


def test_a_refused_claim_stops_claiming_and_fails_the_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = FlakyControlPlane([command(0)], claim_refusal=403)

    tally, _sleeps = claim_loop(control, monkeypatch, expected=1)

    assert control.claims == 1, "a refusal is not repeated"
    assert (tally.claim_errors, tally.claims.retries) == (1, 0)
    assert tally.claim_error_counts == {"HTTPError:403": 1}
    assert "a claim was refused or kept failing for a whole lease" in fatal(
        tally, expected=1
    )


def test_the_scenario_retries_results_and_ownership_reads_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = scenario.ScenarioState(executor_id="correlated-action-000")
    control = FlakyControlPlane([], result_drops=1)

    retried = scenario.command_result(
        control,  # type: ignore[arg-type]
        command(7) | {"step": {"operation": "RESET_GPU", "node_ids": ["node-a"]}},
        failed=False,
        executor_id=state.executor_id,
        sleep=lambda _seconds: None,
    )
    state.retried("result", retried)
    state.ownership_failed(error.URLError(ConnectionResetError(104, "reset")))
    state.ownership_failed(refused(403))

    assert retried == ["RemoteDisconnected"]
    assert len(control.results) == 2, "the result landed on its repeat"
    assert (state.result_retries, state.ownership_retries) == (1, 1)
    assert state.ownership_errors == 1, "a refused read stays fatal"
    assert state.retry_counts == {
        "result:RemoteDisconnected": 1,
        "ownership:URLError:ConnectionResetError": 1,
    }


@pytest.mark.parametrize(
    "script",
    [
        "benchmark_regional_action_executor.py",
        "benchmark_correlated_action_scenario.py",
    ],
)
def test_the_shipped_scripts_import_in_a_bare_interpreter_from_one_directory(
    tmp_path: Path, script: str
) -> None:
    """What a ``python:3.12-slim`` Pod does with the script ConfigMap mounted
    at ``/scripts``: no repository, no site-packages, one directory."""

    shipped = support.executor_scripts(script)
    for name, text in shipped.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    module = script.removesuffix(".py")
    probe = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
        f"import {module} as shipped; print(shipped.transport.__file__)"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )

    assert sorted(shipped) == sorted({script, "benchmark_transport_retry.py"})
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == str(tmp_path / "benchmark_transport_retry.py"), (
        "the script imports the retries shipped beside it"
    )


def job(**status: Any) -> str:
    return json.dumps({"kind": "Job", "status": status})


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        (
            job(
                failed=1,
                conditions=[
                    {
                        "type": "Failed",
                        "status": "True",
                        "reason": "BackoffLimitExceeded",
                    }
                ],
            ),
            "BackoffLimitExceeded",
        ),
        (job(failed=3, active=20), "3 executor Pod(s) failed"),
        (job(active=50, succeeded=0), None),
        ("", None),
        ("not json", None),
    ],
    ids=["failed-condition", "failed-pods", "running", "missing", "unreadable"],
)
def test_an_executor_job_failure_is_read_from_the_job(
    answer: str, failure: str | None
) -> None:
    calls: list[tuple[str, ...]] = []

    def client(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        return answer

    assert support.executor_job_failure("executors", client) == failure
    assert calls == [("get", "job", "executors", "-o", "json")]


def test_the_integrated_runner_stops_waiting_once_its_executor_job_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integrated,
        "workflow_audit",
        lambda _run_id: {
            "workflow_count": 200,
            "terminal_workflow_count": 120,
            "terminal_command_count": 1000,
            "command_count": 1100,
        },
    )
    monkeypatch.setattr(
        integrated,
        "dataplane",
        lambda *_args, **_kwargs: job(
            failed=1,
            conditions=[
                {"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}
            ],
        ),
    )

    audit, history = integrated.wait_for_workflows(
        "run-a", expected_workflows=200, timeout_seconds=1800
    )

    assert audit["executor_job_failure"] == "BackoffLimitExceeded"
    assert len(history) == 1, "no executor is left to finish the rest"


def test_the_action_runner_stops_waiting_once_its_executor_job_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    def running(_run_id: str) -> dict[str, Any]:
        executors = wire.objects[f"job/{action.JOB_NAME}"]
        executors["status"] = {
            "failed": 1,
            "conditions": [
                {"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}
            ],
        }
        return {
            "workflows": {"RUNNING": 1},
            "commands": {"LEASED": 1},
            "budget_holders_max": {"region": 1, "cluster": 1, "class": 1, "node": 1},
        }

    monkeypatch.setattr(action, "database_snapshot", running)

    assert action.main(action_args(tmp_path)) == 1

    history = json.loads((tmp_path / "history.json").read_text())
    assert [item.get("executor_job_failure") for item in history] == [
        "BackoffLimitExceeded"
    ], "it stops at the first poll that sees the failed Job, not at the timeout"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["job_status"] == "Incomplete"


def test_the_executor_summary_reports_retries_apart_from_failures() -> None:
    documents = [
        {
            "executor_id": "action-executor-000",
            "claim_retries": 1,
            "claim_retry_counts": {"RemoteDisconnected": 1},
            "result_retries": 1,
            "result_retry_counts": {"RemoteDisconnected": 1},
            "fatal": [],
        },
        {
            "executor_id": "action-executor-001",
            "result_errors": 1,
            "result_error_counts": {"HTTPError:409": 1},
            "fatal": ["results lost: undelivered after their attempts or refused"],
        },
    ]

    aggregated = action.aggregate_executor_documents(documents)

    assert aggregated["claim_retries"] == 1
    assert aggregated["result_retry_counts"] == {"RemoteDisconnected": 1}
    assert aggregated["result_error_counts"] == {"HTTPError:409": 1}
    assert aggregated["result_errors"] == 1
    assert aggregated["fatal"] == {
        "action-executor-001": [
            "results lost: undelivered after their attempts or refused"
        ]
    }, "only a Pod with a fatal condition is named"
