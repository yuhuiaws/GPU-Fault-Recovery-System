"""The regional executor's ``/metrics`` + ``/healthz`` endpoint (data-plane F9).

The executor kept a dozen counters and published them nowhere a collector could
read: they rode along inside the readiness breadcrumb, behind ``kubectl exec ...
cat``. F7 gave the Pod a scrape annotation, a port named ``metrics`` on :9111
and a keep rule for ``gpu_fault_cluster_executor_.+`` in the per-cluster ADOT
collector; until this endpoint listened, that target read ``up == 0``.

What is pinned here:

* every series matches the family contract and survives F7's keep rule, so a
  renamed metric fails a test instead of vanishing before AMP;
* the counters are the executor's own counters -- ``increment`` is their only
  writer, so a claim/result cycle, a timeout, a fence hold, a report retry, a
  compound command and a transport error move the same numbers the breadcrumb
  carries;
* every number the breadcrumb carries is a series -- the ``increment``
  counters, the claim loop's degraded-cycle streak and the spare sweep's
  total -- so a counter added to the breadcrumb alone (the merge ported five
  that way) fails the contract test instead of staying ``kubectl exec``-only;
* ``/healthz`` asks the exec probe's question of the exec probe's file: the
  loop breadcrumb fresher than 300 s, nothing about the control plane;
* port 0 disables the server and a port that cannot be bound is one ERROR line;
  ``main`` runs the claim loop either way.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest
import yaml

from gpu_fault import dataplane_metrics as metrics_module
from gpu_fault.cluster_executor import (
    CLUSTER_EXECUTOR_METRICS_PREFIX,
    LIVENESS_STALE_AFTER_SECONDS,
    ClusterActionExecutor,
    ClusterExecutorMetrics,
    cluster_executor_metrics,
    executor_health,
    loop_breadcrumb_is_fresh,
    start_executor_metrics,
)
from gpu_fault.cluster_executor import bootstrap as BOOTSTRAP
from gpu_fault.cluster_executor.metrics import (
    COUNTER_SERIES,
    DIRECT_SERIES,
    GAUGE_SERIES,
)
from gpu_fault.dataplane_metrics import MetricsServer
from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator
from gpu_fault.spare_reservation_sweep import SpareReservationSweep
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import RemoteCommandStatus
from tests.execution.test_cluster_executor_batching import (
    FakeClient,
    FakeNodeAdapter,
    all_succeed,
    compound_command,
)
from tests.execution.test_cluster_executor_deadline import StuckAdapter
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    FenceRegistry,
    RecordingAdapter,
    remote_command,
)
from tests.execution.test_cluster_executor_report_retry import FlakyReportClient
from tests.hyperpod.test_spare_reservations import (
    NOW,
    FakeCore,
    FakeLifecycle,
    reserved_node,
    spare,
)

ROOT = Path(__file__).resolve().parents[2]
ADOT_DATAPLANE_MANIFEST = ROOT / "deploy" / "dataplane" / "adot-dataplane.yaml"
DATAPLANE_SCRAPE_JOB = "gpu-fault-dataplane"
METRIC_NAME = re.compile(r"^gpu_fault_cluster_executor_[a-z_]+$")
EXPECTED_METRICS = {
    "gpu_fault_cluster_executor_claims_total": "counter",
    "gpu_fault_cluster_executor_results_succeeded_total": "counter",
    "gpu_fault_cluster_executor_results_failed_total": "counter",
    "gpu_fault_cluster_executor_results_waiting_total": "counter",
    "gpu_fault_cluster_executor_execution_timeouts_total": "counter",
    "gpu_fault_cluster_executor_abandoned_lease_holds_total": "counter",
    "gpu_fault_cluster_executor_fleet_fence_holds_total": "counter",
    "gpu_fault_cluster_executor_lease_lost_total": "counter",
    "gpu_fault_cluster_executor_transport_retries_total": "counter",
    "gpu_fault_cluster_executor_report_failures_total": "counter",
    "gpu_fault_cluster_executor_unexpected_failures_total": "counter",
    "gpu_fault_cluster_executor_lease_renewal_failures_total": "counter",
    "gpu_fault_cluster_executor_results_withheld_total": "counter",
    "gpu_fault_cluster_executor_cancellations_observed_total": "counter",
    "gpu_fault_cluster_executor_barrier_unavailable_holds_total": "counter",
    "gpu_fault_cluster_executor_retryable_adapter_errors_total": "counter",
    "gpu_fault_cluster_executor_retryable_transport_errors_total": "counter",
    "gpu_fault_cluster_executor_batched_commands_total": "counter",
    "gpu_fault_cluster_executor_batched_steps_total": "counter",
    "gpu_fault_cluster_executor_batched_progress_failures_total": "counter",
    "gpu_fault_cluster_executor_spare_reservations_reclaimed_total": "counter",
    "gpu_fault_cluster_executor_in_flight_commands": "gauge",
    "gpu_fault_cluster_executor_consecutive_transport_degraded_cycles": "gauge",
    "gpu_fault_cluster_executor_claim_loop_alive": "gauge",
    "gpu_fault_cluster_executor_stuck_executions": "gauge",
    "gpu_fault_cluster_executor_last_claim_timestamp": "gauge",
    "gpu_fault_cluster_executor_last_loop_iteration_timestamp": "gauge",
}


@pytest.fixture(autouse=True)
def breadcrumbs_outside_shared_tmp(tmp_path, monkeypatch) -> None:
    """Both breadcrumbs default to fixed names in ``/tmp``; keep them here."""

    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        str(tmp_path / "claim-state.json"),
    )


def build_executor(
    client: Any, adapters: list[Any], tmp_path: Path, **overrides: Any
) -> ClusterActionExecutor:
    overrides.setdefault("liveness_state_path", str(tmp_path / "executor-loop-alive"))
    return ClusterActionExecutor(
        client,
        adapters,
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        **overrides,
    )


def samples(body: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, value = line.split(" ", 1)
        values[name] = int(float(value))
    return values


def executor_samples(executor: ClusterActionExecutor) -> dict[str, int]:
    return {
        name.removeprefix(CLUSTER_EXECUTOR_METRICS_PREFIX): value
        for name, value in samples(executor.metrics.family.render()).items()
    }


def get(url: str) -> tuple[int, str]:
    try:
        with urlopen(url, timeout=5) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as error:
        return error.code, error.read().decode("utf-8")


def dataplane_keep_rule() -> re.Pattern[str]:
    """F7's metric-name keep rule, read from the manifest it lives in."""

    documents = [
        document
        for document in yaml.safe_load_all(
            ADOT_DATAPLANE_MANIFEST.read_text(encoding="utf-8")
        )
        if isinstance(document, dict)
    ]
    config_map = next(d for d in documents if d.get("kind") == "ConfigMap")
    config = yaml.safe_load(config_map["data"]["collector.yaml"])
    job = next(
        job
        for job in config["receivers"]["prometheus"]["config"]["scrape_configs"]
        if job["job_name"] == DATAPLANE_SCRAPE_JOB
    )
    keeps = [
        rule["regex"]
        for rule in job["metric_relabel_configs"]
        if rule.get("action") == "keep" and rule.get("source_labels") == ["__name__"]
    ]
    assert len(keeps) == 1, keeps
    return re.compile(f"^(?:{keeps[0]})$")


# --------------------------------------------------------------------- contract


def test_metric_family_contract_names_help_type_and_keep_rule() -> None:
    """Every series is prefixed, lower-snake, typed, explained, and kept."""

    family = cluster_executor_metrics()
    body = family.render()
    lines = body.splitlines()
    names = tuple(samples(body))

    assert family.prefix == CLUSTER_EXECUTOR_METRICS_PREFIX
    assert set(names) == set(EXPECTED_METRICS), sorted(
        set(names) ^ set(EXPECTED_METRICS)
    )
    assert names == family.names(), "render order must follow the family"
    for name, metric_type in EXPECTED_METRICS.items():
        assert METRIC_NAME.match(name), f"{name} breaks the family naming contract"
        assert f"# TYPE {name} {metric_type}" in lines, f"{name} TYPE line missing"
        help_lines = [line for line in lines if line.startswith(f"# HELP {name} ")]
        assert len(help_lines) == 1, f"{name} needs exactly one HELP line"
        assert len(help_lines[0]) > len(f"# HELP {name} ") + 20, (
            f"{name} HELP must say what the number means, not repeat the name"
        )

    keep = dataplane_keep_rule()
    dropped = [name for name in names if not keep.match(name)]
    assert dropped == [], f"executor metrics dropped before AMP: {dropped}"


def test_a_fresh_executor_exports_every_series_at_zero(tmp_path) -> None:
    executor = build_executor(FakeExecutorClient(), [], tmp_path)

    assert isinstance(executor.metrics, ClusterExecutorMetrics), (
        "the executor must own its family so every layer increments one object"
    )
    assert executor_samples(executor) == dict.fromkeys(
        (
            name.removeprefix(CLUSTER_EXECUTOR_METRICS_PREFIX)
            for name in EXPECTED_METRICS
        ),
        0,
    )


def test_every_increment_driven_breadcrumb_counter_is_an_exported_series(
    tmp_path,
) -> None:
    """The breadcrumb and the scrape are two views of one set of numbers.

    A number that reaches the breadcrumb but has no series is ``kubectl
    exec``-only (the merge shipped five counters that way). ``increment``
    counters are mapped by ``COUNTER_SERIES``/``GAUGE_SERIES``; the two values
    the executor sets outside ``increment`` (the claim loop's degraded-cycle
    streak, the spare sweep's total) by ``DIRECT_SERIES``. The ISO claim time
    is a timestamp string with a numeric twin (``last_claim_timestamp``) and
    is the only key allowed to stay out.
    """

    executor = build_executor(FakeExecutorClient(), [], tmp_path)
    incremented = set(COUNTER_SERIES) | set(GAUGE_SERIES)
    exported = incremented | set(DIRECT_SERIES)

    unexported = set(executor.metrics_snapshot()) - {"last_successful_claim_at"}
    unexported -= exported
    assert unexported == set(), (
        f"breadcrumb numbers with no /metrics series: {sorted(unexported)}"
    )
    series = (
        set(COUNTER_SERIES.values())
        | set(GAUGE_SERIES.values())
        | set(DIRECT_SERIES.values())
    )
    assert series <= set(executor_samples(executor)), (
        "every mapped series must be a member of the rendered family"
    )
    assert all(hasattr(executor, attribute) for attribute in incremented), (
        "a mapped attribute the executor lacks is an AttributeError at increment"
    )


# ----------------------------------------------------------------- the counters


@pytest.mark.parametrize(
    ("step_status", "series"),
    [
        (WorkflowStepStatus.SUCCEEDED, "results_succeeded_total"),
        (WorkflowStepStatus.FAILED, "results_failed_total"),
        (WorkflowStepStatus.WAITING, "results_waiting_total"),
    ],
)
def test_a_claim_and_result_cycle_moves_the_claim_and_result_series(
    tmp_path, step_status: WorkflowStepStatus, series: str
) -> None:
    """One claimed command, one posted verdict: the scrape must say both."""

    client = FakeExecutorClient([remote_command("command-a")])
    adapter = RecordingAdapter(
        status=step_status,
        step_error="boom" if step_status is WorkflowStepStatus.FAILED else None,
    )
    executor = build_executor(client, [adapter], tmp_path)
    before = time.time()

    assert executor.run_once() == 1
    posted = client.reported("command-a")
    values = executor_samples(executor)

    assert posted.status is RemoteCommandStatus(step_status.value), posted
    assert values["claims_total"] == 1 == executor.claimed_total
    assert values[series] == 1, values
    others = {
        "results_succeeded_total",
        "results_failed_total",
        "results_waiting_total",
    }
    assert all(values[name] == 0 for name in others - {series}), values
    assert values["in_flight_commands"] == 0, "the batch finished"
    assert values["claim_loop_alive"] == 1, values
    assert values["last_claim_timestamp"] >= int(before), values
    assert values["last_loop_iteration_timestamp"] >= int(before), values


def test_an_empty_claim_marks_the_claim_timestamp_but_counts_no_command(
    tmp_path,
) -> None:
    """A successful round trip with nothing to do is still a live claim."""

    executor = build_executor(FakeExecutorClient(), [], tmp_path)

    assert executor.run_once() == 0
    values = executor_samples(executor)

    assert values["claims_total"] == 0
    assert values["last_claim_timestamp"] > 0, (
        "the claim timestamp is the token/TLS/route proof, commands or not"
    )
    assert values["claim_loop_alive"] == 1


def test_the_execution_timeout_moves_timeouts_and_the_stuck_gauge(tmp_path) -> None:
    client = FakeExecutorClient([remote_command("command-a")])
    adapter = StuckAdapter()
    executor = build_executor(client, [adapter], tmp_path, max_execution_seconds=0.2)

    try:
        assert executor.run_once() == 1
        values = executor_samples(executor)
        assert values["execution_timeouts_total"] == 1, values
        assert values["stuck_executions"] == 1, "the abandoned thread must show"
        assert values["results_failed_total"] == 1, (
            "the timeout verdict is a posted FAILED result"
        )
    finally:
        adapter.released.set()
    assert adapter.returned.wait(10), "the stuck adapter never returned"
    deadline = time.monotonic() + 5
    while executor.stuck_executions and time.monotonic() < deadline:
        time.sleep(0.01)
    assert executor_samples(executor)["stuck_executions"] == 0, (
        "the gauge must follow the attribute back down when the thread returns"
    )


def test_a_fleet_fence_hold_is_counted_as_a_hold_and_a_waiting_result(tmp_path) -> None:
    command = remote_command(
        "command-a", operation=WorkflowOperation.MARK_UNSCHEDULABLE, workflow_steps=True
    )
    client = FakeExecutorClient([command])
    adapter = RecordingAdapter()
    adapter.registry = FenceRegistry()  # type: ignore[attr-defined]
    executor = build_executor(client, [adapter], tmp_path)

    assert executor.run_once() == 1
    values = executor_samples(executor)

    assert client.reported("command-a").details["fleet_preflight_blocked"] is True
    assert values["fleet_fence_holds_total"] == 1 == executor.fleet_fence_holds_total
    assert values["results_waiting_total"] == 1, values
    assert executor.metrics_snapshot()["fleet_fence_holds_total"] == 1, (
        "the breadcrumb must carry the new counter like every other one"
    )


def test_a_transport_failure_on_the_report_counts_one_retry_per_backoff(
    tmp_path,
) -> None:
    client = FlakyReportClient(
        [remote_command("command-a")],
        complete_failures={"command-a": [URLError(socket.timeout("timed out"))]},
    )
    delays: list[float] = []
    executor = build_executor(
        client, [RecordingAdapter()], tmp_path, sleep=delays.append
    )

    assert executor.run_once() == 1
    values = executor_samples(executor)

    assert client.attempts == ["command-a", "command-a"], client.attempts
    assert values["transport_retries_total"] == 1 == len(delays), values
    assert values["results_succeeded_total"] == 1, "the second post landed"
    assert executor.metrics_snapshot()["transport_retries_total"] == 1


def test_a_compound_command_moves_the_batched_command_and_step_series(tmp_path) -> None:
    """One claimed compound command with four covered steps: the scrape must
    count the command once and every step it ran, like the breadcrumb does."""

    client = FakeClient(compound_command())
    executor = build_executor(client, [FakeNodeAdapter(all_succeed())], tmp_path)

    assert executor.run_once() == 1
    values = executor_samples(executor)
    snapshot = executor.metrics_snapshot()

    assert client.reported().status is RemoteCommandStatus.SUCCEEDED
    assert values["batched_commands_total"] == 1 == snapshot["batched_commands_total"]
    assert values["batched_steps_total"] == 4 == snapshot["batched_steps_total"], (
        "the four node-side steps behind the head must each count once"
    )
    assert values["batched_progress_failures_total"] == 0, (
        "every progress post landed on the fake client"
    )
    assert values["claims_total"] == 1 and values["results_succeeded_total"] == 1


def test_a_progress_post_failure_moves_the_batched_progress_failure_series(
    tmp_path,
) -> None:
    client = FakeClient(compound_command(), progress_error=RuntimeError("503"))
    executor = build_executor(client, [FakeNodeAdapter(all_succeed())], tmp_path)

    assert executor.run_once() == 1
    values = executor_samples(executor)

    assert client.reported().status is RemoteCommandStatus.SUCCEEDED, (
        "a failed progress post costs latency, never a step"
    )
    assert values["batched_progress_failures_total"] == 4 == len(client.progress_posts)
    assert values["batched_steps_total"] == 4, values


def test_a_transport_error_from_the_adapter_moves_the_retryable_transport_series(
    tmp_path,
) -> None:
    """A gaierror out of the adapter is the executor's own connectivity: the
    command is held WAITING and the transport counter, not the adapter one,
    moves."""

    client = FakeExecutorClient([remote_command("command-a")])
    adapter = RecordingAdapter(
        raises=socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    )
    executor = build_executor(client, [adapter], tmp_path)

    assert executor.run_once() == 1
    values = executor_samples(executor)

    assert client.reported("command-a").status is RemoteCommandStatus.WAITING
    assert (
        values["retryable_transport_errors_total"]
        == 1
        == executor.retryable_transport_errors_total
    )
    assert values["retryable_adapter_errors_total"] == 0, values
    assert values["results_waiting_total"] == 1, values
    assert executor.metrics_snapshot()["retryable_transport_errors_total"] == 1


def test_the_transport_degraded_streak_is_a_gauge_that_rises_and_resets(
    tmp_path,
) -> None:
    """A cycle that advanced nothing and failed on the executor's own network
    reads 1; the next clean cycle reads 0 again, like the breadcrumb."""

    client = FakeExecutorClient([remote_command("command-a")], [remote_command("b")])
    adapter = RecordingAdapter(
        raises=socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    )
    executor = build_executor(client, [adapter], tmp_path)

    assert executor.run_once() == 1
    degraded = executor_samples(executor)
    adapter.raises = None
    assert executor.run_once() == 1
    clean = executor_samples(executor)

    assert degraded["consecutive_transport_degraded_cycles"] == 1, degraded
    assert clean["consecutive_transport_degraded_cycles"] == 0, clean
    assert clean["retryable_transport_errors_total"] == 1, (
        "the counter keeps the history the gauge does not"
    )
    assert executor.metrics_snapshot()["consecutive_transport_degraded_cycles"] == 0


def test_a_sweep_that_reclaims_a_spare_moves_the_reclaimed_series(tmp_path) -> None:
    core_nodes = {
        "hyperpod-i-1": reserved_node(
            "incident-old", reserved_at=NOW - timedelta(hours=2)
        )
    }
    coordinator = HyperPodSpareCoordinator(
        FakeLifecycle([spare("i-1")]), None, FakeCore(core_nodes), now=lambda: NOW
    )
    sweep = SpareReservationSweep(
        coordinator, ttl_seconds=3600.0, interval_seconds=300.0, now=lambda: NOW
    )
    executor = build_executor(
        FakeExecutorClient(), [], tmp_path, spare_reservation_sweep=sweep
    )

    executor.sweep_spare_reservations()
    values = executor_samples(executor)

    assert sweep.reclaimed_total == 1, "the fixture must reclaim exactly one spare"
    assert values["spare_reservations_reclaimed_total"] == 1, values
    assert executor.metrics_snapshot()["spare_reservations_reclaimed_total"] == 1


def test_a_lost_lease_and_an_abandoned_hold_are_exported(tmp_path) -> None:
    """The two counters that already existed just have to reach the scrape."""

    executor = build_executor(FakeExecutorClient(), [], tmp_path)

    executor.increment("lease_lost_total")
    executor.increment("abandoned_lease_holds_total", 2)
    values = executor_samples(executor)

    assert values["lease_lost_total"] == 1
    assert values["abandoned_lease_holds_total"] == 2


def test_a_failed_breadcrumb_write_reads_as_loop_not_alive(tmp_path) -> None:
    """``claim_loop_alive`` is the writer's own view of its breadcrumb."""

    executor = build_executor(
        FakeExecutorClient(),
        [],
        tmp_path,
        liveness_state_path=str(tmp_path / "missing-dir" / "executor-loop-alive"),
    )

    assert executor.run_once() == 0
    values = executor_samples(executor)

    assert values["claim_loop_alive"] == 0, values
    assert values["last_loop_iteration_timestamp"] == 0, (
        "an iteration whose breadcrumb never landed must not be marked"
    )


def test_a_stopped_loop_reads_as_not_alive(tmp_path) -> None:
    executor = build_executor(FakeExecutorClient(), [], tmp_path, poll_seconds=0.01)
    executor.request_stop("test")

    executor.run()

    assert executor_samples(executor)["claim_loop_alive"] == 0


# --------------------------------------------------------------------- /healthz


def test_the_health_predicate_reads_the_loop_breadcrumb_age(tmp_path) -> None:
    path = tmp_path / "executor-loop-alive"

    assert loop_breadcrumb_is_fresh(str(path), LIVENESS_STALE_AFTER_SECONDS) is False, (
        "no breadcrumb yet must read unhealthy, like the exec probe"
    )
    path.write_text("{}", encoding="utf-8")
    assert loop_breadcrumb_is_fresh(str(path), LIVENESS_STALE_AFTER_SECONDS) is True
    stale = time.time() - LIVENESS_STALE_AFTER_SECONDS - 1
    os.utime(path, (stale, stale))
    assert loop_breadcrumb_is_fresh(str(path), LIVENESS_STALE_AFTER_SECONDS) is False


def test_healthz_follows_the_executors_own_breadcrumb(tmp_path) -> None:
    """200 while the loop breadcrumb is fresh, 503 once it is 300 s old."""

    executor = build_executor(FakeExecutorClient(), [], tmp_path)
    server = MetricsServer(
        executor.metrics.family, port=0, health=executor_health(executor)
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        missing, _ = get(f"{base}/healthz")
        assert executor.run_once() == 0
        fresh, _ = get(f"{base}/healthz")
        status, body = get(f"{base}/metrics")
        stale = time.time() - LIVENESS_STALE_AFTER_SECONDS - 1
        os.utime(executor.liveness_state_path, (stale, stale))
        aged, _ = get(f"{base}/healthz")
    finally:
        server.stop()

    assert (missing, fresh, aged) == (503, 200, 503)
    assert status == 200
    assert "gpu_fault_cluster_executor_claim_loop_alive 1" in body.splitlines(), body


# ------------------------------------------------------------------ bootstrap


def test_port_zero_disables_the_metrics_server(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_METRICS_PORT", "0")
    executor = build_executor(FakeExecutorClient(), [], tmp_path)

    with caplog.at_level(logging.INFO, logger=metrics_module.LOGGER.name):
        assert start_executor_metrics(executor) is None
    assert any("disabled" in record.getMessage() for record in caplog.records), (
        "an operator must be able to see that 0 turned the server off"
    )


def test_the_default_port_is_the_manifests_metrics_port(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_CLUSTER_EXECUTOR_METRICS_PORT", raising=False)
    started: list[int] = []

    def fake_start(family: Any, *, port: int, health: Any) -> None:
        started.append(port)
        return None

    monkeypatch.setattr(BOOTSTRAP, "start_metrics_server", fake_start)
    start_executor_metrics(build_executor(FakeExecutorClient(), [], tmp_path))

    manifest = yaml.safe_load_all(
        (ROOT / "deploy/dataplane/cluster-action-executor.yaml").read_text("utf-8")
    )
    deployment = next(d for d in manifest if d and d.get("kind") == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    metrics_port = next(p for p in container["ports"] if p["name"] == "metrics")
    assert started == [int(metrics_port["containerPort"])], started


def test_a_bind_failure_is_one_error_line_and_main_still_runs_the_loop(
    tmp_path, monkeypatch, caplog
) -> None:
    ran: list[str] = []

    class LoopOnlyExecutor:
        metrics = ClusterExecutorMetrics()
        liveness_state_path = str(tmp_path / "executor-loop-alive")

        def install_signal_handlers(self) -> None:
            ran.append("signals")

        def run(self) -> None:
            ran.append("run")

    monkeypatch.setattr(BOOTSTRAP, "configure_logging", lambda: None)
    monkeypatch.setattr(
        BOOTSTRAP, "validate_gpu_fault_environment", lambda **_kwargs: None
    )
    monkeypatch.setattr(BOOTSTRAP, "executor_from_environment", LoopOnlyExecutor)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen(1)
        monkeypatch.setenv(
            "GPU_FAULT_CLUSTER_EXECUTOR_METRICS_PORT", str(occupied.getsockname()[1])
        )
        with caplog.at_level(logging.ERROR, logger=metrics_module.LOGGER.name):
            BOOTSTRAP.main()

    assert ran == ["signals", "run"], "a metrics bind failure must not stop the loop"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "cannot bind" in errors[0].getMessage(), errors
