"""The executor reports every Kubernetes Node of its cluster to the control plane.

``NodeInventoryReporter`` is the departed-node cleanup's Kubernetes evidence:
one unfiltered, paginated Node LIST per interval, posted as one report. Every
invariant the module docstring promises is a test here: the whole LIST or no
report (a partial list would read as absences), the report time taken before
the LIST, spares named, a failure one warning and the next cycle's retry, an
old control plane (404) said once, and the claim loop never interrupted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    NodeInventoryReporter,
    RegionalExecutorClient,
)
from gpu_fault.kubernetes_node_inventory import NODE_INVENTORY_MAX_NODES
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
)

CLUSTER = "cluster-a"
STARTED = datetime.now(timezone.utc)


def _node(name: str, *, labels: dict[str, str] | None = None, **annotations: str):
    return {
        "metadata": {
            "name": name,
            "labels": labels or {},
            "annotations": dict(annotations),
        }
    }


class FakeCore:
    """``list_node`` with ``limit``/``_continue`` pages, recorded."""

    def __init__(self, *pages: list[dict[str, Any]], error: Exception | None = None):
        self.pages = list(pages)
        self.error = error
        self.calls: list[tuple[int, str | None]] = []

    def list_node(self, *, limit: int, _continue: str | None = None) -> dict[str, Any]:
        self.calls.append((limit, _continue))
        if self.error is not None and len(self.calls) == len(self.pages):
            raise self.error
        index = 0 if _continue is None else int(_continue)
        metadata: dict[str, Any] = {"resourceVersion": "4711"}
        if index + 1 < len(self.pages):
            metadata["continue"] = str(index + 1)
        return {"metadata": metadata, "items": self.pages[index]}


class FakeInventoryClient:
    cluster_id = CLUSTER

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.reports: list[dict[str, Any]] = []

    def report_node_inventory(self, report: dict[str, Any]) -> dict[str, Any]:
        self.reports.append(report)
        if self.error is not None:
            raise self.error
        return {"cluster_id": CLUSTER}


def _reporter(client: Any, core: FakeCore, **overrides: Any) -> NodeInventoryReporter:
    return NodeInventoryReporter(
        client, core, executor_id=EXECUTOR, now=lambda: STARTED, **overrides
    )


def test_every_page_is_one_report_with_the_spares_named() -> None:
    core = FakeCore(
        [_node("node-b"), _node("node-spare", labels={"gpu-fault.io/spare": "true"})],
        [
            _node("node-a"),
            _node("node-reserved", **{"gpu-fault.io/spare-reservation": "incident-1"}),
            _node("node-label-false", labels={"gpu-fault.io/spare": "false"}),
        ],
    )
    client = FakeInventoryClient()

    count = _reporter(client, core, page_size=2).run()

    assert count == 5
    assert core.calls == [(2, None), (2, "1")], "the continue token was followed"
    [report] = client.reports
    assert report["cluster_id"] == CLUSTER and report["executor_id"] == EXECUTOR
    assert report["node_names"] == [
        "node-a",
        "node-b",
        "node-label-false",
        "node-reserved",
        "node-spare",
    ]
    assert report["spare_node_names"] == ["node-reserved", "node-spare"]
    assert report["resource_version"] == "4711"
    assert datetime.fromisoformat(report["observed_at"]) == STARTED


def test_a_page_that_fails_fails_the_whole_report() -> None:
    core = FakeCore(
        [_node("node-a")], [_node("node-b")], error=TimeoutError("apiserver")
    )
    client = FakeInventoryClient()
    reporter = _reporter(client, core, page_size=1)

    with pytest.raises(TimeoutError):
        reporter.run()

    assert client.reports == [], "a partial list would read as absences"
    assert reporter.failures_total == 1


def test_a_cluster_larger_than_one_report_is_refused_not_cut() -> None:
    core = FakeCore(
        [_node(f"node-{index}") for index in range(NODE_INVENTORY_MAX_NODES + 1)]
    )
    client = FakeInventoryClient()

    with pytest.raises(ClusterExecutorError, match="more than"):
        _reporter(client, core).run()
    assert client.reports == []


def test_the_report_is_paced_and_a_failure_never_reaches_the_claim_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = {"now": 0.0}
    client = FakeInventoryClient(
        ClusterExecutorError("regional control plane request failed: timeout")
    )
    reporter = _reporter(
        client, FakeCore([_node("node-a")]), clock=lambda: clock["now"]
    )
    executor = ClusterActionExecutor(
        FakeExecutorClient(),
        [],
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        node_inventory=reporter,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        executor.report_node_inventory()
        executor.report_node_inventory()

    assert len(client.reports) == 1, "not due again inside the interval"
    assert reporter.failures_total == 1
    assert any(
        "Kubernetes node report failed" in record.getMessage()
        for record in caplog.records
    ), "the wrapper must log the failure"
    clock["now"] = reporter.interval_seconds + 1
    executor.report_node_inventory()
    assert len(client.reports) == 2


def test_an_old_control_plane_is_said_once(caplog: pytest.LogCaptureFixture) -> None:
    clock = {"now": 0.0}
    client = FakeInventoryClient(
        ClusterExecutorError(
            "regional control plane rejected request (404)", status_code=404
        )
    )
    reporter = _reporter(
        client, FakeCore([_node("node-a")]), clock=lambda: clock["now"]
    )

    with caplog.at_level(logging.INFO, logger="gpu_fault.cluster_executor"):
        reporter.report()
        clock["now"] = 1000.0
        reporter.report()

    messages = [
        record.getMessage()
        for record in caplog.records
        if "does not take Kubernetes node reports" in record.getMessage()
    ]
    assert len(messages) == 1, messages
    assert not any(record.levelno >= logging.WARNING for record in caplog.records), (
        "a missing route is not a warning"
    )


def test_an_executor_without_a_reporter_is_unchanged() -> None:
    executor = ClusterActionExecutor(
        FakeExecutorClient(), [], executor_id=EXECUTOR, allowed_namespaces={"training"}
    )
    assert executor.node_inventory is None
    executor.report_node_inventory()


@pytest.mark.parametrize(
    "overrides", [{"executor_id": " "}, {"interval_seconds": 0}, {"page_size": 0}]
)
def test_the_reporter_refuses_an_unusable_configuration(overrides: dict) -> None:
    settings = {"executor_id": EXECUTOR, **overrides}
    with pytest.raises(ValueError):
        NodeInventoryReporter(FakeInventoryClient(), FakeCore([]), **settings)


def test_the_client_posts_only_its_own_cluster_and_checks_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RegionalExecutorClient("https://control.example", CLUSTER, "t" * 32)
    posted: list[tuple[str, dict[str, Any]]] = []
    answers = [{"cluster_id": CLUSTER, "recorded": True}, {"cluster_id": "cluster-b"}]

    def fake_post(path: str, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        posted.append((path, payload))
        return answers.pop(0)

    monkeypatch.setattr(client, "_post", fake_post)
    report = {"cluster_id": CLUSTER, "node_names": ["node-a"]}

    assert client.report_node_inventory(report)["recorded"] is True
    assert posted == [("/v1/regional/executors/node-inventory", report)]
    with pytest.raises(ClusterExecutorError, match="another cluster"):
        client.report_node_inventory(report)
    with pytest.raises(ClusterExecutorError, match="names another cluster"):
        client.report_node_inventory({"cluster_id": "cluster-b"})
    assert len(posted) == 2


def test_the_bootstrap_wires_a_reporter_with_the_executor_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault.cluster_executor import bootstrap

    captured: dict[str, Any] = {}

    class Recorder(ClusterActionExecutor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "ClusterActionExecutor", Recorder)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", CLUSTER)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_ID", "cluster-a/executor-0")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "t" * 32)
    monkeypatch.setenv("GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES", "training")
    monkeypatch.setattr(
        bootstrap.KubernetesWorkflowAdapter,
        "__init__",
        lambda self, **kwargs: setattr(self, "core", object())
        or setattr(self, "owner", "gpu-fault-kubernetes-adapter"),
    )
    monkeypatch.setattr(bootstrap, "_node_key_sync_from_environment", lambda *_: None)
    monkeypatch.setattr(
        bootstrap.KubernetesStopOwnershipValidator,
        "from_adapter",
        classmethod(lambda cls, *args, **kwargs: None),
    )

    bootstrap.executor_from_environment()

    reporter = captured["node_inventory"]
    assert isinstance(reporter, NodeInventoryReporter), reporter
    assert reporter.executor_id == "cluster-a/executor-0"
    assert reporter.interval_seconds == timedelta(minutes=1).total_seconds()
