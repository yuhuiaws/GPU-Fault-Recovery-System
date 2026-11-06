"""The load generators' evidence and the INVALID verdict of a saturated client.

Perf plan §5: "results are invalid if any resource exceeds 70%" -- a saturated
client cannot prove product latency. The runners recorded only
``client_cpu_cores`` and left the Pod count, the spread and the CPU gate to
whoever ran the round. The burst and integrated runners now record the load
Pods, the distinct nodes they ran on and each generator's CPU against its Pod's
CPU limit, and a generator above 70% of its limit makes the round INVALID.
Nothing about the load changes: same Pods, no anti-affinity, no calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_capacity_database as database
from scripts.perf import regional_capacity_suite as capacity
from scripts.perf import regional_load_generator as load_generator
from tests.regional._perf_caller_support import (
    TEARDOWN_TAIL,
    CapacityWire,
    capacity_args,
)


def pods(count: int, *, nodes: int, cpu_limit: str = "4") -> str:
    """A ``kubectl get pod -o json`` answer for a load Job's Pods."""

    return json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": f"load-{index}"},
                    "spec": {
                        "nodeName": f"node-{index % nodes}",
                        "containers": [
                            {"name": "sidecar", "resources": {"limits": {"cpu": "1"}}},
                            {
                                "name": "load",
                                "resources": {"limits": {"cpu": cpu_limit}},
                            },
                        ],
                    },
                }
                for index in range(count)
            ]
        }
    )


def reading(cluster: str, cores: float) -> dict[str, Any]:
    return {"cluster_id": cluster, "client_cpu_cores": cores}


@pytest.mark.parametrize(
    ("quantity", "cores"),
    [("4", 4.0), ("4000m", 4.0), ("500m", 0.5), ("1.5", 1.5), ("", None), ("x", None)],
)
def test_cpu_quantities_are_read_in_cores(quantity: str, cores: float | None) -> None:
    assert load_generator.cpu_cores(quantity) == cores


def test_the_load_pods_are_read_with_their_node_and_load_container_limit() -> None:
    calls: list[tuple[str, ...]] = []

    def dataplane(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        return pods(3, nodes=2, cpu_limit="4000m")

    found = load_generator.load_pods(dataplane, "gpu-fault-perf-suite-burst")

    assert calls == [
        (
            "get",
            "pod",
            "-l",
            "batch.kubernetes.io/job-name=gpu-fault-perf-suite-burst",
            "-o",
            "json",
        )
    ], "one read-only list of the Job's Pods"
    assert found == [
        {"name": "load-0", "node": "node-0", "cpu_limit_cores": 4.0},
        {"name": "load-1", "node": "node-1", "cpu_limit_cores": 4.0},
        {"name": "load-2", "node": "node-0", "cpu_limit_cores": 4.0},
    ], "the generator container's limit, not the sidecar's"
    assert load_generator.load_pods(lambda *_a, **_k: "", "job") == [], "no Pods"


def test_a_round_whose_generators_stayed_below_the_limit_is_valid() -> None:
    record = load_generator.section(
        load_generator.load_pods(lambda *_a, **_k: pods(32, nodes=4), "job"),
        [reading(f"perf-cap-{index:03d}", 0.2) for index in range(32)],
    )

    assert record["valid"] is True and record["judged"] is True, record
    assert (record["pods"], record["nodes"], record["max_pods_per_node"]) == (32, 4, 8)
    assert record["cpu_limit_cores"] == 4.0 and record["readings"] == 32
    assert record["cpu_limit_share_max"] == pytest.approx(0.05)
    assert record["mean_below_half_limit"] is True
    assert record["at_least_4_pods"] is True and record["one_pod_per_node"] is False, (
        "the plan's Pod-count and spread gates are recorded as facts"
    )
    assert record["reason"] == (
        "every load generator stayed within 70% of its 4-core CPU limit (max 5%)"
    )


def test_a_generator_above_70_percent_of_its_limit_invalidates_the_round() -> None:
    record = load_generator.section(
        load_generator.load_pods(lambda *_a, **_k: pods(2, nodes=2), "job"),
        [reading("perf-cap-000", 1.0), reading("perf-cap-001", 3.2)],
    )

    assert record["valid"] is False, record
    assert record["reason"] == (
        "the load generator of perf-cap-001 used 3.20 of its 4-core CPU limit "
        "(80% > 70%): a saturated client cannot prove product latency"
    )
    assert record["mean_below_half_limit"] is False, "a 52.5% mean is recorded"


def test_exactly_70_percent_is_still_valid() -> None:
    record = load_generator.section(
        [{"name": "load-0", "node": "n", "cpu_limit_cores": 4.0}],
        [reading("perf-cap-000", 2.8)],
    )

    assert record["valid"] is True, record


def test_without_a_reading_or_a_limit_the_gate_is_not_judged() -> None:
    no_reading = load_generator.section(
        [{"name": "load-0", "node": "n", "cpu_limit_cores": 4.0}], []
    )
    no_limit = load_generator.section(
        [{"name": "load-0", "node": "n", "cpu_limit_cores": None}],
        [reading("perf-cap-000", 3.9)],
    )

    assert no_reading["judged"] is False and no_reading["valid"] is True
    assert no_reading["reason"] == "no client_cpu_cores reading: not judged"
    assert no_limit["judged"] is False and no_limit["valid"] is True
    assert no_limit["reason"] == "no CPU limit: not judged"


def test_an_unreadable_pod_list_still_judges_the_rendered_limit() -> None:
    def dataplane(*_args: str, **_kwargs: Any) -> str:
        raise RuntimeError("command failed (1): kubectl get pod\nboom")

    record = load_generator.evidence(
        dataplane, "job", [reading("perf-cap-000", 3.5)], cpu_limit="4"
    )

    assert record["pods_read_error"] == (
        "RuntimeError: command failed (1): kubectl get pod boom"
    ), "the failed read is recorded"
    assert record["pods"] == 0 and record["cpu_limit_cores"] == 4.0, record
    assert record["valid"] is False, "the CPU gate judges on the rendered limit"


def test_an_invalid_round_beats_its_verdict_and_names_its_reason() -> None:
    invalid = {"load_generator": {"valid": False, "reason": "saturated"}}

    assert database.measured_exit(1, invalid) == load_generator.INVALID_EXIT, (
        "a saturated client invalidates the round whatever its verdict"
    )
    assert database.measured_exit(0, {"load_generator": {"valid": True}}) == 0
    assert database.measured_reason(load_generator.INVALID_EXIT, invalid) == (
        "saturated"
    )
    assert database.step_status(load_generator.INVALID_EXIT, "failed", "saturated") == {
        "status": "invalid",
        "reason": "saturated",
    }
    assert database.step_status(load_generator.INVALID_EXIT, "failed") == {
        "status": "invalid",
        "reason": load_generator.INVALID_REASON,
    }


@pytest.mark.parametrize(("cores", "code"), [(1.8, 0), (3.2, 5)])
def test_the_burst_runner_records_the_generators_and_marks_a_saturated_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cores: float, code: int
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    documents = [{"cluster_id": "perf-cap-000", "events": 1, "client_cpu_cores": cores}]
    monkeypatch.setattr(capacity, "collect_logs", lambda *_args: documents)

    assert capacity.main(capacity_args("all", tmp_path)) == code

    assert wire.events[-4:] == TEARDOWN_TAIL, "an INVALID round is still torn down"
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["load_generator"] == {
        "pods": 1,
        "cpu_request": "2",
        "cpu_limit": "4",
        "anti_affinity": False,
        "no_op_calibration": False,
    }, "run.json records the generators as rendered, unchanged"
    record = json.loads((tmp_path / "summary.json").read_text())["load_generator"]
    assert record["pods"] == 1 and record["cpu_limit_cores"] == 4.0, record
    assert record["client_cpu_cores_max"] == cores, record
    status = json.loads((tmp_path / "status.json").read_text())
    if code:
        assert record["valid"] is False, record
        assert status["status"] == "invalid", status
        assert status["reason"].startswith(
            "the load generator of perf-cap-000 used 3.20 of its 4-core CPU limit"
        ), status
    else:
        assert record["valid"] is True, record
        assert status["status"] == "ok", status
