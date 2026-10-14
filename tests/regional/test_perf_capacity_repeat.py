"""A capacity case run N times in sequence and read as median, min and max.

The single-cluster 1000-node case showed about +/-16 % run-to-run spread on one
build. ``all --repeat N`` runs the whole case N times, each run with its own
suite id, registration, teardown and run directory inside one repeat group,
and ``repeat-summary.json`` reads the headline metrics as median, min, max and
relative spread over the runs that completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_capacity_repeat as repeat
from scripts.perf import regional_capacity_suite as capacity


def summary(throughput: float, **extra: Any) -> dict[str, Any]:
    return {
        "job_status": "Complete",
        "requests": 6500,
        "throughput_req_s": throughput,
        "wall_seconds_max": 50.0,
        "paths": {
            "NVIDIA_KERNEL": {"count": 500, "p50_ms": 10.0, "p99_ms": throughput}
        },
        **extra,
    }


def test_spread_is_median_min_max_and_the_range_over_the_median() -> None:
    assert repeat.spread([10.0, 12.0, 11.0]) == {
        "runs": 3,
        "values": [10.0, 12.0, 11.0],
        "median": 11.0,
        "min": 10.0,
        "max": 12.0,
        "spread": 0.182,
    }, "values stay in run order; the spread is (max - min) / median"
    assert repeat.spread([0.0, 0.0])["spread"] is None, "no spread over a zero median"


def test_the_sections_read_every_reported_metric_and_skip_absent_ones() -> None:
    first = summary(
        200.0,
        fault_p99_ms=900.0,
        processor_timeseries={"consumers": {"mean_in_flight_while_backlog": 12.5}},
    )
    second = summary(240.0, aurora_capacity_max=True)

    metrics, paths = repeat.metric_sections([first, second])

    assert metrics["throughput_req_s"]["values"] == [200.0, 240.0]
    assert metrics["fault_p99_ms"]["runs"] == 1, "a run that lacks it is left out"
    assert metrics["mean_in_flight_while_backlog"]["median"] == 12.5
    assert "aurora_capacity_max" not in metrics, "a flag is not a measurement"
    assert "aurora_cpu_max" not in metrics, "no run reported it"
    assert paths["NVIDIA_KERNEL"]["throughput_req_s"]["values"] == [10.0, 10.0], (
        "a path's throughput is its count over the run's wall time"
    )
    assert paths["NVIDIA_KERNEL"]["p99_ms"]["max"] == 240.0
    assert "p95_ms" not in paths["NVIDIA_KERNEL"]


class Runs:
    """``start`` for a repeat group: one fake complete run per call."""

    def __init__(self, codes: list[int], *, raise_at: int | None = None) -> None:
        self.codes = codes
        self.raise_at = raise_at
        self.started: list[Path] = []

    def __call__(self, artifacts: Path) -> int:
        self.started.append(artifacts)
        index = len(self.started)
        if index == self.raise_at:
            raise RuntimeError("load observation failed")
        code = self.codes[index - 1]
        (artifacts / "run.json").write_text(
            json.dumps({"suite_id": f"suite-{index}", "release_id": "rel-a"})
        )
        (artifacts / "status.json").write_text(
            json.dumps({"status": "ok" if code == 0 else "aborted"})
        )
        (artifacts / "summary.json").write_text(json.dumps(summary(100.0 * index)))
        return code


def read_group(group: Path) -> dict[str, Any]:
    return dict(json.loads((group / repeat.SUMMARY_FILE).read_text()))


def test_a_group_runs_every_case_in_its_own_directory_in_sequence(
    tmp_path: Path,
) -> None:
    runs = Runs([0, 0, 0])
    logged: list[str] = []

    assert repeat.run_repeated(3, tmp_path, runs, logged.append) == 0

    assert runs.started == [tmp_path / f"run-{index}" for index in (1, 2, 3)]
    document = read_group(tmp_path)
    assert (document["complete"], document["completed_runs"]) == (True, 3)
    assert [item["run_id"] for item in document["runs"]] == [
        "suite-1",
        "suite-2",
        "suite-3",
    ], "every run is a run of its own"
    assert document["runs"][0] == {
        "index": 1,
        "artifacts": str(tmp_path / "run-1"),
        "run_id": "suite-1",
        "release_id": "rel-a",
        "exit_code": 0,
        "status": "ok",
        "job_status": "Complete",
    }
    assert (document["release_ids"], document["same_release"]) == (["rel-a"], True)
    assert document["metrics"]["throughput_req_s"]["median"] == 200.0
    assert document["paths"]["NVIDIA_KERNEL"]["p99_ms"]["values"] == [
        100.0,
        200.0,
        300.0,
    ]
    assert logged[0] == f"repeat run 1/3: {tmp_path / 'run-1'}"


def test_the_group_stops_at_the_first_failed_run_and_returns_its_code(
    tmp_path: Path,
) -> None:
    runs = Runs([0, 3, 0])

    assert repeat.run_repeated(3, tmp_path, runs, lambda _line: None) == 3

    assert len(runs.started) == 2, "no run starts after a failure"
    document = read_group(tmp_path)
    assert (document["complete"], document["completed_runs"]) == (False, 1)
    assert [item["exit_code"] for item in document["runs"]] == [0, 3]
    assert document["runs"][1]["status"] == "aborted"
    assert document["metrics"]["throughput_req_s"]["values"] == [100.0], (
        "medians cover the runs that completed"
    )


def test_a_run_that_raises_is_recorded_before_its_error_propagates(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="load observation failed"):
        repeat.run_repeated(3, tmp_path, Runs([0, 0, 0], raise_at=2), print)

    document = read_group(tmp_path)
    assert document["runs"][1]["exit_code"] is None
    assert document["runs"][1]["error"] == "RuntimeError: load observation failed"
    assert document["completed_runs"] == 1


def arguments(**overrides: Any) -> argparse.Namespace:
    values = {
        "repeat": 3,
        "command": "all",
        "suite_id": None,
        "keep_registration": False,
        "no_purge": False,
    }
    return argparse.Namespace(**{**values, **overrides})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"repeat": 1, "command": "run", "suite_id": "run-a"}, None),
        ({}, None),
        ({"repeat": 0}, "at least 1"),
        ({"command": "register"}, "only valid for all"),
        ({"suite_id": "run-a"}, "drop --suite-id"),
        ({"keep_registration": True}, "own teardown"),
        ({"no_purge": True}, "own teardown"),
    ],
)
def test_repeat_refuses_options_that_would_share_a_run(
    overrides: dict[str, Any], expected: str | None
) -> None:
    refusal = repeat.refusal(arguments(**overrides))

    assert (refusal is None) == (expected is None), refusal
    assert expected is None or expected in str(refusal), refusal


class Suite:
    """The suite's ``all`` path with its live execution replaced."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, codes: list[int]) -> None:
        self.codes = codes
        self.calls: list[dict[str, Any]] = []
        monkeypatch.setattr(capacity, "DATAPLANE_CONTEXT", "fake-context")
        monkeypatch.setattr(capacity, "AURORA_CLUSTER_ID", "aurora-a")
        monkeypatch.setattr(
            capacity, "validate_registry_target", lambda **_kwargs: "isolated"
        )
        monkeypatch.setattr(
            capacity, "release_identity", lambda: {"release_id": "rel-a"}
        )
        monkeypatch.setattr(capacity, "execute_capacity_command", self.execute)
        monkeypatch.setattr(capacity, "log", lambda _line: None)

    def execute(self, _args: argparse.Namespace, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        code = self.codes[len(self.calls) - 1]
        artifacts = kwargs["artifacts"]
        capacity.write_status(artifacts, status="ok" if code == 0 else "aborted")
        (artifacts / "summary.json").write_text(json.dumps(summary(250.0)))
        return code


def suite_args(root: Path, *extra: str) -> list[str]:
    return [
        "all",
        "--clusters",
        "1",
        "--nodes-per-cluster",
        "1000",
        "--artifact-root",
        str(root),
        *extra,
    ]


def test_all_with_repeat_runs_the_case_with_a_fresh_suite_id_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = Suite(monkeypatch, [0, 0, 0])

    assert capacity.main(suite_args(tmp_path, "--repeat", "3")) == 0

    (group,) = list((tmp_path / "burst-1c" / "rel-a").iterdir())
    assert [call["artifacts"] for call in suite.calls] == [
        group / f"run-{index}" for index in (1, 2, 3)
    ]
    suite_ids = [call["suite_id"] for call in suite.calls]
    assert len(set(suite_ids)) == 3, "every run registers under its own suite id"
    for call in suite.calls:
        run = json.loads((call["artifacts"] / "run.json").read_text())
        assert (run["suite_id"], run["nodes_per_cluster"]) == (call["suite_id"], 1000)
    document = read_group(group)
    assert [item["run_id"] for item in document["runs"]] == suite_ids
    assert document["metrics"]["throughput_req_s"]["median"] == 250.0


def test_a_failed_repeat_run_is_the_exit_code_of_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = Suite(monkeypatch, [0, 1, 0])

    assert capacity.main(suite_args(tmp_path, "--repeat", "3")) == 1
    assert len(suite.calls) == 2, "the group stops at the failed run"


def test_all_without_repeat_is_the_single_run_it_always_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = Suite(monkeypatch, [0])

    assert capacity.main(suite_args(tmp_path)) == 0

    (artifacts,) = list((tmp_path / "burst-1c" / "rel-a").iterdir())
    assert [call["artifacts"] for call in suite.calls] == [artifacts]
    assert (artifacts / "run.json").is_file(), "the run directory is the stamp dir"
    assert not (artifacts / repeat.SUMMARY_FILE).exists(), "no repeat group"
    assert not (artifacts / "run-1").exists(), "no nested run directory"


def test_the_suite_refuses_a_repeat_that_names_its_suite_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = Suite(monkeypatch, [0, 0])

    with pytest.raises(SystemExit) as refused:
        capacity.main(suite_args(tmp_path, "--repeat", "2", "--suite-id", "run-a"))

    assert refused.value.code == 2, "a usage error"
    assert suite.calls == [], "nothing runs"
