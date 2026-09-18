"""Owned PREEMPT runner lifecycles with local Stores and fake transports."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import preempt037_verdicts as liveness
from scripts.e2e.regional import run_preempt012_acceptance as preempt012
from scripts.e2e.regional import run_preempt036_stuck_workflow_reconcile as sweep


def test_preempt012_passes_the_controlled_host_probe_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: list[Path] = []
    calls: list[str] = []

    def settings(*, state_directory: Path, **kwargs: Any) -> SimpleNamespace:
        paths.append(state_directory)
        return SimpleNamespace(state_directory=state_directory, **kwargs)

    class Host:
        def __init__(self, configuration: Any) -> None:
            assert configuration.state_directory == paths[-1], (
                "host creation must use the attempt's controlled state directory"
            )

        def create(self) -> None:
            calls.append("create")
            raise RuntimeError("fake creation acknowledgement lost")

        def execute(self, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(command)
            return {"timer_active_state": "inactive"}

        def cleanup(self) -> dict[str, bool]:
            calls.append("remove")
            return {}

    monkeypatch.setattr(preempt012, "HostProbeSettings", settings)
    monkeypatch.setattr(preempt012, "HostProbeFixture", Host)
    monkeypatch.setattr(
        preempt012,
        "authorize_execution",
        lambda *a, **k: datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setattr(
        preempt012,
        "read_only_preflight",
        lambda *a, **k: {"errors": [], "predecessor": {"valid": True}},
    )
    regional = SimpleNamespace(
        settings=SimpleNamespace(
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            gpu_context="gpu-context",
            namespace="gpu-ns",
        ),
        evidence_identity=lambda: {
            "release_id": "release-test",
            "cluster_id": "cluster-test",
        },
        node_snapshot=lambda node: {"ownership_annotations": {}},
    )
    predecessor_id, predecessor_file = preempt012.predecessor_path(
        tmp_path, preempt012.CASE_ID, None
    )
    assert predecessor_id is not None and predecessor_file is not None, (
        "the host lifecycle requires an explicit formal predecessor"
    )
    assert (
        preempt012.execute_case(
            Namespace(run_dir=tmp_path, attempt=1),
            regional,
            node="node-a",
            image="image@sha256:" + "a" * 64,
            predecessor_id=predecessor_id,
            predecessor_path_value=predecessor_file,
            environment={},
        )
        == 1
    ), "lost creation acknowledgement must not produce a passing case"
    assert paths == [tmp_path / "cases" / preempt012.CASE_ID / "host-probes"], (
        "host ownership evidence must remain in the case directory"
    )
    assert calls == ["create", "cleanup", "remove"], (
        "creation failure must still attempt host and resource cleanup"
    )


def test_preempt012_uses_the_formal_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor_id, predecessor_file = preempt012.predecessor_path(
        tmp_path, preempt012.CASE_ID, None
    )
    assert predecessor_id is not None and predecessor_file is not None, (
        "the preflight must use an evidence-producing formal predecessor"
    )
    case_dir = tmp_path / "cases" / preempt012.CASE_ID
    case_dir.mkdir(parents=True)
    seen: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        preempt012,
        "predecessor_evidence",
        lambda path, case_id: seen.append((path, case_id)) or {"valid": True},
    )
    monkeypatch.setattr(preempt012, "focused_tests", lambda *a, **k: {"passed": True})
    regional = SimpleNamespace(
        node_snapshot=lambda node: {
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        },
        store_snapshot=lambda **k: {
            "agent": {"lifecycle_state": "ACTIVE"},
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        },
        business_workloads=lambda node: [],
        cpu_blast_snapshot=lambda: {},
    )
    result = preempt012.read_only_preflight(
        regional,
        node="node-a",
        predecessor_id=predecessor_id,
        predecessor_path_value=predecessor_file,
        case_dir=case_dir,
    )
    assert result["errors"] == [], (
        "a valid explicit predecessor must pass the real formal-order revalidation"
    )
    assert seen == [(predecessor_file, predecessor_id)], (
        "preflight must load evidence for the exact predecessor it resolved"
    )


def test_sqlite_runner_uses_and_removes_only_its_temporary_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    sentinel = work / "unrelated.db"
    sentinel.write_text("preserve", encoding="utf-8")
    paths: list[Path] = []

    def run_shape(shape: str, provisioner: Any, **kwargs: Any) -> dict[str, Any]:
        url = provisioner.create(shape)
        store = provisioner.open(url)
        store.close()
        path = Path(url.removeprefix("sqlite:///"))
        paths.append(path)
        assert path.exists(), f"owned SQLite store was not created: {path}"
        assert provisioner.owns(url), "the provisioner lost ownership of its new store"
        with pytest.raises(sweep.RunnerError, match="did not create"):
            provisioner.open("postgresql://unowned.invalid/database")
        return {"verdict": "PASS", "errors": []}

    monkeypatch.setattr(sweep, "run_shape", run_shape)
    assert sweep.main(["--run-dir", str(tmp_path), "--backend", "sqlite"]) == 0
    assert len(paths) == 3 and all(not path.exists() for path in paths)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_shape_closes_store_even_when_seeding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    store = SimpleNamespace(close=lambda: closed.append(True))
    provisioner = SimpleNamespace(
        create=lambda name: "owned-url", open=lambda url: store
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("seed failed")

    monkeypatch.setattr(sweep, "seed_shape", fail)
    with pytest.raises(RuntimeError, match="seed failed"):
        sweep.run_shape("compile-blocked", provisioner, safety_probe="", stats_probe="")
    assert closed == [True]


def test_partial_shape_selection_cannot_write_full_case_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sweep, "run_shape", lambda shape, *a, **k: {"verdict": "PASS", "errors": []}
    )
    monkeypatch.setattr(sweep, "_probe_source", lambda name: "probe")
    assert (
        sweep.main(
            [
                "--run-dir",
                str(tmp_path),
                "--backend",
                "sqlite",
                "--shape",
                "compile-blocked",
            ]
        )
        == 1
    )
    document = json.loads(
        (tmp_path / "cases" / sweep.CASE_ID / f"{sweep.CASE_ID}.json").read_text()
    )
    assert document["verdict"] == "FAIL"
    assert document["errors"] == [
        "all three distinct shapes are required for a case PASS"
    ]


def test_complete_sqlite_runner_exercises_the_shipped_probes(tmp_path: Path) -> None:
    assert sweep.main(["--run-dir", str(tmp_path), "--backend", "sqlite"]) == 0
    document = json.loads(
        (tmp_path / "cases" / sweep.CASE_ID / f"{sweep.CASE_ID}.json").read_text()
    )
    assert document["store_backend"] == "sqlite"
    assert set(document["shapes"]) == set(sweep.SHAPES)
    assert all(value["verdict"] == "PASS" for value in document["shapes"].values()), (
        f"one of the three shipped shape probes failed: {document['shapes']}"
    )
    assert not list((tmp_path / "work").rglob("*.db")), (
        "owned SQLite files survived teardown"
    )


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"status_counts": None},
        {"status_counts": {"UNKNOWN": 1}},
        {"status_counts": {"RUNNING": True}},
        {"status_counts": {"PENDING": -1}},
        {"status_counts": {"SUCCEEDED": 600, "RUNNING": 1}},
    ],
)
def test_liveness_quiescence_requires_complete_known_counts(report: dict) -> None:
    assert liveness.workflow_census_errors(report), (
        f"unsafe workflow census was accepted: {report}"
    )
    assert not liveness.workflow_census_errors(
        {"status_counts": {"SUCCEEDED": 600, "FAILED": 1, "RUNNING": 0}}
    ), "a complete terminal-only census must pass quiescence"


def test_liveness_rejects_a_rule_it_does_not_evaluate() -> None:
    for expression in (
        f"sum({liveness.DISPATCH_METRIC}) > 300",
        f"time() - max by (control_plane_cluster, region) ({liveness.DISPATCH_METRIC}) > 300 or vector(1)",
    ):
        rules = {
            "groups": [
                {
                    "rules": [
                        {
                            "alert": liveness.STALLED_ALERT,
                            "expr": expression,
                            "for": "5m",
                        }
                    ]
                }
            ]
        }
        with pytest.raises(ValueError, match="unsupported"):
            liveness.stall_rule_parameters(json.dumps(rules))


def test_sparse_timeline_and_unknown_recovery_do_not_prove_liveness() -> None:
    assert liveness.stall_timeline_errors(
        [
            {"observed_epoch": 1000, "stalled": True, "periodic_alive": True},
            {"observed_epoch": 1400, "stalled": True, "periodic_alive": True},
        ],
        for_seconds=300,
    ), "a 400-second observation gap must not prove a continuous stall"
    assert liveness.recovery_errors([{}], threshold_seconds=300), (
        "missing dispatch and periodic metrics must not prove recovery"
    )
    assert liveness.restore_errors({"baseline": {}, "restored_state": {}}), (
        "empty baseline and restored state must not prove restoration"
    )
