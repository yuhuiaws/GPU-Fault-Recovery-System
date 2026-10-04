"""NOTIFY runner admission edges: the refusals before any drill is sent.

A superseded case, a NOTIFY-005 node list that is not three distinct nodes or a
mutable training image, a wrong confirmation, a predecessor that did not PASS,
a control plane without a Ready worker Pod, a live notification read that does
not cover every recorded id, and the per-role critical-rank annotation check.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_notification_acceptance as runner
from tests.regional._cov95_ha001_harness import DEADLINE
from tests.regional._cov95_notify_harness import NODES, NotificationSite

IMMUTABLE_IMAGE = "registry.invalid/unit@sha256:" + "a" * 64


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NotificationSite:
    return NotificationSite(monkeypatch, tmp_path)


def test_control_worker_pod_requires_a_ready_pod() -> None:
    target = SimpleNamespace(cluster_id="unit")
    empty = SimpleNamespace(ready_pods=lambda *a: [])
    with pytest.raises(runner.NotificationAcceptanceError, match="no Ready control"):
        runner.control_worker_pod(empty, target)  # type: ignore[arg-type]
    ready = SimpleNamespace(ready_pods=lambda *a: ["worker-0", "worker-1"])
    assert runner.control_worker_pod(ready, target) == "worker-0"  # type: ignore[arg-type]


def test_live_action_completed_records_must_cover_every_recorded_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner,
        "action_completed_records_from_evidence",
        lambda run_dir, **k: {"gpu-reset": [{"notification_id": "n-1"}]},
    )
    reads: list[tuple[Any, ...]] = []

    def pod_json(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        reads.append(arguments)
        return {"records": [{"notification_id": "n-other"}]}

    site = SimpleNamespace(pod_json=pod_json)
    target = SimpleNamespace(cluster_id="unit")
    with pytest.raises(
        runner.NotificationAcceptanceError, match="reads are incomplete"
    ):
        runner.live_action_completed_records(
            site,  # type: ignore[arg-type]
            target,  # type: ignore[arg-type]
            run_dir=tmp_path,
            pod="worker-0",
        )
    assert len(reads) == 1 and "n-1" in reads[0], "the live read names the recorded id"


def test_live_action_completed_records_skip_the_store_without_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner, "action_completed_records_from_evidence", lambda run_dir, **k: {}
    )

    def forbidden(*_arguments: Any, **_keywords: Any) -> Any:
        raise AssertionError("no candidates means no live read")

    result = runner.live_action_completed_records(
        SimpleNamespace(pod_json=forbidden),  # type: ignore[arg-type]
        SimpleNamespace(cluster_id="unit"),  # type: ignore[arg-type]
        run_dir=tmp_path,
        pod="worker-0",
    )
    assert result == {"candidates": {}, "records": []}


def test_notify003_focused_tests_reuse_the_plan_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_arguments: Any, **_keywords: Any) -> Any:
        raise AssertionError("a reusable plan result must not rerun pytest")

    monkeypatch.setattr(runner, "run", forbidden)
    monkeypatch.setattr(
        runner, "reusable_focused_tests", lambda path: {"passed": True, "returncode": 0}
    )
    assert runner.notify003_focused_tests(tmp_path, reuse=True) == {
        "passed": True,
        "returncode": 0,
        "focused_tests_reused": True,
    }


def test_low_utilization_metadata_requires_the_expected_critical_rank_count() -> None:
    def template(role: str, offset: str) -> dict[str, Any]:
        return {
            "template": {
                "metadata": {
                    "labels": {"gpu-fault.io/role": role},
                    "annotations": {
                        "gpu-fault.io/rank-offset": offset,
                        "gpu-fault.io/expected-critical-ranks": "3",
                    },
                }
            }
        }

    workload = {
        "kind": "PyTorchJob",
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": template("master", "0"),
                "Worker": template("worker", "1"),
            }
        },
    }
    assert runner.low_utilization_metadata_errors(workload, expected_pods=3) == []
    assert runner.low_utilization_metadata_errors(workload, expected_pods=2) == [
        "Master expected critical ranks differs",
        "Worker expected critical ranks differs",
    ]


def install_cli(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(runner, "IdentitySite", lambda _: site)
    monkeypatch.setattr(runner, "predecessor_path", lambda *a: (None, None))
    monkeypatch.setattr(
        runner,
        "authorize_execution",
        lambda *a, **kw: events.append("authorize") or DEADLINE,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unit",
            "--run-dir",
            str(site.path),
            "--site",
            str(site.site_file),
            *arguments,
        ],
    )
    return events


def test_notify002_is_superseded_and_never_executes(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_cli(site, monkeypatch, ["--case", "GF-REGIONAL-NOTIFY-002", "--plan"])
    with pytest.raises(runner.NotificationAcceptanceError, match="superseded"):
        runner.main()
    assert not (site.path / "cases").exists(), "a refused case creates no directory"


@pytest.mark.parametrize(
    ("nodes", "image", "fragment"),
    [
        ((NODES[0], NODES[0], NODES[1]), IMMUTABLE_IMAGE, "three distinct --node"),
        (NODES, "registry.invalid/unit:latest", "immutable digest"),
    ],
)
def test_notify005_requires_three_distinct_nodes_and_an_immutable_image(
    site: NotificationSite,
    monkeypatch: pytest.MonkeyPatch,
    nodes: tuple[str, ...],
    image: str,
    fragment: str,
) -> None:
    arguments = [
        "--case",
        "GF-REGIONAL-NOTIFY-005",
        "--plan",
        "--training-image",
        image,
    ]
    for node in nodes:
        arguments.extend(["--node", node])
    install_cli(site, monkeypatch, arguments)
    with pytest.raises(runner.NotificationAcceptanceError, match=fragment):
        runner.main()
    assert not (site.path / "cases").exists(), "a refused case creates no directory"


def test_execute_refuses_a_confirmation_that_does_not_match_the_case(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = install_cli(
        site,
        monkeypatch,
        [
            "--case",
            "GF-REGIONAL-NOTIFY-001",
            "--execute",
            "--confirm",
            "NOTIFY003_EXECUTE",
        ],
    )
    with pytest.raises(
        runner.NotificationAcceptanceError, match="confirmation must be"
    ):
        runner.main()
    assert events == [], "a wrong confirmation must never reach authorization"


def test_execute_refuses_a_predecessor_that_did_not_pass(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = install_cli(
        site,
        monkeypatch,
        [
            "--case",
            "GF-REGIONAL-NOTIFY-001",
            "--execute",
            "--confirm",
            "NOTIFY001_EXECUTE",
        ],
    )
    monkeypatch.setattr(
        runner,
        "predecessor_path",
        lambda *a: ("GF-REGIONAL-HA-004", site.path / "predecessor.json"),
    )
    monkeypatch.setattr(
        runner,
        "predecessor_evidence",
        lambda *a, **k: {"valid": False, "case_id": "GF-REGIONAL-HA-004"},
    )
    with pytest.raises(
        runner.NotificationAcceptanceError, match="predecessor evidence"
    ):
        runner.main()
    assert events == ["authorize"], "the refusal comes after the window is approved"
    assert not (site.path / "cases" / "GF-REGIONAL-NOTIFY-001").exists(), (
        "an invalid predecessor stops before the case directory is created"
    )


def test_notify003_focused_tests_without_a_case_dir_write_no_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_keywords: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=3, stdout="collected", stderr="")

    monkeypatch.setattr(runner, "run", run)
    result = runner.notify003_focused_tests()
    assert result["passed"] is False
    assert result["returncode"] == 3
    assert result["focused_tests_reused"] is False
    assert commands == [result["command"]]
    assert list(tmp_path.iterdir()) == [], "no case directory means no log file"
