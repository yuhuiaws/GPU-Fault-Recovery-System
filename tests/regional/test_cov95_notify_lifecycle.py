from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureAbort
from tests.regional._cov95_ha001_harness import DEADLINE
from tests.regional._cov95_notify_harness import NODES, NotificationSite


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NotificationSite:
    return NotificationSite(monkeypatch, tmp_path)


@pytest.mark.parametrize("live", [False, True])
def test_completion_case_uses_real_handler_replay_and_current_store_records(
    site: NotificationSite, live: bool
) -> None:
    if live:
        site.live_record("gpu-reset")
        site.live_record("workload-restart")
    result = site.run_completion()
    assert result["verdict"] == "PASS", result["checks"]
    assert result["sources"] == {
        kind: "live" if live else "drill" for kind in runner.NOTIFICATION_KINDS
    }
    assert len(site.notifier.sent) == (1 if live else 3)
    assert result["dedup_drill"]["notifier_calls"] == 1
    assert result["dedup_drill"]["command_statuses"] == ["SUCCEEDED"] * 4
    assert result["dedup_drill"]["http_authorization_exercised"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"drill_id": "foreign"},
        {"executed_at": "invalid"},
        {"completed_at": "invalid"},
        {"completed_at": "2000-01-01T00:00:00Z"},
    ],
)
def test_completion_rejects_unbound_or_backwards_drill_evidence(
    site: NotificationSite, override: dict[str, Any]
) -> None:
    site.drill_override = override
    with pytest.raises(runner.NotificationAcceptanceError, match="interval|identity"):
        site.run_completion()
    assert len(site.notifier.sent) == 1


def test_expired_notification_window_prevents_provider_invocation(
    site: NotificationSite,
) -> None:
    site.deadline = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(runner.NotificationAcceptanceError, match="window ended"):
        site.run_completion()
    assert site.notifier.sent == []


@pytest.mark.parametrize("defect", ["missing", "bad-time"])
def test_current_store_evidence_cannot_be_replaced_by_a_drill(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    value = site.live_record("gpu-reset")
    if defect == "missing":
        monkeypatch.setattr(
            site.store,
            "get_notification",
            lambda _: (_ for _ in ()).throw(KeyError("unit missing")),
        )
    else:
        monkeypatch.setattr(
            site.store,
            "get_notification",
            lambda _: value.model_copy(update={"created_at": datetime(2026, 1, 1)}),
        )
    with pytest.raises(
        runner.NotificationAcceptanceError, match="cannot replace|creation time"
    ):
        site.run_completion()
    assert site.notifier.sent == []


@pytest.mark.parametrize(
    "defect", ["none", "disabled", "mixed-pod", "zero-worker", "tests"]
)
def test_notification_config_resolves_envfrom_and_checks_every_replica(
    site: NotificationSite, defect: str
) -> None:
    if defect == "disabled":
        for value in site.env.values():
            value["dispatcher_enabled"] = "false"
    elif defect == "mixed-pod":
        site.observed_override["async_delivery"] = "false"
    elif defect == "zero-worker":
        site.replicas["gpu-fault-control-worker"] = 0
        site.ready_counts["gpu-fault-control-worker"] = 0
    elif defect == "tests":
        site.focused_code = 1
    result = runner.run_notify003(site, site.cluster, case_dir=site.path)
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert site.events == ["focused-tests"]
    assert (site.path / "focused-tests.log").is_file(), (
        "focused test evidence must be persisted"
    )
    assert result["deployment_config"]["gpu-fault-api-ha"]["service_role"] == "ingress"


@pytest.mark.parametrize("mode", ["empty", "expired", "allowed", "denied"])
def test_executor_notification_denial_is_bounded_and_explicit(
    site: NotificationSite, mode: str
) -> None:
    if mode == "empty":
        site.gpu_pods = []
    elif mode == "allowed":
        site.denial = {"result": "ALLOWED"}
    deadline = (
        datetime(2000, 1, 1, tzinfo=timezone.utc) if mode == "expired" else DEADLINE
    )
    if mode in {"empty", "expired"}:
        with pytest.raises(
            runner.NotificationAcceptanceError, match="Executor Pod|window ended"
        ):
            runner.run_notify004(site, site.cluster, maintenance_window_end=deadline)
        assert site.events == []
    else:
        result = runner.run_notify004(
            site, site.cluster, maintenance_window_end=deadline
        )
        assert result["verdict"] == ("PASS" if mode == "denied" else "FAIL")
        assert site.events == ["denial:executor-a", "denial:executor-b"]


def test_two_phase_aggregation_waits_for_rearm_and_cleans_owned_workloads(
    site: NotificationSite,
) -> None:
    result = site.run_aggregation(planned_nodes=site.nodes)
    assert result["verdict"] == "PASS", result.get("error")
    assert result["cleanup_errors"] == []
    assert [len(value["nodes"]) for value in result["observations"]] == [1, 3]
    assert all(len(value["quiesce_polls"]) == 2 for value in result["observations"]), (
        "each phase must observe latch rearming"
    )
    assert len(site.fixtures) == 2
    assert len(set(site.ownership_paths)) == 2
    assert all(path.parent == site.path for path in site.ownership_paths), (
        "test_two_phase_aggregation_waits_for_rearm_and_cleans_owned_workloads: expected all(path.parent == site.path for path in site.ownership_p..."
    )
    assert site.resources == {}
    assert all(fixture.deleted for fixture in site.fixtures), (
        "both owned workloads must be absent"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "planned-uid",
        "late-uid",
        "foreign-workload",
        "submit-ack",
        "silent",
        "duplicate",
        "metadata",
        "cleanup",
        "expired",
    ],
)
def test_aggregation_stops_on_identity_or_evidence_failure(
    site: NotificationSite, defect: str
) -> None:
    planned = {node: dict(value) for node, value in site.nodes.items()}
    if defect == "planned-uid":
        planned[NODES[0]]["uid"] = "different"
        with pytest.raises(runner.NotificationAcceptanceError, match="approved plan"):
            site.run_aggregation(planned_nodes=planned)
        assert site.fixtures == []
        return
    if defect == "late-uid":
        site.node_drift_after = 4
    elif defect == "foreign-workload":
        site.foreign = [
            {
                "metadata": {"name": "foreign", "namespace": "training"},
                "spec": {
                    "nodeName": NODES[0],
                    "containers": [{"resources": {"limits": {"nvidia.com/gpu": "1"}}}],
                },
                "status": {"phase": "Pending"},
            }
        ]
    elif defect == "submit-ack":
        site.submit_failure = True
    elif defect == "silent":
        site.silent = True
    elif defect == "duplicate":
        site.duplicate = True
    elif defect == "metadata":
        site.bad_metadata = True
    elif defect == "cleanup":
        site.delete_failure = True
    else:
        site.deadline = datetime(2000, 1, 1, tzinfo=timezone.utc)
    result = site.run_aggregation(planned_nodes=planned)
    assert result["verdict"] == "FAIL"
    assert result["error"], f"{defect} must produce an actionable failure"
    assert len(site.fixtures) <= (2 if defect == "metadata" else 1)
    if defect != "cleanup":
        assert site.resources == {}
    else:
        assert result["cleanup_errors"], "failed deletion must prevent PASS"
    if defect == "silent":
        assert result["observations"][0]["silent_nodes"] == [NODES[0]]


def test_operator_abort_cleans_the_known_workload_and_propagates(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    def abort(*args: Any, **kwargs: Any) -> Any:
        raise RegionalFixtureAbort(15)

    monkeypatch.setattr(runner, "wait_running_pods", abort)
    with pytest.raises(RegionalFixtureAbort):
        site.run_aggregation()
    assert len(site.fixtures) == 1
    assert site.resources == {}


@pytest.mark.parametrize(
    "case_id",
    [
        "GF-REGIONAL-NOTIFY-001",
        "GF-REGIONAL-NOTIFY-003",
        "GF-REGIONAL-NOTIFY-004",
        "GF-REGIONAL-NOTIFY-005",
    ],
)
def test_main_executes_each_owned_notification_case_with_fake_authorization(
    site: NotificationSite, monkeypatch: pytest.MonkeyPatch, case_id: str
) -> None:
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(runner, "IdentitySite", lambda _: site)
    monkeypatch.setattr(runner, "predecessor_path", lambda *a: (None, None))
    monkeypatch.setattr(runner, "authorize_execution", lambda *a, **kw: DEADLINE)
    directory = site.path / "cases" / case_id
    directory.mkdir(parents=True)
    (directory / "plan.json").write_text(
        json.dumps({"details": {"node_baseline": site.nodes}})
    )
    arguments = [
        "unit",
        "--run-dir",
        str(site.path),
        "--site",
        str(site.site_file),
        "--case",
        case_id,
        "--execute",
        "--confirm",
        case_id.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE",
        "--receipt-evidence",
        str(site.receipt),
        "--ses-window-evidence",
        str(site.dedup),
        "--training-image",
        "registry.invalid/unit@sha256:" + "a" * 64,
    ]
    for node in NODES:
        arguments.extend(["--node", node])
    monkeypatch.setattr(sys, "argv", arguments)
    assert runner.main() == 0
    result = json.loads((directory / f"{case_id}.json").read_text())
    assert result["verdict"] == "PASS"
    assert result["release_id"] == "unit-release"
    assert result["cluster_id"] == "unit-cluster"
