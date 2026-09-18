from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr001_gpu_reset as reset
from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import run_destr008_warm_spare_shortage as shortage
from scripts.e2e.regional import run_destr010_fabric_manager_restart as fabric
from scripts.e2e.regional import run_destr023_idle_cluster_reset as idle
from scripts.e2e.regional import run_destr024_watcher_down_fail_closed as watcher
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_actions import ActionHarness
from tests.regional._cov95_destr_idle import IdleHarness
from tests.regional._cov95_destr_managed import ManagedHarness
from tests.regional._cov95_destr_warm import WarmHarness
from tests.regional.test_cov95_destr_warm_runners import scenario_run


@pytest.mark.parametrize("module", [reset, idle])
@pytest.mark.parametrize(
    ("node_change", "expected"),
    [
        ({"ready": "False"}, "not Ready after reset"),
        (
            {"ownership_annotations": {"owner": "unrestored"}},
            "ownership was not restored",
        ),
    ],
)
def test_reset_terminal_success_cannot_hide_unhealthy_or_owned_node(
    module: Any,
    node_change: dict[str, Any],
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.regional.wait_for_workflow

    def workflow(**kwargs: Any) -> dict[str, Any]:
        state = original(**kwargs)
        h.node.update(node_change)
        return state

    monkeypatch.setattr(h.regional, "wait_for_workflow", workflow)
    code, report = h.execute(tmp_path)
    assert code == 1 and any(expected in error for error in report["errors"]), report
    if module is reset:
        assert report["quiesce_recovery"]["product_restoration_observed"] is True
        assert report["quiesce_recovery"]["runner_restore_attempted"] is False
    else:
        assert report["quiesce_recovery"]["ok"] is True, report
    assert any(name == "host.cleanup" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("module", [reset, idle])
def test_expiry_after_sampler_start_stops_sampler_without_injecting(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.advance_at["host.start-reset-sampler"] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "before XID injection" in report["error"], report
    assert h.injected is False, h.calls
    assert any(name == "host.stop-reset-sampler" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("module", [idle, watcher])
def test_managed_work_appearing_after_preflight_prevents_idle_injection(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.host.execute

    def execute(action: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original(action, *args, **kwargs)
        if action == "snapshot" and not args:
            h.pods = [
                {
                    "metadata": {
                        "name": "new-workload",
                        "namespace": "training",
                        "labels": {idle.MANAGED_LABEL: "true"},
                    },
                    "spec": {"nodeName": "node-a"},
                    "status": {"phase": "Running"},
                }
            ]
        return value

    monkeypatch.setattr(h.host, "execute", execute)
    code, report = h.execute(tmp_path)
    assert code == 1 and "appeared during the wait" in report["error"], report
    assert h.injected is False, h.calls
    assert h.watcher_replicas == 1, h.calls


def test_expiry_after_watcher_scale_down_restores_watcher_without_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.advance_at["watcher.down"] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "before injection" in report["error"], report
    assert not h.injected and h.watcher_replicas == 1, h.calls
    assert any(name == "watchdog.kill" for name, _ in h.calls), h.calls


def test_missing_containment_does_not_invent_an_isolation_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher, tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.regional.wait_for_workflow

    def workflow(**kwargs: Any) -> dict[str, Any]:
        state = original(**kwargs)
        h.node.update(unschedulable=False, taints=[], ownership_annotations={})
        return state

    monkeypatch.setattr(h.regional, "wait_for_workflow", workflow)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["errors"], report
    assert not any(name == "restore.create" for name, _ in h.calls), h.calls
    assert h.watcher_replicas == 1, h.calls


@pytest.mark.parametrize("module", [reset, idle, watcher])
def test_unknown_final_node_state_cannot_certify_cleanup(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.host.cleanup

    def cleanup() -> dict[str, bool]:
        result = original()
        h.failures["node.snapshot"] = RuntimeError("fake final read unavailable")
        return result

    monkeypatch.setattr(h.host, "cleanup", cleanup)
    code, report = h.execute(tmp_path)
    assert code == 1 and "final read unavailable" in report["postflight_error"], report
    assert report["probe_residuals"] == {"pod": False, "configmap": False}, report


@pytest.mark.parametrize("defect", ["ledger", "final-node"])
def test_fabric_replay_and_final_pass_require_exact_node_action_identity(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(fabric, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "ledger":

        def snapshot(result: dict[str, Any], args: Any, _kwargs: Any) -> Any:
            if args[0] == "snapshot" and len(args) > 1:
                result["ledger"] = []
            return result

        h.transforms["host.execute"] = snapshot
    else:

        def cleanup(result: Any, _args: Any, _kwargs: Any) -> Any:
            h.node["uid"] = "replacement"
            return result

        h.transforms["host.cleanup"] = cleanup
    code, report = h.execute(tmp_path)
    assert code == 1 and report["errors"], report
    if defect == "ledger":
        assert (
            "expected Node Agent command ID is absent from the ledger"
            in report["errors"]
        ), report
        assert not any(name == "regional.executor_python" for name, _, _ in h.calls), (
            h.calls
        )
    else:
        assert report["final_node"]["uid"] == "replacement", report


@pytest.mark.parametrize("module", [failover, shortage])
def test_warm_case_expiry_before_prewarm_still_audits_owned_cleanup(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.clock.sleep(61)
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "before image prewarm" in report["error"], report
    names = [name for name, _ in h.calls]
    assert "prewarm.create" not in names and "prewarm.cleanup" in names, names


@pytest.mark.parametrize("module", [failover, shortage])
def test_warm_trigger_non_success_receipt_is_not_workflow_success(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(module, tmp_path, monkeypatch)
    h.injection_status = 503
    if module is failover:
        h.plan(tmp_path)
        code, report = h.execute(tmp_path)
        assert code == 1, report
    else:
        report = scenario_run(h, tmp_path, "no-spare")
    assert report["verdict"] == "FAIL" and "endpoint failed" in report["error"], report
    assert not any(name == "workflow.wait" for name, _ in h.calls), h.calls
    assert any(name == "incident.lookup" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("module", [failover, shortage])
def test_warm_provider_inventory_drift_is_rejected_even_with_expected_workflow(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(module, tmp_path, monkeypatch)
    original = h.warm.wait_for_workflow

    def workflow(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        h.provider["count"] = 3
        return result

    monkeypatch.setattr(h.warm, "wait_for_workflow", workflow)
    if module is failover:
        h.plan(tmp_path)
        code, report = h.execute(tmp_path)
        assert code == 1, report
    else:
        report = scenario_run(h, tmp_path, "no-spare")
    assert report["verdict"] == "FAIL", report
    assert "HyperPod provider inventory changed" in report["errors"], report


def test_failover_requires_fault_agent_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.workflow["agents"] = []
    code, report = h.execute(tmp_path)
    assert code == 1 and "fault Node Agent was not revoked" in report["errors"], report


@pytest.mark.parametrize("defect", ["cpu", "residual", "cleanup-error"])
def test_shortage_matrix_never_hides_postflight_or_prewarm_failure(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "cpu":
        original = h.regional.cpu_blast_snapshot

        def snapshot() -> dict[str, Any]:
            value = original()
            return {"nodes": ["changed"]} if h.posted else value

        monkeypatch.setattr(h.regional, "cpu_blast_snapshot", snapshot)
    elif defect == "residual":
        h.prewarm_residuals["pods"] = True
    else:
        h.failures["prewarm.cleanup"] = RuntimeError("fake prewarm cleanup failed")
    code, report = h.execute(tmp_path)
    assert code == 1, report
    if defect == "cpu":
        assert "control-plane EKS state differs from baseline" in report["errors"], (
            report
        )
    else:
        assert any(report["prewarm_residuals"].values()), report
    assert report["complete_matrix"] is False, report


def test_shortage_provider_replacement_event_is_never_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.events = [{"event_name": "BatchReplaceClusterNodes"}]
    report = scenario_run(h, tmp_path, "no-spare")
    assert report["verdict"] == "FAIL", report
    assert "provider replace/delete mutation appeared" in report["errors"], report


@pytest.mark.parametrize("rerun", [False, True])
def test_managed_expiry_after_prewarm_prevents_group_submission(
    rerun: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch, rerun=rerun)
    h.plan(tmp_path)
    h.advance_at["prewarm.create"] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert (
        code == 1
        and ("group A failed" if rerun else "group D failed") in report["error"]
    ), report
    assert not any(name.endswith(".submit") for name, _, _ in h.calls), h.calls


@pytest.mark.parametrize("defect", ["cache", "provider", "cpu"])
def test_managed_case_rejects_prewarm_failure_and_postflight_blast_changes(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "cache":
        h.transforms["prewarm.cached_nodes"] = lambda *_a: []
    elif defect == "provider":
        h.transforms["regional.provider_events"] = lambda *_a: [
            {"event_name": "RebootClusterNodes"}
        ]
    else:
        h.transforms["regional.cpu_blast_snapshot"] = lambda *_a: {"changed": True}
    code, report = h.execute(tmp_path)
    assert code == 1, report
    if defect == "cache":
        assert "not cached" in report["error"], report
        assert not any(job.submitted for job in h.jobs.values()), h.jobs
        assert not any(name.endswith(".submit") for name, _, _ in h.calls), h.calls
        assert report["workload_residuals"] == {"D": False}, report
    else:
        expected = "provider mutation" if defect == "provider" else "control-plane EKS"
        assert any(expected in error for error in report["errors"]), report


@pytest.mark.parametrize("defect", ["auto-resume", "receipt"])
def test_managed_group_a_refusal_prevents_group_d(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch, rerun=True)
    h.plan(tmp_path)
    if defect == "auto-resume":

        def running(result: dict[str, Any], *_args: Any) -> dict[str, Any]:
            result["workload"]["annotations"][h.module.AUTO_RESUME_ANNOTATION] = "true"
            return result

        h.transforms["workload.job-a.wait_running"] = running
    else:
        h.transforms["regional.post_xid_event"] = lambda *_a: {
            "receipt": {"status": 503}
        }
    code, report = h.execute(tmp_path)
    assert code == 1 and "group A failed" in report["error"], report
    assert not any(name == "workload.job-d.submit" for name, _, _ in h.calls), h.calls
    assert h.jobs["job-a"].deleted, h.jobs


def test_group_d_refusal_must_leave_job_suspend_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    snapshots = 0

    def snapshot(result: dict[str, Any], *_args: Any) -> dict[str, Any]:
        nonlocal snapshots
        snapshots += 1
        if snapshots == 2:
            result["workload"]["suspend"] = True
        return result

    h.transforms["workload.job-d.snapshot"] = snapshot
    code, report = h.execute(tmp_path)
    assert code == 1 and "group D failed" in report["error"], report
    posts = [
        args[0]["record_id"]
        for name, args, _ in h.calls
        if name == "regional.post_xid_event"
    ]
    assert len(posts) == 1 and "-block-" in posts[0], posts


@pytest.mark.parametrize("defect", ["clients", "kmsg", "commands"])
def test_reboot_execute_refuses_invalid_baseline_or_ambiguous_submission(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "commands":

        def store(result: dict[str, Any], *_args: Any) -> dict[str, Any]:
            result["commands"] *= 2
            return result

        h.transforms["regional.store_snapshot"] = store
    else:

        def snapshot(result: dict[str, Any], args: Any, _kwargs: Any) -> dict[str, Any]:
            if args[0] == "snapshot":
                result[
                    "compute_clients" if defect == "clients" else "kmsg_writable"
                ] = ["foreign"] if defect == "clients" else False
            return result

        h.transforms["host.execute"] = snapshot
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert not any(
        name == "regional.kubectl" and "delete" in args for name, args, _ in h.calls
    ), h.calls


@pytest.mark.parametrize("module", [reset, fabric, reboot])
def test_execute_preflight_failure_never_creates_host_probe(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if module is reset:
        h: Any = IdleHarness(module, tmp_path, monkeypatch)
        h.plan(tmp_path)
        h.tests_pass = False
    else:
        h = ActionHarness(module, tmp_path, monkeypatch)
        h.plan(tmp_path)
        h.preflight["errors"] = ["fake unproven preflight"]
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not any(row[0] == "host.create" for row in h.calls), h.calls
