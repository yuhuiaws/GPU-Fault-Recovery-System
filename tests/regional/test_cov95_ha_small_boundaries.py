from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import ha009_refresh as refresh
from scripts.e2e.regional import ha010_verdicts as verdicts
from scripts.e2e.regional import ha_evidence, ha_kubernetes
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004


@pytest.mark.parametrize(
    "module,function",
    [(refresh, "stop_refresh_watchdog"), (ha004, "stop_process_group")],
)
@pytest.mark.parametrize("mode", ["none", "fired", "term", "kill", "error"])
def test_watchdog_stop_handles_every_process_outcome_without_real_signals(
    monkeypatch: pytest.MonkeyPatch, module: Any, function: str, mode: str
) -> None:
    signals, waits = [], []

    def signal(pid: int, value: int) -> None:
        signals.append((pid, value))
        if mode == "error":
            raise PermissionError("unit denied")

    def wait(**kwargs: Any) -> int:
        waits.append(kwargs["timeout"])
        if mode == "kill" and len(waits) == 1:
            raise subprocess.TimeoutExpired("unit", 10)
        return 0

    process = SimpleNamespace(
        pid=12345, returncode=0, poll=lambda: 0 if mode == "fired" else None, wait=wait
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(killpg=signal))
    result = getattr(module, function)(None if mode == "none" else process)
    if mode == "none":
        assert result == {"armed": False}
    elif mode == "fired":
        assert result == {"armed": True, "fired": True, "returncode": 0}
    elif mode == "error":
        assert "PermissionError" in result["stop_error"]
    else:
        assert result["disarmed"] is True
        assert waits == ([10, 10] if mode == "kill" else [10])
        assert len(signals) == (2 if mode == "kill" else 1)


def test_refresh_watchdog_refuses_a_disappeared_owned_job(tmp_path: Path) -> None:
    created = []
    resources = SimpleNamespace(create=created.append, owned=lambda *a: None)
    manifest = {"metadata": {"name": "unit-job"}, "spec": {}}
    with pytest.raises(refresh.CaseError, match="Job disappeared"):
        refresh.start_refresh_watchdog(
            tmp_path, manifest, resources, resume_command=lambda *a: [], delay_seconds=1
        )
    assert created[0]["spec"]["suspend"] is True


@pytest.mark.parametrize("fault", ["gone", "empty-log", "ambiguous-log"])
def test_refresh_result_requires_an_owned_job_and_one_safe_outcome(
    tmp_path: Path, fault: str
) -> None:
    resources = SimpleNamespace(
        create=lambda _: None,
        owned=lambda *a: None if fault == "gone" else {"metadata": {"uid": "unit"}},
    )

    def control(*args: str, **kwargs: Any) -> str:
        if args[0] == "get":
            return json.dumps({"status": {"succeeded": 1}})
        if args[0] == "logs" and fault == "ambiguous-log":
            return "rotated=True restarted=False\nrotated=False restarted=False"
        return ""

    with pytest.raises(refresh.CaseError, match="disappeared|unambiguous"):
        refresh.run_refresh_job(
            tmp_path,
            {"metadata": {"name": "unit"}},
            resources,
            control=control,
            timeout_seconds=200,
        )


def test_kubernetes_mutations_require_uid_owner_and_valid_scale() -> None:
    with pytest.raises(RuntimeError, match="identity is incomplete"):
        ha_kubernetes.delete_pod(lambda *a: None, "", {"uid": "unit", "name": "pod"})
    with pytest.raises(RuntimeError, match="before cordon"):
        ha_kubernetes.node_cordon_patch({"uid": "different"}, uid="unit", owner="unit")
    with pytest.raises(RuntimeError, match="not restorable"):
        ha_kubernetes.node_restore_patch({"unschedulable": True}, owner="unit")
    with pytest.raises(RuntimeError, match="positive replica"):
        ha_kubernetes.scale_patch("unit", 1, False)
    baseline = {
        "uid": "unit",
        "resource_version": "1",
        "unschedulable": False,
        "ha_owner": None,
        "annotations_present": True,
        "taints_present": False,
    }
    operations = ha_kubernetes.node_cordon_patch(baseline, uid="unit", owner="unit")
    assert [item["path"] for item in operations] == [
        "/metadata/uid",
        "/metadata/resourceVersion",
        ha_kubernetes.NODE_OWNER_PATH,
        "/spec/unschedulable",
    ]
    assert (
        ha_kubernetes.node_restore_patch(baseline, owner="unit")[-1]["op"] == "remove"
    )


@pytest.mark.parametrize("defect", ["identity", "predecessor"])
def test_chain_binding_cannot_move_to_another_identity_or_predecessor(
    defect: str,
) -> None:
    planned = {"identity": {"release_id": "unit"}, "predecessor": {"case_id": "old"}}
    current = {
        "errors": [],
        "identity": {"release_id": "unit"},
        "predecessor": {"case_id": "old", "valid": True},
    }
    if defect == "identity":
        current["identity"]["release_id"] = "other"
    else:
        current["predecessor"]["case_id"] = "other"
    with pytest.raises(RuntimeError, match="changed since planning"):
        ha_evidence.require_chain(planned, current)


def test_isolated_chain_does_not_accept_failed_predecessor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ha_evidence, "predecessor_path", lambda *a: ("previous", tmp_path / "previous")
    )
    monkeypatch.setattr(
        ha_evidence, "predecessor_evidence", lambda *a, **kw: {"valid": False}
    )
    with pytest.raises(RuntimeError, match="predecessor is not PASS"):
        ha_evidence.isolated_chain(
            SimpleNamespace(release_id="unit", cluster_id="unit", run_dir=tmp_path),
            "unit-case",
        )


def test_timeline_time_parsing_and_incomplete_replacement_fail_closed() -> None:
    now = datetime.now(timezone.utc)
    assert verdicts.parse_time(now) == now
    assert verdicts.parse_time(datetime(2026, 1, 1)).tzinfo == timezone.utc
    assert verdicts.parse_time("invalid") is None
    errors = verdicts.sampler_errors(
        [{"t": float("nan")}],
        pod="unit",
        stale_seconds=90,
        rds_available_at=now,
        failover_requested_at=now,
    )
    assert errors == ["unit: the probe has invalid sample timestamps"]
    errors = verdicts.replacement_errors(
        {
            "name": "new",
            "ready": True,
            "ready_at": (now - timedelta(seconds=1)).isoformat(),
        },
        deleted_pod="old",
        deleted_at=now,
        budget_seconds=30,
    )
    assert errors == [
        "replacement Pod has no UID",
        "replacement Pod new was Ready before the deletion",
    ]
    assert (
        verdicts.readiness_recovery_at(
            [{"t": now.timestamp(), "healthz": 503}], rds_available_at=now
        )
        is None
    )
