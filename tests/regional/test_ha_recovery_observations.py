from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.e2e.regional.probes import ha006_executor


@pytest.mark.parametrize("module", [ha003, ha004])
def test_wait_reports_cleanup_identity_before_a_store_read_failure(
    module, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = iter(
        [
            {"incident": {"incident_id": "owned-incident"}, "commands": []},
            RuntimeError("Store unavailable"),
        ]
    )

    def snapshot(**kwargs):
        state = next(states)
        if isinstance(state, Exception):
            raise state
        return state

    regional = SimpleNamespace(store_snapshot=snapshot)
    observed = []
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    kwargs = {
        "marker": "unit-marker",
        "observed_after": datetime.now(timezone.utc),
        "timeout_seconds": 10,
        "observe_state": lambda state: observed.append(
            state["incident"]["incident_id"]
        ),
    }
    with pytest.raises(RuntimeError, match="Store unavailable"):
        if module is ha003:
            module.wait_reset_claim(regional, SimpleNamespace(node="node-a"), **kwargs)
        else:
            module.command_timeline(
                regional,
                SimpleNamespace(node="node-a"),
                kill_owner=lambda _: None,
                **kwargs,
            )
    assert observed == ["owned-incident"]


def test_same_executor_replay_is_serialized_and_does_not_repeat_the_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ha006_executor, "WINNER", tmp_path / "winner.json")
    monkeypatch.setattr(ha006_executor.socket, "gethostname", lambda: "unit-pod")
    store = InMemoryStore()
    adapter = ha006_executor.SharedLedgerAdapter(
        SimpleNamespace(save_notification_if_absent=store.save_notification_if_absent),
        run_id="unit-run",
        sleep_seconds=0.01,
    )
    context = SimpleNamespace(
        idempotency_key="workflow/0",
        incident=SimpleNamespace(cluster_id="c", incident_id="i"),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(adapter.execute, [context, context]))
    assert adapter.physical_actions == 1
    assert sorted(result.details["cached"] for result in results) == [False, True]
    assert len(store.list_notifications()) == 1


def test_executor_cleanup_continues_after_log_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    present = set(ha006.PODS)
    calls = []

    def dataplane(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("get", "pod"):
            return args[2] if args[2] in present else ""
        if args[0] == "logs":
            raise RuntimeError("log transport unavailable")
        if args[:2] == ("delete", "pod"):
            present.discard(args[2])
        return ""

    monkeypatch.setattr(ha006, "dataplane", dataplane)
    monkeypatch.setattr(ha006, "cleanup_seed", lambda *a: {"deleted": {}})
    monkeypatch.setattr(ha006, "teardown", lambda **kw: calls.append(("teardown",)))
    monkeypatch.setattr(ha006, "database_residuals", lambda: {"total": 0})
    monkeypatch.setattr(ha006, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(ha006, "kubernetes_residuals", lambda: {"count": 0})
    result = {"verdict": "PASS"}
    resources = SimpleNamespace(
        records={f"Pod/{pod}": {} for pod in ha006.PODS},
        owned=lambda kind, name: name if name in present else None,
        delete=lambda kind, name: dataplane("delete", kind.lower(), name),
    )
    ha006.cleanup_case(
        tmp_path, "unit", {"command_id": "c"}, "n", result, resources=resources
    )
    assert result["verdict"] == "FAIL"
    assert not present, {"remaining_pods": sorted(present), "calls": calls}
    assert ("teardown",) in calls
    assert "postflight" in result
    assert len(result["cleanup_errors"]) == 2


def test_unknown_executor_shutdown_preserves_its_durable_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("Pod API unavailable")

    monkeypatch.setattr(ha006, "dataplane", unavailable)
    calls = []
    monkeypatch.setattr(ha006, "teardown", lambda **kw: calls.append("teardown"))
    monkeypatch.setattr(ha006, "cleanup_seed", lambda *a: calls.append("seed"))
    result = {"verdict": "PASS"}
    resources = SimpleNamespace(
        records={f"Pod/{pod}": {} for pod in ha006.PODS}, owned=unavailable
    )
    ha006.cleanup_case(
        tmp_path, "unit", {"command_id": "c"}, "n", result, resources=resources
    )
    assert result["verdict"] == "FAIL"
    assert result["cleanup_preserved"]
    assert calls == []


@pytest.mark.parametrize("module", [ha005, ha006, ha009])
def test_preflight_refusal_does_not_delete_foreign_fixture_resources(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = module.BASE if module is ha009 else module
    monkeypatch.setattr(base, "database_residuals", lambda: {"total": 1})
    monkeypatch.setattr(base, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(base, "kubernetes_residuals", lambda: {"count": 1})
    calls = []
    monkeypatch.setattr(base, "dataplane", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(base, "teardown", lambda **kw: calls.append(("teardown",)))
    assert (
        module.run_case(
            tmp_path,
            1,
            datetime(2099, 1, 1, tzinfo=timezone.utc),
            **({"all_deployments": True} if module is ha005 else {}),
        )
        == 1
    )
    assert calls == []
    report = json.loads(
        (tmp_path / "cases" / module.CASE_ID / f"{module.CASE_ID}.json").read_text()
    )
    assert report["verdict"] == "FAIL"
