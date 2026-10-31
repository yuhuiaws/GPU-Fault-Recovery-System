"""More category edge branches and direct unit coverage of the helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_common as common
from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional import general_live_restart_outcomes as outcomes
from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    make_settings,
    store_script,
    workflow,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
EMPTY = {
    "incidents": [],
    "workflows": [],
    "batches": [],
    "markers": [],
    "commands": [],
    "health_signals": {},
    "notifications": [],
    "collector_statuses": [],
}


# --------------------------------------------------------------------------- #
# attempts helpers (direct)
# --------------------------------------------------------------------------- #
def _restart_record(status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "recovery": [
            {
                "workflow": workflow(
                    "w",
                    "i",
                    ["RESTART_WORKLOAD"],
                    executions=[
                        {
                            "operation": "RESTART_WORKLOAD",
                            "status": status,
                            "details": details,
                        }
                    ],
                )
            }
        ],
        "containment_workflows": [
            workflow(
                "c",
                "i",
                ["FREEZE_EVIDENCE", "STOP_WORKLOADS"],
                executions=[{"operation": "STOP_WORKLOADS", "status": "SUCCEEDED"}],
            )
        ],
    }


def test_attempts_withheld_and_budget_helpers() -> None:
    # The shapes the old lenient helpers passed are refused now
    # (general_live_restart_outcomes): a WAITING restart is not withheld, no
    # restart at all never reached the budget gate, a count of 0 is not one.
    withheld = outcomes.withheld_errors
    assert "RESTART_WORKLOAD succeeded despite the exhausted budget" in withheld(
        _restart_record("SUCCEEDED", {}), budget=0
    )
    assert withheld(_restart_record("FAILED", {"reason": "other"}), budget=0), (
        "a restart FAILED for another reason is not the budget's withholding"
    )
    assert withheld(_restart_record("WAITING", {}), budget=0), (
        "a WAITING restart is neither run nor withheld"
    )
    with pytest.raises(VariantInconclusive, match="never reached"):
        withheld({"recovery": []}, budget=0)
    exhausted = {
        "reason": "RESTART_BUDGET_EXHAUSTED",
        "restart_count": 0,
        "restart_budget": 0,
    }
    assert withheld(_restart_record("FAILED", exhausted), budget=0) == []
    budget = outcomes.budget_errors
    assert budget({"restart_budget": None}, consumed=1), (
        "budget({'restart_budget': None}, consumed=1)"
    )
    assert budget({"restart_budget": None}, consumed=0) == []
    assert budget(
        {"restart_budget": {"restart_count": 2, "reservation_ids": ["a"]}}, consumed=1
    ), "budget( {'restart_budget': {'restart_count': 2, 'reservation_ids': ['a']}}"
    assert budget(
        {"restart_budget": {"restart_count": 0, "reservation_ids": []}}, consumed=1
    ), "a count of 0 is not one consumed restart"
    assert (
        budget(
            {"restart_budget": {"restart_count": 1, "reservation_ids": ["a"]}},
            consumed=1,
        )
        == []
    )


def test_attempts_created_job_and_escalated() -> None:
    records = {
        "attempts": {"a1": {"recovery": [{"incident": {"job_id": "other-job"}}]}}
    }
    job = type("J", (), {"name": "mine", "attempt_id": "a1"})()
    assert att._created_new_training_job(records, job) is True
    records2 = {"attempts": {"a1": {"recovery": [{"incident": {"job_id": "mine"}}]}}}
    assert att._created_new_training_job(records2, job) is False
    assert att._escalated({"containment_incident": {"state": "ESCALATED"}}), (
        "att._escalated({'containment_incident': {'state': 'ESCALATED'}})"
    )
    assert att._escalated(
        {
            "containment_incident": {"state": "ACTION_PENDING"},
            "recovery": [{"incident": {"state": "QUARANTINED"}}],
        }
    ), "att._escalated( { 'containment_incident': {'state': 'ACTION_PENDING'},"
    assert not att._escalated(
        {"containment_incident": {"state": "ACTION_PENDING"}, "recovery": []}
    ), "not att._escalated( {'containment_incident': {'state': 'ACTION_PENDING"


def test_attempts_restart_once_errors() -> None:
    record = _restart_record("SUCCEEDED", {"restart_attempt_id": "job-r-1"})
    record["decision"] = {"status": "PLAN_CREATED"}
    ledger = {"restart_budget": {"restart_count": 1, "reservation_ids": ["w/0/RW"]}}
    assert outcomes.restart_once_errors(record, ledger) == [
        "recovery workflow w ended RUNNING"
    ], "the fake workflow is RUNNING: a restart is graded on the settled record"
    missing_stop = {**record, "containment_workflows": []}
    assert any(
        "STOP_WORKLOADS never succeeded" in item
        for item in outcomes.restart_once_errors(missing_stop, ledger)
    ), "outcomes.restart_once_errors(missing_stop, ledger)"
    no_restart = {**record, "recovery": []}
    assert any(
        "RESTART_WORKLOAD succeeded 0 times" in item
        for item in outcomes.restart_once_errors(no_restart, ledger)
    ), "outcomes.restart_once_errors(no_restart, ledger)"


# --------------------------------------------------------------------------- #
# common.restore_incidents and runtime_profile_version
# --------------------------------------------------------------------------- #
class _Spare:
    def __init__(self, regional, hp) -> None:
        self.calls: list[str] = []

    def wait_incident_idle(self, incident_id, **k):
        return {"incident": {"state": "ACTION_PENDING"}}

    def create_restore_workflow(self, *, incident_id, node, profile_version, reason):
        return {"workflow_request_id": f"wf-{incident_id}"}

    def wait_workflow_id(self, wf, **k):
        return {
            "status": "SUCCEEDED",
            "official_steps": [{"operation": "VALIDATE_GPU"}],
            "step_executions": [],
        }


def _common_ctx(
    tmp_path: Path,
    *,
    node_snapshot: dict[str, Any],
    survey: dict[str, Any] | None = None,
):
    from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec

    spec = VariantSpec(
        case_id="GF-GENERAL-LIVE-004",
        variant="capacity_full",
        proof_mode=NATIVE,
        lanes=("eks",),
        risk="M",
        injection="i",
        expected="e",
        cleanup="c",
        max_minutes=10,
        execute=lambda ctx: {},
    )
    regional = FakeRegional(
        node_snapshot=node_snapshot,
        cpu_python=store_script(
            {"preflight": {"agent": {"runtime_profile_version": "eks-v1"}}}
        ),
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(spec, node=NODE),
    )
    ctx.survey = survey if survey is not None else {}
    return ctx


def test_restore_incidents_orders_owner_and_skips_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common, "WarmSpareLiveFixture", _Spare)
    node = {
        "name": NODE,
        "uid": "u",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {"gpu-fault.io/incident-id": "inc-owner"},
        "spare_pool_annotations": {},
    }
    ctx = _common_ctx(
        tmp_path,
        node_snapshot=node,
        survey={"agent": {"runtime_profile_version": "eks-v1"}},
    )
    # After restore the node snapshot must look clean; flip it on the second read.
    reads = {"n": 0}
    base_clean = {**node, "ownership_annotations": {}}

    def snapshot(name):
        reads["n"] += 1
        return dict(node if reads["n"] == 1 else base_clean)

    ctx.regional.node_snapshot = snapshot
    results = common.restore_incidents(
        ctx, [{"incident_id": "inc-owner"}, {"incident_id": "inc-2"}], reason="r"
    )
    assert any(r["incident_id"] == "inc-owner" for r in results), (
        "any(r['incident_id'] == 'inc-owner' for r in results)"
    )


def test_restore_incidents_raises_when_still_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailSpare(_Spare):
        def wait_workflow_id(self, wf, **k):
            return {"status": "FAILED", "official_steps": [], "step_executions": []}

    monkeypatch.setattr(common, "WarmSpareLiveFixture", _FailSpare)
    node = {
        "name": NODE,
        "uid": "u",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {"gpu-fault.io/incident-id": "inc-1"},
        "spare_pool_annotations": {},
    }
    ctx = _common_ctx(
        tmp_path,
        node_snapshot=node,
        survey={"agent": {"runtime_profile_version": "eks-v1"}},
    )
    with pytest.raises(Exception, match="ended FAILED|still isolated"):
        common.restore_incidents(ctx, [{"incident_id": "inc-1"}], reason="r")


def test_runtime_profile_version(tmp_path: Path) -> None:
    node = {
        "name": NODE,
        "uid": "u",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "spare_pool_annotations": {},
    }
    ctx = _common_ctx(tmp_path, node_snapshot=node)
    assert common.runtime_profile_version(ctx) == "eks-v1"


# --------------------------------------------------------------------------- #
# network recovery variants
# --------------------------------------------------------------------------- #
def net_survey() -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"ip": True, "tc": True, "systemd-run": True, "systemctl": True},
        "modules": {
            m: {"loaded": True, "available": True}
            for m in ("veth", "dummy", "sch_netem")
        },
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
            "GPU_FAULT_REQUIRED_INTERFACES": "",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def net_probes(oracle: dict[str, Any] | None = None) -> dict[str, FakeProbe]:
    node = FakeProbe(
        {
            "survey": net_survey(),
            "oracle-start": {},
            "oracle-stop": {},
            "oracle-read": oracle or {"records": [], "next_offset": 0, "total": 0},
        }
    )
    netp = FakeProbe(
        {
            "create-veth": [
                {"a": "gfrsaaaaaaa", "b": "gfrsaaaaaab", "link_a": {"operstate": "up"}},
                {"a": "gfrsaaaaaac", "b": "gfrsaaaaaad", "link_a": {"operstate": "up"}},
            ],
            "set-link": lambda args: {"link": {"operstate": args[-1]}},
            "start-drops": {},
            "stop-drops": {},
            "delete-interface": {},
            "reading": {"interfaces": {"gfrsaaaaaaa": {"operstate": "down"}}},
            "cleanup": {},
        }
    )
    win = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )
    return {f"node@{NODE}": node, f"net@{NODE}": netp, f"window@{NODE}": win}


def test_network_link_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        net,
        "restore_incidents",
        lambda ctx, incidents, reason: [
            {"operations": ["VALIDATE_FABRIC", "RESTORE_SCHEDULING"]}
        ],
    )
    spec = next(
        s
        for s in net.VARIANTS
        if s.case_id == "GF-GENERAL-LIVE-008" and s.variant == "recovery"
    )
    inc = incident("l", "network_link_down", device="gfrsaaaaaaa", state="QUARANTINED")
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": {
                    **EMPTY,
                    "incidents": [inc],
                    "workflows": [
                        workflow(
                            "w",
                            "inc-network_link_down",
                            ["QUARANTINE"],
                            status="SUCCEEDED",
                        )
                    ],
                    # Both ends read up again once the link is back up: the
                    # variant closes the window only after the product has.
                    "batches": [
                        host_batch(
                            f"2026-10-24T00:1{minute}:00+00:00",
                            "network_link_down",
                            0.0,
                            device=device,
                        )
                        for minute in range(2)
                        for device in ("gfrsaaaaaaa", "gfrsaaaaaab")
                    ],
                }
            }
        )
    )
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(
        **{**settings.__dict__, "approvals": ("KERNEL_MODULE_LOAD",)}
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=net_probes(),
        settings=settings,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_network_drops_recovery_passes(tmp_path: Path) -> None:
    oracle = {
        "records": [
            {
                "netdev": {
                    "gfrsaaaaaaa": {"tx_dropped": 100 * i},
                    "gfrsaaaaaab": {"rx_dropped": 100 * i},
                }
            }
            for i in range(1, 8)
        ],
        "next_offset": 7,
        "total": 7,
    }
    drops_doc = {
        **EMPTY,
        "incidents": [
            incident(
                "d", "network_drops_delta", device="gfrsaaaaaaa", state="ACTION_PENDING"
            )
        ],
        "workflows": [
            workflow(
                "w",
                "inc-network_drops_delta",
                ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"],
                status="SUCCEEDED",
            )
        ],
        "batches": [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "network_drops_delta",
                0.0,
                device="gfrsaaaaaaa",
            )
            for i in range(2)
        ],
    }
    # 009 counter_reset is driven in tests/regional/test_general_live_network.py
    # (its bursts, the one-process recreate and the reset-tick oracle).
    spec = next(
        s
        for s in net.VARIANTS
        if s.case_id == "GF-GENERAL-LIVE-009" and s.variant == "recovery"
    )
    regional = FakeRegional(cpu_python=store_script({"activity": drops_doc}))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(
        **{**settings.__dict__, "approvals": ("KERNEL_MODULE_LOAD",)}
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=net_probes(oracle),
        settings=settings,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


# --------------------------------------------------------------------------- #
# storage duplicate + inode detected
# --------------------------------------------------------------------------- #
def storage_survey() -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            n: True
            for n in (
                "losetup",
                "mkfs.ext4",
                "mount",
                "fallocate",
                "dmsetup",
                "blockdev",
            )
        },
        "modules": {
            "dm_mod": {"loaded": True, "available": True},
            "dm_delay": {"loaded": True, "available": True},
        },
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "ext4"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            }
        ],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def storage_probes() -> dict[str, FakeProbe]:
    node = FakeProbe(
        {
            "survey": storage_survey(),
            "oracle-start": {},
            "oracle-stop": {},
            "oracle-read": {"records": [], "next_offset": 0, "total": 0},
        }
    )
    store = FakeProbe(
        {
            "fs-create": {
                "name": "fs",
                "mountpoint": "/mnt/gfrs-live/x/fs",
                "statvfs": {"used_percent": 20.0},
            },
            "fs-fill": {"statvfs": {"used_percent": 99.0}},
            "fs-unfill": {},
            "fs-inodes": {"enospc": True},
            "fs-destroy": {},
        }
    )
    win = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
            "start-unit": {},
        }
    )
    return {f"node@{NODE}": node, f"storage@{NODE}": store, f"window@{NODE}": win}


def test_storage_duplicate_and_inode_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        storage,
        "restore_incidents",
        lambda ctx, incidents, reason: [{"status": "SUCCEEDED"}],
    )
    spec = next(s for s in storage.VARIANTS if s.variant == "duplicate")
    regional = FakeRegional(cpu_python=store_script({"activity": None}))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "site_file": tmp_path / "s.yaml"})
    (tmp_path / "s.yaml").write_text("x\n")
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=storage_probes(),
        settings=settings,
    )
    mount = storage._fs_mount(ctx)
    inc = incident("fs", "filesystem_used_percent", device=mount, state="QUARANTINED")
    regional.cpu_python_responder = store_script(
        {
            "activity": {
                **EMPTY,
                "incidents": [inc],
                "workflows": [
                    workflow(
                        "w",
                        "fs",
                        ["MARK_UNSCHEDULABLE", "QUARANTINE"],
                        status="SUCCEEDED",
                    )
                ],
            }
        }
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
