"""More category edge branches and direct unit coverage of the helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_common as common
from scripts.e2e.regional import general_live_network as net
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
def test_attempts_withheld_and_budget_helpers() -> None:
    succeeded = {
        "recovery": [
            {
                "workflow": workflow(
                    "w",
                    "i",
                    ["RESTART_WORKLOAD"],
                    executions=[
                        {
                            "operation": "RESTART_WORKLOAD",
                            "status": "SUCCEEDED",
                            "details": {},
                        }
                    ],
                )
            }
        ]
    }
    assert att._withheld_errors(succeeded) == [
        "RESTART_WORKLOAD succeeded despite an exhausted budget"
    ]
    bad_reason = {
        "recovery": [
            {
                "workflow": workflow(
                    "w",
                    "i",
                    ["RESTART_WORKLOAD"],
                    executions=[
                        {
                            "operation": "RESTART_WORKLOAD",
                            "status": "FAILED",
                            "details": {"reason": "other"},
                        }
                    ],
                )
            }
        ]
    }
    assert att._withheld_errors(bad_reason), "att._withheld_errors(bad_reason)"
    waiting = {
        "recovery": [
            {
                "workflow": workflow(
                    "w",
                    "i",
                    ["RESTART_WORKLOAD"],
                    executions=[
                        {
                            "operation": "RESTART_WORKLOAD",
                            "status": "WAITING",
                            "details": {},
                        }
                    ],
                )
            }
        ]
    }
    assert att._withheld_errors(waiting) == []
    assert att._withheld_errors({"recovery": []}) == []
    assert att._budget_errors({"restart_budget": None}, consumed=1), (
        "att._budget_errors({'restart_budget': None}, consumed=1)"
    )
    assert att._budget_errors({"restart_budget": None}, consumed=0) == []
    assert att._budget_errors(
        {"restart_budget": {"restart_count": 2, "reservation_ids": ["a"]}}, consumed=1
    ), "att._budget_errors( {'restart_budget': {'restart_count': 2, 'reservati"
    assert att._budget_errors(
        {"restart_budget": {"restart_count": 3, "reservation_ids": ["a", "b", "c"]}},
        consumed=1,
    ), "att._budget_errors( {'restart_budget': {'restart_count': 3, 'reservati"
    assert (
        att._budget_errors(
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


def test_attempts_stop_then_restart_errors() -> None:
    missing_stop = {
        "containment_workflows": [workflow("w", "i", ["FREEZE_EVIDENCE"])],
        "recovery": [],
    }
    assert "no STOP_WORKLOADS" in att._stop_then_restart_errors(missing_stop)[0]
    recovery_no_restart = {
        "containment_workflows": [
            workflow("w", "i", ["FREEZE_EVIDENCE", "STOP_WORKLOADS"])
        ],
        "recovery": [{"workflow": workflow("r", "i", ["VALIDATE_GPU"])}],
    }
    assert any(
        "did not plan RESTART_WORKLOAD" in e
        for e in att._stop_then_restart_errors(recovery_no_restart)
    ), "any( 'did not plan RESTART_WORKLOAD' in e for e in att._stop_then_rest"


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


def test_network_drops_recovery_and_counter_reset_and_wrong(tmp_path: Path) -> None:
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
    # counter_reset (batch 173 grading): the reset tick is a ``_network``
    # collection error naming the recreated interface and its counter, with no
    # network_drops_delta reading for it in that batch; the batches postdate the
    # runner's real-clock recreate.
    reset_error = (
        "_network: HostObservationError: gfrsaaaaaaa: rx_dropped reset (400 -> 0); "
        "delta unknown"
    )

    def soon(seconds: float) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def drops(at: str, value: float) -> dict[str, Any]:
        return host_batch(at, "network_drops_delta", value, device="gfrsaaaaaaa")

    reset_doc = {
        **EMPTY,
        "batches": [
            drops(soon(30), 400.0),
            {
                "observed_at": soon(60),
                "batch_id": "host-reset",
                "edge_filter_reasons": [],
                "collection_errors": [reset_error],
                "samples": [],
            },
            drops(soon(90), 0.0),
        ],
    }
    # The old clamp: the recreated counter shipped as a 0 delta, no reset error.
    clamp_doc = {**EMPTY, "batches": [drops(soon(30), 400.0), drops(soon(60), 0.0)]}

    def run(variant: str, activity: dict[str, Any]) -> dict[str, Any]:
        spec = next(
            s
            for s in net.VARIANTS
            if s.case_id == "GF-GENERAL-LIVE-009" and s.variant == variant
        )
        regional = FakeRegional(cpu_python=store_script({"activity": activity}))
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
        return spec.execute(ctx)

    for variant, activity in (("recovery", drops_doc), ("counter_reset", reset_doc)):
        result = run(variant, activity)
        assert result["verdict"] == "PASS", (variant, result["errors"])
    result = run("counter_reset", clamp_doc)
    assert result["verdict"] == "FAIL", result
    assert any("old clamp" in error for error in result["errors"]), result["errors"]
    with pytest.raises(VariantInconclusive, match="surviving baseline"):
        run("counter_reset", EMPTY)


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
