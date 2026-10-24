"""GF-GENERAL-LIVE-011 EFA/NCCL-hung injector/verdict contract over fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_nccl as nc
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import build_registry, coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

SPECS = {spec.variant: spec for spec in nc.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
CLEAN_SURVEY = {
    "residue": {},
    "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
    "tools": {},
}


class FakeNcclJob:
    instances: list["FakeNcclJob"] = []

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        flight_recorder: bool = False,
        keep_active: bool = True,
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.flight_recorder = flight_recorder
        self.keep_active = keep_active
        self.stalled = False
        self.released = False
        FakeNcclJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "nodes": [NODE, PEER],
        }

    def stall(self) -> dict[str, Any]:
        self.stalled = True
        return {}

    def release(self) -> dict[str, Any]:
        self.released = True
        return {}

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeNcclJob.instances = []
    monkeypatch.setattr(nc, "ManagedNcclJob", FakeNcclJob)


def _hung_records(
    *, state="ESCALATED", ops=nc.DIAGNOSTIC_DAG, nodes=(NODE, PEER), count=1
):
    incidents = [
        {
            "incident_id": f"inc-{i}",
            "event_id": f"batch{i}-efa-traffic-hung_suspected",
            "event_type": "GPU_FAULT_GROUP",
            "state": state,
            "attempt_id": "a",
            "job_id": "j",
            "node_ids": list(nodes),
            "workflow_request_id": "wf",
        }
        for i in range(count)
    ]
    return {
        "incidents": incidents,
        "workflows": {
            "wf": {
                "official_steps": [{"operation": op, "parameters": {}} for op in ops],
                "step_executions": [],
            }
        },
    }


def _settings(spec: Any, tmp_path: Path):
    base = make_settings(spec, node=NODE, peer=PEER)
    (tmp_path / "site.yaml").write_text("site: test\n")
    return type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})


def _ctx(
    spec: Any,
    tmp_path: Path,
    doc: dict[str, Any],
    *,
    node_moves=(0, 0),
    peer_moves=(0, 0),
):
    regional = FakeRegional(cpu_python=lambda script, *a: doc)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=_settings(spec, tmp_path),
    )
    node_efa = [{"pair_bytes_total": v} for v in node_moves]
    peer_efa = [{"pair_bytes_total": v} for v in peer_moves]
    ctx.probes[f"node@{NODE}"] = FakeProbe(
        {"survey": CLEAN_SURVEY, "efa-traffic": node_efa}
    )  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = FakeProbe({"efa-traffic": peer_efa})  # type: ignore[assignment]
    return ctx


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_011_variant() -> None:
    rows = coverage_rows()
    build_registry(nc.VARIANTS, rows=rows, categories=[11])
    assert {s.variant for s in nc.VARIANTS} == set(rows["GF-GENERAL-LIVE-011"])
    implemented = {s.variant for s in nc.VARIANTS if s.implemented}
    assert implemented == {
        "active_to_zero",
        "always_low",
        "write_progress",
        "phase_progress",
        "fr_enabled",
        "fr_disabled",
        "simultaneous_nodes",
    }
    assert all(lane == "hp" for spec in nc.VARIANTS for lane in spec.lanes), (
        "011 runs only on the HyperPod lane"
    )
    # Every deferred variant keeps an explicit NOT_RUN reason in the denominator.
    for spec in nc.VARIANTS:
        if not spec.implemented:
            assert spec.deferred.split(":", 1)[0] in {
                "FIXTURE_NOT_READY",
                "SCALE",
                "NOT_AUTHORIZED",
            }, spec.variant


# --------------------------------------------------------------------------- #
# pure verdict helpers
# --------------------------------------------------------------------------- #
def test_dag_errors_detects_order_and_forbidden() -> None:
    assert nc._dag_errors(_hung_records()) == []
    assert nc._dag_errors({"incidents": [], "workflows": {}}) == [
        "no EFA hung_suspected finding reached an incident"
    ]
    wrong = _hung_records(ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"))
    assert any("diagnostic DAG" in item for item in nc._dag_errors(wrong)), wrong
    forbidden = _hung_records(ops=(*nc.DIAGNOSTIC_DAG, "STOP_WORKLOADS"))
    assert any("forbidden" in item for item in nc._dag_errors(forbidden)), forbidden


def test_escalated_and_grouped_and_coverage() -> None:
    assert nc._escalated_errors(_hung_records(state="ESCALATED")) == []
    assert nc._escalated_errors(_hung_records(state="ACTION_PENDING")), (
        "a non-ESCALATED hung incident is a failure"
    )
    assert nc._grouped_errors(_hung_records(count=2), expect_single=True), (
        "two hung incidents for one stall wave must fail the merge check"
    )
    assert nc._grouped_errors(_hung_records(count=1), expect_single=True) == []

    class Ctx:
        class settings:
            node = NODE
            peer_node = PEER

    assert nc._covers_both_nodes(_hung_records(nodes=(NODE, PEER)), Ctx) is True
    assert nc._covers_both_nodes(_hung_records(nodes=(NODE,)), Ctx) is False


def test_fr_errors_read_triage_dump_status() -> None:
    def triage(status):
        return {
            "incidents": [
                {
                    "event_id": "b-efa-traffic-hung_suspected",
                    "state": "ESCALATED",
                    "workflow_request_id": "wf",
                }
            ],
            "workflows": {
                "wf": {
                    "official_steps": [],
                    "step_executions": [
                        {
                            "operation": "COLLECT_HUNG_TRIAGE",
                            "details": {
                                "node_results": {
                                    NODE: {
                                        "data": {
                                            "ranks": [
                                                {"flight_recorder": {"status": status}}
                                            ]
                                        }
                                    }
                                }
                            },
                        }
                    ],
                }
            },
        }

    assert nc._fr_errors(triage("not_configured"), expect_configured=False) == []
    assert nc._fr_errors(triage("dumped"), expect_configured=False), (
        "a dump with FR off must be flagged"
    )
    assert nc._fr_errors(triage("not_configured"), expect_configured=True), (
        "FR configured but not_configured must be flagged"
    )
    assert nc._fr_errors(triage("dumped"), expect_configured=True) == []


# --------------------------------------------------------------------------- #
# execute paths
# --------------------------------------------------------------------------- #
def test_active_to_zero_passes(tmp_path: Path) -> None:
    result = SPECS["active_to_zero"].execute(
        _ctx(SPECS["active_to_zero"], tmp_path, _hung_records())
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeNcclJob.instances[-1].stalled, "the job must have been stalled"


def test_active_to_zero_inconclusive_if_traffic_keeps_moving(tmp_path: Path) -> None:
    ctx = _ctx(
        SPECS["active_to_zero"],
        tmp_path,
        _hung_records(),
        node_moves=(0, 5_000_000_000),
        peer_moves=(0, 0),
    )
    with pytest.raises(VariantInconclusive):
        SPECS["active_to_zero"].execute(ctx)


def test_simultaneous_nodes_fails_when_not_merged(tmp_path: Path) -> None:
    ctx = _ctx(SPECS["simultaneous_nodes"], tmp_path, _hung_records(count=2))
    result = SPECS["simultaneous_nodes"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "merged" in item or "one stall wave" in item for item in result["errors"]
    ), result["errors"]


def test_always_low_passes_when_no_hung(tmp_path: Path) -> None:
    ctx = _ctx(SPECS["always_low"], tmp_path, {"incidents": [], "workflows": {}})
    result = SPECS["always_low"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_fr_disabled_passes_with_not_configured(tmp_path: Path) -> None:
    records = {
        "incidents": [
            {
                "event_id": "b-efa-traffic-hung_suspected",
                "state": "ESCALATED",
                "workflow_request_id": "wf",
                "node_ids": [NODE, PEER],
            }
        ],
        "workflows": {
            "wf": {
                "official_steps": [
                    {"operation": op, "parameters": {}} for op in nc.DIAGNOSTIC_DAG
                ],
                "step_executions": [
                    {
                        "operation": "COLLECT_HUNG_TRIAGE",
                        "details": {
                            "node_results": {
                                NODE: {
                                    "data": {
                                        "ranks": [
                                            {
                                                "flight_recorder": {
                                                    "status": "not_configured"
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        },
                    }
                ],
            }
        },
    }
    result = SPECS["fr_disabled"].execute(_ctx(SPECS["fr_disabled"], tmp_path, records))
    assert result["verdict"] == "PASS", result["errors"]
