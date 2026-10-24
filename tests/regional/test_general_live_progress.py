"""GF-GENERAL-LIVE-027 training-progress injector/verdict contract over fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_progress as pr
from scripts.e2e.regional.general_live_registry import build_registry, coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    node_probe,
)

SPECS = {spec.variant: spec for spec in pr.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLEAN_SURVEY = {
    "residue": {},
    "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
    "tools": {},
}


class FakeProgressJob:
    created: list["FakeProgressJob"] = []

    def __init__(
        self, ctx: Any, *, name: str, ranks: tuple[int, ...] = (0,), gpu_count: int = 8
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.ranks = ranks
        self.writes: list[tuple[int, dict[str, Any]]] = []
        self.stopped: list[int] = []
        FakeProgressJob.created.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "ranks": list(self.ranks),
        }

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        self.writes.append((rank, fields))
        return {}

    def stop_reporter(self, rank: int) -> dict[str, Any]:
        self.stopped.append(rank)
        return {}

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeProgressJob.created = []
    monkeypatch.setattr(pr, "ManagedProgressJob", FakeProgressJob)


def _settings(spec: Any, tmp_path: Path):
    base = make_settings(spec, node=NODE)
    (tmp_path / "site.yaml").write_text("site: test\n")
    return type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})


def _ctx(spec: Any, tmp_path: Path, doc: dict[str, Any]):
    regional = FakeRegional(cpu_python=lambda script, *a: doc)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=_settings(spec, tmp_path),
    )
    node_probe(ctx, FakeProbe({"survey": CLEAN_SURVEY}))
    return ctx


def _incident(kind: str, state: str = "ESCALATED", reasons=()):
    return {
        "incident_id": f"inc-training-{kind}",
        "event_id": f"training-{kind}-deadbeef",
        "state": state,
        "attempt_id": "a",
        "workflow_request_id": "wf",
        "reasons": list(reasons),
    }


def _workflow(ops):
    return {
        "wf": {
            "official_steps": [{"operation": op, "parameters": {}} for op in ops],
            "step_executions": [],
        }
    }


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_027_variant() -> None:
    rows = coverage_rows()
    build_registry(pr.VARIANTS, rows=rows, categories=[27])
    assert {s.variant for s in pr.VARIANTS} == set(rows["GF-GENERAL-LIVE-027"])
    assert all(s.implemented for s in pr.VARIANTS), (
        "every 027 variant is native/implemented"
    )


# --------------------------------------------------------------------------- #
# pure verdict helpers
# --------------------------------------------------------------------------- #
def test_chain_errors_requires_ops_and_forbids_restart() -> None:
    incident = _incident("nonfinite-loss")
    ok = _workflow(("FREEZE_EVIDENCE", "STOP_WORKLOADS", "ESCALATE_SUPPORT"))["wf"]
    assert (
        pr._chain_errors(
            incident,
            ok,
            kind="nonfinite-loss",
            required_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS"),
            allow_stop=True,
            states=frozenset({"ESCALATED"}),
        )
        == []
    )
    # A hang chain that planned STOP_WORKLOADS is a FAIL (no STOP for a stall).
    bad = _workflow(("FREEZE_EVIDENCE", "STOP_WORKLOADS", "VALIDATE_FABRIC"))["wf"]
    errors = pr._chain_errors(
        incident,
        bad,
        kind="hang",
        required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
        allow_stop=False,
        states=frozenset(),
    )
    assert any("forbidden" in item for item in errors), errors


def test_chain_errors_missing_incident() -> None:
    assert pr._chain_errors(
        None, {}, kind="hang", required_ops=(), allow_stop=False, states=frozenset()
    ) == ["no training_hang finding reached an incident"]


def test_no_restart_and_checkpoint_helpers() -> None:
    records = {
        "incidents": [_incident("nonfinite-loss")],
        "workflows": _workflow(("FREEZE_EVIDENCE", "RESTART_WORKLOAD")),
        "decision": {"status": "NO_ACTION"},
    }
    assert pr._no_restart_errors(records) == [
        "a training workflow planned RESTART_WORKLOAD"
    ]
    bad = {
        "official_steps": [
            {
                "operation": "STOP_WORKLOADS",
                "parameters": {"checkpoint_disposition": "RESTORE"},
            }
        ]
    }
    assert pr._checkpoint_errors(bad) == [
        "checkpoint disposition is RESTORE, not application-validated"
    ]
    good = {
        "official_steps": [
            {
                "operation": "STOP_WORKLOADS",
                "parameters": {
                    "checkpoint_disposition": "REQUIRES_APPLICATION_VALIDATION"
                },
            }
        ]
    }
    assert pr._checkpoint_errors(good) == []


def test_incident_matches_by_kind_prefix() -> None:
    records = {"incidents": [_incident("reporter-silent"), _incident("hang")]}
    assert pr._incident(records, "hang")["event_id"] == "training-hang-deadbeef"
    assert pr._incident(records, "straggler") is None


# --------------------------------------------------------------------------- #
# execute paths
# --------------------------------------------------------------------------- #
def test_nan_passes_and_stops_without_restart(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [_incident("nonfinite-loss")],
        "workflows": _workflow(
            ("FREEZE_EVIDENCE", "STOP_WORKLOADS", "ESCALATE_SUPPORT")
        ),
        "decision": {"status": "NO_ACTION"},
        "notifications": [],
    }
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, doc))
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeProgressJob.created[-1]
    assert job.writes and job.writes[0][1].get("numerical_error") is True


def test_nan_fails_when_a_restart_is_planned(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [_incident("nonfinite-loss")],
        "workflows": _workflow(
            (
                "FREEZE_EVIDENCE",
                "STOP_WORKLOADS",
                "RESTART_WORKLOAD",
                "ESCALATE_SUPPORT",
            )
        ),
        "decision": None,
        "notifications": [],
    }
    result = SPECS["nan"].execute(_ctx(SPECS["nan"], tmp_path, doc))
    assert result["verdict"] == "FAIL"
    assert any(
        "RESTART_WORKLOAD" in item or "forbidden" in item for item in result["errors"]
    ), result["errors"]


def test_reporter_silent_fails_if_graded_as_hang(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [_incident("reporter-silent"), _incident("hang")],
        "workflows": _workflow(("FREEZE_EVIDENCE", "VALIDATE_FABRIC")),
        "decision": None,
        "notifications": [],
    }
    result = SPECS["reporter_silent"].execute(
        _ctx(SPECS["reporter_silent"], tmp_path, doc)
    )
    assert result["verdict"] == "FAIL"
    assert any("hang" in item for item in result["errors"]), result["errors"]


def test_old_attempt_passes_when_nothing_is_stopped(tmp_path: Path) -> None:
    doc = {
        "progress_states": [],
        "incidents": [],
        "workflows": {},
        "decision": None,
        "notifications": [],
    }
    result = SPECS["old_attempt"].execute(_ctx(SPECS["old_attempt"], tmp_path, doc))
    assert result["verdict"] == "PASS", result["errors"]
