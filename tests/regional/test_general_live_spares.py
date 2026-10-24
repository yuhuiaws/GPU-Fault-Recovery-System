"""GF-GENERAL-LIVE-030 warm-spare grading/verdict contract over fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_spares as sp
from scripts.e2e.regional.general_live_registry import (
    APPROVALS,
    build_registry,
    coverage_rows,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

SPECS = {spec.variant: spec for spec in sp.VARIANTS}
NODE = "ip-10-0-43-9.us-west-2.compute.internal"


def _node(*, ready="True", health="Schedulable", annotations=None, unschedulable=True):
    return {
        "name": NODE,
        "ready": ready,
        "unschedulable": unschedulable,
        "labels": {sp.HYPERPOD_HEALTH_LABEL: health},
        "annotations": annotations or {},
    }


def _store(**overrides):
    base = {
        "agent_reasons": [],
        "fault_reasons": [],
        "observation_reasons": [],
        "spare_health_incidents": [],
        "spare_health_workflows": {},
    }
    base.update(overrides)
    return base


class FakeSpareSession:
    instances: list["FakeSpareSession"] = []

    def __init__(
        self, ctx: Any, *, node: dict[str, Any], store: dict[str, Any]
    ) -> None:
        self.ctx = ctx
        self.incident_id = f"inc-{ctx.run_id}"
        self._node = node
        self._store = store
        self.declared = False
        self.kubelet_stopped = False
        self.metrics_paused = False
        self.mutations: list[dict[str, Any]] = []
        FakeSpareSession.instances.append(self)

    def declare(self) -> dict[str, Any]:
        self.declared = True
        return {"refusals": []}

    def node(self) -> dict[str, Any]:
        return self._node

    def grade(self) -> dict[str, Any]:
        return self._store

    def stop_kubelet(self) -> dict[str, Any]:
        self.kubelet_stopped = True
        return {}

    def pause_metrics(self) -> dict[str, Any]:
        self.metrics_paused = True
        return {}

    def mutate(self, annotations: dict[str, str | None]) -> Any:
        self.mutations.append(annotations)
        self._node.setdefault("annotations", {}).update(annotations)
        return None


def _patch(monkeypatch, node, store):
    FakeSpareSession.instances = []
    monkeypatch.setattr(
        sp, "SpareSession", lambda ctx: FakeSpareSession(ctx, node=node, store=store)
    )


def _ctx(spec, tmp_path):
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "approvals": spec.approvals})
    return make_context(
        spec, tmp_path=tmp_path, regional=FakeRegional(), probes={}, settings=settings
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_030_variant_and_tokens() -> None:
    rows = coverage_rows()
    build_registry(sp.VARIANTS, rows=rows, categories=[30])
    assert {s.variant for s in sp.VARIANTS} == set(rows["GF-GENERAL-LIVE-030"])
    implemented = {s.variant for s in sp.VARIANTS if s.implemented}
    assert implemented == {
        "not_ready",
        "stale_metrics",
        "configuration_error",
        "reserved_spare",
    }
    assert "DECLARE_WARM_SPARE" in APPROVALS and "SPARE_KUBELET_STOP" in APPROVALS
    for spec in sp.VARIANTS:
        assert all(lane == "hp" for lane in spec.lanes), spec.variant
    deferred = {
        s.variant: s.deferred.split(":", 1)[0] for s in sp.VARIANTS if s.deferred
    }
    assert deferred == {
        "confirmed_hardware": "CAPABILITY_ABSENT",
        "no_spare": "NOT_AUTHORIZED",
        "repair_failure": "CAPABILITY_ABSENT",
        "bounded_recheck": "CAPABILITY_ABSENT",
    }


# --------------------------------------------------------------------------- #
# grading oracle
# --------------------------------------------------------------------------- #
def test_grading_reasons_mirror_health_reasons() -> None:
    reasons = sp.grading_reasons(_node(ready="False"), _store(), our_incident="inc-x")
    assert "Kubernetes node is not Ready" in reasons
    assert sp.reason_class("Kubernetes node is not Ready") == sp.OBSERVATION
    assert sp.reason_class("spare pool state is UNAVAILABLE") == sp.CONFIGURATION
    assert sp.reason_class("reserved by incident inc-y") == sp.CONFIGURATION
    assert sp.reason_class(sp.HARDWARE_PREFIX + "inc-z") == sp.HARDWARE


def test_reserved_spare_rejected_for_other_incident_only() -> None:
    annotations = {
        sp.RESERVATION_ANNOTATION: "inc-holder",
        sp.POOL_STATE_ANNOTATION: "ALLOCATED",
    }
    node = _node(annotations=annotations, unschedulable=False)
    assert "reserved by incident inc-holder" in sp.grading_reasons(
        node, _store(), our_incident="inc-other"
    )
    # Its own holder is not rejected by the reservation (ALLOCATED is allowed).
    assert sp.grading_reasons(node, _store(), our_incident="inc-holder") == []


def test_no_recovery_errors_ignore_injected_pool_state_but_flag_reboot() -> None:
    node = _node(annotations={sp.POOL_STATE_ANNOTATION: "UNAVAILABLE"})
    assert sp.no_recovery_errors(node, _store()) == []
    flagged = _node(
        annotations={sp.INCIDENT_ANNOTATION: "inc-reboot", sp.FAILURES_ANNOTATION: "2"}
    )
    assert len(sp.no_recovery_errors(flagged, _store())) == 2
    store = _store(
        spare_health_incidents=[{"incident_id": "i", "workflow_request_id": "w"}],
        spare_health_workflows={
            "w": {"official_steps": [{"operation": "RESTART_NODE"}]}
        },
    )
    assert any(
        "RESTART_NODE" in item for item in sp.no_recovery_errors(_node(), store)
    ), "a reboot workflow must be flagged as recovery"


def test_block_only_verdict_inconclusive_when_oracle_blind() -> None:
    result = sp.block_only_verdict(
        _node(ready="False"),
        _store(),
        expected_class=sp.OBSERVATION,
        expected_fragment="Kubernetes node is not Ready",
        our_incident="inc-x",
        oracle_ok=False,
        oracle_detail="never NotReady",
    )
    assert result["verdict"] == "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# execute paths
# --------------------------------------------------------------------------- #
def test_not_ready_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _node(ready="False"), _store())
    result = SPECS["not_ready"].execute(_ctx(SPECS["not_ready"], tmp_path))
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeSpareSession.instances[-1].kubelet_stopped, (
        "kubelet must have been stopped"
    )


def test_configuration_error_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _node(), _store())
    result = SPECS["configuration_error"].execute(
        _ctx(SPECS["configuration_error"], tmp_path)
    )
    assert result["verdict"] == "PASS", result["errors"]


def test_reserved_spare_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _node(unschedulable=True), _store())
    result = SPECS["reserved_spare"].execute(_ctx(SPECS["reserved_spare"], tmp_path))
    assert result["verdict"] == "PASS", result["errors"]


def test_stale_metrics_fails_when_product_rebooted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(
        observation_reasons=[
            "collector observation UNKNOWN: GPU_METRICS/STALE_OBSERVATION"
        ],
        spare_health_incidents=[{"incident_id": "i", "workflow_request_id": "w"}],
        spare_health_workflows={
            "w": {"official_steps": [{"operation": "RESTART_NODE"}]}
        },
    )
    _patch(monkeypatch, _node(), store)
    result = SPECS["stale_metrics"].execute(_ctx(SPECS["stale_metrics"], tmp_path))
    assert result["verdict"] == "FAIL"
    assert any("RESTART_NODE" in item for item in result["errors"]), result["errors"]
