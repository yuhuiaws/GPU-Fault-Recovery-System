from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import declare_warm_spare as helper
from scripts.e2e.regional.warm_spare_fixture import (
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
)


def _node(name: str, **overrides: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "name": name,
        "uid": f"uid-{name}",
        "ready": "True",
        "unschedulable": False,
        "gpu_allocatable": 8,
        "taints": [],
        "labels": {
            SPARE_LABEL: None,
            HYPERPOD_HEALTH_LABEL: "Schedulable",
            INSTANCE_GROUP_LABEL: "group-a",
            "node.kubernetes.io/instance-type": "ml.p5en.48xlarge",
        },
        "annotations": {
            SPARE_RESERVATION_ANNOTATION: None,
            SPARE_POOL_STATE_ANNOTATION: None,
        },
    }
    for key, value in overrides.items():
        if key in {"labels", "annotations"}:
            snapshot[key] = {**snapshot[key], **value}
        else:
            snapshot[key] = value
    return snapshot


def _settings(tmp_path: Path, *, fault_node: str = "") -> helper.Settings:
    return helper.Settings(
        node="node-spare",
        fault_node=fault_node,
        hyperpod_cluster="cluster-a",
        baseline=tmp_path / "spare-baseline.json",
    )


def _refusals(settings: helper.Settings, **overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "spare": _node("node-spare"),
        "fault": None,
        "declared": [],
        "workloads": [],
        "state": {"agents": [{"node_id": "node-spare", "lifecycle_state": "ACTIVE"}]},
    }
    arguments.update(overrides)
    return helper.declare_refusals(settings, **arguments)


def test_a_healthy_monitored_node_can_be_declared(tmp_path: Path) -> None:
    assert _refusals(_settings(tmp_path)) == []


def test_declaration_is_refused_without_exactly_one_active_agent(
    tmp_path: Path,
) -> None:
    # "纳入监控" is a hard precondition of the warm-spare path: an unmonitored
    # node has no Agent to fence and no health signal to trust.
    settings = _settings(tmp_path)

    assert "node does not have exactly one ACTIVE Agent" in _refusals(
        settings, state={"agents": []}
    )
    assert "node does not have exactly one ACTIVE Agent" in _refusals(
        settings,
        state={"agents": [{"node_id": "node-spare", "lifecycle_state": "REVOKED"}]},
    )
    # Two Agents claiming one node is ambiguous, not "the first one wins".
    assert "node does not have exactly one ACTIVE Agent" in _refusals(
        settings,
        state={
            "agents": [
                {"node_id": "node-spare", "lifecycle_state": "ACTIVE"},
                {"node_id": "node-spare", "lifecycle_state": "ACTIVE"},
            ]
        },
    )


def test_declaration_is_refused_while_a_gpu_workload_is_running(tmp_path: Path) -> None:
    # Cordoning does not evict. A spare declared under a live job is busy, and
    # the case would allocate a node that is already working.
    refusals = _refusals(
        _settings(tmp_path),
        workloads=[{"node": "node-spare", "namespace": "team", "gpus": 8}],
    )

    assert "node still has an active GPU workload" in refusals


def test_declaration_is_refused_on_a_topology_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path, fault_node="node-fault")
    fault = _node("node-fault", labels={"node.kubernetes.io/instance-type": "ml.p4d"})

    refusals = _refusals(settings, fault=fault)

    assert any("topology does not match" in item for item in refusals), refusals


def test_declaration_is_refused_on_pre_existing_quarantine_ownership(
    tmp_path: Path,
) -> None:
    refusals = _refusals(
        _settings(tmp_path),
        spare=_node("node-spare", annotations={"gpu-fault.io/incident-id": "INC-1"}),
    )

    assert "node carries quarantine ownership from an earlier incident" in refusals, (
        refusals
    )


def test_declaration_is_refused_when_the_pool_state_is_not_available(
    tmp_path: Path,
) -> None:
    # Absent is fine -- the control plane owns this annotation and DESTR-003
    # accepts it unset -- but ALLOCATED means the pool already spent this node.
    assert (
        _refusals(
            _settings(tmp_path),
            spare=_node(
                "node-spare", annotations={SPARE_POOL_STATE_ANNOTATION: "ALLOCATED"}
            ),
        )
    ) == ["spare pool state is not AVAILABLE"]


def test_release_is_refused_for_a_reserved_or_quarantined_node() -> None:
    reserved = _node("node-spare", annotations={SPARE_RESERVATION_ANNOTATION: "INC-1"})
    quarantined = _node(
        "node-spare",
        taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
    )

    assert helper.release_refusals(reserved) == [
        "node is reserved by an incident; release it through the case"
    ]
    assert helper.release_refusals(quarantined) == [
        "node is quarantined; use restore_validated_quarantine.py instead"
    ]
    assert helper.release_refusals(_node("node-spare")) == []


def test_release_is_refused_while_the_pool_still_records_an_allocation() -> None:
    # The pool state outlives the reservation annotation on some paths, and an
    # ALLOCATED node is carrying the failed-over job: dropping its spare label
    # takes it out of the pool while the control plane still holds it.
    allocated = _node(
        "node-spare", annotations={SPARE_POOL_STATE_ANNOTATION: "ALLOCATED"}
    )

    assert helper.release_refusals(allocated) == [
        "node is ALLOCATED by the spare pool; release it through the case"
    ]
    for state in (None, "AVAILABLE"):
        assert (
            helper.release_refusals(
                _node("node-spare", annotations={SPARE_POOL_STATE_ANNOTATION: state})
            )
            == []
        )


def test_release_restores_the_recorded_baseline_not_the_current_state(
    tmp_path: Path,
) -> None:
    # A node that was already cordoned before the declaration must stay cordoned
    # after the release. Restoring "uncordon" unconditionally would hand an
    # operator-cordoned node back to the scheduler.
    settings = _settings(tmp_path)
    baseline = _node("node-spare", unschedulable=True)
    settings.baseline.write_text(
        json.dumps({"node": "node-spare", "baseline": baseline}), encoding="utf-8"
    )
    applied: list[helper.NodePatch] = []

    class FakeWarm:
        def node_snapshot(self, node: str) -> dict[str, Any]:
            del node
            return _node("node-spare", labels={SPARE_LABEL: "true"})

        def spare_nodes(self) -> list[str]:
            return []

        regional = None

    class FakeMutation:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.baseline: dict[str, Any] = {}

        def apply(self, patch: helper.NodePatch) -> None:
            applied.append(patch)

        def restore(self) -> dict[str, Any]:
            self.apply(
                helper.NodePatch(
                    labels={SPARE_LABEL: self.baseline["labels"].get(SPARE_LABEL)},
                    annotations={},
                    unschedulable=bool(self.baseline["unschedulable"]),
                )
            )
            return _node("node-spare", unschedulable=True)

    original = helper.NodeMutationFixture
    helper.NodeMutationFixture = FakeMutation  # type: ignore[assignment,misc]
    try:
        record = helper.release(settings, FakeWarm(), {"node": baseline})
    finally:
        helper.NodeMutationFixture = original  # type: ignore[misc]

    assert applied == [
        helper.NodePatch(labels={SPARE_LABEL: None}, annotations={}, unschedulable=True)
    ], applied
    assert record["released_at"]
    assert json.loads(settings.baseline.read_text(encoding="utf-8"))["released_at"]


def test_release_refuses_an_already_released_or_mismatched_record(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.baseline.write_text(
        json.dumps(
            {"node": "node-spare", "baseline": _node("node-spare"), "released_at": "x"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(helper.RegionalFixtureError, match="already released"):
        helper.release(settings, object(), {})

    settings.baseline.write_text(
        json.dumps({"node": "node-other", "baseline": _node("node-other")}),
        encoding="utf-8",
    )
    with pytest.raises(helper.RegionalFixtureError, match="records node node-other"):
        helper.release(settings, object(), {})


def test_declare_refuses_to_overwrite_an_unreleased_record(tmp_path: Path) -> None:
    # Overwriting would destroy the only record of what the node looked like
    # before it became a spare.
    settings = _settings(tmp_path)
    settings.baseline.write_text(
        json.dumps({"node": "node-spare", "baseline": _node("node-spare")}),
        encoding="utf-8",
    )

    with pytest.raises(helper.RegionalFixtureError, match="unreleased declaration"):
        helper.declare(settings, object(), {})


def test_helper_is_read_only_by_default() -> None:
    arguments = helper.parser().parse_args(["--spare-node", "node-spare"])

    assert arguments.declare is False
    assert arguments.release is False
    assert arguments.confirm == ""
    assert helper.DECLARE_CONFIRMATION == "DECLARE_WARM_SPARE_CORDON"
    assert helper.RELEASE_CONFIRMATION == "RELEASE_WARM_SPARE_UNCORDON"


def test_the_report_is_serialisable_after_a_declaration() -> None:
    # The baseline record embeds the survey and the survey is the report, so
    # attaching the record back unfiltered makes the report contain itself. That
    # only fails at the closing `json.dumps` -- after the node was cordoned --
    # so the operator would see a traceback for a declaration that succeeded.
    report: dict[str, Any] = {"observed_at": "2026-09-05T00:00:00Z"}
    record = {"node": "node-spare", "pre_declaration_survey": report}
    report["declaration"] = helper.without_survey(record)

    assert json.loads(json.dumps(report))["declaration"] == {"node": "node-spare"}


def test_the_spare_is_never_named_by_the_shared_node_flag() -> None:
    # A site profile's `node` is the collector cases' target, a different machine
    # from the spare. If this helper accepted --node it would inherit that value
    # from the profile and cordon the wrong node.
    with pytest.raises(SystemExit):
        helper.parser().parse_args(["--node", "node-spare"])


def test_helper_carries_no_site_topology() -> None:
    source = Path(helper.__file__).read_text(encoding="utf-8")

    assert "/secure/gpu-fault-bootstrap" not in source
    assert "514385905925" not in source
    assert "hyperpod-i-" not in source
