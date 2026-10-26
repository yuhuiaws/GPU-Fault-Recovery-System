"""GF-GENERAL-LIVE-030 warm-spare grading/verdict contract over fakes."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_spares as sp
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import (
    APPROVALS,
    build_registry,
    coverage_rows,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

SPECS = {spec.variant: spec for spec in sp.VARIANTS}
#: The spare candidate (the lane's --node-b) and the lane's own target node.
NODE = "ip-10-0-43-9.us-west-2.compute.internal"
TARGET = "ip-10-0-43-5.us-west-2.compute.internal"


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
        self.verified = False
        self.finished = False
        self.kubelet_stopped = False
        self.awaited_not_ready = False
        self.metrics_paused = False
        self.mutations: list[dict[str, Any]] = []
        #: What ``finish`` reports: the candidate as found and as handed back.
        self.before = {
            **node,
            "ready": "True",
            "labels": {},
            "annotations": {},
            "unschedulable": False,
        }
        self.after = dict(self.before)
        self.residue = {"declared_spares": [], "unexpected": [], "clean": True}
        self.cleanup_results: list[dict[str, Any]] = [
            {"name": "warm-spare", "ok": True, "result": {"released": True}}
        ]
        FakeSpareSession.instances.append(self)

    def verify_idle(self) -> None:
        assert not self.declared, "the idleness proof precedes the declaration"
        self.verified = True

    def declare(self) -> dict[str, Any]:
        assert self.verified, "declare without the idleness proof"
        self.declared = True
        return {"refusals": []}

    def node_snapshot(self) -> dict[str, Any]:
        return self._node

    def grade(self) -> dict[str, Any]:
        return self._store

    def stop_kubelet(self) -> dict[str, Any]:
        self.kubelet_stopped = True
        return {}

    def await_not_ready(self) -> dict[str, Any]:
        assert self.kubelet_stopped, "waiting for NotReady before the stop"
        self.awaited_not_ready = True
        return self._node

    def finish(self) -> dict[str, Any]:
        assert self.declared, "finish without a declaration"
        self.finished = True
        return {
            "spare_before": self.before,
            "spare_after": self.after,
            "residue": self.residue,
            "cleanup": self.cleanup_results,
            "release": self.cleanup_results[-1]["result"],
            "open_incidents_after": [],
            "active_workflows_after": [],
        }

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


def _ctx(spec, tmp_path, **overrides):
    settings = make_settings(spec, node=TARGET, peer=NODE)
    settings = type(settings)(
        **{**settings.__dict__, "approvals": spec.approvals, **overrides}
    )
    return make_context(
        spec, tmp_path=tmp_path, regional=FakeRegional(), probes={}, settings=settings
    )


def _store_responder(**preflight: Any):
    base = {
        "agent": {"lifecycle_state": "ACTIVE"},
        "open_incidents": [],
        "active_workflows": [],
        "collector_statuses": [
            {
                "collector": channel,
                "last_success_at": datetime.now(timezone.utc).isoformat(),
                "last_error_at": None,
            }
            for channel in ("HOST_TELEMETRY", "GPU_INVENTORY", "GPU_METRICS")
        ],
    }
    base.update(preflight)

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        assert "preflight" in script.lower() or "store" in script.lower()
        return dict(base)

    return respond


def _idle_regional(**overrides: Any) -> FakeRegional:
    base: dict[str, Any] = dict(
        node_snapshot={
            "name": NODE,
            "uid": "spare-uid",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "spare_pool_annotations": {},
            "gpu_allocatable": "8",
        },
        node_metadata={
            "name": NODE,
            "labels": {"eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group"},
        },
        cpu_python=_store_responder(),
    )
    base.update(overrides)
    return FakeRegional(**base)


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
        assert spec.lanes == ("eks", "hp"), (
            f"{spec.variant}: 030 declares the lane's second node on both clusters"
        )
        if spec.implemented:
            assert spec.flags == "X", (
                f"{spec.variant}: the candidate is another lane's node"
            )
            assert spec.preflight is sp.spare_preflight, spec.variant
            assert "release the declared spare" in spec.cleanup, spec.variant
    deferred = {
        s.variant: s.deferred.split(":", 1)[0] for s in sp.VARIANTS if s.deferred
    }
    assert deferred == {
        "confirmed_hardware": "CAPABILITY_ABSENT",
        "no_spare": "COVERED_BY_REGIONAL",
        "repair_failure": "CAPABILITY_ABSENT",
        "bounded_recheck": "CAPABILITY_ABSENT",
    }
    no_spare = next(s for s in sp.VARIANTS if s.variant == "no_spare")
    assert "GF-REGIONAL-DESTR-008" in no_spare.deferred and no_spare.lanes == (
        "eks",
        "hp",
    ), "the cross-reference names the regional case; both boards carry it"


# --------------------------------------------------------------------------- #
# the spare candidate: resolution, idleness proof, hand-back proof
# --------------------------------------------------------------------------- #
def test_spare_candidate_is_the_flag_else_node_b_and_scope_falls_back_to_cluster() -> (
    None
):
    spec = SPECS["configuration_error"]
    settings = make_settings(spec, node=TARGET, peer=NODE)
    assert sp.spare_candidate(settings) == NODE, "--node-b by default"
    explicit = dataclasses.replace(settings, spare_candidate="hp-spare")
    assert sp.spare_candidate(explicit) == "hp-spare", "--spare-candidate wins"
    assert sp.spare_candidate(dataclasses.replace(settings, peer_node="")) == ""
    assert sp.spare_scope(settings) == "hp-cluster", "--hyperpod-cluster when given"
    assert (
        sp.spare_scope(dataclasses.replace(settings, hyperpod_cluster=""))
        == settings.regional.cluster_id
    ), "the EKS cluster name is the lane's cluster id"


def test_candidate_errors_name_every_reason_a_node_is_not_idle() -> None:
    spec = SPECS["not_ready"]
    settings = make_settings(spec, node=TARGET, peer=NODE)
    assert sp.candidate_errors(settings, _idle_regional()) == []
    busy = _idle_regional(
        node_snapshot={
            "name": NODE,
            "uid": "u",
            "ready": "False",
            "unschedulable": True,
            "taints": [{"key": "gpu-fault.io/quarantined"}],
            "ownership_annotations": {"gpu-fault.io/incident-id": "i"},
            "spare_pool_annotations": {
                sp.RESERVATION_ANNOTATION: "inc-x",
                sp.POOL_STATE_ANNOTATION: "ALLOCATED",
            },
            "gpu_allocatable": "8",
        },
        node_metadata={
            "name": NODE,
            "labels": {"eks.amazonaws.com/nodegroup": "g", sp.SPARE_LABEL: "true"},
        },
        business=[{"namespace": "team", "name": "trainer"}],
        gpu_pods=[{"node": NODE, "namespace": "ns", "name": "gpu-pod"}],
        cpu_python=_store_responder(
            agent={"lifecycle_state": "REVOKED"},
            open_incidents=[{"incident_id": "i"}],
            active_workflows=[{"request_id": "w"}],
            collector_statuses=[],
        ),
    )
    joined = " ".join(sp.candidate_errors(settings, busy))
    for needle in (
        "not Ready",
        "cordoned",
        "gpu-fault taints",
        "ownership annotations",
        "spare reservation",
        "pool state is ALLOCATED",
        f"{sp.SPARE_LABEL}=true",
        "business workload",
        "GPU-requesting Pods",
        "Node Agent is not ACTIVE",
        "open incidents",
        "active workflows",
        "no successful observation",
    ):
        assert needle in joined, needle
    assert sp.candidate_errors(
        dataclasses.replace(settings, peer_node=""), _idle_regional()
    ) == [
        "030 needs a spare candidate: --spare-candidate <node> "
        "(default: --node-b, the lane's second node)"
    ]
    assert sp.candidate_errors(
        dataclasses.replace(settings, peer_node=TARGET), _idle_regional()
    ) == [f"the spare candidate {TARGET} is the target node itself"]


def test_candidate_errors_refuse_a_foreign_provider_and_eks_node_auto_repair() -> None:
    spec = SPECS["not_ready"]
    settings = make_settings(spec, node=TARGET, peer=NODE)
    repair_on = {"node_repair_config": {"gpu-group": True, "cpu-group": False}}
    errors = sp.candidate_errors(settings, _idle_regional(), recovery=lambda: repair_on)
    assert errors == [
        "EKS node auto repair is enabled on node group(s) ['gpu-group']; a "
        "NotReady spare would race the product with a provider repair"
    ]
    repair_off = {"node_repair_config": {"gpu-group": False, "other": None}}
    assert (
        sp.candidate_errors(settings, _idle_regional(), recovery=lambda: repair_off)
        == []
    )
    assert sp.candidate_errors(settings, _idle_regional(), recovery=None) == [], (
        "without a recovery reader the provider fact is simply not judged"
    )

    class TwoProviders(FakeRegional):
        def node_metadata(self, node: str) -> dict[str, Any]:
            if node == NODE:
                return {"name": NODE, "labels": {"eks.amazonaws.com/nodegroup": "g"}}
            return {
                "name": node,
                "labels": {"sagemaker.amazonaws.com/compute-type": "hyperpod"},
            }

    mixed = TwoProviders(cpu_python=_store_responder())
    errors = sp.candidate_errors(settings, mixed, recovery=lambda: repair_off)
    assert errors == [f"{NODE} is a eks node; the target node {TARGET} is hp"]


def test_pods_not_ready_lists_running_or_pending_pods_without_ready_true() -> None:
    class Regional:
        def kubectl(self, *_: Any, **__: Any) -> str:
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"namespace": "kube-system", "name": "ok"},
                            "status": {
                                "phase": "Running",
                                "conditions": [{"type": "Ready", "status": "True"}],
                            },
                        },
                        {
                            "metadata": {
                                "namespace": "gpu-fault-system",
                                "name": "dcgm",
                            },
                            "status": {
                                "phase": "Running",
                                "conditions": [{"type": "Ready", "status": "False"}],
                            },
                        },
                        {
                            "metadata": {"namespace": "x", "name": "pending"},
                            "status": {"phase": "Pending", "conditions": []},
                        },
                        {
                            "metadata": {"namespace": "x", "name": "done"},
                            "status": {"phase": "Succeeded", "conditions": []},
                        },
                    ]
                }
            )

    assert sp.pods_not_ready(Regional(), NODE) == ["gpu-fault-system/dcgm", "x/pending"]


def test_spare_residue_report_lists_every_spare_mark_and_honours_expected() -> None:
    from scripts.e2e.regional import warm_spare_residue as residue

    def node(
        name: str, *, label: bool = False, cordoned: bool = False, **annotations: str
    ):
        return {
            "metadata": {
                "name": name,
                "labels": {residue.SPARE_LABEL: "true"} if label else {},
                "annotations": annotations,
            },
            "spec": {"unschedulable": cordoned},
        }

    document = {
        "items": [
            node("clean"),
            node("kept", label=True, cordoned=True),
            node(
                "left",
                label=True,
                **{
                    residue.RESERVATION_ANNOTATION: "inc-1",
                    residue.POOL_STATE_ANNOTATION: "ALLOCATED",
                },
            ),
            node("available", **{residue.POOL_STATE_ANNOTATION: "AVAILABLE"}),
        ]
    }
    report = residue.spare_residue_report(document, expected=["kept"])
    assert report == {
        "declared_spares": ["kept", "left"],
        "cordoned_spares": ["kept"],
        "unexpected": ["left"],
        "reserved": {"left": "inc-1"},
        "pool_state": {"left": "ALLOCATED"},
        "expected": ["kept"],
        "clean": False,
    }
    assert residue.spare_residue_report({"items": [node("clean")]})["clean"] is True
    assert residue.RESIDUE_EXIT == 3


def _receipt(**overrides: Any) -> dict[str, Any]:
    before = {
        "name": NODE,
        "ready": "True",
        "unschedulable": False,
        "labels": {},
        "annotations": {sp.POOL_STATE_ANNOTATION: "AVAILABLE"},
    }
    base: dict[str, Any] = {
        "spare_before": before,
        "spare_after": dict(before),
        "residue": {"declared_spares": [], "unexpected": [], "clean": True},
        "cleanup": [{"name": "warm-spare", "ok": True, "result": {"released": True}}],
        "release": {"released": True},
    }
    base.update(overrides)
    return base


def test_restored_errors_accept_a_node_handed_back_as_found() -> None:
    assert sp.restored_errors(_receipt()) == []
    # A node that was cordoned before the declaration stays cordoned: the
    # release restores the recorded baseline, which is not a residue.
    before = {**_receipt()["spare_before"], "unschedulable": True}
    after = dict(before)
    assert sp.restored_errors(_receipt(spare_before=before, spare_after=after)) == []


def test_restored_errors_name_label_cordon_annotations_residue_and_failed_undo() -> (
    None
):
    after = {
        "name": NODE,
        "ready": "Unknown",
        "unschedulable": True,
        "labels": {sp.SPARE_LABEL: "true"},
        "annotations": {
            sp.POOL_STATE_ANNOTATION: "UNAVAILABLE",
            sp.RESERVATION_ANNOTATION: "inc-other",
        },
    }
    errors = sp.restored_errors(
        _receipt(
            spare_after=after,
            residue={"declared_spares": [NODE, "other"], "unexpected": ["other"]},
            cleanup=[{"name": "spare-kubelet", "ok": False, "error": "boom"}],
        )
    )
    joined = " ".join(errors)
    for needle in (
        "spare label is still on the candidate",
        "still cordoned",
        "not Ready after the release (Unknown)",
        f"annotation {sp.POOL_STATE_ANNOTATION} is 'UNAVAILABLE', was 'AVAILABLE'",
        f"annotation {sp.RESERVATION_ANNOTATION} is 'inc-other', was None",
        "left on ['other']",
        "undo spare-kubelet failed: boom",
    ):
        assert needle in joined, needle


def test_finalize_fails_a_graded_pass_when_the_spare_was_not_handed_back() -> None:
    result = {"verdict": "PASS", "stages": {}, "errors": [], "notes": []}
    clean = sp.finalize(dict(result), _receipt())
    assert clean["verdict"] == "PASS" and clean["stages"]["restored"] == []
    assert clean["spare_candidate"] == NODE and clean["spare_release"] == {
        "released": True
    }
    dirty = sp.finalize(
        dict(result),
        _receipt(residue={"declared_spares": [NODE], "unexpected": [NODE]}),
    )
    assert dirty["verdict"] == "FAIL"
    assert any("restored:" in item and "left on" in item for item in dirty["errors"])


def test_preflight_hook_is_the_candidate_proof_with_the_provider_recovery_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = SPECS["configuration_error"]
    settings = make_settings(spec, node=TARGET, peer=NODE)
    settings = dataclasses.replace(settings, hyperpod_cluster="")
    seen: dict[str, Any] = {}

    class FakeWarm:
        def __init__(self, regional: Any, scope: str) -> None:
            seen["scope"] = scope

        def cluster_recovery(self) -> dict[str, Any]:
            seen["read"] = True
            return {"node_repair_config": {"gpu-group": True}}

    import scripts.e2e.regional.warm_spare_fixture as wsf

    monkeypatch.setattr(wsf, "WarmSpareLiveFixture", FakeWarm)
    errors = spec.preflight(settings, _idle_regional())
    assert seen == {"scope": settings.regional.cluster_id, "read": True}
    assert errors and "node auto repair is enabled" in errors[0]


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
    session = FakeSpareSession.instances[-1]
    assert session.kubelet_stopped, "kubelet must have been stopped"
    assert session.verified and session.awaited_not_ready and session.finished, (
        "idleness proof before the declaration, the NotReady wait after the stop, "
        "the hand-back before the verdict"
    )
    assert result["stages"]["restored"] == [] and result["spare_after"]["name"] == NODE
    assert result["spare_release"] == {"released": True}


def test_every_variant_fails_when_the_spare_is_not_handed_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variant in (
        "not_ready",
        "stale_metrics",
        "configuration_error",
        "reserved_spare",
    ):
        node = _node(ready="False") if variant == "not_ready" else _node()
        store = (
            _store(observation_reasons=["collector observation UNKNOWN: GPU_METRICS"])
            if variant == "stale_metrics"
            else _store()
        )
        _patch(monkeypatch, node, store)
        sessions = FakeSpareSession.instances

        def dirty(ctx: Any, *, _node=node, _store=store) -> FakeSpareSession:
            session = FakeSpareSession(ctx, node=_node, store=_store)
            session.after = {**session.after, "labels": {sp.SPARE_LABEL: "true"}}
            return session

        monkeypatch.setattr(sp, "SpareSession", dirty)
        result = SPECS[variant].execute(_ctx(SPECS[variant], tmp_path))
        assert result["verdict"] == "FAIL", (variant, result["errors"])
        assert any(
            "spare label is still on the candidate" in e for e in result["errors"]
        )
        assert sessions[-1].finished, variant


def test_a_busy_candidate_is_inconclusive_before_anything_is_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Busy(FakeSpareSession):
        def verify_idle(self) -> None:
            raise VariantInconclusive(
                "spare candidate is not idle; nothing declared: x"
            )

    monkeypatch.setattr(
        sp, "SpareSession", lambda ctx: Busy(ctx, node=_node(), store=_store())
    )
    FakeSpareSession.instances = []
    with pytest.raises(VariantInconclusive, match="nothing declared"):
        SPECS["configuration_error"].execute(
            _ctx(SPECS["configuration_error"], tmp_path)
        )
    assert not FakeSpareSession.instances[-1].declared


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
