"""``gpu-fault-admin config spare``: the operator lever for a warm spare.

Every refusal the acceptance helper enforced is named here against fakes: the
node read, the patch, the pod list and the control-plane Agent lookup are all
injected, and nothing in this file reaches a cluster or a store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import warm_spare
from gpu_fault.admin.warm_spare import (
    DECLARE_CONFIRMATION,
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    RELEASE_CONFIRMATION,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    AgentState,
    WarmSpareConflict,
    WarmSpareError,
    WarmSpareRequest,
    declare,
    declare_refusals,
    gpu_workloads,
    node_snapshot,
    record_path,
    release,
    release_refusals,
    run_warm_spare,
)

NODE = "node-spare"
FAULT = "node-fault"
ACTIVE = AgentState(lifecycle_state="ACTIVE")


def _raw_node(name: str, **overrides: Any) -> dict[str, Any]:
    """A Kubernetes ``Node`` object as ``kubectl get node -o json`` prints it."""

    labels = {
        HYPERPOD_HEALTH_LABEL: "Schedulable",
        INSTANCE_GROUP_LABEL: "group-a",
        "node.kubernetes.io/instance-type": "ml.p5en.48xlarge",
        **overrides.pop("labels", {}),
    }
    annotations = dict(overrides.pop("annotations", {}))
    node: dict[str, Any] = {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "resourceVersion": "1",
            "labels": {k: v for k, v in labels.items() if v is not None},
            "annotations": {k: v for k, v in annotations.items() if v is not None},
        },
        "spec": {
            "providerID": f"aws:///{name}",
            "unschedulable": overrides.pop("unschedulable", False),
            "taints": overrides.pop("taints", []),
        },
        "status": {
            "conditions": [{"type": "Ready", "status": overrides.pop("ready", "True")}],
            "allocatable": {"nvidia.com/gpu": "8"},
        },
    }
    assert not overrides, f"unknown node overrides: {sorted(overrides)}"
    return node


def _snapshot(name: str = NODE, **overrides: Any) -> dict[str, Any]:
    return node_snapshot(_raw_node(name, **overrides))


def _gpu_pod(node: str, *, phase: str = "Running", gpus: int = 8) -> dict[str, Any]:
    return {
        "metadata": {"namespace": "team", "name": f"job-{node}"},
        "spec": {
            "nodeName": node,
            "containers": [{"resources": {"limits": {"nvidia.com/gpu": str(gpus)}}}],
        },
        "status": {"phase": phase},
    }


class FakeCoreApi:
    """The three CoreV1Api calls the flow makes, over plain JSON objects."""

    def __init__(
        self, *nodes: dict[str, Any], pods: list[dict[str, Any]] | None = None
    ):
        self.nodes = {node["metadata"]["name"]: node for node in nodes}
        self.pods = list(pods or [])
        self.patches: list[tuple[str, dict[str, Any]]] = []
        # Set by a test that wants a patch to be ignored by the cluster.
        self.apply_patches = True

    def read_node(self, name: str) -> dict[str, Any]:
        if name not in self.nodes:
            raise WarmSpareError(f"node {name} not found")
        return json.loads(json.dumps(self.nodes[name]))

    def patch_node(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        self.patches.append((name, json.loads(json.dumps(body))))
        if not self.apply_patches:
            return self.nodes[name]
        node = self.nodes[name]
        for key, value in body.get("metadata", {}).get("labels", {}).items():
            if value is None:
                node["metadata"]["labels"].pop(key, None)
            else:
                node["metadata"]["labels"][key] = value
        if "unschedulable" in body.get("spec", {}):
            node["spec"]["unschedulable"] = body["spec"]["unschedulable"]
        return node

    def list_pod_for_all_namespaces(self) -> dict[str, Any]:
        return {"items": list(self.pods)}

    def list_node(self, label_selector: str) -> dict[str, Any]:
        key, _, value = label_selector.partition("=")
        return {
            "items": [
                node
                for node in self.nodes.values()
                if node["metadata"]["labels"].get(key) == value
            ]
        }


class ConflictingCoreApi(FakeCoreApi):
    """``FakeCoreApi`` with the API server's optimistic concurrency.

    A patch carrying a stale ``resourceVersion`` is refused as a conflict; one
    carrying the current version applies and moves the version on, as the real
    server does. ``attempts`` records the version every attempt carried, so a
    test can tell a retry against a fresh read from a blind repeat.
    """

    def __init__(
        self, *nodes: dict[str, Any], pods: list[dict[str, Any]] | None = None
    ):
        super().__init__(*nodes, pods=pods)
        self.attempts: list[str | None] = []

    def patch_node(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        metadata = self.nodes[name]["metadata"]
        version = body.get("metadata", {}).get("resourceVersion")
        self.attempts.append(version)
        if version != metadata["resourceVersion"]:
            raise WarmSpareConflict(
                f"kubectl patch node {name} failed: resource version conflict "
                "(the node changed between survey and patch)"
            )
        result = super().patch_node(name, body)
        metadata["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)
        return result


def _moved_on(api: FakeCoreApi, node: str = NODE) -> dict[str, Any]:
    """Another writer updated ``node`` after the survey; returns the raw node."""

    raw = api.nodes[node]
    raw["metadata"]["resourceVersion"] = str(
        int(raw["metadata"]["resourceVersion"]) + 5
    )
    return raw


def _refusals(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "spare": _snapshot(),
        "fault": None,
        "fault_node": "",
        "declared": [],
        "workloads": [],
        "agent": ACTIVE,
    }
    arguments.update(overrides)
    return declare_refusals(NODE, **arguments)


# --- the node snapshot ------------------------------------------------------


def test_the_snapshot_keeps_exactly_the_fields_the_refusals_read() -> None:
    snapshot = _snapshot(
        unschedulable=True,
        taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
        annotations={"gpu-fault.io/incident-id": "INC-1"},
        labels={SPARE_LABEL: "true"},
    )

    assert snapshot["name"] == NODE
    assert snapshot["ready"] == "True"
    assert snapshot["unschedulable"] is True
    assert snapshot["gpu_allocatable"] == 8
    assert snapshot["taints"] == [
        {"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}
    ]
    assert snapshot["labels"][SPARE_LABEL] == "true"
    assert snapshot["labels"][HYPERPOD_HEALTH_LABEL] == "Schedulable"
    assert snapshot["annotations"]["gpu-fault.io/incident-id"] == "INC-1"
    assert snapshot["annotations"][SPARE_RESERVATION_ANNOTATION] is None


def test_gpu_workloads_keep_only_live_gpu_pods_on_the_node() -> None:
    pods = [
        _gpu_pod(NODE),
        _gpu_pod(NODE, phase="Succeeded"),
        _gpu_pod("node-other"),
        {
            "metadata": {"namespace": "kube-system", "name": "coredns"},
            "spec": {"nodeName": NODE, "containers": [{"resources": {}}]},
            "status": {"phase": "Running"},
        },
    ]

    assert gpu_workloads(pods, NODE) == [
        {
            "namespace": "team",
            "name": f"job-{NODE}",
            "node": NODE,
            "phase": "Running",
            "gpu_count": 8,
        }
    ]


# --- the declare refusal matrix ----------------------------------------------


def test_a_healthy_monitored_node_can_be_declared() -> None:
    assert _refusals() == []


def test_declaration_is_refused_when_the_node_is_not_ready() -> None:
    assert "node is not Ready" in _refusals(spare=_snapshot(ready="False"))


def test_a_labeled_and_cordoned_node_is_refused_as_already_declared() -> None:
    whole = _snapshot(labels={SPARE_LABEL: "true"}, unschedulable=True)

    assert _refusals(spare=whole, declared=[NODE]) == [
        "node is already declared as a spare and cordoned",
        "node is already in the declared spare set",
    ]


def test_a_labeled_but_schedulable_spare_can_be_redeclared_to_complete_the_cordon() -> (
    None
):
    """A validated restore that released the spare's quarantine also uncordoned
    it (DESTR-003 cleanup, 2026-09-08); the pool needs it cordoned again and
    the declaration is the supported way to cordon."""

    half = _snapshot(labels={SPARE_LABEL: "true"}, unschedulable=False)

    assert _refusals(spare=half, declared=[NODE]) == []


def test_declaration_is_refused_when_hyperpod_health_is_not_schedulable() -> None:
    refusals = _refusals(spare=_snapshot(labels={HYPERPOD_HEALTH_LABEL: "Pending"}))

    assert "HyperPod health label is not Schedulable" in refusals


def test_declaration_is_refused_when_the_node_carries_a_reservation() -> None:
    refusals = _refusals(
        spare=_snapshot(annotations={SPARE_RESERVATION_ANNOTATION: "INC-1"})
    )

    assert "node already carries a spare reservation" in refusals


def test_declaration_is_refused_when_the_pool_state_is_not_available() -> None:
    # Absent is fine -- the control plane owns this annotation -- but ALLOCATED
    # means the pool already spent this node.
    assert _refusals(
        spare=_snapshot(annotations={SPARE_POOL_STATE_ANNOTATION: "ALLOCATED"})
    ) == ["spare pool state is not AVAILABLE"]
    assert (
        _refusals(
            spare=_snapshot(annotations={SPARE_POOL_STATE_ANNOTATION: "AVAILABLE"})
        )
        == []
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"taints": [{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}]},
        {"annotations": {"gpu-fault.io/incident-id": "INC-1"}},
        {"annotations": {"gpu-fault.io/fencing-token": "7"}},
        {"annotations": {"gpu-fault.io/previous-unschedulable": "false"}},
    ],
)
def test_declaration_is_refused_on_quarantine_ownership(overrides: dict) -> None:
    refusals = _refusals(spare=_snapshot(**overrides))

    assert "node carries quarantine ownership from an earlier incident" in refusals


def test_declaration_is_refused_while_a_gpu_workload_is_running() -> None:
    # Cordoning does not evict. A spare declared under a live job is busy.
    refusals = _refusals(workloads=gpu_workloads([_gpu_pod(NODE)], NODE))

    assert "node still has an active GPU workload" in refusals


def test_declaration_is_refused_without_exactly_one_active_agent() -> None:
    # "纳入监控" is a hard precondition: an unmonitored node has no Agent to
    # fence and no health signal to trust.
    missing = _refusals(agent=AgentState(lifecycle_state=None))
    revoked = _refusals(agent=AgentState(lifecycle_state="REVOKED"))

    assert "node does not have exactly one ACTIVE Agent" in missing
    assert "node does not have exactly one ACTIVE Agent" in revoked


def test_an_unreachable_store_is_a_named_refusal_never_a_pass() -> None:
    refusals = _refusals(
        agent=AgentState(lifecycle_state=None, error="no Running CPU ingress Pod")
    )

    assert refusals == [
        "Node Agent state could not be read from the control-plane store: "
        "no Running CPU ingress Pod"
    ]


def test_declaration_is_refused_when_the_spare_is_the_fault_node() -> None:
    refusals = _refusals(fault=_snapshot(NODE), fault_node=NODE)

    assert "spare and fault node are identical" in refusals


def test_declaration_is_refused_on_a_topology_mismatch() -> None:
    fault = _snapshot(FAULT, labels={"node.kubernetes.io/instance-type": "ml.p4d"})

    refusals = _refusals(fault=fault, fault_node=FAULT)

    assert any(item.startswith("topology does not match") for item in refusals), (
        refusals
    )
    assert _refusals(fault=_snapshot(FAULT), fault_node=FAULT) == []


# --- the release refusals ----------------------------------------------------


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "node": NODE,
        "cluster_id": "cluster-a",
        "declared_at": "2026-09-08T00:00:00Z",
        "baseline": {"labels": {SPARE_LABEL: None}, "unschedulable": False},
        "pre_declaration_survey": {"node": _snapshot()},
    }
    record.update(overrides)
    return record


def test_release_is_refused_for_a_reserved_allocated_or_quarantined_node() -> None:
    reserved = _snapshot(annotations={SPARE_RESERVATION_ANNOTATION: "INC-1"})
    allocated = _snapshot(annotations={SPARE_POOL_STATE_ANNOTATION: "ALLOCATED"})
    quarantined = _snapshot(
        taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}]
    )
    owned = _snapshot(annotations={"gpu-fault.io/incident-id": "INC-1"})

    assert release_refusals(reserved, _record()) == [
        "node is reserved by an incident; release it through the incident"
    ]
    assert release_refusals(allocated, _record()) == [
        "node is ALLOCATED by the spare pool; release it through the incident"
    ]
    assert release_refusals(quarantined, _record()) == [
        "node is quarantined; restore the quarantine before releasing the spare"
    ]
    assert release_refusals(owned, _record()) == [
        "node is quarantined; restore the quarantine before releasing the spare"
    ]
    assert release_refusals(_snapshot(), _record()) == []


def test_release_is_refused_without_an_unreleased_declaration_record() -> None:
    assert release_refusals(_snapshot(), None) == [
        "node was not declared by gpu-fault-admin (no unreleased record)"
    ]
    assert release_refusals(_snapshot(), _record(released_at="x")) == [
        "node was not declared by gpu-fault-admin (no unreleased record)"
    ]


# --- declare / release against the fakes -------------------------------------


def _survey(api: FakeCoreApi, node: str = NODE) -> dict[str, Any]:
    return warm_spare.survey(
        api,
        node=node,
        fault_node="",
        cluster_id="cluster-a",
        agent_lookup=lambda _cluster, _node: ACTIVE,
    )


def test_declare_writes_the_baseline_before_patching_and_verifies_after(
    tmp_path: Path,
) -> None:
    api = FakeCoreApi(_raw_node(NODE))
    record_file = record_path(tmp_path, NODE)
    seen_at_patch: list[dict[str, Any]] = []
    original = api.patch_node

    def patch_node(name: str, body: dict[str, Any]) -> dict[str, Any]:
        # The record must already be on disk when the cluster is touched, so
        # an interrupted declaration still says what to put back.
        seen_at_patch.append(json.loads(record_file.read_text(encoding="utf-8")))
        return original(name, body)

    api.patch_node = patch_node  # type: ignore[method-assign]

    record = declare(
        api,
        node=NODE,
        record=record_file,
        reference="CHG-1",
        actor="alice",
        survey=_survey(api),
    )

    assert record_file == tmp_path / "warm-spares" / f"{NODE}.json"
    (before,) = seen_at_patch
    assert before["baseline"] == {"labels": {SPARE_LABEL: None}, "unschedulable": False}
    assert before["confirmation"] == DECLARE_CONFIRMATION
    assert before["reference"] == "CHG-1"
    assert before["actor"] == "alice"
    assert "declared_state" not in before
    assert api.patches == [
        (
            NODE,
            {
                "metadata": {
                    "uid": f"uid-{NODE}",
                    "resourceVersion": "1",
                    "labels": {SPARE_LABEL: "true"},
                },
                "spec": {"unschedulable": True},
            },
        )
    ]
    assert record["declared_state"]["labels"][SPARE_LABEL] == "true"
    assert record["declared_state"]["unschedulable"] is True
    assert record["declared_spares"] == [NODE]
    written = json.loads(record_file.read_text(encoding="utf-8"))
    assert written["declared_state"] == record["declared_state"]
    assert record_file.stat().st_mode & 0o077 == 0, "record holds node identities"


def test_declare_fails_when_the_patch_did_not_take_effect(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE))
    api.apply_patches = False

    with pytest.raises(WarmSpareError, match="did not take effect"):
        declare(
            api,
            node=NODE,
            record=record_path(tmp_path, NODE),
            reference="CHG-1",
            actor="alice",
            survey=_survey(api),
        )


@pytest.mark.parametrize(
    "record",
    [
        # The declaration took effect: declared_state is written only after
        # the patch was verified on the node.
        _record(
            declared_state=_snapshot(labels={SPARE_LABEL: "true"}, unschedulable=True)
        ),
        # No declared_state, but the node no longer reads as the baseline the
        # record holds: something did change, and release must put it back.
        _record(baseline={"labels": {SPARE_LABEL: None}, "unschedulable": True}),
        _record(baseline={"labels": {SPARE_LABEL: "true"}, "unschedulable": False}),
    ],
    ids=["took-effect", "cordon-baseline-differs", "label-baseline-differs"],
)
def test_declare_refuses_to_overwrite_an_unreleased_record(
    tmp_path: Path, record: dict[str, Any]
) -> None:
    # Overwriting would destroy the only record of what the node looked like
    # before it became a spare.
    api = FakeCoreApi(_raw_node(NODE))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(WarmSpareError, match="unreleased declaration"):
        declare(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-1",
            actor="alice",
            survey=_survey(api),
        )
    assert api.patches == [], "a refused declaration must not touch the node"
    assert json.loads(record_file.read_text(encoding="utf-8")) == record, (
        "a refused declaration must leave the existing record untouched"
    )


def test_declare_supersedes_an_interrupted_declaration_whose_patch_never_took_effect(
    tmp_path: Path,
) -> None:
    # The live sequence behind this test: a declaration wrote its record, the
    # patch hit a resource version conflict, and the node was never changed.
    # The record has no declared_state and its baseline still describes the
    # node, so there is nothing to release; the next declaration keeps it as
    # forensic history instead of demanding a no-op --release first.
    api = FakeCoreApi(_raw_node(NODE))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    interrupted = _record(
        reference="CHG-0",
        actor="carol",
        superseded_interrupted_declarations=[
            {"reference": "CHG-00", "superseded_at": "2026-09-08T01:00:00Z"}
        ],
    )
    record_file.write_text(json.dumps(interrupted), encoding="utf-8")

    record = declare(
        api,
        node=NODE,
        record=record_file,
        reference="CHG-1",
        actor="alice",
        survey=_survey(api),
    )

    assert record["reference"] == "CHG-1", (
        "the new declaration carries its own reference"
    )
    assert record["declared_state"]["labels"][SPARE_LABEL] == "true", (
        "the fresh declaration must take effect"
    )
    assert len(api.patches) == 1, "exactly one patch: the fresh declaration"
    earlier, superseded = record["superseded_interrupted_declarations"]
    assert earlier == {
        "reference": "CHG-00",
        "superseded_at": "2026-09-08T01:00:00Z",
    }, "history already carried by the interrupted record is kept, oldest first"
    assert superseded["reference"] == "CHG-0", "the interrupted record is kept"
    assert superseded["actor"] == "carol", "the interrupted record keeps its actor"
    assert superseded["baseline"] == {
        "labels": {SPARE_LABEL: None},
        "unschedulable": False,
    }, "the interrupted record keeps its baseline"
    assert superseded["superseded_at"], "the history entry must say when"
    assert "pre_declaration_survey" not in superseded, (
        "history holds the document minus its surveys"
    )
    assert "superseded_interrupted_declarations" not in superseded, (
        "history is kept flat, never nested"
    )
    written = json.loads(record_file.read_text(encoding="utf-8"))
    assert (
        written["superseded_interrupted_declarations"]
        == (record["superseded_interrupted_declarations"])
    ), "the history must be persisted with the new record"


def test_release_restores_the_recorded_baseline_not_the_current_state(
    tmp_path: Path,
) -> None:
    # A node that was already cordoned before the declaration must stay cordoned
    # after the release. Restoring "uncordon" unconditionally would hand an
    # operator-cordoned node back to the scheduler.
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(
        json.dumps(
            _record(baseline={"labels": {SPARE_LABEL: None}, "unschedulable": True})
        ),
        encoding="utf-8",
    )

    record = release(
        api,
        node=NODE,
        record=record_file,
        reference="CHG-2",
        actor="bob",
        survey=_survey(api),
    )

    assert api.patches == [
        (
            NODE,
            {
                "metadata": {
                    "uid": f"uid-{NODE}",
                    "resourceVersion": "1",
                    "labels": {SPARE_LABEL: None},
                },
                "spec": {"unschedulable": True},
            },
        )
    ]
    assert record["released_at"], "the record must be marked released"
    assert record["release_reference"] == "CHG-2"
    assert record["release_actor"] == "bob"
    assert record["restored_state"]["unschedulable"] is True
    assert record["restored_state"]["labels"][SPARE_LABEL] is None
    assert json.loads(record_file.read_text(encoding="utf-8"))["released_at"], (
        "released_at must be persisted"
    )


def test_release_uncordons_a_node_that_was_schedulable_before(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record()), encoding="utf-8")

    record = release(
        api,
        node=NODE,
        record=record_file,
        reference="CHG-2",
        actor="bob",
        survey=_survey(api),
    )

    assert record["restored_state"]["unschedulable"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cluster_id": "cluster-b"}, "cluster"),
        ({"cluster_id": None}, "cluster"),
        ({"pre_declaration_survey": {"node": {"uid": "replaced-node"}}}, "UID"),
        ({"pre_declaration_survey": {}}, "UID"),
        ({"baseline": {}}, "baseline"),
        ({"baseline": {"labels": {}, "unschedulable": False}}, "baseline"),
    ],
)
def test_release_refuses_unbound_or_incomplete_baselines_before_patching(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record(**overrides)), encoding="utf-8")

    with pytest.raises(WarmSpareError, match=message):
        release(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-2",
            actor="bob",
            survey=_survey(api),
        )
    assert api.patches == []
    assert not json.loads(record_file.read_text()).get("released_at"), (
        "rejected release must not record a release timestamp"
    )


def test_release_does_not_record_success_when_the_patch_did_not_restore_the_node(
    tmp_path: Path,
) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))
    api.apply_patches = False
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record()), encoding="utf-8")

    with pytest.raises(WarmSpareError, match="did not restore"):
        release(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-2",
            actor="bob",
            survey=_survey(api),
        )
    assert not json.loads(record_file.read_text()).get("released_at"), (
        "unrestored node must not be recorded as released"
    )


def test_release_patch_is_bound_to_the_surveyed_resource_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))
    before = _survey(api)
    api.nodes[NODE]["metadata"]["resourceVersion"] = "2"
    api.nodes[NODE]["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] = "inc-new"
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record()), encoding="utf-8")
    original = api.patch_node

    def checked_patch(node, body):
        assert body["metadata"]["uid"] == f"uid-{NODE}"
        if (
            body["metadata"].get("resourceVersion")
            != api.nodes[node]["metadata"]["resourceVersion"]
        ):
            raise WarmSpareError("resource version conflict")
        return original(node, body)

    monkeypatch.setattr(api, "patch_node", checked_patch)
    with pytest.raises(WarmSpareError, match="resource version conflict"):
        release(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-2",
            actor="bob",
            survey=before,
        )
    assert api.patches == []
    assert api.nodes[NODE]["spec"]["unschedulable"] is True


# --- resource version conflicts ----------------------------------------------


def _declare(
    api: FakeCoreApi, tmp_path: Path, survey: dict[str, Any]
) -> dict[str, Any]:
    return declare(
        api,
        node=NODE,
        record=record_path(tmp_path, NODE),
        reference="CHG-1",
        actor="alice",
        survey=survey,
    )


def test_declare_retries_a_stale_resource_version_against_a_fresh_read(
    tmp_path: Path,
) -> None:
    # A kubelet heartbeat or a device plugin re-registering moves the node's
    # resourceVersion without touching the label or the cordon. The
    # declaration re-reads the node and retries with the fresh version.
    api = ConflictingCoreApi(_raw_node(NODE))
    before = _survey(api)
    raw = _moved_on(api)
    raw["metadata"]["annotations"]["node.alpha.kubernetes.io/ttl"] = "0"
    raw["status"]["allocatable"]["nvidia.com/gpu"] = "0"

    record = _declare(api, tmp_path, before)

    assert api.attempts == ["1", "6"], (
        "one attempt with the surveyed version, one with the freshly read one"
    )
    assert [body["metadata"]["resourceVersion"] for _name, body in api.patches] == [
        "6"
    ], "the patch that applied was bound to the fresh resource version"
    assert api.nodes[NODE]["metadata"]["labels"][SPARE_LABEL] == "true", (
        "the retried declaration must label the node"
    )
    assert api.nodes[NODE]["spec"]["unschedulable"] is True, (
        "the retried declaration must cordon the node"
    )
    assert record["declared_state"]["resource_version"] == "7", (
        "declared_state is read back after the retried patch"
    )


def test_release_retries_a_stale_resource_version_against_a_fresh_read(
    tmp_path: Path,
) -> None:
    api = ConflictingCoreApi(
        _raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True)
    )
    before = _survey(api)
    _moved_on(api)
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record()), encoding="utf-8")

    record = release(
        api,
        node=NODE,
        record=record_file,
        reference="CHG-2",
        actor="bob",
        survey=before,
    )

    assert api.attempts == ["1", "6"], "the release retries once with the fresh version"
    assert api.nodes[NODE]["spec"]["unschedulable"] is False, (
        "the retried release must uncordon the node"
    )
    assert SPARE_LABEL not in api.nodes[NODE]["metadata"]["labels"], (
        "the retried release must remove the spare label"
    )
    assert record["released_at"], "the retried release must be recorded"


@pytest.mark.parametrize(
    ("labels", "unschedulable", "named"),
    [({SPARE_LABEL: "true"}, False, "spare label"), ({}, True, "unschedulable")],
    ids=["someone-labeled-it", "someone-cordoned-it"],
)
def test_a_conflict_retry_refuses_a_spare_label_or_cordon_changed_since_the_survey(
    tmp_path: Path, labels: dict[str, str], unschedulable: bool, named: str
) -> None:
    # The retry is honest only while the concurrent change touched nothing the
    # patch is about to change. A hand label or cordon between survey and patch
    # is refused, never silently overwritten.
    api = ConflictingCoreApi(_raw_node(NODE))
    before = _survey(api)
    raw = _moved_on(api)
    raw["metadata"]["labels"].update(labels)
    raw["spec"]["unschedulable"] = unschedulable

    with pytest.raises(
        WarmSpareError, match="changed between survey and patch"
    ) as failure:
        _declare(api, tmp_path, before)

    assert named in str(failure.value), "the refusal names the field that changed"
    assert api.attempts == ["1"], "no retry over a changed label or cordon"
    assert api.patches == [], "nothing was written over the concurrent change"
    assert api.nodes[NODE]["metadata"]["labels"].get(SPARE_LABEL) == labels.get(
        SPARE_LABEL
    ), "the node keeps the label the other writer left"
    assert api.nodes[NODE]["spec"]["unschedulable"] is unschedulable, (
        "the node keeps the cordon state the other writer left"
    )


def test_a_conflict_retry_refuses_a_spare_reservation_that_appeared_since_the_survey(
    tmp_path: Path,
) -> None:
    # The pool reserved this spare for an incident inside the conflict window.
    # The release refusals were computed before that; retrying would hand a
    # reserved spare back to the scheduler under a running workflow.
    api = ConflictingCoreApi(
        _raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True)
    )
    before = _survey(api)
    raw = _moved_on(api)
    raw["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] = "inc-new"
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record()), encoding="utf-8")

    with pytest.raises(
        WarmSpareError, match="changed between survey and patch"
    ) as failure:
        release(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-2",
            actor="bob",
            survey=before,
        )

    assert "spare reservation" in str(failure.value), (
        "the refusal names the reservation that appeared"
    )
    assert api.attempts == ["1"], "no retry over a reservation that appeared"
    assert api.patches == [], "nothing was written over the reservation"
    assert api.nodes[NODE]["spec"]["unschedulable"] is True, (
        "the reserved spare stays cordoned"
    )
    assert api.nodes[NODE]["metadata"]["labels"][SPARE_LABEL] == "true", (
        "the reserved spare keeps its label"
    )
    assert not json.loads(record_file.read_text(encoding="utf-8")).get("released_at"), (
        "the refused release must not be recorded as released"
    )


@pytest.mark.parametrize(
    "ownership",
    [
        {"taints": [{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}]},
        {"annotations": {"gpu-fault.io/incident-id": "INC-9"}},
    ],
    ids=["quarantine-taint", "incident-annotation"],
)
def test_a_conflict_retry_refuses_quarantine_ownership_that_appeared_since_the_survey(
    tmp_path: Path, ownership: dict[str, Any]
) -> None:
    # The control plane quarantined the node inside the conflict window: it
    # now belongs to an incident, and the declaration must not label it a
    # spare over that ownership.
    api = ConflictingCoreApi(_raw_node(NODE))
    before = _survey(api)
    raw = _moved_on(api)
    raw["spec"]["taints"] = ownership.get("taints", [])
    raw["metadata"]["annotations"].update(ownership.get("annotations", {}))

    with pytest.raises(
        WarmSpareError, match="changed between survey and patch"
    ) as failure:
        _declare(api, tmp_path, before)

    assert "quarantine ownership" in str(failure.value), (
        "the refusal names the ownership that appeared"
    )
    assert api.attempts == ["1"], "no retry over quarantine ownership that appeared"
    assert api.patches == [], "nothing was written over the quarantine"
    assert SPARE_LABEL not in api.nodes[NODE]["metadata"]["labels"], (
        "the quarantined node is not labeled a spare"
    )
    assert api.nodes[NODE]["spec"]["unschedulable"] is False, (
        "the node's cordon is left to the incident that owns it"
    )


def _contended(api: ConflictingCoreApi) -> None:
    """Something else writes the node between every read and our patch."""

    original = api.patch_node

    def contended(name: str, body: dict[str, Any]) -> dict[str, Any]:
        _moved_on(api, name)
        return original(name, body)

    api.patch_node = contended  # type: ignore[method-assign]


def test_three_consecutive_conflicts_fail_with_a_conflict_naming_error(
    tmp_path: Path,
) -> None:
    api = ConflictingCoreApi(_raw_node(NODE))
    before = _survey(api)
    _contended(api)

    with pytest.raises(WarmSpareError, match="resource version conflict") as failure:
        _declare(api, tmp_path, before)

    assert not isinstance(failure.value, WarmSpareConflict), (
        "the bounded failure is final, not another retryable conflict"
    )
    assert api.attempts == ["1", "6", "11"], (
        "three attempts, each bound to the version last read"
    )
    assert api.patches == [], "nothing was written"
    assert api.nodes[NODE]["spec"]["unschedulable"] is False, "the node is unchanged"
    assert "declared_state" not in json.loads(
        record_path(tmp_path, NODE).read_text(encoding="utf-8")
    ), "the record must not claim a declaration that never took effect"


def test_a_non_conflict_patch_failure_is_not_retried(tmp_path: Path) -> None:
    api = ConflictingCoreApi(_raw_node(NODE))
    before = _survey(api)
    calls: list[str] = []

    def forbidden(name: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body["metadata"]["resourceVersion"])
        raise WarmSpareError(f"kubectl patch node {name} failed: Forbidden: <redacted>")

    api.patch_node = forbidden  # type: ignore[method-assign]

    with pytest.raises(WarmSpareError, match="Forbidden") as failure:
        _declare(api, tmp_path, before)

    assert calls == ["1"], "a non-conflict failure is never retried"
    assert str(failure.value) == (
        f"kubectl patch node {NODE} failed: Forbidden: <redacted>"
    ), "a non-conflict failure propagates unchanged"


def test_a_declaration_interrupted_by_conflicts_is_completed_by_the_next_command(
    tmp_path: Path,
) -> None:
    # The operator's path through the command: the first --declare exhausts its
    # attempts and leaves a record without declared_state; the next --declare
    # supersedes that record instead of demanding a --release of nothing.
    api = ConflictingCoreApi(_raw_node(NODE))
    _contended(api)
    with pytest.raises(WarmSpareError, match="resource version conflict"):
        run_warm_spare(
            _site(),
            _request(
                tmp_path,
                mode="declare",
                reference="CHG-1",
                confirmation=DECLARE_CONFIRMATION,
            ),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    left_behind = json.loads(record_path(tmp_path, NODE).read_text(encoding="utf-8"))
    assert "declared_state" not in left_behind, "the interrupted record has no state"
    assert api.nodes[NODE]["spec"]["unschedulable"] is False, "the node is unchanged"

    del api.patch_node  # the cluster calms down
    report = run_warm_spare(
        _site(),
        _request(
            tmp_path,
            mode="declare",
            reference="CHG-2",
            confirmation=DECLARE_CONFIRMATION,
        ),
        api=api,
        agent_lookup=lambda *_: ACTIVE,
    )

    assert report["ready"] is True, "the second declaration succeeds without --release"
    assert report["declaration"]["reference"] == "CHG-2", "the report is the new one"
    (superseded,) = report["declaration"]["superseded_interrupted_declarations"]
    assert superseded["reference"] == "CHG-1", "the interrupted declaration is history"
    assert api.nodes[NODE]["spec"]["unschedulable"] is True, "the node is cordoned"
    assert api.nodes[NODE]["metadata"]["labels"][SPARE_LABEL] == "true", (
        "the node is labeled"
    )


def test_kubectl_conflict_stderr_is_a_named_conflict_that_stays_redacted() -> None:
    marker = "synthetic-unstructured-auth-value-24680"

    class Conflict:
        stdout = ""
        stderr = (
            "Error from server (Conflict): Operation cannot be fulfilled on nodes "
            f'"{NODE}": the object has been modified; please apply your changes to '
            f"the latest version and try again\n{marker}"
        )
        returncode = 1

    api = warm_spare.KubectlNodeApi(["kubectl"], run=lambda *_a, **_k: Conflict())

    with pytest.raises(WarmSpareConflict) as failure:
        api.patch_node(NODE, {"spec": {"unschedulable": True}})

    message = str(failure.value)
    assert message.startswith(f"kubectl patch node {NODE} failed: "), (
        "the conflict names the failed kubectl verb"
    )
    assert "resource version conflict" in message, "the conflict is named as such"
    assert marker not in message, "server output stays redacted"

    class Forbidden:
        stdout = ""
        stderr = f"Error from server (Forbidden): nodes is forbidden\n{marker}"
        returncode = 1

    api = warm_spare.KubectlNodeApi(["kubectl"], run=lambda *_a, **_k: Forbidden())
    with pytest.raises(WarmSpareError) as other:
        api.patch_node(NODE, {"spec": {"unschedulable": True}})
    assert not isinstance(other.value, WarmSpareConflict), (
        "only the conflict wording is a retryable conflict"
    )
    assert "Forbidden" in str(other.value), "other failures keep their redacted code"


def test_unknown_spare_pool_state_refuses_release() -> None:
    refusals = release_refusals(
        _snapshot(annotations={SPARE_POOL_STATE_ANNOTATION: "UNKNOWN"}), _record()
    )
    assert "spare pool state is unknown or unavailable" in refusals


def test_release_refuses_an_already_released_or_mismatched_record(
    tmp_path: Path,
) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record(released_at="x")), encoding="utf-8")

    with pytest.raises(WarmSpareError, match="already released"):
        release(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-2",
            actor="bob",
            survey=_survey(api),
        )

    record_file.write_text(json.dumps(_record(node="node-other")), encoding="utf-8")
    with pytest.raises(WarmSpareError, match="records node node-other"):
        release(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-2",
            actor="bob",
            survey=_survey(api),
        )
    assert api.patches == []


def test_the_record_path_refuses_a_node_name_that_escapes_the_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(WarmSpareError, match="node name"):
        record_path(tmp_path, "../site.yaml")


# --- the command flow --------------------------------------------------------


def _request(state_dir: Path, **overrides: Any) -> WarmSpareRequest:
    values: dict[str, Any] = {
        "state_dir": state_dir,
        "node": NODE,
        "fault_node": "",
        "cluster_id": None,
        "reference": None,
        "mode": "check",
        "confirmation": "",
    }
    values.update(overrides)
    return WarmSpareRequest(**values)


def _site(clusters: int = 1) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        release_config={
            "clusters": [
                {"cluster_id": f"cluster-{index}", "context": f"ctx-{index}"}
                for index in range(clusters)
            ],
            "gpu_kubeconfig": "/state/gpu.kubeconfig",
        },
        environment={},
    )


def test_the_read_only_check_reports_refusals_and_touches_nothing(
    tmp_path: Path,
) -> None:
    api = FakeCoreApi(_raw_node(NODE, ready="False"))

    report = run_warm_spare(
        _site(), _request(tmp_path), api=api, agent_lookup=lambda *_: ACTIVE
    )

    assert report["mode"] == "check"
    assert report["ready"] is False
    assert report["refusals"] == ["node is not Ready"]
    assert report["cluster_id"] == "cluster-0"
    assert api.patches == []
    assert not (tmp_path / "warm-spares").exists(), "a check must write no record"


def test_declare_needs_the_exact_confirmation_and_a_reference(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE))

    with pytest.raises(WarmSpareError, match=f"--confirm {DECLARE_CONFIRMATION}"):
        run_warm_spare(
            _site(),
            _request(tmp_path, mode="declare", reference="CHG-1", confirmation="yes"),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    with pytest.raises(WarmSpareError, match="requires --reference"):
        run_warm_spare(
            _site(),
            _request(tmp_path, mode="declare", confirmation=DECLARE_CONFIRMATION),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    with pytest.raises(WarmSpareError, match="reference is invalid"):
        run_warm_spare(
            _site(),
            _request(
                tmp_path,
                mode="declare",
                reference="x",
                confirmation=DECLARE_CONFIRMATION,
            ),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    assert api.patches == []


def test_declare_then_release_round_trip_through_the_command(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE))
    lookups: list[tuple[str, str]] = []

    def agent_lookup(cluster_id: str, node: str) -> AgentState:
        lookups.append((cluster_id, node))
        return ACTIVE

    declared = run_warm_spare(
        _site(),
        _request(
            tmp_path,
            mode="declare",
            reference="CHG-1",
            confirmation=DECLARE_CONFIRMATION,
        ),
        api=api,
        agent_lookup=agent_lookup,
    )

    assert declared["mode"] == "declare"
    assert declared["refusals"] == []
    assert declared["declaration"]["node"] == NODE
    assert "pre_declaration_survey" not in declared["declaration"], (
        "the report must not contain itself"
    )
    json.dumps(declared)
    assert lookups == [("cluster-0", NODE)]
    assert api.nodes[NODE]["spec"]["unschedulable"] is True
    assert api.nodes[NODE]["metadata"]["labels"][SPARE_LABEL] == "true"

    released = run_warm_spare(
        _site(),
        _request(
            tmp_path,
            mode="release",
            reference="CHG-2",
            confirmation=RELEASE_CONFIRMATION,
        ),
        api=api,
        agent_lookup=agent_lookup,
    )

    assert released["mode"] == "release"
    assert released["release"]["released_at"], "release must stamp released_at"
    assert api.nodes[NODE]["spec"]["unschedulable"] is False
    assert SPARE_LABEL not in api.nodes[NODE]["metadata"]["labels"]


def test_declare_with_refusals_raises_and_writes_no_record(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE), pods=[_gpu_pod(NODE)])

    with pytest.raises(WarmSpareError, match="active GPU workload"):
        run_warm_spare(
            _site(),
            _request(
                tmp_path,
                mode="declare",
                reference="CHG-1",
                confirmation=DECLARE_CONFIRMATION,
            ),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    assert api.patches == []
    assert not record_path(tmp_path, NODE).exists(), "refused: no record written"


def test_release_with_refusals_raises_before_any_patch(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))

    with pytest.raises(WarmSpareError, match="no unreleased record"):
        run_warm_spare(
            _site(),
            _request(
                tmp_path,
                mode="release",
                reference="CHG-2",
                confirmation=RELEASE_CONFIRMATION,
            ),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    assert api.patches == []


def test_a_multi_cluster_site_needs_the_cluster_named(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE))

    with pytest.raises(WarmSpareError, match="--cluster-id"):
        run_warm_spare(
            _site(clusters=2),
            _request(tmp_path),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    report = run_warm_spare(
        _site(clusters=2),
        _request(tmp_path, cluster_id="cluster-1"),
        api=api,
        agent_lookup=lambda *_: ACTIVE,
    )
    assert report["cluster_id"] == "cluster-1"
    with pytest.raises(WarmSpareError, match="not in the managed site"):
        run_warm_spare(
            _site(),
            _request(tmp_path, cluster_id="cluster-9"),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )


def test_the_default_node_api_binds_the_site_gpu_kubeconfig_and_refuses_none() -> None:
    api = warm_spare.node_api_for_site(_site(), {"cluster_id": "c", "context": "ctx"})

    assert api.kubectl == [
        "kubectl",
        "--kubeconfig",
        "/state/gpu.kubeconfig",
        "--context",
        "ctx",
    ]
    bare = _site()
    bare.release_config.pop("gpu_kubeconfig")
    with pytest.raises(Exception, match="no GPU kubeconfig"):
        warm_spare.node_api_for_site(bare, {"cluster_id": "c", "context": "ctx"})


def test_the_kubectl_node_api_reads_patches_and_lists_through_kubectl() -> None:
    calls: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def run(command: list[str], **_kwargs: Any) -> Completed:
        calls.append(command)
        return Completed(json.dumps({"metadata": {"name": NODE}, "items": []}))

    api = warm_spare.KubectlNodeApi(["kubectl", "--context", "ctx"], run=run)

    api.read_node(NODE)
    api.patch_node(NODE, {"spec": {"unschedulable": True}})
    api.list_pod_for_all_namespaces()
    api.list_node(f"{SPARE_LABEL}=true")

    assert calls[0] == [
        "kubectl",
        "--context",
        "ctx",
        "get",
        "node",
        NODE,
        "-o",
        "json",
    ]
    assert calls[1][:6] == ["kubectl", "--context", "ctx", "patch", "node", NODE]
    assert "--type=merge" in calls[1]
    assert json.loads(calls[1][-1]) == {"spec": {"unschedulable": True}}
    assert calls[2] == [
        "kubectl",
        "--context",
        "ctx",
        "get",
        "pod",
        "--all-namespaces",
        "-o",
        "json",
    ]
    assert calls[3] == [
        "kubectl",
        "--context",
        "ctx",
        "get",
        "node",
        "-l",
        f"{SPARE_LABEL}=true",
        "-o",
        "json",
    ]


def test_a_failing_kubectl_is_a_named_error() -> None:
    class Completed:
        stdout = ""
        stderr = "Unable to connect to the server"
        returncode = 1

    api = warm_spare.KubectlNodeApi(["kubectl"], run=lambda *_a, **_k: Completed())

    with pytest.raises(WarmSpareError, match="Unable to connect"):
        api.read_node(NODE)


def test_kubectl_failure_does_not_expose_credential_helper_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "synthetic-unstructured-auth-value-13579"

    class Completed:
        stdout = ""
        stderr = f"Forbidden: exec helper failed\n{marker}"
        returncode = 1

    monkeypatch.setattr(
        warm_spare, "run_command", lambda *_args, **_kwargs: Completed()
    )
    api = warm_spare.KubectlNodeApi(["kubectl"])
    with pytest.raises(WarmSpareError) as failure:
        api.read_node(NODE)
    message = str(failure.value)
    assert marker not in message
    assert "Forbidden" in message
    assert "redacted" in message


def test_the_control_plane_agent_lookup_never_passes_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault.admin.bootstrap_common import BootstrapError

    def unreachable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise BootstrapError("found no Running CPU ingress Pod")

    monkeypatch.setattr(warm_spare, "run_control_plane_script", unreachable)
    lookup = warm_spare.control_plane_agent_lookup(_site())

    state = lookup("cluster-0", NODE)

    assert state.lifecycle_state is None
    assert state.error == "found no Running CPU ingress Pod"

    monkeypatch.setattr(
        warm_spare,
        "run_control_plane_script",
        lambda *_a, **_k: {"found": True, "lifecycle_state": "ACTIVE"},
    )
    assert lookup("cluster-0", NODE) == ACTIVE
    monkeypatch.setattr(
        warm_spare, "run_control_plane_script", lambda *_a, **_k: {"found": False}
    )
    assert lookup("cluster-0", NODE) == AgentState(lifecycle_state=None)
