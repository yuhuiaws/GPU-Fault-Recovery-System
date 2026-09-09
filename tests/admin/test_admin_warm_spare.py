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
        "declared_at": "2026-09-08T00:00:00Z",
        "baseline": {"labels": {SPARE_LABEL: None}, "unschedulable": False},
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
                "metadata": {"labels": {SPARE_LABEL: "true"}},
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


def test_declare_refuses_to_overwrite_an_unreleased_record(tmp_path: Path) -> None:
    # Overwriting would destroy the only record of what the node looked like
    # before it became a spare.
    api = FakeCoreApi(_raw_node(NODE))
    record_file = record_path(tmp_path, NODE)
    record_file.parent.mkdir(parents=True)
    record_file.write_text(json.dumps(_record()), encoding="utf-8")

    with pytest.raises(WarmSpareError, match="unreleased declaration"):
        declare(
            api,
            node=NODE,
            record=record_file,
            reference="CHG-1",
            actor="alice",
            survey=_survey(api),
        )
    assert api.patches == []


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
                "metadata": {"labels": {SPARE_LABEL: None}},
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
