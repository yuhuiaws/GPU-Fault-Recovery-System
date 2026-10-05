"""The upgrade wave skips nodes the product itself isolated and ends PARTIAL.

Live 2026-10-05 (mixed site): the product quarantined ip-10-0-41-164 during the
EKS node wave, the node-state gate called the cordon + quarantine taint unsafe,
and the whole release -- HyperPod included -- rolled back for twenty minutes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_commands as COMMANDS
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_preflight as PREFLIGHT
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_product_isolation as ISOLATION
from gpu_fault_release import regional_release_resume_validation as RESUME
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import (
    ReleaseExecutionPlan,
    diff_from_changed,
)
from tests.regional._cov95_release_support import (
    ResourceRelease,
    mixed_probe_result,
    previous_snapshot,
)
from tests.regional._release_orchestrator_support import config_file, phase_release

NODES = ("node-a", "node-b", "node-c")
IDENTITY = ("b" * 64, "e" * 64)
INCIDENT = "inc-kernel-log-kmsg-xid-46"


def _node(
    name: str,
    *,
    taints: list[dict[str, str]] | None = None,
    annotations: dict[str, str] | None = None,
    unschedulable: bool = False,
    ready: bool = True,
) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"},
            "annotations": annotations or {},
        },
        "spec": {"unschedulable": unschedulable, "taints": taints or []},
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


def _quarantined(name: str, incident: str = INCIDENT) -> dict[str, Any]:
    return _node(
        name,
        unschedulable=True,
        taints=[
            {
                "key": "gpu-fault.io/quarantined",
                "value": "incident-abc",
                "effect": "NoSchedule",
            },
            {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"},
        ],
        annotations={
            "gpu-fault.io/incident-id": incident,
            "gpu-fault.io/fencing-token": "1",
        },
    )


@pytest.mark.parametrize(
    ("node", "reasons"),
    [
        (_quarantined("n"), ["quarantine-taint", "cordoned-by-incident"]),
        (
            _node("n", taints=[{"key": "gpu-fault.io/quiesce", "effect": "NoExecute"}]),
            ["quiesce-taint"],
        ),
        (
            _node(
                "n",
                unschedulable=True,
                annotations={"gpu-fault.io/incident-id": INCIDENT},
            ),
            ["cordoned-by-incident"],
        ),
        (_node("n", unschedulable=True), None),
        (_node("n", ready=False), None),
        (_node("n", taints=[{"key": "nvidia.com/gpu", "value": "present"}]), None),
        (_node("n", annotations={"gpu-fault.io/incident-id": INCIDENT}), None),
    ],
    ids=[
        "quarantine",
        "quiesce",
        "cordon-with-incident",
        "bare-cordon",
        "not-ready",
        "foreign-taint",
        "annotation-only",
    ],
)
def test_only_the_products_own_isolation_is_excused(node, reasons) -> None:
    isolation = ISOLATION.product_isolation(node)

    if reasons is None:
        assert isolation is None, "a hard failure stays a hard failure"
    else:
        assert isolation["reasons"] == reasons
        assert isolation["node_id"] == "n"
        assert isolation["ready"] is True


def test_the_quarantine_record_names_the_incident_and_the_taint() -> None:
    isolation = ISOLATION.product_isolation(_quarantined("ip-10-0-41-164"))

    assert isolation == {
        "node_id": "ip-10-0-41-164",
        "reasons": ["quarantine-taint", "cordoned-by-incident"],
        "incident_id": INCIDENT,
        "fencing_token": "1",
        "quarantine_taint_value": "incident-abc",
        "unschedulable": True,
        "ready": True,
    }


def test_the_scan_reads_live_nodes_and_keeps_only_target_nodes() -> None:
    release = ResourceRelease()
    items = release.documents[("gpu-a", "nodes", "")]["items"]
    items.append(_quarantined("node-b"))
    items.append(_quarantined("node-outside"))
    target = release.config.clusters[0]

    excluded = ISOLATION.product_isolated_nodes(release, target, ("node-a", "node-b"))

    assert [item["node_id"] for item in excluded] == ["node-b"]
    assert excluded[0]["incident_id"] == INCIDENT
    assert release.reads[-1] == ("gpu-a", "nodes", ""), "a fresh read, not a snapshot"


def _roll(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str = "upgrade",
    isolated: tuple[str, ...] = ("node-c",),
    nodes: tuple[str, ...] = NODES,
    real_fleet_record: bool = False,
    real_finalize: bool = False,
) -> tuple[Any, dict[str, Any], Any]:
    """Run ``roll_node_runtime`` with every seam stubbed; return what it handed on.

    ``real_fleet_record`` keeps the fleet record creation and records the
    control-plane fleet commands instead; ``real_finalize`` keeps the steady-state
    finalisation and records the Agent wait.
    """

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    seen: dict[str, Any] = {"deploys": [], "scans": 0}
    monkeypatch.setattr(release, "_target_node_names", lambda _target: nodes)

    def scan(_release: Any, _target: Any, names: tuple[str, ...]) -> list[dict]:
        seen["scans"] += 1
        return [
            ISOLATION.product_isolation(_quarantined(name))
            for name in names
            if name in isolated
        ]

    monkeypatch.setattr(RUNTIME, "product_isolated_nodes", scan)
    monkeypatch.setattr(
        RUNTIME,
        "target_node_failure_domains",
        lambda _release, _target, names: {name: "zone-a" for name in names},
    )
    monkeypatch.setattr(
        RUNTIME, "ensure_pre_node_mutation_barrier", lambda *_args, **_kwargs: None
    )

    def deploy_reconciler(_target: Any, **kwargs: Any) -> tuple[str, str]:
        seen["deploys"].append(kwargs)
        return IDENTITY

    monkeypatch.setattr(release, "_deploy_reconciler", deploy_reconciler)
    monkeypatch.setattr(
        RUNTIME, "finish_legacy_node_runtime_rollback", lambda *_a, **_k: None
    )

    def create(_release: Any, _target: Any, candidate: Any, policy: Any, *_a, **_k):
        seen["candidate"] = candidate
        seen["policy"] = policy
        return "dep-1", {"status": "PLANNED"}

    if real_fleet_record:
        seen["fleet_ids"] = []
        seen["fleet_commands"] = []
        monkeypatch.setattr(
            release,
            "_fleet_deployment_id",
            lambda _target, **kwargs: seen["fleet_ids"].append(kwargs) or "dep-1",
        )
        monkeypatch.setattr(
            release,
            "_fleet_command",
            lambda op, body: seen["fleet_commands"].append((op, body))
            or {"status": "SUCCEEDED"},
        )
        monkeypatch.setattr(
            RUNTIME, "terminalize_stranded_cluster_rollouts", lambda *_a, **_k: ()
        )
        monkeypatch.setattr(RUNTIME, "apply_fleet_request_identity", lambda *_a: None)
    else:
        monkeypatch.setattr(RUNTIME, "_create_fleet_rollout", create)
    monkeypatch.setattr(
        RUNTIME, "live_agent_node_names", lambda _release, _target, names: names
    )
    monkeypatch.setattr(
        RUNTIME,
        "run_fleet_waves",
        lambda _release, _target, context, _deployment: seen.__setitem__(
            "context", context
        ),
    )

    def finalize(_release: Any, _target: Any, candidate: Any, *_a, **_k):
        seen["finalize_candidate"] = candidate
        return IDENTITY

    if real_finalize:
        monkeypatch.setattr(
            release,
            "_wait_agents",
            lambda _target, _sha, **kwargs: seen.__setitem__("wait", kwargs),
        )
    else:
        monkeypatch.setattr(RUNTIME, "_finalize_node_runtime", finalize)
    outcome: Any = None
    try:
        outcome = RUNTIME.roll_node_runtime(
            release,
            target,
            phase=phase,
            wheel_cm=release.executor_wheel_cm,
            bundle_cm=release.bundle_cm,
            artifact_sha=release.node_wheel_sha,
            config_digest=config.agent_config_digest,
            candidate_preflight_completed=True,
        )
    except ReleaseError as exc:
        outcome = exc
    return release, seen, outcome


def test_an_upgrade_rolls_the_other_nodes_and_ends_partial(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release, seen, outcome = _roll(tmp_path, monkeypatch)

    assert isinstance(outcome, ISOLATION.ProductIsolatedNodesError), outcome
    assert isinstance(outcome, ORCHESTRATION.ClusterLocalReleaseError), (
        "cluster-local: the orchestrator pauses this cluster, no rollback"
    )
    assert "node-c (" + INCIDENT + ")" in str(outcome)
    assert "submit-remediation --disposition restore" in str(outcome)
    candidate = seen["candidate"]
    assert candidate.node_names == ("node-a", "node-b"), "the wave rolls the rest"
    assert [item["node_id"] for item in candidate.excluded_nodes] == ["node-c"]
    assert seen["context"].node_names == ("node-a", "node-b"), (
        "the wave safety gate protects the included nodes only"
    )
    assert seen["finalize_candidate"] is candidate, (
        "the included nodes converged before the PARTIAL verdict"
    )
    recorded = release.state[ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY]["gpu-a"]
    assert recorded["release_id"] == release.release_id
    assert recorded["nodes"][0]["incident_id"] == INCIDENT
    assert ISOLATION.recorded_product_isolated_nodes(release.state, "gpu-a") == (
        "node-c",
    )
    narration = capsys.readouterr().err
    assert "fleet-node-excluded" in narration and "nodes=node-c" in narration


@pytest.mark.parametrize("phase", ["join", "bootstrap", "rollback"])
def test_other_phases_keep_refusing_isolated_nodes(
    tmp_path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    release, seen, outcome = _roll(tmp_path, monkeypatch, phase=phase)

    assert seen["scans"] == 0, "no exclusion outside the upgrade wave"
    assert seen["candidate"].node_names == NODES
    assert seen["candidate"].excluded_nodes == ()
    assert outcome == IDENTITY
    assert ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY not in release.state


def test_a_cluster_with_nothing_left_to_roll_pauses_before_any_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, seen, outcome = _roll(tmp_path, monkeypatch, isolated=NODES)

    assert isinstance(outcome, ISOLATION.ProductIsolatedNodesError), outcome
    assert seen["deploys"] == [] and "candidate" not in seen
    assert ISOLATION.recorded_product_isolated_nodes(release.state, "gpu-a") == NODES


def test_an_upgrade_without_isolation_is_unchanged(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, seen, outcome = _roll(tmp_path, monkeypatch, isolated=())

    assert outcome == IDENTITY
    assert seen["candidate"].node_names == NODES
    assert seen["candidate"].excluded_nodes == ()
    assert ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY not in release.state


def test_a_partial_wave_opens_its_own_fleet_deployment_record() -> None:
    def fleet_id(**kwargs: Any) -> str:
        return FLEET.fleet_deployment_id(
            SimpleNamespace(
                release_id="cand", state={"fleet_rollout_transaction": "1"}
            ),
            SimpleNamespace(cluster_id="gpu-a"),
            phase="upgrade",
            artifact_sha="a" * 64,
            bundle_sha="b" * 64,
            template_sha="c" * 64,
            config_digest="d" * 64,
            runtime_profile_version="profile-v1",
            **kwargs,
        )

    whole = fleet_id()
    partial = fleet_id(node_subset=("node-b", "node-a"))

    assert partial != whole, (
        "the resume that rolls every node must not hit the partial record"
    )
    assert partial == fleet_id(node_subset=("node-a", "node-b")), "order-free"
    assert partial.startswith("release-upgrade-cand-"), partial


def _candidate(**overrides: Any) -> PREFLIGHT.NodeMutationPreflight:
    fields: dict[str, Any] = {
        "phase": "upgrade",
        "node_names": ("node-a", "node-b"),
        "wheel_cm": "wheel",
        "bundle_cm": "bundle",
        "artifact_sha": "a" * 64,
        "config_digest": "d" * 64,
        "runtime_profile_version": "profile-v1",
        "executor_wheel_filename": None,
        "node_compatibility_digest": "c" * 64,
        "bundle_sha256": IDENTITY[0],
        "template_sha256": IDENTITY[1],
        "template_config_map": None,
        "max_unavailable": 1,
        "runtime_image": None,
        "node_installer_image": None,
    }
    fields.update(overrides)
    return PREFLIGHT.NodeMutationPreflight(**fields)


@pytest.mark.parametrize("isolated", [(), ("node-c",)])
def test_the_steady_state_reconciler_and_the_wait_stay_off_excluded_nodes(
    tmp_path, monkeypatch: pytest.MonkeyPatch, isolated: tuple[str, ...]
) -> None:
    _release, seen, outcome = _roll(
        tmp_path, monkeypatch, isolated=isolated, real_finalize=True
    )

    steady = seen["deploys"][-1]
    if isolated:
        assert isinstance(outcome, ISOLATION.ProductIsolatedNodesError), outcome
        assert steady["allowed_node_names"] == ("node-a", "node-b"), (
            "no installer Job may cycle on the quarantined node until the resume"
        )
        assert seen["wait"]["node_names"] == ("node-a", "node-b")
    else:
        assert outcome == IDENTITY
        assert steady["allowed_node_names"] is None
        assert "node_names" not in seen["wait"], "the steady-state call is unchanged"


@pytest.mark.parametrize("isolated", [(), ("node-c",)])
def test_the_fleet_record_of_a_partial_wave_names_its_node_subset(
    tmp_path, monkeypatch: pytest.MonkeyPatch, isolated: tuple[str, ...]
) -> None:
    _release, seen, _outcome = _roll(
        tmp_path, monkeypatch, isolated=isolated, real_fleet_record=True
    )

    create = [body for op, body in seen["fleet_commands"] if op == "create"]
    included = tuple(name for name in NODES if name not in isolated)
    assert [body["request"]["node_ids"] for body in create] == [list(included)], (
        "the fleet record covers exactly the included nodes"
    )
    if isolated:
        assert seen["fleet_ids"][-1]["node_subset"] == included
    else:
        assert "node_subset" not in seen["fleet_ids"][-1]


def test_the_orchestrator_pauses_the_cluster_and_keeps_the_rest() -> None:
    targets = tuple(SimpleNamespace(cluster_id=name) for name in ("gpu-a", "gpu-b"))
    saves: list[Any] = []

    def upgrade(target: Any, *_args: Any, **_kwargs: Any) -> None:
        if target.cluster_id == "gpu-b":
            raise ISOLATION.ProductIsolatedNodesError(
                "gpu-b", [ISOLATION.product_isolation(_quarantined("node-c"))]
            )

    release = phase_release([], saves, clusters=targets, _upgrade_gpu_target=upgrade)

    with pytest.raises(ORCHESTRATION.PartialClusterRolloutError) as failure:
        ORCHESTRATION.upgrade_gpu_clusters(
            release,
            diff=diff_from_changed({"node_wheel"}),
            plan=ReleaseExecutionPlan((Component.AGENT,)),
            previous={},
            completed_phases=set(),
            completed_clusters=set(),
            registry_staged=False,
        )

    assert saves[-1]["phase"] == "data-plane-paused"
    assert saves[-1]["failure_scope"] == "cluster-local"
    assert saves[-1]["release_lifecycle"] == "PAUSED"
    assert saves[-1]["completed_cluster_ids"] == ["gpu-a"], "gpu-a keeps its roll"
    assert saves[-1]["failed_cluster_ids"] == ["gpu-b"]
    assert INCIDENT in saves[-1]["cluster_attempts"]["gpu-b"]["details"]["error"]
    assert "the release ends PARTIAL" in str(failure.value)


def _paused_checkpoint(release: ResourceRelease) -> dict[str, Any]:
    return {
        "release_id": release.release_id,
        "cluster_ids": ["gpu-a"],
        "cluster_registry_digest": release.cluster_registry_digest,
        "cluster_attempts": {"gpu-a": {"state": "FAILED"}},
        "completed_phases": [],
        "component_progress": {
            "clusters": {"gpu-a": {Component.AGENT.value: {"status": "FAILED"}}}
        },
        ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY: {
            "gpu-a": {
                "release_id": release.release_id,
                "nodes": [{"node_id": "node-b", "incident_id": INCIDENT}],
            }
        },
    }


@pytest.mark.parametrize("fault", [None, "whole-record", "stale-release"])
def test_resume_validation_binds_the_partial_record_to_the_included_nodes(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    release = ResourceRelease()
    release.node_names = ("node-a", "node-b")
    previous = previous_snapshot(release)
    previous["agent_identities"]["gpu-a"]["node_ids"] = ["node-a", "node-b"]
    loaded = _paused_checkpoint(release)
    if fault == "stale-release":
        loaded[ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY]["gpu-a"]["release_id"] = "x"
    ids: list[dict[str, Any]] = []
    monkeypatch.setattr(
        release,
        "_fleet_deployment_id",
        lambda _target, **kwargs: ids.append(kwargs) or "fleet-gpu-a",
    )
    response = mixed_probe_result(release)
    response["deployment"]["nodes"] = [
        {"node_id": name, "status": "READY"}
        for name in (("node-a", "node-b") if fault == "whole-record" else ("node-a",))
    ]
    probes: list[dict[str, Any]] = []

    def probe(_release: Any, **kwargs: Any) -> str:
        probes.append(json.loads(kwargs["input_text"]))
        return json.dumps(response)

    monkeypatch.setattr(RESUME, "exec_cpu_ingress_probe", probe)
    monkeypatch.setattr(RESUME, "validate_gpu_rollback_target", lambda *_a, **_k: None)

    def validate() -> None:
        RESUME.validate_resume_checkpoint(
            release,
            loaded=loaded,
            previous=previous,
            plan=ReleaseExecutionPlan((Component.AGENT,)),
        )

    if fault is None:
        validate()
        assert ids[-1]["node_subset"] == ("node-a",), "the partial wave's record"
        assert probes[-1]["node_ids"] == ["node-a", "node-b"], (
            "the excluded node's Agent must still be inside the window to resume"
        )
    else:
        with pytest.raises(ReleaseError, match="contract drifted"):
            validate()
        if fault == "stale-release":
            assert "node_subset" not in ids[-1], (
                "another release's exclusions do not describe this transaction"
            )


def test_record_keeps_this_release_and_clears_a_cluster_with_nothing_excluded() -> None:
    release = SimpleNamespace(
        release_id="cand",
        state={
            ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY: {
                "gpu-a": {"release_id": "old", "nodes": [{"node_id": "x"}]},
                "gpu-b": {"release_id": "cand", "nodes": [{"node_id": "node-z"}]},
            }
        },
    )
    excluded = [ISOLATION.product_isolation(_quarantined("node-c"))]

    ISOLATION.record_product_isolated_nodes(
        release, SimpleNamespace(cluster_id="gpu-a"), excluded
    )
    recorded = release.state[ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY]
    assert set(recorded) == {"gpu-a", "gpu-b"}, "the stale release's entry is replaced"
    assert recorded["gpu-a"]["nodes"][0]["node_id"] == "node-c"

    ISOLATION.record_product_isolated_nodes(
        release, SimpleNamespace(cluster_id="gpu-a"), []
    )
    ISOLATION.record_product_isolated_nodes(
        release, SimpleNamespace(cluster_id="gpu-b"), []
    )
    assert ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY not in release.state


def test_operator_recovery_commands_restore_then_resume() -> None:
    entries = {
        "gpu-a": {
            "release_id": "cand",
            "nodes": [
                {"node_id": "node-c", "incident_id": INCIDENT},
                {"node_id": "node-d", "incident_id": None},
            ],
        }
    }

    commands = ISOLATION.operator_recovery_commands(entries)

    restore = (
        'gpu-fault-admin submit-remediation --state-dir "$STATE_DIR" '
        f"--incident-id {INCIDENT} --disposition restore --reference <CHG>"
    )
    assert commands[:2] == [f"{restore} --plan", restore]
    assert commands[2].startswith(
        "# gpu-a/node-d: isolated without an incident annotation"
    ), commands
    assert commands[-1] == ISOLATION.RESUME_COMMAND
    assert ISOLATION.operator_recovery_commands({}) == []


def test_status_names_the_excluded_nodes_and_the_way_out() -> None:
    state = {
        "release_id": "cand",
        "phase": "partial-convergence",
        ISOLATION.PRODUCT_ISOLATED_NODES_STATE_KEY: {
            "gpu-eks-cluster": {
                "release_id": "cand",
                "nodes": [{"node_id": "ip-10-0-41-164", "incident_id": INCIDENT}],
            },
            "stale": {"release_id": "old", "nodes": [{"node_id": "x"}]},
        },
    }

    summary = COMMANDS.product_isolation_summary(state)

    assert set(summary["product_isolated_nodes"]) == {"gpu-eks-cluster"}
    assert summary["operator_recovery"][-1] == ISOLATION.RESUME_COMMAND
    assert COMMANDS.product_isolation_summary({**state, "phase": "complete"}) == {}
    report = {
        "healthy": True,
        "live_release": {
            "release_id": "cand",
            "phase": "partial-convergence",
            "transaction_committed": False,
            **summary,
        },
        "next_deploy": {"kind": "FULL", "resume": True, "action": "upgrade"},
        "health": {"summary": {"PASS": 1}},
    }

    lines = COMMANDS.status_header_lines(report, json_destination="stdout")

    assert len(lines) == 6
    assert lines[2] == "next deploy: FULL (resume upgrade)"
    assert lines[3] == (
        "partial: 1 node(s) excluded by the product's own isolation: "
        f"gpu-eks-cluster/ip-10-0-41-164 ({INCIDENT}); restore the incident(s), "
        "then rerun deploy -- commands in live_release.operator_recovery"
    )
    plain = copy.deepcopy(report)
    del plain["live_release"]["product_isolated_nodes"]
    assert len(COMMANDS.status_header_lines(plain, json_destination="stdout")) == 5


def test_the_candidate_default_has_no_exclusions() -> None:
    assert _candidate().excluded_nodes == ()
    assert replace(_candidate(), node_names=("node-a",)).excluded_nodes == ()
