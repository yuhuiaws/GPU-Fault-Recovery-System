"""The per-cluster evidence behind a membership verb's scope, and its last steps.

* The ``workflow_safety`` and ``rollout_wave_safety`` probes attribute every
  blocker, open command and destructive workflow to its incident's cluster
  (``""`` when the incident cannot be read), so a join's preflight and its
  waves can gate on the joining cluster alone.
* ``sync-state`` -- the step before a join's activation, where a refusal
  rolled the join back -- judges the target's Agent set only when scoped;
  another cluster's mismatch (a node a spot reclaim just took) is logged.
* ``drain-cluster`` -- a removal's first engine step -- waits for the drained
  cluster's remote commands only.

Only modules that existed before the change are imported.
"""

from __future__ import annotations

import io
import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault_release import regional_release_agent_set as AGENT_SET
from gpu_fault_release import regional_release_online_registry as REGISTRY
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import regional_release_store_preflight as STORE
from gpu_fault_release import regional_verify_scope as SCOPE
from gpu_fault_release.regional_release_config import ReleaseError
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.regional._cov95_release_support import ResourceRelease

PROBES = Path(__file__).resolve().parents[2] / "deploy/control-plane/regional/probes"
NODE = "ip-10-0-38-176.us-west-2.compute.internal"


def pending(name: str, cluster_id: str) -> tuple[Any, Any]:
    """A destructive record never handed out, and its incident on ``cluster_id``."""

    workflow = workflow_request(
        f"workflow-{name}",
        f"incident-{name}",
        status=WorkflowStatus.PENDING,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        official_steps=[workflow_step(WorkflowOperation.RESET_GPU, node_ids=[NODE])],
    )
    incident = fault_incident(
        f"incident-{name}",
        f"event-{name}",
        cluster_id=cluster_id,
        node_ids=[NODE],
        state=IncidentState.ESCALATED,
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
    )
    return workflow, incident


def run_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    pairs: list[tuple[Any, Any]],
    *,
    unreadable: frozenset[str] = frozenset(),
    remote: dict[str, Any] | None = None,
) -> dict[str, Any]:
    incidents = {incident.incident_id: incident for _workflow, incident in pairs}

    def get_incident(incident_id: str) -> Any:
        if incident_id in unreadable:
            raise RuntimeError("the Store read timed out")
        return incidents[incident_id]

    store = SimpleNamespace(
        remote_command_stats=lambda: remote or {"by_status": {}},
        list_workflows=lambda **_kwargs: [workflow for workflow, _ in pairs],
        get_incident=get_incident,
        get_workflow=lambda request_id: (_ for _ in ()).throw(KeyError(request_id)),
        list_remote_commands=lambda **_kwargs: [],
        list_agents=lambda _cluster_id: [],
    )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        staticmethod(lambda: SimpleNamespace(store=store)),
    )
    request = {"cluster_id": "gpu-b", "nodes": [], "wave": []}
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({**request, "minimum_lease_remaining_seconds": 30})),
    )
    runpy.run_path(str(PROBES / f"{name}.py"))
    return json.loads(capsys.readouterr().out)


def test_the_workflow_safety_probe_attributes_every_blocker_to_its_cluster(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = [pending("a1", "gpu-a"), pending("a2", "gpu-a"), pending("b1", "gpu-b")]
    pairs.append(pending("lost", "gpu-a"))

    report = run_probe(
        monkeypatch,
        capsys,
        "workflow_safety",
        pairs,
        unreadable=frozenset({"incident-lost"}),
    )

    assert report["blocker_count"] == 4, report
    assert report["blocker_count_by_cluster"] == {"gpu-a": 2, "gpu-b": 1, "": 1}
    assert report["blockers_by_cluster"] == {
        "gpu-a": ["workflow-a1", "workflow-a2"],
        "gpu-b": ["workflow-b1"],
        "": ["workflow-lost"],
    }, "an unreadable incident is filed under no cluster, which gates every scope"


def test_the_wave_probe_attributes_open_commands_and_workflows_to_clusters(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = {
        "by_status": {"PENDING": 3, "LEASED": 1, "SUCCEEDED": 9},
        "by_cluster_status": {
            "gpu-a": {"PENDING": 3, "SUCCEEDED": 4},
            "gpu-b": {"LEASED": 1, "SUCCEEDED": 5},
            "gpu-c": {"SUCCEEDED": 1},
        },
    }

    report = run_probe(
        monkeypatch,
        capsys,
        "rollout_wave_safety",
        [pending("a1", "gpu-a"), pending("b1", "gpu-b")],
        remote=remote,
    )

    assert report["open_remote"] == {"PENDING": 3, "LEASED": 1, "WAITING": 0}
    assert report["open_remote_by_cluster"] == {
        "gpu-a": {"PENDING": 3},
        "gpu-b": {"LEASED": 1},
        "gpu-c": {},
    }
    assert report["destructive_workflow_count_by_cluster"] == {"gpu-a": 1, "gpu-b": 1}
    assert report["destructive_workflows_by_cluster"] == {
        "gpu-a": ["workflow-a1"],
        "gpu-b": ["workflow-b1"],
    }
    young = report["never_dispatched_young_destructive_workflows"]
    assert sorted(item["cluster_id"] for item in young) == ["gpu-a", "gpu-b"], young


def test_the_wave_probe_leaves_the_command_breakdown_out_without_the_stores(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Store older than ``by_cluster_status``: the engine gates on the total."""

    report = run_probe(
        monkeypatch,
        capsys,
        "rollout_wave_safety",
        [],
        remote={"by_status": {"PENDING": 1}},
    )

    assert "open_remote_by_cluster" not in report
    assert report["open_remote"]["PENDING"] == 1


# -- sync-state's Agent set ----------------------------------------------------------


class TwoClusters(ResourceRelease):
    """Two clusters' nodes; the identity probe answers ``records``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__(("gpu-a", "gpu-b"))
        self.nodes = {"gpu-a": ("node-a1", "node-a2"), "gpu-b": ("node-b1",)}
        self.records = records

    def _target_node_names(self, target: Any) -> tuple[str, ...]:
        return self.nodes[target.cluster_id]


def agent(cluster_id: str, node_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "node_id": node_id,
        "agent_protocol_version": 4,
        "agent_version": "0.10.0",
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "a" * 64,
        "installer_bundle_sha256": "b" * 64,
        "installer_template_sha256": "c" * 64,
        "policy_version": "policy-v1",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "d" * 64,
        "node_action_key_version": 1,
        **fields,
    }


def scoped(release: Any, verb: str, *targets: str) -> Any:
    release.verify_scope = SCOPE.VerifyScope(verb, targets)
    return release


@pytest.fixture
def no_reread(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The identity probe answers the release's ``records``; the re-read wait
    is recorded, the lease facts left out."""

    slept: list[float] = []
    monkeypatch.setattr(AGENT_SET, "SLEEP", slept.append)
    monkeypatch.setattr(AGENT_SET, "_lease_facts", lambda *_a: {})
    monkeypatch.setattr(
        STATE,
        "exec_cpu_ingress_probe",
        lambda release, **_k: json.dumps(release.records),
    )
    return slept


def test_a_scoped_sync_state_logs_another_clusters_agent_set_and_commits(
    no_reread: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    """Before: ``gpu-a active Agent set does not match HyperPod nodes`` -- a node
    a spot reclaim just took still listed, its Agent's lease lapsed -- refused
    the join's sync-state, before activation, and the join was rolled back."""

    release = scoped(
        TwoClusters([agent("gpu-a", "node-a1"), agent("gpu-b", "node-b1")]),
        SCOPE.JOIN_VERB,
        "gpu-b",
    )

    identities = AGENT_SET.capture_agent_identities_matching_nodes(release)

    assert identities["gpu-b"]["node_ids"] == ["node-b1"]
    assert no_reread == [], "another cluster's set is not waited for"
    lines = [
        line for line in capsys.readouterr().err.splitlines() if "not-gating" in line
    ]
    assert len(lines) == 1, lines
    assert "cluster=gpu-a check=agent_set" in lines[0]
    assert "nodes without an active Agent=['node-a2']" in lines[0]


def test_a_scoped_sync_state_logs_a_cluster_left_without_agents(
    no_reread: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    release = scoped(
        TwoClusters([agent("gpu-b", "node-b1")]), SCOPE.REMOVE_VERB, "gpu-removed"
    )
    release.nodes["gpu-b"] = ("node-b1",)

    identities = AGENT_SET.capture_agent_identities_matching_nodes(release)

    assert list(identities) == ["gpu-b"], "the cluster without Agents is left out"
    err = capsys.readouterr().err
    assert "cluster=gpu-a check=agent_set" in err
    assert "has no active Agent identity to capture" in err


def test_the_targets_agent_set_still_refuses_the_sync_state(
    no_reread: list[float],
) -> None:
    release = scoped(
        TwoClusters(
            [agent("gpu-a", "node-a1"), agent("gpu-a", "node-a2"), agent("gpu-b", "x")]
        ),
        SCOPE.JOIN_VERB,
        "gpu-b",
    )

    with pytest.raises(ReleaseError, match="gpu-b active Agent set does not match"):
        AGENT_SET.capture_agent_identities_matching_nodes(release)
    assert no_reread == [AGENT_SET.AGENT_SET_RECHECK_SECONDS], "re-read once"


@pytest.mark.parametrize(
    "records,problem",
    [
        ([agent("gpu-b", "node-b1")], "gpu-a has no active Agent identity to capture"),
        (
            [
                agent("gpu-a", "node-a1"),
                agent("gpu-a", "node-a2", agent_version="0.9.0"),
                agent("gpu-b", "node-b1"),
            ],
            "gpu-a active Agent identities differ",
        ),
        (
            [agent("gpu-a", "node-a1"), agent("gpu-b", "node-b1")],
            "gpu-a active Agent set does not match",
        ),
    ],
    ids=["no-agents", "identities-differ", "set-mismatch"],
)
def test_an_unscoped_capture_still_refuses_every_cluster(
    no_reread: list[float], records: list[dict[str, Any]], problem: str
) -> None:
    """A deploy's capture is its rollback snapshot: every cluster gates."""

    with pytest.raises(ReleaseError, match=problem):
        AGENT_SET.capture_agent_identities_matching_nodes(TwoClusters(records))


def test_the_state_module_capture_groups_through_the_agent_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = TwoClusters([])
    records = [agent("gpu-a", "node-a1"), agent("gpu-b", "node-b1")]
    monkeypatch.setattr(
        STATE, "exec_cpu_ingress_probe", lambda *_a, **_k: json.dumps(records)
    )

    identities = STATE.capture_agent_identities(release)

    assert {key: value["node_ids"] for key, value in identities.items()} == {
        "gpu-a": ["node-a1"],
        "gpu-b": ["node-b1"],
    }


# -- drain-cluster -------------------------------------------------------------------


class Drained(ResourceRelease):
    """A removal's ``drain-cluster``: the real idle check, a recorded publish."""

    _remote_commands_are_idle = STORE.remote_commands_are_idle

    def __init__(self) -> None:
        super().__init__(("gpu-a", "gpu-b"))
        self.published: list[dict[str, str]] = []

    def _target(self, cluster_id: str) -> Any:
        return next(
            target for target in self.config.clusters if target.cluster_id == cluster_id
        )


def drain(
    monkeypatch: pytest.MonkeyPatch, cluster_ids: list[str], **open_by_cluster: Any
) -> Drained:
    release = Drained()
    by_status = {"PENDING": 0, "LEASED": 0, "WAITING": 0}
    for cells in open_by_cluster.values():
        for status, count in cells.items():
            by_status[status] += count
    stats = {"by_status": by_status, "by_cluster_status": open_by_cluster}
    monkeypatch.setattr(STORE, "cpu_ingress_pod_if_running", lambda _r: "api-ha-0")
    monkeypatch.setattr(
        STORE,
        "exec_cpu_ingress_probe",
        lambda _release, *, script, **_k: (
            json.dumps(stats)
            if "remote_command_stats()" in script and "print(bad" not in script
            else ("" if not any(by_status.values()) else str(by_status))
        ),
    )
    monkeypatch.setattr(
        REGISTRY,
        "publish_current_registry",
        lambda release, **kwargs: release.published.append(
            kwargs["lifecycle_overrides"]
        ),
    )
    REGISTRY.drain_registry_clusters(release, cluster_ids)
    return release


def test_a_removal_drain_does_not_wait_for_another_clusters_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before: ``remote commands are PENDING/LEASED/WAITING`` -- gpu-b's -- held
    gpu-a's removal at its drain until they expired."""

    release = drain(monkeypatch, ["gpu-a"], **{"gpu-b": {"PENDING": 2}})

    assert release.published == [{"gpu-a": "DRAINING"}]
    assert release.verify_scope is None, "the drain's scope is undone afterwards"
    assert "cluster=gpu-b check=remote_commands_idle" in capsys.readouterr().err


@pytest.mark.parametrize(
    "cluster_ids,open_by_cluster",
    [
        (["gpu-a"], {"gpu-a": {"WAITING": 1}}),
        (["gpu-a"], {"gpu-gone": {"PENDING": 1}}),
        (["gpu-a", "gpu-b"], {"gpu-b": {"LEASED": 1}}),
    ],
    ids=["drained-clusters-own", "unmanaged-cluster", "uninstall-drains-every-cluster"],
)
def test_a_drain_still_waits_for_what_it_could_strand(
    monkeypatch: pytest.MonkeyPatch,
    cluster_ids: list[str],
    open_by_cluster: dict[str, dict[str, int]],
) -> None:
    with pytest.raises(
        ReleaseError, match="remote commands are PENDING/LEASED/WAITING"
    ):
        drain(monkeypatch, cluster_ids, **open_by_cluster)


# -- the scope's count helpers ---------------------------------------------------------


def joining(release: Any) -> Any:
    return scoped(release, SCOPE.JOIN_VERB, "gpu-b")


@pytest.mark.parametrize(
    "by_cluster",
    [None, {"gpu-a": "two"}, {"gpu-a": -1, "gpu-b": 3}, {"gpu-a": 1}],
    ids=["absent", "unreadable", "negative", "short-of-the-total"],
)
def test_a_count_breakdown_that_cannot_be_trusted_gates_in_full(
    by_cluster: Any,
) -> None:
    container: dict[str, Any] = {}

    gating = SCOPE.gating_count(
        joining(ResourceRelease(("gpu-a", "gpu-b"))),
        container,
        check="workflow_safety",
        total=2,
        by_cluster=by_cluster,
        finding=lambda cluster, count: f"{count} of {cluster}",
    )

    assert (gating, container) == (2, {})


def test_another_clusters_zero_count_files_nothing() -> None:
    container: dict[str, Any] = {}

    gating = SCOPE.gating_count(
        joining(ResourceRelease(("gpu-a", "gpu-b"))),
        container,
        check="workflow_safety",
        total=1,
        by_cluster={"gpu-a": 0, "gpu-b": 1},
        finding=lambda cluster, count: f"{count} of {cluster}",
    )

    assert (gating, container) == (1, {})


@pytest.mark.parametrize(
    "by_cluster_status",
    [
        None,
        {"gpu-a": ["PENDING"]},
        {"gpu-a": {"PENDING": "many"}},
        {"gpu-a": {"PENDING": 1}},
    ],
    ids=["absent", "not-a-mapping", "unreadable", "short-of-the-fleet"],
)
def test_an_open_command_breakdown_that_cannot_be_trusted_gates_in_full(
    by_cluster_status: Any,
) -> None:
    container: dict[str, Any] = {}

    gating = SCOPE.gating_open_commands(
        joining(ResourceRelease(("gpu-a", "gpu-b"))),
        container,
        check="remote_commands_idle",
        open_by_status={"PENDING": 2, "LEASED": 0},
        by_cluster_status=by_cluster_status,
    )

    assert (gating, container) == ({"PENDING": 2, "LEASED": 0}, {})


def test_a_finding_is_narrated_once_however_often_a_gate_polls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    release = joining(ResourceRelease(("gpu-a", "gpu-b")))
    for _poll in range(3):
        SCOPE.file_not_gating(release, {}, "fleet_wave_safety", "gpu-a", "same")
    SCOPE.narrate_not_gating(SimpleNamespace(), "x", "gpu-a", "unscoped: silent")

    lines = [line for line in capsys.readouterr().err.splitlines() if line]
    assert len(lines) == 1, lines
    assert lines[0].startswith("not-gating "), lines
    assert 'finding="same"' in lines[0], lines


def test_another_clusters_unreadable_node_list_is_logged(
    no_reread: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    class Unreadable(TwoClusters):
        def _target_node_names(self, target: Any) -> tuple[str, ...]:
            if target.cluster_id == "gpu-a":
                raise ReleaseError("gpu-a API server: connection refused")
            return super()._target_node_names(target)

    release = scoped(
        Unreadable(
            [
                agent("gpu-a", "node-a1"),
                agent("gpu-a", "node-a2"),
                agent("gpu-b", "node-b1"),
            ]
        ),
        SCOPE.JOIN_VERB,
        "gpu-b",
    )

    identities = AGENT_SET.capture_agent_identities_matching_nodes(release)

    assert sorted(identities) == ["gpu-a", "gpu-b"]
    assert "node list unreadable: gpu-a API server" in capsys.readouterr().err
