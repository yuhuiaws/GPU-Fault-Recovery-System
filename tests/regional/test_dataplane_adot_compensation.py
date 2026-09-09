"""Rollback compensates the per-GPU-cluster ADOT collector from a snapshot (F10 fix 1).

The first wiring re-rendered the candidate's collector manifest with the previous
image on every cluster in the candidate config, so a rollback kept the
candidate's keep filters and relabels, created collectors on clusters the
candidate never touched, and left a collector running on a cluster whose role
the candidate had just added. These tests pin the compensation: the previous
release's live objects are captured per cluster before the OBSERVABILITY node
runs, and rollback restores each cluster from ITS snapshot -- objects put back
where a collector was live, the candidate's objects deleted where none was. The
capture iterates the candidate config, so a cluster the candidate ADDED is in
the snapshot too (every object absent -> ``removed``); ``scaled-to-zero`` is
reached only by a cluster added to site.yaml between the transaction's opening
and its rollback, and a cluster removed from site.yaml in that window refuses
the rollback. The previous-image path survives only for a state captured before
the snapshot existed, and the rollback record says which path ran.

Fix round 2: the data-plane half is its own rollback phase, after the CPU
control plane is back; one cluster failing does not stop the others or the
expected-rules restore, and the phase then fails naming every failed cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_dataplane_observability as DATAPLANE
from gpu_fault_release import regional_gpu_bootstrap as BOOTSTRAP
from gpu_fault_release import regional_manifest_snapshot as SNAPSHOT
from gpu_fault_release import regional_observability_rollback as CONTROL_PLANE
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_progress as PROGRESS
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_rendering import DATAPLANE_ADOT_DEPLOYMENT
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-adot-writer"
PREVIOUS_ADOT_IMAGE = "adot@sha256:" + "0" * 64
NAMESPACE = "gpu-fault-system"


def _target(cluster_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        cluster_id=cluster_id,
        context=f"{cluster_id}-context",
        region="us-east-1",
        adot_irsa_role_arn=ROLE_ARN,
    )


def _gpu(target: Any, *arguments: str) -> list[str]:
    return ["kubectl", "--context", target.context, *arguments]


def _declared() -> tuple[dict[str, str], ...]:
    return DATAPLANE.declared_dataplane_adot_objects()


def _live_object(kind: str, name: str, marker: str = "previous") -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1" if kind == "Deployment" else "v1",
        "kind": kind,
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "uid": "1a2b",
            "resourceVersion": "4711",
            "annotations": {
                "kubectl.kubernetes.io/last-applied-configuration": "{}",
                "gpu-fault.io/previous-release": "kept",
            },
        },
        "spec": {"marker": marker},
        "status": {"observedGeneration": 3},
    }


# --- capture --------------------------------------------------------------------


def test_the_capture_reads_each_clusters_collector_through_its_own_context() -> None:
    """The snapshot is per GPU cluster, read with that cluster's kubectl context.

    The control-plane capture reads through ``_cpu``; a data-plane capture that
    did the same would snapshot the wrong cluster and a rollback built on it
    would compensate nothing while reporting PASSED.
    """
    live = {
        (item["resource"], item["name"]): _live_object(item["kind"], item["name"])
        for item in _declared()
    }
    reads: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> str:
        reads.append(list(arguments))
        index = arguments.index("get")
        document = live.get((arguments[index + 1], arguments[index + 2]))
        return json.dumps(document) if document is not None else ""

    release = SimpleNamespace(
        config=SimpleNamespace(namespace=NAMESPACE),
        runner=SimpleNamespace(run=run),
        _cpu=lambda *args: ["kubectl", "--kubeconfig", "/secure/cpu", *args],
        _gpu=_gpu,
    )

    snapshot = DATAPLANE.capture_dataplane_adot_snapshot(release, _target("gpu-a"))

    assert reads, "nothing was read"
    for arguments in reads:
        assert arguments[:3] == ["kubectl", "--context", "gpu-a-context"], arguments
        assert "--ignore-not-found" in arguments
    assert snapshot["present"] is True
    assert snapshot["namespace"] == NAMESPACE
    assert snapshot["absent"] == []
    assert [item["kind"] for item in snapshot["objects"]] == [
        item["kind"] for item in _declared()
    ]
    for item in snapshot["objects"]:
        assert "status" not in item
        assert "uid" not in item["metadata"]


def test_a_cluster_without_a_collector_is_captured_as_absent_not_refused() -> None:
    """Unlike the control-plane collector, a missing data-plane collector is a
    legitimate previous state (no IRSA role yet): the snapshot records every
    declared object as absent so rollback can delete what the candidate adds."""
    release = SimpleNamespace(
        config=SimpleNamespace(namespace=NAMESPACE),
        runner=SimpleNamespace(run=lambda *_args, **_kwargs: ""),
        _gpu=_gpu,
    )

    snapshot = DATAPLANE.capture_dataplane_adot_snapshot(release, _target("gpu-b"))

    assert snapshot["present"] is False
    assert snapshot["objects"] == []
    assert [item["name"] for item in snapshot["absent"]] == [
        item["name"] for item in _declared()
    ]
    assert any(
        item["name"] == DATAPLANE_ADOT_DEPLOYMENT for item in snapshot["absent"]
    ), snapshot["absent"]


def test_the_observability_snapshot_carries_every_clusters_collector_state() -> None:
    """The release binds the combined capture, so ``previous.observability`` holds
    the control-plane blobs AND the per-cluster collector snapshots plus the
    expected-collector rule namespace, keyed by cluster id in config order."""
    assert (
        MODULE.RegionalRelease._capture_observability_snapshot
        is DATAPLANE.capture_observability_snapshot_with_dataplane
    )
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=(_target("gpu-a"), _target("gpu-b")),
            aws_region="us-east-1",
            health=SimpleNamespace(amp_workspace_id="ws-a"),
        ),
        _capture_dataplane_adot_snapshot=lambda target: (
            calls.append(target.cluster_id) or {"present": True, "objects": []}
        ),
        _capture_dataplane_expected_rules=lambda: {
            "present": True,
            "data_base64": "Z3JvdXBzOiBbXQo=",
        },
    )

    state = DATAPLANE.capture_dataplane_adot_state(release)

    assert sorted(calls) == ["gpu-a", "gpu-b"]
    assert list(state["clusters"]) == ["gpu-a", "gpu-b"]
    assert state["expected_rules"]["present"] is True


def test_expected_rules_capture_distinguishes_absent_from_unreadable() -> None:
    def probe_output(arguments: list[str], **_kwargs: Any) -> tuple[int, str, str]:
        assert arguments[:3] == ["aws", "amp", "describe-rule-groups-namespace"]
        assert DATAPLANE.DATAPLANE_EXPECTED_RULE_NAMESPACE in arguments
        return probe_output.answer  # type: ignore[attr-defined]

    release = SimpleNamespace(
        config=SimpleNamespace(
            aws_region="us-east-1", health=SimpleNamespace(amp_workspace_id="ws-a")
        ),
        runner=SimpleNamespace(probe_output=probe_output),
    )

    probe_output.answer = (  # type: ignore[attr-defined]
        0,
        json.dumps({"ruleGroupsNamespace": {"data": "Z3JvdXBzOiBbXQo="}}),
        "",
    )
    assert DATAPLANE.capture_dataplane_expected_rules(release) == {
        "present": True,
        "data_base64": "Z3JvdXBzOiBbXQo=",
    }

    probe_output.answer = (  # type: ignore[attr-defined]
        254,
        "",
        "An error occurred (ResourceNotFoundException) when calling ...",
    )
    assert DATAPLANE.capture_dataplane_expected_rules(release) == {
        "present": False,
        "data_base64": None,
    }

    probe_output.answer = (255, "", "Unable to locate credentials")  # type: ignore[attr-defined]
    with pytest.raises(ReleaseError, match="expected-collector rule namespace"):
        DATAPLANE.capture_dataplane_expected_rules(release)


# --- restore through the public rollback entry ---------------------------------


def _identity() -> dict[str, Any]:
    return {
        "agent_protocol_version": 3,
        "agent_version": "0.10.0",
        "artifact_sha256": "artifact",
        "compatibility_digest": "compatibility",
        "installer_bundle_sha256": None,
        "installer_template_sha256": None,
        "policy_version": "catalog",
        "runtime_profile_version": "profile-v1",
        "config_digest": "config",
        "node_action_key_version": 2,
        "node_ids": ["node-a"],
    }


PHASE = "dataplane_observability_restore"
PHASE_DONE = "rollback-dataplane-observability-restored"
NOT_FOUND = "An error occurred (ResourceNotFoundException) when calling ..."


def _describe_answer(
    status: str = "ACTIVE", *, reason: str | None = None
) -> tuple[int, str, str]:
    document: dict[str, Any] = {"statusCode": status}
    if reason is not None:
        document["statusReason"] = reason
    return (
        0,
        json.dumps(
            {"ruleGroupsNamespace": {"status": document, "data": "Z3JvdXBzOiBbXQo="}}
        ),
        "",
    )


class _Rollback:
    """Drive ``rollback_release`` with only the observability restores live.

    ``failing_status`` names the kube contexts whose ``rollout status`` fails;
    ``apply_output`` is what ``kubectl apply`` prints on every cluster (the
    default, an empty string, is read as "changed"); ``cpu`` adds the CPU
    finalize to the completed phases so the CPU restore phase is planned.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clusters: tuple[SimpleNamespace, ...],
        *,
        expected_rules_present: bool = True,
        failing_status: frozenset[str] = frozenset(),
        apply_output: str = "",
        cpu: bool = False,
    ) -> None:
        self.phase_calls: list[str] = []
        for name in (
            "_stage_rollback_controller",
            "_rollback_gpu_clusters",
            "_restore_rollback_cpu",
            "_verify_and_complete_rollback",
            "cleanup_candidate_rollout_state",
        ):
            monkeypatch.setattr(
                ORCHESTRATION,
                name,
                lambda *_args, _name=name, **_kwargs: self.phase_calls.append(_name),
            )
        self.clusters = clusters
        self.calls: list[list[str]] = []
        self.probes: list[list[str]] = []
        self.applied: list[tuple[str, Any]] = []
        self.candidate_renders: list[str] = []
        self.scaled: list[tuple[list[str], str, int]] = []
        self.saves: list[dict[str, Any]] = []
        self.control_plane_restored: list[Any] = []
        self.expected_rules_present = expected_rules_present

        def run(arguments: list[str], **_kwargs: Any) -> str:
            self.calls.append(list(arguments))
            # The fake AMP: a delete makes the namespace gone and a create makes
            # it exist, so the restore's post-write waits see what AMP would say.
            if arguments[:2] == ["aws", "amp"]:
                if arguments[2] == "delete-rule-groups-namespace":
                    self.expected_rules_present = False
                elif arguments[2] == "create-rule-groups-namespace":
                    self.expected_rules_present = True
            if arguments[0] == "kubectl" and "apply" in arguments:
                path = Path(arguments[arguments.index("-f") + 1])
                self.applied.append(
                    (arguments[2], json.loads(path.read_text(encoding="utf-8")))
                )
                return apply_output
            if (
                arguments[0] == "kubectl"
                and "status" in arguments
                and arguments[2] in failing_status
            ):
                raise ReleaseError(f"command failed (1): kubectl on {arguments[2]}")
            return ""

        def probe_output(arguments: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            self.probes.append(list(arguments))
            if self.expected_rules_present:
                return _describe_answer()
            return 254, "", NOT_FOUND

        completed = ["uploaded", "observability-ready"]
        nodes = ["observability", "verify"]
        if cpu:
            completed.append("cpu-finalized")
            nodes.insert(1, "cpu_finalize")

        def control_plane(snapshot: Any) -> None:
            self.control_plane_restored.append(snapshot)
            self.phase_calls.append("_restore_observability_snapshot")

        self.release = SimpleNamespace(
            state={
                "release_diff": {
                    "kind": str(DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY),
                    "changed": ["observability_adot"],
                },
                "execution_plan": {"nodes": nodes},
                "completed_phases": completed,
            },
            runtime_image="candidate-runtime",
            node_installer_image="registry.example/installer:candidate",
            config=SimpleNamespace(
                clusters=clusters,
                auto_rollback=True,
                namespace=NAMESPACE,
                aws_region="us-east-1",
                health=SimpleNamespace(amp_workspace_id="ws-a"),
            ),
            runner=SimpleNamespace(run=run, probe_output=probe_output),
            _gpu=_gpu,
            _refresh_aurora_credentials=lambda: None,
            _require_no_inflight_installs=lambda **_kwargs: None,
            _save_state=lambda phase, **updates: self.saves.append(
                {"phase": phase, **updates}
            ),
            _restore_observability_snapshot=control_plane,
            _apply_gpu_adot_collector=lambda target, *, image: (
                self.candidate_renders.append(f"{target.cluster_id}@{image}")
            ),
            _scale_if_present=lambda kubectl, deployment, replicas, **_kw: (
                self.scaled.append((list(kubectl), deployment, replicas))
            ),
        )

    def run(self, observability: dict[str, Any]) -> None:
        previous = {
            "metadata": {
                "required-agent-artifact-sha256": "artifact",
                "required-agent-config-digest": "config",
                "required-regional-executor-artifact-sha256": "executor",
            },
            "cpu_wheel": "wheel",
            "runtime_image": "registry.example/runtime@sha256:" + "e" * 64,
            "node_installer_image": "registry.example/installer@sha256:" + "f" * 64,
            "runtime_profile_version": "profile-v1",
            "adot_image": PREVIOUS_ADOT_IMAGE,
            "agent_identities": {t.cluster_id: _identity() for t in self.clusters},
            "observability": {"adot": [], **observability},
        }
        ORCHESTRATION.rollback_release(self.release, state=previous)

    def kubectl(self, context: str) -> list[list[str]]:
        return [
            arguments
            for arguments in self.calls
            if arguments[:3] == ["kubectl", "--context", context]
        ]

    def phase(self) -> dict[str, Any]:
        """The data-plane phase's durable timing entry from the last save."""
        return dict(self.saves[-1]["rollback_timing"]["phases"][PHASE])

    def record(self) -> dict[str, Any]:
        return dict(self.phase()["details"])

    def saved_phases(self) -> list[str]:
        return [str(save["phase"]) for save in self.saves]


def _snapshot_present(marker: str = "previous") -> dict[str, Any]:
    objects = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": DATAPLANE_ADOT_DEPLOYMENT, "namespace": NAMESPACE},
            "data": {"collector.yaml": marker},
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": DATAPLANE_ADOT_DEPLOYMENT, "namespace": NAMESPACE},
            "spec": {"replicas": 1},
        },
    ]
    return {"present": True, "namespace": NAMESPACE, "objects": objects, "absent": []}


def _snapshot_absent() -> dict[str, Any]:
    return {
        "present": False,
        "namespace": NAMESPACE,
        "objects": [],
        "absent": [
            {"resource": item["resource"], "name": item["name"]} for item in _declared()
        ],
    }


def test_rollback_restores_each_cluster_from_its_own_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(i) the SNAPSHOT objects are applied, never the candidate's rendering;
    (ii) a cluster with no collector before the release has the candidate's
    objects deleted, not a collector created (this is also what a cluster the
    candidate ADDED looks like: the capture iterates the candidate config, so
    it is in the snapshot with every object absent); a configured cluster the
    snapshot never saw -- one added to site.yaml after the transaction opened
    -- is scaled to zero; and the expected-collector rule namespace goes back
    to what it was."""
    harness = _Rollback(
        monkeypatch, (_target("gpu-a"), _target("gpu-b"), _target("gpu-c"))
    )

    harness.run(
        {
            "dataplane_adot": {
                "clusters": {"gpu-a": _snapshot_present(), "gpu-b": _snapshot_absent()},
                "expected_rules": {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="},
            }
        }
    )

    assert harness.candidate_renders == [], (
        "the candidate manifest was rendered during a snapshot rollback"
    )
    assert harness.control_plane_restored == [{"adot": []}] or (
        harness.control_plane_restored[0]["adot"] == []
    )
    # gpu-a: the previous objects, as one List, through gpu-a's context; then
    # the restored Deployment is restarted and waited for.
    assert harness.applied == [
        (
            "gpu-a-context",
            {
                "apiVersion": "v1",
                "kind": "List",
                "items": _snapshot_present()["objects"],
            },
        )
    ]
    verbs_a = [
        next(word for word in arguments if word in {"apply", "delete", "rollout"})
        for arguments in harness.kubectl("gpu-a-context")
    ]
    assert verbs_a == ["apply", "rollout", "rollout"], verbs_a
    # gpu-b: every declared object deleted in reverse order, nothing applied.
    deletes_b = [
        arguments
        for arguments in harness.kubectl("gpu-b-context")
        if "delete" in arguments
    ]
    assert [arguments[-2] for arguments in deletes_b] == [
        item["name"] for item in reversed(_declared())
    ]
    assert all("--ignore-not-found" in arguments for arguments in deletes_b), deletes_b
    assert not any(
        "apply" in arguments for arguments in harness.kubectl("gpu-b-context")
    ), "a collector was created on a cluster that had none before the release"
    # gpu-c: not in the snapshot -> joined site.yaml after the transaction
    # opened, so the candidate never captured it -> scaled to zero.
    assert harness.scaled == [
        (["kubectl", "--context", "gpu-c-context"], DATAPLANE_ADOT_DEPLOYMENT, 0)
    ]
    # The expected-collector rules go back to the previous definition.
    aws = [arguments for arguments in harness.calls if arguments[0] == "aws"]
    assert [arguments[2] for arguments in aws] == ["put-rule-groups-namespace"]
    assert DATAPLANE.DATAPLANE_EXPECTED_RULE_NAMESPACE in aws[0]
    # And the durable record names the path and the per-cluster outcome.
    assert harness.record() == {
        "path": "snapshot",
        "clusters": {
            "gpu-a": "restored",
            "gpu-b": "removed",
            "gpu-c": "scaled-to-zero",
        },
        "expected_rules": "restored",
    }


def test_a_cluster_the_candidate_never_touched_is_reapplied_from_its_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision pinned: the restore does not consult per-cluster progress.

    Re-applying a cluster's own snapshot is idempotent (apply reports
    `unchanged`; the deletes are `--ignore-not-found`) and costs one collector
    restart; reading progress to skip it would tie the compensation to the
    progress record's shape and leave a partially applied cluster unrestored.
    What must never happen is the candidate rendering being applied to it.
    """
    harness = _Rollback(monkeypatch, (_target("gpu-a"), _target("gpu-b")))
    harness.release.state["cluster_progress"] = {
        "gpu-a": {"components": {"observability": "COMPLETED"}}
    }

    harness.run(
        {
            "dataplane_adot": {
                "clusters": {
                    "gpu-a": _snapshot_present("a"),
                    "gpu-b": _snapshot_present("b"),
                },
                "expected_rules": {"present": False, "data_base64": None},
            }
        }
    )

    assert harness.candidate_renders == []
    assert [
        (context, body["items"][0]["data"]) for context, body in harness.applied
    ] == [
        ("gpu-a-context", {"collector.yaml": "a"}),
        ("gpu-b-context", {"collector.yaml": "b"}),
    ]
    assert harness.scaled == []
    assert harness.record()["clusters"] == {"gpu-a": "restored", "gpu-b": "restored"}
    # No expected rules before the release: the namespace the candidate put is
    # deleted (the probe says it exists), so no per-cluster rule lingers.
    aws = [arguments for arguments in harness.calls if arguments[0] == "aws"]
    assert [arguments[2] for arguments in aws] == ["delete-rule-groups-namespace"]
    assert harness.record()["expected_rules"] == "deleted"


def test_an_absent_expected_rule_namespace_is_not_deleted_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Rollback(monkeypatch, (_target("gpu-a"),), expected_rules_present=False)

    harness.run(
        {
            "dataplane_adot": {
                "clusters": {"gpu-a": _snapshot_present()},
                "expected_rules": {"present": False, "data_base64": None},
            }
        }
    )

    assert [a for a in harness.calls if a[0] == "aws" and "delete" in a[2]] == []
    assert harness.record()["expected_rules"] == "absent"


def test_a_state_captured_before_the_snapshot_falls_back_to_the_previous_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed transaction keeps the snapshot it was started with. Without the
    per-cluster half the only honest move is the old one -- the candidate's
    manifest with the previous image on every configured cluster -- and the
    record has to say so instead of claiming a snapshot restore."""
    harness = _Rollback(monkeypatch, (_target("gpu-a"), _target("gpu-b")))

    harness.run({})

    assert harness.candidate_renders == [
        f"gpu-a@{PREVIOUS_ADOT_IMAGE}",
        f"gpu-b@{PREVIOUS_ADOT_IMAGE}",
    ]
    assert harness.applied == [] and harness.scaled == []
    assert harness.record() == {
        "path": "previous-image",
        "image": PREVIOUS_ADOT_IMAGE,
        "clusters": {
            "gpu-a": "candidate-manifest-with-previous-image",
            "gpu-b": "candidate-manifest-with-previous-image",
        },
    }


def test_a_snapshot_cluster_missing_from_the_config_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Rollback(monkeypatch, (_target("gpu-a"),))

    with pytest.raises(ReleaseError, match="gpu-z"):
        harness.run(
            {
                "dataplane_adot": {
                    "clusters": {"gpu-z": _snapshot_present()},
                    "expected_rules": {"present": False, "data_base64": None},
                }
            }
        )

    assert harness.applied == [] and harness.scaled == [], (
        "a snapshot that cannot be restored still mutated a cluster"
    )
    # LOW-1: the WHOLE observability snapshot is validated before either half
    # mutates anything -- the control-plane AMP blobs and collector included.
    assert harness.control_plane_restored == [], (
        "the control-plane observability was restored on a snapshot whose "
        "data-plane half cannot be put back"
    )
    assert harness.calls == [] and harness.probes == [], (
        f"a refused snapshot still issued commands: {harness.calls + harness.probes}"
    )


def test_a_fallback_state_without_the_previous_image_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOW-1 for the previous-image path: the missing ``adot_image`` is found
    before the control-plane half is put back, not after."""
    harness = _Rollback(monkeypatch, (_target("gpu-a"),))
    previous_without_image = {"observability": {"adot": []}}

    with pytest.raises(ReleaseError, match="previous ADOT image"):
        DATAPLANE.restore_control_plane_observability(
            harness.release, previous_without_image
        )

    assert harness.control_plane_restored == [], (
        "the control-plane snapshot was restored although the data-plane half "
        "cannot be compensated"
    )


# --- fix round 2, MEDIUM-2: an own phase, after the control plane, that continues


def test_the_dataplane_phase_runs_after_the_control_plane_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control-plane observability snapshot is still restored where it was
    (before the data plane); the per-cluster collectors get their own phase,
    checkpointed after ``rollback-cpu-restored``: a GPU cluster whose API is
    unreachable is a common reason the release failed in the first place, and
    it must not stop the previous control plane from coming back."""
    harness = _Rollback(monkeypatch, (_target("gpu-a"),), cpu=True)

    harness.run(
        {
            "dataplane_adot": {
                "clusters": {"gpu-a": _snapshot_present()},
                "expected_rules": {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="},
            }
        }
    )

    phases = harness.saved_phases()
    assert "rollback-cpu-restored" in phases, phases
    assert phases.index("rollback-observability-restored") < phases.index(
        "rollback-cpu-restored"
    ), phases
    assert phases.index("rollback-cpu-restored") < phases.index(PHASE_DONE), phases
    assert phases.index(PHASE_DONE) < phases.index("rollback-restored"), phases
    # The control-plane half ran in its own phase before the CPU restore, the
    # data-plane objects were touched only after it.
    first_kubectl = next(
        index for index, call in enumerate(harness.calls) if call[0] == "kubectl"
    )
    assert len(harness.control_plane_restored) == 1, harness.control_plane_restored
    assert harness.control_plane_restored[0]["adot"] == []
    assert harness.phase_calls.index("_restore_observability_snapshot") < (
        harness.phase_calls.index("_restore_rollback_cpu")
    ), harness.phase_calls
    assert first_kubectl >= 0
    assert harness.record()["clusters"] == {"gpu-a": "restored"}
    assert harness.phase()["status"] == "COMPLETED"
    for name in (PHASE_DONE, "rollback-dataplane-observability-restoring"):
        assert name in ADMIN.ROLLBACK_PHASES, (
            f"{name} is not a rollback phase the admin resume recognises"
        )


def test_one_failed_cluster_does_not_stop_the_others_or_the_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpu-a's ``rollout status`` fails: gpu-b is still restored, the expected
    rules are still put back, and the phase then raises ONE error naming gpu-a,
    so the rollback is honestly FAILED (a resume re-runs the phase) while the
    durable record shows what did and did not happen per cluster."""
    harness = _Rollback(
        monkeypatch,
        (_target("gpu-a"), _target("gpu-b")),
        failing_status=frozenset({"gpu-a-context"}),
    )

    with pytest.raises(ReleaseError, match="gpu-a") as failure:
        harness.run(
            {
                "dataplane_adot": {
                    "clusters": {
                        "gpu-a": _snapshot_present("a"),
                        "gpu-b": _snapshot_present("b"),
                    },
                    "expected_rules": {
                        "present": True,
                        "data_base64": "Z3JvdXBzOiBbXQo=",
                    },
                }
            }
        )

    assert "gpu-b" not in str(failure.value).split("restored")[0], (
        f"the error names a cluster that was restored as failed: {failure.value}"
    )
    assert [context for context, _ in harness.applied] == [
        "gpu-a-context",
        "gpu-b-context",
    ], "the failure on gpu-a stopped gpu-b from being restored"
    aws = [arguments for arguments in harness.calls if arguments[0] == "aws"]
    assert "put-rule-groups-namespace" in [arguments[2] for arguments in aws], (
        "the expected-rules restore was skipped after a cluster failed"
    )
    phase = harness.phase()
    assert phase["status"] == "FAILED", phase
    record = phase["details"]
    assert record["clusters"]["gpu-b"] == "restored", record
    assert record["clusters"]["gpu-a"].startswith("failed: "), record
    assert "gpu-a-context" in record["clusters"]["gpu-a"], record
    assert record["expected_rules"] == "restored", record
    assert record["path"] == "snapshot", record
    assert "gpu-a" in record["error"], record
    assert PHASE_DONE not in harness.saves[-1]["rollback_completed_phases"], (
        "a phase that failed on a cluster was checkpointed as complete"
    )


# --- fix round 2, LOW-4: an unchanged apply does not restart the collector -------

UNCHANGED_APPLY = (
    f"configmap/{DATAPLANE_ADOT_DEPLOYMENT} unchanged\n"
    f"deployment.apps/{DATAPLANE_ADOT_DEPLOYMENT} unchanged\n"
)


def test_an_unchanged_apply_skips_the_collector_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster the candidate never touched applies as ``unchanged`` on every
    object; restarting its collector anyway costs a 30-60 s Recreate gap on a
    cluster that was never wrong. The record says so per cluster."""
    harness = _Rollback(monkeypatch, (_target("gpu-a"),), apply_output=UNCHANGED_APPLY)

    harness.run(
        {
            "dataplane_adot": {
                "clusters": {"gpu-a": _snapshot_present()},
                "expected_rules": {"present": False, "data_base64": None},
            }
        }
    )

    verbs = [
        next(word for word in arguments if word in {"apply", "delete", "rollout"})
        for arguments in harness.kubectl("gpu-a-context")
    ]
    assert verbs == ["apply"], verbs
    assert harness.record()["clusters"] == {"gpu-a": "restored-unchanged"}


def test_apply_snapshot_objects_reports_whether_anything_changed() -> None:
    outputs: list[str] = []
    release = SimpleNamespace(
        runner=SimpleNamespace(run=lambda *_a, **_k: outputs.pop(0)),
        _cpu=lambda *args: ["kubectl", *args],
    )
    objects = [{"kind": "ConfigMap", "metadata": {"name": "x"}}]

    outputs[:] = ["configmap/x unchanged\ndeployment.apps/y unchanged"]
    assert SNAPSHOT.apply_snapshot_objects(release, objects) is False
    outputs[:] = ["configmap/x unchanged\ndeployment.apps/y configured"]
    assert SNAPSHOT.apply_snapshot_objects(release, objects) is True
    # Output the caller cannot parse (a stub runner, a dry run) is "changed":
    # the restart is the safe default, skipping it is the optimisation.
    outputs[:] = [""]
    assert SNAPSHOT.apply_snapshot_objects(release, objects) is True
    assert SNAPSHOT.apply_snapshot_objects(release, []) is False


def test_the_control_plane_collector_restore_is_gated_the_same_way() -> None:
    """The installer already skips the control-plane collector's restart on an
    ``unchanged`` pair (install-amp-monitoring.sh); its rollback twin does now."""
    calls: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> str:
        calls.append(list(arguments))
        if "apply" in arguments:
            return "configmap/gpu-fault-adot unchanged\ndeployment.apps/gpu-fault-adot unchanged"
        return ""

    release = SimpleNamespace(
        runner=SimpleNamespace(run=run), _cpu=lambda *args: ["kubectl", *args]
    )
    snapshot = {
        "adot": {
            "namespace": NAMESPACE,
            "objects": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "gpu-fault-adot"},
                }
            ],
            "absent": [],
        }
    }

    CONTROL_PLANE.restore_adot_objects(release, snapshot)

    assert not any("rollout" in arguments for arguments in calls), calls


def test_the_control_plane_snapshot_is_validated_whole_before_the_amp_puts() -> None:
    """LOW-1 on the control-plane half: an ADOT object list that cannot be put
    back is found before the AMP rule and Alertmanager blobs are rewritten."""
    calls: list[list[str]] = []
    release = SimpleNamespace(
        config=SimpleNamespace(
            aws_region="us-east-1", health=SimpleNamespace(amp_workspace_id="ws-a")
        ),
        runner=SimpleNamespace(run=lambda a, **_k: calls.append(list(a)) or ""),
        _cpu=lambda *args: ["kubectl", *args],
    )

    with pytest.raises(ReleaseError, match="ADOT collector"):
        CONTROL_PLANE.restore_observability_snapshot(
            release,
            {
                "rule_namespace": "rules",
                "rules_data_base64": "cnVsZXM=",
                "alertmanager_data_base64": "YWxlcnRz",
                "adot": {"namespace": NAMESPACE, "objects": [], "absent": []},
            },
        )

    assert calls == [], f"the AMP blobs were rewritten before validation: {calls}"


# --- fix round 2, LOW-5 / MEDIUM-3: the rules namespace's state is read, not guessed


def _rules_release(
    answers: list[tuple[int, str, str]],
) -> tuple[Any, list[list[str]], list[float]]:
    calls: list[list[str]] = []
    slept: list[float] = []

    def probe_output(arguments: list[str], **_kwargs: Any) -> tuple[int, str, str]:
        calls.append(list(arguments))
        return answers.pop(0)

    release = SimpleNamespace(
        config=SimpleNamespace(
            aws_region="us-east-1", health=SimpleNamespace(amp_workspace_id="ws-a")
        ),
        runner=SimpleNamespace(
            run=lambda a, **_k: calls.append(list(a)) or "", probe_output=probe_output
        ),
    )
    return release, calls, slept


def test_a_transient_describe_failure_is_not_read_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boolean probe turned a throttle or a credentials error into "the
    namespace is absent", and the restore then tried to CREATE over an existing
    namespace (or skipped a delete). Only ResourceNotFoundException is absence."""
    release, calls, _slept = _rules_release([(255, "", "Unable to locate credentials")])

    with pytest.raises(ReleaseError, match="expected-collector rule namespace"):
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": False, "data_base64": None}
        )

    assert [c[2] for c in calls] == ["describe-rule-groups-namespace"], calls


def test_a_namespace_still_deleting_is_waited_for_before_the_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The candidate's installer may have just deleted the namespace (AMP deletes
    asynchronously); a put on a DELETING namespace is a ConflictException. The
    restore waits for it to settle -- here it disappears, so it is created."""
    release, calls, slept = _rules_release(
        [
            _describe_answer("DELETING"),
            _describe_answer("DELETING"),
            (254, "", NOT_FOUND),
            _describe_answer("ACTIVE"),
        ]
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", slept.append)

    outcome = DATAPLANE.restore_dataplane_expected_rules(
        release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
    )

    assert outcome == "restored"
    assert [c[2] for c in calls] == [
        "describe-rule-groups-namespace",
        "describe-rule-groups-namespace",
        "describe-rule-groups-namespace",
        "create-rule-groups-namespace",
        "describe-rule-groups-namespace",
    ], calls
    assert len(slept) == 2, slept


def test_a_namespace_still_updating_is_waited_for_then_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, calls, slept = _rules_release(
        [
            _describe_answer("UPDATING"),
            _describe_answer("ACTIVE"),
            _describe_answer("ACTIVE"),
        ]
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", slept.append)

    outcome = DATAPLANE.restore_dataplane_expected_rules(
        release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
    )

    assert outcome == "restored"
    assert [c[2] for c in calls][-2] == "put-rule-groups-namespace", calls
    assert len(slept) == 1, slept


# --- the restore waits out its OWN write, as the installer does ----------------------


def test_the_put_is_recorded_as_restored_only_once_amp_reports_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AMP validates a put asynchronously: the namespace goes UPDATING and the
    previous rules keep serving until it settles. Returning ``restored`` at the
    put meant the record was written while AMP was still validating, the next
    writer (a resume, the next deploy's installer) could hit a ConflictException,
    and a rejected definition was never noticed. The restore now polls describe
    until ACTIVE, exactly as the installer's ``wait_for_amp_definition`` does."""
    release, calls, slept = _rules_release(
        [
            _describe_answer("ACTIVE"),  # the pre-write settle check
            _describe_answer("UPDATING"),
            _describe_answer("UPDATING"),
            _describe_answer("ACTIVE"),
        ]
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", slept.append)

    outcome = DATAPLANE.restore_dataplane_expected_rules(
        release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
    )

    assert outcome == "restored"
    assert [c[2] for c in calls] == [
        "describe-rule-groups-namespace",
        "put-rule-groups-namespace",
        "describe-rule-groups-namespace",
        "describe-rule-groups-namespace",
        "describe-rule-groups-namespace",
    ], f"the put was not followed by a describe loop until ACTIVE: {calls}"
    assert slept == [DATAPLANE.EXPECTED_RULES_SETTLE_SECONDS] * 2, slept


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("exists", "verb", "status"),
    [
        (True, "put-rule-groups-namespace", "UPDATE_FAILED"),
        (False, "create-rule-groups-namespace", "CREATION_FAILED"),
    ],
)
def test_a_definition_amp_rejects_fails_the_restore_naming_the_reason(
    monkeypatch: pytest.MonkeyPatch, exists: bool, verb: str, status: str
) -> None:
    """A rejected definition leaves the OLD rules serving; the restore must say
    so, with AMP's ``statusReason``, instead of recording ``restored``."""
    reason = "rule group gpu-fault-dataplane-expected: parse error at line 3"
    release, calls, _slept = _rules_release(
        [
            _describe_answer("ACTIVE") if exists else (254, "", NOT_FOUND),
            _describe_answer(status, reason=reason),
        ]
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", lambda _s: None)

    with pytest.raises(ReleaseError, match=status) as caught:
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
        )

    assert reason in str(caught.value), caught.value
    assert [c[2] for c in calls][1] == verb, calls


def test_a_put_that_never_reaches_active_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, calls, slept = _rules_release(
        [_describe_answer("ACTIVE")] + [_describe_answer("UPDATING")] * 5
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", slept.append)
    monkeypatch.setattr(DATAPLANE, "EXPECTED_RULES_SETTLE_ATTEMPTS", 3)

    with pytest.raises(ReleaseError, match="still UPDATING after 15s"):
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
        )

    assert [c[2] for c in calls][1] == "put-rule-groups-namespace", calls
    assert len(slept) == 3, slept


def test_a_namespace_that_vanishes_after_the_put_is_not_recorded_as_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, _calls, _slept = _rules_release(
        [_describe_answer("ACTIVE"), (254, "", NOT_FOUND)]
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", lambda _s: None)

    with pytest.raises(ReleaseError, match="disappeared"):
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
        )


def test_the_delete_is_recorded_as_deleted_only_once_the_namespace_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delete is asynchronous too (``wait_for_amp_definition_gone`` in the
    installer): a namespace still DELETING is a ConflictException for the next
    create, so ``deleted`` means describe answered ResourceNotFoundException."""
    release, calls, slept = _rules_release(
        [_describe_answer("ACTIVE"), _describe_answer("DELETING"), (254, "", NOT_FOUND)]
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", slept.append)

    outcome = DATAPLANE.restore_dataplane_expected_rules(
        release, {"present": False, "data_base64": None}
    )

    assert outcome == "deleted"
    assert [c[2] for c in calls] == [
        "describe-rule-groups-namespace",
        "delete-rule-groups-namespace",
        "describe-rule-groups-namespace",
        "describe-rule-groups-namespace",
    ], f"the delete was not followed by a describe loop until gone: {calls}"
    assert len(slept) == 1, slept


def test_a_delete_that_never_settles_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, _calls, _slept = _rules_release(
        [_describe_answer("ACTIVE")] + [_describe_answer("DELETING")] * 5
    )
    monkeypatch.setattr(DATAPLANE, "_sleep", lambda _s: None)
    monkeypatch.setattr(DATAPLANE, "EXPECTED_RULES_SETTLE_ATTEMPTS", 3)

    with pytest.raises(ReleaseError, match="still DELETING after 15s"):
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": False, "data_base64": None}
        )


def test_a_namespace_that_never_settles_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, _calls, _slept = _rules_release([_describe_answer("CREATING")] * 4)
    monkeypatch.setattr(DATAPLANE, "_sleep", lambda _s: None)
    monkeypatch.setattr(DATAPLANE, "EXPECTED_RULES_SETTLE_ATTEMPTS", 3)

    with pytest.raises(ReleaseError, match="CREATING"):
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": True, "data_base64": "Z3JvdXBzOiBbXQo="}
        )


def test_the_replayed_manifest_narrative_is_true_for_both_collectors() -> None:
    """The docstring used to promise that an observability template edit is
    genuinely undone; that was only true for the control-plane half. It now has
    to say the data-plane collector is snapshot-restored too, and name the one
    case where the candidate manifest IS replayed (a pre-snapshot state)."""
    text = str(PROGRESS.__doc__ or "") + "\n" + _module_source(PROGRESS)
    assert "observability_manifests" not in PROGRESS.REPLAYED_MANIFEST_CHANGES
    for needle in ("per GPU cluster", "previous-image"):
        assert needle in text, (
            f"the REPLAYED_MANIFEST_CHANGES narrative lacks {needle!r}"
        )


def _module_source(module: Any) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


# --- F4: the skip branch scales a previously applied collector down ---------------


class _Runner:
    dry_run = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.probes: list[list[str]] = []

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        self.calls.append((list(arguments), kwargs))
        return ""

    def probe(self, arguments: list[str], **_kwargs: Any) -> bool:
        self.probes.append(list(arguments))
        return True


def _config(tmp_path: Path, *, adot_irsa_role_arn: str | None = None) -> Path:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    if adot_irsa_role_arn:
        value["clusters"][0]["adot_irsa_role_arn"] = adot_irsa_role_arn
    value["health"] = {"amp_workspace_id": "ws-a"}
    target = tmp_path / "with-health.json"
    target.write_text(json.dumps(value))
    return target


def test_the_apply_skip_branch_scales_a_leftover_collector_to_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing a cluster's role is a release (the digest moves), and the
    OBSERVABILITY node then takes the skip branch: the collector applied by the
    previous release must not keep running with credentials that no longer
    exist, so the skip scales it to zero -- the same lever ``remove_cluster``
    uses. The preflight's skip stays read-only."""
    release = MODULE.RegionalRelease(
        MODULE.ReleaseConfig.load(_config(tmp_path)), _Runner()
    )
    target = release.config.clusters[0]

    BOOTSTRAP.apply_gpu_adot_collector(release, target)

    assert "gpu-a: data-plane ADOT collector not applied" in capsys.readouterr().err
    assert release.runner.probes == [
        [
            "kubectl",
            "--context",
            target.context,
            "-n",
            NAMESPACE,
            "get",
            "deployment",
            DATAPLANE_ADOT_DEPLOYMENT,
        ]
    ]
    scale = [arguments for arguments, _ in release.runner.calls if "scale" in arguments]
    assert len(scale) == 1 and scale[0][:3] == ["kubectl", "--context", target.context]
    assert f"deployment/{DATAPLANE_ADOT_DEPLOYMENT}" in scale[0]
    assert "--replicas=0" in scale[0]
    assert not any("apply" in arguments for arguments, _ in release.runner.calls), (
        "the skip branch applied a manifest"
    )

    release.runner.calls.clear()
    release.runner.probes.clear()
    BOOTSTRAP.preflight_gpu_adot_collector(release, target)
    assert release.runner.calls == [] and release.runner.probes == []
