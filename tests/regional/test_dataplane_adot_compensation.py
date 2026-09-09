"""Rollback compensates the per-GPU-cluster ADOT collector from a snapshot (F10 fix 1).

The first wiring re-rendered the candidate's collector manifest with the previous
image on every cluster in the candidate config, so a rollback kept the
candidate's keep filters and relabels, created collectors on clusters the
candidate never touched, and left a collector running on a cluster whose role
the candidate had just added. These tests pin the compensation: the previous
release's live objects are captured per cluster before the OBSERVABILITY node
runs, and rollback restores each cluster from ITS snapshot -- objects put back
where a collector was live, the candidate's objects deleted where none was,
and a cluster the snapshot never saw scaled to zero. The previous-image path
survives only for a state captured before the snapshot existed, and the
rollback record says which path ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_dataplane_observability as DATAPLANE
from gpu_fault_release import regional_gpu_bootstrap as BOOTSTRAP
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


class _Rollback:
    """Drive ``rollback_release`` with only the observability restore live."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clusters: tuple[SimpleNamespace, ...],
        *,
        expected_rules_present: bool = True,
    ) -> None:
        for name in (
            "_stage_rollback_controller",
            "_rollback_gpu_clusters",
            "_restore_rollback_cpu",
            "_verify_and_complete_rollback",
            "cleanup_candidate_rollout_state",
        ):
            monkeypatch.setattr(ORCHESTRATION, name, lambda *_args, **_kwargs: None)
        self.clusters = clusters
        self.calls: list[list[str]] = []
        self.applied: list[tuple[str, Any]] = []
        self.candidate_renders: list[str] = []
        self.scaled: list[tuple[list[str], str, int]] = []
        self.saves: list[dict[str, Any]] = []
        self.control_plane_restored: list[Any] = []
        self.expected_rules_present = expected_rules_present

        def run(arguments: list[str], **_kwargs: Any) -> str:
            self.calls.append(list(arguments))
            if arguments[0] == "kubectl" and "apply" in arguments:
                path = Path(arguments[arguments.index("-f") + 1])
                self.applied.append(
                    (arguments[2], json.loads(path.read_text(encoding="utf-8")))
                )
            return ""

        def probe(arguments: list[str], **_kwargs: Any) -> bool:
            return self.expected_rules_present

        self.release = SimpleNamespace(
            state={
                "release_diff": {
                    "kind": str(DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY),
                    "changed": ["observability_adot"],
                },
                "execution_plan": {"nodes": ["observability", "verify"]},
                "completed_phases": ["uploaded", "observability-ready"],
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
            runner=SimpleNamespace(run=run, probe=probe),
            _gpu=_gpu,
            _refresh_aurora_credentials=lambda: None,
            _require_no_inflight_installs=lambda **_kwargs: None,
            _save_state=lambda phase, **updates: self.saves.append(
                {"phase": phase, **updates}
            ),
            _restore_observability_snapshot=self.control_plane_restored.append,
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

    def record(self) -> dict[str, Any]:
        phases = self.saves[-1]["rollback_timing"]["phases"]
        return phases["observability_restore"]["details"]["dataplane_adot"]


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
    objects deleted, not a collector created; a cluster in the candidate config
    the snapshot never saw is scaled to zero; and the expected-collector rule
    namespace goes back to what it was."""
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
    # gpu-c: not in the snapshot -> the candidate created it -> scaled to zero.
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
