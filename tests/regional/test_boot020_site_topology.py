"""BOOT-020 over a site with one GPU cluster parked at zero nodes.

HyperPod keeps four nodes beside a plain EKS cluster parked at zero nodes: the
shape the live site runs in since 2026-11-07. The run must accept the engine's
explicit zero-node entries for the parked cluster and still require the full
node behaviour where there are nodes (``boot020_site_topology``). Each defect
below is one the site-wide verdicts let pass: the parked cluster's components
or Reconciler roll standing in for Agents that never moved.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import (
    regional_parked_cluster,
    regional_release_gpu_rollout,
    regional_release_node_preflight,
)
from scripts.e2e.regional import boot020_site_topology as topology
from scripts.e2e.regional import run_boot020_release_rolling as boot020
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from tests.regional._boot020_site_shape import (
    FULL_DIGEST,
    FULL_PROFILE,
    PARKED_PROOF,
    SITE_DIGEST,
    SITE_PROFILE,
    agent_identity,
    cluster_timing,
    node_shape,
    waves,
    zero_node_identity,
    zero_node_shape,
)
from tests.regional._cov95_boot_extra_rolling import (
    BackendModel,
    ReleaseEngine,
    RollingEntry,
)

HP = "hp-cluster-hypd"
EKS = "gpu-eks-cluster"
HP_NODES = [f"hyperpod-i-{index}" for index in range(4)]
SKIPPED_ALL = list(regional_parked_cluster.SKIPPED_NODE_WORK)
RELEASES = {
    "noop": "release-site",
    "control_plane": "release-site",
    "executor": "release-B",
    "agent": "release-C",
    "full": "release-D",
}
PLANS = {
    "control_plane": {},
    "executor": ["collector", "executor", "reconciler", "watcher"],
    "agent": ["agent", "reconciler"],
    "full": ["agent", "collector", "executor", "reconciler", "watcher"],
}
CPU = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
GPU = (
    boot020.EXECUTOR_DEPLOYMENT,
    "gpu-fault-completion-watcher",
    "gpu-fault-kubernetes-node-resource-collector",
    "gpu-fault-node-installer-reconciler",
)
PINS = {
    "rolled-back": {"required": "a" * 64, "compatible": []},
    "staged": {"required": "a" * 64, "compatible": ["b" * 64]},
    "finalized": {"required": "b" * 64, "compatible": []},
}


def capture(release: str) -> dict[str, Any]:
    """A GPU cluster's capture: executor wheel, Reconciler bundle and template."""

    wheel, bundle = {
        "site": ("executor-v1", "bundle-v1"),
        "executor": ("executor-v2", "bundle-v1"),
        "agent": ("executor-v2", "bundle-v2"),
        "full": ("executor-v3", "bundle-v3"),
    }[release]
    return {
        "wheel": wheel,
        "reconciler_wheel": wheel,
        "bundle": f"cm-{bundle}",
        "bundle_sha256": f"{bundle}-sha",
        "template": "cm-template-v1",
        "template_sha256": "template-v1-sha",
    }


class ParkedSite:
    """HyperPod with four nodes beside an EKS cluster parked at zero nodes,
    with switches for the defects the per-cluster verdicts must catch."""

    def __init__(self, *defects: str) -> None:
        self.defects = set(defects)
        self.completed: set[str] = set()
        self.restored = False
        self.live: dict[str, Any] = {
            "cpu_wheel": "cpu-v1",
            "clusters": {HP: capture("site"), EKS: capture("site")},
            "agent_identities": {
                HP: agent_identity(HP_NODES),
                EKS: zero_node_identity(),
            },
        }
        self.initial_live = copy.deepcopy(self.live)
        # The site's own deploy left the parked cluster's record.
        self.records: dict[str, Any] = {EKS: self._record("release-site", "upgrade")}
        self.shape = {HP: node_shape(HP_NODES), EKS: zero_node_shape()}
        self.cpu_generations = dict.fromkeys(CPU, 1)
        self.cpu_identities = {
            name: {"replicas": 3, "template_sha256": f"{name}-v1"} for name in CPU
        }
        self.gpu_generations = {cid: dict.fromkeys(GPU, 1) for cid in (HP, EKS)}
        self.calls: list[tuple[Any, ...]] = []
        self.commits: list[str] = []

    def _record(
        self, release_id: str, phase: str, skipped: list[str] | None = None
    ) -> dict[str, Any]:
        reconciler = self.live["clusters"][EKS]
        return {
            **{key: PARKED_PROOF[key] for key in ("state", "provider", "reason")},
            "groups": copy.deepcopy(PARKED_PROOF["groups"]),
            "release_id": release_id,
            "phase": phase,
            "skipped": list(SKIPPED_ALL if skipped is None else skipped),
            "reconciler": {
                "bundle_sha256": reconciler["bundle_sha256"],
                "template_sha256": reconciler["template_sha256"],
            },
        }

    # --- reads -----------------------------------------------------------

    def classify(self, scenario: str) -> dict[str, Any]:
        scenario = "noop" if scenario == "restore" else scenario
        if scenario == "noop" and "full" in self.completed and not self.restored:
            return {"kind": "FULL", "changed": ["runtime_profile"]}
        kind = (
            "NOOP" if scenario in self.completed else boot020.EXPECTED_KINDS[scenario]
        )
        field = {
            "control_plane": "admin_config_worker",
            "executor": "executor_manifests",
            "agent": "node_bundle",
            "full": "runtime_profile",
        }.get(scenario)
        return {"kind": kind, "changed": [] if kind == "NOOP" or not field else [field]}

    def snapshot(self, scenario: str, *, live: bool = True) -> dict[str, Any]:
        return {
            "phase": "complete",
            "release_id": RELEASES.get(scenario, "release-site"),
            "transaction_committed": True,
            "live": copy.deepcopy(self.live) if live else None,
            "cpu_generations": dict(self.cpu_generations),
            "cpu_identities": copy.deepcopy(self.cpu_identities),
            "gpu_generations": copy.deepcopy(self.gpu_generations),
            "next_deploy": None,
        }

    def node_safety(self, scenario: str) -> dict[str, Any]:
        return {"safe": True, "clusters": {HP: [], EKS: []}, "shape": self.shape}

    def profile_versions(self, scenario: str) -> dict[str, Any]:
        hp_full = (
            SITE_PROFILE if "full-profile-unchanged" in self.defects else FULL_PROFILE
        )
        return {
            "clusters": {
                HP: {"version": hp_full, "agent_config_digest": FULL_DIGEST},
                EKS: {"version": "eks-v1-boot020", "agent_config_digest": "e" * 64},
            },
            "site_clusters": {
                HP: {"version": SITE_PROFILE, "agent_config_digest": SITE_DIGEST},
                EKS: {"version": "eks-v1", "agent_config_digest": "f" * 64},
            },
            "registered_versions": ["eks-v1-boot020", FULL_PROFILE],
            "permanent": True,
        }

    def cpu_env_alignment(self, scenario: str) -> dict[str, Any]:
        return {"aligned": True, "deployments": {}}

    def resume_rollback(self, scenario: str) -> dict[str, Any]:
        return {"resumed": False, "phase": "complete"}

    def release_state(self) -> dict[str, Any]:
        return {"phase": "complete", "release_id": "release-site"}

    def candidate_releases(self) -> dict[str, dict[str, str]]:
        return {
            name: {"release_id": release, "admin_config_sha256": "a" * 64}
            for name, release in RELEASES.items()
        }

    def resume_upgrade(self, scenario: str) -> dict[str, Any]:
        return {"scenario": scenario, "phase": "complete", "error": None}

    # --- the deploys --------------------------------------------------------

    def deploy(
        self,
        scenario: str,
        *,
        diff: dict[str, Any],
        fault_phase: str | None = None,
        resume: bool = False,
        auto_rollback: bool | None = None,
        commit: bool = False,
    ) -> dict[str, Any]:
        self.calls.append((scenario, fault_phase, resume, auto_rollback, diff["kind"]))
        before = copy.deepcopy(self.records)
        if diff["kind"] != "NOOP" and not resume:
            # A new transaction starts from an empty state (upgrade_release);
            # a NOOP rewrites the state it loaded.
            self.records = {}
        restore = scenario == "noop" and "full" in self.completed
        release_id = "release-site" if restore else RELEASES[scenario]
        if fault_phase and auto_rollback:
            plan, timing = self._rollback(scenario, release_id)
            return self._result(
                "rolled-back",
                release_id,
                before,
                pins="rolled-back",
                injected_failure=fault_phase,
                rollback_plan=plan,
                rollback_timing=timing,
            )
        if fault_phase:
            return self._result(
                "failed",
                release_id,
                before,
                pins="staged",
                injected_failure=fault_phase,
            )
        if restore:
            self.live = copy.deepcopy(self.initial_live)
            if "restore-mismatch" in self.defects:
                self.live["agent_identities"][HP] = agent_identity(HP_NODES, "agent")
            self.restored = True
            self.records[EKS] = self._record("release-site", "upgrade")
            self.cpu_generations = {
                name: value + 1 for name, value in self.cpu_generations.items()
            }
        else:
            self._forward(scenario, release_id, resume=resume)
        self.completed.add(scenario)
        if commit:
            self.commits.append(scenario)
        return self._result(
            "complete", release_id, before, pins="finalized", transaction_committed=True
        )

    def _result(
        self,
        phase: str,
        release_id: str,
        before: dict[str, Any],
        *,
        pins: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "release_id": release_id,
            "injected_failure": None,
            "pins": {
                **PINS[pins],
                "candidate": "b" * 64,
                "previous_required": "a" * 64,
            },
            "zero_node_clusters": copy.deepcopy(self.records),
            "zero_node_clusters_before": before,
            "operation_duration_seconds": 30.0,
            **extra,
        }

    def _rollback(self, scenario: str, release_id: str) -> tuple[dict, dict]:
        components = list(PLANS[scenario])
        clusters = {HP: components, EKS: components} if components else {}
        if scenario == "agent" and "plan-masks-hp" in self.defects:
            clusters = {HP: ["reconciler"], EKS: ["agent", "reconciler"]}
        hp_waves = waves(HP_NODES) if scenario in {"agent", "full"} else None
        if scenario == "agent":
            hp_waves = self._damaged_waves(hp_waves or {})
        if scenario == "executor" and "executor-waves" in self.defects:
            hp_waves = waves(HP_NODES)
        timing: dict[str, Any] = {
            "t_safe_seconds": 860.0,
            "t_full_seconds": 920.0,
            "clusters": {
                cid: cluster_timing(
                    comps,
                    hp_waves
                    if cid == HP
                    else (waves(["ghost"]) if "eks-waves" in self.defects else None),
                )
                for cid, comps in clusters.items()
            },
        }
        if scenario in {"executor", "agent", "full"}:
            skipped = ["node-preflight-batch"] if scenario == "executor" else None
            if scenario == "executor" and "record-skipped" in self.defects:
                skipped = None
            phase = "upgrade" if "record-phase" in self.defects else "rollback"
            if not (scenario == "agent" and "record-missing" in self.defects):
                self.records[EKS] = self._record(release_id, phase, skipped)
        plan = {
            "clusters": clusters,
            "global_components": ["cpu-finalize", "cpu-stage"],
            "restores_data_plane": bool(clusters),
            "needs_controller": scenario in {"agent", "full"},
        }
        return plan, timing

    def _damaged_waves(self, node_waves: dict[str, Any]) -> dict[str, Any]:
        first = HP_NODES[0]
        if "wave-drop" in self.defects:
            node_waves.pop(HP_NODES[-1])
        if "wave-repeat" in self.defects:
            node_waves[HP_NODES[-1]]["nodes"].append(first)
        if "wave-order" in self.defects:
            node_waves[first]["agents_converged_at_epoch"] = 1.0
        if "wave-incomplete" in self.defects:
            node_waves[first]["status"] = "STARTED"
        if "wave-missing-stamp" in self.defects:
            node_waves[first].pop("reconciler_applied_at_epoch")
        return node_waves

    def _forward(self, scenario: str, release_id: str, *, resume: bool) -> None:
        if scenario == "control_plane":
            self.live["cpu_wheel"] = "cpu-v2"
            self.cpu_generations["gpu-fault-control-worker"] += 1
            self.cpu_identities["gpu-fault-control-worker"] = {
                "replicas": 2,
                "template_sha256": "gpu-fault-control-worker-v2",
            }
            if "cpu-touches-record" in self.defects:
                self.records[EKS] = self._record(release_id, "upgrade")
        elif scenario == "executor":
            for cid in (HP, EKS):
                self.live["clusters"][cid] = capture("executor")
                self.gpu_generations[cid][boot020.EXECUTOR_DEPLOYMENT] += 1
            if resume:
                self.cpu_generations = {
                    name: value + 1 for name, value in self.cpu_generations.items()
                }
            self.records[EKS] = self._record(
                release_id, "upgrade", ["node-preflight-batch"]
            )
        elif scenario == "agent":
            for cid in (HP, EKS):
                self.live["clusters"][cid] = capture("agent")
            if "hp-agents-stuck" not in self.defects:
                self.live["agent_identities"][HP] = agent_identity(HP_NODES, "agent")
            if "eks-node-appears" in self.defects:
                self.live["agent_identities"][EKS] = agent_identity(["ip-10-0-1-1"])
            record = self._record(release_id, "upgrade")
            if "record-reconciler" in self.defects:
                record["reconciler"]["bundle_sha256"] = "bundle-v1-sha"
            self.records[EKS] = record
            if "hp-record" in self.defects:
                self.records[HP] = self._record(release_id, "upgrade")
        elif scenario == "full":
            self.live["cpu_wheel"] = "cpu-v3"
            for cid in (HP, EKS):
                self.live["clusters"][cid] = capture("full")
            identity = agent_identity(HP_NODES, "full")
            if "full-profile-unchanged" in self.defects:
                identity.update(
                    runtime_profile_version=SITE_PROFILE, config_digest=SITE_DIGEST
                )
            if "full-wrong-digest" in self.defects:
                identity["config_digest"] = SITE_DIGEST
            self.live["agent_identities"][HP] = identity
            self.records[EKS] = self._record(release_id, "upgrade")


def recorder(tmp_path: Path, declared: list[str] | None = None) -> EvidenceRecorder:
    return EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id=boot020.CASE_ID,
        inputs={
            "configs": "unit",
            "zero_node_clusters": [EKS] if declared is None else declared,
        },
    )


def test_the_parked_site_passes_with_the_parked_path_on_record(tmp_path: Path) -> None:
    result = boot020.run_release_rolling(ParkedSite(), recorder(tmp_path))

    assert result["verdict"] == "PASS", result.get("error")
    assert result["site_shape"] == {HP: {"nodes": 4}, EKS: {"zero_nodes": "parked"}}
    assert result["zero_node_clusters"] == [EKS]
    assert result["registered_runtime_profile_versions"] == [
        "eks-v1-boot020",
        FULL_PROFILE,
    ]
    stages = result["stages"]
    assert stages["site_shape"]["declared_zero_node_clusters"] == [EKS]
    phases = {
        key: stages[key]["zero_node_clusters"][EKS]["phase"]
        for key in (
            "executor_injected_failure_and_rollback",
            "executor_resumed",
            "agent_injected_failure_and_rollback",
            "agent_apply_after_rollback",
            "full_injected_failure_and_rollback",
            "full_apply_after_rollback",
            "restore_apply",
        )
    }
    assert phases == {
        "executor_injected_failure_and_rollback": "rollback",
        "executor_resumed": "upgrade",
        "agent_injected_failure_and_rollback": "rollback",
        "agent_apply_after_rollback": "upgrade",
        "full_injected_failure_and_rollback": "rollback",
        "full_apply_after_rollback": "upgrade",
        "restore_apply": "upgrade",
    }, "the evidence carries the parked cluster's record of every GPU deploy"
    waves_seen = stages["agent_injected_failure_and_rollback"]["rollback_timing"][
        "clusters"
    ]
    assert sorted(waves_seen[HP]["waves"]) == HP_NODES
    assert "waves" not in waves_seen[EKS], "the parked cluster rolls no wave"


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        (
            "hp-agents-stuck",
            "Agents of hp-cluster-hypd did not take the release's node",
        ),
        ("plan-masks-hp", r"rollback of hp-cluster-hypd \(nodes\) lacks \['agent'\]"),
        ("wave-drop", "do not cover each of its nodes exactly once"),
        ("wave-repeat", "do not cover each of its nodes exactly once"),
        ("wave-order", "timestamps are out of order"),
        ("wave-incomplete", "did not complete"),
        ("wave-missing-stamp", "lacks one of its five timestamps"),
        ("executor-waves", "the Executor-only rollback restores no Agent"),
        ("eks-waves", "rolled Fleet waves although it has no node"),
        ("record-missing", "recorded no skipped node work for gpu-eks-cluster"),
        ("record-phase", "is not this release's rollback"),
        ("record-skipped", "is not this release's rollback"),
        ("record-reconciler", "is not the one captured after it"),
        (
            "hp-record",
            "node work for hp-cluster-hypd in a deploy although it has nodes",
        ),
        (
            "cpu-touches-record",
            "node work for gpu-eks-cluster in a deploy that does no GPU",
        ),
        ("eks-node-appears", "holds no explicit zero-node entry"),
        ("full-profile-unchanged", "changes no Runtime Profile of hp-cluster-hypd"),
        ("full-wrong-digest", "do not report the FULL Runtime Profile"),
        ("restore-mismatch", "are not back to what noop_before observed"),
    ],
)
def test_a_defect_the_site_wide_verdicts_let_pass_fails_the_stage(
    tmp_path: Path, defect: str, message: str
) -> None:
    evidence = recorder(tmp_path)

    with pytest.raises(boot020.AcceptanceCheckError, match=message):
        boot020.run_release_rolling(ParkedSite(defect), evidence)

    assert evidence.document["verdict"] == "FAIL", evidence.document.get("error")


@pytest.mark.parametrize(
    ("shape", "declared", "message"),
    [
        (
            {HP: node_shape(HP_NODES), EKS: zero_node_shape()},
            [],
            "gpu-eks-cluster has no node .* and is not declared",
        ),
        (
            {HP: node_shape(HP_NODES), EKS: zero_node_shape()},
            [EKS, HP],
            "hp-cluster-hypd is declared with --zero-node-cluster but has 4 node",
        ),
        (
            {
                HP: node_shape(HP_NODES),
                EKS: zero_node_shape({**PARKED_PROOF, "state": "unavailable"}),
            },
            [EKS],
            "is unavailable, not parked",
        ),
        (
            {HP: node_shape(HP_NODES), EKS: {"zero_nodes": None, "unproven": "unit"}},
            [EKS],
            "zero-node proof could not be read",
        ),
        (
            {HP: zero_node_shape(), EKS: zero_node_shape()},
            [EKS, HP],
            "no GPU cluster has nodes",
        ),
        (
            {HP: node_shape(HP_NODES), EKS: zero_node_shape()},
            [EKS, "gpu-other"],
            "names no cluster of the site: gpu-other",
        ),
        ({}, [EKS], "reported no site shape"),
    ],
)
def test_a_site_shape_the_run_cannot_judge_is_refused_before_any_change(
    tmp_path: Path, shape: dict[str, Any], declared: list[str], message: str
) -> None:
    site = ParkedSite()
    site.shape = shape

    with pytest.raises(boot020.AcceptanceCheckError, match=message):
        boot020.run_release_rolling(site, recorder(tmp_path, declared))

    assert site.calls == [], "the refusal came after the site was changed"


def test_a_resume_over_another_site_shape_is_refused(tmp_path: Path) -> None:
    first = recorder(tmp_path)
    with pytest.raises(boot020.AcceptanceCheckError):
        boot020.run_release_rolling(ParkedSite("wave-drop"), first)
    assert first.document["stages"]["site_shape"]["clusters"][HP] == node_shape(
        HP_NODES
    )

    replaced = ParkedSite()
    replaced.shape = {
        HP: node_shape([*HP_NODES[:3], "hyperpod-i-replacement"]),
        EKS: zero_node_shape(),
    }
    with pytest.raises(boot020.AcceptanceCheckError, match="site shape changed"):
        boot020.resume_release_rolling(replaced, recorder(tmp_path))
    assert replaced.calls == [], "the resume converged a site of another shape"


def test_node_safety_reports_each_clusters_nodes_or_its_zero_node_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [SimpleNamespace(cluster_id=cid) for cid in (HP, EKS, "gpu-lost")]
    release = SimpleNamespace(config=SimpleNamespace(clusters=targets))
    listed = {HP: [{"metadata": {"name": name}} for name in HP_NODES]}
    proofs = {EKS: {**PARKED_PROOF, "parked": True, "proven_by": ["unit"]}}
    asked: list[str] = []
    monkeypatch.setattr(
        regional_release_gpu_rollout,
        "gpu_node_items",
        lambda _release, target, *, fresh: listed.get(target.cluster_id, []),
    )
    monkeypatch.setattr(
        regional_release_node_preflight,
        "validate_target_node_state",
        lambda *_args: None,
    )

    def without_nodes(_release: Any, target: Any) -> dict[str, Any] | None:
        asked.append(target.cluster_id)
        return proofs.get(target.cluster_id)

    monkeypatch.setattr(regional_parked_cluster, "without_nodes", without_nodes)

    result = topology.live_node_safety(
        SimpleNamespace(**{"_release": lambda _scenario: release}), "noop"
    )

    assert result["safe"] is True, result
    assert result["shape"][HP] == {"nodes": sorted(HP_NODES)}
    assert result["shape"][EKS] == {
        "zero_nodes": regional_parked_cluster.summary(proofs[EKS])
    }
    assert result["shape"]["gpu-lost"]["zero_nodes"] is None
    assert asked == [EKS, "gpu-lost"], (
        "the proof is read only for a cluster with no node"
    )


class RecordingEngine(ReleaseEngine):
    """The recording engine with the parked cluster's record in its state."""

    def __init__(self, model: Any, config: Any, runner: Any) -> None:
        super().__init__(model, config, runner)
        self.state["zero_node_clusters"] = {
            EKS: {"phase": "upgrade", "release_id": "r"}
        }

    def noop(self, diff: Any) -> None:
        super().noop(diff)
        self.state["zero_node_clusters"] = {
            EKS: {"phase": "upgrade", "release_id": "n"}
        }


class RecordingModel(BackendModel):
    def release(self, config: Any, runner: Any) -> RecordingEngine:
        engine = RecordingEngine(self, config, runner)
        self.engines.append(engine)
        return engine


def test_deploy_returns_the_zero_node_records_it_found_and_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel()
    backend = model.backend()
    monkeypatch.setattr(boot020, "time", model)

    result = backend.deploy("noop", diff={"kind": "NOOP", "changed": []})

    assert result["zero_node_clusters_before"] == {
        EKS: {"phase": "upgrade", "release_id": "r"}
    }
    assert result["zero_node_clusters"] == {
        EKS: {"phase": "upgrade", "release_id": "n"}
    }
    assert model.engines[0].calls[0][0] == "noop", "the deploy engine ran the release"
    assert model.engines[1].calls == [("load-state",)], (
        "a second release read the state"
    )


def test_profile_versions_name_only_the_versions_the_drill_adds() -> None:
    def config(version: str, hp_version: str) -> SimpleNamespace:
        return SimpleNamespace(
            runtime_profile_source=Path("/profile.yaml"),
            runtime_profile_template_source=None,
            runtime_profile_version=version,
            runtime_profile_registration_cluster_id=EKS,
            agent_config_digest="1" * 64,
            clusters=[
                SimpleNamespace(
                    cluster_id=HP,
                    overrides_runtime_profile=True,
                    runtime_profile_source=Path("/hp.yaml"),
                    runtime_profile_template_source=None,
                    runtime_profile_version=hp_version,
                    agent_config_digest="2" * 64,
                ),
                SimpleNamespace(
                    cluster_id=EKS,
                    overrides_runtime_profile=False,
                    runtime_profile_version="",
                    agent_config_digest="",
                ),
            ],
        )

    record = topology.profile_versions_record(
        config("eks-v1-boot020", "hyperpod-v1-boot020"), config("eks-v1", "hyperpod-v1")
    )

    assert record["registered_versions"] == ["eks-v1-boot020", "hyperpod-v1-boot020"]
    assert record["clusters"][HP]["version"] == "hyperpod-v1-boot020"
    assert record["clusters"][EKS] == {
        "version": "eks-v1-boot020",
        "agent_config_digest": "1" * 64,
        "own_profile": False,
    }
    assert record["site_clusters"][HP]["version"] == "hyperpod-v1"
    assert record["permanent"] is True, "registered versions are immutable"


def test_the_entrypoint_binds_the_declared_clusters_and_guards_before_the_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    roundtrips: list[str] = []

    def entry_for(root: Path, site: ParkedSite) -> RollingEntry:
        root.mkdir()
        entry = RollingEntry(root, monkeypatch)
        monkeypatch.setattr(boot020, "LiveReleaseRollingBackend", lambda _configs: site)
        monkeypatch.setattr(
            boot020,
            "public_config_roundtrip",
            lambda *_a, **_k: roundtrips.append(root.name) or {"restored": True},
        )
        return entry

    declared = entry_for(tmp_path / "declared", ParkedSite())
    monkeypatch.setattr(
        sys,
        "argv",
        [*declared.argv, "--zero-node-cluster", EKS, "--zero-node-cluster", EKS],
    )
    assert boot020.main() == 0, "the declared parked site did not pass"
    document = json.loads(declared.path.read_text(encoding="utf-8"))
    assert document["inputs"]["zero_node_clusters"] == [EKS]
    assert document["inputs"]["acceptance_contract"] == 6
    assert declared.calls[0][1]["details"]["zero_node_clusters"] == [EKS]
    assert document["zero_node_clusters"] == [EKS]
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"

    undeclared = entry_for(tmp_path / "undeclared", ParkedSite())
    monkeypatch.setattr(sys, "argv", undeclared.argv)
    with pytest.raises(boot020.AcceptanceCheckError, match="is not declared"):
        boot020.main()
    assert roundtrips == ["declared"], "the public roundtrip ran before the guard"
    refused = json.loads(undeclared.path.read_text(encoding="utf-8"))
    assert refused["verdict"] == "FAIL", refused
