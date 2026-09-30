from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.run_boot020_release_rolling import (
    EXECUTOR_DEPLOYMENT,
    STAGES,
    AcceptanceCheckError,
    LiveReleaseRollingBackend,
    configure_gpu_kubeconfig,
    deployment_generations,
    resume_release_rolling,
    resume_target,
    run_release_rolling,
)

PREVIOUS_EXECUTOR_PIN = "a" * 64
CANDIDATE_EXECUTOR_PIN = "b" * 64

ROOT = Path(__file__).resolve().parents[2]


class FakeReleaseRollingBackend:
    def __init__(self) -> None:
        self.completed = set()
        self.live = {
            "cpu_wheel": "cpu-v1",
            "runtime_profile_version": "profile-v1",
            "clusters": {
                "cluster-a": {
                    "wheel": "executor-v1",
                    "reconciler_wheel": "node-v1",
                    "bundle": "bundle-v1",
                }
            },
        }
        self.cpu_generations = {
            "gpu-fault-api-ha": 1,
            "gpu-fault-control-worker": 1,
            "gpu-fault-telemetry-spool-worker": 1,
        }
        # What a CPU Deployment rollout actually changes; generation alone also
        # moves when the release stamps an annotation (live 2026-09-20).
        self.cpu_identities = {
            name: {"replicas": 3, "template_sha256": f"{name}-template-v1"}
            for name in self.cpu_generations
        }
        self.stamp_ingress_identity = False
        # A previous attempt's automatic rollback died mid-way (live 2026-09-20
        # 08:02Z: the HyperPod software update rebooted every CPU node during the
        # control_plane stage; the engine left phase rollback-failed).
        self.rollback_pending = False
        # Blockers the engine's own pre-node-mutation gate would raise on
        # (live 2026-09-20 08:33Z: the warm spare carried a quarantine taint from
        # an inconclusive DCGM diagnosis and the executor stage failed after
        # ~50 minutes of control-plane work).
        self.unsafe_nodes: list[dict] = []
        # Keyed by the real Deployment names: the executor stage asserts the
        # generation changes are exactly {GPU_EXECUTOR_DEPLOYMENT} per cluster.
        self.gpu_generations = {
            "cluster-a": {
                name: 1
                for name in (
                    EXECUTOR_DEPLOYMENT,
                    "gpu-fault-completion-watcher",
                    "gpu-fault-kubernetes-node-resource-collector",
                    "gpu-fault-node-installer-reconciler",
                )
            }
        }
        self.calls = []
        self.commits: list[str] = []
        # Every live snapshot costs 20-30 s of kubectl reads, so the runner is
        # held to the number it takes, not only to the deploys it issues.
        self.snapshots: list[str] = []
        # The restore stage (see FakeBackend in test_boot020_release_rolling).
        self.initial_live = copy.deepcopy(self.live)
        self.restored = False
        self.cpu_env_drift = False
        self.uncommitted_restore = False

    def cpu_env_alignment(self, scenario: str) -> dict:
        return {"aligned": not self.cpu_env_drift, "deployments": {}}

    def classify(self, scenario: str) -> dict:
        scenario = "noop" if scenario == "restore" else scenario
        if scenario == "noop" and "full" in self.completed and not self.restored:
            return {"kind": "FULL", "changed": ["runtime_profile"]}
        kind = (
            "NOOP"
            if scenario in self.completed
            else {
                "noop": "NOOP",
                "control_plane": "CONTROL_PLANE_ONLY",
                "executor": "DATA_PLANE_COMPATIBLE",
                "agent": "DATA_PLANE_COMPATIBLE",
                "full": "FULL",
            }[scenario]
        )
        fields = {
            "control_plane": "admin_config_worker",
            "executor": "executor_wheel",
            "agent": "node_bundle",
            "full": "runtime_profile",
        }
        return {"kind": kind, "changed": [] if kind == "NOOP" else [fields[scenario]]}

    def snapshot(self, scenario: str, *, live: bool = True) -> dict:
        self.snapshots.append(scenario)
        self.light_snapshots = getattr(self, "light_snapshots", []) + (
            [] if live else [scenario]
        )
        return {
            "phase": "complete",
            "release_id": scenario,
            "live": copy.deepcopy(self.live),
            "cpu_generations": dict(self.cpu_generations),
            "cpu_identities": copy.deepcopy(self.cpu_identities),
            "gpu_generations": copy.deepcopy(self.gpu_generations),
            "next_deploy": self.classify(scenario),
        }

    @staticmethod
    def _pins(phase: str) -> dict:
        """The two-phase executor pin window the runner asserts at each checkpoint."""

        return {
            "rolled-back": {"required": PREVIOUS_EXECUTOR_PIN, "compatible": []},
            "staged": {
                "required": PREVIOUS_EXECUTOR_PIN,
                "compatible": [CANDIDATE_EXECUTOR_PIN],
            },
            "finalized": {"required": CANDIDATE_EXECUTOR_PIN, "compatible": []},
        }[phase] | {
            "candidate": CANDIDATE_EXECUTOR_PIN,
            "previous_required": PREVIOUS_EXECUTOR_PIN,
        }

    def node_safety(self, scenario: str) -> dict:
        # Kept out of ``calls``: those are the deploys the site saw.
        self.safety_checks = getattr(self, "safety_checks", []) + [scenario]
        return {
            "safe": not self.unsafe_nodes,
            "clusters": {"cluster-a": list(self.unsafe_nodes)},
        }

    def resume_rollback(self, scenario: str) -> dict:
        self.calls.append(("resume_rollback", scenario))
        if not self.rollback_pending:
            return {"resumed": False, "phase": "complete"}
        self.rollback_pending = False
        return {
            "resumed": True,
            "phase_before": "rollback-failed",
            "phase": "rolled-back",
        }

    def deploy(
        self,
        scenario: str,
        *,
        diff: dict,
        fault_phase=None,
        resume=False,
        auto_rollback=None,
        commit=False,
    ) -> dict:
        if commit:
            self.commits.append(scenario)
        if self.rollback_pending:
            raise RuntimeError("a rollback is in progress and must finish first")
        self.calls.append((scenario, fault_phase, resume, auto_rollback, diff["kind"]))
        plans = {
            "control_plane": {
                "clusters": {},
                "global_components": ["cpu_stage", "cpu_finalize"],
                "restores_data_plane": False,
                "needs_controller": False,
            },
            "executor": {
                "clusters": {
                    "cluster-a": ["collector", "executor", "reconciler", "watcher"]
                },
                "global_components": [],
                "restores_data_plane": True,
                "needs_controller": False,
            },
            "agent": {
                "clusters": {"cluster-a": ["reconciler", "agent"]},
                "global_components": [],
                "restores_data_plane": True,
                "needs_controller": True,
            },
            "full": {
                "clusters": {"cluster-a": ["executor", "reconciler", "agent"]},
                "global_components": ["cpu_stage"],
                "restores_data_plane": True,
                "needs_controller": True,
            },
        }
        if fault_phase:
            if auto_rollback:
                # A CPU rollback restores EVERY CPU role from the snapshot with a
                # fresh restart token (live 2026-09-20 07:32Z: api-ha and spool got
                # new ReplicaSets during a worker-only rollback), so identities
                # and generations of untouched roles move before the real apply.
                self.cpu_generations = {
                    name: value + 2 for name, value in self.cpu_generations.items()
                }
                for name, identity in self.cpu_identities.items():
                    identity["template_sha256"] = f"{name}-template-restored"
                return {
                    "phase": "rolled-back",
                    "injected_failure": fault_phase,
                    "rollback_plan": plans[scenario],
                    "rollback_timing": {"t_safe_seconds": 30.0, "t_full_seconds": 45.0},
                    "pins": self._pins("rolled-back"),
                    "operation_duration_seconds": 45.0,
                }
            return {
                "phase": "failed",
                "injected_failure": fault_phase,
                "pins": self._pins("staged"),
                "operation_duration_seconds": 1.0,
            }
        if scenario == "control_plane":
            self.live["cpu_wheel"] = "cpu-v2"
            # The worker rolls (replicas + template); the admin-config stamp bumps
            # every CPU Deployment's generation without a rollout.
            self.cpu_generations = {
                name: value + 1 for name, value in self.cpu_generations.items()
            }
            self.cpu_identities["gpu-fault-control-worker"] = {
                "replicas": 2,
                "template_sha256": "gpu-fault-control-worker-template-v2",
            }
            if self.stamp_ingress_identity:
                self.cpu_identities["gpu-fault-api-ha"]["replicas"] = 4
        elif scenario == "executor":
            # The interrupted attempt already staged the CPU side; the resume
            # rolls the executor Deployment and then finalizes the pins, which
            # is the one further CPU rollout (cpu-finalized) every release makes.
            self.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
            self.gpu_generations["cluster-a"] = dict.fromkeys(
                self.gpu_generations["cluster-a"], 2
            )
            if resume:
                self.cpu_generations = {
                    name: value + 1 for name, value in self.cpu_generations.items()
                }
        elif scenario == "agent":
            self.live["clusters"]["cluster-a"]["reconciler_wheel"] = "node-v2"
            self.live["clusters"]["cluster-a"]["bundle"] = "bundle-v2"
            self.gpu_generations["cluster-a"]["gpu-fault-completion-watcher"] = 2
        elif scenario == "full":
            self.live = {
                "cpu_wheel": "cpu-v3",
                "runtime_profile_version": "profile-v2",
                "clusters": {
                    "cluster-a": {
                        "wheel": "executor-v3",
                        "reconciler_wheel": "node-v2",
                        "bundle": "bundle-v2",
                    }
                },
            }
            self.cpu_generations = dict.fromkeys(self.cpu_generations, 3)
            self.gpu_generations = {
                "cluster-a": {EXECUTOR_DEPLOYMENT: 3, "gpu-fault-completion-watcher": 3}
            }
        elif scenario == "noop" and "full" in self.completed:
            # The restore: the site is back on its own release, every CPU role
            # rolled once more (a profile change restarts the CPU Pods).
            self.restored = True
            self.live = copy.deepcopy(self.initial_live)
            self.cpu_generations = dict.fromkeys(self.cpu_generations, 4)
        self.completed.add(scenario)
        return {
            "phase": "complete",
            # The engine leaves a release uncommitted; only a requested commit
            # (the restore stage asks for it) makes it a committed transaction.
            "transaction_committed": not (
                scenario == "noop"
                and self.restored
                and (self.uncommitted_restore or not commit)
            ),
            "injected_failure": None,
            "pins": self._pins("finalized"),
            "operation_duration_seconds": 10.0 if resume else 20.0,
        }


def test_control_plane_stage_reads_rollouts_from_identity_not_generation(
    tmp_path: Path,
) -> None:
    # Live 2026-09-20 (BOOT-020 a3): the admin-config stamp moved api-ha and spool
    # generations; only a replicas or template change is a rollout.
    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    backend = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(evidence, case_id="GF-REGIONAL-BOOT-020", inputs={})
    result = run_release_rolling(backend, recorder)
    assert result["verdict"] == "PASS"

    damaged = FakeReleaseRollingBackend()
    damaged.stamp_ingress_identity = True
    recorder = EvidenceRecorder(
        tmp_path / "damaged.json", case_id="GF-REGIONAL-BOOT-020", inputs={}
    )
    with pytest.raises(AcceptanceCheckError, match="CPU role rollout differs"):
        run_release_rolling(damaged, recorder)


def test_boot020_runner_covers_diff_resume_and_rollback(tmp_path: Path) -> None:
    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    backend = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = run_release_rolling(backend, recorder)

    assert result["status"] == "COMPLETED"
    assert backend.calls == [
        ("noop", None, False, None, "NOOP"),
        ("control_plane", "cpu-finalized", False, True, "CONTROL_PLANE_ONLY"),
        ("control_plane", None, False, None, "CONTROL_PLANE_ONLY"),
        ("executor", "data-converged", False, True, "DATA_PLANE_COMPATIBLE"),
        ("executor", "cpu-staged", False, False, "DATA_PLANE_COMPATIBLE"),
        ("executor", None, True, False, "DATA_PLANE_COMPATIBLE"),
        ("agent", "data-converged", False, True, "DATA_PLANE_COMPATIBLE"),
        ("agent", None, False, None, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
        # The restore stage gives the site back its own config as a committed
        # release (the noop candidate classifies FULL against candidate D).
        ("noop", None, False, None, "FULL"),
    ]
    assert (
        result["stages"]["full_rollback_snapshot"]["live"]
        == (result["stages"]["full_before"]["live"])
    )
    stages = result["stages"]
    # Executor stage: the two-phase pin window at each checkpoint, only the
    # executor Deployment rolled, and the CPU generation held across the resume.
    assert stages["executor_injected_failure_and_rollback"]["pins"]["required"] == (
        PREVIOUS_EXECUTOR_PIN
    )
    assert stages["executor_interrupted_failure"]["pins"]["compatible"] == [
        CANDIDATE_EXECUTOR_PIN
    ]
    assert stages["executor_resumed"]["pins"] == {
        "required": CANDIDATE_EXECUTOR_PIN,
        "compatible": [],
        "candidate": CANDIDATE_EXECUTOR_PIN,
        "previous_required": PREVIOUS_EXECUTOR_PIN,
    }
    interrupted = stages["executor_interrupted_snapshot"]["cpu_generations"]
    after = stages["executor_after"]["cpu_generations"]
    assert {name: after[name] - interrupted[name] for name in after} == {
        name: 1 for name in after
    }, "the resume finalizes the pins (one CPU roll) and never repeats the stage"
    # Mid-transaction the two planes run different runtime images and the
    # engine's previous-release capture refuses the mixed fleet (live
    # 2026-09-11), so the interrupted snapshot must be the light form.
    assert backend.light_snapshots == ["executor"], backend.light_snapshots
    before = stages["executor_before"]["gpu_generations"]["cluster-a"]
    after = stages["executor_after"]["gpu_generations"]["cluster-a"]
    assert {name for name in after if after[name] != before[name]} == {
        EXECUTOR_DEPLOYMENT,
        "gpu-fault-completion-watcher",
        "gpu-fault-kubernetes-node-resource-collector",
        "gpu-fault-node-installer-reconciler",
    }
    assert stages["control_plane_injected_failure_and_rollback"]["rollback_plan"][
        "global_components"
    ] == ["cpu_stage", "cpu_finalize"]
    assert evidence.stat().st_mode & 0o777 == 0o600
    # A stage's ``*_after`` and the next stage's ``*_before`` observe the same
    # live state -- only read-only classifications run between them -- so the
    # runner takes the ``*_after`` once and reuses it, marked, as the next
    # ``*_before``. ``noop_before`` has no predecessor and stays fresh.
    # The executor stage takes one extra mid-stage snapshot
    # (``executor_interrupted_snapshot``) so the CPU generations can be held
    # across the resume; like the ``*_rollback_snapshot`` reads it is never a
    # stage boundary and is not reused.
    assert backend.snapshots == [
        "noop",
        "noop",
        "control_plane",
        "control_plane",
        "executor",
        "executor",
        "executor",
        "agent",
        "agent",
        "full",
        "full",
        # ``restore_before`` reuses ``full_after``; one fresh ``restore_after``.
        "restore",
    ], backend.snapshots
    stages = result["stages"]
    assert "reused_from" not in stages["noop_before"]
    for previous, stage in zip(STAGES, STAGES[1:]):
        before = stages[f"{stage}_before"]
        assert before["reused_from"] == f"{previous}_after", (stage, before)
        assert before["live"] == stages[f"{previous}_after"]["live"], stage
        assert (
            before["cpu_generations"] == stages[f"{previous}_after"]["cpu_generations"]
        )
        assert (
            before["gpu_generations"] == stages[f"{previous}_after"]["gpu_generations"]
        )
        # ``next_deploy`` is the one config-dependent field; a reused snapshot
        # reports the new stage's own classification, which is what a fresh
        # ``next_deploy`` computes for a committed ``complete`` release.
        assert before["next_deploy"] == stages[f"{stage}_classification"], stage
        assert "reused_from" not in stages[f"{stage}_after"]
        if stage != "restore":  # the restore is a plain release: no rollback drill
            assert "reused_from" not in stages[f"{stage}_rollback_snapshot"]


def test_boot020_replayed_after_snapshot_is_never_reused(tmp_path: Path) -> None:
    """Only a ``*_after`` taken fresh in this process may seed the next
    ``*_before``: one replayed from evidence describes an older observation."""

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(FakeReleaseRollingBackend(), recorder)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in [key for key in document["stages"] if key.startswith("full")]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor", "agent"})
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = run_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED"
    # ``agent_after`` replayed, so ``full_before`` was observed afresh.
    assert second.snapshots == ["full", "full", "full"], second.snapshots
    assert "reused_from" not in result["stages"]["full_before"]


CPU_ARGS = ["kubectl", "--kubeconfig", "/cpu.kubeconfig"]


def _context_args(context: str) -> list[str]:
    return ["kubectl", "--context", context]


class FakeRelease:
    """A release whose ``_get_json`` answers ``get deployment`` lists per
    kube context and records every argv it was asked for.

    ``load_state`` and ``capture_previous`` are what the engine's own
    ``_load_state``/``_capture_previous`` answer; the fake records that they
    were called inside its read snapshot.
    """

    def __init__(
        self, listings: dict[str, list[dict]], *, load_state=None, capture_previous=None
    ) -> None:
        self.config = SimpleNamespace(
            namespace="gpu-fault",
            cpu_kubeconfig="/cpu.kubeconfig",
            clusters=[
                SimpleNamespace(cluster_id="cluster-a", context="ctx-a"),
                SimpleNamespace(cluster_id="cluster-b", context="ctx-b"),
            ],
        )
        self.listings = listings
        self.load_state = load_state
        self.capture_previous = capture_previous
        self.reads: list[list[str]] = []
        self.calls: list[str] = []
        self.snapshot_depth = 0
        self.reads_inside_snapshot = 0
        self.state: dict | None = None

    def _cpu(self, *args: str) -> list[str]:
        return [*CPU_ARGS, *args]

    def _gpu(self, target, *args: str) -> list[str]:
        return [*_context_args(target.context), *args]

    def _get_json(self, args: list[str]) -> dict:
        self.reads.append(list(args))
        if self.snapshot_depth:
            self.reads_inside_snapshot += 1
        assert args[-2:] == ["get", "deployment"], args
        context = args[2]
        return {"items": self.listings[context]}

    @contextlib.contextmanager
    def _read_snapshot(self):
        self.snapshot_depth += 1
        try:
            yield
        finally:
            self.snapshot_depth -= 1

    def _load_state(self) -> dict:
        self.calls.append("load_state")
        assert self.snapshot_depth == 1
        self.state = self.load_state()
        return self.state

    def _capture_previous(self) -> dict:
        self.calls.append("capture_previous")
        assert self.snapshot_depth == 1
        return self.capture_previous()


def _deployment_item(name: str, generation: int) -> dict:
    # The live listing always carries replicas and a Pod template; the snapshot
    # reads both for the rollout identity next to the generation.
    return {
        "metadata": {"name": name, "generation": generation},
        "spec": {"replicas": 1, "template": {"metadata": {"labels": {"app": name}}}},
    }


def test_boot020_generations_come_from_one_list_per_context() -> None:
    release = FakeRelease(
        {
            "/cpu.kubeconfig": [
                _deployment_item("gpu-fault-api-ha", 7),
                _deployment_item("gpu-fault-control-worker", 3),
                _deployment_item("unrelated", 99),
            ],
            "ctx-a": [_deployment_item("gpu-fault-cluster-action-executor", 5)],
        }
    )

    cpu = deployment_generations(
        release, CPU_ARGS, ("gpu-fault-api-ha", "gpu-fault-control-worker")
    )
    gpu = deployment_generations(
        release,
        _context_args(release.config.clusters[0].context),
        ("gpu-fault-cluster-action-executor",),
    )

    assert cpu == {"gpu-fault-api-ha": 7, "gpu-fault-control-worker": 3}
    assert gpu == {"gpu-fault-cluster-action-executor": 5}
    # Exactly one list read per kube context, with the argv the engine's own
    # `prime_deployment_snapshot` uses so a warm read cache answers it.
    assert release.reads == [
        [
            "kubectl",
            "--kubeconfig",
            "/cpu.kubeconfig",
            "-n",
            "gpu-fault",
            "get",
            "deployment",
        ],
        ["kubectl", "--context", "ctx-a", "-n", "gpu-fault", "get", "deployment"],
    ]


def test_boot020_live_snapshot_reads_once_per_context_inside_one_snapshot(
    monkeypatch,
) -> None:
    """One ``snapshot()`` loads the state once, captures the previous release
    once, derives ``next_deploy`` from the engine's own ``next_deploy`` and
    lists Deployments once per context, all inside one read snapshot."""

    from gpu_fault_release import regional_deployment_inventory as inventory

    cpu_items = [
        _deployment_item(name, index + 1)
        for index, name in enumerate(inventory.CPU_RUNTIME_DEPLOYMENTS)
    ]
    gpu_items = [
        _deployment_item(name, index + 10)
        for index, name in enumerate(
            (*inventory.DEPLOYMENTS, inventory.GPU_RECONCILER_DEPLOYMENT)
        )
    ]
    state = {"phase": "complete", "release_id": "rel-1", "previous": {}}
    release = FakeRelease(
        {"/cpu.kubeconfig": cpu_items, "ctx-a": gpu_items, "ctx-b": gpu_items},
        load_state=lambda: state,
        capture_previous=lambda: {
            "release_id": "rel-1",
            "clusters": {"cluster-a": {}, "cluster-b": {}},
        },
    )

    backend = LiveReleaseRollingBackend({"noop": Path("/noop.json")})
    monkeypatch.setattr(backend, "_release", lambda scenario, **_: release)

    def _next_deploy(given_release, given_state) -> dict:
        release.calls.append("next_deploy")
        assert given_release is release and given_state is state
        assert release.snapshot_depth == 1
        return {"kind": "NOOP", "changed": []}

    monkeypatch.setattr(backend, "commands", SimpleNamespace(next_deploy=_next_deploy))

    snapshot = backend.snapshot("noop")

    assert release.calls == ["load_state", "capture_previous", "next_deploy"]
    assert snapshot == {
        "phase": "complete",
        "release_id": "rel-1",
        "live": {"release_id": "rel-1", "clusters": {"cluster-a": {}, "cluster-b": {}}},
        "cpu_generations": {
            name: index + 1
            for index, name in enumerate(inventory.CPU_RUNTIME_DEPLOYMENTS)
        },
        "cpu_identities": {
            name: {
                "replicas": 1,
                "template_sha256": hashlib.sha256(
                    json.dumps(
                        {"metadata": {"labels": {"app": name}}},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
            for name in inventory.CPU_RUNTIME_DEPLOYMENTS
        },
        "gpu_generations": {
            "cluster-a": {
                name: index + 10
                for index, name in enumerate(
                    (*inventory.DEPLOYMENTS, inventory.GPU_RECONCILER_DEPLOYMENT)
                )
            },
            "cluster-b": {
                name: index + 10
                for index, name in enumerate(
                    (*inventory.DEPLOYMENTS, inventory.GPU_RECONCILER_DEPLOYMENT)
                )
            },
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
    }
    # One list per context (CPU, cluster-a, cluster-b), nothing per Deployment,
    # and every read inside the snapshot the whole call shares.
    assert len(release.reads) == 3, release.reads
    assert release.reads_inside_snapshot == 3
    assert release.snapshot_depth == 0


def test_boot020_live_snapshot_keeps_next_deploy_none_when_it_fails(
    monkeypatch,
) -> None:
    """``build_release_summary`` swallowed a ``next_deploy`` failure into
    ``next_deploy_error`` and the snapshot recorded ``None``; the direct call
    keeps that recorded value."""

    from gpu_fault_release import regional_deployment_inventory as inventory

    cpu_items = [
        _deployment_item(name, 1) for name in inventory.CPU_RUNTIME_DEPLOYMENTS
    ]
    gpu_items = [
        _deployment_item(name, 1)
        for name in (*inventory.DEPLOYMENTS, inventory.GPU_RECONCILER_DEPLOYMENT)
    ]
    release = FakeRelease(
        {"/cpu.kubeconfig": cpu_items, "ctx-a": gpu_items, "ctx-b": gpu_items},
        load_state=lambda: {"phase": "complete", "release_id": "rel-1"},
        capture_previous=lambda: {"clusters": {}},
    )
    backend = LiveReleaseRollingBackend({"noop": Path("/noop.json")})
    monkeypatch.setattr(backend, "_release", lambda scenario, **_: release)

    def _next_deploy(*_):
        raise RuntimeError("classification unavailable")

    monkeypatch.setattr(backend, "commands", SimpleNamespace(next_deploy=_next_deploy))

    assert backend.snapshot("noop")["next_deploy"] is None


def test_boot020_configures_explicit_gpu_kubeconfig(
    tmp_path: Path, monkeypatch
) -> None:
    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    monkeypatch.delenv("KUBECONFIG", raising=False)

    result = configure_gpu_kubeconfig(kubeconfig)

    assert result == kubeconfig.resolve()
    assert os.environ["KUBECONFIG"] == str(kubeconfig.resolve())


def test_boot020_runner_resumes_at_a_later_stage(tmp_path: Path) -> None:
    """A rerun over the same evidence must not touch the site for stages that
    already ran; --start-stage skips them and records where it restarted."""

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    first = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(first, recorder)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in (
        "agent_apply_after_rollback",
        "agent_after",
        "agent_next_classification",
    ):
        document["stages"].pop(name)
    for name in [key for key in document["stages"] if key.startswith("full")]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor"})
    second.live["cpu_wheel"] = "cpu-v2"
    second.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = run_release_rolling(second, resumed, start_stage="agent")

    assert result["status"] == "COMPLETED", result["status"]
    assert result["stages"]["resumed_at_agent"]["skipped_stages"] == [
        "noop",
        "control_plane",
        "executor",
    ], result["stages"]["resumed_at_agent"]
    # Only the missing agent apply and the whole full stage touched the site.
    assert second.calls == [
        ("agent", None, False, None, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
    ], second.calls
    # ``agent_after`` was taken fresh here, so it seeds ``full_before``.
    assert second.snapshots == ["agent", "full", "full"], second.snapshots
    assert result["stages"]["full_before"]["reused_from"] == "agent_after"


def test_boot020_runner_refuses_to_resume_without_earlier_evidence(
    tmp_path: Path,
) -> None:
    recorder = EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={"configs": "test"},
    )
    with pytest.raises(RuntimeError, match="earlier stages have no recorded result"):
        run_release_rolling(FakeReleaseRollingBackend(), recorder, start_stage="agent")


def test_boot020_resume_target_is_the_first_stage_without_a_passed_marker() -> None:
    # A stage only carries its ``_passed`` marker once every assertion in it
    # ran, so the marker -- not a mid-stage record -- is what proves a stage is
    # done and can be replayed.
    partial = {
        "stages": {
            "noop_passed": {},
            "control_plane_passed": {},
            "executor_classification": {},
            "executor_before": {},
        }
    }
    assert resume_target(partial) == (2, "executor")
    done = {"stages": {f"{stage}_passed": {} for stage in STAGES}}
    assert resume_target(done) == (len(STAGES), None)
    assert resume_target({"stages": {}}) == (0, "noop")


def test_boot020_refuses_to_start_while_a_gpu_node_blocks_data_plane_mutation(
    tmp_path: Path,
) -> None:
    """Live 2026-09-20 08:33Z: the warm spare carried gpu-fault.io/quarantined
    (inconclusive DCGM diagnosis at 07:35Z); the engine's pre-node-mutation gate
    refused the executor stage after the drill, noop and control_plane stages had
    already spent ~50 minutes. The same gate must run before the first stage."""
    backend = FakeReleaseRollingBackend()
    backend.unsafe_nodes = [
        {
            "node_id": "spare-1",
            "reason": "blocking-taint",
            "taint_key": "gpu-fault.io/quarantined",
        }
    ]
    recorder = EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={},
    )
    with pytest.raises(AcceptanceCheckError, match="spare-1.*quarantined"):
        run_release_rolling(backend, recorder)
    assert backend.calls == [] and backend.safety_checks == ["noop"], (
        "the site was touched before the gate"
    )
    assert recorder.document["node_safety"]["safe"] is False

    with pytest.raises(AcceptanceCheckError, match="spare-1"):
        resume_release_rolling(backend, recorder)

    backend.unsafe_nodes = []
    healthy = EvidenceRecorder(
        tmp_path / "ok.json", case_id="GF-REGIONAL-BOOT-020", inputs={}
    )
    assert run_release_rolling(backend, healthy)["verdict"] == "PASS"
    assert healthy.document["node_safety"]["safe"] is True


def test_boot020_auto_resume_first_resumes_a_failed_rollback(tmp_path: Path) -> None:
    """Live 2026-09-20 08:02Z: a HyperPod software update rebooted every CPU node
    inside the control_plane stage; the injected upgrade failed for real and the
    automatic rollback died on an unavailable webhook, leaving phase
    rollback-failed. The engine refuses a new transaction over a pending rollback,
    so the convergence must resume the rollback first, and say so in the evidence."""
    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    first = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(evidence, case_id="GF-REGIONAL-BOOT-020", inputs={})
    run_release_rolling(first, recorder)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in [
        key
        for key in document["stages"]
        if key.startswith("control_plane")
        or key.startswith("executor")
        or key.startswith("agent")
        or key.startswith("full")
    ]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    second = FakeReleaseRollingBackend()
    second.rollback_pending = True
    resumed = EvidenceRecorder(evidence, case_id="GF-REGIONAL-BOOT-020", inputs={})
    result = resume_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED", result.get("error")
    convergence = result["convergence"][-1]
    assert convergence["converged_to"] == "noop"
    assert convergence["rollback_resume"]["resumed"] is True
    assert second.calls[:2] == [
        ("resume_rollback", "noop"),
        ("noop", None, False, None, "NOOP"),
    ], "the pending rollback was not resumed before the convergence deploy"


def test_boot020_auto_resume_reconverges_and_replays_passed_stages(
    tmp_path: Path,
) -> None:
    """--resume reads the evidence, converges the live state back to the failed
    stage's precondition, and restarts only that stage -- the semantics are
    unchanged, only the cost of a driver defect is."""

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    first = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(first, recorder)

    # A driver assertion fails partway through the agent stage: it never reached
    # its end-of-stage marker and everything from it onward is stale.
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in [
        key
        for key in document["stages"]
        if key == "agent" or key.startswith("agent_") or key.startswith("full")
    ]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    # A fresh backend at the executor precondition -- what the agent stage's
    # auto-rollback left live -- reopens the same evidence.
    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor"})
    second.live["cpu_wheel"] = "cpu-v2"
    second.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
    second.gpu_generations["cluster-a"]["executor"] = 2
    second.cpu_generations = {"ingress": 2, "worker": 2}
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = resume_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED", result.get("status")
    assert result["stages"]["resumed_at_agent"]["skipped_stages"] == [
        "noop",
        "control_plane",
        "executor",
    ]
    # The convergence deployed the executor precondition once, and because the
    # live state already matched it, as a NOOP.
    assert result["convergence"][-1]["converged_to"] == "executor"
    assert [call for call in second.calls if call[0] == "executor"] == [
        ("executor", None, False, None, "NOOP")
    ]
    # The passed stages were replayed, never re-run against the site.
    assert not any(
        call[0] == "control_plane" or (call[0] == "noop" and call[-1] == "NOOP")
        for call in second.calls
    ), "a passed stage was re-run against the site instead of replayed"
    # The agent stage ran again from its injected-failure classification, then
    # the full stage ran.
    assert second.calls == [
        # The convergence first asks the engine to finish any pending rollback
        # (nothing pending here), then deploys the executor precondition.
        ("resume_rollback", "executor"),
        ("executor", None, False, None, "NOOP"),
        ("agent", "data-converged", False, True, "DATA_PLANE_COMPATIBLE"),
        ("agent", None, False, None, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
        ("noop", None, False, None, "FULL"),
    ], second.calls
    # The resumed stage's ``before`` is observed afresh -- the convergence
    # deploy sits between it and any earlier snapshot -- and only the in-process
    # ``agent_after`` is reused for ``full_before`` (and ``full_after`` for
    # ``restore_before``).
    assert second.snapshots == ["agent", "agent", "agent", "full", "full", "restore"], (
        second.snapshots
    )
    assert "reused_from" not in result["stages"]["agent_before"]
    assert result["stages"]["full_before"]["reused_from"] == "agent_after"


def test_boot020_auto_resume_over_completed_evidence_touches_nothing(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(FakeReleaseRollingBackend(), recorder)

    second = FakeReleaseRollingBackend()
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = resume_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED"
    assert second.calls == [], "a completed run must not touch the site to resume"


def test_boot020_start_stage_cannot_skip_an_unpassed_terminal_assertion(
    tmp_path: Path,
) -> None:
    recorder = EvidenceRecorder(
        tmp_path / "case.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={"configs": "test"},
    )
    recorder.stage("noop_after", lambda: {"live": "observed-but-not-validated"})
    backend = FakeReleaseRollingBackend()

    with pytest.raises(RuntimeError, match="earlier stages"):
        run_release_rolling(backend, recorder, start_stage="control_plane")

    assert backend.calls == [], "a snapshot without its PASS marker authorizes nothing"


def test_boot020_restore_stage_returns_the_site_to_its_own_committed_release(
    tmp_path: Path,
) -> None:
    """Live 2026-09-26: BOOT-020 left the site on candidate D (uncommitted, Pods on
    the candidate's start-up env) and the operator's restore deploy failed. The
    runner now restores the site itself and proves the end state."""

    backend = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={},
    )
    result = run_release_rolling(backend, recorder)

    stages = result["stages"]
    assert stages["restore_classification"]["kind"] == "FULL", (
        "the site config is a real change against candidate D"
    )
    assert stages["restore_apply"]["phase"] == "complete", stages["restore_apply"]
    assert stages["restore_apply"]["transaction_committed"] is True, (
        "the restore is a committed release, not a candidate left open"
    )
    assert backend.commits == ["noop"], (
        "only the restore asks the engine to commit; candidates stay open"
    )
    assert stages["restore_next_classification"]["kind"] == "NOOP", stages[
        "restore_next_classification"
    ]
    assert stages["restore_cpu_env_alignment"]["aligned"] is True, stages[
        "restore_cpu_env_alignment"
    ]
    assert stages["restore_after"]["live"] == backend.initial_live, (
        "the live state is back where the chain started"
    )
    assert result["release_id"] == "restore", "the final identity is the restored site"
    assert "restore_passed" in stages, "the restore stage carries its pass marker"


@pytest.mark.parametrize("defect", ["cpu_env_drift", "uncommitted_restore"])
def test_boot020_restore_stage_refuses_a_site_that_is_not_fully_restored(
    tmp_path: Path, defect: str
) -> None:
    backend = FakeReleaseRollingBackend()
    setattr(backend, defect, True)
    recorder = EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={},
    )
    expected = (
        "start-up environment" if defect == "cpu_env_drift" else "committed transaction"
    )
    with pytest.raises(AcceptanceCheckError, match=expected):
        run_release_rolling(backend, recorder)
    saved = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert saved["verdict"] == "FAIL", saved
    assert "restore_passed" not in saved["stages"], "no pass marker on a refusal"
    assert "full_passed" in saved["stages"], "the earlier stages keep their pass"


def test_boot020_spec_no_longer_owes_a_restore_deploy_after_the_case() -> None:
    doc = (ROOT / "docs" / "区域模式端到端验收测试用例.md").read_text(encoding="utf-8")
    start = doc.index("### GF-REGIONAL-BOOT-020")
    section = doc[start : doc.index("### GF-REGIONAL-BOOT-021", start)]
    assert "restore_next_classification" in section, "the spec names the NOOP re-check"
    assert "restore_cpu_env_alignment" in section, "the spec names the Pod env check"
    assert "恢复部署" not in section, "the runner restores the site; no deploy is owed"
    assert "自己恢复" in section, "the spec says the runner restores the site itself"
