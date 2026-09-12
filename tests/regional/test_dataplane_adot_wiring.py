"""The data-plane ADOT collector is wired into the release engine (F10).

F7 rendered the per-GPU-cluster collector (``deploy/dataplane/adot-dataplane.yaml``)
and gave the release ``apply_gpu_adot_collector``; nothing called it. These tests
pin the wiring: the collector is applied wherever the DCGM exporter is applied,
its inputs move the observability digest so a brownfield site that adds the IRSA
role later gets a release that applies it, ``remove-cluster`` scales it down,
and an automatic rollback compensates it -- from a per-cluster snapshot
(``test_dataplane_adot_compensation.py``), or, for a state captured before that
snapshot existed, by re-rendering with the previous ADOT image (pinned here).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_dataplane_observability as DATAPLANE
from gpu_fault_release import regional_gpu_bootstrap as BOOTSTRAP
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_gpu_rollout as GPU_ROLLOUT
from gpu_fault_release import regional_release_mutation_preflight as PREFLIGHT
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_rendering import DATAPLANE_ADOT_DEPLOYMENT
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-adot-writer"
TARGET = SimpleNamespace(cluster_id="gpu-a")
Component = DIFF.ReleaseComponent


class _Runner:
    dry_run = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.probes: list[list[str]] = []

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        self.calls.append((list(arguments), kwargs))
        return ""

    def probe(self, arguments: list[str], **_kwargs: Any) -> bool:
        # No Deployment on the cluster: the skip branch's scale-down probe
        # answers "absent" and nothing is scaled.
        self.probes.append(list(arguments))
        return False


def _config(
    tmp_path: Path,
    *,
    adot_irsa_role_arn: str | None = None,
    amp_workspace_id: str | None = "ws-a",
    name: str = "release.json",
) -> Path:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    if adot_irsa_role_arn:
        value["clusters"][0]["adot_irsa_role_arn"] = adot_irsa_role_arn
    if amp_workspace_id:
        value["health"] = {"amp_workspace_id": amp_workspace_id}
    target = tmp_path / name
    target.write_text(json.dumps(value))
    return target


def _release(path: Path) -> Any:
    return MODULE.RegionalRelease(MODULE.ReleaseConfig.load(path), _Runner())


# --- the seven hooks -------------------------------------------------------


def test_the_release_class_binds_the_collector_apply_and_preflight() -> None:
    """The engine reaches the collector through the same seam as DCGM.

    Class attributes are what the phase tests stub, so a hook that imported the
    function directly would be untestable and, worse, unnoticed if it were
    dropped.
    """
    assert (
        MODULE.RegionalRelease._apply_gpu_adot_collector
        is BOOTSTRAP.apply_gpu_adot_collector
    )
    assert (
        MODULE.RegionalRelease._preflight_gpu_adot_collector
        is BOOTSTRAP.preflight_gpu_adot_collector
    )


def _gpu_recorder(calls: list[str]) -> dict[str, Any]:
    return {
        "executor_wheel_cm": "wheel",
        "bundle_cm": "bundle",
        "node_wheel_sha": "a" * 64,
        "config": SimpleNamespace(agent_config_digest="c" * 64),
        "_ensure_gpu_namespace": lambda _target: calls.append("namespace"),
        "_ensure_connection_secret": lambda _target: calls.append("secret"),
        "_quiesce_gpu_executor": lambda _target: calls.append("quiesce"),
        "_verify_gpu_control_plane_endpoint": lambda _target: calls.append("endpoint"),
        "_apply_gpu_dcgm_exporter": lambda _target, **_kwargs: calls.append("dcgm"),
        "_apply_gpu_adot_collector": lambda _target, **_kwargs: calls.append("adot"),
        "_apply_gpu_deployments": lambda *_args, **_kwargs: calls.append("executor"),
        "_roll_node_runtime": lambda _target, **_kwargs: calls.append("node-runtime"),
        "_apply_observability": lambda **_kwargs: calls.append("observability"),
    }


def _assert_collector_is_staged_with_dcgm_before_the_executor(calls: list[str]) -> None:
    # The gate, DCGM and the collector run as one stage
    # (``regional_release_gpu_stage``); only their membership is fixed.
    assert calls[:3] == ["namespace", "secret", "quiesce"], calls
    assert set(calls[3:6]) == {"endpoint", "dcgm", "adot"}, calls
    assert calls[6:8] == ["executor", "node-runtime"], calls


def test_bootstrap_applies_the_collector_right_after_dcgm() -> None:
    """A bootstrapping cluster gets its scrape path before the Executor rolls.

    The collector needs nothing from the Executor and the Executor's first
    faults are the ones the watcher alerts exist to catch, so the collector goes
    up first -- exactly where DCGM does, for the same reason.
    """
    calls: list[str] = []

    ORCHESTRATION.bootstrap_gpu_target(SimpleNamespace(**_gpu_recorder(calls)), TARGET)

    _assert_collector_is_staged_with_dcgm_before_the_executor(calls)
    assert len(calls) == 8, calls


def test_join_applies_the_collector_right_after_dcgm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    for name in ("prepare_join_registry", "ensure_runtime_profile"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "join_target", lambda _self, _cluster_id: TARGET)
    release = SimpleNamespace(
        **_gpu_recorder(calls),
        bundle_sha="b" * 64,
        executor_wheel_sha="e" * 64,
        _ensure_contexts=lambda: None,
        _update_registry=lambda _target, *, remove: None,
        _gpu=lambda _target, *arguments: ["kubectl", *arguments],
        _upload_config_map=lambda *_args, **_kwargs: None,
        _config_map_data=lambda _name: {
            "required-agent-artifact-sha256": "a" * 64,
            "required-regional-executor-artifact-sha256": "e" * 64,
        },
    )
    release.config.executor_wheel = Path("executor.whl")
    release.config.bundle = Path("bundle.tar.gz")

    MODULE.RegionalRelease.join_cluster(release, "gpu-a")

    _assert_collector_is_staged_with_dcgm_before_the_executor(calls)
    # Fix round 2 (LOW-3): the joined cluster's absence rule is rendered
    # once it is up, not on the next deploy.
    assert calls[8:] == ["observability"], calls


def test_upgrade_applies_the_collector_on_the_observability_node() -> None:
    """An OBSERVABILITY plan node visits the cluster and records its progress."""
    calls: list[str] = []
    progress: list[tuple[tuple[Component, ...], str]] = []
    release = SimpleNamespace(**_gpu_recorder(calls))

    GPU_ROLLOUT.upgrade_gpu_target(
        release,
        TARGET,
        DIFF.ReleaseDiff(
            kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
            changed=frozenset({"observability_adot"}),
        ),
        DIFF.ReleaseExecutionPlan(nodes=(Component.DCGM, Component.OBSERVABILITY)),
        progress=lambda components, status, _details: progress.append(
            (components, status)
        ),
    )

    assert set(calls) == {"dcgm", "adot"}, calls
    observability = [
        status
        for components, status in progress
        if components == (Component.OBSERVABILITY,)
    ]
    assert observability == ["STARTED", "COMPLETED"], progress


def test_upgrade_without_the_observability_node_leaves_the_collector_alone() -> None:
    calls: list[str] = []

    GPU_ROLLOUT.upgrade_gpu_target(
        SimpleNamespace(**_gpu_recorder(calls)),
        TARGET,
        DIFF.ReleaseDiff(
            kind=DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
            changed=frozenset({"dcgm"}),
        ),
        DIFF.ReleaseExecutionPlan(nodes=(Component.DCGM,)),
    )

    assert calls == ["dcgm"]


def test_an_observability_only_plan_still_visits_the_gpu_clusters() -> None:
    """The cluster loop used to return before an observability-only release.

    That was right while observability meant the control-plane collector and
    the AMP rules; now the component has a per-cluster half, so the gate has to
    let the plan through to ``_upgrade_gpu_target``.
    """
    visited: list[str] = []
    release = SimpleNamespace(
        state={},
        config=SimpleNamespace(clusters=(TARGET,), upgrade_max_parallel_clusters=1),
        _upgrade_gpu_target=lambda target, *_args, **_kwargs: visited.append(
            target.cluster_id
        ),
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
    )

    ORCHESTRATION.upgrade_gpu_clusters(
        release,
        diff=DIFF.diff_from_changed({"observability_adot"}),
        plan=DIFF.ReleaseExecutionPlan(nodes=(Component.OBSERVABILITY,)),
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    assert visited == ["gpu-a"]


def test_the_mutation_preflight_dry_runs_the_collector_for_the_plan() -> None:
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(TARGET,), agent_config_digest="c" * 64),
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        _preflight_gpu_dcgm_exporter=lambda _target: calls.append("dcgm"),
        _preflight_gpu_adot_collector=lambda _target: calls.append("adot"),
        _preflight_gpu_deployments=lambda *_args, **_kwargs: calls.append("gpu"),
        _preflight_node_runtime=lambda *_args, **_kwargs: calls.append("node"),
    )

    PREFLIGHT.preflight_upgrade_mutations(
        release, DIFF.ReleaseExecutionPlan(nodes=(Component.OBSERVABILITY,))
    )
    assert calls == ["adot"]

    calls.clear()
    PREFLIGHT.preflight_upgrade_mutations(
        release, DIFF.ReleaseExecutionPlan(nodes=(Component.DCGM,))
    )
    assert calls == ["dcgm"]


def test_remove_cluster_scales_the_collector_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collector is not in ``inventory.DEPLOYMENTS``, so it needs its own line.

    A removed cluster whose collector keeps running keeps writing the cluster's
    series into the site's AMP workspace under the site's labels, which is the
    one thing ``remove-cluster`` exists to stop.
    """
    for name in ("revoke_registry_cluster", "purge_registry_cluster"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    release = _release(_config(tmp_path, adot_irsa_role_arn=ROLE_ARN))
    scaled: list[tuple[str, int]] = []
    monkeypatch.setattr(release, "_update_registry", lambda *_args, **_kwargs: None)
    # The rule re-render at the end is pinned by its own test below.
    monkeypatch.setattr(release, "_apply_observability", lambda **_kwargs: None)
    monkeypatch.setattr(
        release,
        "_scale_if_present",
        lambda _arguments, deployment, replicas, **_kwargs: scaled.append(
            (deployment, replicas)
        ),
    )

    release.remove_cluster("gpu-a")

    assert (DATAPLANE_ADOT_DEPLOYMENT, 0) in scaled, scaled
    assert (MODULE.inventory.GPU_RECONCILER_DEPLOYMENT, 0) in scaled, scaled


def test_bootstrap_cleanup_scales_the_collector_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleaned-up bootstrap must not leave a collector writing to AMP."""
    release = _release(_config(tmp_path, adot_irsa_role_arn=ROLE_ARN))
    scaled: list[str] = []
    monkeypatch.setattr(
        release,
        "_scale_if_present",
        lambda _arguments, deployment, _replicas, **_kwargs: scaled.append(deployment),
    )
    monkeypatch.setattr(release, "_cancel_active_installer_jobs", lambda _t: None)
    monkeypatch.setattr(release, "_save_state", lambda *_args, **_kwargs: None)
    release.state = {}

    MODULE.RegionalRelease._cleanup_bootstrap(release)

    assert DATAPLANE_ADOT_DEPLOYMENT in scaled, scaled


# --- fix round 2: bootstrap, join and remove re-render the expected rules ---------


class _InstallerRunner:
    """Records the installer invocation and reads the rendered rules file while
    it still exists (the release removes the temporary directory afterwards)."""

    dry_run = False

    def __init__(self) -> None:
        self.installs: list[tuple[list[str], dict[str, str], str | None]] = []
        self.other: list[list[str]] = []

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        if arguments[:2] != ["bash", str(DATAPLANE.AMP_MONITORING_INSTALLER)]:
            self.other.append(list(arguments))
            return ""
        handed = None
        if "--dataplane-expected-rules" in arguments:
            path = Path(arguments[arguments.index("--dataplane-expected-rules") + 1])
            handed = path.read_text(encoding="utf-8")
        self.installs.append((list(arguments), dict(kwargs.get("env") or {}), handed))
        return ""

    def probe(self, _arguments: list[str], **_kwargs: Any) -> bool:
        return False


def _site_target(cluster_id: str, *, role: str | None = ROLE_ARN) -> SimpleNamespace:
    return SimpleNamespace(
        cluster_id=cluster_id,
        context=f"{cluster_id}-context",
        region="us-east-1",
        adot_irsa_role_arn=role,
    )


class _SiteRelease(SimpleNamespace):
    """A release whose observability step is the REAL one -- the very function
    the class binds -- and whose every other step only records its name."""

    _apply_observability = DATAPLANE.apply_observability
    _target = MODULE.RegionalRelease._target


def _site_release(
    calls: list[str], *, clusters: tuple[SimpleNamespace, ...]
) -> _SiteRelease:
    runner = _InstallerRunner()
    recorded = _gpu_recorder(calls)
    # The class method is the real step; the recorder's stand-in must not
    # shadow it as an instance attribute.
    recorded.pop("_apply_observability")
    release = _SiteRelease(
        **recorded,
        runner=runner,
        state={},
        release_id="release-1",
        adot_image="adot@sha256:" + "a" * 64,
        bundle_sha="b" * 64,
        executor_wheel_sha="e" * 64,
        _ensure_contexts=lambda: None,
        _apply_rds_ca_bundle=lambda: calls.append("rds-ca"),
        _bootstrap_cpu_is_current=lambda: False,
        _save_state=lambda phase, **_updates: calls.append(f"save:{phase}"),
        _require_cpu_secrets=lambda **_kwargs: None,
        _initialize_registry=lambda: calls.append("registry"),
        _upload_release=lambda *_args, **_kwargs: calls.append("upload"),
        _ensure_schema=lambda: calls.append("schema"),
        _apply_cpu=lambda **_kwargs: calls.append("cpu"),
        _apply_nlb=lambda: calls.append("nlb"),
        _validate_release=lambda: calls.append("validate"),
        _update_registry=lambda _target, *, remove: None,
        _upload_config_map=lambda *_args, **_kwargs: None,
        _scale_if_present=lambda *_args, **_kwargs: calls.append("scale"),
        _cpu=lambda *arguments: ["kubectl", "--kubeconfig", "/secure/cpu", *arguments],
        _gpu=lambda target, *arguments: [
            "kubectl",
            "--context",
            target.context,
            *arguments,
        ],
        _config_map_data=lambda _name: {
            "required-agent-artifact-sha256": "a" * 64,
            "required-regional-executor-artifact-sha256": "e" * 64,
        },
    )
    release.config = SimpleNamespace(
        clusters=clusters,
        namespace="gpu-fault-system",
        aws_region="us-east-1",
        cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        cpu_kubeconfig="/secure/cpu.kubeconfig",
        auto_rollback=False,
        agent_config_digest="c" * 64,
        executor_wheel=Path("executor.whl"),
        bundle=Path("bundle.tar.gz"),
        health=SimpleNamespace(
            amp_workspace_id="ws-a",
            amp_rule_namespace="gpu-fault-control-plane-capacity",
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts",
            require_confirmed_sns_subscription=True,
        ),
        notifications=SimpleNamespace(admin_email="ops@example.com"),
    )
    return release


def _installs(
    release: _SiteRelease,
) -> list[tuple[list[str], dict[str, str], str | None]]:
    return list(release.runner.installs)


def test_the_release_class_binds_the_observability_step() -> None:
    assert MODULE.RegionalRelease._apply_observability is DATAPLANE.apply_observability


def test_the_observability_step_runs_the_installer_with_the_rendered_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOW-6: ``_apply_observability`` -> installer argv and environment. The
    orchestration tests stub this step by name, so nothing else pinned it."""
    monkeypatch.setenv("GPU_FAULT_TEST_MARKER", "inherited")
    release = _site_release([], clusters=(_site_target("gpu-a"),))

    MODULE.RegionalRelease._apply_observability(release)

    (arguments, environment, handed), *rest = _installs(release)
    assert rest == [], "the installer ran more than once"
    assert arguments[:2] == ["bash", str(DATAPLANE.AMP_MONITORING_INSTALLER)]
    assert arguments[2] == "--dataplane-expected-rules", arguments
    assert handed == DATAPLANE.render_dataplane_expected_rules(release)
    assert handed is not None and 'gpu_cluster="gpu-a"' in handed
    expected_environment = {
        "AWS_REGION": "us-east-1",
        "CPU_EKS_CLUSTER": "control",
        "CPU_KUBECONFIG": "/secure/cpu.kubeconfig",
        "AMP_WORKSPACE_ID": "ws-a",
        "SNS_TOPIC_NAME": "gpu-fault-alerts",
        "NAMESPACE": "gpu-fault-system",
        "RULE_NAMESPACE": "gpu-fault-control-plane-capacity",
        "GPU_FAULT_ADOT_IMAGE": release.adot_image,
        "GPU_FAULT_ENABLE_ADOT": "true",
        "GPU_FAULT_ENABLE_AMP": "true",
        "GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION": "true",
        "GPU_FAULT_ALERT_EMAIL": "ops@example.com",
    }
    for key, value in expected_environment.items():
        assert environment.get(key) == value, (key, environment.get(key))
    assert environment.get("GPU_FAULT_TEST_MARKER") == "inherited", (
        "the installer environment no longer inherits the process environment"
    )


def test_bootstrap_puts_the_expected_rules_once_the_clusters_are_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEDIUM-1: the admin bootstrap runs the installer bare (it does not know
    the expected set), and every ``save_state`` records the current
    observability digest, so a freshly bootstrapped site with a role never got
    its ``gpu-fault-dataplane-expected`` namespace until an unrelated
    observability input moved the digest. The engine's bootstrap renders them."""
    calls: list[str] = []
    monkeypatch.setattr(MODULE, "ensure_runtime_profile", lambda _r: None)
    monkeypatch.setattr(
        MODULE,
        "bootstrap_gpu_clusters",
        lambda _release, _completed: calls.append("gpu-clusters"),
    )
    release = _site_release(calls, clusters=(_site_target("gpu-a"),))

    MODULE.RegionalRelease.bootstrap(release)

    installs = _installs(release)
    assert len(installs) == 1, f"the bootstrap ran the installer {len(installs)}x"
    arguments, _environment, handed = installs[0]
    assert arguments[2] == "--dataplane-expected-rules", arguments
    assert handed is not None and 'gpu_cluster="gpu-a"' in handed
    assert calls.index("gpu-clusters") < calls.index("validate")
    assert release.runner.other and "apply" in release.runner.other[0], (
        "the prerequisites apply did not run"
    )


def test_bootstrap_of_a_role_less_site_asks_for_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(MODULE, "ensure_runtime_profile", lambda _r: None)
    monkeypatch.setattr(MODULE, "bootstrap_gpu_clusters", lambda *_a: None)
    release = _site_release(calls, clusters=(_site_target("gpu-a", role=None),))

    MODULE.RegionalRelease.bootstrap(release)

    (arguments, _environment, handed), *rest = _installs(release)
    assert rest == []
    assert arguments[2:] == ["--no-dataplane-expected-rules"], arguments
    assert handed is None


def test_join_re_renders_the_expected_rules_with_the_joined_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOW-3: a joined cluster with a role had no absence rule until the next
    deploy; join re-runs the (idempotent) installer with the new set."""
    calls: list[str] = []
    for name in ("prepare_join_registry", "ensure_runtime_profile"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    joined = _site_target("gpu-b")
    monkeypatch.setattr(MODULE, "join_target", lambda _self, _cluster_id: joined)
    release = _site_release(calls, clusters=(_site_target("gpu-a"), joined))

    MODULE.RegionalRelease.join_cluster(release, "gpu-b")

    (arguments, _environment, handed), *rest = _installs(release)
    assert rest == []
    assert arguments[2] == "--dataplane-expected-rules", arguments
    assert handed is not None
    assert 'gpu_cluster="gpu-b"' in handed and 'gpu_cluster="gpu-a"' in handed
    assert calls.index("node-runtime") < len(calls), calls


def test_remove_re_renders_the_expected_rules_without_the_removed_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOW-3: the removed cluster's collector is scaled to 0, so its absence
    rule would fire 15 minutes later until the next deploy. The config handed
    to ``remove_cluster`` still names the cluster (``_target`` finds it there),
    so the render has to leave it out explicitly."""
    calls: list[str] = []
    for name in ("revoke_registry_cluster", "purge_registry_cluster"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    release = _site_release(
        calls, clusters=(_site_target("gpu-a"), _site_target("gpu-b"))
    )

    MODULE.RegionalRelease.remove_cluster(release, "gpu-b")

    (arguments, _environment, handed), *rest = _installs(release)
    assert rest == []
    assert arguments[2] == "--dataplane-expected-rules", arguments
    assert handed is not None
    assert 'gpu_cluster="gpu-a"' in handed and 'gpu_cluster="gpu-b"' not in handed
    assert "scale" in calls, "the collector was not scaled down"


def test_remove_of_the_last_expected_cluster_asks_for_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    for name in ("revoke_registry_cluster", "purge_registry_cluster"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    release = _site_release(calls, clusters=(_site_target("gpu-a"),))

    MODULE.RegionalRelease.remove_cluster(release, "gpu-a")

    (arguments, _environment, handed), *rest = _installs(release)
    assert rest == []
    assert arguments[2:] == ["--no-dataplane-expected-rules"], arguments
    assert handed is None


# --- rollback -----------------------------------------------------------------

PREVIOUS_ADOT_IMAGE = "adot@sha256:" + "0" * 64


def _rollback(
    monkeypatch: pytest.MonkeyPatch,
    previous_extra: dict[str, Any],
    calls: list[tuple[str, Any]],
) -> None:
    """Drive the public rollback entry with only the observability restore live.

    The controller, data-plane, CPU and verify phases are recorded away; the
    state says an observability-only release completed its
    ``observability-ready`` phase, which is what puts OBSERVABILITY into the
    compensation plan.
    """
    for name in (
        "_stage_rollback_controller",
        "_rollback_gpu_clusters",
        "_restore_rollback_cpu",
        "_verify_and_complete_rollback",
        "cleanup_candidate_rollout_state",
    ):
        monkeypatch.setattr(ORCHESTRATION, name, lambda *_args, **_kwargs: None)
    clusters = (TARGET, SimpleNamespace(cluster_id="gpu-b"))
    identity = {
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
    release = SimpleNamespace(
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
        config=SimpleNamespace(clusters=clusters, auto_rollback=True),
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _save_state=lambda _phase, **_updates: None,
        _restore_observability_snapshot=lambda snapshot: calls.append(
            ("snapshot", snapshot)
        ),
        _apply_gpu_adot_collector=lambda target, *, image: calls.append(
            ("adot", (target.cluster_id, image))
        ),
    )
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
        "agent_identities": {target.cluster_id: dict(identity) for target in clusters},
        "observability": {"adot": []},
        **previous_extra,
    }
    ORCHESTRATION.rollback_release(release, state=previous)


def test_observability_rollback_puts_every_collector_on_the_previous_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback for a state whose observability snapshot predates the
    per-cluster collector capture (``dataplane_adot`` key absent): every
    configured cluster gets the candidate manifest with the previous
    ``adot_image``, after the control-plane snapshot. A state that carries the
    key never takes this path (``test_dataplane_adot_compensation.py``)."""
    calls: list[tuple[str, Any]] = []

    _rollback(monkeypatch, {"adot_image": PREVIOUS_ADOT_IMAGE}, calls)

    assert calls == [
        ("snapshot", {"adot": []}),
        ("adot", ("gpu-a", PREVIOUS_ADOT_IMAGE)),
        ("adot", ("gpu-b", PREVIOUS_ADOT_IMAGE)),
    ]


def test_observability_rollback_refuses_without_a_previous_adot_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    with pytest.raises(ReleaseError, match="previous ADOT image"):
        _rollback(monkeypatch, {}, calls)

    # Fix round 2 (LOW-1): the whole snapshot is validated before either half
    # is put back, so the control-plane restore never ran.
    assert [name for name, _ in calls] == []


def test_rollback_gpu_adot_collectors_reapplies_the_previous_image_per_cluster() -> (
    None
):
    """The previous-image fallback helper, on its own."""
    calls: list[tuple[str, str]] = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(TARGET, SimpleNamespace(cluster_id="gpu-b"))),
        _apply_gpu_adot_collector=lambda target, *, image: calls.append(
            (target.cluster_id, image)
        ),
    )

    BOOTSTRAP.rollback_gpu_adot_collectors(release, {"adot_image": PREVIOUS_ADOT_IMAGE})

    assert calls == [("gpu-a", PREVIOUS_ADOT_IMAGE), ("gpu-b", PREVIOUS_ADOT_IMAGE)]
    with pytest.raises(ReleaseError, match="previous ADOT image"):
        BOOTSTRAP.rollback_gpu_adot_collectors(release, {})


# --- digest inputs (F7 re-review F1) ---------------------------------------------


def test_adding_a_cluster_irsa_role_moves_the_observability_digest_only(
    tmp_path: Path,
) -> None:
    """Deploy first, create the role later, deploy again: the second deploy must
    not be a NOOP. The role is not in the registry digest (adding it must not
    re-stage the cluster registry), so the collector's own digest has to carry
    it."""
    without = _release(_config(tmp_path, name="without.json"))
    with_role = _release(
        _config(tmp_path, adot_irsa_role_arn=ROLE_ARN, name="with.json")
    )

    assert without.observability_adot_digest != with_role.observability_adot_digest
    assert without.cluster_registry_digest == with_role.cluster_registry_digest
    assert without.observability_rules_digest == with_role.observability_rules_digest
    assert without.dcgm_digest == with_role.dcgm_digest


def test_the_amp_workspace_moves_the_observability_digest(tmp_path: Path) -> None:
    first = _release(_config(tmp_path, amp_workspace_id="ws-a", name="a.json"))
    second = _release(_config(tmp_path, amp_workspace_id="ws-b", name="b.json"))

    assert first.observability_adot_digest != second.observability_adot_digest


def test_the_collector_manifest_moves_the_observability_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit to adot-dataplane.yaml alone must select the OBSERVABILITY node."""
    config = _config(tmp_path, adot_irsa_role_arn=ROLE_ARN)
    baseline = _release(config).observability_adot_digest
    root = tmp_path / "root"
    for relative in ("deploy/dataplane", "deploy/observability"):
        shutil.copytree(ROOT / relative, root / relative)
    manifest = root / "deploy/dataplane/adot-dataplane.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
    )
    monkeypatch.setattr(MODULE, "ROOT", root)

    assert _release(config).observability_adot_digest != baseline


def test_the_observability_plan_node_follows_its_digest_and_manifests() -> None:
    """Only the observability inputs select the node; a reconciler edit does not."""
    for changed in (
        {"observability_adot"},
        {"observability_manifests"},
        {"adot_image"},
        {"observability_rules"},
    ):
        plan = DIFF.build_execution_plan(DIFF.diff_from_changed(changed))
        assert plan.has(Component.OBSERVABILITY), changed
        assert not plan.has(Component.DCGM, Component.EXECUTOR), changed
    plan = DIFF.build_execution_plan(DIFF.diff_from_changed({"node_manifests"}))
    assert not plan.has(Component.OBSERVABILITY), plan.nodes


def test_a_release_with_the_role_applies_the_collector_and_without_it_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through the real class attribute, not a recorder."""
    applied = _release(_config(tmp_path, adot_irsa_role_arn=ROLE_ARN, name="a.json"))
    BOOTSTRAP.apply_gpu_adot_collector(applied, applied.config.clusters[0])
    verbs = [
        arguments[arguments.index("apply") + 1 : arguments.index("apply") + 3]
        for arguments, _kwargs in applied.runner.calls
        if "apply" in arguments
    ]
    assert len(verbs) == 2, applied.runner.calls
    assert any(
        f"deployment/{DATAPLANE_ADOT_DEPLOYMENT}" in arguments
        for arguments, _kwargs in applied.runner.calls
    ), applied.runner.calls

    skipped = _release(_config(tmp_path, name="b.json"))
    BOOTSTRAP.apply_gpu_adot_collector(skipped, skipped.config.clusters[0])
    # The skip only probes for a leftover collector to scale down (F10 fix 1,
    # F4); with none present it applies and scales nothing.
    assert skipped.runner.calls == []
    assert [arguments[-3:] for arguments in skipped.runner.probes] == [
        ["get", "deployment", DATAPLANE_ADOT_DEPLOYMENT]
    ]
    assert "gpu-a: data-plane ADOT collector not applied" in capsys.readouterr().err
