"""A rollback restarts only the CPU tiers whose start-up inputs it changes.

Both CPU stages of a rollback -- the controller stage, which puts the previous
pins in front of the data restore, and the CPU restore -- used to force-restart
all three tiers. An Executor-only rollback therefore paid two full CPU rolls for
pins every running Pod hot-loads from ``gpu-fault-release-metadata``, and
BOOT-020 measured its Executor-only T_safe above the 900 s limit.

``RollbackRestartPlan`` decides per tier from proof. These tests pin which
restored input restarts which tier, that an incomplete proof restarts (fail
closed), what the apply environment carries, and the evidence each stage
leaves in ``rollback_timing``.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_progress as PROGRESS
from gpu_fault_release import regional_release_rollback_restart as RESTART
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_release_config import ReleaseError
from scripts.e2e.regional import run_boot020_release_rolling as boot020
from tests.regional.test_release_rollback_container_env import _rollback_environment
from tests.regional.test_release_rollback_state import _rollback_previous

NAMESPACE = "gpu-fault-system"
EMAIL = "gpu-fault-email"
EMAIL_BACKUP = "gpu-fault-email-rollback-candidate"
# Base64, as a Secret's ``data`` carries it; compared, never recorded.
EMAIL_DATA = {"email-sender": "c2VuZGVy", "site-id": "c2l0ZQ=="}
CHANGED_EMAIL = "b3RoZXItc2VuZGVy"
DEPLOYMENTS = RESTART.ROLE_DEPLOYMENTS
EVERY_TIER = list(RESTART.ALL_ROLES)
FORCE = "GPU_FAULT_FORCE_ROLE_RESTART"
CONVERGED = {name: "converged" for name in DEPLOYMENTS.values()}


def _iso(epoch: float) -> str:
    """A Kubernetes timestamp: whole seconds, UTC, ``Z``."""

    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deployment(name: str) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "generation": 4},
        "spec": {
            "replicas": 2,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {
                    "labels": {"app": name},
                    "annotations": {"kubectl.kubernetes.io/restartedAt": "forward"},
                },
                "spec": {"containers": [{"name": "app"}, {"name": "sidecar"}]},
            },
        },
    }


def _pod(deployment: str, index: int, started: float) -> dict[str, Any]:
    return {
        "metadata": {
            "name": f"{deployment}-{index}",
            "labels": {"app": deployment},
            "creationTimestamp": _iso(started),
        },
        "spec": {"containers": [{"name": "app"}, {"name": "sidecar"}]},
        "status": {
            "containerStatuses": [
                {"name": name, "state": {"running": {"startedAt": _iso(started + 5)}}}
                for name in ("sidecar", "app")
            ]
        },
    }


class Site:
    """The CPU cluster as a rollback stage reads and writes it.

    The forward transaction captured the previous release half an hour ago;
    the role Pods have run since an hour before that capture. By default
    nothing a rollback restores differs from what they started with, which is
    the Executor-only rollback.
    """

    def __init__(self) -> None:
        self.captured_at = time.time() - 1800.0
        self.deployments = {name: _deployment(name) for name in DEPLOYMENTS.values()}
        self.pods = [
            _pod(name, index, self.captured_at - 3600.0)
            for name in DEPLOYMENTS.values()
            for index in (0, 1)
        ]
        self.secrets = {EMAIL: dict(EMAIL_DATA), EMAIL_BACKUP: dict(EMAIL_DATA)}
        self.config_maps = {
            f"{name}-config-{domain}": {"GPU_FAULT_PROCESSOR_WORKERS": "24"}
            for name in DEPLOYMENTS.values()
            for domain in ("core", "processor")
        }
        self.snapshots = copy.deepcopy(self.config_maps)
        self.baseline = RESTART.cpu_restart_baseline(self.deployments, self.captured_at)
        self.registry_backup = False
        self.unreadable: set[str] = set()
        self.calls: list[str] = []

    def previous(self, **extra: Any) -> dict[str, Any]:
        """The previous-release record the forward capture wrote."""

        return {
            "secret_backups": {"cpu": {"source": EMAIL, "backup": EMAIL_BACKUP}},
            "cpu_role_config_maps": copy.deepcopy(self.snapshots),
            RESTART.BASELINE_KEY: copy.deepcopy(self.baseline),
            **extra,
        }

    def get_json(self, arguments: list[str]) -> dict[str, Any]:
        at = arguments.index("get")
        kind, names = arguments[at + 1], arguments[at + 2 :]
        self.calls.append(f"get {kind}")
        if kind in self.unreadable:
            raise ReleaseError(f"kubectl get {kind} failed")
        if kind == "secret":
            return {"data": dict(self.secrets[names[0]])}
        if kind == "deployment":
            return {
                "items": [
                    copy.deepcopy(self.deployments[name])
                    for name in names
                    if name in self.deployments
                ]
            }
        if kind == "pods":
            return {"items": copy.deepcopy(self.pods)}
        raise AssertionError(f"unexpected read: {arguments}")

    def config_maps_data(self, names: list[str]) -> dict[str, dict[str, str]]:
        self.calls.append("get configmap")
        if "configmap" in self.unreadable:
            raise ReleaseError("kubectl get configmap failed")
        return {name: dict(self.config_maps[name]) for name in names}

    def restore_secret(
        self, _kubectl: list[str], *, source: str, backup: str
    ) -> dict[str, str]:
        self.calls.append("restore secret")
        self.secrets[source] = dict(self.secrets[backup])
        return dict(self.secrets[backup])

    def restore_registry_backup(self) -> bool:
        self.calls.append("restore registry")
        restored, self.registry_backup = self.registry_backup, False
        return restored

    def publish_restored_registry(self) -> None:
        self.calls.append("publish registry")

    def restore_role_config_maps(self, snapshots: object) -> bool:
        self.calls.append("restore config-maps")
        if not snapshots or not isinstance(snapshots, dict):
            return False
        for name, data in snapshots.items():
            self.config_maps[name] = dict(data)
        return True

    def run(self, arguments: list[str], **_kwargs: Any) -> str:
        at = arguments.index("patch")
        name = arguments[at + 2]
        self.config_maps[name].update(json.loads(arguments[-1])["data"])
        self.calls.append(f"patch {name}")
        return ""

    def roll(self, roles: list[str]) -> None:
        """What the apply does to a tier it restarts: a new token, new Pods."""

        now = time.time()
        for role in roles:
            name = DEPLOYMENTS[role]
            template = self.deployments[name]["spec"]["template"]
            template["metadata"]["annotations"]["kubectl.kubernetes.io/restartedAt"] = (
                _iso(now)
            )
            self.pods = [
                pod for pod in self.pods if pod["metadata"]["labels"]["app"] != name
            ] + [_pod(name, index, now) for index in (2, 3)]

    def release(self, **stubs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            **{
                "config": SimpleNamespace(namespace=NAMESPACE),
                "runner": SimpleNamespace(run=self.run),
                "_cpu": lambda *args: ["kubectl", *args],
                "_get_json": self.get_json,
                "_config_maps_data": self.config_maps_data,
                "_restore_secret": self.restore_secret,
                "_restore_registry_backup": self.restore_registry_backup,
                "_publish_restored_registry": self.publish_restored_registry,
                "_restore_cpu_role_config_maps": self.restore_role_config_maps,
                **stubs,
            }
        )


def _stage(
    site: Site,
    previous: dict[str, Any] | None = None,
    *,
    agent_config: dict[str, str] | None = None,
    restore_config_maps: bool = False,
    report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One rollback stage's decision and evidence, in the orchestration's order.

    ``agent_config`` is the controller stage's ``-config-core`` patch,
    ``restore_config_maps`` the CPU restore's role ConfigMap restore, and
    ``report`` what the apply would write back.
    """

    previous = site.previous() if previous is None else previous
    with RESTART.RollbackRestartPlan(site.release(), previous) as plan:
        plan.restore_secret_and_registry()
        if agent_config is not None:
            plan.patch_controller_config(agent_config)
        if restore_config_maps:
            plan.restore_role_config_maps(previous.get("cpu_role_config_maps"))
        arguments = plan.environment_arguments()
        assert Path(arguments["role_restart_report"]).parent == plan.directory, (
            "the apply reports into the stage's own directory"
        )
        if report is not None:
            plan.report_path.write_text(json.dumps(report), encoding="utf-8")
        return arguments, plan.details()["cpu_restart"]


def _restarted(evidence: dict[str, Any]) -> dict[str, list[str]]:
    return dict(evidence["restarted"])


def _narrated(capsys: pytest.CaptureFixture[str], kind: str) -> list[str]:
    """The narration lines of one kind (``<kind> <stamp> field=value ...``)."""

    return [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.split(" ", 1)[0] == kind
    ]


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


def test_an_untouched_tier_keeps_its_pods_in_both_stages() -> None:
    """The Executor-only rollback: no CPU input changes, nothing restarts.

    The Secret restore writes what the Secret already holds, there is no
    registry backup, no Agent identity to patch, the role ConfigMaps equal the
    snapshot, and every Deployment and Pod is the one the capture saw.
    """

    site = Site()

    controller, controller_evidence = _stage(site)
    restore, restore_evidence = _stage(site, restore_config_maps=True)

    for arguments, evidence in (
        (controller, controller_evidence),
        (restore, restore_evidence),
    ):
        assert arguments["force_restart_roles"] == []
        assert _restarted(evidence) == {}
        assert evidence["kept"] == EVERY_TIER
    assert "patch" not in " ".join(site.calls), "no Agent identity to stage"
    assert "publish registry" not in site.calls, "there was no registry backup"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda spec: spec.update(replicas=3), id="replicas"),
        pytest.param(
            lambda spec: spec["template"]["metadata"]["annotations"].update(
                {"gpu-fault.io/runtime-image": "candidate"}
            ),
            id="template",
        ),
        pytest.param(
            lambda spec: spec.update(strategy={"type": "Recreate"}), id="strategy"
        ),
    ],
)
def test_a_spec_change_restarts_only_that_tier(
    change: Callable[[dict[str, Any]], None],
) -> None:
    """A control-plane change to one Deployment restarts that tier alone."""

    site = Site()
    change(site.deployments["gpu-fault-api-ha"]["spec"])

    arguments, evidence = _stage(site, restore_config_maps=True)

    assert arguments["force_restart_roles"] == ["ingress"]
    assert _restarted(evidence) == {"ingress": ["spec-changed"]}
    assert evidence["kept"] == ["spool", "worker"]


def test_a_generation_bump_alone_proves_nothing_either_way() -> None:
    """Kubernetes bumps ``generation`` for metadata-only writes too; the proof
    compares the spec itself, so an unchanged spec keeps the tier."""

    site = Site()
    site.deployments["gpu-fault-control-worker"]["metadata"]["generation"] = 9
    site.deployments["gpu-fault-control-worker"]["metadata"]["annotations"] = {
        "gpu-fault.io/pin-config-sha256": "candidate"
    }

    arguments, _evidence = _stage(site)

    assert arguments["force_restart_roles"] == []


def test_a_registry_restore_restarts_every_tier() -> None:
    site = Site()
    site.registry_backup = True

    arguments, evidence = _stage(site)

    assert arguments["force_restart_roles"] == EVERY_TIER
    assert _restarted(evidence) == {role: ["registry-restored"] for role in EVERY_TIER}
    assert site.calls.index("restore registry") < site.calls.index("publish registry")


def test_a_secret_restore_that_changes_the_data_restarts_every_tier(
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = Site()
    site.secrets[EMAIL]["email-sender"] = CHANGED_EMAIL

    arguments, evidence = _stage(site)

    assert arguments["force_restart_roles"] == EVERY_TIER
    assert _restarted(evidence) == {role: ["secret-changed"] for role in EVERY_TIER}
    assert site.secrets[EMAIL] == EMAIL_DATA, "the backup's data is back"
    recorded = json.dumps(evidence) + capsys.readouterr().err
    for value in (*EMAIL_DATA.values(), CHANGED_EMAIL):
        assert value not in recorded, "Secret data is compared, never recorded"


def test_without_a_secret_backup_the_secret_is_neither_read_nor_restored() -> None:
    site = Site()
    previous = site.previous(secret_backups={})

    arguments, _evidence = _stage(site, previous)

    assert arguments["force_restart_roles"] == []
    assert "get secret" not in site.calls and "restore secret" not in site.calls


def test_the_agent_patch_restarts_only_the_tiers_whose_config_core_it_changes() -> None:
    """``-config-core`` is ``envFrom`` env: a tier whose copy already holds the
    previous Agent identity keeps its Pods, the others restart."""

    site = Site()
    for name in DEPLOYMENTS.values():
        site.config_maps[f"{name}-config-core"]["GPU_FAULT_AGENT_CONFIG"] = "previous"
    site.config_maps["gpu-fault-control-worker-config-core"][
        "GPU_FAULT_AGENT_CONFIG"
    ] = "candidate"

    arguments, evidence = _stage(
        site, agent_config={"GPU_FAULT_AGENT_CONFIG": "previous"}
    )

    assert arguments["force_restart_roles"] == ["worker"]
    assert _restarted(evidence) == {"worker": ["config-core-changed"]}
    assert [call for call in site.calls if call.startswith("patch")] == [
        f"patch {name}-config-core" for name in inventory.CPU_RUNTIME_DEPLOYMENTS
    ], "every tier's -config-core is patched, changed or not"
    assert {
        site.config_maps[f"{name}-config-core"]["GPU_FAULT_AGENT_CONFIG"]
        for name in DEPLOYMENTS.values()
    } == {"previous"}


def test_a_role_config_map_restore_restarts_only_the_tier_it_changes() -> None:
    site = Site()
    site.config_maps["gpu-fault-telemetry-spool-worker-config-processor"][
        "GPU_FAULT_PROCESSOR_WORKERS"
    ] = "32"

    arguments, evidence = _stage(site, restore_config_maps=True)

    assert arguments["force_restart_roles"] == ["spool"]
    assert _restarted(evidence) == {"spool": ["config-maps-changed"]}
    assert (
        site.config_maps["gpu-fault-telemetry-spool-worker-config-processor"]
        == site.snapshots["gpu-fault-telemetry-spool-worker-config-processor"]
    )


def test_without_a_config_map_snapshot_every_tier_restarts() -> None:
    """The apply then renders the current ConfigMaps, which the running Pods
    may not have started from."""

    site = Site()

    arguments, evidence = _stage(
        site, site.previous(cpu_role_config_maps={}), restore_config_maps=True
    )

    assert arguments["force_restart_roles"] == EVERY_TIER
    assert _restarted(evidence) == {
        role: ["no-config-map-snapshot"] for role in EVERY_TIER
    }


def _created_after_capture(site: Site, pod: dict[str, Any]) -> None:
    pod["metadata"]["creationTimestamp"] = _iso(site.captured_at + 60)


def _created_just_before_capture(site: Site, pod: dict[str, Any]) -> None:
    # Inside the clock-skew margin: not proven older than the capture.
    pod["metadata"]["creationTimestamp"] = _iso(site.captured_at - 30)


def _container_restarted(site: Site, pod: dict[str, Any]) -> None:
    pod["status"]["containerStatuses"][1]["state"]["running"]["startedAt"] = _iso(
        site.captured_at + 60
    )


def _container_waiting(_site: Site, pod: dict[str, Any]) -> None:
    pod["status"]["containerStatuses"][0]["state"] = {
        "waiting": {"reason": "CrashLoopBackOff"}
    }


def _container_unreported(_site: Site, pod: dict[str, Any]) -> None:
    pod["status"]["containerStatuses"].pop()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(_created_after_capture, "pod-started-after-capture", id="new"),
        pytest.param(
            _created_just_before_capture, "pod-started-after-capture", id="margin"
        ),
        pytest.param(
            _container_restarted, "container-restarted-after-capture", id="restart"
        ),
        pytest.param(_container_waiting, "container-not-running", id="waiting"),
        pytest.param(_container_unreported, "container-not-running", id="no-status"),
    ],
)
def test_a_pod_that_may_hold_the_candidate_s_env_restarts_its_tier(
    change: Callable[[Site, dict[str, Any]], None], reason: str
) -> None:
    """A Pod or container that started after the capture resolved its env from
    whatever the forward transaction had written by then."""

    site = Site()
    (pod,) = [
        pod
        for pod in site.pods
        if pod["metadata"]["name"] == "gpu-fault-control-worker-1"
    ]
    change(site, pod)

    arguments, evidence = _stage(site)

    assert arguments["force_restart_roles"] == ["worker"]
    assert _restarted(evidence) == {"worker": [reason]}


def test_terminating_and_foreign_pods_do_not_count() -> None:
    site = Site()
    after = site.captured_at + 60
    leaving = _pod("gpu-fault-api-ha", 9, after)
    leaving["metadata"]["deletionTimestamp"] = _iso(after + 5)
    foreign = _pod("gpu-fault-adot", 0, after)
    site.pods += [leaving, foreign]

    arguments, _evidence = _stage(site)

    assert arguments["force_restart_roles"] == []


def test_without_a_baseline_every_tier_restarts() -> None:
    """A transaction captured before the baseline existed cannot prove a tier."""

    site = Site()
    previous = site.previous()
    del previous[RESTART.BASELINE_KEY]

    arguments, evidence = _stage(site, previous)

    assert arguments["force_restart_roles"] == EVERY_TIER
    assert _restarted(evidence) == {role: ["no-baseline"] for role in EVERY_TIER}
    assert "get pods" not in site.calls, "nothing to prove the Pods against"


def test_a_baseline_without_one_deployment_restarts_that_tier() -> None:
    site = Site()
    del site.baseline["spec_sha256"]["gpu-fault-telemetry-spool-worker"]

    arguments, evidence = _stage(site)

    assert arguments["force_restart_roles"] == ["spool"]
    assert _restarted(evidence) == {"spool": ["no-baseline"]}


def test_a_missing_deployment_restarts_its_tier() -> None:
    site = Site()
    del site.deployments["gpu-fault-control-worker"]

    arguments, evidence = _stage(site)

    assert arguments["force_restart_roles"] == ["worker"]
    assert _restarted(evidence) == {"worker": ["deployment-missing"]}


@pytest.mark.parametrize(
    ("kind", "what"),
    [
        ("secret", "secret"),
        ("deployment", "deployments"),
        ("pods", "pods"),
        ("configmap", "config-maps"),
    ],
)
def test_a_read_the_proof_needs_that_fails_restarts_every_tier(
    kind: str, what: str, capsys: pytest.CaptureFixture[str]
) -> None:
    site = Site()
    site.unreadable.add(kind)

    arguments, evidence = _stage(site, restore_config_maps=True)

    assert arguments["force_restart_roles"] == EVERY_TIER
    assert _restarted(evidence) == {role: [f"{what}-unreadable"] for role in EVERY_TIER}
    (line,) = _narrated(capsys, "rollback-cpu-restart-proof")
    assert f" read={what} error=ReleaseError " in line


def test_a_deadline_is_never_mistaken_for_an_unreadable_input() -> None:
    def expired(_arguments: list[str]) -> dict[str, Any]:
        raise DeploymentDeadlineExceeded("deployment deadline exceeded: rollback")

    site = Site()
    plan = RESTART.RollbackRestartPlan(site.release(_get_json=expired), site.previous())

    with plan, pytest.raises(DeploymentDeadlineExceeded):
        plan.environment_arguments()


# --------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------


def test_the_decision_and_its_reasons_are_narrated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = Site()
    site.deployments["gpu-fault-api-ha"]["spec"]["replicas"] = 3
    site.pods[0]["metadata"]["creationTimestamp"] = _iso(site.captured_at + 60)

    _stage(site)

    (line,) = _narrated(capsys, "rollback-cpu-restart")
    assert (
        " forced=spool,ingress "
        "reasons=spool:pod-started-after-capture;ingress:spec-changed "
    ) in line


def test_the_evidence_names_the_fallback_restarts_and_every_verdict() -> None:
    """A kept tier that did not prove the restored pins on ``/healthz`` was
    restarted by the apply after all; the evidence says so, and why."""

    site = Site()
    site.deployments["gpu-fault-api-ha"]["spec"]["replicas"] = 3
    report = {
        "forced": ["ingress"],
        "fallback": ["worker"],
        "pin_wait": {
            "gpu-fault-telemetry-spool-worker": "converged",
            "gpu-fault-control-worker": "unconverged",
            "gpu-fault-api-ha": "converged",
        },
        "restart_wait": {"gpu-fault-control-worker": "converged"},
    }

    _arguments, evidence = _stage(site, report=report)

    assert evidence == {
        "pin_wait": report["pin_wait"],
        "restart_wait": report["restart_wait"],
        "restarted": {"ingress": ["spec-changed"], "worker": ["pin-proof-failed"]},
        "kept": ["spool"],
    }


def test_an_apply_that_wrote_no_report_is_recorded_as_such() -> None:
    _arguments, evidence = _stage(Site())

    assert evidence["report"] == "missing"
    assert "pin_wait" not in evidence


def test_the_stage_directory_lives_exactly_as_long_as_the_stage() -> None:
    site = Site()
    plan = RESTART.RollbackRestartPlan(site.release(), site.previous())

    with plan:
        directory = plan.directory
        assert directory.is_dir(), "the apply writes its report here"
    assert not directory.exists(), "the stage's inputs are gone with it"
    with pytest.raises(RuntimeError, match="outside its stage"):
        _ = plan.directory


def test_the_tiers_are_the_cpu_runtime_deployments() -> None:
    assert set(DEPLOYMENTS.values()) == set(inventory.CPU_RUNTIME_DEPLOYMENTS)
    assert RESTART.ALL_ROLES == ("spool", "worker", "ingress"), (
        "the apply script's canonical role order"
    )


def test_the_baseline_digests_the_whole_spec_of_every_captured_deployment() -> None:
    site = Site()
    changed = copy.deepcopy(site.deployments)
    changed["gpu-fault-api-ha"]["spec"]["template"]["spec"]["containers"].pop()

    baseline = RESTART.cpu_restart_baseline(changed, site.captured_at)

    assert baseline["captured_at"] == site.captured_at
    assert sorted(baseline["spec_sha256"]) == sorted(DEPLOYMENTS.values())
    assert (
        baseline["spec_sha256"]["gpu-fault-api-ha"]
        != (site.baseline["spec_sha256"]["gpu-fault-api-ha"])
    )
    assert (
        baseline["spec_sha256"]["gpu-fault-control-worker"]
        == (site.baseline["spec_sha256"]["gpu-fault-control-worker"])
    )


# --------------------------------------------------------------------------
# The apply environment
# --------------------------------------------------------------------------


def test_the_rollback_environment_carries_the_per_tier_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _rollback_environment(
        monkeypatch,
        force_restart_roles=["worker", "ingress"],
        role_restart_report="/tmp/inputs/role-restart.json",
    )

    assert environment[FORCE] == "true"
    assert environment[RESTART.FORCE_ROLES_VARIABLE] == "worker,ingress"
    assert environment[RESTART.REPORT_VARIABLE] == "/tmp/inputs/role-restart.json"


def test_an_empty_decision_still_selects_the_per_tier_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set but empty is what tells the apply to restart nothing and prove the
    pins of every tier instead."""

    environment = _rollback_environment(
        monkeypatch,
        force_restart_roles=[],
        role_restart_report="/tmp/inputs/role-restart.json",
    )

    assert environment[FORCE] == "false"
    assert environment[RESTART.FORCE_ROLES_VARIABLE] == ""
    assert environment[RESTART.REPORT_VARIABLE] == "/tmp/inputs/role-restart.json"


def test_without_a_decision_the_rollback_keeps_the_all_tier_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither variable is ever inherited from the operator's shell."""

    monkeypatch.setenv(RESTART.FORCE_ROLES_VARIABLE, "")
    monkeypatch.setenv(RESTART.REPORT_VARIABLE, "/stray/role-restart.json")

    environment = _rollback_environment(monkeypatch)

    assert environment[FORCE] == "true"
    assert RESTART.FORCE_ROLES_VARIABLE not in environment
    assert RESTART.REPORT_VARIABLE not in environment


def test_a_forward_apply_never_takes_a_per_tier_decision_from_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ROLLOUT, "apply_failure_domain_map", lambda _release: "")
    monkeypatch.setattr(ROLLOUT, "reconcile_namespace_labels", lambda _release: {})
    applied: list[dict[str, str]] = []
    release = SimpleNamespace(
        runner=SimpleNamespace(
            run=lambda arguments, **kwargs: applied.append(kwargs["env"]) or ""
        ),
        state={},
        _load_state=lambda: {},
    )

    ROLLOUT.render_and_apply_cpu_roles(
        release,
        {
            RESTART.FORCE_ROLES_VARIABLE: "",
            RESTART.REPORT_VARIABLE: "/stray/role-restart.json",
        },
    )

    assert len(applied) == 2, "the render and the apply"
    for environment in applied:
        assert not set(RESTART.ROLE_RESTART_VARIABLES) & set(environment)


# --------------------------------------------------------------------------
# Through the rollback
# --------------------------------------------------------------------------


def _executor_only_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "execution_plan": {
            "nodes": [
                DIFF.ReleaseComponent.EXECUTOR.value,
                DIFF.ReleaseComponent.VERIFY.value,
            ]
        },
        "completed_phases": ["uploaded", "schema-ready"],
    }
    state["component_progress"] = PROGRESS.update_component_progress(
        state,
        DIFF.ReleaseComponent.EXECUTOR,
        PROGRESS.PROGRESS_STARTED,
        cluster_id="gpu-a",
        observed_at_epoch=time.time(),
    )
    return state


def _rollback(
    monkeypatch: pytest.MonkeyPatch, site: Site
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Roll an Executor-only transaction back with both CPU stages live.

    The apply is simulated on ``site``: it restarts exactly the tiers it was
    told to and reports every tier converged. Returns what each stage handed
    the environment builder, and the saved ``rollback_timing`` phases.
    """

    for name in (
        "_rollback_gpu_clusters",
        "_verify_and_complete_rollback",
        "cleanup_candidate_rollout_state",
    ):
        monkeypatch.setattr(ORCHESTRATION, name, lambda *_args, **_kwargs: None)
    builds: list[dict[str, Any]] = []

    def build(**kwargs: Any) -> dict[str, str]:
        builds.append(kwargs)
        return {
            RESTART.FORCE_ROLES_VARIABLE: ",".join(kwargs["force_restart_roles"]),
            RESTART.REPORT_VARIABLE: kwargs["role_restart_report"],
        }

    def apply(_release: Any, environment: dict[str, str]) -> None:
        forced = [
            role
            for role in environment[RESTART.FORCE_ROLES_VARIABLE].split(",")
            if role
        ]
        site.roll(forced)
        Path(environment[RESTART.REPORT_VARIABLE]).write_text(
            json.dumps(
                {
                    "forced": forced,
                    "fallback": [],
                    "pin_wait": CONVERGED,
                    "restart_wait": {},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ORCHESTRATION, "build_rollback_environment", build)
    monkeypatch.setattr(ORCHESTRATION, "_apply_rollback_cpu_environment", apply)
    saved: list[tuple[str, dict[str, Any]]] = []
    release = site.release(
        state=_executor_only_state(),
        runtime_image="candidate-runtime",
        node_installer_image="registry.example/installer:candidate",
        config=SimpleNamespace(
            namespace=NAMESPACE,
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
            schema_rollback_compatible=True,
            wheel=SimpleNamespace(name="wheel"),
            for_rollback=lambda digest, admin_config: SimpleNamespace(
                digest=digest, admin_config=admin_config
            ),
        ),
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _save_state=lambda phase, **updates: saved.append((phase, updates)),
        _config_map_sha=lambda *_args: "4" * 64,
    )
    previous = {
        **_rollback_previous(),
        "admin_config": {},
        "clusters": {},
        **site.previous(),
    }

    ORCHESTRATION.rollback_release(release, state=previous)

    phases = dict(saved)["rollback-cpu-restored"]["rollback_timing"]["phases"]
    return builds, phases


def test_an_executor_only_rollback_forces_no_cpu_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BOOT-020 case: both CPU stages run, neither restarts a tier."""

    builds, phases = _rollback(monkeypatch, Site())

    assert [build["force_restart_roles"] for build in builds] == [[], []], (
        "the controller stage, then the CPU restore"
    )
    for phase in ("controller_stage", "cpu_restore"):
        evidence = phases[phase]["details"]["cpu_restart"]
        assert evidence["restarted"] == {}, phase
        assert evidence["kept"] == EVERY_TIER, phase
        assert evidence["pin_wait"] == CONVERGED, phase


def test_a_secret_restore_keeps_the_forced_restart_through_the_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller stage restarts every tier for the restored Secret; the CPU
    restore cannot prove Pods started after the capture, so it restarts them
    again rather than trust them. A forced restart is never lost."""

    site = Site()
    site.secrets[EMAIL]["email-sender"] = CHANGED_EMAIL

    builds, phases = _rollback(monkeypatch, site)

    assert [build["force_restart_roles"] for build in builds] == [
        EVERY_TIER,
        EVERY_TIER,
    ]
    assert phases["controller_stage"]["details"]["cpu_restart"]["restarted"] == {
        role: ["secret-changed"] for role in EVERY_TIER
    }
    assert phases["cpu_restore"]["details"]["cpu_restart"]["restarted"] == {
        role: ["pod-started-after-capture", "spec-changed"] for role in EVERY_TIER
    }


# --------------------------------------------------------------------------
# The limits the change is measured against
# --------------------------------------------------------------------------


def test_the_boot020_rto_limits_are_unchanged() -> None:
    """The fix earns the Executor-only budget back; it does not move it."""

    assert boot020.RTO_LIMITS == {
        "control_plane": {"safe": 1200.0, "full": 1500.0},
        "executor": {"safe": 900.0, "full": 1200.0, "resume": 1200.0},
        "agent": {"safe": 1800.0, "full": 2100.0},
        "full": {"safe": 2700.0, "full": 3000.0},
    }
