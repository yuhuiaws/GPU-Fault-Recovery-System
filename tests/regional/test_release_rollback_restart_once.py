"""A rollback rolls each CPU tier once, even when both of its CPU stages run.

An Agent or FULL rollback (or any whose controller stage restores a start-up
input, such as the email Secret) restarts every CPU tier in the controller
stage. The CPU restore used to judge each tier against the forward capture
only, found its Pods newer than the capture, and restarted it again: a second
full roll, about 240 s on the 3-node site (api-ha surge-first, one Deployment
after the other), against BOOT-020's Agent and FULL T_safe limits.

The controller stage now records what each tier it restarted starts from
(``restarted_onto``), and the CPU restore keeps a tier whose restart is still
live and was onto exactly its own runtime image, container env and role
ConfigMaps; any other tier restarts as before. These tests pin the record, the
CPU restore's judgement of it, and -- through ``rollback_release`` on a
simulated apply and clock -- that the rollback pays one roll per tier and
leaves no Pod on a candidate start-up input.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.container_env_snapshot import pod_container_env
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_progress as PROGRESS
from gpu_fault_release import regional_release_rollback_restart as RESTART
from gpu_fault_release import regional_release_timing as TIMING
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_rollback_context import (
    CONTAINER_ENV_FILE_VARIABLE,
)
from tests.regional.test_release_rollback_restart import (
    CHANGED_EMAIL,
    CONVERGED,
    DEPLOYMENTS,
    EMAIL,
    EMAIL_DATA,
    EVERY_TIER,
    NAMESPACE,
    PREVIOUS_PINS,
    RUNTIME,
    Site,
    _container_env,
    _narrated,
    _pod,
)
from tests.regional.test_release_rollback_state import _rollback_previous

WORKER = DEPLOYMENTS["worker"]
CANDIDATE_RUNTIME = "registry.example/runtime@sha256:" + "c" * 64


# --------------------------------------------------------------------------
# The controller stage's restarts, and the CPU restore after them
# --------------------------------------------------------------------------


def _controller_stage(
    site: Site,
    *,
    image: str = RUNTIME,
    agent_config: dict[str, str] | None = None,
    apply: Callable[[list[str]], object] | None = None,
) -> dict[str, Any]:
    """The controller stage on ``site``: proof, simulated apply, record.

    Returns the ``rollback_timing`` it leaves once completed.
    """

    with RESTART.RollbackRestartPlan(site.release(), site.previous()) as plan:
        plan.restore_secret_and_registry()
        if agent_config is not None:
            plan.patch_controller_config(agent_config)
        forced = plan.environment_arguments()["force_restart_roles"]
        if apply is None:
            site.apply(forced, image=image)
        else:
            apply(forced)
        plan.record_restarts()
        details = plan.details()
    return {
        "phases": {
            RESTART.CONTROLLER_STAGE: {"status": "COMPLETED", "details": details}
        }
    }


def _cpu_restore(
    site: Site, timing: dict[str, Any], *, image: str = RUNTIME
) -> tuple[list[str], dict[str, Any]]:
    """The CPU restore's decision and evidence after ``timing``'s controller
    stage, against its own target: ``image`` and the captured container env."""

    previous = site.previous()
    with RESTART.RollbackRestartPlan(site.release(), previous, timing) as plan:
        plan.restore_secret_and_registry()
        plan.restore_role_config_maps(previous["cpu_role_config_maps"])
        arguments = plan.environment_arguments(
            image, previous["cpu_role_container_env"]
        )
        return arguments["force_restart_roles"], plan.details()["cpu_restart"]


def _restarted_onto(timing: dict[str, Any]) -> dict[str, Any]:
    details = timing["phases"][RESTART.CONTROLLER_STAGE]["details"]
    return dict(details["cpu_restart"][RESTART.RESTART_RECORD_KEY])


def _secret_changed(site: Site) -> Site:
    """The forward transaction changed the email Secret: the controller stage
    restarts every tier for its restore."""

    site.secrets[EMAIL]["email-sender"] = CHANGED_EMAIL
    return site


def test_the_controller_stage_records_what_each_restarted_tier_starts_from(
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = _secret_changed(Site())

    timing = _controller_stage(site)

    record = _restarted_onto(timing)
    assert sorted(record) == sorted(EVERY_TIER)
    for role, name in DEPLOYMENTS.items():
        live = site.deployments[name]
        assert record[role] == {
            "spec_sha256": RESTART.deployment_spec_sha256(live),
            "restarted_at": "restart-1",
            "runtime_image": RUNTIME,
            "container_env_sha256": canonical_sha256(pod_container_env(live)),
            "config_maps_sha256": canonical_sha256(site.tier_config_maps(name)),
        }, role
    (line,) = _narrated(capsys, "rollback-cpu-restart-record")
    assert " recorded=spool,worker,ingress " in line
    recorded = json.dumps(timing)
    for value in (*EMAIL_DATA.values(), CHANGED_EMAIL):
        assert value not in recorded, "the record holds digests, never Secret data"


def test_only_the_tiers_the_controller_stage_restarted_are_recorded() -> None:
    site = Site()
    site.config_maps[f"{WORKER}-config-core"]["GPU_FAULT_REQUIRED_POLICY_VERSION"] = (
        "candidate-catalog"
    )

    timing = _controller_stage(site, agent_config=dict(PREVIOUS_PINS))

    assert sorted(_restarted_onto(timing)) == ["worker"]
    details = timing["phases"][RESTART.CONTROLLER_STAGE]["details"]["cpu_restart"]
    assert details["restarted"] == {"worker": ["config-core-changed"]}


def test_a_restart_token_the_apply_did_not_mint_is_not_recorded() -> None:
    """A tier proves the controller stage's restart only by a token minted
    after the stage restored its inputs; the one it already had proves
    nothing about when its Pods resolved them."""

    site = _secret_changed(Site())

    timing = _controller_stage(site, apply=lambda _forced: site.apply([]))

    assert _restarted_onto(timing) == {}


@pytest.mark.parametrize(
    "previous_change",
    [
        pytest.param(
            lambda previous: previous.pop(RESTART.BASELINE_KEY), id="no-baseline"
        ),
        pytest.param(
            lambda previous: previous.update(cpu_role_config_maps={}),
            id="no-config-map-snapshot",
        ),
    ],
)
def test_nothing_is_recorded_without_the_proof_s_reads_or_a_snapshot(
    previous_change: Callable[[dict[str, Any]], object],
) -> None:
    """Without the token read before the apply, or the ConfigMap snapshot the
    CPU restore restores, a restart cannot be judged later: no record, so the
    CPU restore restarts those tiers again."""

    site = _secret_changed(Site())
    previous = site.previous()
    previous_change(previous)
    with RESTART.RollbackRestartPlan(site.release(), previous) as plan:
        plan.restore_secret_and_registry()
        forced = plan.environment_arguments()["force_restart_roles"]
        assert site.apply(forced) == EVERY_TIER
        plan.record_restarts()
        assert plan.details()["cpu_restart"][RESTART.RESTART_RECORD_KEY] == {}


def test_a_record_read_that_fails_records_nothing_and_decides_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = _secret_changed(Site())

    def apply_then_lose_the_reads(forced: list[str]) -> None:
        site.apply(forced)
        site.unreadable.add("configmap")

    timing = _controller_stage(site, apply=apply_then_lose_the_reads)

    assert _restarted_onto(timing) == {}
    details = timing["phases"][RESTART.CONTROLLER_STAGE]["details"]["cpu_restart"]
    assert details["restarted"] == {role: ["secret-changed"] for role in EVERY_TIER}, (
        "the decision was taken before the apply; a failed read only loses the record"
    )
    lines = _narrated(capsys, "rollback-cpu-restart-record")
    assert [line for line in lines if " error=ReleaseError " in line], (
        "the lost record is narrated with the read's error"
    )


def test_the_cpu_restore_keeps_a_tier_the_controller_stage_restarted_onto_its_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = _secret_changed(Site())
    timing = _controller_stage(site)
    pods = copy.deepcopy(site.pods)

    forced, evidence = _cpu_restore(site, timing)

    assert forced == []
    assert evidence["restarted"] == {} and evidence["kept"] == EVERY_TIER
    assert evidence["controller_stage_restarted"] == EVERY_TIER
    assert site.pods == pods
    *_controller, line = _narrated(capsys, "rollback-cpu-restart")
    assert (
        " forced=none reasons=none controller_stage_restarted=spool,worker,ingress "
    ) in line


def _old_pod_left(site: Site) -> None:
    site.pods.append(site.started(_pod(WORKER, 9, site.captured_at - 3600.0)))


def _container_crashing(site: Site) -> None:
    (pod, *_rest) = [
        pod for pod in site.pods if pod["metadata"]["labels"]["app"] == WORKER
    ]
    pod["status"]["containerStatuses"][0]["state"] = {"waiting": {}}


def _processor_changed(site: Site) -> None:
    site.config_maps[f"{WORKER}-config-processor"]["GPU_FAULT_PROCESSOR_WORKERS"] = "32"


def _processor_restored(site: Site) -> None:
    # An earlier CPU restore attempt restored the ConfigMaps and then failed.
    name = f"{WORKER}-config-processor"
    site.config_maps[name] = dict(site.snapshots[name])


def _env_added(site: Site) -> None:
    site.template_env[WORKER] = _container_env(WORKER, GPU_FAULT_NEW_KNOB="1")


@pytest.mark.parametrize(
    ("before", "between", "image", "expected"),
    [
        pytest.param(
            None,
            None,
            CANDIDATE_RUNTIME,
            {role: ["runtime-image-changed"] for role in EVERY_TIER},
            id="runtime-image",
        ),
        pytest.param(
            _env_added,
            None,
            RUNTIME,
            {"worker": ["container-env-changed"]},
            id="container-env",
        ),
        pytest.param(
            _processor_changed,
            _processor_restored,
            RUNTIME,
            {"worker": ["config-maps-changed"]},
            id="config-maps-on-resume",
        ),
        pytest.param(
            None,
            lambda site: site.deployments[WORKER]["spec"].update(replicas=3),
            RUNTIME,
            {"worker": ["spec-changed"]},
            id="spec",
        ),
        pytest.param(
            None,
            _old_pod_left,
            RUNTIME,
            {"worker": ["pod-not-from-controller-stage"]},
            id="old-pod",
        ),
        pytest.param(
            None,
            _container_crashing,
            RUNTIME,
            {"worker": ["container-not-running"]},
            id="container",
        ),
    ],
)
def test_the_cpu_restore_restarts_a_tier_the_controller_stage_restarted_elsewhere(
    before: Callable[[Site], None] | None,
    between: Callable[[Site], None] | None,
    image: str,
    expected: dict[str, list[str]],
) -> None:
    """A tier restarted onto other start-up inputs than the CPU restore's --
    the candidate's runtime image, the checked-out template's env instead of
    the captured one, ConfigMaps the restore has to put back -- or one whose
    Deployment or Pods are no longer that restart's, restarts again."""

    site = _secret_changed(Site())
    if before is not None:
        before(site)
    timing = _controller_stage(site, image=image)
    if between is not None:
        between(site)

    forced, evidence = _cpu_restore(site, timing)

    assert forced == list(expected)
    assert evidence["restarted"] == expected
    assert evidence["controller_stage_restarted"] == EVERY_TIER


def test_a_deleted_deployment_restarts_its_tier_after_the_controller_stage() -> None:
    site = _secret_changed(Site())
    timing = _controller_stage(site)
    del site.deployments[WORKER]

    forced, evidence = _cpu_restore(site, timing)

    assert forced == ["worker"]
    assert evidence["restarted"] == {"worker": ["deployment-missing"]}


def test_without_the_restore_s_target_the_record_is_not_used() -> None:
    """The record is compared with what the stage restarts a tier onto; a
    stage that names no target proves every tier against the capture."""

    site = _secret_changed(Site())
    timing = _controller_stage(site)

    with RESTART.RollbackRestartPlan(site.release(), site.previous(), timing) as plan:
        forced = plan.environment_arguments()["force_restart_roles"]
        assert "controller_stage_restarted" not in plan.details()["cpu_restart"]

    assert forced == EVERY_TIER


def _entry(**changes: Any) -> dict[str, Any]:
    record = {field: f"{field}-value" for field in RESTART.RESTART_RECORD_FIELDS}
    return {
        "phases": {
            RESTART.CONTROLLER_STAGE: {
                "status": "COMPLETED",
                "details": {
                    "cpu_restart": {
                        RESTART.RESTART_RECORD_KEY: {
                            "worker": record,
                            "spool": {**record, **changes},
                            "adot": record,
                        }
                    }
                },
            }
        }
    }


def test_only_a_completed_controller_stage_s_whole_entries_count() -> None:
    """A controller stage that started or failed may not have restarted what it
    recorded, and an entry missing a field cannot be compared."""

    assert sorted(RESTART.controller_stage_restarts(_entry())) == ["spool", "worker"]
    assert sorted(RESTART.controller_stage_restarts(_entry(spec_sha256=""))) == [
        "worker"
    ]
    assert sorted(RESTART.controller_stage_restarts(_entry(restarted_at=7))) == [
        "worker"
    ]
    for status in ("STARTED", "FAILED"):
        timing = _entry()
        timing["phases"][RESTART.CONTROLLER_STAGE]["status"] = status
        assert RESTART.controller_stage_restarts(timing) == {}, status
    for timing in (None, {}, {"phases": {}}, {"phases": {"controller_stage": []}}):
        assert RESTART.controller_stage_restarts(timing) == {}


# --------------------------------------------------------------------------
# Through the rollback, on the rollback's clock
# --------------------------------------------------------------------------

#: What an apply costs the rollback, as BOOT-020 measured it on the 3-node
#: site: a forced roll of api-ha is surge-first, one Pod at a time (about 47 s
#: for each of its three), and the control and spool workers follow one
#: Deployment after the other -- 228-249 s for the three together. Rendering,
#: the schedulability preflight and the /healthz pin proof come on top.
ROLL_SECONDS = {"spool": 50.0, "worker": 50.0, "ingress": 141.0}
APPLY_SECONDS = 20.0
ONE_ROLL = sum(ROLL_SECONDS.values())
#: The components each scenario's forward transaction began, globally and on
#: the GPU cluster: what the compensation plan rolls back.
TRANSACTIONS = {
    "executor": ((), (DIFF.ReleaseComponent.EXECUTOR,)),
    "agent": ((), (DIFF.ReleaseComponent.AGENT,)),
    "full": (
        (DIFF.ReleaseComponent.CPU_STAGE, DIFF.ReleaseComponent.CPU_FINALIZE),
        (DIFF.ReleaseComponent.EXECUTOR, DIFF.ReleaseComponent.AGENT),
    ),
}


def _transaction(scenario: str) -> dict[str, Any]:
    global_components, cluster_components = TRANSACTIONS[scenario]
    state: dict[str, Any] = {
        "execution_plan": {
            "nodes": [
                component.value
                for component in (
                    *global_components,
                    *cluster_components,
                    DIFF.ReleaseComponent.VERIFY,
                )
            ]
        },
        "completed_phases": ["uploaded", "schema-ready"],
    }
    for component, cluster_id in (
        *((component, None) for component in global_components),
        *((component, "gpu-a") for component in cluster_components),
    ):
        state["component_progress"] = PROGRESS.update_component_progress(
            state,
            component,
            PROGRESS.PROGRESS_STARTED,
            cluster_id=cluster_id,
            observed_at_epoch=time.time(),
        )
    return state


class Rollback:
    """``rollback_release`` over ``site`` with both CPU stages live.

    The apply is simulated as ``Site.apply`` (the role-split apply and
    Kubernetes): the stage's runtime image and container env, a new restart
    token for each forced tier, a roll of each tier whose template changed. It
    costs ``APPLY_SECONDS`` plus ``ROLL_SECONDS`` per rolled tier on the
    rollback's clock, which every ``rollback_timing`` entry is measured on.
    The data restore, the rollout cleanup and the final verification are
    no-ops; ``runtime_image`` is the candidate's, the controller stage's.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        site: Site,
        scenario: str = "executor",
        *,
        runtime_image: str = RUNTIME,
    ) -> None:
        for name in (
            "_rollback_gpu_clusters",
            "_verify_and_complete_rollback",
            "cleanup_candidate_rollout_state",
            "validate_rollback_node_template",
        ):
            monkeypatch.setattr(ORCHESTRATION, name, lambda *_args, **_kwargs: None)
        monkeypatch.setattr(ORCHESTRATION, "build_rollback_environment", self.build)
        monkeypatch.setattr(
            ORCHESTRATION, "_apply_rollback_cpu_environment", self.apply
        )
        monkeypatch.setattr(TIMING, "time", self)
        self.now = time.time()
        self.site = site
        self.builds: list[dict[str, Any]] = []
        self.rolled: list[list[str]] = []
        self.saved: list[tuple[str, dict[str, Any]]] = []
        self.failing_applies: set[int] = set()
        self.release = site.release(
            state=_transaction(scenario),
            runtime_image=runtime_image,
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
            _save_state=lambda phase, **updates: self.saved.append((phase, updates)),
            _config_map_sha=lambda *_args: "4" * 64,
        )
        self.previous = {
            **_rollback_previous(),
            "admin_config": {},
            "clusters": {},
            "cluster_runtime_profiles": {},
            **site.previous(),
        }

    def time(self) -> float:
        """The rollback's clock (``regional_release_timing`` reads it)."""

        return self.now

    def build(self, **kwargs: Any) -> dict[str, str]:
        self.builds.append(kwargs)
        environment = {
            "GPU_FAULT_RUNTIME_IMAGE": kwargs["runtime_image"],
            RESTART.FORCE_ROLES_VARIABLE: ",".join(kwargs["force_restart_roles"]),
            RESTART.REPORT_VARIABLE: kwargs["role_restart_report"],
        }
        if kwargs.get("previous_container_env_file"):
            environment[CONTAINER_ENV_FILE_VARIABLE] = kwargs[
                "previous_container_env_file"
            ]
        return environment

    def apply(self, _release: Any, environment: dict[str, str]) -> None:
        if len(self.rolled) in self.failing_applies:
            self.failing_applies.discard(len(self.rolled))
            raise ReleaseError("simulated apply failure")
        forced = [
            role
            for role in environment[RESTART.FORCE_ROLES_VARIABLE].split(",")
            if role
        ]
        snapshot = environment.get(CONTAINER_ENV_FILE_VARIABLE)
        rolled = self.site.apply(
            forced,
            image=environment["GPU_FAULT_RUNTIME_IMAGE"],
            env=(
                json.loads(Path(snapshot).read_text(encoding="utf-8"))
                if snapshot
                else None
            ),
        )
        self.rolled.append(rolled)
        self.now += APPLY_SECONDS + sum(ROLL_SECONDS[role] for role in rolled)
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

    def run(self) -> dict[str, Any]:
        """Roll back; returns the saved ``rollback_timing``."""

        ORCHESTRATION.rollback_release(self.release, state=self.previous)
        return self.timing

    def resume(self) -> dict[str, Any]:
        """Roll back again from the last saved state, as a new run would."""

        phase, updates = self.saved[-1]
        self.release.state = {**self.release.state, **updates, "phase": phase}
        return self.run()

    @property
    def timing(self) -> dict[str, Any]:
        timing: dict[str, Any] = self.saved[-1][1]["rollback_timing"]
        return timing


def _cpu_restart(timing: dict[str, Any], phase: str) -> dict[str, Any]:
    evidence: dict[str, Any] = timing["phases"][phase]["details"]["cpu_restart"]
    return evidence


def _seconds(timing: dict[str, Any], phase: str) -> float:
    seconds: float = timing["phases"][phase]["duration_seconds"]
    return seconds


def _assert_no_pod_keeps_a_candidate_start_up_input(site: Site) -> None:
    """BOOT-020's final check, per Pod: each starts from exactly the previous
    image, the captured container env and the captured role ConfigMaps."""

    for pod in site.pods:
        name = pod["metadata"]["labels"]["app"]
        assert pod["start_up"] == site.restored_start_up(name), pod["metadata"]["name"]


def test_an_executor_only_rollback_forces_no_cpu_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BOOT-020 case: both CPU stages run, neither restarts a tier."""

    rollback = Rollback(monkeypatch, Site())

    timing = rollback.run()

    assert [build["force_restart_roles"] for build in rollback.builds] == [[], []], (
        "the controller stage, then the CPU restore"
    )
    assert rollback.rolled == [[], []]
    for phase in ("controller_stage", "cpu_restore"):
        evidence = _cpu_restart(timing, phase)
        assert evidence["restarted"] == {}, phase
        assert evidence["kept"] == EVERY_TIER, phase
        assert evidence["pin_wait"] == CONVERGED, phase
        assert _seconds(timing, phase) == APPLY_SECONDS, phase
    assert timing["t_safe_seconds"] == 2 * APPLY_SECONDS


def _secret(site: Site) -> None:
    _secret_changed(site)


def _agent(site: Site) -> None:
    # A node-runtime release whose catalog moved the policy version; its CPU
    # stage rolled every tier onto it.
    site.forward(
        config_maps={
            f"{name}-config-core": {
                "GPU_FAULT_REQUIRED_POLICY_VERSION": "candidate-catalog"
            }
            for name in DEPLOYMENTS.values()
        }
    )


def _full(site: Site) -> None:
    # Candidate D: a suffixed Runtime Profile, rolled onto every tier.
    site.forward(
        config_maps={
            f"{name}-config-core": {
                "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1-boot020"
            }
            for name in DEPLOYMENTS.values()
        }
    )


@pytest.mark.parametrize(
    ("scenario", "forward", "reason"),
    [
        pytest.param("executor", _secret, "secret-changed", id="secret-restore"),
        pytest.param("agent", _agent, "config-core-changed", id="agent"),
        pytest.param("full", _full, "config-core-changed", id="full"),
    ],
)
def test_a_tier_the_controller_stage_restarted_is_rolled_once(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    forward: Callable[[Site], None],
    reason: str,
) -> None:
    """Agent and FULL rollbacks (and any controller stage that restores a
    start-up input) used to pay two full CPU rolls: the controller stage
    restarted every tier, and the CPU restore restarted each again because its
    Pods were newer than the capture. The CPU restore now keeps the Pods of
    that restart, which already start from what it restores, so the rollback's
    clock shows one roll -- about 240 s less T_safe -- and no Pod is left on a
    candidate input."""

    site = Site()
    forward(site)
    rollback = Rollback(monkeypatch, site, scenario)

    timing = rollback.run()

    controller = _cpu_restart(timing, "controller_stage")
    restore = _cpu_restart(timing, "cpu_restore")
    assert reason in controller["restarted"]["worker"]
    assert sorted(controller["restarted_onto"]) == sorted(EVERY_TIER)
    assert [build["force_restart_roles"] for build in rollback.builds] == [
        EVERY_TIER,
        [],
    ]
    assert restore["restarted"] == {}
    assert restore["kept"] == EVERY_TIER
    assert restore["controller_stage_restarted"] == EVERY_TIER
    assert rollback.rolled == [EVERY_TIER, []], "the CPU restore rolls nothing"
    assert _seconds(timing, "controller_stage") == APPLY_SECONDS + ONE_ROLL
    assert _seconds(timing, "cpu_restore") == APPLY_SECONDS
    assert timing["t_safe_seconds"] == 2 * APPLY_SECONDS + ONE_ROLL
    assert timing["t_safe_seconds"] < 2 * ONE_ROLL, "less than the old two rolls"
    _assert_no_pod_keeps_a_candidate_start_up_input(site)


def _candidate_image(site: Site) -> None:
    site.forward(image=CANDIDATE_RUNTIME)


def _candidate_env(site: Site) -> None:
    for name in DEPLOYMENTS.values():
        site.template_env[name] = _container_env(name, GPU_FAULT_NEW_KNOB="1")
    site.forward()


def _candidate_worker_config(site: Site) -> None:
    site.forward(
        config_maps={
            f"{WORKER}-config-processor": {"GPU_FAULT_PROCESSOR_WORKERS": "32"},
            f"{WORKER}-config-core": {
                "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1-boot020"
            },
        }
    )


@pytest.mark.parametrize(
    ("forward", "runtime_image", "again"),
    [
        pytest.param(
            _candidate_image,
            CANDIDATE_RUNTIME,
            {role: ["runtime-image-changed"] for role in EVERY_TIER},
            id="runtime-image",
        ),
        pytest.param(
            _candidate_env,
            RUNTIME,
            {role: ["container-env-changed"] for role in EVERY_TIER},
            id="container-env",
        ),
        pytest.param(
            _candidate_worker_config,
            RUNTIME,
            {"worker": ["config-maps-changed"]},
            id="role-config-map",
        ),
    ],
)
def test_a_tier_restarted_onto_another_input_is_restarted_again(
    monkeypatch: pytest.MonkeyPatch,
    forward: Callable[[Site], None],
    runtime_image: str,
    again: dict[str, list[str]],
) -> None:
    """The controller stage restarts onto the candidate's runtime image and the
    checked-out template's container env, and leaves the role ConfigMaps it
    does not patch as the forward transaction wrote them; when one of those is
    not what the CPU restore restores, the second roll is the price of putting
    every Pod on the previous start-up inputs."""

    site = Site()
    forward(site)
    rollback = Rollback(monkeypatch, site, "full", runtime_image=runtime_image)

    timing = rollback.run()

    assert _cpu_restart(timing, "cpu_restore")["restarted"] == again
    assert rollback.rolled == [EVERY_TIER, list(again)]
    assert _seconds(timing, "cpu_restore") == APPLY_SECONDS + sum(
        ROLL_SECONDS[role] for role in again
    )
    _assert_no_pod_keeps_a_candidate_start_up_input(site)


def test_a_resumed_cpu_restore_judges_the_role_config_maps_the_restart_saw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first CPU restore restored the worker's ConfigMaps and failed before
    its apply rolled anything. The resumed one finds them already restored --
    its own restore changes nothing -- but the controller stage's record says
    what the worker's Pods resolved, so the worker still restarts; the other
    tiers keep the controller stage's Pods."""

    site = Site()
    _candidate_worker_config(site)
    rollback = Rollback(monkeypatch, site, "full")
    rollback.failing_applies.add(1)

    with pytest.raises(ReleaseError, match="simulated apply failure"):
        rollback.run()
    assert rollback.saved[-1][0] == "rollback-cpu-restoring"
    assert (
        site.tier_config_maps(WORKER) == site.restored_start_up(WORKER)["config_maps"]
    ), "the failed attempt restored them"
    timing = rollback.resume()

    restore = _cpu_restart(timing, "cpu_restore")
    assert restore["restarted"] == {"worker": ["config-maps-changed"]}
    assert restore["controller_stage_restarted"] == EVERY_TIER
    assert rollback.rolled == [EVERY_TIER, ["worker"]]
    assert len(rollback.builds) == 3, "the controller stage is not run again"
    _assert_no_pod_keeps_a_candidate_start_up_input(site)
