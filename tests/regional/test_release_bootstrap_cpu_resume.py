"""A bootstrap reports success only over a control plane it applied or proved.

Bootstrap cleanup scales every CPU Deployment to 0 replicas and keeps the
release metadata. The live CPU check that lets a resumed attempt skip its
schema ensure and CPU apply compared each Deployment's rollout with its own
``spec.replicas``, so 0 of 0 passed: after a cleanup, an attempt that failed
or died before its CPU apply was followed by one that skipped it and wrote
``complete`` over a control plane of 0 replicas. The bootstrap checkpoint
also kept an older attempt's ``resume_phase``, which a ``complete`` state
carried on: a deploy that bootstrapped over a ``complete`` state whose
control plane was gone resumed past the CPU apply it no longer had.

Every deploy here runs as the administrator CLI runs it, the preflight
(``build_deploy_preflight_report``) and then the deploy (``run_deploy`` into
``RegionalRelease.bootstrap``) as separate unpinned processes with the real
manifest pin enforcement, over an in-memory CPU cluster: the real live CPU
check reads Deployments rendered from the generated role manifests and stamped
from the real CPU apply environment, as the role-split apply stamps them.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_prerequisite_handoff as HANDOFF
from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional import test_release_bootstrap_handoff as handoff_support
from tests.regional._release_orchestrator_support import ROOT
from tests.regional.test_release_bootstrap_foreign_takeover import (
    adopted_predecessor,
    deploy,
    deploy_preflight,
    real_manifest_pin,
    stub_bootstrap,
)
from tests.regional.test_release_bootstrap_handoff import select_candidate

handoff_release = handoff_support.handoff_release

GENERATED = ROOT / "deploy/control-plane/regional/generated"
DEFAULT_RUNTIME_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"
CPU = ["kubectl", "--context", "cpu"]
METADATA = "gpu-fault-release-metadata"
ADOT = next(
    name
    for name in inventory.CPU_DEPLOYMENTS
    if name not in inventory.CPU_RUNTIME_DEPLOYMENTS
)
# The role Deployments the CPU apply renders, their manifest and role digest.
ROLES = {
    "gpu-fault-telemetry-spool-worker": ("gpu-fault-telemetry-spool-worker", "SPOOL"),
    "gpu-fault-control-worker": ("gpu-fault-control-worker", "WORKER"),
    "gpu-fault-api-ha": ("gpu-fault-api-ha-ingress", "INGRESS"),
}
METADATA_PINS = {
    "required-agent-artifact-sha256": "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256",
    "required-agent-compatibility-digest": (
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST"
    ),
    "required-regional-executor-artifact-sha256": (
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256"
    ),
    "required-regional-executor-compatibility-digest": (
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST"
    ),
    "required-agent-config-digest": "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST",
}


class Killed(BaseException):
    """The deploy process dies: no failure checkpoint and no cleanup run."""


def rolled_out(replicas: int, generation: int) -> dict[str, int]:
    status = {"observedGeneration": generation}
    if replicas:
        status.update(
            replicas=replicas,
            readyReplicas=replicas,
            updatedReplicas=replicas,
            availableReplicas=replicas,
        )
    return status


class LiveCpu:
    """The CPU cluster's control-plane Deployments and release metadata."""

    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.deployments: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, str] = {}
        # Raise after this role's Deployment, as an apply that dies part-way.
        self.fail_after: str | None = None

    def write(self, name: str, document: dict[str, Any], replicas: int) -> None:
        old = self.deployments.get(name)
        document["metadata"]["uid"] = old["metadata"]["uid"] if old else f"{name}-uid"
        generation = old["metadata"]["generation"] + 1 if old else 1
        document["metadata"]["generation"] = generation
        document["spec"]["replicas"] = replicas
        document["status"] = rolled_out(replicas, generation)
        self.deployments[name] = document

    def apply(self) -> None:
        """What a CPU apply that returns leaves: every role rolled out."""
        environment = RENDERING.build_cpu_apply_environment(
            self.instance, finalize=True
        )
        self.metadata = {key: environment[name] for key, name in METADATA_PINS.items()}
        replicas = {
            "WORKER": int(environment["GPU_FAULT_CONTROL_WORKER_REPLICAS"]),
            "SPOOL": int(environment["GPU_FAULT_TELEMETRY_SPOOL_REPLICAS"]),
        }
        for name, (manifest, role) in ROLES.items():
            text = (
                (GENERATED / f"{manifest}.yaml")
                .read_text(encoding="utf-8")
                .replace(
                    "gpu-fault-control-plane-wheel-0100",
                    environment["GPU_FAULT_WHEEL_CONFIGMAP"],
                )
                .replace(
                    "namespace: gpu-fault-system",
                    f"namespace: {environment['GPU_FAULT_NAMESPACE']}",
                )
                .replace(DEFAULT_RUNTIME_IMAGE, environment["GPU_FAULT_RUNTIME_IMAGE"])
            )
            document = next(
                item
                for item in yaml.safe_load_all(text)
                if item and item.get("kind") == "Deployment"
            )
            document["metadata"].setdefault("annotations", {})[
                "gpu-fault.io/admin-config-sha256"
            ] = environment["GPU_FAULT_ADMIN_CONFIG_SHA256"]
            document["spec"]["template"]["metadata"].setdefault(
                "annotations", {}
            ).update(
                {
                    "gpu-fault.io/artifact-sha256": environment[
                        "GPU_FAULT_WHEEL_SHA256"
                    ],
                    "gpu-fault.io/runtime-image": environment[
                        "GPU_FAULT_RUNTIME_IMAGE"
                    ],
                    "gpu-fault.io/notification-config-sha256": environment[
                        "GPU_FAULT_NOTIFICATION_CONFIG_SHA256"
                    ],
                    "gpu-fault.io/role-config-sha256": environment[
                        f"GPU_FAULT_ADMIN_CONFIG_{role}_SHA256"
                    ],
                    "gpu-fault.io/runtime-profile-version": environment[
                        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"
                    ],
                }
            )
            self.write(name, document, replicas.get(role, document["spec"]["replicas"]))
            if self.fail_after == name:
                self.fail_after = None
                raise ReleaseError(f"{name} rollout did not finish")

    def observability(self) -> None:
        self.write(
            ADOT,
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": ADOT, "namespace": self.namespace},
                "spec": {"template": {"metadata": {}, "spec": {}}},
            },
            1,
        )

    def scale(self, name: str, replicas: int) -> None:
        document = self.deployments.get(name)
        if document is not None:
            self.write(name, document, replicas)

    @property
    def namespace(self) -> str:
        return str(self.instance.config.namespace)

    def read(self, arguments: list[str]) -> str | None:
        """The model's answer to a CPU ``get``, or None when it is not modelled."""
        if arguments[: len(CPU)] != CPU or "get" not in arguments:
            return None
        kind, name = arguments[arguments.index("get") + 1 :][:2]
        if kind == "deployment" and name in inventory.CPU_DEPLOYMENTS:
            document = self.deployments.get(name)
            return json.dumps(document) if document else ""
        if kind == "configmap" and name == METADATA:
            if not self.metadata:
                return ""
            return json.dumps(
                {
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": METADATA,
                        "namespace": self.namespace,
                        "uid": "metadata-uid",
                    },
                    "data": self.metadata,
                }
            )
        return None

    def up(self, image: str) -> bool:
        """Every role at its replicas, ready, on ``image``; ADOT up too."""
        roles = [self.deployments.get(name) for name in ROLES]
        adot = self.deployments.get(ADOT)
        return bool(
            adot
            and adot["status"].get("readyReplicas")
            and all(
                document
                and document["spec"]["template"]["spec"]["containers"][0]["image"]
                == image
                and document["status"].get("readyReplicas", 0)
                == document["spec"]["replicas"]
                for document in roles
            )
            and roles[2]["spec"]["replicas"] == 3
            and roles[1]["spec"]["replicas"] > 0
        )


def live_site(
    instance: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[LiveCpu, list[Any]]:
    """The real bootstrap and live CPU check over the in-memory CPU cluster."""
    gpu_calls = stub_bootstrap(instance, monkeypatch)
    live = LiveCpu(instance)
    events = instance.runner.events
    run = instance.runner.run

    def cpu_run(arguments: list[str], **kwargs: Any) -> str:
        answer = live.read(list(arguments))
        return str(run(arguments, **kwargs)) if answer is None else answer

    def apply_cpu(*, finalize: bool, **_kwargs: Any) -> None:
        events.append("cpu-finalize" if finalize else "cpu-stage")
        live.apply()

    def observability() -> None:
        events.append("observability")
        live.observability()

    def scale(kubectl: list[str], deployment: str, replicas: int, **_: Any) -> None:
        if kubectl[: len(CPU)] == CPU:
            live.scale(deployment, replicas)
        else:
            events.append("scale-if-present")

    def cpu_is_current() -> bool:
        events.append("live-cpu-check")
        return ADMIN.bootstrap_cpu_is_current(instance)

    def gpu_clusters(_release: Any, completed: set[str]) -> None:
        gpu_calls.append(sorted(completed))
        completed.update(target.cluster_id for target in instance.config.clusters)

    monkeypatch.setattr(instance.runner, "run", cpu_run)
    monkeypatch.setattr(instance, "_apply_cpu", apply_cpu)
    monkeypatch.setattr(instance, "_apply_control_plane_observability", observability)
    monkeypatch.setattr(instance, "_scale_if_present", scale)
    monkeypatch.setattr(instance, "_bootstrap_cpu_is_current", cpu_is_current)
    monkeypatch.setattr(rollout, "bootstrap_gpu_clusters", gpu_clusters)
    monkeypatch.setattr(
        instance,
        "_config_map_data",
        lambda name: STATE.config_map_data(instance, name),
        raising=False,
    )
    monkeypatch.setattr(
        instance,
        "_deployment_wheel",
        lambda arguments, name: STATE.deployment_wheel(instance, arguments, name),
        raising=False,
    )
    # The alert channel variables; the model reads none of them.
    monkeypatch.setattr(
        instance.config, "notification_environment", lambda: {}, raising=False
    )
    return live, gpu_calls


def fail_once(
    instance: Any, monkeypatch: pytest.MonkeyPatch, name: str, *, kill: bool = False
) -> None:
    """The next call of bootstrap step ``name`` fails, or the process dies in it."""
    step = getattr(instance, name)

    def failing(*_args: Any, **_kwargs: Any) -> None:
        monkeypatch.setattr(instance, name, step)
        if kill:
            raise Killed(name)
        raise ReleaseError(f"{name} failed")

    monkeypatch.setattr(instance, name, failing)


def candidate_b(instance: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """Candidate B's bootstrap over candidate A's adopted, cleaned-up attempt."""
    adopted_predecessor(instance, "bootstrap-cleaned")
    select_candidate(instance)
    digest = real_manifest_pin(instance, monkeypatch)
    instance.rendered_manifest_digest = digest
    return digest


def attempt(instance: Any) -> list[str]:
    """One deploy, preflight then deploy; the events it recorded."""
    mark = len(instance.runner.events)
    deploy_preflight(instance)
    deploy(instance)
    return instance.runner.events[mark:]


def failed_attempt(instance: Any, *, killed: bool = False) -> list[str]:
    mark = len(instance.runner.events)
    deploy_preflight(instance)
    with pytest.raises(Killed if killed else ReleaseError):
        deploy(instance)
    return instance.runner.events[mark:]


@pytest.mark.parametrize(
    ("step", "killed"),
    [
        ("_prepare_nlb", False),
        ("_prepare_nlb", True),
        ("_ensure_schema", False),
        ("_ensure_schema", True),
    ],
)
def test_resume_after_cleanup_reapplies_the_scaled_down_control_plane(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, step: str, killed: bool
) -> None:
    instance = handoff_release
    live, _gpu_calls = live_site(instance, monkeypatch)
    candidate_b(instance, monkeypatch)
    instance.config.auto_rollback = True
    fail_once(instance, monkeypatch, "_validate_release")
    first = failed_attempt(instance)
    assert {"schema", "cpu-finalize"} <= set(first), "the first attempt never applied"
    assert instance.runner.cloud_state["phase"] == "bootstrap-cleaned", (
        "the failed first attempt was not cleaned up"
    )
    assert live.deployments and all(
        value["spec"]["replicas"] == 0 for value in live.deployments.values()
    ), "bootstrap cleanup did not scale the control plane to 0"

    instance.config.auto_rollback = False
    fail_once(instance, monkeypatch, step, kill=killed)
    second = failed_attempt(instance, killed=killed)
    assert "cpu-finalize" not in second, "the second attempt reached its CPU apply"

    final = attempt(instance)
    state = instance.runner.cloud_state
    assert state["phase"] == "complete", f"the resume did not complete: {state}"
    assert {"schema", "cpu-finalize"} <= set(final), (
        "a resume after cleanup trusted the 0-replica control plane as applied"
    )
    assert live.up(instance.runtime_image), (
        "bootstrap completed over a control plane that is not up"
    )


@pytest.mark.parametrize("mode", ["preflight", "deploy-only", "older-plan"])
def test_complete_state_without_its_control_plane_bootstraps_it_again(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    instance = handoff_release
    live, gpu_calls = live_site(instance, monkeypatch)
    candidate_b(instance, monkeypatch)
    fail_once(instance, monkeypatch, "_wait_nlb")
    failed_attempt(instance)
    state = instance.runner.cloud_state
    assert (state["phase"], state["resume_phase"]) == (
        "bootstrap-failed",
        "bootstrap-cpu-ready",
    ), f"the first attempt did not stop at its CPU checkpoint: {state}"
    resumed = attempt(instance)
    assert instance.runner.cloud_state["phase"] == "complete" and (
        "cpu-finalize" not in resumed
    ), "the same candidate did not resume from its own CPU checkpoint"

    # The control plane is deleted out of band; the state still reads complete.
    live.deployments.clear()
    if mode == "older-plan":
        # The plan it finished was approved on a tree it no longer renders.
        instance.runner.cloud_state["approved_manifest_sha256"] = "b" * 64
    mark = len(gpu_calls)
    events = len(instance.runner.events)
    if mode == "preflight":
        deploy_preflight(instance)
    deploy(instance)
    final = instance.runner.events[events:]
    state = instance.runner.cloud_state
    assert state["phase"] == "complete" and REPAIR.REPAIR_KEY not in state, (
        f"the bootstrap over the complete state did not complete: {state}"
    )
    assert {"schema", "cpu-finalize"} <= set(final), (
        "the complete state lent the CPU checkpoint of the attempt that finished it"
    )
    assert gpu_calls[mark:] == [[]], "the complete state lent its completed clusters"
    assert live.up(instance.runtime_image), "bootstrap completed with no control plane"


@pytest.mark.parametrize("step", ["_ensure_schema", "_apply_cpu"])
def test_new_candidate_never_resumes_onto_the_replaced_candidates_control_plane(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    instance = handoff_release
    live, _gpu_calls = live_site(instance, monkeypatch)
    adopted_predecessor(instance, "bootstrap-endpoint-ready")
    # Candidate A's own control plane, applied and up.
    live.apply()
    live.observability()
    old_image = instance.runtime_image
    assert live.up(old_image), "candidate A's control plane is not up"
    wheel = (instance.wheel_sha, instance.wheel_cm)
    select_candidate(instance)
    # B ships A's wheel and pins on another runtime image: only the image
    # tells the two control planes apart.
    instance.wheel_sha, instance.wheel_cm = wheel
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)
    assert instance.runtime_image != old_image, "the candidates share an image"
    if step == "_apply_cpu":
        # B's apply dies after the control worker: the ingress is still A's.
        live.fail_after = "gpu-fault-control-worker"
    else:
        fail_once(instance, monkeypatch, step)
    first = failed_attempt(instance)
    state = instance.runner.cloud_state
    assert "live-cpu-check" not in first and state["phase"] == "bootstrap-failed", (
        "the takeover attempt trusted the replaced candidate's control plane"
    )
    assert REPAIR.REPAIR_KEY not in state, "the takeover attempt did not adopt"
    images = {
        name: live.deployments[name]["spec"]["template"]["spec"]["containers"][0][
            "image"
        ]
        for name in ROLES
    }
    assert old_image in images.values(), "A's control plane was fully replaced"
    assert ADMIN.bootstrap_cpu_is_current(instance) is False, (
        "the live CPU check took A's runtime image for B's control plane"
    )
    # The same control plane on B's image is B's: the image alone stopped it.
    saved = copy.deepcopy(live.deployments)
    for name in ROLES:
        template = live.deployments[name]["spec"]["template"]
        pod = template["spec"]
        for container in [*pod.get("initContainers", []), *pod["containers"]]:
            container["image"] = instance.runtime_image
        template["metadata"]["annotations"]["gpu-fault.io/runtime-image"] = (
            instance.runtime_image
        )
    assert ADMIN.bootstrap_cpu_is_current(instance) is True, (
        "the probe control plane differs from B's in more than the image"
    )
    live.deployments = saved

    final = attempt(instance)
    state = instance.runner.cloud_state
    assert state["phase"] == "complete", f"B's resume did not complete: {state}"
    assert {"schema", "cpu-finalize"} <= set(final), (
        "B resumed onto the replaced candidate's control plane"
    )
    assert live.up(instance.runtime_image), "B completed over A's runtime image"
    # Only an attempt that ensured B's schema here may ask the live check.
    assert ("live-cpu-check" in final) is (step == "_apply_cpu"), (
        "the live CPU check ran without, or was skipped despite, B's own schema"
    )


def next_candidate(
    instance: Any, monkeypatch: pytest.MonkeyPatch, name: str, digit: str
) -> str:
    """Candidate ``name``: its own images, wheel and rendered tree."""
    instance.release_id = f"candidate-{name}"
    instance.runtime_image = "registry/cpu@sha256:" + digit * 64
    instance.executor_image = "registry/executor@sha256:" + digit * 64
    instance.wheel_sha = digit * 64
    instance.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + instance.wheel_sha[:12]
    instance.approved_manifest_digest = None
    payload = {"cpu": {"gpu-fault-api": [{"kind": "Deployment", "candidate": name}]}}
    monkeypatch.setattr(
        RENDERING, "_render_release_payload", lambda _release: copy.deepcopy(payload)
    )
    digest = RENDERING.rendered_release_manifest_sha256(
        SimpleNamespace(approved_manifest_digest=None)
    )
    instance.rendered_manifest_digest = digest
    return digest


def legacy_complete_journal(
    instance: Any, monkeypatch: pytest.MonkeyPatch, live: LiveCpu
) -> dict[str, Any]:
    """What an older preflight left over candidate B's lost ``complete`` state.

    It journaled the state as it found it, with no takeover; the bootstrap
    entry, every later preflight and the standalone rollback refused that
    baseline as invalid bootstrap progress.
    """
    candidate_b(instance, monkeypatch)
    attempt(instance)
    assert instance.runner.cloud_state["phase"] == "complete", "B never completed"
    live.deployments.clear()
    with monkeypatch.context() as older:
        older.setattr(
            REPAIR, "foreign_bootstrap_baseline", lambda _release, state: state
        )
        deploy_preflight(instance)
    state = copy.deepcopy(instance.runner.cloud_state)
    record = state.get(REPAIR.REPAIR_KEY)
    assert (
        state["phase"] == "complete"
        and isinstance(record, dict)
        and "predecessor" not in record
        and record["binding"]["release_id"] == "candidate-b"
    ), "the older preflight's journal was not reproduced"
    return state


@pytest.mark.parametrize("preflight", [True, False])
def test_older_journal_over_a_lost_complete_state_is_retired_by_its_candidate(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, preflight: bool
) -> None:
    instance = handoff_release
    live, gpu_calls = live_site(instance, monkeypatch)
    record = legacy_complete_journal(instance, monkeypatch, live)[REPAIR.REPAIR_KEY]
    mark = len(gpu_calls)
    events = len(instance.runner.events)
    if preflight:
        deploy_preflight(instance)
        taken = instance.runner.cloud_state
        successor = taken[REPAIR.REPAIR_KEY]
        assert taken["phase"] == "bootstrap-started" and (
            successor["supersedes"]["attempt_id"] == record["attempt_id"]
            and successor["predecessor"]["phase"] == "complete"
        ), "the preflight did not take the lost complete state over"
        assert successor["previous_refresher"] == record["previous_refresher"], (
            "the successor dropped the refresher B first repaired over"
        )
    deploy(instance)
    final = instance.runner.events[events:]
    state = instance.runner.cloud_state
    assert state["phase"] == "complete" and REPAIR.REPAIR_KEY not in state, (
        f"B did not bootstrap over its older journal: {state}"
    )
    assert state[HANDOFF.HANDOFF_AUDIT_KEY]["attempt_id"] == record["attempt_id"], (
        "adoption lost the audit of the retired journal"
    )
    assert state[HANDOFF.PREDECESSOR_AUDIT_KEY]["phase"] == "complete", (
        "adoption lost the audit of the complete state it replaced"
    )
    assert {"schema", "cpu-finalize"} <= set(final) and gpu_calls[mark:] == [[]], (
        "the lost complete state lent its control plane or clusters"
    )
    assert live.up(instance.runtime_image), "bootstrap completed with no control plane"


def test_older_journal_over_a_lost_complete_state_is_retired_by_rollback(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    live, _gpu_calls = live_site(instance, monkeypatch)
    legacy_complete_journal(instance, monkeypatch, live)
    with pytest.raises(ReleaseError, match="repair restored; rerun rollback"):
        REPAIR.restore_standalone_prerequisite(instance)
    state = instance.runner.cloud_state
    assert REPAIR.REPAIR_KEY not in state and state["phase"] == "complete", (
        "the standalone rollback did not retire the older journal"
    )
    final = attempt(instance)
    assert instance.runner.cloud_state["phase"] == "complete" and (
        {"schema", "cpu-finalize"} <= set(final)
    ), "the deploy after the rollback did not bootstrap the lost control plane"
    assert live.up(instance.runtime_image), "bootstrap completed with no control plane"


def test_older_journal_over_a_complete_state_names_its_exit_to_another_candidate(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    live, _gpu_calls = live_site(instance, monkeypatch)
    legacy = legacy_complete_journal(instance, monkeypatch, live)
    next_candidate(instance, monkeypatch, "c", "7")
    mark = len(instance.runner.events)
    with pytest.raises(
        ReleaseError,
        match="a deploy of its candidate candidate-b retires it when the control "
        "plane is gone, and that candidate's rollback restores its refresher",
    ):
        deploy_preflight(instance)
    assert instance.runner.cloud_state == legacy, "the refusal rewrote the state"
    assert not (
        {"persist", "apply", "delete", "job-create"}
        & set(instance.runner.events[mark:])
    ), "the refused handoff mutated the site"


@pytest.mark.parametrize("mode", ["unrun", "failed", "killed", "cleaned"])
@pytest.mark.parametrize("phase", ["bootstrap-cleaned", "bootstrap-failed"])
def test_third_candidate_takes_over_an_unadopted_takeover_journal(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, phase: str, mode: str
) -> None:
    instance = handoff_release
    live, gpu_calls = live_site(instance, monkeypatch)
    old = adopted_predecessor(instance, phase)
    select_candidate(instance)
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)
    deploy_preflight(instance)
    journal = copy.deepcopy(instance.runner.cloud_state[REPAIR.REPAIR_KEY])
    assert journal["predecessor"]["release_id"] == "candidate", "B never took over"
    if mode != "unrun":
        # B's bootstrap stops before adopting its journal.
        instance.config.auto_rollback = mode == "cleaned"
        fail_once(instance, monkeypatch, "_prepare_nlb", kill=mode == "killed")
        with pytest.raises(Killed if mode == "killed" else ReleaseError):
            deploy(instance)
    state = instance.runner.cloud_state
    assert state[REPAIR.REPAIR_KEY]["attempt_id"] == journal["attempt_id"], (
        "B adopted or replaced its takeover journal"
    )
    instance.config.auto_rollback = False
    next_candidate(instance, monkeypatch, "c", "7")
    final = attempt(instance)
    state = instance.runner.cloud_state
    assert state["phase"] == "complete" and REPAIR.REPAIR_KEY not in state, (
        f"C did not bootstrap over B's unadopted takeover: {state}"
    )
    assert state["release_id"] == "candidate-c", "C's bootstrap kept B's identity"
    assert state[HANDOFF.PREDECESSOR_AUDIT_KEY] == journal["predecessor"], (
        "C's adoption lost the audit of the state B's takeover replaced"
    )
    assert state[HANDOFF.HANDOFF_AUDIT_KEY]["attempt_id"] == journal["attempt_id"], (
        "C's adoption lost the audit of B's journal"
    )
    assert state["bootstrap_database_origin"] == old["bootstrap_database_origin"], (
        "the empty-database origin of the first bootstrap was not carried forward"
    )
    assert {"schema", "cpu-finalize"} <= set(final) and "live-cpu-check" not in final, (
        "C trusted a control plane that is not its own"
    )
    assert gpu_calls[-1] == [], "C reused another candidate's completed clusters"
    assert live.up(instance.runtime_image), "C completed with no control plane"


def test_fourth_candidate_takes_over_the_third_candidates_unadopted_successor(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    live, _gpu_calls = live_site(instance, monkeypatch)
    old = adopted_predecessor(instance, "bootstrap-cleaned")
    select_candidate(instance)
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)
    deploy_preflight(instance)
    taken_over = copy.deepcopy(instance.runner.cloud_state[REPAIR.REPAIR_KEY])
    next_candidate(instance, monkeypatch, "c", "7")
    deploy_preflight(instance)
    successor = copy.deepcopy(instance.runner.cloud_state[REPAIR.REPAIR_KEY])
    assert successor["supersedes"]["attempt_id"] == taken_over["attempt_id"] and (
        successor["predecessor"] == taken_over["predecessor"]
    ), "C's successor lost B's journal or the takeover audit"
    next_candidate(instance, monkeypatch, "d", "8")
    final = attempt(instance)
    state = instance.runner.cloud_state
    assert state["phase"] == "complete" and state["release_id"] == "candidate-d", (
        f"D did not bootstrap over C's unadopted successor: {state}"
    )
    assert state[HANDOFF.PREDECESSOR_AUDIT_KEY] == taken_over["predecessor"], (
        "D's adoption lost the audit of the state B's takeover replaced"
    )
    assert state[HANDOFF.HANDOFF_AUDIT_KEY]["attempt_id"] == successor["attempt_id"], (
        "D's adoption lost the audit of C's journal"
    )
    assert state["bootstrap_database_origin"] == old["bootstrap_database_origin"], (
        "the empty-database origin of the first bootstrap was not carried forward"
    )
    assert {"schema", "cpu-finalize"} <= set(final), "D skipped its own CPU apply"
    assert live.up(instance.runtime_image), "D completed with no control plane"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("phase", "bootstrap-cpu-ready", "takeover handoff progress is invalid"),
        ("resume_phase", "bootstrap-cpu-ready", "takeover handoff progress is invalid"),
        ("completed_cluster_ids", None, "completion scope changed"),
    ],
)
def test_takeover_journal_handoff_still_refuses_progress(
    handoff_release: Any,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    instance = handoff_release
    live_site(instance, monkeypatch)
    adopted_predecessor(instance, "bootstrap-cleaned")
    select_candidate(instance)
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)
    deploy_preflight(instance)
    progressed = instance.runner.cloud_state
    progressed[field] = (
        [instance.config.clusters[0].cluster_id] if value is None else value
    )
    before = copy.deepcopy(progressed)
    next_candidate(instance, monkeypatch, "c", "7")
    mark = len(instance.runner.events)
    with pytest.raises(ReleaseError, match=message):
        deploy_preflight(instance)
    assert instance.runner.cloud_state == before, "the refusal rewrote the state"
    assert not (
        {"persist", "apply", "delete", "job-create"}
        & set(instance.runner.events[mark:])
    ), "the refused handoff mutated the site"


def test_takeover_on_another_schema_never_reuses_the_empty_database_origin(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    live_site(instance, monkeypatch)
    old = adopted_predecessor(instance, "bootstrap-cleaned")
    select_candidate(instance)
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)
    # B ships another schema than the one A's empty-database proof was bound to.
    instance.config.database_schema_version += 1
    instance.approved_manifest_digest = None
    report = ADMIN.build_deploy_preflight_report(instance)
    assert report["healthy"] is False and "origin proof is missing or unbound" in (
        json.dumps(report["checks"])
    ), f"the preflight accepted A's origin for B's schema: {report['checks']}"
    mark = len(instance.runner.events)
    with pytest.raises(ReleaseError, match="origin proof is missing or unbound"):
        deploy(instance)
    state = instance.runner.cloud_state
    assert REPAIR.REPAIR_KEY in state and (
        state["bootstrap_database_origin"] == old["bootstrap_database_origin"]
    ), "B adopted its journal over another schema's origin proof"
    assert not ({"schema", "cpu-finalize"} & set(instance.runner.events[mark:])), (
        "B ensured its schema or applied its CPU over another schema's origin"
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        # A ``complete`` state lends nothing, its stale resume point included.
        (
            {
                "phase": "complete",
                "resume_phase": "bootstrap-cpu-ready",
                "completed_cluster_ids": ["gpu-a"],
                "bootstrap_cpu_release_id": "candidate-b",
            },
            (set(), False, False),
        ),
        # A killed attempt never resumes from an older attempt's checkpoint.
        (
            {
                "phase": "bootstrap-started",
                "resume_phase": "bootstrap-cpu-ready",
                "completed_cluster_ids": ["gpu-a"],
                "bootstrap_cpu_release_id": "candidate-b",
            },
            ({"gpu-a"}, False, True),
        ),
        # The live check stands in only after this candidate's schema ensure.
        (
            {
                "phase": "bootstrap-failed",
                "resume_phase": "bootstrap-started",
                "completed_cluster_ids": ["gpu-a"],
            },
            ({"gpu-a"}, False, False),
        ),
        (
            {
                "phase": "bootstrap-failed",
                "resume_phase": "bootstrap-started",
                "bootstrap_cpu_release_id": "candidate",
            },
            (set(), False, False),
        ),
        # This candidate's own failure checkpoint still resumes, as before.
        (
            {
                "phase": "bootstrap-failed",
                "resume_phase": "bootstrap-cpu-ready",
                "completed_cluster_ids": ["gpu-a"],
            },
            ({"gpu-a"}, True, False),
        ),
        (
            {
                "phase": "bootstrap-cleaned",
                "resume_phase": "bootstrap-started",
                "bootstrap_cpu_release_id": "candidate-b",
            },
            (set(), False, False),
        ),
    ],
)
def test_resume_context_reads_only_this_attempts_own_evidence(
    state: dict[str, Any], expected: tuple[set[str], bool, bool]
) -> None:
    resumed = rollout.bootstrap_resume_context(
        {"release_id": "candidate-b", **state}, "candidate-b"
    )
    assert resumed == expected, f"{state['phase']} lent {resumed}"
