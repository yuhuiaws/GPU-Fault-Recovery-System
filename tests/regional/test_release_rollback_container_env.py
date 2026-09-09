"""Rollback gives the previous image exactly the container env it ran with.

Observed live 2026-09-07: a newer release engine rolled back to an older
image, restored the old image reference, and re-rendered the CPU role
Deployments from the *current* template. The current template carried
``env`` entries the older image had never heard of, the application failed
fast on ``received unknown GPU_FAULT_* environment variable(s)``, and the
release ended ``rollback-failed``. The ConfigMap side of the same problem was
already solved (``cpu_role_config_maps`` is captured and re-applied verbatim);
Deployment-level ``env``/``envFrom`` were not captured at all.

These tests pin the three halves of the fix: the capture stores every
container's ``env``/``envFrom`` verbatim from one read per Deployment, the
rollback validates the snapshot before it mutates anything and hands it to the
renderer through a file, and a transaction opened before the capture existed
falls back to the old behaviour and says so in the durable rollback record.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_rollback_context as ROLLBACK_CONTEXT
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional.test_release_rollback_state import (
    _rollback_previous,
    stub_rollback_phases,
)

NAMESPACE = "gpu-fault-system"
PREVIOUS_RUNTIME = "registry.example/runtime@sha256:" + "e" * 64
VARIABLE = ROLLBACK_CONTEXT.CONTAINER_ENV_FILE_VARIABLE


def _app_env(deployment: str) -> list[dict[str, Any]]:
    role = "ingress" if deployment == inventory.CPU_INGRESS_DEPLOYMENT else "worker"
    return [
        {"name": "GPU_FAULT_SERVICE_ROLE", "value": role},
        {
            "name": "GPU_FAULT_STORE_URL",
            "valueFrom": {
                "secretKeyRef": {"name": "gpu-fault-aurora", "key": "postgres-url"}
            },
        },
        {
            "name": "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
            "valueFrom": {
                "configMapKeyRef": {
                    "name": "gpu-fault-failure-domains",
                    "key": "map.json",
                    "optional": True,
                }
            },
        },
    ]


def _app_env_from(deployment: str) -> list[dict[str, Any]]:
    return [
        {"configMapRef": {"name": f"{deployment}-config-core"}},
        {"configMapRef": {"name": f"{deployment}-config-processor"}},
    ]


def _document(deployment: str, *, env: list[dict[str, Any]] | None = None) -> dict:
    return {
        "spec": {
            "template": {
                "spec": {
                    "initContainers": [
                        {
                            "name": "init",
                            "image": PREVIOUS_RUNTIME,
                            "env": [{"name": "GPU_FAULT_INIT_ONLY", "value": "1"}],
                        }
                    ],
                    "containers": [
                        {
                            "name": "app",
                            "image": PREVIOUS_RUNTIME,
                            "env": _app_env(deployment) if env is None else env,
                            "envFrom": _app_env_from(deployment),
                        }
                    ],
                }
            }
        }
    }


def _documents() -> dict[str, dict]:
    return {name: _document(name) for name in inventory.CPU_RUNTIME_DEPLOYMENTS}


def _expected_snapshot() -> dict[str, dict[str, dict[str, list]]]:
    return {
        deployment: {
            "init": {
                "env": [{"name": "GPU_FAULT_INIT_ONLY", "value": "1"}],
                "envFrom": [],
            },
            "app": {"env": _app_env(deployment), "envFrom": _app_env_from(deployment)},
        }
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
    }


def _reader(
    documents: dict[str, dict], reads: list[str], **stubs: Any
) -> SimpleNamespace:
    """A release double whose Deployment reads are served from ``documents``.

    ``stubs`` replaces or extends the defaults, so a test that drives the whole
    ``capture_previous`` path hands in the pins it needs up front.
    """

    def get_json(arguments: list[str]) -> dict:
        assert arguments[arguments.index("get") + 1] == "deployment"
        name = arguments[-1]
        reads.append(name)
        return documents[name]

    defaults = dict(
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["cpu", *args],
        _get_json=get_json,
        _config_map_data=lambda name: {"GPU_FAULT_PROCESSOR_WORKERS": "24"},
    )
    return SimpleNamespace(**{**defaults, **stubs})


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


def test_capture_reads_each_deployment_once_and_keeps_every_entry() -> None:
    """One `kubectl get deployment` per role serves both role snapshots.

    The env lists are copied verbatim -- literals, Secret refs, optional
    ConfigMap key refs, init containers -- and detached from the read
    document so a later mutation of the live object cannot leak into the
    snapshot.
    """

    documents = _documents()
    reads: list[str] = []
    release = _reader(documents, reads)

    deployments = STATE.cpu_role_deployments(release)
    snapshot = STATE.cpu_role_container_env(release, deployments)
    config_maps = STATE.cpu_role_config_maps(release, deployments)

    assert reads == list(inventory.CPU_RUNTIME_DEPLOYMENTS), (
        "each CPU role Deployment must be read exactly once for both snapshots"
    )
    assert snapshot == _expected_snapshot()
    assert set(config_maps) == {
        f"{deployment}-config-{domain}"
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
        for domain in ("core", "processor")
    }
    documents[inventory.CPU_INGRESS_DEPLOYMENT]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"].clear()
    assert snapshot == _expected_snapshot(), "snapshot aliases the live document"


def test_capture_refuses_a_literal_under_a_sensitive_name() -> None:
    documents = _documents()
    worker = "gpu-fault-control-worker"
    documents[worker] = _document(
        worker, env=[{"name": "GPU_FAULT_CLUSTER_TOKEN", "value": "plaintext"}]
    )
    release = _reader(documents, [])

    with pytest.raises(ReleaseError, match="sensitive-looking env names"):
        STATE.cpu_role_container_env(release)


def test_capture_fails_closed_when_a_role_deployment_is_missing() -> None:
    documents = _documents()
    del documents["gpu-fault-telemetry-spool-worker"]
    release = _reader(documents, [])

    with pytest.raises(KeyError):
        STATE.cpu_role_container_env(release)


def test_capture_previous_records_the_container_env_beside_the_config_maps() -> None:
    """The full CPU capture path writes the new key into `previous`."""

    documents = _documents()
    reads: list[str] = []
    state = {
        "release_id": "previous",
        "runtime_image": PREVIOUS_RUNTIME,
        "node_installer_image": "registry.example/installer@sha256:" + "f" * 64,
        "adot_image": "registry.example/adot@sha256:" + "a" * 64,
    }
    release = _reader(
        documents,
        reads,
        state=state,
        config=SimpleNamespace(
            namespace=NAMESPACE,
            clusters=(),
            bundle=Path("bundle.tar.gz"),
            wheel=SimpleNamespace(name="wheel"),
        ),
        runtime_image="registry.example/runtime@sha256:" + "c" * 64,
        node_installer_image=state["node_installer_image"],
        adot_image=state["adot_image"],
        _config_map_data=lambda name: (
            {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1"}
            if name == "gpu-fault-api-ha-config-core"
            else {}
        ),
        _deployment_wheel=lambda _args, name: f"{name}-wheel",
        _config_map_binary_key=lambda _args, name: f"{name}-key",
        _config_map_sha=lambda *_args: "4" * 64,
        _remote_command_stats=lambda: {"executor_internal_error_total": 0},
    )
    plan = STATE.ReleaseExecutionPlan(
        nodes=(STATE.ReleaseComponent.CPU_FINALIZE, STATE.ReleaseComponent.VERIFY)
    )

    previous = STATE.capture_previous(release, plan)

    assert previous["cpu_role_container_env"] == _expected_snapshot()
    assert set(previous["cpu_role_config_maps"]) == {
        f"{deployment}-config-{domain}"
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
        for domain in ("core", "processor")
    }
    assert previous["runtime_image"] == PREVIOUS_RUNTIME


# --------------------------------------------------------------------------
# Snapshot validation and the rollback environment
# --------------------------------------------------------------------------


def test_snapshot_validator_falls_back_when_the_key_is_absent() -> None:
    assert ROLLBACK_CONTEXT.previous_container_env_snapshot(None) is None
    assert ROLLBACK_CONTEXT.previous_container_env_snapshot({}) is None


def test_snapshot_validator_returns_a_detached_verbatim_copy() -> None:
    snapshot = _expected_snapshot()
    validated = ROLLBACK_CONTEXT.previous_container_env_snapshot(snapshot)
    assert validated == snapshot
    snapshot["gpu-fault-api-ha"]["app"]["env"].clear()
    assert validated == _expected_snapshot()


def _without_deployment() -> dict:
    snapshot = _expected_snapshot()
    del snapshot["gpu-fault-control-worker"]
    return snapshot


def _with_unknown_deployment() -> dict:
    snapshot = _expected_snapshot()
    snapshot["gpu-fault-adot"] = snapshot["gpu-fault-api-ha"]
    return snapshot


def _mutated(mutate) -> dict:
    snapshot = _expected_snapshot()
    mutate(snapshot["gpu-fault-api-ha"]["app"])
    return snapshot


@pytest.mark.parametrize(
    ("snapshot", "detail"),
    [
        pytest.param("not-a-mapping", "expected a Deployment mapping", id="scalar"),
        pytest.param(_without_deployment(), "do not match", id="missing-deployment"),
        pytest.param(_with_unknown_deployment(), "do not match", id="extra-deployment"),
        pytest.param(
            {name: {} for name in inventory.CPU_RUNTIME_DEPLOYMENTS},
            "lists no containers",
            id="no-containers",
        ),
        pytest.param(
            _mutated(lambda app: app.pop("envFrom")),
            "exactly env and envFrom",
            id="missing-envFrom",
        ),
        pytest.param(
            _mutated(lambda app: app["env"].append({"value": "x"})),
            "without a name",
            id="nameless-entry",
        ),
        pytest.param(
            _mutated(
                lambda app: app["env"].append(
                    {"name": "GPU_FAULT_X", "value": "1", "valueFrom": {}}
                )
            ),
            "exactly one of value or valueFrom",
            id="value-and-valueFrom",
        ),
        pytest.param(
            _mutated(
                lambda app: app["env"].append(
                    {"name": "GPU_FAULT_ADMIN_PASSWORD", "value": "hunter2"}
                )
            ),
            "sensitive env",
            id="sensitive-literal",
        ),
        pytest.param(
            _mutated(lambda app: app["envFrom"].append("gpu-fault-x-config-core")),
            "malformed envFrom",
            id="envFrom-scalar",
        ),
    ],
)
def test_snapshot_validator_rejects_malformed_or_partial_snapshots(
    snapshot: object, detail: str
) -> None:
    with pytest.raises(ReleaseError, match=detail):
        ROLLBACK_CONTEXT.previous_container_env_snapshot(snapshot)


def _rollback_environment(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> dict[str, str]:
    monkeypatch.setattr(
        ROLLBACK_CONTEXT, "admin_config_renderer_environment", lambda _config: {}
    )
    monkeypatch.setattr(ROLLBACK_CONTEXT, "notification_digest", lambda _n: "f" * 64)
    admin_config = SimpleNamespace(
        sha256=lambda: "a" * 64,
        role_sha256=lambda: {
            "ingress": "i" * 64,
            "worker": "w" * 64,
            "spool": "s" * 64,
        },
    )
    return ROLLBACK_CONTEXT.build_rollback_environment(
        rollback_config=SimpleNamespace(
            admin_config=admin_config,
            cpu_kubeconfig="/nonexistent/cpu.kubeconfig",
            aws_region="us-west-2",
            namespace=NAMESPACE,
            notifications=SimpleNamespace(
                allow_email=True, acknowledge_external_alert_channel=False
            ),
        ),
        metadata={},
        cpu_wheel="previous-cpu-wheel",
        cpu_sha="c" * 64,
        artifact="artifact",
        config_digest="config",
        runtime_profile_version="profile-v1",
        runtime_image=PREVIOUS_RUNTIME,
        **overrides,
    )


def test_build_rollback_environment_points_the_renderer_at_the_snapshot_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _rollback_environment(
        monkeypatch, previous_container_env_file="/tmp/inputs/previous.json"
    )
    assert environment[VARIABLE] == "/tmp/inputs/previous.json"
    assert environment["GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS"] == "false"


def test_build_rollback_environment_never_inherits_the_variable_from_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the validated snapshot may switch the renderer into rollback mode."""

    monkeypatch.setitem(os.environ, VARIABLE, "/stray/from/the/operator/shell")
    environment = _rollback_environment(monkeypatch)
    assert VARIABLE not in environment


# --------------------------------------------------------------------------
# The CPU restore phase of a rollback
# --------------------------------------------------------------------------


def _stub_every_phase_but_the_cpu_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave only the CPU restore live inside ``rollback_release``.

    Controller staging, the data-plane restore, the rollout cleanup and the
    final verification all reach a cluster; as no-ops they let the run drive
    the real CPU restore through the same ``run_phase`` bookkeeping a live
    rollback uses, so what it returns is read back from the saved record.
    """

    for name in (
        "_stage_rollback_controller",
        "_rollback_gpu_clusters",
        "_verify_and_complete_rollback",
        "cleanup_candidate_rollout_state",
    ):
        monkeypatch.setattr(ORCHESTRATION, name, lambda *_args, **_kwargs: None)


def _restore_fake(
    mutations: list[str], saved: list[tuple[str, dict[str, Any]]]
) -> SimpleNamespace:
    return SimpleNamespace(
        state={},
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
        runner=SimpleNamespace(
            probe=lambda *_args, **_kwargs: False, run=lambda *_args, **_kwargs: ""
        ),
        _cpu=lambda *args: ["cpu", *args],
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _save_state=lambda phase, **updates: saved.append((phase, updates)),
        _restore_secret=lambda *_a, **_k: mutations.append("secret"),
        _restore_registry_backup=lambda: mutations.append("registry"),
        _restore_cpu_role_config_maps=lambda snapshots: (
            mutations.append("config-maps") or bool(snapshots)
        ),
        _config_map_sha=lambda *_args: "4" * 64,
    )


def _previous(**extra: Any) -> dict[str, Any]:
    """The previous-release record the rollback compensates towards.

    ``_rollback_previous`` carries the pins the rollback identity is resolved
    from -- its runtime image is this file's ``PREVIOUS_RUNTIME`` -- and the
    CPU half adds what the restore itself consumes.
    """

    return {
        **_rollback_previous(),
        "admin_config": {},
        "secret_backups": {"cpu": {"source": "gpu-fault-aurora", "backup": "bk"}},
        "cpu_role_config_maps": {"gpu-fault-api-ha-config-core": {"A": "1"}},
        **extra,
    }


def _restore(
    monkeypatch: pytest.MonkeyPatch, mutations: list[str], previous: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Roll back to ``previous`` with only the CPU restore live; returns the
    ``(phase, updates)`` pairs the release state was saved with."""

    _stub_every_phase_but_the_cpu_restore(monkeypatch)
    saved: list[tuple[str, dict[str, Any]]] = []
    ORCHESTRATION.rollback_release(_restore_fake(mutations, saved), state=previous)
    return saved


def _cpu_restore_details(saved: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """The CPU restore's details as ``status`` and a post-mortem read them:
    from the ``rollback_timing`` record saved at ``rollback-cpu-restored``."""

    phases = dict(saved)["rollback-cpu-restored"]["rollback_timing"]["phases"]
    return phases["cpu_restore"]["details"]


def _intercept_render(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Record what the environment builder was given and what the apply saw.

    The snapshot file lives in a temp dir that is gone once the restore
    returns, so its content is read while the apply step is running.
    """

    builds: list[dict[str, Any]] = []
    applied: list[dict[str, str]] = []

    def build(**kwargs: Any) -> dict[str, str]:
        path = kwargs.get("previous_container_env_file")
        builds.append(
            {
                **kwargs,
                "file_content": (
                    json.loads(Path(path).read_text(encoding="utf-8")) if path else None
                ),
            }
        )
        environment = {"GPU_FAULT_RUNTIME_IMAGE": kwargs["runtime_image"]}
        if path:
            environment[VARIABLE] = path
        return environment

    def apply(_self: Any, environment: dict[str, str]) -> None:
        applied.append(
            {
                **environment,
                "file_exists": str(Path(environment[VARIABLE]).is_file())
                if VARIABLE in environment
                else "n/a",
            }
        )

    monkeypatch.setattr(ORCHESTRATION, "build_rollback_environment", build)
    monkeypatch.setattr(ORCHESTRATION, "_apply_rollback_cpu_environment", apply)
    return builds, applied


def test_restore_rollback_cpu_hands_the_captured_env_to_the_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds, applied = _intercept_render(monkeypatch)
    mutations: list[str] = []
    snapshot = _expected_snapshot()

    saved = _restore(
        monkeypatch,
        mutations,
        _previous(cpu_role_container_env=copy.deepcopy(snapshot)),
    )

    assert len(builds) == 1 and len(applied) == 1
    assert builds[0]["file_content"] == snapshot, (
        "the renderer must receive the snapshot verbatim"
    )
    assert builds[0]["preserve_role_config_maps"] is True
    assert builds[0]["runtime_image"] == PREVIOUS_RUNTIME, (
        "the previous image, never the candidate's, is what the renderer gets"
    )
    assert applied[0]["file_exists"] == "True", "file must outlive the apply step"
    assert _cpu_restore_details(saved) == {"cpu_container_env": "snapshot"}
    assert mutations == ["secret", "registry", "config-maps"]


def test_restore_rollback_cpu_falls_back_and_records_it_for_old_snapshots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transaction opened before the capture existed still rolls back.

    It renders from the current template as before -- the only thing the
    engine can do without a snapshot -- and leaves a durable trace so a later
    ``unknown GPU_FAULT environment variable`` failure is explainable.
    """

    builds, applied = _intercept_render(monkeypatch)

    details = _cpu_restore_details(_restore(monkeypatch, [], _previous()))

    assert builds[0]["previous_container_env_file"] is None
    assert VARIABLE not in applied[0]
    assert details["cpu_container_env"] == "current-template"
    assert "predates" in details["cpu_container_env_note"]
    narration = capsys.readouterr().err
    assert "rollback-cpu-container-env" in narration
    assert "source=current-template" in narration
    assert "reason=snapshot-predates-capture" in narration


def test_restore_rollback_cpu_refuses_a_malformed_snapshot_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds, applied = _intercept_render(monkeypatch)
    mutations: list[str] = []

    with pytest.raises(ReleaseError, match="container environment snapshot"):
        _restore(
            monkeypatch,
            mutations,
            _previous(cpu_role_container_env=_without_deployment()),
        )

    assert mutations == [], "a bad snapshot must stop before the Secret restore"
    assert builds == [] and applied == []


def test_rollback_records_the_cpu_restore_details_in_the_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What `_restore_rollback_cpu` returns lands in `rollback_timing`.

    That record is saved with the release state, so `status` and a post-mortem
    can read where the container environment came from long after the log
    lines are gone.
    """

    stub_rollback_phases(monkeypatch)
    monkeypatch.setattr(
        ORCHESTRATION,
        "_restore_rollback_cpu",
        lambda *_args, **_kwargs: {"cpu_container_env": "current-template"},
    )
    saved: list[tuple[str, dict[str, Any]]] = []
    release = SimpleNamespace(
        state={},
        runtime_image="candidate-runtime",
        node_installer_image="registry.example/installer:candidate",
        config=SimpleNamespace(
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
            schema_rollback_compatible=True,
        ),
        _save_state=lambda phase, **updates: saved.append((phase, updates)),
        _refresh_aurora_credentials=lambda *_args, **_kwargs: None,
        _require_no_inflight_installs=lambda *_args, **_kwargs: None,
    )

    ORCHESTRATION.rollback_release(release, state=_rollback_previous())

    phases = dict(saved)["rollback-cpu-restored"]["rollback_timing"]["phases"]
    assert phases["cpu_restore"]["status"] == "COMPLETED"
    assert phases["cpu_restore"]["details"] == {"cpu_container_env": "current-template"}
    assert "details" not in phases["controller_stage"], (
        "phases that return nothing must not grow an empty details block"
    )
