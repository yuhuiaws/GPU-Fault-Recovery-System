"""The release engine for a plain EKS cluster's DCGM host engine mode.

``host`` (every selected node group declares ``gpu-fault.io/dcgm-hostengine=
host``): the exporter dials the node's ``nvidia-dcgm`` like the HyperPod text,
the node installer, reconciler and Agent carry ``GPU_FAULT_DCGM_HOSTENGINE=
host``. ``embedded``: today's eks variant, byte for byte. A mode flip is the
``dcgm_hostengines`` release input -- exporter, reconciler and Agent waves --
the forward waves wait for the reconciler's mode annotation, a rollback
restores the mode the previous release ran, and ``status --full`` shows the
mode and every node's DCGM state the last node preflight observed. The admin
half (discovery, site.yaml, node-scope) is
``tests/admin/test_admin_dcgm_hostengine.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.cluster_scope import DCGM_HOSTENGINE_ENV
from gpu_fault.node_installer_rendering import INSTALLER_DCGM_HOSTENGINE_ANNOTATION
from gpu_fault_release import regional_admin_checks_coverage as COVERAGE
from gpu_fault_release import regional_gpu_bootstrap as GPU_BOOTSTRAP
from gpu_fault_release import regional_node_batch as BATCH
from gpu_fault_release import regional_release_dcgm as DCGM
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    agents_converged,
    upgrade_gpu_target,
)
from tests.regional import test_release_node_image_targets as IMAGE_TARGETS
from tests.regional._release_orchestrator_support import config_file
from tests.regional.test_regional_release_eks_managed import (
    RecordingRunner,
    _eks_cluster,
    _hyperpod_cluster,
)
from tests.regional.test_release_node_image_targets import (
    ARTIFACT,
    BUNDLE,
    CONFIG,
    rollback,
)

#: The rollback/rollout scenario of the node image target tests, reused.
node_release = IMAGE_TARGETS.node_release
IMAGE = "registry.example/dcgm@sha256:" + "b" * 64
REMOTE = "DCGM_REMOTE_HOSTENGINE_INFO"
HOST_ENTRY = {"dcgm_hostengine": "host", "dcgm_hostengine_source": "node-group-label"}


def _render(**options: Any) -> str:
    return RENDERING.render_dcgm_exporter_manifest(
        namespace="gpu-fault-system", image=IMAGE, **options
    )


def _daemonset(text: str) -> dict[str, Any]:
    return next(
        document
        for document in yaml.safe_load_all(text)
        if isinstance(document, dict) and document.get("kind") == "DaemonSet"
    )


def _exporter_env(daemonset: dict[str, Any]) -> dict[str, Any]:
    (container,) = [
        item
        for item in daemonset["spec"]["template"]["spec"]["containers"]
        if item["name"] == "exporter"
    ]
    return {item["name"]: item.get("value") for item in container.get("env", [])}


def _release(tmp_path: Path, *clusters: dict[str, Any]) -> MODULE.RegionalRelease:
    release = MODULE.RegionalRelease(
        MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=list(clusters))),
        RecordingRunner(),
    )
    release.dcgm_exporter_image = IMAGE
    return release


def _eks(tmp_path: Path, name: str = "gpu-eks-cluster", **fields: Any) -> dict:
    value = _eks_cluster(tmp_path)
    value.update(cluster_id=name, context=f"{name}-context", **fields)
    return value


# --- rendering --------------------------------------------------------------------


def test_the_host_variant_dials_the_node_engine_and_keeps_the_eks_variant() -> None:
    embedded = _daemonset(_render(provider="eks-managed"))
    host = _daemonset(_render(provider="eks-managed", dcgm_hostengine="host"))
    assert host["metadata"]["annotations"]["gpu-fault.io/dcgm-hostengine"] == "host"
    host_env, embedded_env = _exporter_env(host), _exporter_env(embedded)
    assert host_env[REMOTE] == "127.0.0.1:5555", (
        "exactly one hostengine per node: the exporter dials nvidia-dcgm"
    )
    assert {key: value for key, value in host_env.items() if key != REMOTE} == (
        embedded_env
    ), "everything else of the eks variant (NVIDIA_* env, collectors) is kept"
    pod = host["spec"]["template"]["spec"]
    assert pod["tolerations"] == [
        {"key": "nvidia.com/gpu", "operator": "Exists"},
        {"key": "gpu-fault.io/quarantined", "operator": "Exists"},
    ], "still no quiesce toleration: the quiesce stops nvidia-dcgm and evicts it"
    for variant in (host, embedded):
        variant["metadata"]["annotations"].pop("gpu-fault.io/dcgm-hostengine")
        for container in variant["spec"]["template"]["spec"]["containers"]:
            container["env"] = [
                item for item in container.get("env", []) if item["name"] != REMOTE
            ]
    assert host == embedded, "the two variants differ only in the engine they use"


def test_the_embedded_variant_is_byte_identical_to_the_variant_before() -> None:
    """Rebuilt here from the checked-in DaemonSet the way the eks variant was
    always rendered: a site without the label applies the same bytes."""

    hyperpod_text = _render()
    documents = list(yaml.safe_load_all(hyperpod_text))
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != "DaemonSet":
            continue
        document.setdefault("metadata", {}).setdefault("annotations", {})[
            "gpu-fault.io/dcgm-hostengine"
        ] = "embedded"
        pod = document["spec"]["template"]["spec"]
        pod["tolerations"] = [
            {"key": "nvidia.com/gpu", "operator": "Exists"},
            {"key": "gpu-fault.io/quarantined", "operator": "Exists"},
        ]
        for container in pod["containers"]:
            if container["name"] == "exporter":
                container["env"] = [
                    item for item in container["env"] if item["name"] != REMOTE
                ] + [
                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                    {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "utility"},
                ]
    expected = yaml.safe_dump_all(documents, sort_keys=False)
    assert _render(provider="eks-managed") == expected
    assert _render(provider="eks-managed", dcgm_hostengine="embedded") == expected
    assert _render(provider="hyperpod", dcgm_hostengine="host") == hyperpod_text, (
        "the HyperPod text ignores the mode: it always dials its host service"
    )


def test_each_cluster_is_applied_its_own_mode_and_a_rollback_its_captured_one(
    tmp_path: Path,
) -> None:
    release = _release(
        tmp_path, _hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY)
    )
    hyperpod, eks = release.config.clusters

    def applied(target: Any, **options: Any) -> list[str]:
        before = len(release.runner.calls)  # type: ignore[attr-defined]
        GPU_BOOTSTRAP.apply_gpu_dcgm_exporter(release, target, **options)
        return [
            str(call_options.get("input_text") or "")
            for arguments, call_options in release.runner.calls[before:]  # type: ignore[attr-defined]
            if arguments[-3:] == ["apply", "-f", "-"]
            and "kind: DaemonSet" in str(call_options.get("input_text") or "")
        ]

    (host_text,) = applied(eks)
    assert _exporter_env(_daemonset(host_text))[REMOTE] == "127.0.0.1:5555"
    (restored,) = applied(eks, dcgm_hostengine="embedded")
    assert REMOTE not in _exporter_env(_daemonset(restored)), (
        "a rollback renders the mode the previous DaemonSet ran"
    )
    (hyperpod_text,) = applied(hyperpod)
    assert hyperpod_text == _render(), "HyperPod applies the checked-in text"


def test_the_plan_payload_carries_one_eks_variant_per_mode_in_use(
    tmp_path: Path,
) -> None:
    def payload(*clusters: dict[str, Any]) -> dict[str, Any]:
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        root.mkdir()
        return RENDERING.render_release_payload(_release(root, *clusters))

    host_only = payload(_hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY))
    assert "dcgm_eks_managed" not in host_only, "no cluster runs the embedded variant"
    assert {item["kind"] for item in host_only["dcgm_eks_managed_host"]} == {
        "DaemonSet"
    }, "the approver reads the host variant"
    assert REMOTE in json.dumps(host_only["dcgm_eks_managed_host"])
    both = payload(
        _eks(tmp_path, "gpu-eks-a", **HOST_ENTRY), _eks(tmp_path, "gpu-eks-b")
    )
    assert {"dcgm_eks_managed", "dcgm_eks_managed_host"} <= set(both), (
        "a site running both modes shows both texts"
    )
    embedded_only = payload(_hyperpod_cluster(tmp_path), _eks(tmp_path))
    assert "dcgm_eks_managed_host" not in embedded_only
    assert "dcgm_eks_managed" in embedded_only, "today's plan, unchanged"


# --- release input --------------------------------------------------------------------


def test_a_mode_flip_is_a_data_plane_release_of_exporter_reconciler_and_agents(
    tmp_path: Path,
) -> None:
    host = _release(tmp_path, _hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY))
    digest = DCGM.dcgm_hostengines_digest(host.config)
    assert DCGM.cluster_dcgm_hostengines(host.config) == {"gpu-eks-cluster": "host"}
    legacy_state: dict[str, Any] = {}
    assert (
        DCGM.DCGM_HOSTENGINES_INPUT in DIFF.classify_release(host, legacy_state).changed
    ), "declaring host mode is a change against every state recorded before"
    recorded = {DCGM.DCGM_HOSTENGINES_STATE_KEY: digest}
    assert (
        DCGM.DCGM_HOSTENGINES_INPUT not in DIFF.classify_release(host, recorded).changed
    )
    embedded_root = tmp_path / "embedded"
    embedded_root.mkdir()
    embedded = _release(
        embedded_root, _hyperpod_cluster(embedded_root), _eks(embedded_root)
    )
    assert (
        DCGM.DCGM_HOSTENGINES_INPUT
        not in DIFF.classify_release(embedded, legacy_state).changed
    ), "a site that never declared host mode keeps its diff exactly as before"
    assert (
        DCGM.DCGM_HOSTENGINES_INPUT in DIFF.classify_release(embedded, recorded).changed
    ), "leaving host mode is a change too"
    diff = DIFF.diff_from_changed({DCGM.DCGM_HOSTENGINES_INPUT})
    assert diff.kind is ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(ReleaseComponent.DCGM), "the exporter is re-rendered"
    assert plan.has(ReleaseComponent.RECONCILER), "the reconciler identity moves"
    assert plan.has(ReleaseComponent.AGENT), "the Agents are re-installed in waves"
    assert not plan.has(ReleaseComponent.EXECUTOR, ReleaseComponent.CPU_STAGE), (
        "nothing else rolls for a mode flip"
    )


def test_the_release_state_records_the_modes_it_rolled(tmp_path: Path) -> None:
    release = _release(
        tmp_path, _hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY)
    )
    STATE.save_state(release, "dcgm-test")
    assert release.state[DCGM.DCGM_HOSTENGINES_STATE_KEY] == (
        DCGM.dcgm_hostengines_digest(release.config)
    )
    assert (
        DCGM.DCGM_HOSTENGINES_INPUT
        not in DIFF.classify_release(release, dict(release.state)).changed
    ), "the next deploy of the same site sees no flip"


# --- node runtime: reconciler environment, waits, rollback ------------------------


def _reconciler_environment(release: Any, target: Any) -> dict[str, str]:
    return RENDERING.build_reconciler_environment(
        release,
        target,
        wheel_cm="wheel-cm",
        bundle_cm="bundle-cm",
        artifact_sha="a" * 64,
        config_digest="b" * 64,
    )


def test_the_reconciler_environment_carries_the_mode_on_eks_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DCGM_HOSTENGINE_ENV, "host")  # never inherited
    release = _release(
        tmp_path,
        _hyperpod_cluster(tmp_path),
        _eks(tmp_path, "gpu-eks-a", **HOST_ENTRY),
        _eks(tmp_path, "gpu-eks-b"),
    )
    hyperpod, host, embedded = release.config.clusters
    environments = [
        _reconciler_environment(release, item) for item in release.config.clusters
    ]
    assert [item[DCGM_HOSTENGINE_ENV] for item in environments] == [
        "",
        "host",
        "embedded",
    ]
    hyperpod_env = environments[0]
    assert RENDERING.installer_product_inputs_digest(hyperpod_env) == (
        RENDERING.installer_product_inputs_digest(
            {k: v for k, v in hyperpod_env.items() if k != DCGM_HOSTENGINE_ENV}
        )
    ), "a HyperPod site's recorded products keep their digest"
    assert RENDERING.installer_product_inputs_digest(environments[1]) != (
        RENDERING.installer_product_inputs_digest(
            {**environments[1], DCGM_HOSTENGINE_ENV: "embedded"}
        )
    ), "a mode flip never reuses the other mode's rendered installer template"


def _eks_target(scenario: SimpleNamespace, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        **vars(scenario.target),
        provider="eks-managed",
        node_selector="gpu-fault.io/control-plane=site-a",
        node_selector_source="label-convention",
        dcgm_hostengine=mode,
        dcgm_hostengine_source="node-group-label" if mode == "host" else "default",
    )


@pytest.mark.parametrize("mode", ["host", "embedded"])
def test_forward_waves_wait_for_the_agents_installed_in_the_cluster_mode(
    node_release: SimpleNamespace, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = node_release.release
    waits: list[dict[str, Any]] = []
    monkeypatch.setattr(
        release, "_wait_agents", lambda *_args, **options: waits.append(options)
    )
    target = _eks_target(node_release, mode)
    upgrade_gpu_target(
        release,
        target,
        ReleaseDiff(ReleaseChangeKind.DATA_PLANE_COMPATIBLE, frozenset()),
        ReleaseExecutionPlan(nodes=(ReleaseComponent.AGENT,)),
    )
    assert waits and {item.get("dcgm_hostengine") for item in waits} == {mode}, (
        "every forward wave and the steady-state wait hold the Agents to the mode"
    )
    assert {item[DCGM_HOSTENGINE_ENV] for item in node_release.environments} == {mode}


def test_the_mode_annotation_decides_convergence_only_when_asked() -> None:
    target = SimpleNamespace(
        provider="eks-managed", node_selector="gpu-fault.io/control-plane=site-a"
    )

    def node(annotation: str | None) -> dict[str, Any]:
        annotations = {
            "gpu-fault.io/installer-state": "Succeeded",
            "gpu-fault.io/installer-artifact-sha256": "a" * 64,
        }
        if annotation is not None:
            annotations[INSTALLER_DCGM_HOSTENGINE_ANNOTATION] = annotation
        return {
            "metadata": {
                "name": "node-a",
                "labels": {"gpu-fault.io/control-plane": "site-a"},
                "annotations": annotations,
            }
        }

    assert agents_converged([node(None)], target, "a" * 64), "no expectation"
    assert not agents_converged(
        [node(None)], target, "a" * 64, dcgm_hostengine="host"
    ), "an Agent installed before the flip is still pending"
    assert agents_converged([node("host")], target, "a" * 64, dcgm_hostengine="host"), (
        "the reconciler re-installed the Agent in host mode"
    )
    assert not agents_converged(
        [node("host")], target, "a" * 64, dcgm_hostengine="embedded"
    ), "leaving host mode waits for the reconciler to remove the annotation"
    assert agents_converged(
        [node(None)], target, "a" * 64, dcgm_hostengine="embedded"
    ), "an embedded Agent carries no annotation"
    hyperpod = SimpleNamespace(provider="hyperpod", hyperpod_cluster_name="hp")
    assert DCGM.forward_dcgm_hostengine(hyperpod, "upgrade") is None
    assert DCGM.forward_dcgm_hostengine(target, "rollback") is None, (
        "a rollback's reconciler may predate the annotation"
    )
    assert DCGM.forward_dcgm_hostengine(target, "join") == "embedded"


@pytest.mark.parametrize(
    ("captured", "expected"), [("embedded", "embedded"), (None, "host")]
)
def test_a_rollback_restores_the_mode_the_previous_release_ran(
    node_release: SimpleNamespace, captured: str | None, expected: str
) -> None:
    applied: list[dict[str, Any]] = []
    node_release.release._apply_gpu_dcgm_exporter = (
        lambda target, **options: applied.append(options)
    )
    binding = node_release.previous["clusters"]["gpu-a"]
    binding["dcgm_image"] = IMAGE
    if captured is not None:
        binding["dcgm_hostengine"] = captured
    target = _eks_target(node_release, "host")
    node_release.target = target
    rollback(node_release, ReleaseComponent.RECONCILER)
    assert {item[DCGM_HOSTENGINE_ENV] for item in node_release.environments} == {
        expected
    }, "the reconciler is restored for the previous mode when it was captured"
    rollback(node_release, ReleaseComponent.DCGM)
    assert applied == [{"image": IMAGE, "dcgm_hostengine": captured}], (
        "the exporter is re-rendered for the captured mode (None: the target's)"
    )


def test_the_previous_snapshot_captures_the_live_exporter_mode() -> None:
    daemonset = {
        "metadata": {"annotations": {"gpu-fault.io/dcgm-hostengine": "host"}},
        "spec": {"template": {"spec": {"containers": [{"image": IMAGE}]}}},
    }
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=lambda _arguments: daemonset,
    )
    eks = SimpleNamespace(cluster_id="gpu-eks", provider="eks-managed")
    hyperpod = SimpleNamespace(cluster_id="gpu-a", provider="hyperpod")
    assert DCGM.capture_exporter_identity(release, eks) == {
        "dcgm_image": IMAGE,
        "dcgm_hostengine": "host",
    }
    assert DCGM.capture_exporter_identity(release, hyperpod) == {"dcgm_image": IMAGE}
    previous = {"clusters": {"gpu-eks": {"dcgm_hostengine": "host"}, "gpu-a": {}}}
    assert DCGM.captured_dcgm_hostengine(previous, eks) == "host"
    assert DCGM.captured_dcgm_hostengine({"clusters": {"gpu-eks": {}}}, eks) is None
    assert DCGM.captured_dcgm_hostengine(previous, hyperpod) is None
    assert DCGM.daemonset_dcgm_hostengine({"metadata": {}}) is None


# --- node preflight observations and status ---------------------------------------


def _verdict(node: str, state: str, *, conflict: bool = False) -> str:
    return json.dumps(
        {
            "node_id": node,
            "status": "PASSED",
            "dcgm": {"mode": "host", "state": state, "conflict": conflict},
        }
    )


def test_the_preflight_batch_returns_every_nodes_dcgm_state() -> None:
    assert BATCH.preflight_dcgm_observation(
        "INFO checking\n" + _verdict("node-a", "host-active") + "\n"
    ) == {"mode": "host", "state": "host-active", "conflict": False}
    assert BATCH.preflight_dcgm_observation('{"status": "PASSED"}') is None, (
        "a HyperPod verdict carries no DCGM object"
    )
    assert BATCH.preflight_dcgm_observation("ERROR: no verdict") is None


def test_a_node_preflight_records_its_dcgm_states_for_status(
    node_release: SimpleNamespace,
) -> None:
    release = node_release.release
    original = release.runner.run

    def run(arguments, **options):
        result = original(arguments, **options)
        if options.get("capture"):
            summary = json.loads(result)
            summary["dcgm_hostengine"] = {
                "node-a": {"mode": "host", "state": "host-active", "conflict": False}
            }
            return json.dumps(summary)
        return result

    release.runner = SimpleNamespace(dry_run=False, run=run)
    target = _eks_target(node_release, "host")
    RUNTIME.preflight_node_runtime(
        release,
        target,
        phase="upgrade",
        wheel_cm="test-wheel",
        bundle_cm=release.bundle_cm,
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        bundle_sha256=BUNDLE,
    )
    record = release.state[DCGM.NODE_DCGM_STATE_KEY]["gpu-a"]
    assert record["mode"] == "host" and record["observed_at"], record
    assert record["nodes"] == {
        "node-a": {"mode": "host", "state": "host-active", "conflict": False}
    }


def _status_release(state: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=[]),
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=lambda _arguments: {"items": []},
    )


def _status_node(name: str, annotation: str | None) -> dict[str, Any]:
    annotations = (
        {} if annotation is None else {INSTALLER_DCGM_HOSTENGINE_ANNOTATION: annotation}
    )
    return {"metadata": {"name": name, "annotations": annotations}}


def test_status_shows_the_mode_pending_agents_and_every_nodes_state() -> None:
    target = SimpleNamespace(
        cluster_id="gpu-eks",
        provider="eks-managed",
        hyperpod_cluster_name="",
        node_selector="gpu-fault.io/control-plane=site-a",
        node_selector_source="label-convention",
        **HOST_ENTRY,
    )
    state = {
        DCGM.NODE_DCGM_STATE_KEY: {
            "gpu-eks": {
                "mode": "host",
                "observed_at": "observed",
                "nodes": {
                    "node-a": {"state": "host-active", "conflict": False},
                    "node-b": {"state": "installed-inactive", "conflict": False},
                },
            }
        }
    }
    nodes = {"items": [_status_node("node-a", "host"), _status_node("node-b", None)]}
    details = COVERAGE.node_scope_details(
        _status_release(state), target, target.node_selector, nodes
    )
    assert details["dcgm_hostengine"] == "host"
    assert details["dcgm_hostengine_source"] == "node-group-label"
    assert details["dcgm_hostengine_pending_nodes"] == ["node-b"], (
        "node-b's Agent still runs the embedded grading"
    )
    assert details["dcgm_hostengine_nodes"] == {
        "node-a": "host-active",
        "node-b": "installed-inactive",
    }
    assert details["dcgm_hostengine_conflict_nodes"] == ["node-b"], (
        "a host-mode node without a working nvidia-dcgm is flagged by name"
    )
    assert details["dcgm_hostengine_observed_at"] == "observed", (
        "the states are the last preflight's observation, not a live probe"
    )
    embedded = SimpleNamespace(
        **{
            **vars(target),
            "dcgm_hostengine": "embedded",
            "dcgm_hostengine_source": "default",
        }
    )
    state[DCGM.NODE_DCGM_STATE_KEY]["gpu-eks"]["nodes"]["node-b"] = {
        "state": "host-active",
        "conflict": True,
    }
    embedded_details = COVERAGE.node_scope_details(
        _status_release(state), embedded, embedded.node_selector, nodes
    )
    assert embedded_details["dcgm_hostengine_conflict_nodes"] == ["node-b"], (
        "embedded mode flags the node whose host engine runs next to the exporter's"
    )
    assert embedded_details["dcgm_hostengine_pending_nodes"] == ["node-a"], (
        "node-a still carries the host annotation"
    )


def test_status_reads_the_persisted_observations_without_replacing_the_state() -> None:
    persisted = {
        DCGM.NODE_DCGM_STATE_KEY: {
            "gpu-eks": {"mode": "embedded", "observed_at": "t", "nodes": {}}
        }
    }
    reads: list[list[str]] = []

    def get_json(arguments: list[str]) -> dict[str, Any]:
        reads.append(arguments)
        return {"data": {"state.json": json.dumps(persisted)}}

    release = SimpleNamespace(
        state={},
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: ["cpu", *arguments],
        _get_json=get_json,
    )
    target = SimpleNamespace(
        cluster_id="gpu-eks", provider="eks-managed", dcgm_hostengine="embedded"
    )
    details = DCGM.node_dcgm_details(release, target, {"items": []})
    assert details["dcgm_hostengine_observed_at"] == "t"
    assert release.state == {}, "a status read never replaces the in-memory state"
    assert reads and reads[0][-1] == STATE.STATE_CONFIG_MAP, (
        "the status read targets the release state ConfigMap"
    )
    assert DCGM.RELEASE_STATE_CONFIG_MAP == STATE.STATE_CONFIG_MAP


def test_status_holds_a_hyperpod_cluster_to_its_host_engine_too() -> None:
    """HyperPod is always host mode: the last preflight's per-node states are
    shown and a node without a working nvidia-dcgm is flagged, but no Agent is
    ever "pending" (the reconciler annotation is an eks-managed identity)."""

    target = SimpleNamespace(
        cluster_id="gpu-a",
        provider="hyperpod",
        hyperpod_cluster_name="hp-gpu-a",
        node_selector="",
        node_selector_source="",
    )
    state = {
        DCGM.NODE_DCGM_STATE_KEY: {
            "gpu-a": {
                "mode": "host",
                "observed_at": "observed",
                "nodes": {
                    "hyperpod-i-1": {"state": "host-active", "conflict": False},
                    "hyperpod-i-2": {"state": "host-failed", "conflict": False},
                },
            }
        }
    }
    details = DCGM.node_dcgm_details(
        _status_release(state), target, {"items": [_status_node("hyperpod-i-1", None)]}
    )
    assert details == {
        "dcgm_hostengine": "host",
        "dcgm_hostengine_source": "provider",
        "dcgm_hostengine_observed_at": "observed",
        "dcgm_hostengine_nodes": {
            "hyperpod-i-1": "host-active",
            "hyperpod-i-2": "host-failed",
        },
        "dcgm_hostengine_conflict_nodes": ["hyperpod-i-2"],
    }, details
