"""What the data plane serves decides a DCGM host engine mode, not a digest.

Live 2026-10-17, in this order: a release flipping the plain EKS cluster to
host mode was refused by the node preflight and rolled back; the failure
recovery rebuilt ``site.yaml`` from the attempt's baseline (written after its
bootstrap had recorded ``dcgmHostEngine: host``) and ``sync-state`` committed
the digest of that site. The next, successful release of the same head saw no
``dcgm_hostengines`` change: it re-installed the Agents in host mode and never
re-rendered the exporter, which kept running its embedded engine next to the
host ``nvidia-dcgm`` -- the conflict the mode exists to prevent -- while
``status --full`` reported everything converged.

Now the release diff probes the live exporter and reconciler
(``dcgm_hostengine_drift``), ``status --full``/verify refuse a live render that
serves another mode than recorded, and ``sync-state`` records the mode the
exporters serve and keeps the last node preflight's observations.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.cluster_scope import DCGM_HOSTENGINE_ENV
from gpu_fault_release import regional_gpu_bootstrap as GPU_BOOTSTRAP
from gpu_fault_release import regional_release_dcgm as DCGM
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_deployment_inventory import GPU_RECONCILER_DEPLOYMENT
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseChangeKind, ReleaseComponent
from tests.regional.test_regional_release_eks_managed import _hyperpod_cluster
from tests.regional.test_release_dcgm_hostengine import HOST_ENTRY, _eks, _release

EMBEDDED_DIGEST = DCGM.dcgm_hostengines_digest(SimpleNamespace(clusters=()))


def _target(cluster_id: str, provider: str = "eks-managed", **fields: Any):
    return SimpleNamespace(
        cluster_id=cluster_id,
        context=f"{cluster_id}-context",
        provider=provider,
        hyperpod_cluster_name="" if provider == "eks-managed" else cluster_id,
        **fields,
    )


def _exporter(mark: str | None, *, remote: bool = False) -> dict[str, Any]:
    annotations = {} if mark is None else {DCGM.DAEMONSET_ANNOTATION: mark}
    env = [{"name": DCGM.REMOTE_HOSTENGINE_ENV, "value": "127.0.0.1:5555"}]
    return {
        "metadata": {"name": DCGM.EXPORTER_DAEMONSET, "annotations": annotations},
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"name": "exporter", "env": env if remote else []}]
                }
            }
        },
    }


def _reconciler(mode: str | None) -> dict[str, Any]:
    env = [] if mode is None else [{"name": DCGM_HOSTENGINE_ENV, "value": mode}]
    return {
        "spec": {
            "template": {"spec": {"containers": [{"name": "reconciler", "env": env}]}}
        }
    }


class LiveRelease:
    """Each cluster's live exporter DaemonSet and reconciler Deployment."""

    def __init__(self, targets: list[Any], live: dict[str, dict[str, Any]]) -> None:
        self.runner = SimpleNamespace(dry_run=False)
        self.config = SimpleNamespace(
            clusters=targets,
            namespace="gpu-fault-system",
            delivery_component_digests={"cpu": "a" * 64},
        )
        self.live = live
        self.reads: list[list[str]] = []

    def _gpu(self, target: Any, *arguments: str) -> list[str]:
        return ["kubectl", "--context", target.context, *arguments]

    def _get_json(self, command: list[str]) -> dict[str, Any]:
        self.reads.append(command)
        assert "--ignore-not-found" in command, "an absent render is not an error"
        context = command[command.index("--context") + 1].removesuffix("-context")
        kind = "daemonset" if "daemonset" in command else "deployment"
        return self.live.get(context, {}).get(kind) or {}


# --- what the live renders serve ----------------------------------------------------


def test_the_served_variant_is_the_mark_else_what_the_render_does() -> None:
    assert DCGM.exporter_served_mode(_exporter("host")) == "host"
    assert (
        DCGM.exporter_served_mode(_exporter("embedded", remote=True)) == "embedded"
    ), "the mark wins"
    assert DCGM.exporter_served_mode(_exporter(None)) == "embedded", (
        "an exporter rendered before the mark dials no host engine"
    )
    assert DCGM.exporter_served_mode(_exporter(None, remote=True)) == "host"
    assert DCGM.exporter_served_mode({}) is None, "no DaemonSet serves nothing"
    assert DCGM.reconciler_served_mode(_reconciler("host")) == "host"
    assert DCGM.reconciler_served_mode(_reconciler(None)) == "embedded", (
        "a reconciler before the variable installs embedded Agents"
    )
    assert DCGM.reconciler_served_mode(_reconciler("hosted")) is None
    assert DCGM.reconciler_served_mode({}) is None
    assert DCGM.REMOTE_HOSTENGINE_ENV == RENDERING.REMOTE_HOSTENGINE_ENV


def test_the_probe_names_every_cluster_whose_live_renders_disagree() -> None:
    live_finding = _target("gpu-eks-cluster", **HOST_ENTRY)
    agreeing = _target("gpu-eks-b", **HOST_ENTRY)
    left_host = _target("gpu-eks-c")
    not_bootstrapped = _target("gpu-eks-d", **HOST_ENTRY)
    hyperpod = _target("hp-gpu-a", provider="hyperpod")
    release = LiveRelease(
        [live_finding, agreeing, left_host, not_bootstrapped, hyperpod],
        {
            "gpu-eks-cluster": {
                "daemonset": _exporter("embedded"),
                "deployment": _reconciler("host"),
            },
            "gpu-eks-b": {
                "daemonset": _exporter("host", remote=True),
                "deployment": _reconciler("host"),
            },
            "gpu-eks-c": {
                "daemonset": _exporter("host", remote=True),
                "deployment": _reconciler("host"),
            },
        },
    )

    drift = DCGM.dcgm_hostengine_drift(release)

    assert drift == {
        "gpu-eks-cluster": {
            "recorded": "host",
            "exporter": "embedded",
            "reconciler": "host",
        },
        "gpu-eks-c": {"recorded": "embedded", "exporter": "host", "reconciler": "host"},
    }, "the live finding (Agents host, exporter embedded), and a record left host"
    contexts = {command[command.index("--context") + 1] for command in release.reads}
    assert "hp-gpu-a-context" not in contexts, "HyperPod serves no mode: not read"
    assert len(release.reads) == 8, "one exporter and one reconciler read per cluster"


@pytest.mark.parametrize("gate", ["dry-run", "no-control-plane", "bare-double"])
def test_the_probe_never_reads_a_cluster_it_must_not(gate: str) -> None:
    release: Any = LiveRelease(
        [_target("gpu-eks-cluster", **HOST_ENTRY)],
        {"gpu-eks-cluster": {"daemonset": _exporter("embedded")}},
    )
    if gate == "dry-run":
        release.runner.dry_run = True
    elif gate == "no-control-plane":
        release.config.delivery_component_digests = {}
    else:
        release = SimpleNamespace(config=release.config, runner=release.runner)
    assert DCGM.dcgm_hostengine_drift(release) == {}
    if gate != "bare-double":
        assert release.reads == []


def test_status_and_verify_refuse_a_live_render_that_serves_another_mode() -> None:
    target = _target("gpu-eks-cluster", **HOST_ENTRY)
    release = LiveRelease(
        [target],
        {
            "gpu-eks-cluster": {
                "daemonset": _exporter("embedded"),
                "deployment": _reconciler("host"),
            }
        },
    )

    with pytest.raises(ReleaseError) as refusal:
        DCGM.check_dcgm_hostengine_agreement(release, target)

    assert str(refusal.value) == (
        "gpu-eks-cluster DCGM host engine disagreement: site.yaml records host; "
        f"the exporter DaemonSet {DCGM.EXPORTER_DAEMONSET} serves embedded; "
        "gpu-fault-admin deploy re-renders it for the recorded mode"
    )
    release.live["gpu-eks-cluster"]["daemonset"] = _exporter("host", remote=True)
    assert DCGM.check_dcgm_hostengine_agreement(release, target) == {
        "dcgm_hostengine_served": {"exporter": "host", "reconciler": "host"}
    }
    assert (
        DCGM.check_dcgm_hostengine_agreement(
            release, _target("hp-gpu-a", provider="hyperpod")
        )
        == {}
    ), "HyperPod is always host"


# --- sync-state ---------------------------------------------------------------------


class SyncRelease(LiveRelease):
    def __init__(self, persisted: dict[str, Any] | None, live: dict[str, Any]) -> None:
        super().__init__(
            [_target("gpu-eks-cluster", **HOST_ENTRY), _target("hp-gpu-a", "hyperpod")],
            live,
        )
        self.config.release_manifest_schema_version = 4
        self.persisted = persisted
        self.state: dict[str, Any] = {}
        self.saved: list[tuple[str, dict[str, Any]]] = []

    def _load_state(self) -> dict[str, Any]:
        if self.persisted is None:
            raise ReleaseError("regional release state is missing")
        self.state = dict(self.persisted)
        return self.state

    def _capture_previous(self) -> dict[str, Any]:
        return {"live_runtime_image": "runtime"}

    def _save_state(self, phase: str, **updates: Any) -> None:
        # The record's own digest, as ``_write_state`` writes it, then updates.
        self.state.update(
            {
                DCGM.DCGM_HOSTENGINES_STATE_KEY: DCGM.dcgm_hostengines_digest(
                    self.config
                ),
                **updates,
            },
            phase=phase,
        )
        self.saved.append((phase, dict(self.state)))


def test_sync_state_records_the_served_mode_and_keeps_the_observations() -> None:
    observed = {
        "mode": "host",
        "observed_at": "2026-10-17T13:44:30+00:00",
        "nodes": {"ip-10-0-42-179": {"state": "installed-inactive", "conflict": False}},
    }
    release = SyncRelease(
        {
            "phase": "rolled-back",
            DCGM.NODE_DCGM_STATE_KEY: {
                "gpu-eks-cluster": observed,
                "removed-cluster": observed,
            },
        },
        {"gpu-eks-cluster": {"daemonset": _exporter("embedded")}},
    )

    MODULE.sync_release_state(release)

    ((phase, state),) = release.saved
    assert phase == "complete" and state["transaction_committed"] is True
    assert state[DCGM.DCGM_HOSTENGINES_STATE_KEY] == EMBEDDED_DIGEST, (
        "the refused host mode is not committed as live: the exporter serves embedded"
    )
    assert state[DCGM.NODE_DCGM_STATE_KEY] == {"gpu-eks-cluster": observed}, (
        "the refused attempt's observations outlive the re-commit, for the "
        "clusters the site still manages"
    )
    assert state["previous"] is None, "the rest is written afresh"

    unread = SyncRelease(None, {})
    unread.runner.dry_run = True
    MODULE.sync_release_state(unread)
    ((_phase, state),) = unread.saved
    assert DCGM.NODE_DCGM_STATE_KEY not in state, "nothing to carry from no state"
    assert state[DCGM.DCGM_HOSTENGINES_STATE_KEY] == (
        DCGM.dcgm_hostengines_digest(unread.config)
    ), "a render that cannot be read counts as recorded"


# --- the live sequence --------------------------------------------------------------


class LiveRunner:
    """The live site after the refused release: exporter embedded, reconciler
    and Agents host. ``apply -f -`` of a DaemonSet replaces the live one."""

    dry_run = False

    def __init__(self) -> None:
        self.daemonset = _exporter("embedded")
        self.applied: list[str] = []

    def run(self, arguments: list[str], **options: Any) -> str:
        if "get" in arguments and "daemonset" in arguments:
            return json.dumps(self.daemonset)
        if "get" in arguments and GPU_RECONCILER_DEPLOYMENT in arguments:
            return json.dumps(_reconciler("host"))
        text = str(options.get("input_text") or "")
        if arguments[-3:] == ["apply", "-f", "-"] and "kind: DaemonSet" in text:
            self.applied.append(text)
            self.daemonset = next(
                item
                for item in yaml.safe_load_all(text)
                if isinstance(item, dict) and item.get("kind") == "DaemonSet"
            )
        return ""


def test_a_refused_flip_then_a_release_of_the_same_head_re_renders_the_exporter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _release(
        tmp_path, _hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY)
    )
    release.config = replace(
        release.config,
        delivery_component_digests={
            **release.config.delivery_component_digests,
            "cpu": "c" * 64,
        },
    )
    hyperpod, eks = release.config.clusters
    # What the refused release's sync-state committed before this batch: the
    # site's own record, host mode included, as if it had been rolled.
    STATE.save_state(release, "complete")
    stale = dict(release.state)
    assert stale[DCGM.DCGM_HOSTENGINES_STATE_KEY] == (
        DCGM.dcgm_hostengines_digest(release.config)
    )
    runner = LiveRunner()
    release.runner = runner
    for probe in ("_observability_drift", "_aurora_refresh_drift"):
        monkeypatch.setattr(release, probe, lambda: False)
    monkeypatch.setattr(DIFF, "cpu_pod_profile_drift", lambda _release: False)
    monkeypatch.setattr(DIFF, "node_scope_drift", lambda _release: {})

    with pytest.raises(ReleaseError, match="DCGM host engine disagreement"):
        DCGM.check_dcgm_hostengine_agreement(release, eks)
    diff = DIFF.classify_release(release, stale)

    assert DCGM.DCGM_HOSTENGINES_INPUT not in diff.changed, (
        "the recorded digest says host: alone it hides the embedded exporter"
    )
    assert DCGM.DCGM_HOSTENGINE_DRIFT_INPUT in diff.changed
    assert diff.kind is ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(ReleaseComponent.DCGM), "the exporter is re-rendered"
    assert plan.has(ReleaseComponent.RECONCILER) and plan.has(ReleaseComponent.AGENT)

    # The rollout's DCGM stage step for the planned component (the release's
    # ``_apply_gpu_dcgm_exporter`` is this function).
    GPU_BOOTSTRAP.apply_gpu_dcgm_exporter(release, eks)

    (applied,) = runner.applied
    daemonset = runner.daemonset
    assert daemonset["metadata"]["annotations"][DCGM.DAEMONSET_ANNOTATION] == "host"
    (container,) = [
        item
        for item in daemonset["spec"]["template"]["spec"]["containers"]
        if item["name"] == "exporter"
    ]
    assert {item["name"]: item.get("value") for item in container["env"]}[
        DCGM.REMOTE_HOSTENGINE_ENV
    ] == "127.0.0.1:5555", "the exporter dials the node's nvidia-dcgm"
    assert DCGM.dcgm_hostengine_drift(release) == {}, "nothing left to roll"
    assert DCGM.check_dcgm_hostengine_agreement(release, eks) == {
        "dcgm_hostengine_served": {"exporter": "host", "reconciler": "host"}
    }
    assert DCGM.check_dcgm_hostengine_agreement(release, hyperpod) == {}
    assert applied.count("kind: DaemonSet") == 1
