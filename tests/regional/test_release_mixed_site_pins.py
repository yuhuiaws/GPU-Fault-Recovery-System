"""A mixed-provider site has one Agent pin per cluster, forward and back.

Live 2026-10-04 (settle deploy of 5a2603b39e41 over a HyperPod + plain-EKS
site): the finalize barrier judged the ``eks-v1`` Agents against the site's
``hyperpod-v1`` config digest ("pin-not-aligned" for both EKS nodes) and the
automatic rollback then refused with "previous Agent runtime_profile_version
differs across clusters". Both paths now carry the per-cluster profile.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.config import default_admin_config
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_agent_set as agent_set
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_legacy as LEGACY
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release import regional_release_rollback_context as ROLLBACK
from gpu_fault_release import regional_release_state as state
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import ReleaseExecutionPlan
from tests.regional._cov95_release_support import (
    NEW_IMAGE,
    OLD_IMAGE,
    ResourceRelease,
    deployment,
    previous_snapshot,
)
from tests.regional._release_orchestrator_support import ingress_pod_list_json

SITE_DIGEST = "c" * 64
EKS_DIGEST = "e" * 64
ARTIFACT = "a" * 64
COMPATIBILITY = "b" * 64


def _site_pin() -> dict[str, str]:
    return {
        "artifact_sha256": ARTIFACT,
        "compatibility_digest": COMPATIBILITY,
        "agent_protocol_version": "3",
        "config_digest": SITE_DIGEST,
    }


# --- the finalize barrier -------------------------------------------------------


def _run_barrier_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    request: dict[str, Any],
    agents_by_cluster: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Run the probe program against a store whose Agents heartbeat on every read."""

    reads = {"count": 0}
    started = datetime(2026, 10, 4, 13, 50, tzinfo=timezone.utc)

    def list_agents(cluster_id: str) -> list[Any]:
        reads["count"] += 1
        now = datetime.now(timezone.utc)
        return [
            SimpleNamespace(
                lifecycle_state="ACTIVE",
                last_seen_at=started + timedelta(seconds=reads["count"]),
                lease_expires_at=now + timedelta(seconds=85),
                **record,
            )
            for record in agents_by_cluster.get(cluster_id, [])
        ]

    monkeypatch.setitem(
        sys.modules,
        "gpu_fault.app",
        SimpleNamespace(
            ApplicationContext=SimpleNamespace(
                from_environment=lambda: SimpleNamespace(
                    store=SimpleNamespace(list_agents=list_agents)
                )
            )
        ),
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    exec(  # noqa: S102 - the probe is a program; running it is the assertion
        compile(
            PROBES.probe_source("agent_heartbeat_barrier"),
            "<probe:agent_heartbeat_barrier>",
            "exec",
        ),
        {"__name__": "agent_heartbeat_barrier"},
    )
    return json.loads(capsys.readouterr().out)


def _agent(node_id: str, config_digest: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "artifact_sha256": ARTIFACT,
        "compatibility_digest": COMPATIBILITY,
        "agent_protocol_version": 3,
        "config_digest": config_digest,
    }


_MIXED_AGENTS = {
    "hp": [_agent("hp-node-1", SITE_DIGEST)],
    "gpu-eks-cluster": [_agent("ip-10-0-33-205", EKS_DIGEST)],
}


def _barrier_request(**extra: Any) -> dict[str, Any]:
    return {
        "expected_by_cluster": {
            "hp": ["hp-node-1"],
            "gpu-eks-cluster": ["ip-10-0-33-205"],
        },
        "required_identity": _site_pin(),
        "timeout_seconds": 0,
        "poll_seconds": 0,
        "minimum_lease_remaining_seconds": 30,
        **extra,
    }


def test_barrier_probe_judges_each_cluster_against_its_own_pin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _run_barrier_probe(
        monkeypatch,
        capsys,
        _barrier_request(
            required_identities={
                "hp": _site_pin(),
                "gpu-eks-cluster": {**_site_pin(), "config_digest": EKS_DIGEST},
            }
        ),
        _MIXED_AGENTS,
    )
    assert result == {"status": "PASSED", "expected_count": 2, "refreshed_count": 2}


def test_barrier_probe_without_cluster_pins_reproduces_the_live_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pre-fix payload: the EKS Agent is measured against the site digest."""

    result = _run_barrier_probe(monkeypatch, capsys, _barrier_request(), _MIXED_AGENTS)
    assert result["status"] == "FAILED"
    assert result["refreshed_count"] == 1
    (blocker,) = result["blockers"]
    assert blocker["cluster_id"] == "gpu-eks-cluster"
    assert blocker["reason"] == "pin-not-aligned"
    assert blocker["observed_pin"]["config_digest"] == EKS_DIGEST
    assert blocker["required_pin"]["config_digest"] == SITE_DIGEST, (
        "a pin blocker says which pin it saw and which it expected"
    )


def test_barrier_probe_still_refuses_a_cluster_pin_that_is_not_met(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail-closed: a cluster pin is a stricter yardstick, never a bypass."""

    result = _run_barrier_probe(
        monkeypatch,
        capsys,
        _barrier_request(
            required_identities={
                "gpu-eks-cluster": {**_site_pin(), "config_digest": "f" * 64}
            }
        ),
        _MIXED_AGENTS,
    )
    assert result["status"] == "FAILED"
    assert [item["node_id"] for item in result["blockers"]] == ["ip-10-0-33-205"]


def test_candidate_pin_identities_follow_each_clusters_runtime_profile() -> None:
    release = SimpleNamespace(
        node_wheel_sha=ARTIFACT,
        config=SimpleNamespace(
            component_digests={"node_runtime": COMPATIBILITY},
            agent_protocol_version=3,
            agent_config_digest=SITE_DIGEST,
            clusters=(
                SimpleNamespace(cluster_id="hp"),
                SimpleNamespace(
                    cluster_id="gpu-eks-cluster", agent_config_digest=EKS_DIGEST
                ),
            ),
        ),
    )
    assert FLEET.candidate_agent_pin_identities(release) == {
        "hp": _site_pin(),
        "gpu-eks-cluster": {**_site_pin(), "config_digest": EKS_DIGEST},
    }
    single = SimpleNamespace(
        node_wheel_sha=ARTIFACT,
        config=SimpleNamespace(
            component_digests={"node_runtime": COMPATIBILITY},
            agent_protocol_version=3,
            agent_config_digest=SITE_DIGEST,
            clusters=(SimpleNamespace(cluster_id="hp"),),
        ),
    )
    assert FLEET.candidate_agent_pin_identities(single) == {"hp": _site_pin()}, (
        "a single-profile site pins every cluster to the site identity, as before"
    )


class _BarrierRunner:
    dry_run = False

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def run(self, args: list[str], **kwargs: Any) -> str:
        self.calls.append((args, kwargs))
        if "get" in args and "pod" in args:
            return ingress_pod_list_json("cpu-pod") if "json" in args else "cpu-pod"
        return self.result


def test_barrier_forwards_the_per_cluster_pins_and_names_the_open_window() -> None:
    runner = _BarrierRunner(
        '{"status":"PASSED","expected_count":1,"refreshed_count":1}'
    )
    release = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: list(args),
    )
    pins = {"gpu-eks-cluster": {**_site_pin(), "config_digest": EKS_DIGEST}}
    FLEET.wait_candidate_cpu_agent_heartbeats(
        release,
        {"gpu-eks-cluster": {"node_ids": ["ip-10-0-33-205"]}},
        required_identity=_site_pin(),
        required_identities=pins,
        timeout_seconds=1,
        poll_seconds=0,
    )
    payload = json.loads(str(runner.calls[1][1]["input_text"]))
    assert payload["required_identity"] == _site_pin()
    assert payload["required_identities"] == pins

    failing = _BarrierRunner(
        json.dumps(
            {
                "status": "FAILED",
                "expected_count": 1,
                "refreshed_count": 0,
                "blocker_count": 1,
                "blockers": [
                    {"node_id": "ip-10-0-33-205", "reason": "pin-not-aligned"}
                ],
            }
        )
    )
    release.runner = failing
    with pytest.raises(ReleaseError, match="window stays open"):
        FLEET.wait_candidate_cpu_agent_heartbeats(
            release,
            {"gpu-eks-cluster": {"node_ids": ["ip-10-0-33-205"]}},
            required_identities=pins,
            timeout_seconds=1,
            poll_seconds=0,
        )


# --- the rollback -----------------------------------------------------------------


def _identity(version: str, **extra: Any) -> dict[str, Any]:
    return {
        "agent_version": "0.10.0",
        "policy_version": "catalog",
        "runtime_profile_version": version,
        "node_ids": ["node"],
        **extra,
    }


_MIXED_IDENTITIES = {
    "hp": _identity("hyperpod-v1"),
    "gpu-eks-cluster": _identity("eks-v1"),
}
_EKS_OVERRIDE = {
    "gpu-eks-cluster": {"version": "eks-v1", "agent_config_digest": EKS_DIGEST}
}


def test_rollback_controller_config_keeps_the_site_profile_and_tolerates_cluster_profiles() -> (
    None
):
    config = LEGACY.rollback_controller_config(
        _MIXED_IDENTITIES,
        site_runtime_profile_version="hyperpod-v1",
        cluster_runtime_profiles=_EKS_OVERRIDE,
    )
    assert config["GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"] == "hyperpod-v1", (
        "the control-plane env carries the site pin; the EKS cluster keeps eks-v1 "
        "through its restored registry registration"
    )
    assert config["GPU_FAULT_REQUIRED_AGENT_VERSION"] == "0.10.0"
    assert config["GPU_FAULT_REQUIRED_POLICY_VERSION"] == "catalog"

    derived = LEGACY.rollback_controller_config(
        _MIXED_IDENTITIES, cluster_runtime_profiles=_EKS_OVERRIDE
    )
    assert derived["GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"] == "hyperpod-v1", (
        "without a recorded site version the clusters on the site profile define it"
    )


def test_rollback_controller_config_refuses_a_version_no_profile_explains() -> None:
    with pytest.raises(
        ReleaseError,
        match="gpu-eks-cluster previous Agent runtime_profile_version 'eks-v1'",
    ):
        LEGACY.rollback_controller_config(
            _MIXED_IDENTITIES,
            site_runtime_profile_version="hyperpod-v1",
            cluster_runtime_profiles={},
        )
    with pytest.raises(
        ReleaseError, match="runtime_profile_version differs across clusters"
    ):
        (
            LEGACY.rollback_controller_config(_MIXED_IDENTITIES),
            "the historical rule stands",
        )
    with pytest.raises(ReleaseError, match="policy_version differs"):
        LEGACY.rollback_controller_config(
            {
                "hp": _identity("hyperpod-v1"),
                "gpu-eks-cluster": _identity("eks-v1", policy_version="other"),
            },
            site_runtime_profile_version="hyperpod-v1",
            cluster_runtime_profiles=_EKS_OVERRIDE,
        )


def _rollback_release(
    changed: list[str] | None, overrides: bool = True
) -> SimpleNamespace:
    clusters = [SimpleNamespace(cluster_id="hp", overrides_runtime_profile=False)]
    if overrides:
        clusters.append(
            SimpleNamespace(
                cluster_id="gpu-eks-cluster",
                overrides_runtime_profile=True,
                runtime_profile_source=None,
                runtime_profile_template_source=None,
                runtime_profile_version="eks-v1",
                agent_config_digest=EKS_DIGEST,
            )
        )
    return SimpleNamespace(
        config=SimpleNamespace(
            clusters=tuple(clusters),
            runtime_profile_version="hyperpod-v1",
            agent_config_digest=SITE_DIGEST,
        ),
        state={}
        if changed is None
        else {"release_diff": {"kind": "DATA_PLANE_COMPATIBLE", "changed": changed}},
    )


def test_previous_cluster_profiles_prefer_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _rollback_release(["runtime_image"])
    recorded = {
        "gpu-eks-cluster": {"version": "eks-v0", "agent_config_digest": "d" * 64}
    }
    assert (
        ROLLBACK.previous_cluster_runtime_profiles(
            release, {"cluster_runtime_profiles": recorded}
        )
        == recorded
    )
    assert (
        ROLLBACK.previous_cluster_runtime_profiles(
            release, {"cluster_runtime_profiles": {}}
        )
        == {}
    ), "an explicit empty record means no cluster ran its own profile"


def test_previous_cluster_profiles_fall_back_to_the_candidate_when_no_profile_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live rollback-failed transaction: its snapshot predates the capture and
    its recorded diff moved images and wheels only, so the candidate's overrides
    are the previous release's."""

    monkeypatch.setattr(
        ROLLBACK,
        "cluster_runtime_profiles",
        lambda _config: {
            "gpu-eks-cluster": {
                "version": "eks-v1",
                "agent_config_digest": EKS_DIGEST,
                "policy_sha256": "p" * 64,
            }
        },
    )
    steps: list[dict[str, Any]] = []
    monkeypatch.setattr(
        ROLLBACK,
        "narrate_step",
        lambda kind, **fields: steps.append({"kind": kind, **fields}),
    )
    release = _rollback_release(["control_plane_wheel", "runtime_image", "node_bundle"])
    profiles = ROLLBACK.previous_cluster_runtime_profiles(release, {"metadata": {}})
    assert profiles["gpu-eks-cluster"]["version"] == "eks-v1"
    assert steps == [
        {
            "kind": "rollback-cluster-profiles",
            "source": "candidate-config",
            "reason": "snapshot-predates-capture; no profile component changed",
            "clusters": "gpu-eks-cluster",
        }
    ], "the log says where the per-cluster pins came from"


@pytest.mark.parametrize(
    ("changed", "detail"),
    [
        (["agent_config", "runtime_image"], "changed agent_config"),
        (["cluster_runtime_profiles"], "changed cluster_runtime_profiles"),
        (None, "recorded no release diff"),
    ],
)
def test_previous_cluster_profiles_refuse_when_the_previous_pins_are_unknowable(
    monkeypatch: pytest.MonkeyPatch, changed: list[str] | None, detail: str
) -> None:
    monkeypatch.setattr(
        ROLLBACK,
        "cluster_runtime_profiles",
        lambda _config: {
            "gpu-eks-cluster": {"version": "eks-v1", "agent_config_digest": EKS_DIGEST}
        },
    )
    with pytest.raises(ReleaseError, match=detail) as refusal:
        ROLLBACK.previous_cluster_runtime_profiles(
            _rollback_release(changed), {"metadata": {}}
        )
    assert "gpu-eks-cluster cannot be derived" in str(refusal.value)


def test_previous_cluster_profiles_are_empty_for_a_single_profile_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ROLLBACK, "cluster_runtime_profiles", lambda _config: {})
    assert (
        ROLLBACK.previous_cluster_runtime_profiles(
            _rollback_release(None, overrides=False), {"metadata": {}}
        )
        == {}
    ), "a HyperPod-only snapshot without the key behaves exactly as before"


def test_rollback_controller_config_for_uses_the_previous_cluster_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _rollback_release(["runtime_image"])
    previous = {
        "agent_identities": _MIXED_IDENTITIES,
        "cluster_runtime_profiles": _EKS_OVERRIDE,
    }
    config = ROLLBACK.rollback_controller_config_for(release, previous, "hyperpod-v1")
    assert config["GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"] == "hyperpod-v1"


def test_previous_snapshot_records_the_live_per_cluster_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = ResourceRelease(("gpu-a", "gpu-b"))
    release.state = {
        "release_id": "old",
        "runtime_image": NEW_IMAGE,
        "admin_config": default_admin_config().as_dict(),
        "cluster_runtime_profiles": {
            "gpu-b": {"version": "eks-v1", "agent_config_digest": EKS_DIGEST}
        },
    }
    release.node_installer_image = OLD_IMAGE
    release.adot_image = OLD_IMAGE
    for name in inventory.CPU_RUNTIME_DEPLOYMENTS:
        release.documents[("cpu", "deployment", name)] = deployment(name)
    release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")] = {
        "data": {}
    }
    monkeypatch.setattr(state, "remote_command_stats", lambda _release: {})
    monkeypatch.setattr(
        state, "capture_previous_monitoring", lambda *_args, **_kwargs: ({}, OLD_IMAGE)
    )
    identities = previous_snapshot(release)["agent_identities"]
    monkeypatch.setattr(state, "capture_agent_identities", lambda _release: identities)
    monkeypatch.setattr(agent_set, "agent_lease_facts", lambda _release: {})
    barrier = threading.Barrier(2)

    def capture(_release: Any, target: Any, **_kwargs: Any) -> tuple[Any, Any, str]:
        barrier.wait(timeout=5)
        return {}, {target.cluster_id: NEW_IMAGE}, OLD_IMAGE

    monkeypatch.setattr(state, "capture_gpu_cluster_snapshot", capture)
    result = state.capture_previous(
        release, ReleaseExecutionPlan(nodes=(Component.EXECUTOR,))
    )
    assert result["cluster_runtime_profiles"] == {
        "gpu-b": {"version": "eks-v1", "agent_config_digest": EKS_DIGEST}
    }, "the rollback restores each cluster to the profile the live record names"
    assert (
        result["cluster_runtime_profiles"]
        is not release.state["cluster_runtime_profiles"]
    )
