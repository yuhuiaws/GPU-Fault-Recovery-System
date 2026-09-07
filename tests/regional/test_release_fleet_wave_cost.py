"""What one wave costs, and which of those costs are the release's own.

The 2026-09-05 upgrade rolled four nodes as four waves and each wave spent
12.5s in safety (three `kubectl exec` probes, two of them the same wave-safety
probe five seconds apart) and 23-48s in install, most of the tail being the
Reconciler's poll boundary. These tests pin the three costs this work package
removes: the duplicate probe, the wave size the built-in cap already computed,
and the whole-fleet `get nodes` behind every convergence poll.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_release_agent_convergence as CONVERGENCE
from gpu_fault_release import regional_release_config as RELEASE_CONFIG
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_probes as PROBES
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]

CLUSTER = "hp-cluster"
ARTIFACT = "a" * 64
BUNDLE = "b" * 64
TEMPLATE = "e" * 64
CONFIG = "c" * 64
TARGET = SimpleNamespace(cluster_id="gpu-1", hyperpod_cluster_name=CLUSTER)

_DEPLOYMENT_IN_PROGRESS = {
    "status": "IN_PROGRESS",
    "waves": [["node-a"], ["node-b"]],
    "nodes": [
        {"node_id": "node-a", "status": "READY"},
        {"node_id": "node-b", "status": "PENDING"},
    ],
}
_DEPLOYMENT_DONE = {
    "waves": [["node-a"], ["node-b"]],
    "nodes": [
        {"node_id": "node-a", "status": "READY"},
        {"node_id": "node-b", "status": "READY"},
    ],
}


def _wave_context() -> Any:
    return FLEET.FleetWaveContext(
        phase="upgrade",
        deployment_id="deployment",
        node_names=("node-a", "node-b"),
        paused_identity=(BUNDLE, TEMPLATE),
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        expected_profile="profile-v1",
        executor_wheel_filename=None,
        expected_compatibility="compatibility",
        desired_bundle=BUNDLE,
        desired_template=TEMPLATE,
        template_config_map=None,
        max_unavailable=1,
        runtime_image=None,
        node_installer_image=None,
        allow_legacy_identity=False,
        agent_identity=None,
    )


def _snapshot(minimum_lease: float | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "open_remote": {},
        "destructive_workflow_count": 0,
        "agent_blocker_count": 0,
        "agent_blockers": [],
    }
    if minimum_lease is not None:
        snapshot["minimum_lease_remaining_seconds"] = minimum_lease
    return snapshot


def _run_one_wave(
    monkeypatch: pytest.MonkeyPatch,
    *,
    minimum_lease: float | None,
    probe_seconds: float = 8.0,
) -> int:
    """Roll one wave over a clock the wave-safety probe itself advances.

    ``probe_seconds`` is what one `kubectl exec` of the probe costs on the live
    cluster, which is exactly the time the lease evidence ages by before the
    post-lease margin is checked. Returns how many probe execs the wave paid for.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    clock = [1000.0]
    monkeypatch.setattr(FLEET.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(FLEET.time, "sleep", lambda _seconds: None)
    probes: list[dict[str, Any]] = []

    def snapshot(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        probes.append(kwargs)
        clock[0] += probe_seconds
        return _snapshot(minimum_lease)

    monkeypatch.setattr(FLEET, "rollout_wave_safety_snapshot", snapshot)
    monkeypatch.setattr(
        FLEET, "hand_wave_to_reconciler", lambda *_a, **_k: (BUNDLE, TEMPLATE)
    )
    reads = iter(({"status": "SUCCEEDED", **_DEPLOYMENT_DONE},))
    release = SimpleNamespace(
        _fleet_command=lambda operation, _payload: (
            {"node_ids": ["node-b"]} if operation == "next-wave" else next(reads)
        ),
        _wait_agents=lambda *_a, **_k: None,
    )

    FLEET.run_fleet_waves(
        release, TARGET, _wave_context(), dict(_DEPLOYMENT_IN_PROGRESS)
    )
    return len(probes)


def test_wave_runs_one_safety_probe_when_lease_margin_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second probe asked a question the first one already answered.

    Both execs report the same Agent leases; the only difference is the margin
    they are compared against. A lease that had 120s left when the first probe
    ran still has more than the post-lease margin once the probe and the
    `next-wave` call are paid for, and that is arithmetic, not a cluster read.
    """

    assert _run_one_wave(monkeypatch, minimum_lease=120.0) == 1


def test_wave_reprobes_when_lease_margin_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: too little proven margin means the probe runs again.

    35s of lease minus the 8s the probe itself cost leaves 27s, under the 30s
    the wave requires after taking the lease, so the evidence is re-read rather
    than assumed.
    """

    assert _run_one_wave(monkeypatch, minimum_lease=35.0) == 2


def test_wave_reprobes_when_the_snapshot_omits_the_lease_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe too old to report the minimum lease is not evidence of margin."""

    assert _run_one_wave(monkeypatch, minimum_lease=None) == 2


def _probe_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    nodes: tuple[str, ...],
    wave: tuple[str, ...],
    leases: dict[str, float],
) -> dict[str, Any]:
    """Run the wave-safety probe program and read the report it prints.

    The probe is shipped to the Pod as source and executed by `python3 -c`, so
    the only way to observe what it computes is to execute it here too, with the
    product modules it imports standing in for a live control plane.
    """

    now = datetime.now(timezone.utc)
    agents = [
        SimpleNamespace(
            node_id=node,
            lifecycle_state="ACTIVE",
            last_seen_at=now,
            lease_expires_at=now + timedelta(seconds=remaining),
        )
        for node, remaining in leases.items()
    ]
    store = SimpleNamespace(
        remote_command_stats=lambda: {"by_status": {}},
        list_workflows=lambda **_kwargs: [],
        list_agents=lambda _cluster_id: agents,
    )
    modules = {
        "gpu_fault.app": SimpleNamespace(
            ApplicationContext=SimpleNamespace(
                from_environment=lambda: SimpleNamespace(store=store)
            )
        ),
        "gpu_fault.models": SimpleNamespace(
            WorkflowStatus=SimpleNamespace(
                PENDING="PENDING",
                SAFETY_PENDING="SAFETY_PENDING",
                BLOCKED="BLOCKED",
                RUNNING="RUNNING",
            )
        ),
        "gpu_fault.operation_registry": SimpleNamespace(
            DESTRUCTIVE_OPERATIONS=frozenset()
        ),
        "gpu_fault.workflow_resolution": SimpleNamespace(
            verified_restore_successor=lambda *_args: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    request = {
        "cluster_id": "gpu-1",
        "nodes": list(nodes),
        "wave": list(wave),
        "minimum_lease_remaining_seconds": 30,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))

    # The probe is a program, not a module: running it *is* the assertion.
    exec(
        compile(
            PROBES.probe_source("rollout_wave_safety"),
            "<probe:rollout_wave_safety>",
            "exec",
        ),
        {"__name__": "rollout_wave_safety"},
    )

    return json.loads(capsys.readouterr().out)


def test_wave_safety_probe_reports_the_minimum_lease_outside_the_wave(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The engine now decides the post-lease margin on this number.

    It has to be the smallest lease among the nodes that *stay up*: the wave's
    own agents are about to be restarted, so their leases say nothing about the
    margin the cluster retains, and including one would let a wave skip the
    re-probe on the strength of a lease it is itself about to end.
    """

    report = _probe_report(
        monkeypatch,
        capsys,
        nodes=("node-a", "node-b", "node-c", "node-d"),
        wave=("node-d",),
        leases={"node-a": 300.0, "node-b": 120.0, "node-c": 900.0, "node-d": 10.0},
    )

    assert report["agent_blocker_count"] == 0, report
    assert report["minimum_lease_remaining_seconds"] == pytest.approx(120.0, abs=5.0), (
        report
    )


def test_wave_safety_probe_reports_no_minimum_when_the_wave_is_the_fleet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No node outside the wave means no proven margin, not an unbounded one.

    `wave_lease_margin_holds` reads `null` as "not evidence" and re-probes, so
    the probe must report it rather than omitting the field or reporting a
    number nobody measured.
    """

    report = _probe_report(
        monkeypatch,
        capsys,
        nodes=("node-a",),
        wave=("node-a",),
        leases={"node-a": 300.0},
    )

    assert report["minimum_lease_remaining_seconds"] is None, report


def test_wave_lease_margin_holds_rejects_values_that_are_not_seconds() -> None:
    """A JSON `true` is an `int` in Python, and it is not one second of lease.

    Everything here comes from a probe whose output shape can change under an
    older Pod, so anything that is not a real number fails closed and the wave
    pays for the second exec it would otherwise have skipped.
    """

    for minimum in (True, False, None, "120", [120], {}):
        assert not FLEET.wave_lease_margin_holds(
            {"minimum_lease_remaining_seconds": minimum}, elapsed_seconds=0.0
        ), f"invalid minimum type {type(minimum).__name__} must fail closed"
    assert not FLEET.wave_lease_margin_holds({}, elapsed_seconds=0.0), (
        "a snapshot without the lease minimum must fail closed"
    )
    assert not FLEET.wave_lease_margin_holds(None, elapsed_seconds=0.0), (
        "a missing snapshot must fail closed"
    )
    assert FLEET.wave_lease_margin_holds(
        {"minimum_lease_remaining_seconds": 120}, elapsed_seconds=0.0
    ), "an integer count of seconds is still evidence"


def _policy_release(configured: int) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(upgrade_max_unavailable=configured))


def test_auto_upgrade_max_unavailable_uses_size_cap() -> None:
    """`0` means "as wide as the built-in cap already allows".

    The site default of 1 turned a four-node cluster into four waves, each
    paying a full safety and Reconciler round trip, while the cap computed for
    exactly this purpose said four nodes could move together.
    """

    auto = _policy_release(0)

    assert FLEET.node_rollout_policy(auto, 4, phase="upgrade").max_unavailable == 4
    assert FLEET.node_rollout_policy(auto, 64, phase="upgrade").max_unavailable == 8
    assert FLEET.node_rollout_policy(auto, 256, phase="upgrade").max_unavailable == 16
    assert FLEET.node_rollout_policy(auto, 512, phase="upgrade").max_unavailable == 32
    explicit = FLEET.node_rollout_policy(_policy_release(2), 64, phase="upgrade")
    assert explicit.max_unavailable == 2
    unknown = FLEET.node_rollout_policy(
        auto,
        {"node-a": "UNKNOWN", "node-b": "zone-a", "node-c": "zone-a"},
        phase="upgrade",
    )
    assert unknown.max_unavailable == 1, "an unknown topology has no blast radius"


def test_auto_widens_a_single_domain_cluster_to_the_whole_wave() -> None:
    """The production shape: four nodes, one availability zone.

    With every node in one failure domain there is no cross-domain spread to
    ration, so the per-domain limit has to equal the wave -- if it stayed at the
    `ceil(effective / domains)` reading of a multi-domain fleet it would clamp
    the wave straight back to one node and auto would buy nothing here.
    """

    policy = FLEET.node_rollout_policy(
        _policy_release(0),
        {f"node-{index}": "zone-a" for index in range(4)},
        phase="upgrade",
    )

    assert policy.max_unavailable == 4
    assert policy.max_unavailable_per_failure_domain == 4


def test_first_wave_stays_a_single_node_canary() -> None:
    """Auto widens the steady waves, never the canary."""

    policy = FLEET.node_rollout_policy(_policy_release(0), 64, phase="upgrade")

    assert policy.first_wave_max_unavailable == 1


def test_release_config_accepts_auto_upgrade_max_unavailable(tmp_path: Path) -> None:
    """The rendered site value has to survive release-config validation."""

    path = config_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["release"]["upgrade_max_unavailable"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")

    assert RELEASE_CONFIG.ReleaseConfig.load(path).upgrade_max_unavailable == 0


def test_release_config_still_bounds_upgrade_max_unavailable(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["release"]["upgrade_max_unavailable"] = 33
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RELEASE_CONFIG.ReleaseError, match=r"0\.\.32"):
        RELEASE_CONFIG.ReleaseConfig.load(path)


def _node(name: str) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {"sagemaker.amazonaws.com/cluster-name": CLUSTER},
            "annotations": {
                "gpu-fault.io/installer-state": "Succeeded",
                "gpu-fault.io/installer-artifact-sha256": ARTIFACT,
                "gpu-fault.io/installer-config-digest": CONFIG,
                "gpu-fault.io/installer-bundle-sha256": BUNDLE,
                "gpu-fault.io/installer-template-sha256": TEMPLATE,
                "gpu-fault.io/installer-node-uid": f"uid-{name}",
            },
        }
    }


def _convergence_release(
    reads: list[list[str]], *, single_object: bool = False
) -> SimpleNamespace:
    def get_json(arguments: list[str]) -> dict[str, Any]:
        reads.append(list(arguments))
        names = [
            item
            for item in arguments[arguments.index("nodes") + 1 :]
            if not item.startswith("-")
        ]
        if single_object:
            # `kubectl get nodes <one-name> -o json` answers with the object.
            return _node(names[0])
        return {"items": [_node(name) for name in names]}

    return SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        bundle_sha=BUNDLE,
        node_template_sha=TEMPLATE,
        config=SimpleNamespace(
            agent_config_digest=CONFIG,
            runtime_profile_version="profile-v1",
            release_manifest_schema_version=3,
            upgrade_max_unavailable=0,
        ),
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=get_json,
        _agent_heartbeats_converged=lambda *_a, **_k: True,
    )


def test_wait_agents_lists_only_wave_nodes_for_small_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll for two nodes read every node in the fleet, every five seconds.

    On a large cluster that listing is the most expensive read in the release
    and it is repeated for the whole install window, so a wave small enough to
    name its nodes names them.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.setattr(
        CONVERGENCE,
        "gpu_node_items",
        lambda *_a, **_k: pytest.fail("a named wave must not list the whole fleet"),
    )
    reads: list[list[str]] = []

    CONVERGENCE.wait_agents(
        _convergence_release(reads),
        TARGET,
        ARTIFACT,
        bundle_sha=BUNDLE,
        template_sha=TEMPLATE,
        config_digest=CONFIG,
        node_names=("node-b", "node-a"),
        timeout_seconds=900,
    )

    assert reads == [["get", "nodes", "node-a", "node-b", "--ignore-not-found"]], reads


def test_wait_agents_falls_back_to_the_label_listing_for_large_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beyond sixteen names the argument list stops being the cheaper read."""

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    wave = tuple(f"node-{index:02d}" for index in range(17))
    listed: list[bool] = []
    monkeypatch.setattr(
        CONVERGENCE,
        "gpu_node_items",
        lambda *_a, **_k: listed.append(True) or [_node(name) for name in wave],
    )
    reads: list[list[str]] = []

    CONVERGENCE.wait_agents(
        _convergence_release(reads),
        TARGET,
        ARTIFACT,
        bundle_sha=BUNDLE,
        template_sha=TEMPLATE,
        config_digest=CONFIG,
        node_names=wave,
        timeout_seconds=900,
    )

    assert listed == [True] and reads == []


def test_wait_agents_reads_a_single_named_node_as_one_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`kubectl get nodes <one>` answers with the object, not with a list.

    A one-node wave is the common case on the production fleet, so parsing only
    `items` would make every such wave observe an empty cluster and time out.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.setattr(
        CONVERGENCE,
        "gpu_node_items",
        lambda *_a, **_k: pytest.fail("a named wave must not list the whole fleet"),
    )
    CONVERGENCE.wait_agents(
        _convergence_release([], single_object=True),
        TARGET,
        ARTIFACT,
        bundle_sha=BUNDLE,
        template_sha=TEMPLATE,
        config_digest=CONFIG,
        node_names=("node-a",),
        timeout_seconds=900,
    )


def test_reconciler_poll_boundary_is_five_seconds() -> None:
    """Two of these boundaries sit inside every wave's install window.

    A 15s reconcile pass meant a wave waited up to 15s for the installer Job to
    be created and up to another 15s for the next node, which is most of the
    measured 23-48s install on a one-node wave.
    """

    documents = [
        document
        for document in yaml.safe_load_all(
            (ROOT / "deploy/dataplane/node-installer-reconciler.yaml").read_text(
                encoding="utf-8"
            )
        )
        if isinstance(document, dict) and document.get("kind") == "Deployment"
    ]
    environment = {
        str(item.get("name")): str(item.get("value"))
        for document in documents
        for container in document["spec"]["template"]["spec"]["containers"]
        for item in container.get("env") or []
    }

    assert environment["GPU_FAULT_RECONCILE_SECONDS"] == "5"
