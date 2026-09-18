from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.admin import node_key_custody_activation as engine
from gpu_fault.admin import node_key_custody_activation_io as adapter
from gpu_fault.admin import node_key_custody_preflight as preflight
from gpu_fault.admin.node_key_custody_models import CustodyError
from tests.admin._security_activation_io_world import ActivationWorld


@pytest.fixture
def world(tmp_path, monkeypatch):
    return ActivationWorld(tmp_path, monkeypatch)


@pytest.mark.parametrize("defect", ["none", "release", "arn", "context"])
def test_host_preflight_uses_the_shared_release_engine_and_exact_authorized_pins(
    world, monkeypatch, defect
):
    target = SimpleNamespace(
        eks_cluster_arn="foreign" if defect == "arn" else world.gpu_arn,
        context="foreign" if defect == "context" else "gpu-context",
    )
    release = SimpleNamespace(
        release_id="foreign" if defect == "release" else "release-a",
        _target=lambda _cluster: target,
        executor_wheel_cm="executor-wheel",
        bundle_cm="node-bundle",
        node_wheel_sha="a" * 64,
        node_template_sha="d" * 64,
        executor_image=world.images["executor"],
        node_installer_image="installer-image",
    )
    site = object()
    calls = []
    site_reads = []
    monkeypatch.setattr(
        preflight,
        "load_site",
        lambda *args, **kwargs: site_reads.append((args, kwargs)) or site,
    )
    monkeypatch.setattr(
        preflight, "build_release", lambda value: release if value is site else None
    )
    monkeypatch.setattr(
        preflight,
        "run_node_installer_preflight",
        lambda selected, cluster, candidate: calls.append(
            (selected, cluster, candidate)
        ),
    )
    if defect != "none":
        with pytest.raises(CustodyError, match="authorized target"):
            adapter.CustodyActivationIO.host_preflight(world.io)
        assert calls == []
        return
    adapter.CustodyActivationIO.host_preflight(world.io)
    assert calls[0][:2] == (release, target)
    candidate = calls[0][2]
    assert candidate.node_names == ("node-a",) and candidate.max_unavailable == 1
    assert candidate.artifact_sha == "a" * 64
    assert candidate.template_sha256 == "d" * 64
    assert candidate.bundle_sha256 == "c" * 64
    assert candidate.node_compatibility_digest == "b" * 64
    assert site_reads == [
        (
            (world.context.state_dir / "site.yaml",),
            {"repository_root": world.context.repository_root},
        )
    ]


def test_activation_factory_constructs_the_bound_io_without_side_effects(world):
    concrete = engine.activation_io(world, world.context, world.authorization)
    assert isinstance(concrete, adapter.CustodyActivationIO), (
        "the activation factory must retain the bound production I/O adapter"
    )
    assert world.calls == []


@pytest.mark.parametrize("node", [None, "foreign-node"])
def test_unbound_rotation_target_never_creates_an_activation_driver(world, node):
    with pytest.raises(CustodyError, match="single-node"):
        adapter.CustodyActivationIO(
            world,
            world.context,
            world.authorization.model_copy(update={"rotate_node": node}),
        )
    assert world.calls == []


def test_target_key_drift_cannot_be_adopted_as_the_expected_projection(world):
    with pytest.raises(CustodyError, match="verified signed completion"):
        world.io.verify_expected_keys()
    world.bind_provisioned()
    world.key_values["node-a"] = "not-approved-" + "x" * 40
    world.write_keys()
    with pytest.raises(CustodyError, match="signed completion"):
        world.io.verify_expected_keys()
    assert world.patches == []


@pytest.mark.parametrize("raw", ["not-json", "[]", "{}", '{"phase":"complete"}'])
def test_activation_rejects_unknown_release_state(world, raw):
    world.records["cpu", "configmap", "gpu-fault-regional-release-state"]["data"][
        "state.json"
    ] = raw
    with pytest.raises(CustodyError, match="committed authorized release"):
        world.io.release_state()


@pytest.mark.parametrize("body", ["[]", "null", "not-json"])
def test_cpu_probe_rejects_incomplete_private_replies(world, monkeypatch, body):
    original = world.run
    monkeypatch.setattr(
        world,
        "run",
        lambda command, **kwargs: body
        if "exec" in command
        else original(command, **kwargs),
    )
    with pytest.raises(CustodyError, match="invalid evidence"):
        world.io.agents()


@pytest.mark.parametrize(
    "field", ["open_remote", "destructive_workflow_count", "agent_blocker_count"]
)
def test_any_fleet_safety_blocker_prevents_activation(world, monkeypatch, field):
    value = {
        "open_remote": {"PENDING": 0, "LEASED": 0, "WAITING": 0},
        "destructive_workflow_count": 0,
        "agent_blocker_count": 0,
    }
    if field == "open_remote":
        value[field]["LEASED"] = 1
    else:
        value[field] = 1
    monkeypatch.setattr(world.io, "cpu_probe", lambda *_a, **_k: value)
    with pytest.raises(CustodyError, match="fleet safety"):
        world.io.safety()
    assert world.patches == []


@pytest.mark.parametrize("defect", ["agent-count", "stale-agent", "image", "wave"])
def test_activation_admission_rejects_unhealthy_or_unbound_inputs(world, defect):
    if defect == "agent-count":
        original = world.io.agents
        world.io.agents = lambda: {"node-a": original()["node-a"]}
    elif defect == "stale-agent":
        record = world.store.get_agent("cluster-a", "node-a")
        world.store.save_agent(record.model_copy(update={"lease_expires_at": None}))
    elif defect == "image":
        world.records["gpu", "deployment", adapter.GPU_EXECUTOR_DEPLOYMENT]["spec"][
            "template"
        ]["spec"]["containers"][0]["image"] = "foreign-image"
    else:
        world.records["gpu", "configmap", "wave"]["data"]["allowed-nodes"] = "node-b"
    with pytest.raises(CustodyError):
        world.io.capture()
    assert world.patches == []


def test_guard_will_not_drain_a_new_incarnation_using_the_old_generation(world):
    snapshot = world.io.capture()
    current = world.store.get_agent("cluster-a", "node-a")
    world.store.save_agent(
        current.model_copy(update={"generation": current.generation + 1})
    )
    with pytest.raises(CustodyError, match="changed before drain"):
        world.io.guard(snapshot)


def test_readable_projection_metadata_requires_the_original_container(world):
    pod = world.io.consumer_pods("gpu", adapter.GPU_EXECUTOR_DEPLOYMENT)[0]
    pod["status"]["containerStatuses"] = []
    with pytest.raises(CustodyError, match="container identity"):
        world.io.projected_key_precedes_process("gpu", pod, "a" * 64)


def test_readonly_activation_observation_has_a_finite_wait(world, monkeypatch):
    snapshot = world.io.capture()
    ticks = iter([0.0, 121.0])
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(ticks))
    with pytest.raises(CustodyError, match="no fresh new Agent"):
        world.io.observe(snapshot)
