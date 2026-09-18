from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.node_key_custody_models import CustodyError
from gpu_fault.admin.node_key_custody_pods import cpu_role_container
from gpu_fault.container_env_snapshot import ROLE_DEPLOYMENTS
from tests.admin._security_activation_io_world import ActivationWorld
from tests.regional._security_consumer_pods import converged_deployment

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rendered_roles(monkeypatch):
    source = next(
        item
        for item in yaml.safe_load_all(
            (
                ROOT / "deploy/control-plane/base/control-plane-deployment.yaml"
            ).read_text()
        )
        if item and item.get("kind") == "Deployment"
    )
    source["spec"]["template"]["spec"]["containers"][0]["image"] = (
        "registry.invalid/runtime@sha256:" + "f" * 64
    )
    output = io.StringIO()
    renderer = ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"
    with monkeypatch.context() as patch, contextlib.redirect_stdout(output):
        patch.setattr(sys, "argv", [str(renderer), "--json"])
        patch.setattr(sys, "stdin", io.StringIO(json.dumps(source)))
        runpy.run_path(str(renderer), run_name="__main__")
    return {
        item["metadata"]["name"]: item
        for item in json.loads(output.getvalue())["items"]
        if item["kind"] == "Deployment"
    }


@pytest.fixture
def world(tmp_path, monkeypatch, rendered_roles):
    selected = ActivationWorld(tmp_path, monkeypatch)
    for name in ROLE_DEPLOYMENTS:
        document = selected.records["cpu", "deployment", name]
        document["spec"]["template"]["spec"]["containers"] = rendered_roles[name][
            "spec"
        ]["template"]["spec"]["containers"]
        document["spec"]["replicas"] = 1
        selected.records["cpu", "deployment", name] = converged_deployment(document)
    return selected


@pytest.mark.parametrize("name", ROLE_DEPLOYMENTS)
def test_capture_and_projection_accept_the_actual_rendered_cpu_container(world, name):
    world.io.capture()
    pod = world.io.consumer_pods("cpu", name)[0]
    world.bind_provisioned()
    digest = world.io.completed.cpu.keys["node-a"].sha256
    assert world.io.projected_key_precedes_process("cpu", pod, digest), (
        "the actual rendered role container must prove its projected key"
    )
    expected = world.records["cpu", "deployment", name]["spec"]["template"]["spec"][
        "containers"
    ][0]["name"]
    executions = [call for call in world.calls if "exec" in call]
    projection = executions[-1]
    assert projection[projection.index("-c") + 1] == expected, (
        "the projection command must select the renderer's container name"
    )


def test_full_activation_refreshes_ingress_worker_and_spool_using_their_rendered_names(
    world,
):
    snapshot = world.io.capture()
    world.io.guard(snapshot)
    world.io.fence(snapshot)
    world.key_values["node-a"] = "rendered-role-rotation-" + "x" * 40
    world.write_keys()
    world.bind_provisioned()
    world.io.refresh_executor(snapshot)
    world.io.install(snapshot)
    result = world.io.refresh_cpu(snapshot)
    assert set(result["activation_markers"]) == set(ROLE_DEPLOYMENTS), (
        "activation must refresh every configured CPU role"
    )
    assert world.io.observe(snapshot)["independent_witness"] == "NOT_PROVED", (
        "operational activation cannot manufacture an independent witness"
    )
    assert world.io.unfence(snapshot)["restored"] is True, (
        "successful activation must restore its original installer wave"
    )
    names = {
        call[call.index("-c") + 1]
        for call in world.calls
        if "exec" in call and "gpu-context" not in call
    }
    assert names == {"api", "control-worker", "telemetry-spool-worker"}, (
        "all three real role containers must be addressed during activation"
    )


@pytest.mark.parametrize("defect", ["unknown-role", "ambiguous-container"])
def test_unknown_or_ambiguous_cpu_container_never_gets_guessed(rendered_roles, defect):
    name = "gpu-fault-control-worker"
    spec = rendered_roles[name]["spec"]["template"]["spec"]
    if defect == "unknown-role":
        name = "unmanaged-deployment"
    else:
        spec["containers"].append({"name": "unrelated", "image": "another"})
    with pytest.raises(CustodyError, match="ambiguous or unknown"):
        cpu_role_container(name, spec)
