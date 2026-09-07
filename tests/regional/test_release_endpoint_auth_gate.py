from __future__ import annotations

import json
from pathlib import Path

import yaml

from gpu_fault_release import regional_gpu_bootstrap as GPU_BOOTSTRAP
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]


class EndpointCheckRunner:
    dry_run = False

    def __init__(self) -> None:
        self.applied: list[str] = []
        self.calls: list[list[str]] = []

    def run(self, arguments, *, input_text=None, capture=False, **_kwargs):
        self.calls.append(list(arguments))
        if input_text is not None:
            self.applied.append(input_text)
        if capture and "jsonpath={.status.phase}" in arguments:
            return "Succeeded"
        if capture and "logs" in arguments:
            return json.dumps({"tls": "verified", "cluster_token": "accepted"})
        return ""


def endpoint_check_pod(tmp_path: Path) -> dict:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = EndpointCheckRunner()
    release = MODULE.RegionalRelease(config, runner)
    release.runtime_image = "example.invalid/runtime@sha256:" + "0" * 64

    GPU_BOOTSTRAP.verify_gpu_control_plane_endpoint(release, config.clusters[0])

    assert runner.applied, "the endpoint gate applied no Pod"
    return yaml.safe_load(runner.applied[-1])


def test_endpoint_gate_probes_an_authenticated_route(tmp_path: Path) -> None:
    """The gate has to spend the token, not just reach the hostname.

    ``/healthz`` answers 200 for every caller, so a token the registry does not
    hold passes it. The first caller that notices is the Executor claim loop a
    phase later, once the Secret is installed and the whole data plane has been
    rolled onto it. Probing an authenticated route here is what turns that into
    a failure the endpoint rollback can undo by restoring one Secret.
    """

    pod = endpoint_check_pod(tmp_path)
    script = pod["spec"]["containers"][0]["command"][-1]
    compile(script, "endpoint-gate", "exec")

    assert "/v1/regional/executors/incident-ownership" in script, (
        "the gate never calls a route that checks the cluster token"
    )
    assert '"Authorization": "Bearer " + token' in script
    assert '"X-GPU-Fault-Cluster-ID": cluster' in script
    assert "error.code in (401, 403)" in script, (
        "a rejected token has to be named, not re-raised as a bare HTTPError"
    )


def test_endpoint_gate_reads_the_token_from_a_mount_not_the_environment(
    tmp_path: Path,
) -> None:
    pod = endpoint_check_pod(tmp_path)
    container = pod["spec"]["containers"][0]
    script = container["command"][-1]

    assert 'pathlib.Path("/auth/cluster-token")' in script
    names = {entry["name"] for entry in container["env"]}
    assert not any("TOKEN" in name for name in names), (
        f"the cluster token leaked into the Pod spec environment: {sorted(names)}"
    )

    mounted = {
        volume["name"]: volume["secret"]
        for volume in pod["spec"]["volumes"]
        if "secret" in volume
    }
    keys = {item["key"] for item in mounted["auth"]["items"]}
    assert keys == {"cluster-token", "cluster-id"}, (
        "the gate mounts more of the connection Secret than the probe needs"
    )
    assert mounted["auth"]["secretName"] == "gpu-fault-regional-connection"


def test_endpoint_gate_probe_id_cannot_name_a_real_incident(tmp_path: Path) -> None:
    """The probe must stay a read of nothing, so it can never own real work."""

    pod = endpoint_check_pod(tmp_path)
    env = {
        entry["name"]: entry.get("value")
        for entry in pod["spec"]["containers"][0]["env"]
    }

    assert env["PROBE_INCIDENT_ID"].startswith("gpu-fault-endpoint-gate-"), (
        "the probe id is not namespaced to the gate, so it could collide with a "
        f"real incident: {env['PROBE_INCIDENT_ID']}"
    )
