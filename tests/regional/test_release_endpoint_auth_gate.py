from __future__ import annotations

import json
from pathlib import Path

import pytest
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


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class PhasedRunner(EndpointCheckRunner):
    """A probe Pod whose phase reads are scripted, with the last one repeating.

    Bounded `kubectl wait` verdicts are scripted too; a ``False`` consumes the
    wait's ``--timeout`` on the clock, as kubectl giving up at it would.
    """

    def __init__(self, clock: FakeClock, phases: list[str], verdicts: list[bool]):
        super().__init__()
        self.clock = clock
        self.phases = list(phases)
        self.verdicts = list(verdicts)
        self.waits: list[tuple[list[str], float]] = []

    def run(self, arguments, *, input_text=None, capture=False, **kwargs):
        if capture and "jsonpath={.status.phase}" in arguments:
            self.calls.append(list(arguments))
            return self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]
        if capture and "logs" in arguments:
            return "probe log"
        return super().run(arguments, input_text=input_text, capture=capture, **kwargs)

    def probe(self, arguments, *, timeout_seconds=None):
        self.waits.append((list(arguments), self.clock.now))
        met = self.verdicts.pop(0) if self.verdicts else False
        if not met:
            self.clock.now += _wait_timeout(arguments)
        return met


def _wait_timeout(arguments: list[str]) -> float:
    (flag,) = [value for value in arguments if value.startswith("--timeout=")]
    return float(flag.removeprefix("--timeout=").removesuffix("s"))


def _gate(tmp_path: Path, monkeypatch, runner: PhasedRunner) -> None:
    from gpu_fault_release import regional_release_rollout_wait as WAIT

    monkeypatch.setattr(GPU_BOOTSTRAP, "time", runner.clock)
    monkeypatch.setattr(WAIT, "time", runner.clock)
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, runner)
    release.runtime_image = "example.invalid/runtime@sha256:" + "0" * 64
    GPU_BOOTSTRAP.verify_gpu_control_plane_endpoint(release, config.clusters[0])


def test_endpoint_gate_wakes_the_moment_the_probe_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    """The 5s between phase reads is a bounded `kubectl wait`, not a timer."""

    runner = PhasedRunner(FakeClock(), ["Pending", "Succeeded"], [True])

    _gate(tmp_path, monkeypatch, runner)

    assert [arguments[-4:] for arguments, _ in runner.waits] == [
        [
            "wait",
            "pod/gpu-fault-control-plane-endpoint-check",
            "--for=jsonpath={.status.phase}=Succeeded",
            "--timeout=5s",
        ]
    ]
    assert "-n" in runner.waits[0][0]
    assert runner.clock.sleeps == [] and runner.clock.now == 0.0


def test_endpoint_gate_still_reads_a_failed_probe_within_the_interval(
    tmp_path: Path, monkeypatch
) -> None:
    runner = PhasedRunner(FakeClock(), ["Pending", "Failed"], [])

    with pytest.raises(MODULE.ReleaseError, match="GPU DNS/TLS check failed"):
        _gate(tmp_path, monkeypatch, runner)

    assert runner.clock.now == 5.0, "the Failed phase was not read at the next interval"


def test_endpoint_gate_keeps_its_300s_deadline(tmp_path: Path, monkeypatch) -> None:
    runner = PhasedRunner(FakeClock(), ["Running"], [])

    with pytest.raises(MODULE.ReleaseError, match="GPU DNS/TLS check timed out"):
        _gate(tmp_path, monkeypatch, runner)

    assert runner.clock.now == 300.0
    assert all(
        _wait_timeout(arguments) <= 300 - at for arguments, at in runner.waits
    ), runner.waits
    assert len(runner.waits) == 60
