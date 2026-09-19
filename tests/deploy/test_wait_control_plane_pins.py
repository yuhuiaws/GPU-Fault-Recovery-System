"""The pin convergence wait proves what rolling the tier used to guarantee.

``apply-control-plane-role-split.sh`` no longer rolls a control-plane tier for
a pin change: the Pods poll ``gpu-fault-release-metadata`` themselves and
report the digest they serve as ``fleet_pins.content_sha256`` on ``/healthz``.
The wait asks every Pod of the selected role Deployments through ``kubectl
exec``; here a fake kubectl (handed over as ``--kubectl``, so the suite's
cluster-binary guard never sees a ``kubectl``) runs the very probe the wait
sends, against a local HTTP server standing in for the Pod, so the port
lookup, the probe and the verdict are the ones that ship.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/control-plane/tools/wait_control_plane_pins.py"
MODULE = lazy_script_module(SCRIPT)
EXPECTED = "a" * 64
STALE = "b" * 64
INGRESS = "gpu-fault-api-ha"
WORKER = "gpu-fault-control-worker"
SPOOL = "gpu-fault-telemetry-spool-worker"
# Answers the two calls the wait makes: the Pod list of one Deployment (from
# the STATE file) and ``exec <pod> -- <python> -c <probe>``, which runs the
# probe with this interpreter so it reaches the local stand-in server.
FAKE_KUBECTL = """#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as handle:
    handle.write(json.dumps(args) + "\\n")
with open(os.environ["STATE"]) as handle:
    state = json.load(handle)
if "get" in args:
    counter = os.environ["CALLS"] + ".gets"
    count = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
    with open(counter, "w") as handle:
        handle.write(str(count))
    if count <= int(os.environ.get("FAIL_GETS", "0")):
        print("Unable to connect to the server: connection reset by peer", file=sys.stderr)
        raise SystemExit(1)
    deployment = next(arg for arg in args if arg.startswith("app="))[4:]
    print(json.dumps({"items": state["pods"].get(deployment, [])}))
elif "exec" in args:
    code = args[args.index("-c") + 1]
    raise SystemExit(subprocess.run([sys.executable, "-c", code]).returncode)
else:
    raise SystemExit("unexpected kubectl call: " + " ".join(args))
"""


class PodServer:
    """One Pod's uvicorn stand-in: scripted ``/healthz`` answers, last repeats."""

    def __init__(self, *answers: tuple[int, dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.requests = 0
        pod = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                pod.requests += 1
                status, body = pod.answers[min(pod.requests, len(pod.answers)) - 1]
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_arguments: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.server.server_address[1])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def healthz(digest: str) -> dict[str, Any]:
    return {"status": "ok", "fleet_pins": {"content_sha256": digest}}


def pod(
    name: str,
    deployment: str,
    *,
    port: int | None,
    phase: str = "Running",
    terminating: bool = False,
) -> dict[str, Any]:
    container: dict[str, Any] = {"name": "api"}
    if port is not None:
        container["ports"] = [{"name": "http", "containerPort": port}]
    metadata: dict[str, Any] = {"name": name, "labels": {"app": deployment}}
    if terminating:
        metadata["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    return {
        "metadata": metadata,
        "spec": {"containers": [container]},
        "status": {"phase": phase},
    }


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.kubectl = tmp_path / "fake-kubectl"
        self.kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
        self.kubectl.chmod(0o755)
        self.calls = tmp_path / "calls.jsonl"
        self.state = tmp_path / "state.json"
        self.servers: list[PodServer] = []
        self.clock = 0.0
        self.sleeps: list[float] = []
        monkeypatch.setenv("CALLS", str(self.calls))
        monkeypatch.setenv("STATE", str(self.state))
        monkeypatch.delenv("GPU_FAULT_SPOOL_WORKER_METRICS_PORT", raising=False)
        monkeypatch.delenv("FAIL_GETS", raising=False)
        self.monkeypatch = monkeypatch

    def serve(self, *answers: tuple[int, dict[str, Any]]) -> PodServer:
        server = PodServer(*answers)
        self.servers.append(server)
        return server

    def argv(
        self,
        pods: dict[str, list[dict[str, Any]]],
        *,
        deployments: list[str] | None = None,
        timeout: int = 30,
        poll: str = "2",
        kubeconfig: str | None = None,
    ) -> list[str]:
        self.state.write_text(json.dumps({"pods": pods}), encoding="utf-8")
        argv = [
            "--namespace",
            "gpu-fault-system",
            "--expected-sha256",
            EXPECTED,
            "--timeout-seconds",
            str(timeout),
            "--deployments",
            ",".join(deployments or list(pods)),
            "--poll-seconds",
            poll,
            "--kubectl",
            str(self.kubectl),
        ]
        if kubeconfig is not None:
            argv.extend(["--kubeconfig", kubeconfig])
        return argv

    def run(self, pods: dict[str, list[dict[str, Any]]], **options: Any) -> int:
        def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)
            self.clock += seconds

        return int(
            MODULE.main(
                self.argv(pods, **options), monotonic=lambda: self.clock, sleep=sleep
            )
        )

    def kubectl_calls(self) -> list[list[str]]:
        if not self.calls.exists():
            return []
        return [
            json.loads(line)
            for line in self.calls.read_text(encoding="utf-8").splitlines()
        ]

    def probes(self, name: str) -> list[list[str]]:
        return [
            call
            for call in self.kubectl_calls()
            if "exec" in call and call[call.index("exec") + 1] == name
        ]

    def close(self) -> None:
        for server in self.servers:
            server.close()


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    instance = Harness(tmp_path, monkeypatch)
    yield instance
    instance.close()


def test_converges_when_every_pod_serves_the_expected_digest(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    servers = [harness.serve((200, healthz(EXPECTED))) for _index in range(3)]
    code = harness.run(
        {
            INGRESS: [
                pod("api-0", INGRESS, port=servers[0].port),
                pod("api-1", INGRESS, port=servers[1].port),
            ],
            WORKER: [pod("worker-0", WORKER, port=servers[2].port)],
        }
    )

    assert code == 0, capsys.readouterr().err
    assert harness.sleeps == [], "a converged control plane must not wait a poll"
    assert [server.requests for server in servers] == [1, 1, 1]
    captured = capsys.readouterr()
    assert "3 Pods" in captured.out and EXPECTED in captured.out, captured.out
    assert all(
        call[call.index("-n") + 1] == "gpu-fault-system"
        for call in harness.kubectl_calls()
    ), "every kubectl call must be scoped to the namespace"


def test_polls_until_a_lagging_pod_serves_the_digest(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = harness.serve((200, healthz(EXPECTED)))
    lagging = harness.serve(
        (200, healthz(STALE)), (200, healthz(STALE)), (200, healthz(EXPECTED))
    )
    code = harness.run(
        {
            INGRESS: [
                pod("api-0", INGRESS, port=prompt.port),
                pod("api-1", INGRESS, port=lagging.port),
            ]
        }
    )

    assert code == 0, capsys.readouterr().err
    assert harness.sleeps == [2.0, 2.0]
    assert lagging.requests == 3
    assert prompt.requests == 3, (
        "a converged Pod must be re-read on every poll: a replica restarted "
        "mid-wait serves its start-up snapshot until its first ConfigMap read"
    )
    assert capsys.readouterr().err.count("polling again") == 2


def test_timeout_reports_every_pod_digest_or_error_and_fails(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = harness.serve((200, healthz(STALE)))
    refused = harness.serve((200, healthz(EXPECTED)))
    refused.close()
    old_wheel = harness.serve((200, {"status": "ok"}))
    code = harness.run(
        {
            INGRESS: [
                pod("api-0", INGRESS, port=stale.port),
                pod("api-1", INGRESS, port=refused.port),
            ],
            WORKER: [pod("worker-0", WORKER, port=old_wheel.port)],
        },
        poll="10",
    )

    assert code == 1
    assert harness.clock >= 30, "the wait gave up before its window closed"
    report = capsys.readouterr().err.splitlines()
    assert f"{INGRESS}/api-0: {STALE}" in report
    assert any(
        line.startswith(f"{INGRESS}/api-1: error: ") and "refused" in line.lower()
        for line in report
    ), report
    assert any(
        line.startswith(f"{WORKER}/worker-0: error: ") and "fleet_pins" in line
        for line in report
    ), report
    assert report[-1] == (
        f"control-plane pins did not converge to {EXPECTED} within 30s"
    )


def test_503_healthz_with_the_expected_digest_counts_as_converged(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    """A replica whose readiness is degraded still reports the pins it serves;
    the HTTP status of /healthz is not the pin verdict."""

    degraded = harness.serve((503, {**healthz(EXPECTED), "status": "degraded"}))
    code = harness.run({INGRESS: [pod("api-0", INGRESS, port=degraded.port)]})

    assert code == 0, capsys.readouterr().err
    assert degraded.requests == 1


def test_spool_worker_without_a_named_http_port_uses_the_metrics_port_fallback(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    server = harness.serve((200, healthz(EXPECTED)))
    harness.monkeypatch.setenv("GPU_FAULT_SPOOL_WORKER_METRICS_PORT", str(server.port))
    code = harness.run({SPOOL: [pod("spool-0", SPOOL, port=None)]})

    assert code == 0, capsys.readouterr().err
    (probe,) = harness.probes("spool-0")
    assert f"http://127.0.0.1:{server.port}/healthz" in probe[probe.index("-c") + 1]
    assert probe[probe.index("--") + 1] == "/opt/gpu-fault/control-plane/bin/python"


def test_terminating_pods_are_left_out_of_the_verdict(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    """A replica on its way out (an old ReplicaSet draining after a roll) never
    converges and never serves the release; it must not hold the apply."""

    live = harness.serve((200, healthz(EXPECTED)))
    leaving = harness.serve((200, healthz(STALE)))
    code = harness.run(
        {
            INGRESS: [
                pod("api-0", INGRESS, port=live.port),
                pod("api-old", INGRESS, port=leaving.port, terminating=True),
            ]
        }
    )

    assert code == 0, capsys.readouterr().err
    assert harness.probes("api-old") == [], "a terminating Pod was probed"


def test_a_pending_pod_blocks_convergence_until_it_runs(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    live = harness.serve((200, healthz(EXPECTED)))
    code = harness.run(
        {
            INGRESS: [
                pod("api-0", INGRESS, port=live.port),
                pod("api-1", INGRESS, port=None, phase="Pending"),
            ]
        },
        poll="10",
    )

    assert code == 1
    assert f"{INGRESS}/api-1: error: Pod phase Pending" in (
        capsys.readouterr().err.splitlines()
    )
    assert harness.probes("api-1") == [], "a Pod that is not Running was probed"


def test_a_transient_pod_list_failure_is_retried_within_the_window(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    server = harness.serve((200, healthz(EXPECTED)))
    harness.monkeypatch.setenv("FAIL_GETS", "1")
    code = harness.run({INGRESS: [pod("api-0", INGRESS, port=server.port)]})

    assert code == 0, capsys.readouterr().err
    assert harness.sleeps == [2.0]
    assert sum("get" in call for call in harness.kubectl_calls()) == 2
    assert "connection reset" in capsys.readouterr().err


def test_a_deployment_scaled_to_zero_has_nothing_to_converge(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spool tier is scaled to zero when spool admission is off; the
    rollout gate just before this wait already proved the replica counts."""

    server = harness.serve((200, healthz(EXPECTED)))
    code = harness.run({WORKER: [pod("worker-0", WORKER, port=server.port)], SPOOL: []})

    assert code == 0, capsys.readouterr().err
    assert f"{SPOOL}: no Pods" in capsys.readouterr().err


def test_kubeconfig_reaches_every_kubectl_call(
    harness: Harness, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = harness.serve((200, healthz(EXPECTED)))
    code = harness.run(
        {INGRESS: [pod("api-0", INGRESS, port=server.port)]},
        kubeconfig=str(tmp_path / "cpu"),
    )

    assert code == 0, capsys.readouterr().err
    calls = harness.kubectl_calls()
    assert calls, "no kubectl call was made"
    assert all(call[:2] == ["--kubeconfig", str(tmp_path / "cpu")] for call in calls), (
        calls
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--expected-sha256", "A" * 64],
        ["--expected-sha256", "a" * 63],
        ["--timeout-seconds", "29"],
        ["--timeout-seconds", "1m"],
        ["--deployments", ""],
        ["--deployments", "gpu-fault-api-ha,Not_A_Name"],
        ["--poll-seconds", "0"],
        ["--namespace", ""],
    ],
)
def test_invalid_arguments_exit_2_without_touching_the_cluster(
    harness: Harness, arguments: list[str]
) -> None:
    argv = harness.argv({INGRESS: []})
    for flag, value in zip(arguments[::2], arguments[1::2], strict=True):
        argv[argv.index(flag) + 1] = value
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2, result.stderr
    assert "usage:" in result.stderr, result.stderr
    assert harness.kubectl_calls() == []


def test_command_line_entry_converges_against_a_live_clock(
    harness: Harness, tmp_path: Path
) -> None:
    server = harness.serve((200, healthz(EXPECTED)))
    argv = harness.argv(
        {INGRESS: [pod("api-0", INGRESS, port=server.port)]},
        kubeconfig=str(tmp_path / "cpu"),
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert EXPECTED in result.stdout
    assert server.requests == 1
