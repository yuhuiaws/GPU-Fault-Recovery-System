from __future__ import annotations

import io
import json
import runpy
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

from gpu_fault import fleet_cli

TOKEN = "local-test-execution-credential"
BASE = "https://control.invalid"
DEPLOYMENT = {
    "deployment_id": "release/a",
    "cluster_id": "cluster-a",
    "desired_agent_version": "test-version",
    "desired_artifact_sha256": "a" * 64,
    "status": "PLANNED",
    "nodes": [{"node_id": "node-a", "status": "PENDING"}],
}


def test_run_deployment_authenticates_reads_and_final_readiness() -> None:
    calls = []

    def requester(base, path, *, payload=None, token=None):
        assert token == TOKEN, (
            "every protected fleet request needs the selected credential"
        )
        calls.append((path, payload))
        if path.endswith("/readiness"):
            return {"ready": True}
        return {**DEPLOYMENT, "status": "SUCCEEDED"}

    result = fleet_cli.run_deployment(
        BASE,
        TOKEN,
        "release/a",
        "unused",
        poll_interval_seconds=1,
        wave_timeout_seconds=30,
        requester=requester,
    )
    assert result["readiness"]["ready"] is True, (
        "the final authenticated gate is enforced"
    )
    assert [path for path, _ in calls] == [
        "/v1/fleet/deployments/release%2Fa",
        "/v1/fleet/readiness",
    ], "read-only completion does not request a new wave"


def test_wave_transport_is_bounded_before_waiting_for_heartbeats() -> None:
    updates = []

    def requester(base, path, *, payload=None, token=None):
        if path.endswith("/next-wave"):
            return {"node_ids": ["node-a"]}
        if "/nodes/" in path:
            updates.append(payload)
            return {}
        return dict(DEPLOYMENT)

    def runner(command, **kwargs):
        assert 0 < kwargs["timeout"] <= 30, (
            "transport must not hang outside its wave deadline"
        )
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(SystemExit, match="transport failed"):
        fleet_cli.run_deployment(
            BASE,
            TOKEN,
            "release/a",
            "install {node_id}",
            poll_interval_seconds=1,
            wave_timeout_seconds=30,
            requester=requester,
            runner=runner,
            monotonic=lambda: 0,
        )
    assert updates[0]["status"] == "FAILED", "timeout stops further waves"
    assert "completion is unknown" in updates[0]["reason"], (
        "local timeout is not remote cancellation proof"
    )


@pytest.mark.parametrize(
    ("arguments", "method", "suffix"),
    [
        (["agents"], "GET", "/v1/fleet/agents"),
        (["agents", "--cluster-id", "a&b"], "GET", "/v1/fleet/agents?cluster_id=a%26b"),
        (
            ["readiness", "--cluster-id", "a", "--nodes", "node-a,node-b"],
            "POST",
            "/v1/fleet/readiness",
        ),
        (
            ["deployment", "--deployment-id", "a/b"],
            "GET",
            "/v1/fleet/deployments/a%2Fb",
        ),
        (
            ["next-wave", "--deployment-id", "a/b"],
            "POST",
            "/v1/fleet/deployments/a%2Fb/next-wave",
        ),
        (
            [
                "create-deployment",
                "--cluster-id",
                "a",
                "--nodes",
                "node-a",
                "--agent-version",
                "version",
                "--artifact-sha256",
                "a" * 64,
                "--policy-version",
                "policy",
                "--runtime-profile-version",
                "profile",
                "--config-digest",
                "b" * 64,
            ],
            "POST",
            "/v1/fleet/deployments",
        ),
    ],
)
def test_cli_preserves_credentials_method_and_scoped_payload(
    monkeypatch, capsys, arguments, method, suffix
) -> None:
    calls = []

    def urlopen(request, *, timeout):
        calls.append(request)
        assert request.method == method and request.full_url == BASE + suffix, (
            "selected identities must be encoded without changing the endpoint"
        )
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["x-gpu-fault-execution-token"] == TOKEN, (
            "authentication is not optional on reads"
        )
        assert timeout == 30, "HTTP calls retain a bounded timeout"
        return io.BytesIO(b'{"accepted":true}')

    monkeypatch.setattr(fleet_cli.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-fleet",
            "--control-plane-url",
            BASE,
            *arguments,
            "--execution-token",
            TOKEN,
        ],
    )
    fleet_cli.main()
    output = capsys.readouterr().out
    assert json.loads(output) == {"accepted": True}, (
        "CLI returns structured server output"
    )
    assert TOKEN not in output, "credentials do not enter normal command output"
    assert len(calls) == 1, "read and create commands do not issue extra mutations"
    if arguments[0] == "create-deployment":
        body = json.loads(calls[0].data)
        assert (
            body["node_ids"] == ["node-a"]
            and body["desired_agent_version"] == "version"
        ), "the release identity and target scope reach the API unchanged"


@pytest.mark.parametrize("status", [401, 403, 503])
def test_cli_surfaces_http_refusal_without_treating_it_as_an_empty_result(
    monkeypatch, status
) -> None:
    def refuse(request, **kwargs):
        raise HTTPError(
            request.full_url, status, "refused", {}, io.BytesIO(b"request refused")
        )

    monkeypatch.setattr(fleet_cli.urllib_request, "urlopen", refuse)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-fleet",
            "--control-plane-url",
            BASE,
            "agents",
            "--execution-token",
            TOKEN,
        ],
    )
    with pytest.raises(SystemExit, match=f"rejected request \\({status}\\)"):
        fleet_cli.main()


@pytest.mark.parametrize("nodes", ["", "node-a,node-a"])
def test_cli_refuses_empty_or_duplicate_node_scopes(nodes) -> None:
    with pytest.raises(SystemExit) as error:
        fleet_cli.parser().parse_args(
            [
                "--control-plane-url",
                BASE,
                "readiness",
                "--execution-token",
                TOKEN,
                "--cluster-id",
                "a",
                "--nodes",
                nodes,
            ]
        )
    assert error.value.code == 2, "invalid scope fails before transport"


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_poll_budget_must_be_a_positive_integer(value) -> None:
    with pytest.raises(SystemExit) as error:
        fleet_cli.parser().parse_args(
            [
                "--control-plane-url",
                BASE,
                "run-deployment",
                "--execution-token",
                TOKEN,
                "--deployment-id",
                "a",
                "--transport-command",
                "install",
                "--poll-interval-seconds",
                value,
            ]
        )
    assert error.value.code == 2, "invalid timing cannot reach the rollout loop"


def test_empty_credential_is_not_emitted_as_a_header(monkeypatch) -> None:
    def refuse(request, **kwargs):
        headers = {key.lower() for key, _ in request.header_items()}
        assert "x-gpu-fault-execution-token" not in headers, (
            "an empty value is not authentication"
        )
        raise HTTPError(request.full_url, 401, "unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(fleet_cli.urllib_request, "urlopen", refuse)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-fleet",
            "--control-plane-url",
            BASE,
            "agents",
            "--execution-token",
            "",
        ],
    )
    with pytest.raises(SystemExit, match="401"):
        fleet_cli.main()


def test_run_command_forwards_explicit_timing_to_the_deployment_driver(
    monkeypatch, capsys
) -> None:
    calls = []

    def deploy(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ready": True}

    monkeypatch.setattr(fleet_cli, "run_deployment", deploy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-fleet",
            "--control-plane-url",
            BASE,
            "run-deployment",
            "--execution-token",
            TOKEN,
            "--deployment-id",
            "release/a",
            "--transport-command",
            "install {node_id}",
            "--poll-interval-seconds",
            "3",
            "--wave-timeout-seconds",
            "75",
        ],
    )
    fleet_cli.main()
    assert calls == [
        (
            (BASE, TOKEN, "release/a", "install {node_id}"),
            {"poll_interval_seconds": 3, "wave_timeout_seconds": 75},
        )
    ], "explicit timing is neither lost nor left as strings"
    assert json.loads(capsys.readouterr().out) == {"ready": True}, (
        "driver result is serialized"
    )


def test_module_entrypoint_keeps_authenticated_read_only_behavior(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        fleet_cli.urllib_request, "urlopen", lambda *args, **kwargs: io.BytesIO()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-fleet",
            "--control-plane-url",
            BASE,
            "agents",
            "--execution-token",
            TOKEN,
        ],
    )
    runpy.run_path(str(Path(fleet_cli.__file__)), run_name="__main__")
    assert json.loads(capsys.readouterr().out) == {}, (
        "an empty success response remains an object"
    )


@pytest.mark.parametrize(
    ("command", "failure", "reason"),
    [
        ("install {unknown}", None, "invalid transport command template"),
        ('"unterminated', None, "invalid transport command template"),
        ("", None, "transport command is empty"),
        (
            "install",
            OSError("executable unavailable"),
            "OSError: executable unavailable",
        ),
        ("install", "x" * 1100, "transport exited with status 9"),
    ],
)
def test_transport_failure_is_reported_and_stops_later_waves(
    command, failure, reason
) -> None:
    updates = []
    requested = []

    def requester(base, path, *, payload=None, token=None):
        requested.append(path)
        if path.endswith("/next-wave"):
            return {"node_ids": ["node-a"]}
        if "/nodes/" in path:
            updates.append(payload)
            return {}
        return DEPLOYMENT

    def runner(arguments, **kwargs):
        if isinstance(failure, OSError):
            raise failure
        assert isinstance(failure, str), (
            "invalid templates must not invoke a subprocess"
        )
        return subprocess.CompletedProcess(arguments, 9, "", failure)

    with pytest.raises(SystemExit, match="transport failed"):
        fleet_cli.run_deployment(
            BASE,
            TOKEN,
            "release/a",
            command,
            poll_interval_seconds=1,
            wave_timeout_seconds=30,
            requester=requester,
            runner=runner,
        )
    assert updates[0]["status"] == "FAILED" and reason in updates[0]["reason"], (
        "the node failure preserves the actionable transport cause"
    )
    assert len(updates[0]["reason"]) < 1100, "transport output is bounded"
    assert sum(path.endswith("/next-wave") for path in requested) == 1, (
        "failure cannot advance a wave"
    )


def test_expired_wave_cannot_start_a_transport() -> None:
    clock = {"now": 0}
    updates = []

    def requester(base, path, *, payload=None, token=None):
        if path.endswith("/next-wave"):
            clock["now"] = 31
            return {"node_ids": ["node-a"]}
        if "/nodes/" in path:
            updates.append(payload)
            return {}
        return DEPLOYMENT

    def forbidden(*args, **kwargs):
        pytest.fail("expired wave started a transport")

    with pytest.raises(SystemExit, match="expired before start"):
        fleet_cli.run_deployment(
            BASE,
            TOKEN,
            "release/a",
            "install",
            poll_interval_seconds=1,
            wave_timeout_seconds=30,
            requester=requester,
            runner=forbidden,
            monotonic=lambda: clock["now"],
        )
    assert updates[0]["reason"] == "transport deadline expired before start", (
        "no remote action was started"
    )


@pytest.mark.parametrize("phase", ["initial", "heartbeat", "timeout", "readiness"])
def test_failed_or_timed_out_fleet_proof_never_reports_success(phase) -> None:
    calls = []
    clock = {"now": 0}
    initial = True

    def requester(base, path, *, payload=None, token=None):
        nonlocal initial
        calls.append((path, payload))
        if path.endswith("/readiness"):
            return {"ready": False, "reasons": ["stale identity"]}
        if path.endswith("/next-wave"):
            return {"node_ids": ["node-a"]}
        if "/nodes/" in path:
            return {}
        if initial:
            initial = False
            status = (
                "FAILED"
                if phase == "initial"
                else ("SUCCEEDED" if phase == "readiness" else "RUNNING")
            )
            return {**DEPLOYMENT, "status": status}
        clock["now"] = 31
        return {
            **DEPLOYMENT,
            "status": "RUNNING",
            "nodes": [
                {
                    "node_id": "node-a",
                    "status": "FAILED" if phase == "heartbeat" else "INSTALLING",
                }
            ],
        }

    with pytest.raises(SystemExit):
        fleet_cli.run_deployment(
            BASE,
            TOKEN,
            "release/a",
            "install",
            poll_interval_seconds=1,
            wave_timeout_seconds=30,
            requester=requester,
            runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
            monotonic=lambda: clock["now"],
        )
    updates = [payload for path, payload in calls if "/nodes/" in path]
    if phase == "timeout":
        assert updates == [
            {
                "status": "FAILED",
                "reason": "target heartbeat was not observed before timeout",
            }
        ], "only unconfirmed node installation is failed by the local timeout"
    else:
        assert not updates, (
            "existing fleet failures and readiness rejection do not invent node transitions"
        )


def test_multiple_waves_poll_without_rerunning_completed_transports() -> None:
    nodes = [
        {"node_id": "node-a", "status": "READY"},
        {"node_id": "node-b", "status": "INSTALLING"},
    ]
    reads = iter(
        [
            DEPLOYMENT,
            {**DEPLOYMENT, "status": "RUNNING", "nodes": nodes},
            {**DEPLOYMENT, "status": "RUNNING", "nodes": nodes},
            {**DEPLOYMENT, "status": "RUNNING", "nodes": nodes},
            {
                **DEPLOYMENT,
                "status": "SUCCEEDED",
                "nodes": [{**node, "status": "READY"} for node in nodes],
            },
        ]
    )
    waves = iter([["node-a"], ["node-b"]])
    commands = []
    sleeps = []
    clock = {"now": 0}

    def requester(base, path, *, payload=None, token=None):
        assert token == TOKEN, "all polling and final checks remain authenticated"
        if path.endswith("/readiness"):
            return {"ready": True}
        if path.endswith("/next-wave"):
            return {"node_ids": next(waves)}
        return next(reads)

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    result = fleet_cli.run_deployment(
        BASE,
        TOKEN,
        "release/a",
        "install {node_id}",
        poll_interval_seconds=1,
        wave_timeout_seconds=30,
        requester=requester,
        runner=runner,
        sleep=sleep,
        monotonic=lambda: clock["now"],
    )
    assert commands == [["install", "node-a"], ["install", "node-b"]], (
        "polling never resubmits physical work"
    )
    assert sleeps == [1], "unconfirmed heartbeat polling is paced"
    assert result["deployment"]["status"] == "SUCCEEDED", (
        "all waves and final readiness completed"
    )
