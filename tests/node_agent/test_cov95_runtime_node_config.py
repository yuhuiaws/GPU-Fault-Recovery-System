from __future__ import annotations

import socket
import ssl
from collections.abc import Callable
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor
from gpu_fault.node_agent import app as node_app
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import FakeRunner, command, envelope
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"secret": "short"}, "at least 32"),
        ({"node_action_key_version": 3}, "key version"),
        ({"inflight_wait_timeout_seconds": 9}, "in-flight wait timeout"),
        ({"inflight_wait_timeout_seconds": 7201}, "in-flight wait timeout"),
        ({"allowed_operations": {WorkflowOperation.RESTART_WORKLOAD}}, "unsupported"),
        ({"diagnostic_retention_seconds": 3599}, "at least one hour"),
        ({"diagnostic_max_archives": 0}, "max archives"),
        ({"python_stack_tool": "relative/py-spy"}, "path must be absolute"),
        ({"field_diagnostic_timeout_seconds": 59}, "Field Diagnostic timeout"),
        ({"field_diagnostic_timeout_seconds": 7201}, "Field Diagnostic timeout"),
        ({"service_quiesce_enabled": True}, "requires a quiesce manager"),
    ],
)
def test_invalid_runtime_policy_never_reaches_a_node_command(
    node_factory: Callable[..., NodeActionExecutor],
    changes: dict[str, Any],
    message: str,
) -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError, match=message):
        node_factory(runner=runner, **changes)
    assert runner.commands == []


@pytest.mark.parametrize(
    ("operation", "settings", "drift", "message"),
    [
        (WorkflowOperation.QUIESCE_GPU_SERVICES, {}, {}, "GPU reset is disabled"),
        (
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            {"reset_enabled": True},
            {},
            "GPU service quiesce is disabled",
        ),
        (
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            {"reset_enabled": True},
            {"service_quiesce_enabled": True},
            "quiesce manager is unavailable",
        ),
        (
            WorkflowOperation.RESET_GPU,
            {},
            {"service_quiesce_enabled": True},
            "quiesce manager is unavailable",
        ),
        (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            {},
            {},
            "full fabric reset requires GPU service quiesce",
        ),
        (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            {},
            {"service_quiesce_enabled": True},
            "quiesce manager is unavailable",
        ),
        (
            WorkflowOperation.RESTORE_GPU_SERVICES,
            {},
            {},
            "GPU service restore is disabled",
        ),
        (
            WorkflowOperation.RESTORE_GPU_SERVICES,
            {},
            {"service_quiesce_enabled": True},
            "quiesce manager is unavailable",
        ),
    ],
)
def test_missing_quiesce_prerequisites_fail_without_any_process_action(
    node_factory: Callable[..., NodeActionExecutor],
    operation: WorkflowOperation,
    settings: dict[str, Any],
    drift: dict[str, Any],
    message: str,
) -> None:
    runner = FakeRunner()
    agent = node_factory(allowed_operations={operation}, runner=runner, **settings)
    for name, value in drift.items():
        setattr(agent, name, value)
    result = agent.execute(envelope(command(operation)))
    assert result.status.value == "FAILED"
    assert message in result.error
    assert result.retryable is False
    assert runner.commands == []
    assert agent.ledger.get(result.command_id) == result


@pytest.mark.parametrize("host_mode", ["private", "fallback", "dns-error"])
@pytest.mark.parametrize("mutual_tls", [False, True])
def test_server_tls_and_host_selection_reach_only_the_recording_server(
    monkeypatch: pytest.MonkeyPatch, host_mode: str, mutual_tls: bool
) -> None:
    import uvicorn

    resolutions = []
    calls = []
    app = object()

    def resolve(name: str, port: Any, **kwargs: Any) -> list:
        resolutions.append(name)
        if host_mode == "dns-error" and name == "unit.domain":
            raise socket.gaierror("synthetic DNS failure")
        values = ["invalid", "127.0.0.1", "0.0.0.0", "169.254.169.254", "8.8.8.8"]
        if host_mode != "fallback":
            values.append("10.0.0.5")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in values]

    monkeypatch.setattr(node_app.socket, "getfqdn", lambda: "unit.domain")
    monkeypatch.setattr(node_app.socket, "gethostname", lambda: "unit")
    monkeypatch.setattr(node_app.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(node_app, "create_node_agent_app", lambda: app)
    monkeypatch.setattr(
        uvicorn, "run", lambda value, **kwargs: calls.append((value, kwargs))
    )
    monkeypatch.setenv("GPU_FAULT_NODE_AGENT_TLS_CERT", "/unit/server.crt")
    monkeypatch.setenv("GPU_FAULT_NODE_AGENT_TLS_KEY", "/unit/server.key")
    if mutual_tls:
        monkeypatch.setenv("GPU_FAULT_NODE_AGENT_TLS_CLIENT_CA", "/unit/client-ca.crt")
        monkeypatch.setenv(
            "GPU_FAULT_NODE_AGENT_TLS_KEY_PASSWORD", "synthetic-unit-value"
        )
    node_app.run()
    assert len(calls) == 1 and calls[0][0] is app
    options = calls[0][1]
    assert options["host"] == ("127.0.0.1" if host_mode == "fallback" else "10.0.0.5")
    assert options["port"] == 9099
    assert options["ssl_certfile"] == "/unit/server.crt"
    assert options["ssl_keyfile"] == "/unit/server.key"
    if mutual_tls:
        assert options["ssl_cert_reqs"] == ssl.CERT_REQUIRED
        assert options["ssl_ca_certs"] == "/unit/client-ca.crt"
        assert options["ssl_keyfile_password"] == "synthetic-unit-value"
    else:
        assert "ssl_cert_reqs" not in options
        assert "ssl_keyfile_password" not in options
    assert resolutions == (
        ["unit.domain"] if host_mode == "private" else ["unit.domain", "unit"]
    )


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"GPU_FAULT_NODE_AGENT_TLS_CERT": "/unit/cert"}, "needs both"),
        ({"GPU_FAULT_NODE_AGENT_TLS_KEY": "/unit/key"}, "needs both"),
        ({"GPU_FAULT_NODE_AGENT_TLS_CLIENT_CA": "/unit/ca"}, "verification requires"),
    ],
)
def test_incomplete_tls_configuration_never_starts_serving(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], message: str
) -> None:
    import uvicorn

    calls = []
    monkeypatch.setattr(node_app.socket, "getfqdn", lambda: "unit")
    monkeypatch.setattr(node_app.socket, "gethostname", lambda: "unit")
    monkeypatch.setattr(node_app.socket, "getaddrinfo", lambda *args, **kwargs: [])
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(args))
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        node_app.run()
    assert calls == []
