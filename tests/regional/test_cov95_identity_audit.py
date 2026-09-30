from __future__ import annotations

import contextlib
import io
import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import (
    AgentHeartbeat,
    FleetCompatibilityPolicy,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_auth_backlog as backlog
from tests._builders import build_context
from tests.regional._cov95_identity_support import ASGIBridge
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration
from tests.regional.test_multi_cluster_fixture_review import pod_document


def bound_api(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ASGIBridge]:
    context = build_context()
    context.regional_mode = True
    context.execution_token = "e" * 32
    for cluster, token in (("a", TOKEN_A), ("b", TOKEN_B)):
        context.store.save_regional_cluster(registration(cluster, token))
    bridge = ASGIBridge(context)
    monkeypatch.setattr(urllib.request, "urlopen", bridge.urlopen)
    monkeypatch.setattr(ssl, "create_default_context", lambda **kwargs: object())
    monkeypatch.setattr(ApplicationContext, "from_environment", lambda: context)
    return context, bridge


def probe(script: str, *arguments: str) -> dict[str, Any]:
    output = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(output):
        patch.setattr(sys, "argv", ["unit-probe", *arguments])
        exec(compile(script, "<cov95-identity-probe>", "exec"), {})
    return json.loads(output.getvalue())


def matrix_arguments(tmp_path: Path) -> list[str]:
    for name, value in (("a", TOKEN_A), ("b", TOKEN_B)):
        path = tmp_path / name
        path.write_text(value, encoding="ascii")
        path.chmod(0o600)
    return [
        "matrix",
        "--url",
        "https://unit.invalid",
        "--ca-file",
        str(tmp_path / "public-ca"),
        "--cluster-a",
        "a",
        "--cluster-b",
        "b",
        "--token-a-file",
        str(tmp_path / "a"),
        "--token-b-file",
        str(tmp_path / "b"),
        "--executor-artifact-sha256",
        "a" * 64,
        "--executor-compatibility-digest",
        "b" * 64,
    ]


def test_auth_matrix_request_builders_exercise_actual_authorization_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, bridge = bound_api(monkeypatch)
    args = audit.parser().parse_args(matrix_arguments(tmp_path))
    before = probe(audit.STORE_NEGATIVE_PROBE, "a", "b")
    results = audit.run_matrix(args)
    audit.validate_matrix(results, cluster_a="a")
    after = probe(audit.STORE_NEGATIVE_PROBE, "a", "b")
    assert before == after
    assert set(results) == set(audit.expected_statuses(results))
    assert len(bridge.requests) == len(results)
    assert context.store.list_remote_commands() == []
    assert context.store.list_agents("b") == []
    assert context.store.list_attempt_observations("b") == []
    assert context.store.list_raw_evidence("b") == []
    assert audit.probe_claims_leased_nothing(results) is True
    assert TOKEN_A not in json.dumps(results) and TOKEN_B not in json.dumps(results)


@pytest.mark.parametrize("store_reads", [False, True])
def test_auth_audit_main_sandwiches_requests_between_store_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    store_reads: bool,
) -> None:
    _context, bridge = bound_api(monkeypatch)
    calls = []
    argv = ["audit", *matrix_arguments(tmp_path)]
    if store_reads:
        argv.extend(
            [
                "--cpu-kubeconfig",
                str(tmp_path / "cpu"),
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "configmap" in command:
            calls.append(("release", len(bridge.requests)))
            value = {"data": {"state.json": '{"release_id":"unit-release"}'}}
        elif "exec" in command:
            calls.append(("store", len(bridge.requests)))
            value = probe(kwargs["input_text"], *command[command.index("-") + 1 :])
        else:
            assert "pod" in command
            calls.append(("pod", len(bridge.requests)))
            value = pod_document()
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    monkeypatch.setattr(audit, "run_fixture_command", run)
    assert audit.main() == (0 if store_reads else 1)
    result = json.loads(capsys.readouterr().out)
    assert audit.GUARDED_AUTH008_CASE not in result["verdicts"]
    assert audit.GUARDED_AUTH008_CASE in result["not_evaluated"]
    if store_reads:
        assert calls[:3] == [("release", 0), ("pod", 0), ("store", 0)]
        assert calls[-1] == ("store", len(bridge.requests))
        assert set(result["verdicts"].values()) == {"PASS"}
        assert not (tmp_path / "run" / "cases" / audit.GUARDED_AUTH008_CASE).exists(), (
            "the diagnostic matrix must not publish guarded AUTH008 evidence"
        )
    else:
        assert calls == []
        assert result["verdicts"]["GF-REGIONAL-AUTH-005"] == "FAIL"


@pytest.mark.parametrize(
    "defect", ["missing-list", "not-ready", "empty", "healthy-after-unready"]
)
def test_audit_api_pod_selection_rejects_unknown_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    document = pod_document()
    if defect == "missing-list":
        document = {}
    elif defect == "empty":
        document["items"] = []
    elif defect == "not-ready":
        document["items"][0]["status"]["conditions"] = []
    else:
        unready = {
            "metadata": {"name": "newer", "creationTimestamp": "2099"},
            "status": {"phase": "Pending"},
        }
        document["items"].append(unready)
    monkeypatch.setattr(
        audit,
        "run_fixture_command",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(document)),
    )
    if defect == "healthy-after-unready":
        assert audit.kubectl_api_pod(tmp_path / "cpu", "gpu-system") == "api"
    else:
        with pytest.raises(RuntimeError, match="incomplete|no Ready"):
            audit.kubectl_api_pod(tmp_path / "cpu", "gpu-system")


def test_store_probe_refuses_nonobject_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        audit,
        "run_fixture_command",
        lambda *args, **kwargs: SimpleNamespace(stdout="log-prefix\n[]"),
    )
    with pytest.raises(RuntimeError, match="did not return an object"):
        audit.store_snapshot(tmp_path / "cpu", "gpu-system", "api", ["a", "b"])


@pytest.mark.parametrize("defect", ["none", "same-cluster", "peer-release", "pending"])
def test_auth008_real_claim_protocol_owns_only_its_b_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    context, bridge = bound_api(monkeypatch)
    # The CPU Pod exports the release pins, not GPU_FAULT_RELEASE_ID.
    monkeypatch.delenv("GPU_FAULT_RELEASE_ID", raising=False)
    pins = {
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "c" * 64,
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": "d" * 64,
    }
    for name, value in pins.items():
        monkeypatch.setenv(name, value)
    # The bound API derives its executor policy from this environment too.
    monkeypatch.setenv(
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST", "b" * 64
    )
    events = []

    class Regional:
        def __init__(self, cluster: str) -> None:
            self.cluster = cluster

        def evidence_identity(self) -> dict[str, str]:
            return {
                "release_id": "foreign-release"
                if defect == "peer-release" and self.cluster == "b"
                else "unit-release",
                "cluster_id": self.cluster,
            }

        def release_pins(self) -> dict[str, str]:
            return dict(pins)

        def executor_python(
            self, script: str, *args: str, **kwargs: Any
        ) -> dict[str, Any]:
            assert kwargs["attempts"] == 1
            if script == backlog.AUTH008_EXECUTOR_IDENTITY_PROBE:
                return {
                    "cluster_id": self.cluster,
                    "executor_id": self.cluster + "/actual",
                    "artifact": "a" * 64,
                    "compatibility": "b" * 64,
                    "protocol": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
                    "probe_owner_configured": False,
                }
            assert script == backlog.AUTH008_CLAIM_PROBE
            with monkeypatch.context() as patch:
                patch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://unit.invalid")
                patch.setenv(
                    "GPU_FAULT_CONTROL_PLANE_CA_FILE", str(tmp_path / "public-ca")
                )
                patch.setenv(
                    "GPU_FAULT_CONTROL_PLANE_TOKEN",
                    TOKEN_A if self.cluster == "a" else TOKEN_B,
                )
                return probe(script, *args)

        def cpu_python(
            self, script: str, argument: str, **kwargs: Any
        ) -> dict[str, Any]:
            action, receipt = json.loads(argument)
            events.append(action)
            assert (tmp_path / f"auth008-intent-{receipt['nonce']}.json").exists(), (
                "AUTH008 candidate mutation preceded its ownership receipt"
            )
            result = probe(script, argument)
            if defect == "pending" and action == "seed":
                result["pending_unleased"] = False
            return result

    site = SimpleNamespace(regional=lambda target: Regional(target.cluster_id))
    primary, secondary = (
        SimpleNamespace(cluster_id="a"),
        SimpleNamespace(cluster_id="a" if defect == "same-cluster" else "b"),
    )
    if defect in {"same-cluster", "peer-release"}:
        with pytest.raises(
            backlog.IdentityAcceptanceError, match="distinct|release identity"
        ):
            backlog.run_auth008(site, primary, secondary, case_dir=tmp_path)
        assert events == [] and bridge.requests == []
        return
    result = backlog.run_auth008(site, primary, secondary, case_dir=tmp_path)
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert events[0] == "seed" and events[-1] == "cleanup"
    commands = context.store.list_remote_commands()
    assert len(commands) == 1
    assert commands[0].cluster_id == "b"
    assert commands[0].step.operation is WorkflowOperation.FREEZE_EVIDENCE
    assert commands[0].step.node_ids == []
    assert commands[0].status.value == "FAILED"
    assert commands[0].lease_token is None and commands[0].lease_owner is None
    assert (
        context.store.get_workflow(commands[0].workflow_request_id).status.value
        == "SUPERSEDED"
    )
    assert len(bridge.requests) == (5 if defect == "none" else 0)


@pytest.mark.parametrize("defect", ["none", "nested-cluster", "wrong-node-key"])
def test_auth_nested_identity_and_node_scoped_signature_use_actual_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    context, _bridge = bound_api(monkeypatch)
    now = datetime.now(timezone.utc)
    key_a, key_b = "a" * 64, "b" * 64
    context.fleet_registry = FleetRegistry(
        context.store,
        "m" * 64,
        FleetCompatibilityPolicy(required_node_action_key_version=2),
        node_secrets={"node-a": key_a, "node-b": key_b},
        now=lambda: now,
    )
    heartbeat = AgentHeartbeat(
        cluster_id="b" if defect == "nested-cluster" else "a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=3,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="policy",
        runtime_profile_version="profile",
        config_digest="b" * 64,
        node_action_key_version=2,
        observed_at=now,
        allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
    )
    envelope = SignedAgentHeartbeat(
        heartbeat=heartbeat,
        signature=sign_agent_heartbeat(
            heartbeat, key_b if defect == "wrong-node-key" else key_a
        ),
    )
    response = audit.post(
        "https://unit.invalid",
        tmp_path / "public-ca",
        "/v1/fleet/agents/heartbeat",
        cluster_id="a",
        token=TOKEN_A,
        payload=envelope.model_dump(mode="json"),
    )
    if defect == "none":
        assert response["status"] == 200
        assert response["body"]["node_action_key_version"] == 2
        assert [agent.node_id for agent in context.store.list_agents("a")] == ["node-a"]
    else:
        assert response == {
            "status": 403 if defect == "nested-cluster" else 409,
            "body": {
                "detail": audit.PAYLOAD_BINDING_DENIAL
                if defect == "nested-cluster"
                else "invalid agent heartbeat signature"
            },
        }
        assert context.store.list_agents("a") == []
    assert context.store.list_agents("b") == []
    assert context.store.list_remote_commands() == []


def test_get_builder_preserves_optional_cluster_and_token_headers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _context, bridge = bound_api(monkeypatch)
    assert (
        audit.get(
            "https://unit.invalid/",
            tmp_path / "public-ca",
            "/healthz",
            cluster_id="a",
            token=TOKEN_A,
        )["status"]
        == 200
    )
    request = bridge.requests[-1]
    assert request.full_url == "https://unit.invalid/healthz"
    assert request.get_header("X-gpu-fault-cluster-id") == "a"
    assert request.get_header("Authorization") == "Bearer " + TOKEN_A


def test_matrix_validation_reports_bad_status_and_foreign_claims() -> None:
    results = {
        name: {
            "status": status,
            "body": {"detail": audit.EXPECTED_DETAILS[name]}
            if name in audit.EXPECTED_DETAILS
            else {"commands": []},
        }
        for name, status in audit.expected_statuses({}).items()
    }
    results["AUTH-001"]["status"] = True
    results["AUTH-008-A-normal"]["body"] = {"commands": [{"cluster_id": "b"}]}
    with pytest.raises(AssertionError):
        audit.validate_matrix(results, cluster_a="a")
    errors = audit.matrix_errors(results, cluster_a="a")
    assert "status True, expected 401" in errors["AUTH-001"]
    assert errors["AUTH-008-A-normal"] == ["a claimed command is foreign"]
