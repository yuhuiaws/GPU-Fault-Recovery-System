from __future__ import annotations

import asyncio
import contextlib
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import identity_auth_backlog as backlog
from scripts.e2e.regional.identity_acceptance_common import (
    ACCEPTANCE_PROBE_OWNER,
    ClusterTarget,
    IdentityAcceptanceError,
)
from scripts.e2e.regional.identity_auth_probes import AUTH008_EXECUTOR_IDENTITY_PROBE
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests._builders import asgi_client, build_context
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration

# The release pins a CPU Pod exports; the AUTH-008 probe binds its receipt to
# these because CPU Pods carry no GPU_FAULT_RELEASE_ID.
RELEASE_PINS = {
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
    "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "c" * 64,
    "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": "d" * 64,
}
# The in-process API derives its executor policy from the same environment, so
# the Pod environment also names the compatibility digest the identities carry.
POD_ENVIRONMENT = {
    **RELEASE_PINS,
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
}
FIXTURE_ERROR_TEXT = (
    "cpu Pod probe failed: command failed (1): <sensitive output redacted> "
    "Authorization: Bearer probe-bearer-value token=probe-token-value"
)


def cluster_target(cluster: str, directory: Path) -> ClusterTarget:
    return ClusterTarget(
        cluster_id=cluster,
        context=f"context-{cluster}",
        region="us-east-1",
        hyperpod_cluster_name=f"hyperpod-{cluster}",
        eks_cluster_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{cluster}",
        executor_role_arn=f"arn:aws:iam::123456789012:role/executor-{cluster}",
        control_plane_url="https://unit.invalid",
        ca_file=directory / "public-ca",
    )


def execute_probe(
    monkeypatch: pytest.MonkeyPatch, script: str, argument: str
) -> dict[str, Any]:
    output = io.StringIO()
    with monkeypatch.context() as local, contextlib.redirect_stdout(output):
        local.setattr(sys, "argv", ["probe", argument])
        exec(compile(script, "<auth008-probe>", "exec"), {})
    result = json.loads(output.getvalue())
    assert isinstance(result, dict), "AUTH-008 probe must return a JSON object"
    return result


def backlog_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> tuple[Any, Any, Any, list[str]]:
    context = build_context()
    context.regional_mode = True
    for cluster, token in (("a", TOKEN_A), ("b", TOKEN_B)):
        context.store.save_regional_cluster(registration(cluster, token))
    monkeypatch.setattr(ApplicationContext, "from_environment", lambda: context)
    monkeypatch.delenv("GPU_FAULT_RELEASE_ID", raising=False)
    for name, value in POD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    events: list[str] = []
    receipt: dict[str, Any] = {}
    pin_reads: list[int] = []

    if defect == "cluster-filter":
        original_claim = context.store.claim_remote_commands

        def wrong_cluster(cluster: str, *args: Any, **kwargs: Any) -> Any:
            return original_claim("b" if cluster == "a" else cluster, *args, **kwargs)

        monkeypatch.setattr(context.store, "claim_remote_commands", wrong_cluster)

    class Regional:
        def __init__(self, cluster: str) -> None:
            self.cluster = cluster

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "release-test", "cluster_id": self.cluster}

        def release_pins(self) -> dict[str, str]:
            pin_reads.append(1)
            if defect == "pins-rolled" and len(pin_reads) > 1:
                return {
                    **RELEASE_PINS,
                    "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "e" * 64,
                }
            return dict(RELEASE_PINS)

        def executor_python(self, script: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["attempts"] == 1
            return {
                "cluster_id": self.cluster,
                "executor_id": f"{self.cluster}/actual-executor",
                "artifact": "a" * 64,
                "compatibility": "b" * 64,
                "protocol": 3
                if defect == "legacy-protocol"
                else CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
                "probe_owner_configured": defect == "configured-owner",
            }

        def cpu_python(
            self, script: str, argument: str, **kwargs: Any
        ) -> dict[str, Any]:
            assert kwargs["attempts"] == 1
            action, identity = json.loads(argument)
            receipt.update(identity)
            events.append(action)
            assert (tmp_path / f"auth008-intent-{identity['nonce']}.json").is_file(), (
                "owned IDs must be persisted before any candidate write"
            )
            if defect == "cleanup" and action == "cleanup":
                raise RuntimeError("cleanup unavailable")
            if defect == "fixture-error" and action == "seed":
                raise RegionalFixtureError(FIXTURE_ERROR_TEXT)
            value = execute_probe(monkeypatch, script, argument)
            if defect == "seed-ack" and action == "seed":
                raise RuntimeError("creation ACK lost")
            return value

    async def post(
        regional: Regional, header: str, executor_id: str, identity: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            **audit.claim_payload(
                executor_id=executor_id,
                artifact_sha256=identity["artifact"],
                compatibility_digest=identity["compatibility"],
            ),
            "executor_protocol_version": identity["protocol"],
        }
        events.append(f"claim:{regional.cluster}:{header}:{executor_id}")
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/regional/executors/claim",
                headers={
                    "Authorization": "Bearer "
                    + (TOKEN_A if regional.cluster == "a" else TOKEN_B),
                    "X-GPU-Fault-Cluster-ID": header,
                },
                json=payload,
            )
        body = response.json()
        if defect == "claim-ack" and body.get("commands"):
            raise RuntimeError("claim ACK lost")
        if "commands" in body:
            body["commands"] = [
                {
                    key: command[key]
                    for key in (
                        "command_id",
                        "cluster_id",
                        "workflow_request_id",
                        "incident_id",
                        "status",
                    )
                }
                for command in body["commands"]
            ]
        if defect == "positive-empty" and executor_id.endswith("/b") and header == "b":
            body["commands"] = []
        if defect == "wrong-403" and response.status_code == 403:
            body = {"detail": "unrelated denial"}
        return {"status": response.status_code, "body": body}

    def claim(regional: Regional, **kwargs: Any) -> dict[str, Any]:
        return asyncio.run(
            post(
                regional,
                kwargs["header_cluster"],
                kwargs["executor_id"],
                kwargs["identity"],
            )
        )

    monkeypatch.setattr(backlog, "auth008_claim", claim)
    site = SimpleNamespace(regional=lambda target: Regional(target.cluster_id))
    return site, context, receipt, events


@pytest.mark.parametrize("probe_owner", [False, True])
def test_auth008_deployed_identity_probe_advertises_current_protocol(
    monkeypatch: pytest.MonkeyPatch, probe_owner: bool
) -> None:
    for name, value in {
        "GPU_FAULT_CLUSTER_ID": "a",
        "GPU_FAULT_CLUSTER_EXECUTOR_ID": "a/owned-executor",
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
        "GPU_FAULT_RESET_OWNER": (
            ACCEPTANCE_PROBE_OWNER if probe_owner else "gpu-fault-node-agent"
        ),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(socket, "gethostname", lambda: "owned-executor")
    result = execute_probe(monkeypatch, AUTH008_EXECUTOR_IDENTITY_PROBE, "")
    assert result == {
        "cluster_id": "a",
        "executor_id": "a/owned-executor",
        "artifact": "a" * 64,
        "compatibility": "b" * 64,
        "protocol": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        "probe_owner_configured": probe_owner,
    }


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "cluster-filter",
        "seed-ack",
        "claim-ack",
        "cleanup",
        "positive-empty",
        "wrong-403",
        "legacy-protocol",
        "pins-rolled",
    ],
)
def test_auth008_uses_owned_b_backlog_and_retires_it_without_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, context, receipt, events = backlog_site(monkeypatch, tmp_path, defect)
    result = auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert receipt["required_pins"] == RELEASE_PINS, (
        "the receipt must carry the pins the CPU probe binds to"
    )
    assert receipt["release_id"] == "release-test", "release_id stays as evidence"
    if defect == "pins-rolled":
        assert result["checks"]["release_unchanged"] is False
        assert result["checks"]["b_can_claim_exact_candidate"] is True
    assert events[0] == "seed"
    assert events[-1] == "cleanup"
    commands = context.store.list_remote_commands()
    assert len(commands) == 1
    command = commands[0]
    assert command.cluster_id == "b"
    assert command.step.operation.value == "FREEZE_EVIDENCE"
    assert command.step.execution_owner == ACCEPTANCE_PROBE_OWNER
    assert command.step.node_ids == command.incident.node_ids == []
    assert command.incident.job_id is None
    assert command.result_details.get("no_action_executed") in {None, True}
    if defect != "cleanup":
        assert command.status is RemoteCommandStatus.FAILED
        assert command.lease_owner is command.lease_token is None
        workflow = context.store.get_workflow(command.workflow_request_id)
        assert workflow.status.value == "SUPERSEDED"
        assert workflow.execution_lease_expires_at is None
        assert result["checks"]["owned_records_retired"] is True
    if defect == "none":
        assert result["candidate_before"] == result["candidate_after_negatives"]
        assert result["checks"]["b_can_claim_exact_candidate"] is True
        assert any(item.endswith("b/actual-executor") for item in events), (
            "the spoofed A request must advertise the observed B executor identity"
        )
    if defect == "legacy-protocol":
        assert result["entries"]["AUTH-008-A-normal"]["status"] == 503
        assert result["entries"]["AUTH-008-A-fake-executor"]["status"] == 503
        assert result["entries"]["AUTH-008-B-header-A-token"]["status"] == 403
        assert result["entries"]["AUTH-008-A-header-B-token"]["status"] == 403
        assert command.last_lease_owner is None
        assert "positive_b_claim" not in result
    assert len(receipt["nonce"]) == 32
    assert TOKEN_A not in (tmp_path / "auth008-details.json").read_text()
    assert TOKEN_B not in (tmp_path / "auth008-details.json").read_text()


def test_auth008_refuses_a_probe_owner_on_a_physical_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, context, _receipt, events = backlog_site(
        monkeypatch, tmp_path, "configured-owner"
    )
    with pytest.raises(IdentityAcceptanceError, match="unsafe"):
        auth.run_auth008(
            site,
            cluster_target("a", tmp_path),
            cluster_target("b", tmp_path),
            case_dir=tmp_path,
        )
    assert events == []
    assert context.store.list_remote_commands() == []


def test_auth008_cleanup_refuses_foreign_ownership_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, context, receipt, _events = backlog_site(monkeypatch, tmp_path, "cleanup")
    result = auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    assert result["verdict"] == "FAIL"
    command = context.store.list_remote_commands()[0]
    assert command.status is RemoteCommandStatus.LEASED, (
        "foreign-claimant cleanup requires a successfully leased candidate"
    )
    changed_receipt = {**receipt, "claimants": ["foreign"]}
    with pytest.raises(RuntimeError, match="unowned claimant"):
        execute_probe(
            monkeypatch,
            auth.AUTH008_BACKLOG_PROBE,
            json.dumps(["cleanup", changed_receipt]),
        )
    assert (
        context.store.get_remote_command(command.command_id).status
        is RemoteCommandStatus.LEASED
    )
    for _ in range(2):
        cleaned = execute_probe(
            monkeypatch, auth.AUTH008_BACKLOG_PROBE, json.dumps(["cleanup", receipt])
        )
        assert cleaned["retired"] is True


def test_auth008_partial_seed_is_retired_without_removing_audit_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, context, receipt, _events = backlog_site(monkeypatch, tmp_path, "none")

    def fail_ensure(command: Any) -> None:
        raise RuntimeError("command creation failed")

    monkeypatch.setattr(context.store, "ensure_remote_command", fail_ensure)
    result = auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    assert result["verdict"] == "FAIL"
    assert result["checks"]["owned_records_retired"] is True
    workflow = context.store.get_workflow("auth008-workflow-" + receipt["nonce"])
    assert workflow.status.value == "SUPERSEDED"
    assert context.store.list_remote_commands() == []


def test_auth008_refuses_existing_backlog_and_keeps_its_old_ownership_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, context, _receipt, _events = backlog_site(monkeypatch, tmp_path, "cleanup")
    primary, secondary = (cluster_target("a", tmp_path), cluster_target("b", tmp_path))
    first = auth.run_auth008(site, primary, secondary, case_dir=tmp_path)
    command = context.store.list_remote_commands()[0]
    old_state = command.model_dump_json()
    old_receipt = (tmp_path / first["ownership_receipt"]).read_bytes()
    second = auth.run_auth008(site, primary, secondary, case_dir=tmp_path)
    assert second["verdict"] == "FAIL"
    assert first["ownership_receipt"] != second["ownership_receipt"]
    assert (tmp_path / first["ownership_receipt"]).read_bytes() == old_receipt
    assert context.store.list_remote_commands() == [command]
    assert (
        context.store.get_remote_command(command.command_id).model_dump_json()
        == old_state
    )


@pytest.mark.parametrize("record", ["command", "workflow", "incident"])
def test_auth008_cleanup_never_overwrites_changed_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record: str
) -> None:
    site, context, receipt, _events = backlog_site(monkeypatch, tmp_path, "cleanup")
    auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    assert (
        context.store.list_remote_commands()[0].status is RemoteCommandStatus.LEASED
    ), "ownership-drift cleanup requires a successfully leased candidate"
    getter_name = {
        "command": "get_remote_command",
        "workflow": "get_workflow",
        "incident": "get_incident",
    }[record]
    original_getter = getattr(context.store, getter_name)

    def changed(key: str) -> Any:
        return original_getter(key).model_copy(update={"fencing_token": 2})

    monkeypatch.setattr(context.store, getter_name, changed)
    with pytest.raises(RuntimeError, match="ownership changed"):
        execute_probe(
            monkeypatch, auth.AUTH008_BACKLOG_PROBE, json.dumps(["cleanup", receipt])
        )
    assert context.store.list_remote_commands()[0].status is RemoteCommandStatus.LEASED


@pytest.mark.parametrize("name", sorted(RELEASE_PINS))
def test_auth008_probe_refuses_a_pod_whose_release_pin_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    site, context, receipt, _events = backlog_site(monkeypatch, tmp_path, "cleanup")
    auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    command = context.store.list_remote_commands()[0]
    assert command.status is RemoteCommandStatus.LEASED, (
        "pin-drift cleanup requires a successfully leased candidate"
    )
    monkeypatch.setenv(name, "f" * 64)
    with pytest.raises(RuntimeError, match=name) as refused:
        execute_probe(
            monkeypatch, auth.AUTH008_BACKLOG_PROBE, json.dumps(["cleanup", receipt])
        )
    assert "release changed" in str(refused.value)
    assert (
        context.store.get_remote_command(command.command_id).status
        is RemoteCommandStatus.LEASED
    ), "a rolled release must not clean up another release's candidate"


@pytest.mark.parametrize("name", sorted(RELEASE_PINS))
def test_auth008_probe_refuses_a_pod_that_exports_no_release_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    site, context, receipt, _events = backlog_site(monkeypatch, tmp_path, "cleanup")
    auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    monkeypatch.delenv(name)
    with pytest.raises(RuntimeError, match="not exported") as refused:
        execute_probe(
            monkeypatch, auth.AUTH008_BACKLOG_PROBE, json.dumps(["seed", receipt])
        )
    assert not isinstance(refused.value, KeyError), (
        "a missing pin must be a refusal with a message, never a KeyError"
    )
    assert name in str(refused.value)
    assert context.store.list_remote_commands()[0].status is RemoteCommandStatus.LEASED


@pytest.mark.parametrize(
    "receipt_pins",
    [
        None,
        {},
        {"GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "c" * 64},
        {**RELEASE_PINS, "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": ""},
        {**RELEASE_PINS, "GPU_FAULT_RELEASE_ID": "release-test"},
    ],
)
def test_auth008_probe_refuses_a_receipt_without_complete_pins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, receipt_pins: Any
) -> None:
    site, context, receipt, _events = backlog_site(monkeypatch, tmp_path, "cleanup")
    auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    weakened = {key: value for key, value in receipt.items() if key != "required_pins"}
    if receipt_pins is not None:
        weakened["required_pins"] = receipt_pins
    with pytest.raises(RuntimeError, match="release pins are incomplete"):
        execute_probe(
            monkeypatch, auth.AUTH008_BACKLOG_PROBE, json.dumps(["cleanup", weakened])
        )
    assert context.store.list_remote_commands()[0].status is RemoteCommandStatus.LEASED


def test_auth008_records_the_sanitized_fixture_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, context, _receipt, events = backlog_site(
        monkeypatch, tmp_path, "fixture-error"
    )
    result = auth.run_auth008(
        site,
        cluster_target("a", tmp_path),
        cluster_target("b", tmp_path),
        case_dir=tmp_path,
    )
    assert result["verdict"] == "FAIL"
    assert result["error_type"] == "RegionalFixtureError"
    assert result["error"].startswith("cpu Pod probe failed: command failed (1)"), (
        result["error"]
    )
    assert "<sensitive output redacted>" in result["error"]
    for secret in ("probe-bearer-value", "probe-token-value"):
        assert secret not in result["error"]
    details = json.loads((tmp_path / "auth008-details.json").read_text())
    assert details["error"] == result["error"]
    assert details["error_type"] == "RegionalFixtureError"
    assert events == ["seed", "cleanup"], "cleanup still runs after a probe failure"
    assert result["checks"]["owned_records_retired"] is True
    assert context.store.list_remote_commands() == []
