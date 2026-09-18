"""Offline checks: no live cases, physical adapters, or durable writes."""

from __future__ import annotations

import io
import json
import os
import secrets
import select
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from gpu_fault.app import ApplicationContext
from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalRegistryHead,
    RegionalRegistryRevision,
    cluster_token_sha256,
)
from gpu_fault.settings import StoreSettings
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.postgres.pool import StoreCredentials
from gpu_fault.store.shared.errors import NotFoundError
from scripts.e2e.regional import audit_regional_command_protocol_live as protocol
from scripts.e2e.regional.command_protocol_probes import ProtocolAuditError
from tests.regional.test_command_protocol_audit_contracts import FakeStore, bare_audit


@pytest.fixture
def credential_input() -> tuple[dict[str, Any], Any]:
    now = datetime.now(timezone.utc)
    tokens = [secrets.token_urlsafe(48), secrets.token_urlsafe(48)]
    registrations = [
        RegionalClusterRegistration(
            cluster_id=f"perf-cap-{index:03d}",
            region="us-west-2",
            hyperpod_cluster_name=f"perf-cap-{index:03d}",
            eks_cluster_arn=(
                f"arn:aws:eks:us-west-2:000000000000:cluster/perf-cap-{index:03d}"
            ),
            token_sha256=cluster_token_sha256(token),
            synthetic=True,
            synthetic_run_id="unit-capacity-run",
            synthetic_expires_at=now + timedelta(hours=1),
            allowed_namespaces=["training"],
            agent_endpoint_allowed_cidrs=["127.0.0.1/32"],
        )
        for index, token in enumerate(tokens)
    ]
    revision = RegionalRegistryRevision.build(
        generation=2,
        registrations=registrations,
        previous_generation=1,
        required_member_ids=[],
        reason="unit-only fixture",
    )
    head = RegionalRegistryHead(
        generation=revision.generation, content_sha256=revision.content_sha256
    )
    store = SimpleNamespace(
        get_regional_registry_head=lambda: head,
        get_regional_registry_revision=lambda generation: revision,
        list_agents=lambda cluster_id: [],
        remote_command_stats=lambda: {"open_by_cluster": {}},
        close=lambda: None,
    )
    envelope = {
        "schema_version": 1,
        "purpose": "regional-command-audit",
        "cluster_id": "perf-cap-000",
        "other_cluster_id": "perf-cap-001",
        "synthetic_run_id": "unit-capacity-run",
        "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "registry_generation": head.generation,
        "registry_content_sha256": head.content_sha256,
        "credentials": [
            {"cluster_id": registration.cluster_id, "token": token}
            for registration, token in zip(registrations, tokens, strict=True)
        ],
    }
    return envelope, store


def decode(document: dict[str, Any]) -> protocol.AuditCredentialEnvelope:
    return protocol.decode_credentials(
        json.dumps(document).encode(),
        cluster_id="perf-cap-000",
        other_cluster_id="perf-cap-001",
        synthetic_run_id="unit-capacity-run",
    )


def assert_redacted(error: BaseException, document: dict[str, Any]) -> None:
    output = "".join(traceback.format_exception(error))
    for credential in document["credentials"]:
        assert credential["token"] not in output, (
            "credential escaped exception boundary"
        )


@pytest.mark.parametrize(
    ("hot_mode", "queue_mode"),
    [(None, None), ("dedicated", "dual"), ("dual", "dedicated")],
)
def test_cpu_store_settings_native_modes_and_state_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hot_mode: str | None,
    queue_mode: str | None,
) -> None:
    from gpu_fault.store.postgres import store as native
    from scripts.component_wheels import component_modules

    assert {
        StoreSettings.__module__,
        ControlPlaneStore.__module__,
        StoreCredentials.__module__,
        native.PostgresStore.__module__,
    } <= component_modules("control_plane")
    current_url = "postgresql://unit@db.invalid/current"
    path = tmp_path / "store-url"
    path.write_text(current_url)
    path.chmod(0o600)
    for key, value in {
        "GPU_FAULT_STORE_URL": "postgresql://unit@db.invalid/stale",
        "GPU_FAULT_STORE_URL_FILE": str(path),
        "GPU_FAULT_POSTGRES_POOL_MIN_SIZE": "2",
        "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "5",
        "GPU_FAULT_POSTGRES_POOL_TIMEOUT_SECONDS": "7",
        "GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS": "11",
        "GPU_FAULT_POSTGRES_POOL_MAX_IDLE_SECONDS": "19",
        "GPU_FAULT_POSTGRES_POOL_MAX_LIFETIME_SECONDS": "23",
        "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT": "true",
    }.items():
        monkeypatch.setenv(key, value)
    for key, mode_value in (
        ("GPU_FAULT_POSTGRES_HOT_STATE_MODE", hot_mode),
        ("GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE", queue_mode),
    ):
        if mode_value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, mode_value)
    pool_options: dict[str, Any] = {}
    schema_requests: list[bool] = []
    queries: list[tuple[str, Any]] = []

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def execute(self, query: str, parameters: Any) -> None:
            queries.append((" ".join(query.split()), parameters))

        def fetchone(self) -> None:
            return None

    database = SimpleNamespace(cursor=Cursor, close=lambda: None)

    def open_pool(factory: Any, credentials: Any, **kwargs: Any) -> Any:
        assert credentials.conninfo() == current_url
        pool_options.update(kwargs)
        return object()

    monkeypatch.setattr(native, "open_writer_pool", open_pool)
    monkeypatch.setattr(native, "PooledPostgresDatabase", lambda *a, **k: database)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY", "1")
    monkeypatch.setattr(
        native.PostgresStore,
        "_initialize_schema_state",
        lambda self, initialize: schema_requests.append(initialize),
    )
    store = protocol.open_credential_audit_store()
    try:
        assert type(store) is native.PostgresStore
        assert store.hot_state_mode == (hot_mode or "dedicated")
        assert store.processor_queue_state_mode == (queue_mode or "dedicated")
        assert (pool_options["min_size"], pool_options["max_size"]) == (2, 5)
        assert pool_options["timeout"] == 7
        assert (pool_options["max_idle"], pool_options["max_lifetime"]) == (19, 23)
        assert "statement_timeout=11000" in pool_options["kwargs"]["options"]
        assert schema_requests == [False], "audit must never request schema bootstrap"
        # In this release, workflow/command reads use dedicated state dispatch
        # independently of the telemetry hot-state and processor queue modes.
        for read in (store.get_workflow, store.get_remote_command):
            with pytest.raises(NotFoundError):
                read("unit-missing")
        assert queries == [
            (
                "SELECT payload FROM gpu_fault_control_records WHERE kind=%s AND key=%s",
                (kind, "unit-missing"),
            )
            for kind in ("workflow", "remote_command")
        ]
    finally:
        store.close()


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "extra-credential",
        "duplicate-cluster",
        "wrong-purpose",
        "wrong-run",
        "wrong-order",
        "third-identity",
        "expired",
        "naive",
        "long-lived",
        "boolean-version",
        "bad-token-type",
    ],
)
def test_envelope_rejections_do_not_echo_any_input(
    credential_input: tuple[dict[str, Any], Any], fault: str
) -> None:
    document, _ = credential_input
    original = json.loads(json.dumps(document))
    if fault == "extra":
        document["unexpected"] = document["credentials"][0]["token"]
    elif fault == "extra-credential":
        document["credentials"].append(dict(document["credentials"][0]))
    elif fault == "duplicate-cluster":
        document["credentials"][1]["cluster_id"] = "perf-cap-000"
    elif fault == "wrong-purpose":
        document["purpose"] = "capacity-load"
    elif fault == "wrong-run":
        document["synthetic_run_id"] = "another-run"
    elif fault == "wrong-order":
        document["cluster_id"], document["other_cluster_id"] = (
            document["other_cluster_id"],
            document["cluster_id"],
        )
    elif fault == "third-identity":
        document["credentials"][1]["cluster_id"] = "perf-cap-002"
    elif fault == "expired":
        document["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
    elif fault == "naive":
        document["expires_at"] = datetime.now().isoformat()
    elif fault == "long-lived":
        document["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat()
    elif fault == "boolean-version":
        document["schema_version"] = True
    else:
        document["credentials"][0]["token"] = {
            "value": original["credentials"][0]["token"]
        }
    with pytest.raises(ProtocolAuditError, match="envelope rejected") as caught:
        decode(document)
    assert_redacted(caught.value, original)


def test_duplicate_json_keys_and_oversize_are_rejected(
    credential_input: tuple[dict[str, Any], Any],
) -> None:
    document, _ = credential_input
    encoded = json.dumps(document)
    duplicate = encoded[:-1] + ',"schema_version":1}'
    for raw in (
        duplicate.encode(),
        b"{" + document["credentials"][0]["token"].encode(),
        b" " * (protocol.CREDENTIAL_INPUT_LIMIT + 1),
    ):
        with pytest.raises(ProtocolAuditError) as caught:
            protocol.decode_credentials(
                raw,
                cluster_id="perf-cap-000",
                other_cluster_id="perf-cap-001",
                synthetic_run_id="unit-capacity-run",
            )
        assert_redacted(caught.value, document)


def test_hot_registry_path_never_reads_startup_tokens_or_builds_adapters(
    credential_input: tuple[dict[str, Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    document, store = credential_input
    monkeypatch.setenv("GPU_FAULT_REGIONAL_CLUSTERS_JSON", "stale-startup-data")
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: pytest.fail(
            "fresh-input audit must not bootstrap an ApplicationContext"
        ),
    )
    monkeypatch.setattr(protocol, "open_credential_audit_store", lambda: store)
    envelope = decode(document)
    audit = protocol.LiveProtocolAudit(
        cluster_id="perf-cap-000",
        other_cluster_id="perf-cap-001",
        executor_sha256="a" * 64,
        executor_digest="b" * 64,
        credential_envelope=envelope,
    )
    preflight = audit.preflight()
    assert preflight["errors"] == []
    assert preflight["isolated_cluster"] is False
    assert preflight["executor_ready_replicas"] is None
    assert (
        preflight["synthetic_registry_scope"]["physical_executor_absence"]
        == "NOT_OBSERVED"
    )
    assert set(audit.tokens) == protocol.PERF_AUDIT_CLUSTER_IDS
    assert audit.registry["perf-cap-000"]["synthetic_run_id"] == "unit-capacity-run"
    for credential in document["credentials"]:
        assert credential["token"] not in repr(envelope)
        assert credential["token"] not in json.dumps(preflight)


def test_startup_credential_path_uses_non_bootstrapping_store_and_closes_it(
    credential_input: tuple[dict[str, Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    document, store = credential_input
    closed: list[bool] = []
    store.close = lambda: closed.append(True)
    monkeypatch.setenv(
        "GPU_FAULT_REGIONAL_CLUSTERS_JSON", json.dumps(document["credentials"])
    )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: pytest.fail("startup tokens must not bootstrap an ApplicationContext"),
    )
    monkeypatch.setattr(protocol, "open_credential_audit_store", lambda: store)
    audit = protocol.LiveProtocolAudit(
        cluster_id="perf-cap-000",
        other_cluster_id="perf-cap-001",
        executor_sha256="a" * 64,
        executor_digest="b" * 64,
    )
    assert audit.tokens == {
        item["cluster_id"]: item["token"] for item in document["credentials"]
    }
    audit.close()
    assert closed == [True], "the audit must release the Store it opened"


@pytest.mark.parametrize(
    "overrides",
    [
        {"cluster_id": "physical-cluster"},
        {"isolated_cluster": True},
        {"executor_ready_replicas": 0},
        {"executor_ready_replicas": 2},
    ],
)
def test_input_cannot_turn_caller_assertions_into_isolation_proof(
    credential_input: tuple[dict[str, Any], Any],
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    document, _ = credential_input
    monkeypatch.setattr(
        protocol,
        "open_credential_audit_store",
        lambda: pytest.fail("scope must be checked before opening a Store"),
    )
    arguments = {
        "cluster_id": "perf-cap-000",
        "other_cluster_id": "perf-cap-001",
        "executor_sha256": "a" * 64,
        "executor_digest": "b" * 64,
        "credential_envelope": decode(document),
        **overrides,
    }
    with pytest.raises(ProtocolAuditError, match="isolation assertions"):
        protocol.LiveProtocolAudit(**arguments)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-token",
        "wrong-digest",
        "old-generation",
        "wrong-run",
        "not-synthetic",
        "expired",
        "disabled",
        "draining",
        "physical-arn",
        "agent-present",
        "registry-race",
        "validation-error",
        "retiring-token",
        "shared-token",
        "tampered-revision",
    ],
)
def test_current_durable_scope_is_authoritative_and_errors_are_redacted(
    credential_input: tuple[dict[str, Any], Any], fault: str
) -> None:
    document, store = credential_input
    original = json.loads(json.dumps(document))
    revision = store.get_regional_registry_revision(2)
    if fault == "wrong-token":
        document["credentials"][0]["token"] = secrets.token_urlsafe(48)
    elif fault == "wrong-digest":
        document["registry_content_sha256"] = "0" * 64
    elif fault == "old-generation":
        document["registry_generation"] = 1
    elif fault == "tampered-revision":
        changed = revision.registrations[0].model_copy(update={"enabled": False})
        revision = revision.model_copy(
            update={"registrations": [changed, revision.registrations[1]]}
        )
        store.get_regional_registry_revision = lambda generation: revision
    elif fault == "agent-present":
        store.list_agents = lambda cluster_id: [SimpleNamespace(node_id="unexpected")]
    elif fault == "registry-race":
        head = store.get_regional_registry_head()
        reads = iter([head, head.model_copy(update={"generation": 3})])
        store.get_regional_registry_head = lambda: next(reads)
    elif fault == "validation-error":

        def malformed_revision(generation: int) -> Any:
            return RegionalRegistryRevision.model_validate(
                {"credentials": original["credentials"]}
            )

        store.get_regional_registry_revision = malformed_revision
        with pytest.raises(ValidationError):
            malformed_revision(2)
    else:
        change_sets: dict[str, dict[str, Any]] = {
            "wrong-run": {"synthetic_run_id": "another-run"},
            "not-synthetic": {
                "synthetic": False,
                "synthetic_run_id": None,
                "synthetic_expires_at": None,
            },
            "expired": {
                "synthetic_expires_at": datetime.now(timezone.utc)
                - timedelta(seconds=1)
            },
            "disabled": {"enabled": False},
            "draining": {"lifecycle_state": "DRAINING"},
            "physical-arn": {
                "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/physical"
            },
            "retiring-token": {
                "token_sha256": cluster_token_sha256(secrets.token_urlsafe(48)),
                "retiring_token_sha256": revision.registrations[0].token_sha256,
                "token_rotation_expires_at": (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ),
            },
            "shared-token": {"token_sha256": revision.registrations[1].token_sha256},
        }
        changes = change_sets[fault]
        if fault == "shared-token":
            document["credentials"][0]["token"] = document["credentials"][1]["token"]
        values = revision.registrations[0].model_dump()
        values.update(changes)
        changed = RegionalClusterRegistration.model_validate(values)
        revision = RegionalRegistryRevision.build(
            generation=2,
            previous_generation=1,
            registrations=[changed, revision.registrations[1]],
            required_member_ids=[],
            reason="unit-only changed fixture",
        )
        head = RegionalRegistryHead(
            generation=2, content_sha256=revision.content_sha256
        )
        store.get_regional_registry_head = lambda: head
        store.get_regional_registry_revision = lambda generation: revision
        document["registry_content_sha256"] = revision.content_sha256
    with pytest.raises(ProtocolAuditError, match="durable credential") as caught:
        protocol.current_credential_registry(store, decode(document))
    assert_redacted(caught.value, original)


def test_pipe_input_and_emitted_code_use_separate_channels(
    credential_input: tuple[dict[str, Any], Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document, _ = credential_input
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "wb") as writer:
        writer.write(json.dumps(document).encode())
    with os.fdopen(read_fd, "rb") as reader:
        stdin = io.TextIOWrapper(reader)
        monkeypatch.setattr(sys, "stdin", stdin)
        monkeypatch.setattr(
            sys, "argv", ["audit", "--emit-probe", "--credentials-stdin"]
        )
        monkeypatch.setattr(
            ApplicationContext,
            "from_environment",
            lambda: pytest.fail("emission must not open a Store"),
        )
        assert protocol.main() == 0
        source = capsys.readouterr().out
        module = ModuleType("command_audit_bundle")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setitem(
            sys.modules, "command_protocol_probes", ModuleType("placeholder")
        )
        exec(
            compile(source, "<credential-free-emitted-probe>", "exec"), module.__dict__
        )
        monkeypatch.setattr(sys, "argv", ["-c"])
        observed = module.read_credentials(
            cluster_id="perf-cap-000",
            other_cluster_id="perf-cap-001",
            synthetic_run_id="unit-capacity-run",
        )
        assert observed.synthetic_run_id == "unit-capacity-run"
        for credential in document["credentials"]:
            assert credential["token"] not in source


def test_python_source_on_stdin_cannot_also_supply_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreadableInput:
        def fileno(self) -> int:
            pytest.fail("source-on-stdin rejection must happen before credential reads")

    monkeypatch.setattr(sys, "argv", ["-"])
    monkeypatch.setattr(sys, "stdin", UnreadableInput())
    with pytest.raises(ProtocolAuditError, match="consumes stdin as source"):
        protocol.read_credentials(
            cluster_id="perf-cap-000",
            other_cluster_id="perf-cap-001",
            synthetic_run_id="unit-capacity-run",
        )


def test_group_readable_stdin_file_is_rejected_without_echo(
    credential_input: tuple[dict[str, Any], Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, _ = credential_input
    path = tmp_path / "unit-input.json"
    path.write_text(json.dumps(document))
    path.chmod(0o640)
    with path.open() as stdin:
        monkeypatch.setattr(sys, "argv", ["-c"])
        monkeypatch.setattr(sys, "stdin", stdin)
        with pytest.raises(ProtocolAuditError, match="protected") as caught:
            protocol.read_credentials(
                cluster_id="perf-cap-000",
                other_cluster_id="perf-cap-001",
                synthetic_run_id="unit-capacity-run",
            )
        assert_redacted(caught.value, document)
        assert stdin.tell() == 0, "unprotected input must not be consumed"


def test_main_does_not_log_validation_error_representations(
    credential_input: tuple[dict[str, Any], Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document, _ = credential_input

    def fail(arguments: Any) -> Any:
        return RegionalRegistryRevision.model_validate({"extra": document})

    monkeypatch.setattr(protocol, "run_with_deadline", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--cluster-id",
            "perf-cap-000",
            "--other-cluster-id",
            "perf-cap-001",
            "--executor-sha256",
            "a" * 64,
            "--executor-digest",
            "b" * 64,
            "--case",
            "GF-REGIONAL-CMD-001",
            "--credentials-stdin",
            "--synthetic-run-id",
            "unit-capacity-run",
        ],
    )
    assert protocol.main() == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["verdict"] == "FAIL"
    for credential in document["credentials"]:
        assert credential["token"] not in captured.out + captured.err


def test_deadline_blocks_new_seeds_and_cleans_only_owned_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    store = FakeStore()
    audit = bare_audit(store=store)
    audit.deadline = protocol.AuditDeadline(10, 2)
    order = []

    def expire() -> None:
        order.append("first")
        audit.created_commands.add("cmd-audit-test-owned")
        clock[0] = 10.0
        audit.seed("must-not-be-written")

    monkeypatch.setattr(audit, "run_001", expire)
    monkeypatch.setattr(audit, "run_002", lambda: order.append("second"))
    summary = audit.run(("GF-REGIONAL-CMD-001", "GF-REGIONAL-CMD-002"))
    assert summary["verdict"] == "FAIL"
    assert summary["not_run"] == ["GF-REGIONAL-CMD-002"]
    assert order == ["first"]
    assert store.deleted == [("remote_command", "cmd-audit-test-owned")]
    assert not audit.created_commands, (
        "test_deadline_blocks_new_seeds_and_cleans_only_owned_records: expected no audit.created_commands"
    )
    with pytest.raises(protocol.AuditStopped):
        audit.seed("still-must-not-be-written")


def test_credential_validity_covers_hard_stop_and_cleanup_grace(
    credential_input: tuple[dict[str, Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _ = credential_input
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    deadline = protocol.AuditDeadline(300, 30)
    envelope = decode(document)
    for lifetime in (299, 310, 330):
        short = envelope.model_copy(
            update={
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=lifetime)
                ).isoformat()
            }
        )
        with pytest.raises(ProtocolAuditError, match="credential lifetime"):
            deadline.require_credential_lifetime(short)
    deadline.require_credential_lifetime(envelope)
    assert deadline.work_until == 300 and deadline.hard_until == 330
    assert deadline.cleanup_remaining == 30, (
        "validation cannot consume or renew cleanup"
    )


def test_short_lived_input_is_refused_before_opening_any_store(
    credential_input: tuple[dict[str, Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _ = credential_input
    document["expires_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=310)
    ).isoformat()
    monkeypatch.setattr(protocol, "read_credentials", lambda **kwargs: decode(document))
    monkeypatch.setattr(protocol.AuditDeadline, "__enter__", lambda self: self)
    monkeypatch.setattr(protocol.AuditDeadline, "__exit__", lambda *args: None)
    monkeypatch.setattr(
        protocol,
        "open_credential_audit_store",
        lambda: pytest.fail("short-lived input must not open the Store"),
    )
    arguments = protocol.parser().parse_args(
        [
            "--case",
            "GF-REGIONAL-CMD-001",
            "--credentials-stdin",
            "--synthetic-run-id",
            "unit-capacity-run",
        ]
    )
    summary = protocol.run_with_deadline(arguments)
    assert summary["verdict"] == "FAIL"
    assert summary["error"] == "audit aborted: ProtocolAuditError"
    assert summary["not_run"] == ["GF-REGIONAL-CMD-001"]


def test_full_selection_requires_an_explicit_total_cleanup_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = [
        argument
        for case_id in protocol.AUDITED_CASE_IDS
        for argument in ("--case", case_id)
    ]
    arguments = protocol.parser().parse_args(selection)
    with pytest.raises(ProtocolAuditError, match="explicit cumulative"):
        protocol.run_with_deadline(arguments)
    arguments = protocol.parser().parse_args(selection + ["--cleanup-seconds", "120"])
    assert arguments.cleanup_seconds == 120


def test_cleanup_budget_is_cumulative_and_never_renewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    deadline = protocol.AuditDeadline(10, 2)
    with deadline.cleanup():
        clock[0] = 1.5
    with pytest.raises(protocol.AuditStopped, match="cleanup budget"):
        with deadline.cleanup():
            clock[0] = 2.0
    with pytest.raises(protocol.AuditStopped, match="cleanup budget"):
        with deadline.cleanup():
            pytest.fail("cleanup allowance must not reset after exhaustion")


@pytest.mark.parametrize(
    "mode", ["cooperative", "blocked-producer", "blocked-cleanup", "early-cleanup"]
)
def test_cpu_watchdog_survives_closed_input_and_bounds_its_own_producer(
    mode: str,
) -> None:
    # Only a disposable local interpreter is signalled; no Store is constructed.
    source = r"""
import signal, sys, time
from scripts.e2e.regional.audit_regional_command_protocol_live import (
    AuditDeadline, AuditStopped,
)
mode = sys.argv[1]
assert sys.stdin.buffer.read() == b""
with AuditDeadline(3 if mode == "early-cleanup" else 0.3, 0.2) as deadline:
    if mode == "early-cleanup":
        print("cleanup-start", flush=True)
        with deadline.cleanup():
            time.sleep(10)
    if mode == "blocked-producer":
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        time.sleep(10)
    except AuditStopped:
        with deadline.cleanup():
            if mode == "blocked-cleanup":
                time.sleep(10)
        print("owned-cleanup-complete", flush=True)
"""
    with subprocess.Popen(
        [sys.executable, "-B", "-c", source, mode],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
    ) as child:
        try:
            if mode == "early-cleanup":
                assert child.stdout is not None
                ready, _, _ = select.select([child.stdout], [], [], 5)
                assert ready, "owned interpreter must reach its cleanup boundary"
                assert child.stdout.readline() == b"cleanup-start\n"
            output, _ = child.communicate(timeout=1.5 if mode == "early-cleanup" else 5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
    if mode == "cooperative":
        assert child.returncode == 0, "CPU-local stop must allow bounded cleanup"
        assert output == b"owned-cleanup-complete\n"
    else:
        expected_signal = -9 if mode == "blocked-producer" else -14
        assert child.returncode == expected_signal, (
            "only the owned interpreter may stop"
        )
        assert b"owned-cleanup-complete" not in output
