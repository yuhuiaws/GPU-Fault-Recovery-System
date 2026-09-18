"""Drive the owned CMD audit against the local ASGI application, never a Pod."""

from __future__ import annotations

import asyncio
import builtins
import io
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import closing
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError, SqliteStore
from scripts.e2e.regional import audit_regional_command_protocol_live as protocol
from scripts.e2e.regional import command_protocol_probes as probes
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration


@pytest.fixture
def audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[protocol.LiveProtocolAudit]:
    store = SqliteStore(str(tmp_path / "audit.db"))
    context = build_context(store=store)
    context.regional_mode = True
    context.execution_token = "e" * 32
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    monkeypatch.setattr(protocol, "open_credential_audit_store", lambda: context.store)
    monkeypatch.setenv(
        "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
        json.dumps(
            [
                {
                    "cluster_id": "cluster-a",
                    "token": TOKEN_A,
                    "hyperpod_cluster_name": "hp-cluster-a",
                },
                {
                    "cluster_id": "cluster-b",
                    "token": TOKEN_B,
                    "hyperpod_cluster_name": "hp-cluster-b",
                },
            ]
        ),
    )
    subject = protocol.LiveProtocolAudit(
        cluster_id="cluster-a",
        other_cluster_id="cluster-b",
        executor_sha256="a" * 64,
        executor_digest="b" * 64,
        isolated_cluster=True,
    )

    def request(
        method: str,
        path: str,
        *,
        cluster_id: str | None,
        token: str | None,
        payload: Any = None,
    ) -> tuple[int, Any]:
        async def call() -> tuple[int, Any]:
            headers = {}
            if cluster_id is not None:
                headers["X-GPU-Fault-Cluster-ID"] = cluster_id
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            async with asgi_client(context) as client:
                response = await client.request(
                    method, path, headers=headers, json=payload
                )
                return response.status_code, response.json()

        return asyncio.run(call())

    monkeypatch.setattr(subject, "_request", request)
    try:
        yield subject
    finally:
        store.close()


def test_cmd009_accepts_the_actual_caller_derived_not_found_contract(
    audit: Any,
) -> None:
    audit.run_009()
    assert audit.results["GF-REGIONAL-CMD-009"]["missing"]["status"] == 404
    assert audit.results["GF-REGIONAL-CMD-009"]["cross_cluster"]["status"] == 404


def test_cmd010_proves_safe_stale_fence_terminalization(audit: Any) -> None:
    audit.run_010()
    recorded = audit.results["GF-REGIONAL-CMD-010"]
    assert recorded["complete_status"] == 200
    assert recorded["command_after"]["status"] == "FAILED"
    assert recorded["command_after"]["status_source"] == "stale-fence"
    assert recorded["command_after"]["post_stale_fence_status"] == "SUCCEEDED"
    assert recorded["reclaimed"] is False


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16])
def test_audit_scenarios_use_real_local_model_and_store_contracts(
    audit: Any, number: int
) -> None:
    getattr(audit, f"run_{number:03d}")()
    assert f"GF-REGIONAL-CMD-{number:03d}" in audit.results


def test_cmd007_rejects_consistently_wrong_terminal_result(
    audit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        audit, "complete", lambda *a, **k: (200, {"status": "FAILED", "error": "wrong"})
    )
    with pytest.raises(probes.ProtocolAuditError, match="terminal|SUCCEEDED"):
        audit.run_007()


def test_cmd008_rejects_record_drift_even_when_the_result_is_refused(
    audit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = audit.complete

    def reject_with_drift(
        command_id: str, *, payload: dict[str, Any]
    ) -> tuple[int, Any]:
        response: tuple[int, Any] = complete(command_id, payload=payload)
        command = audit.store.get_remote_command(command_id)
        changed = command.model_copy(update={"result_details": {"unexpected": True}})
        monkeypatch.setattr(audit.store, "get_remote_command", lambda _: changed)
        return response

    monkeypatch.setattr(audit, "complete", reject_with_drift)
    with pytest.raises(
        probes.ProtocolAuditError, match="PENDING changed the command record"
    ):
        audit.run_008()
    assert "GF-REGIONAL-CMD-008" not in audit.results, (
        "record drift must not produce successful CMD-008 evidence"
    )


@pytest.mark.parametrize(
    ("status", "commands", "count"),
    [
        (201, [{"command_id": "owned"}], 1),
        (503, [{"command_id": "owned"}], 1),
        (200, [], 1),
        (200, [{"command_id": "foreign"}], 1),
        (200, [{"command_id": "owned"}, {"command_id": "owned"}], 2),
        (200, [None], 1),
    ],
)
def test_claim_proof_refuses_wrong_codes_missing_or_duplicate_ids(
    status: int, commands: list[Any], count: int
) -> None:
    with pytest.raises(probes.ProtocolAuditError):
        protocol.LiveProtocolAudit.claim_records(
            status, {"commands": commands}, {"owned"}, count=count
        )


def test_case_cleanup_removes_links_and_submission_records(audit: Any) -> None:
    summary = audit.run(
        ("GF-REGIONAL-CMD-001", "GF-REGIONAL-CMD-014", "GF-REGIONAL-CMD-016")
    )
    assert summary["verdict"] == "PASS"
    assert audit.store.list_remote_commands() == []
    assert audit.store.list_workflows() == []
    assert audit.store.get_incident_by_event(f"{audit.run_id}-cmd014-event") is None
    with pytest.raises(NotFoundError):
        audit.store.get_hyperpod_submission("hp-cluster-a", f"{audit.run_id}-cmd016")


def test_seed_commit_ack_loss_still_cleans_and_stops_next_case(
    audit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    save = audit.store.save_incident_and_workflow

    def lost_ack(*args: Any, **kwargs: Any) -> Any:
        save(*args, **kwargs)
        raise RuntimeError("seed committed but acknowledgement lost")

    monkeypatch.setattr(audit.store, "save_incident_and_workflow", lost_ack)
    summary = audit.run(("GF-REGIONAL-CMD-014", "GF-REGIONAL-CMD-015"))
    assert summary["verdict"] == "FAIL"
    assert summary["not_run"] == ["GF-REGIONAL-CMD-015"]
    assert "cleanup_error" not in summary["results"]["GF-REGIONAL-CMD-014"]
    assert audit.store.list_workflows() == []
    assert audit.store.get_incident_by_event(f"{audit.run_id}-cmd014-event") is None


def test_submission_seed_registers_all_cleanup_keys_before_reservation(
    audit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve = audit.store.reserve_hyperpod_submission
    cluster_names = ("hp-cluster-a", "hp-cluster-b")
    suffixes = ("", "-missing", "-unreserved")
    key = f"{audit.run_id}-cmd016"

    def lost_ack(record: Any) -> Any:
        assert len(audit.created_auxiliary) == 6, (
            "all possible submission keys must be tracked before the first reserve"
        )
        for name in cluster_names:
            for suffix in suffixes:
                reserve(
                    record.model_copy(
                        update={"cluster_name": name, "idempotency_key": key + suffix}
                    )
                )
        raise RuntimeError("submission committed but acknowledgement lost")

    monkeypatch.setattr(audit.store, "reserve_hyperpod_submission", lost_ack)
    summary = audit.run(("GF-REGIONAL-CMD-016",))
    result = summary["results"]["GF-REGIONAL-CMD-016"]
    assert summary["verdict"] == "FAIL", "lost acknowledgement must fail the audit"
    assert "submission committed but acknowledgement lost" in result["error"], (
        "the audit must retain the reservation failure"
    )
    assert "cleanup_error" not in result, "all registered submission keys must clean"
    assert not audit.created_auxiliary, "successful cleanup must clear tracked keys"
    for name in cluster_names:
        for suffix in suffixes:
            with pytest.raises(NotFoundError):
                audit.store.get_hyperpod_submission(name, key + suffix)


def test_bundled_probe_loads_and_runs_submission_checks_without_a_checkout(
    audit: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = protocol.probe_source()
    original_import = builtins.__import__

    def import_without_checkout(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scripts" or name.startswith("scripts."):
            raise ImportError("no checkout in the Pod")
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(
        sys.modules, "command_protocol_probes", ModuleType("placeholder")
    )
    monkeypatch.setattr(builtins, "__import__", import_without_checkout)
    namespace: dict[str, Any] = {"__name__": "bundled_protocol", "__package__": None}
    exec(compile(source, "<protocol-bundle>", "exec"), namespace)
    assert namespace["LiveProtocolAudit"].claim_records(
        200, {"commands": [{"command_id": "owned"}]}, {"owned"}
    ) == [{"command_id": "owned"}]
    namespace["LiveProtocolAudit"].run_008(audit)
    assert audit.results["GF-REGIONAL-CMD-008"] == {
        "invalid_statuses": {
            "PENDING": 422,
            "LEASED": 422,
            "FAILED_NO_ERROR": 422,
            "FAILED_EMPTY_ERROR": 422,
            "MISSING_TOKEN": 422,
            "EXTRA_FIELD": 422,
        },
        "waiting_status": 200,
    }, "the standalone bundle must retain the complete CMD-008 result checks"
    namespace["run_hyperpod_submission_case"](audit)
    assert audit.results["GF-REGIONAL-CMD-016"]["anonymous"] == {
        "POST /v1/regional/executors/hyperpod-submissions/reserve": 401,
        "GET /v1/regional/executors/hyperpod-submissions": 401,
        "POST /v1/regional/executors/hyperpod-submissions/outcome": 401,
    }
    audit.close()
    with closing(SqliteStore(str(tmp_path / "audit.db"))) as observer:
        with pytest.raises(NotFoundError):
            observer.get_hyperpod_submission("hp-cluster-a", f"{audit.run_id}-cmd016")


def test_standalone_bundle_output_never_opens_the_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: pytest.fail("bundle emission must not access a Store or live identity"),
    )
    monkeypatch.setattr(
        protocol,
        "open_credential_audit_store",
        lambda: pytest.fail("bundle emission must not open the audit Store"),
    )
    monkeypatch.setattr(sys, "argv", ["command-audit", "--emit-probe"])
    assert protocol.main() == 0
    source = capsys.readouterr().out
    monkeypatch.setitem(
        sys.modules, "command_protocol_probes", ModuleType("placeholder")
    )
    namespace: dict[str, Any] = {"__name__": "emitted_probe", "__package__": None}
    exec(compile(source, "<emitted-probe>", "exec"), namespace)
    assert namespace["parser"]().parse_args(["--emit-probe"]).emit_probe is True
    assert (
        namespace["LiveProtocolAudit"].claim_records(200, {"commands": []}, set()) == []
    )


@pytest.mark.parametrize("status", [200, 409])
def test_extracted_transport_preserves_http_status_body_and_scope(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    sent = []

    class Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        @property
        def status(self) -> int:
            return status

        def read(self) -> bytes:
            return b'{"observed":true}'

    def urlopen(request: Any, *, timeout: int) -> Response:
        sent.append(request)
        assert timeout == 20
        if status == 409:
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "conflict",
                Message(),
                io.BytesIO(b'{"observed":true}'),
            )
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    result = probes.request_json(
        "POST",
        "/v1/regional/executors/claim",
        cluster_id="cluster-a",
        token="test-credential",
        payload={"max_commands": 1},
    )
    assert result == (status, {"observed": True})
    assert len(sent) == 1
    assert sent[0].full_url == "http://127.0.0.1:8080/v1/regional/executors/claim"
    assert sent[0].get_header("X-gpu-fault-cluster-id") == "cluster-a"
    assert sent[0].get_header("Authorization") == "Bearer test-credential"
    assert json.loads(sent[0].data) == {"max_commands": 1}
