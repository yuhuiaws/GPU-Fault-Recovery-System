"""Contracts of the live CMD protocol audit that hold without a cluster.

The audit itself needs an API Pod; what CI can own is the harness around the
cases -- that one failing case is recorded and the rest still run, that the
preflight refuses a queue a production executor could drain, that every lease
the boundary sweep takes is handed back, that a case's verdict comes from the
observed answers -- and the evidence the CMD-011 local guard fixture writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_executor_local_guards as local_guards
from scripts.e2e.regional import audit_regional_command_protocol_live as audit_module
from scripts.e2e.regional.audit_regional_command_protocol_live import (
    AUDITED_CASE_IDS,
    LiveProtocolAudit,
    ProtocolAuditError,
    write_evidence_from_summary,
)


class FakeStore:
    def __init__(self, open_by_cluster: dict[str, int] | None = None) -> None:
        self.open_by_cluster = open_by_cluster or {}
        self.deleted: list[tuple[str, str]] = []

    def remote_command_stats(self) -> dict[str, Any]:
        return {"open_by_cluster": dict(self.open_by_cluster)}

    def _delete(self, kind: str, key: str) -> None:
        self.deleted.append((kind, key))


def bare_audit(
    *,
    store: FakeStore | None = None,
    run_dir: Path | None = None,
    isolated_cluster: bool = True,
    executor_ready_replicas: int | None = None,
) -> LiveProtocolAudit:
    """An audit object without ``__init__``: no ApplicationContext, no env."""

    audit = LiveProtocolAudit.__new__(LiveProtocolAudit)
    audit.cluster_id = "cluster-a"
    audit.other_cluster_id = "cluster-b"
    audit.executor_sha256 = "s" * 64
    audit.executor_digest = "d" * 64
    audit.run_dir = run_dir
    audit.isolated_cluster = isolated_cluster
    audit.executor_ready_replicas = executor_ready_replicas
    audit.release_id = "release-1"
    audit.run_id = "cmd-audit-test"
    audit.test_owner = "cmd-audit-test-owner"
    audit.store = store or FakeStore()
    audit.tokens = {"cluster-a": "a" * 32, "cluster-b": "b" * 32}
    audit.registry = {}
    audit.created_commands = set()
    audit.created_workflows = set()
    audit.created_incidents = set()
    audit.results = {}
    audit.preflight_result = {}
    return audit


def test_a_failing_case_is_recorded_and_the_remaining_cases_still_run(
    tmp_path: Path,
) -> None:
    store = FakeStore()
    audit = bare_audit(store=store, run_dir=tmp_path)
    order: list[str] = []

    def passing(case_id: str):
        def method() -> None:
            order.append(case_id)
            audit.created_commands.add(f"{case_id}-command")
            audit.record(case_id, observed=True)

        return method

    def failing() -> None:
        order.append("GF-REGIONAL-CMD-002")
        audit.created_commands.add("cmd002-command")
        raise ProtocolAuditError("lease_seconds=9 answered 200")

    for case_id in AUDITED_CASE_IDS:
        setattr(audit, f"run_{case_id.rsplit('-', 1)[1]}", passing(case_id))
    audit.run_002 = failing  # type: ignore[method-assign]

    summary = audit.run()

    assert order == list(AUDITED_CASE_IDS), "a failing case aborted the rest"
    assert summary["verdict"] == "FAIL"
    assert summary["verdicts"]["GF-REGIONAL-CMD-002"] == "FAIL"
    assert summary["verdicts"]["GF-REGIONAL-CMD-003"] == "PASS"
    assert (
        "lease_seconds=9 answered 200"
        in summary["results"]["GF-REGIONAL-CMD-002"]["error"]
    )
    # The failing case's seeded object was still cleaned up.
    assert ("remote_command", "cmd002-command") in store.deleted
    assert not audit.created_commands, "a case left seeded commands behind"
    failed = json.loads(
        (tmp_path / "cases/GF-REGIONAL-CMD-002/GF-REGIONAL-CMD-002.json").read_text()
    )
    passed = json.loads(
        (tmp_path / "cases/GF-REGIONAL-CMD-016/GF-REGIONAL-CMD-016.json").read_text()
    )
    assert failed["verdict"] == "FAIL" and "lease_seconds" in failed["error"]
    assert passed["verdict"] == "PASS"
    assert passed["case_id"] == "GF-REGIONAL-CMD-016"
    assert passed["cluster_id"] == "cluster-a"
    assert passed["release_id"] == "release-1"
    assert passed["details"] == {"observed": True}


def test_preflight_refuses_an_open_queue_and_a_live_executor() -> None:
    busy = bare_audit(store=FakeStore({"cluster-a": 2}), isolated_cluster=True)
    with pytest.raises(ProtocolAuditError, match="2 open remote command"):
        busy.preflight()

    live = bare_audit(isolated_cluster=False, executor_ready_replicas=2)
    with pytest.raises(ProtocolAuditError, match="production executor may still"):
        live.preflight()

    unknown = bare_audit(isolated_cluster=False, executor_ready_replicas=None)
    with pytest.raises(ProtocolAuditError, match="executor-ready-replicas 0"):
        unknown.preflight()

    scaled_down = bare_audit(isolated_cluster=False, executor_ready_replicas=0)
    assert scaled_down.preflight()["errors"] == []
    isolated = bare_audit(isolated_cluster=True)
    assert isolated.preflight()["open_remote_commands"] == 0


def test_run_refuses_to_start_without_a_passing_preflight() -> None:
    audit = bare_audit(store=FakeStore({"cluster-a": 1}))

    with pytest.raises(ProtocolAuditError, match="preflight refused"):
        audit.run()

    assert audit.results == {}


def test_cmd001_hands_back_every_lease_as_waiting_and_seeds_only_its_owner() -> None:
    audit = bare_audit()
    seeded_owners: list[str | None] = []
    returned: list[dict[str, Any]] = []
    backlog = [f"cmd-audit-test-cmd001-{index}-command" for index in range(3)]

    def seed(suffix: str, **kwargs: Any) -> SimpleNamespace:
        seeded_owners.append(kwargs.get("owner"))
        return SimpleNamespace(command_id=f"cmd-audit-test-{suffix}-command")

    def claim(**kwargs: Any) -> tuple[int, Any]:
        value = kwargs["max_commands"]
        if value in {1, 25, "5"}:
            count = min(len(backlog), int(value))
            return 200, {
                "commands": [
                    {"command_id": command_id, "lease_token": f"lease-{command_id}"}
                    for command_id in backlog[:count]
                ]
            }
        return 422, {"detail": "invalid"}

    def complete(command_id: str, *, payload: dict[str, Any]) -> tuple[int, Any]:
        returned.append({"command_id": command_id, **payload})
        return 200, {"status": payload["status"]}

    audit.seed = seed  # type: ignore[method-assign]
    audit.claim = claim  # type: ignore[method-assign]
    audit.complete = complete  # type: ignore[method-assign]

    audit.run_001()

    assert seeded_owners == [None, None, None], (
        "CMD-001 must seed with the audit's test-only owner, not a real adapter"
    )
    # 1 -> 1 lease, 25 -> 3 leases, "5" -> 3 leases; every one came back WAITING.
    assert len(returned) == 7
    assert all(item["status"] == "WAITING" for item in returned), (
        "a leased command must be handed back as WAITING, never completed"
    )
    assert {item["command_id"] for item in returned} == set(backlog)
    recorded = audit.results["GF-REGIONAL-CMD-001"]
    assert recorded["leased_and_returned"] == 7
    assert recorded["statuses"] == {
        "0": 422,
        "1": 200,
        "25": 200,
        "26": 422,
        "-1": 422,
        "'5'": 200,
        "5.5": 422,
    }


def test_cmd001_refuses_a_lease_on_a_command_it_did_not_seed() -> None:
    audit = bare_audit()
    audit.seed = lambda suffix, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        command_id=f"cmd-audit-test-{suffix}-command"
    )
    audit.claim = lambda **kwargs: (  # type: ignore[method-assign]
        (200, {"commands": [{"command_id": "remote-production", "lease_token": "x"}]})
        if kwargs["max_commands"] == 1
        else (422, {})
    )
    audit.complete = lambda *args, **kwargs: (200, {})  # type: ignore[method-assign]

    with pytest.raises(ProtocolAuditError, match="did not seed"):
        audit.run_001()


def test_cmd009_requires_the_unknown_and_foreign_404_to_be_indistinguishable() -> None:
    audit = bare_audit()
    answers = iter(
        [
            (404, {"detail": "remote command not found"}),
            (404, {"detail": "command belongs to another cluster"}),
        ]
    )
    audit.seed = lambda suffix, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        command_id="cmd-audit-test-cmd009-other-command"
    )
    audit.complete = lambda *args, **kwargs: next(answers)  # type: ignore[method-assign]

    with pytest.raises(ProtocolAuditError, match="distinguishable"):
        audit.run_009()


def test_cmd007_records_the_replay_statuses_it_observed_and_fails_on_a_rewrite() -> (
    None
):
    first = {"status": "SUCCEEDED", "last_lease_owner": "cmd007"}

    def make_audit(replays: list[tuple[int, Any]]) -> LiveProtocolAudit:
        audit = bare_audit()
        answers = iter([(200, first), *replays])
        audit.seed = lambda suffix, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
            command_id="cmd-audit-test-cmd007-command"
        )
        audit.claim = lambda **kwargs: (  # type: ignore[method-assign]
            200,
            {"commands": [{"command_id": "c", "lease_token": "t"}]},
        )
        audit.complete = lambda *args, **kwargs: next(answers)  # type: ignore[method-assign]
        return audit

    clean = make_audit([(200, first), (200, first), (200, first)])
    clean.run_007()
    assert clean.results["GF-REGIONAL-CMD-007"]["replay_statuses"] == [200, 200, 200]

    rewritten = make_audit([(200, first), (200, {**first, "status": "FAILED"})])
    with pytest.raises(ProtocolAuditError, match="changed record"):
        rewritten.run_007()


def test_claim_defaults_to_the_test_only_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    audit = bare_audit()
    sent: list[dict[str, Any]] = []

    class Response:
        status = 200

        def read(self) -> bytes:
            return b'{"commands": []}'

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def urlopen(request: Any, timeout: float) -> Response:
        sent.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(audit_module.urllib.request, "urlopen", urlopen)

    audit.claim()
    audit.claim(owners=[])
    audit.claim(owners=["gpu-fault-node-agent"])

    assert [item["execution_owners"] for item in sent] == [
        ["cmd-audit-test-owner"],
        [],
        ["gpu-fault-node-agent"],
    ]


def test_write_evidence_from_summary_writes_one_file_per_recorded_case(
    tmp_path: Path,
) -> None:
    summary = {
        "run_id": "cmd-audit-abc",
        "cluster_id": "cluster-a",
        "other_cluster_id": "cluster-b",
        "verdict": "FAIL",
        "preflight": {"open_remote_commands": 0},
        "results": {
            "GF-REGIONAL-CMD-001": {"verdict": "PASS", "statuses": {"1": 200}},
            "GF-REGIONAL-CMD-005": {"verdict": "FAIL", "error": "order"},
        },
    }

    written = write_evidence_from_summary(summary, tmp_path, release_id="rel-9")

    assert [path.name for path in written] == [
        "GF-REGIONAL-CMD-001.json",
        "GF-REGIONAL-CMD-005.json",
    ]
    first = json.loads(written[0].read_text(encoding="utf-8"))
    assert first["verdict"] == "PASS"
    assert first["release_id"] == "rel-9"
    assert first["details"] == {"statuses": {"1": 200}}
    second = json.loads(written[1].read_text(encoding="utf-8"))
    assert second["verdict"] == "FAIL" and second["error"] == "order"
    with pytest.raises(ProtocolAuditError, match="no per-case results"):
        write_evidence_from_summary({"results": {}}, tmp_path)


def test_local_guard_fixture_writes_evidence_for_iso002_and_cmd011(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = local_guards.main(["--run-dir", str(tmp_path), "--release-id", "rel-1"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"GF-REGIONAL-ISO-002", "GF-REGIONAL-CMD-011"}
    for case_id in local_guards.CASE_IDS:
        document = json.loads(
            (tmp_path / "cases" / case_id / f"{case_id}.json").read_text("utf-8")
        )
        assert document["case_id"] == case_id
        assert document["verdict"] == "PASS"
        assert document["release_id"] == "rel-1"
        assert document["observed"] == payload[case_id]
        assert document["errors"] == []


def test_local_guard_fixture_reports_a_guard_that_stopped_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = SimpleNamespace(
        status=SimpleNamespace(value="SUCCEEDED"), status_source="adapter", error=None
    )
    executor = SimpleNamespace(_execute=lambda command: accepted, unexpected_failures=0)
    monkeypatch.setattr(local_guards, "_executor", lambda cluster_id: executor)

    payload, errors = local_guards.evaluate("cluster-a", "cluster-b")

    assert payload["GF-REGIONAL-CMD-011"]["status"] == "SUCCEEDED"
    assert errors["GF-REGIONAL-CMD-011"] and errors["GF-REGIONAL-ISO-002"]
    with pytest.raises(AssertionError):
        local_guards.run("cluster-a", "cluster-b")
