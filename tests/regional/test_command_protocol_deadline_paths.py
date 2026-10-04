"""Refusal and stop paths of the live CMD audit's CPU deadline.

``AuditDeadline`` is the only thing standing between a hung audit and a
control-plane queue it still holds leases on. These tests run it in-process --
the real pidfd watchdog, the real signal handlers, restored afterwards -- and
pin the paths the subprocess-based tests cannot measure: invalid budgets, a
credential that does not outlive the hard stop, the cooperative and signalled
stops, an exhausted cleanup budget, and the two pre-flight refusals of
``__enter__``. The driver around it (``run_with_deadline`` and ``main``) is
exercised with a scripted audit so the summary it prints after an abort, a
failed owned cleanup or a refused argument set is what an operator will see.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.regional import regional_registry_content_sha256
from scripts.e2e.regional import audit_regional_command_protocol_live as protocol
from scripts.e2e.regional import command_audit_summary
from scripts.e2e.regional.command_protocol_probes import ProtocolAuditError
from tests.regional.test_command_protocol_audit_contracts import FakeStore, bare_audit
from tests.regional.test_command_protocol_secure_input import (
    credential_input as credential_input_fixture,  # noqa: F401  (pytest fixture)
)
from tests.regional.test_command_protocol_secure_input import decode


def credential_input(request: pytest.FixtureRequest) -> tuple[dict[str, Any], Any]:
    """The secure-input suite's valid envelope and registry store."""

    document, store = request.getfixturevalue("credential_input_fixture")
    return document, store


AuditDeadline = protocol.AuditDeadline
AuditStopped = protocol.AuditStopped


def envelope(expires_at: str) -> Any:
    return argparse.Namespace(expires_at=expires_at)


@pytest.mark.parametrize(
    "overall, cleanup",
    [
        (0, 30),
        (-1.0, 30),
        (901, 30),
        (300, 0),
        (300, 121),
        (float("inf"), 30),
        (300, float("nan")),
        (True, 30),
        ("300", 30),
    ],
)
def test_deadline_refuses_budgets_outside_the_cpu_envelope(
    overall: Any, cleanup: Any
) -> None:
    with pytest.raises(ProtocolAuditError, match="invalid CPU audit deadline"):
        AuditDeadline(overall, cleanup)


def test_credential_lifetime_must_outlive_the_hard_stop() -> None:
    deadline = AuditDeadline(60, 10)
    far = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    deadline.require_credential_lifetime(envelope(far))

    short = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    with pytest.raises(ProtocolAuditError, match="cover CPU work and cleanup"):
        deadline.require_credential_lifetime(envelope(short))

    naive = (datetime.now() + timedelta(hours=1)).isoformat()
    with pytest.raises(ProtocolAuditError, match="cover CPU work and cleanup"):
        deadline.require_credential_lifetime(envelope(naive))

    with pytest.raises(ProtocolAuditError, match="cover CPU work and cleanup"):
        deadline.require_credential_lifetime(envelope("not-a-timestamp"))


def test_cooperative_stop_only_marks_while_a_signal_aborts() -> None:
    deadline = AuditDeadline(60, 10)

    deadline.stop()
    assert deadline.stopped is True

    fresh = AuditDeadline(60, 10)
    with pytest.raises(AuditStopped, match="CPU audit stopped"):
        fresh.stop(signal.SIGTERM, None)
    assert fresh.stopped is True

    cleaning = AuditDeadline(60, 10)
    cleaning.cleaning = True
    cleaning.stop(signal.SIGTERM, None)  # a signal during cleanup does not raise
    assert cleaning.stopped is True


def test_check_work_raises_once_stopped_or_past_the_work_deadline() -> None:
    deadline = AuditDeadline(60, 10)
    deadline.check_work()

    deadline.stopped = True
    with pytest.raises(AuditStopped, match="deadline or supervision lost"):
        deadline.check_work()

    expired = AuditDeadline(60, 10)
    expired.work_until = 0.0
    with pytest.raises(AuditStopped):
        expired.check_work()
    assert expired.stopped is True


def test_cleanup_budget_is_refused_when_exhausted_or_reentered() -> None:
    deadline = AuditDeadline(60, 10)
    deadline.cleanup_remaining = 0
    with pytest.raises(AuditStopped, match="cleanup budget expired"):
        with deadline.cleanup():
            pass
    assert deadline.stopped is True

    nested = AuditDeadline(60, 10)
    with nested.cleanup():
        with pytest.raises(AuditStopped, match="cleanup budget expired"):
            with nested.cleanup():
                pass
    assert nested.cleaning is False, "the outer cleanup still releases the flag"


def test_cleanup_that_overruns_its_remaining_budget_stops_the_audit() -> None:
    deadline = AuditDeadline(60, 10)
    deadline.cleanup_remaining = 1e-9

    with pytest.raises(AuditStopped, match="cleanup budget expired"):
        with deadline.cleanup():
            pass

    assert deadline.stopped is True
    assert deadline.cleaning is False
    assert deadline.cleanup_remaining == 0


def test_enter_requires_pidfd_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "pidfd_open")

    with pytest.raises(ProtocolAuditError, match="requires Linux pidfd supervision"):
        with AuditDeadline(5, 1):
            pass


def test_enter_refuses_to_replace_an_armed_interval_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signal, "getitimer", lambda _which: (12.5, 0.0))

    with pytest.raises(ProtocolAuditError, match="cannot replace an existing timer"):
        with AuditDeadline(5, 1):
            pass


def test_enter_arms_a_watchdog_and_exit_restores_the_handlers() -> None:
    before = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGUSR1, signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    }

    with AuditDeadline(30, 5) as deadline:
        assert deadline.watchdog is not None
        assert deadline.watchdog.poll() is None, "the watchdog must be alive"
        assert signal.getsignal(signal.SIGUSR1) == deadline.stop
        deadline.check_work()

    assert deadline.stopped is True
    assert deadline.watchdog.poll() == 0, "a dismissed watchdog exits cleanly"
    assert {signum: signal.getsignal(signum) for signum in before} == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_cleanup_inside_a_supervised_audit_arms_and_disarms_the_alarm() -> None:
    with AuditDeadline(30, 5) as deadline:
        with deadline.cleanup():
            armed, _interval = signal.getitimer(signal.ITIMER_REAL)
            assert 0 < armed <= 5, "cleanup is bounded by a native alarm"
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert deadline.cleanup_remaining < 5

    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_a_watchdog_that_exited_is_a_lost_supervision() -> None:
    deadline = AuditDeadline(30, 5)
    deadline.watchdog = argparse.Namespace(poll=lambda: 0)  # type: ignore[assignment]

    with pytest.raises(AuditStopped, match="supervision lost"):
        deadline.check_work()
    assert deadline.stopped is True


def test_exit_kills_a_watchdog_that_ignores_dismissal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    deadline = AuditDeadline(30, 5)
    deadline.__enter__()
    watchdog = deadline.watchdog
    assert watchdog is not None
    original_wait = watchdog.wait
    attempts: list[float | None] = []

    def wait(timeout: float | None = None) -> int:
        attempts.append(timeout)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired(cmd="watchdog", timeout=timeout or 0)
        return original_wait(timeout=timeout)

    monkeypatch.setattr(watchdog, "wait", wait)

    deadline.__exit__(None, None, None)

    assert attempts == [1, 1]
    assert watchdog.returncode is not None and watchdog.returncode != 0, (
        "the second wait follows a kill"
    )


# --------------------------------------------------------------------------- #
# audit object: close with a deadline, preflight on an unknown queue
# --------------------------------------------------------------------------- #
class ClosingStore(FakeStore):
    def __init__(self, stats: Any = None) -> None:
        super().__init__()
        self.closed = 0
        self.stats = stats

    def close(self) -> None:
        self.closed += 1

    def remote_command_stats(self) -> Any:
        return self.stats if self.stats is not None else super().remote_command_stats()


def test_close_stops_the_deadline_before_releasing_the_store() -> None:
    store = ClosingStore()
    audit = bare_audit(store=store)
    deadline = AuditDeadline(60, 10)
    audit.deadline = deadline

    audit.close()

    assert deadline.stopped is True
    assert store.closed == 1
    assert deadline.cleanup_remaining < 10, "the close ran inside the cleanup budget"


def test_preflight_refuses_a_queue_whose_state_is_unknown() -> None:
    audit = bare_audit(store=ClosingStore(stats={"open_by_cluster": None}))

    with pytest.raises(ProtocolAuditError, match="queue state is unknown"):
        audit.open_commands_in_cluster()


def test_case_evidence_document_lifts_the_error_and_release() -> None:
    document = protocol.case_evidence_document(
        case_id="GF-REGIONAL-CMD-003",
        details={"verdict": "FAIL", "error": "lease was not handed back", "seen": 2},
        run_id="cmd-audit-test",
        cluster_id="cluster-a",
        other_cluster_id="cluster-b",
        preflight={"open_commands": 0},
        release_id="release-7",
    )

    assert document["verdict"] == "FAIL"
    assert document["error"] == "lease was not handed back"
    assert document["release_id"] == "release-7"
    assert document["details"] == {"error": "lease was not handed back", "seen": 2}


# --------------------------------------------------------------------------- #
# run_with_deadline / main
# --------------------------------------------------------------------------- #
def arguments(**overrides: Any) -> argparse.Namespace:
    value = argparse.Namespace(
        case=["GF-REGIONAL-CMD-001"],
        cleanup_seconds=None,
        overall_seconds=30,
        credentials_stdin=False,
        synthetic_run_id="",
        cluster_id="cluster-a",
        other_cluster_id="cluster-b",
        executor_sha256="s" * 64,
        executor_digest="d" * 64,
        run_dir=None,
        isolated_cluster=True,
        executor_ready_replicas=None,
        release_id="",
    )
    for key, item in overrides.items():
        setattr(value, key, item)
    return value


def test_run_with_deadline_defaults_the_cleanup_budget_for_a_single_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class ScriptedAudit:
        def __init__(self, **kwargs: Any) -> None:
            seen["kwargs"] = kwargs
            self.closed = False
            seen["audit"] = self

        def run(self, cases: tuple[str, ...]) -> dict[str, Any]:
            seen["cases"] = cases
            return {"verdict": "PASS", "verdicts": {case: "PASS" for case in cases}}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(protocol, "LiveProtocolAudit", ScriptedAudit)

    summary = protocol.run_with_deadline(arguments())

    assert summary["verdict"] == "PASS"
    assert seen["cases"] == ("GF-REGIONAL-CMD-001",)
    assert seen["kwargs"]["credential_envelope"] is None
    assert (
        seen["kwargs"]["deadline"].hard_until - seen["kwargs"]["deadline"].work_until
        == 30
    )
    assert seen["audit"].closed is True


def test_run_with_deadline_requires_a_cumulative_budget_for_several_cases() -> None:
    with pytest.raises(ProtocolAuditError, match="cumulative --cleanup-seconds"):
        protocol.run_with_deadline(
            arguments(case=["GF-REGIONAL-CMD-001", "GF-REGIONAL-CMD-002"])
        )


def test_run_with_deadline_reports_an_aborted_audit_and_a_failed_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAudit:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def run(self, cases: tuple[str, ...]) -> dict[str, Any]:
            raise RuntimeError("store went away")

        def close(self) -> None:
            raise AuditStopped("cleanup budget exhausted")

    monkeypatch.setattr(protocol, "LiveProtocolAudit", FailingAudit)

    summary = protocol.run_with_deadline(arguments(cleanup_seconds=5))

    assert summary == {
        "verdict": "FAIL",
        "not_run": ["GF-REGIONAL-CMD-001"],
        "error": "audit aborted: RuntimeError",
        "cleanup_error": "owned cleanup: AuditStopped",
        "recovery_required": True,
    }
    assert "store went away" not in json.dumps(summary), "no exception text leaks"


def test_run_with_deadline_without_an_audit_object_has_nothing_to_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**kwargs: Any) -> Any:
        raise ProtocolAuditError("durable credential scope rejected")

    monkeypatch.setattr(protocol, "LiveProtocolAudit", refuse)

    summary = protocol.run_with_deadline(arguments(cleanup_seconds=5))

    assert summary == {
        "verdict": "FAIL",
        "not_run": ["GF-REGIONAL-CMD-001"],
        "error": "audit aborted: ProtocolAuditError",
    }


def _argv(monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(
        sys, "argv", ["audit_regional_command_protocol_live.py", *extra]
    )


def test_main_emits_the_probe_source_without_touching_the_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _argv(monkeypatch, "--emit-probe")
    monkeypatch.setattr(protocol, "probe_source", lambda: "print('probe')\n")

    assert protocol.main() == 0

    assert capsys.readouterr().out == "print('probe')\n"


def test_main_refuses_an_incomplete_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _argv(
        monkeypatch,
        "--cluster-id",
        "cluster-a",
        "--other-cluster-id",
        "cluster-b",
        "--executor-sha256",
        "s" * 64,
    )

    with pytest.raises(SystemExit, match="--executor-digest is required"):
        protocol.main()


def test_main_refuses_to_run_without_an_explicit_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _argv(
        monkeypatch,
        "--cluster-id",
        "cluster-a",
        "--other-cluster-id",
        "cluster-b",
        "--executor-sha256",
        "s" * 64,
        "--executor-digest",
        "d" * 64,
    )

    with pytest.raises(SystemExit, match="explicit --case is required"):
        protocol.main()


def test_main_requires_credentials_stdin_and_synthetic_run_id_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _argv(
        monkeypatch,
        "--cluster-id",
        "cluster-a",
        "--other-cluster-id",
        "cluster-b",
        "--executor-sha256",
        "s" * 64,
        "--executor-digest",
        "d" * 64,
        "--case",
        "GF-REGIONAL-CMD-001",
        "--synthetic-run-id",
        "run-1",
    )

    with pytest.raises(SystemExit, match="--credentials-stdin requires"):
        protocol.main()


def test_main_prints_a_refusal_summary_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _argv(
        monkeypatch,
        "--cluster-id",
        "cluster-a",
        "--other-cluster-id",
        "cluster-b",
        "--executor-sha256",
        "s" * 64,
        "--executor-digest",
        "d" * 64,
        "--case",
        "GF-REGIONAL-CMD-001",
    )

    def refuse(_arguments: argparse.Namespace) -> dict[str, Any]:
        raise ProtocolAuditError("CPU audit requires Linux pidfd supervision")

    monkeypatch.setattr(protocol, "run_with_deadline", refuse)

    assert protocol.main() == 1

    assert json.loads(capsys.readouterr().out) == {
        "verdict": "FAIL",
        "error": "audit refused: ProtocolAuditError",
    }


# --------------------------------------------------------------------------- #
# __exit__ without a live watchdog, and a watchdog that never arms
# --------------------------------------------------------------------------- #
def test_exit_without_enter_only_stops_and_clears_the_timer() -> None:
    deadline = AuditDeadline(30, 5)

    deadline.__exit__(None, None, None)

    assert deadline.stopped is True
    assert deadline.watchdog is None
    assert deadline.handlers == {}
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_exit_tolerates_a_watchdog_without_pipes() -> None:
    deadline = AuditDeadline(30, 5)
    waited: list[float | None] = []
    deadline.watchdog = argparse.Namespace(  # type: ignore[assignment]
        stdin=None, stdout=None, wait=lambda timeout=None: waited.append(timeout)
    )

    deadline.__exit__(None, None, None)

    assert waited == [1], "the child is still reaped once"
    assert deadline.stopped is True


def test_a_watchdog_that_does_not_arm_is_refused_and_dismissed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(protocol.select, "select", lambda *_a, **_k: ([], [], []))
    deadline = AuditDeadline(30, 5)

    with pytest.raises(ProtocolAuditError, match="watchdog did not arm"):
        deadline.__enter__()

    assert deadline.watchdog is not None
    assert deadline.watchdog.poll() is not None, "__exit__ dismissed the child"
    assert signal.getsignal(signal.SIGUSR1) != deadline.stop, "handlers restored"


# --------------------------------------------------------------------------- #
# credential input and store opening refusals
# --------------------------------------------------------------------------- #
def test_group_readable_stdin_is_an_unprotected_credential_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "credentials.json"
    source.write_text('{"not": "read"}', encoding="utf-8")
    source.chmod(0o644)
    monkeypatch.setattr(sys, "argv", ["audit_regional_command_protocol_live.py"])
    reads: list[int] = []
    with source.open("rb") as handle:

        class Stdin:
            buffer = argparse.Namespace(read=lambda n: reads.append(n) or b"{}")

            @staticmethod
            def fileno() -> int:
                return handle.fileno()

            @staticmethod
            def isatty() -> bool:
                return False

        monkeypatch.setattr(sys, "stdin", Stdin())
        with pytest.raises(ProtocolAuditError, match="protected credential stdin"):
            protocol.read_credentials(
                cluster_id="perf-cap-000",
                other_cluster_id="perf-cap-001",
                synthetic_run_id="unit-capacity-run",
            )
    assert reads == [], "nothing is read from an unprotected channel"


def test_open_credential_audit_store_requires_a_postgres_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "")
    with pytest.raises(ProtocolAuditError, match="Store initialization rejected"):
        protocol.open_credential_audit_store()

    monkeypatch.setenv("GPU_FAULT_STORE_URL", f"sqlite:///{tmp_path / 'state.db'}")
    with pytest.raises(ProtocolAuditError, match="Store initialization rejected"):
        protocol.open_credential_audit_store()
    assert not (tmp_path / "state.db").exists(), "no SQLite store is ever created"


class CountingStore:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.head_reads = 0

    def get_regional_registry_head(self) -> Any:
        self.head_reads += 1
        return self.inner.get_regional_registry_head()

    def get_regional_registry_revision(self, generation: int) -> Any:
        return self.inner.get_regional_registry_revision(generation)

    def list_agents(self, cluster_id: str) -> Any:
        return self.inner.list_agents(cluster_id)


def test_an_already_expired_envelope_is_refused_before_any_store_read(
    request: pytest.FixtureRequest,
) -> None:
    document, inner = credential_input(request)
    store = CountingStore(inner)
    envelope = decode(document)
    for stale in (
        (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        datetime.now().isoformat(),  # naive
    ):
        expired = envelope.model_copy(update={"expires_at": stale})
        with pytest.raises(ProtocolAuditError, match="durable credential scope"):
            protocol.current_credential_registry(store, expired)
    assert store.head_reads == 0


def test_a_revision_listing_one_cluster_twice_is_refused(
    request: pytest.FixtureRequest,
) -> None:
    document, inner = credential_input(request)
    revision = inner.get_regional_registry_revision(2)
    first = revision.registrations[0]
    duplicated = argparse.Namespace(
        generation=revision.generation,
        registrations=[first, first],
        content_sha256=regional_registry_content_sha256([first, first]),
    )
    head = inner.get_regional_registry_head().model_copy(
        update={"content_sha256": duplicated.content_sha256}
    )
    document["registry_content_sha256"] = head.content_sha256
    inner.get_regional_registry_head = lambda: head
    inner.get_regional_registry_revision = lambda generation: duplicated
    store = CountingStore(inner)

    with pytest.raises(ProtocolAuditError, match="durable credential scope"):
        protocol.current_credential_registry(store, decode(document))
    assert store.head_reads == 1, "refused after the first head read"


def test_credentials_for_only_one_of_the_two_clusters_are_refused(
    request: pytest.FixtureRequest,
) -> None:
    document, inner = credential_input(request)
    store = CountingStore(inner)
    envelope = decode(document)
    partial = envelope.model_copy(update={"credentials": envelope.credentials[:1]})

    with pytest.raises(ProtocolAuditError, match="durable credential scope"):
        protocol.current_credential_registry(store, partial)
    assert store.head_reads == 1, "selection is checked before the head recheck"


def test_case_evidence_document_without_release_or_error_stays_minimal() -> None:
    document = protocol.case_evidence_document(
        case_id="GF-REGIONAL-CMD-001",
        details={"verdict": "PASS", "leases": 3},
        run_id="cmd-audit-test",
        cluster_id="cluster-a",
        other_cluster_id="cluster-b",
        preflight={},
    )

    assert "release_id" not in document and "error" not in document
    assert document["details"] == {"leases": 3}


def test_main_write_evidence_delegates_to_the_summary_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    _argv(monkeypatch, "--write-evidence", str(summary))
    seen: list[Any] = []

    def write_evidence_main(arguments: argparse.Namespace, writer: Any) -> int:
        seen.append((arguments.write_evidence, writer))
        return 3

    monkeypatch.setattr(
        command_audit_summary, "write_evidence_main", write_evidence_main
    )

    assert protocol.main() == 3

    assert seen == [(summary, protocol.write_evidence_from_summary)]
