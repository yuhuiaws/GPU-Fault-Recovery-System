"""Execute the emitted producer and consume its stdout through the real wrapper."""

from __future__ import annotations

import copy
import io
import json
import os
import socket
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
    redirect_stdout,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast
from unittest.mock import patch

import pytest

from gpu_fault_release import regional_release_store_probe as probe
from gpu_fault_release import regional_release_store_proof as proof
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_runtime_identity import (
    BASE_RUNTIME_PATH,
    CONTROL_PLANE_PATH,
    CONTROL_PLANE_PYTHON,
)
from tests.regional._prerequisite_repair_support import repair_release

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

PRIVATE = "fixture-private-envelope-material"
FIELDS = (
    "run_id",
    "identity_sha256",
    "finished_at",
    "safe",
    "blockers",
    "database_state",
    "schema_version",
)


class EmptyDatabase:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.error: Exception | None = None

    def transaction(self) -> AbstractContextManager[None]:
        return nullcontext()

    def cursor(self) -> AbstractContextManager[EmptyDatabase]:
        return nullcontext(self)

    def execute(self, command: str, *_arguments: Any) -> None:
        assert command.strip().startswith(("SET ", "SELECT ")), (
            "the producer attempted a database mutation"
        )
        self.commands.append(command)
        if self.error is not None:
            raise self.error

    def fetchall(self) -> list[tuple[str, str, str]]:
        return []


class EnvelopeRoundTrip:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psycopg
        from psycopg.conninfo import make_conninfo

        self.release = repair_release(monkeypatch)
        self.original_run = self.release.runner.run
        self.make_conninfo = make_conninfo
        self.database = EmptyDatabase()
        self.connect_error: Exception | None = None
        self.close_error: Exception | None = None
        self.connections: list[dict[str, Any]] = []
        self.closed = 0
        self.producer_calls = 0
        self.producer_stdout = ""
        self.command: list[str] = []
        self.environment: dict[str, str] = {}
        self.identity: dict[str, Any] = {}
        self.dsn = ""
        self.records: list[dict[str, Any]] = []
        self.change_environment: Callable[[dict[str, str]], None] = lambda _: None
        self.corrupt_stdout: Callable[[dict[str, Any]], None] = lambda _: None
        monkeypatch.setattr(psycopg, "connect", self.connect)
        monkeypatch.setattr(psycopg.Connection, "connect", forbidden_io)
        monkeypatch.setattr(self.release.runner, "run", self.run)

    @contextmanager
    def connect(self, **arguments: Any) -> Iterator[EmptyDatabase]:
        self.connections.append(arguments)
        if self.connect_error is not None:
            raise self.connect_error
        try:
            yield self.database
        finally:
            self.closed += 1
        if self.close_error is not None:
            raise self.close_error

    def run(self, arguments: list[str], **options: Any) -> str:
        if not any(
            value.endswith("/wait-for-kubernetes-job.sh") for value in arguments
        ):
            return str(self.original_run(arguments, **options))
        job = self.release.runner.jobs[arguments[3]]
        container = job["spec"]["template"]["spec"]["containers"][0]
        command = container["command"]
        self.command = list(command)
        assert command[:3] == [CONTROL_PLANE_PYTHON, "-I", "-c"], (
            "the round-trip must execute the actual emitted probe program"
        )
        environment = {
            item["name"]: item["value"] for item in container["env"] if "value" in item
        }
        self.identity = json.loads(environment["GPU_FAULT_PROOF_DATABASE"])
        self.dsn = self.make_conninfo(
            host=self.identity["endpoint"],
            port=self.identity["port"],
            dbname=self.identity["database"],
            user=self.identity["username"],
            password=PRIVATE,
            sslmode="verify-full",
            sslrootcert=probe.CA_PATH,
        )
        environment["GPU_FAULT_STORE_URL"] = self.dsn
        self.change_environment(environment)
        self.environment = dict(environment)
        stdout = io.StringIO()
        with patch.dict(os.environ, environment, clear=True), redirect_stdout(stdout):
            exec(
                compile(command[3], "<store-proof-job>", "exec"),
                {"__name__": "__main__"},
            )
        self.producer_calls += 1
        self.producer_stdout = stdout.getvalue()
        # Negative cases corrupt only a freshly executed producer's transport.
        result = json.loads(self.producer_stdout)
        self.corrupt_stdout(result)
        self.release.runner.outputs[arguments[3]] = json.dumps(result)
        job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
        self.release.runner.events.append("store-job")
        return ""

    def prove(self) -> dict[str, Any]:
        return proof.bootstrap_store_proof(
            cast("RegionalRelease", self.release),
            lambda record: self.records.append(copy.deepcopy(record)),
        )

    def assert_cleaned(self) -> None:
        assert self.producer_calls == 1, "a fabricated result bypassed producer main()"
        assert self.release.runner.jobs == {}, "the owned proof Job was not removed"
        assert [record["status"] for record in self.records] == [
            "PLANNED",
            "RUNNING",
            "REMOVED",
        ], "the round-trip bypassed UID-bound Job lifecycle checks"
        assert self.records[-1]["uid"] == self.records[1]["uid"], (
            "cleanup did not retain the created Job UID"
        )


def forbidden_io(*_arguments: Any, **_options: Any) -> NoReturn:
    pytest.fail("Store envelope tests must not perform external I/O")


@pytest.fixture
def round_trip(monkeypatch: pytest.MonkeyPatch) -> EnvelopeRoundTrip:
    for target, name in (
        (subprocess, "Popen"),
        (subprocess, "run"),
        (os, "system"),
        (socket, "getaddrinfo"),
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
    ):
        monkeypatch.setattr(target, name, forbidden_io)
    return EnvelopeRoundTrip(monkeypatch)


def rejected(round_trip: EnvelopeRoundTrip, reason: str) -> dict[str, Any]:
    with pytest.raises(
        ReleaseError, match="safety proof failed or is incomplete"
    ) as caught:
        round_trip.prove()
    round_trip.assert_cleaned()
    error = caught.value
    assert error.__cause__ is None and error.__suppress_context__, (
        "proof rejection can expose an unsafe exception chain"
    )
    rendered = "".join(traceback.format_exception(error))
    assert PRIVATE not in rendered, "proof diagnostics disclosed private material"
    assert round_trip.dsn not in rendered, "proof diagnostics disclosed a DSN"
    for name in ("endpoint", "username", "cluster_resource_id", "master_secret_arn"):
        assert str(round_trip.identity[name]) not in rendered, (
            "proof diagnostics disclosed a database identity value"
        )
    diagnostics: dict[str, Any] = json.loads(str(error).partition("; diagnostics=")[2])
    assert len(str(error)) < 2048, "proof rejection diagnostics exceeded their bound"
    assert diagnostics["validation_reason"] == reason, (
        "the rejection did not identify the failed validation category"
    )
    assert set(diagnostics) <= {
        "validation_reason",
        "probe_stage",
        "error_type",
        *(f"has_{name}" for name in FIELDS),
        "run_id_match",
        "identity_match",
        "timestamp_aware",
        "timestamp_after_start",
        "timestamp_not_future",
        "safe",
        "blockers_clear",
        "database_schema",
        "empty_requirement",
    }, "a proof field escaped the diagnostic key allowlist"
    assert all(
        type(value) is bool
        for name, value in diagnostics.items()
        if name not in {"validation_reason", "probe_stage", "error_type"}
    ), "diagnostics must contain only categories and booleans"
    return diagnostics


def test_empty_producer_output_passes_real_wrapper(
    round_trip: EnvelopeRoundTrip,
) -> None:
    result = round_trip.prove()
    round_trip.assert_cleaned()
    assert result == {
        **json.loads(round_trip.producer_stdout),
        "job_uid": round_trip.records[-1]["uid"],
    }, "the wrapper did not consume the actual producer envelope"
    assert (
        result["safe"] is True and result["database_state"] == "uninitialized_empty"
    ), "the real empty-database producer output was rejected"
    assert set(result) == {*FIELDS, "job_uid"}, "success gained diagnostic-only fields"
    assert round_trip.closed == len(round_trip.connections) == 1, (
        "the producer leaked or skipped its connection context"
    )
    arguments = round_trip.connections[0]
    assert arguments["sslmode"] == "verify-full", "the proof weakened verified TLS"
    assert arguments["sslrootcert"] == probe.CA_PATH, "the proof changed its CA binding"
    assert "default_transaction_read_only=on" in arguments["options"], (
        "the producer omitted its read-only connection policy"
    )
    assert round_trip.database.commands[0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    ), "the proof skipped its read-only transaction"


def test_store_probe_command_selects_its_component_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    round_trip = EnvelopeRoundTrip(monkeypatch)
    round_trip.prove()
    component = tmp_path / "component-bin"
    base = tmp_path / "base-bin"
    for directory, marker in ((component, "component"), (base, "base")):
        directory.mkdir()
        executable = directory / CONTROL_PLANE_PYTHON
        executable.write_text(
            "#!/bin/sh\n"
            '[ "$1" = "-I" ] && [ "$2" = "-c" ] || exit 2\n'
            f"printf '%s\\n' '{marker}'\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    image_path = round_trip.environment.get("PATH", BASE_RUNTIME_PATH)
    component_directory = CONTROL_PLANE_PATH.split(os.pathsep)[0]
    environment = {
        "PATH": os.pathsep.join(
            str(component if directory == component_directory else base)
            for directory in image_path.split(os.pathsep)
        )
    }
    completed = subprocess.run(
        round_trip.command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, "the emitted Python command did not start"
    assert completed.stdout.strip() == "component", (
        "the Store proof selected base Python without the control-plane wheel"
    )
    assert image_path == CONTROL_PLANE_PATH, (
        "the Job must retain the declared component interpreter search path"
    )


@pytest.mark.parametrize(
    ("stage", "error_type"),
    [
        ("import_driver", "ModuleNotFoundError"),
        ("read_identity", "JSONDecodeError"),
        ("bind_connection", "ValueError"),
        ("connect", "OperationalError"),
        ("inspect_database", "ProgrammingError"),
        ("close_connection", "OperationalError"),
        ("build_envelope", "KeyError"),
    ],
)
def test_caught_producer_failures_survive_cleanup_as_safe_diagnostics(
    round_trip: EnvelopeRoundTrip,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    error_type: str,
) -> None:
    import psycopg

    if stage == "import_driver":
        monkeypatch.setitem(sys.modules, "psycopg", None)
    elif stage == "read_identity":
        round_trip.change_environment = lambda env: env.update(
            GPU_FAULT_PROOF_DATABASE=PRIVATE
        )
    elif stage == "bind_connection":
        round_trip.change_environment = lambda env: env.update(
            GPU_FAULT_STORE_URL=round_trip.dsn.replace("verify-full", "require")
        )
    elif stage == "connect":
        round_trip.connect_error = psycopg.OperationalError(PRIVATE)
    elif stage == "inspect_database":
        round_trip.database.error = psycopg.ProgrammingError(PRIVATE)
    elif stage == "close_connection":
        round_trip.close_error = psycopg.OperationalError(PRIVATE)
    else:
        round_trip.change_environment = lambda env: env.__delitem__(
            "GPU_FAULT_PROOF_RUN_ID"
        )
    diagnostics = rejected(round_trip, "missing_field")
    assert json.loads(round_trip.producer_stdout) == {
        "safe": False,
        "probe_stage": stage,
        "error_type": error_type,
    }, "the producer lost its bounded error envelope"
    assert (
        diagnostics["probe_stage"] == stage and diagnostics["error_type"] == error_type
    ), "the wrapper discarded the producer's failure category"
    assert diagnostics["has_finished_at"] is False, (
        "a failed producer was given a fabricated completion timestamp"
    )
    captured = capsys.readouterr()
    assert captured.out == captured.err == "", "the producer leaked uncaptured output"


@pytest.mark.parametrize("field", FIELDS)
def test_missing_success_fields_never_become_a_proof(
    round_trip: EnvelopeRoundTrip, field: str
) -> None:
    round_trip.corrupt_stdout = lambda result: result.__delitem__(field)
    diagnostics = rejected(round_trip, "missing_field")
    assert diagnostics[f"has_{field}"] is False, "the missing field was not identified"
    assert field in json.loads(round_trip.producer_stdout), (
        "the corruption test did not start with real producer output"
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("run_id", PRIVATE, "run_id_match"),
        ("run_id", [], "run_id_match"),
        ("identity_sha256", PRIVATE, "identity_match"),
        ("identity_sha256", {}, "identity_match"),
        ("safe", False, "safe"),
        ("safe", 1, "safe"),
        ("safe", "true", "safe"),
        ("blockers", {}, "blockers_clear"),
        ("blockers", [], "blockers_clear"),
        (
            "blockers",
            {"workflow": 1, "remote_command": 0, "observation": 0},
            "blockers_clear",
        ),
        (
            "blockers",
            {"workflow": "0", "remote_command": 0, "observation": 0},
            "blockers_clear",
        ),
        ("database_state", PRIVATE, "database_schema"),
        ("database_state", [], "database_schema"),
        ("schema_version", "0", "database_schema"),
        ("schema_version", {}, "database_schema"),
        ("schema_version", 1, "database_schema"),
        ("finished_at", PRIVATE, "invalid_timestamp"),
        ("finished_at", None, "invalid_type"),
        ("finished_at", 1, "invalid_type"),
        ("finished_at", [], "invalid_type"),
        ("finished_at", {}, "invalid_type"),
        ("finished_at", "2000-01-01T00:00:00", "timestamp_aware"),
        ("finished_at", "2000-01-01T00:00:00+00:00", "timestamp_after_start"),
        ("finished_at", "2999-01-01T00:00:00+00:00", "timestamp_not_future"),
    ],
    ids=[
        "wrong-run",
        "run-type",
        "wrong-identity",
        "identity-type",
        "unsafe",
        "safe-integer",
        "safe-string",
        "missing-blockers",
        "blocker-type",
        "active-blocker",
        "blocker-count-type",
        "unknown-state",
        "state-type",
        "schema-string",
        "schema-object",
        "wrong-schema",
        "timestamp-invalid",
        "timestamp-null",
        "timestamp-integer",
        "timestamp-list",
        "timestamp-object",
        "timestamp-naive",
        "timestamp-old",
        "timestamp-future",
    ],
)
def test_corrupt_producer_envelopes_remain_rejected(
    round_trip: EnvelopeRoundTrip, field: str, value: Any, reason: str
) -> None:
    round_trip.corrupt_stdout = lambda result: result.update({field: value})
    diagnostics = rejected(round_trip, reason)
    assert all(diagnostics[f"has_{name}"] for name in FIELDS), (
        "this case must exercise an invalid value, not an absent field"
    )
    assert json.loads(round_trip.producer_stdout)["safe"] is True, (
        "the transport corruption bypassed a successful producer execution"
    )


@pytest.mark.parametrize("value", [PRIVATE, [PRIVATE], {"private": PRIVATE}])
def test_untrusted_diagnostic_fields_are_not_echoed(
    round_trip: EnvelopeRoundTrip, value: Any
) -> None:
    round_trip.corrupt_stdout = lambda result: result.update(
        safe=False, probe_stage=value, error_type=value, unrecognized=PRIVATE
    )
    diagnostics = rejected(round_trip, "safe")
    assert diagnostics["probe_stage"] == diagnostics["error_type"] == "unknown", (
        "unrecognized diagnostic values escaped the allowlist"
    )


@pytest.mark.parametrize("base", [Exception, RuntimeError])
def test_exception_names_messages_and_chains_are_never_disclosed(
    round_trip: EnvelopeRoundTrip, base: type[Exception]
) -> None:
    failure = type(PRIVATE, (base,), {})(PRIVATE)
    failure.add_note(PRIVATE)
    failure.__cause__ = ValueError(PRIVATE)
    round_trip.connect_error = failure
    try:
        raise RuntimeError(PRIVATE)
    except RuntimeError:
        diagnostics = rejected(round_trip, "missing_field")
    expected = "RuntimeError" if base is RuntimeError else "unknown"
    assert diagnostics["error_type"] == expected, (
        "a custom exception was not reduced to an allowlisted category"
    )
    assert PRIVATE not in round_trip.producer_stdout, (
        "the producer emitted an arbitrary exception class, message, or chain"
    )


def test_rejection_preserves_predicate_short_circuiting(
    round_trip: EnvelopeRoundTrip,
) -> None:
    round_trip.corrupt_stdout = lambda result: result.update(
        run_id=PRIVATE, identity_sha256=PRIVATE, safe=False
    )
    diagnostics = rejected(round_trip, "run_id_match")
    assert diagnostics["run_id_match"] is False, "the first mismatch was not recorded"
    assert "identity_match" not in diagnostics and "safe" not in diagnostics, (
        "diagnostics evaluated predicates after the first failure"
    )


def test_initialized_claim_does_not_satisfy_empty_bootstrap(
    round_trip: EnvelopeRoundTrip,
) -> None:
    round_trip.corrupt_stdout = lambda result: result.update(
        database_state="initialized",
        schema_version=round_trip.release.config.database_schema_version,
    )
    diagnostics = rejected(round_trip, "empty_requirement")
    assert diagnostics["database_schema"] is True, (
        "the test did not reach the independent empty-database requirement"
    )
