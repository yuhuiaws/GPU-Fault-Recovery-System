"""Owned-I/O witnesses for store lifecycle, public config, and epoch preflight."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.config import (
    load_desired_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.config_file import (
    initialize_desired_admin_config,
    load_admin_config_file,
)
from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_lifecycle import INSTALLATION_FILE
from gpu_fault.schema_migrations import POSTGRES_SCHEMA_MIGRATIONS
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseConfig
from scripts.e2e.regional import boot020_admin_config as admin_config
from scripts.e2e.regional import boot024_epoch_preflight as epoch
from scripts.e2e.regional import boot_store_lifecycle_evidence as lifecycle
from tests.admin.test_admin_site import site_file
from tests.regional.test_acceptance_alignment_lifecycle_witness import (
    documents as existing_lifecycle_documents,
)
from tests.regional.test_release_diff import _release as release_diff_fixture
from tests.regional.test_release_diff import _state as existing_release_state

lifecycle_documents: Callable[
    [], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
] = existing_lifecycle_documents
release_state_fixture: Callable[[], dict[str, Any]] = existing_release_state

OBSERVATION_TIME = datetime(2026, 9, 15, 12, tzinfo=UTC)
CPU_ROLES = {
    "worker": "gpu-fault-control-worker",
    "ingress": "gpu-fault-api-ha",
    "spool": "gpu-fault-telemetry-spool-worker",
}


def migration_rows(version: int) -> list[list[Any]]:
    return [
        [item.version, item.name, item.checksum]
        for item in POSTGRES_SCHEMA_MIGRATIONS
        if item.version <= version
    ]


@dataclass
class ProbeRows:
    rows: list[tuple[Any, ...]]

    def fetchone(self) -> tuple[Any, ...]:
        assert self.rows, "the emitted probe requested an absent singleton row"
        return self.rows[0]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return copy.deepcopy(self.rows)


class LifecycleIO(CommandRunner):
    """Execute the emitted probe with owned rows and record its actual I/O."""

    def __init__(self, site: RenderedSite) -> None:
        super().__init__()
        self.site = site
        self.version = 17
        self.replica = False
        self.rows = [
            ("workflow", "history", {"status": "SUCCEEDED", "steps": ["observed"]}),
            ("remote_command", "command", {"status": "FAILED", "exit_code": 1}),
        ]
        self.identity = {"release_id": "release-a", "cluster_id": "gpu-a"}
        self.namespace_uid = "original-namespace"
        _before, after, _uninstall = lifecycle_documents()
        self.release = copy.deepcopy(after["release_state"])
        self.release["unrelated_debug_field"] = "example-not-part-of-the-witness"
        self.live_jobs: list[dict[str, Any]] = []
        self.sql: list[tuple[str, object]] = []
        self.kube_reads: list[tuple[str, ...]] = []
        self.connection_events: list[str] = []
        self.connection_options: list[dict[str, Any]] = []
        self.aws_reads: list[tuple[str, ...]] = []
        self.fixture_targets: list[tuple[Path, str]] = []

    @contextmanager
    def connect(self, dsn: str, **options: Any) -> Iterator[LifecycleIO]:
        assert dsn == "postgresql://example.invalid/owned-probe", (
            "the probe tried to use credentials outside its owned fixture"
        )
        self.connection_options.append(options)
        self.connection_events.append("connect")
        try:
            yield self
        finally:
            self.connection_events.append("disconnect")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.connection_events.append("transaction-enter")
        try:
            yield
        finally:
            self.connection_events.append("transaction-exit")

    def execute(self, statement: str, parameters: object = None) -> ProbeRows:
        self.sql.append((statement, copy.deepcopy(parameters)))
        if statement == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY":
            return ProbeRows([])
        if statement == "SELECT pg_is_in_recovery()":
            return ProbeRows([(self.replica,)])
        if statement == "SELECT version FROM gpu_fault_schema_version WHERE singleton":
            return ProbeRows([(self.version,)])
        if statement == (
            "SELECT version,name,checksum FROM gpu_fault_schema_migrations ORDER BY version"
        ):
            return ProbeRows([tuple(item) for item in migration_rows(self.version)])
        if statement.startswith(
            "SELECT kind,key,payload FROM gpu_fault_control_records "
        ):
            return ProbeRows(copy.deepcopy(self.rows))
        pytest.fail(f"unexpected store probe statement: {statement}")

    def evidence_identity(self) -> dict[str, str]:
        return dict(self.identity)

    def cpu_python(self, script: str) -> dict[str, Any]:
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(script, "<owned-boot-store-probe>", "exec"), {})
        value: dict[str, Any] = json.loads(output.getvalue())
        return value

    def kubectl(self, plane: str, *args: str) -> str:
        assert plane == "cpu", "store observations must never contact the GPU plane"
        self.kube_reads.append(args)
        responses = {
            ("get", "configmap", "gpu-fault-regional-release-state", "-o", "json"): {
                "data": {"state.json": json.dumps(self.release)}
            },
            (
                "get",
                "namespace",
                str(self.site.release_config["namespace"]),
                "-o",
                "json",
            ): {"metadata": {"uid": self.namespace_uid}},
            ("get", "jobs", "-o", "json"): {"items": self.live_jobs},
        }
        assert args in responses, (
            "the lifecycle witness attempted a mutation or unknown read"
        )
        return json.dumps(responses[args])

    def aws_json(
        self,
        region: str,
        *arguments: str,
        mutate: bool = False,
        sensitive: bool = False,
    ) -> dict[str, Any]:
        config = self.site.release_config
        cluster = config["health"]["aurora_cluster_id"]
        assert region == config["aws_region"], "Aurora discovery changed region"
        assert arguments == (
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            cluster,
        ), "lifecycle evidence attempted an AWS mutation or an unrelated database read"
        self.aws_reads.append(arguments)
        return {
            "DBClusters": [
                {
                    "DBClusterIdentifier": cluster,
                    "DBClusterArn": f"arn:aws:rds:{region}:123456789012:cluster:{cluster}",
                    "Engine": "aurora-postgresql",
                    "Status": "available",
                    "DbClusterResourceId": "example-original-database",
                    "Endpoint": "example.invalid",
                    "DatabaseName": "owned",
                    "MasterUsername": "example",
                    "MasterUserSecret": {
                        "SecretArn": f"arn:aws:secretsmanager:{region}:123456789012:secret:example"
                    },
                    "Port": 5432,
                }
            ]
        }

    def fixture(self, path: Path, cluster_id: str) -> SimpleNamespace:
        self.fixture_targets.append((path, cluster_id))
        assert path == self.site.source and cluster_id == "gpu-a", (
            "the observation selected a different installed target"
        )
        return SimpleNamespace(regional=self)


@pytest.fixture
def lifecycle_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LifecycleIO:
    value = LifecycleIO(load_site(site_file(tmp_path)))
    monkeypatch.setattr(lifecycle, "SiteFixture", value.fixture)
    monkeypatch.setattr(lifecycle, "CommandRunner", lambda: value)
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=value.connect))
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://example.invalid/owned-probe"
    )
    return value


@pytest.mark.parametrize(
    "keys", [None, ["history", "command"]], ids=["terminal", "retained-keys"]
)
@pytest.mark.parametrize(
    "fallback", [False, True], ids=["repair-jobs", "bootstrap-jobs"]
)
def test_observation_executes_read_only_probe_and_projects_only_bound_job_fields(
    lifecycle_io: LifecycleIO, keys: list[str] | None, fallback: bool
) -> None:
    state_dir = lifecycle_io.site.source.parent
    installation = {"installation_id": "example-installation"}
    if fallback:
        write_json_atomic(state_dir / INSTALLATION_FILE, installation)
        jobs = lifecycle_io.release.pop("aurora_prerequisite_repair")["jobs"]
        lifecycle_io.release["bootstrap_store_safety"]["proof_jobs"] = jobs
    else:
        jobs = lifecycle_io.release["aurora_prerequisite_repair"]["jobs"]
    jobs["job"]["run_id"] = "example-proof-run"
    jobs["job"]["spec_sha256"] = "a" * 64
    jobs["job"]["unrelated_debug_field"] = "not-an-identity"
    lifecycle_io.live_jobs = [
        {"metadata": {"name": "unrelated", "uid": "unrelated-uid"}}
    ]

    report = lifecycle.observe(state_dir, keys=keys)

    assert report["store"] == {
        "schema_version": lifecycle_io.version,
        "migrations": migration_rows(lifecycle_io.version),
        "records": {
            f"{kind}/{key}": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for kind, key, payload in lifecycle_io.rows
        },
    }, "the emitted probe did not hash the rows it actually read"
    query, parameters = lifecycle_io.sql[-1]
    assert parameters == (
        ["workflow", "remote_command"],
        keys or ["SUCCEEDED", "FAILED", "SUPERSEDED", "CANCELLED", "TIMED_OUT"],
    )
    assert ("key=ANY(%s)" if keys else "payload->>'status'=ANY(%s)") in query, (
        "capture and verification used the wrong SQL record selection"
    )
    assert lifecycle_io.connection_options == [
        {
            "autocommit": True,
            "connect_timeout": 10,
            "options": (
                "-c default_transaction_read_only=on -c statement_timeout=30000 "
                "-c lock_timeout=5000"
            ),
        }
    ]
    assert lifecycle_io.sql[0][0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )
    assert lifecycle_io.connection_events == [
        "connect",
        "transaction-enter",
        "transaction-exit",
        "disconnect",
    ], "the observation leaked a database transaction or connection"
    assert report["database"]["cluster_resource_id"] == "example-original-database"
    assert report["identity"] == lifecycle_io.identity
    assert report["namespace_uid"] == lifecycle_io.namespace_uid
    assert (
        report["state_dir_sha256"]
        == hashlib.sha256(str(state_dir.resolve()).encode()).hexdigest()
    )
    assert report["installation"] == (installation if fallback else {})
    assert report["live_jobs"] == [{"name": "unrelated", "uid": "unrelated-uid"}]
    assert report["release_state"]["aurora_prerequisite_repair"]["jobs"] == {
        "job": {
            key: jobs["job"][key]
            for key in ("name", "uid", "owner_uid", "run_id", "spec_sha256", "status")
        }
    }, "proof-job observation retained fields outside the identity contract"
    assert "unrelated_debug_field" not in report["release_state"]
    assert lifecycle_io.fixture_targets == [(lifecycle_io.site.source, "gpu-a")]
    assert datetime.fromisoformat(report["observed_at"]).tzinfo is not None


def test_lifecycle_observation_refuses_an_empty_gpu_baseline(
    lifecycle_io: LifecycleIO,
) -> None:
    path = lifecycle_io.site.source
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["clusters"] = []
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="installed baseline GPU"):
        lifecycle.observe(path.parent)
    assert lifecycle_io.fixture_targets == [], "an empty fleet reached cluster I/O"
    assert lifecycle_io.connection_events == []


def test_emitted_store_probe_refuses_a_replica_and_closes_its_transaction(
    lifecycle_io: LifecycleIO,
) -> None:
    lifecycle_io.replica = True
    with pytest.raises(RuntimeError, match="requires the writer"):
        lifecycle.observe(lifecycle_io.site.source.parent)
    assert [query for query, _params in lifecycle_io.sql] == [
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY",
        "SELECT pg_is_in_recovery()",
    ], "replica rejection must precede schema and historical record reads"
    assert lifecycle_io.connection_events == [
        "connect",
        "transaction-enter",
        "transaction-exit",
        "disconnect",
    ]
    assert lifecycle_io.kube_reads == [] and lifecycle_io.aws_reads == []


def schema_documents(mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before, after, _uninstall = lifecycle_documents()
    before["observed_at"] = OBSERVATION_TIME.isoformat()
    after["observed_at"] = (OBSERVATION_TIME + timedelta(seconds=10)).isoformat()
    after["namespace_uid"] = before["namespace_uid"]
    if mode == "fail-forward":
        after["release_state"].update(phase="failed", transaction_committed=False)
    return before, after


@pytest.mark.parametrize("mode", ["schema", "fail-forward"])
@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("naive-before", "timestamps are unordered or unbound"),
        ("naive-after", "timestamps are unordered or unbound"),
        ("backwards-time", "timestamps are unordered or unbound"),
        ("history", "historical schema migrations changed"),
        ("namespace", "replaced the original namespace"),
        ("no-advance", "approved pre-schema snapshot"),
        ("no-schema-ready", "approved pre-schema snapshot"),
        ("snapshot-schema", "approved pre-schema snapshot"),
        ("snapshot-mode", "approved pre-schema snapshot"),
        ("snapshot-unavailable", "approved pre-schema snapshot"),
        ("snapshot-id", "approved pre-schema snapshot"),
        ("not-committed", "schema candidate did not commit"),
    ],
)
def test_schema_witness_rejects_unbound_observations_and_unapproved_advances(
    mode: str, damage: str, expected: str
) -> None:
    before, after = schema_documents(mode)
    release = after["release_state"]
    acceptance = release["schema_change_acceptance"]
    if damage == "naive-before":
        before["observed_at"] = OBSERVATION_TIME.replace(tzinfo=None).isoformat()
    elif damage == "naive-after":
        after["observed_at"] = OBSERVATION_TIME.replace(tzinfo=None).isoformat()
    elif damage == "backwards-time":
        after["observed_at"] = (OBSERVATION_TIME - timedelta(seconds=1)).isoformat()
    elif damage == "history":
        after["store"]["migrations"][0][2] = "changed-history"
    elif damage == "namespace":
        after["namespace_uid"] = "replacement"
    elif damage == "no-advance":
        after["store"] = copy.deepcopy(before["store"])
    elif damage == "no-schema-ready":
        release["completed_phases"] = []
    elif damage.startswith("snapshot-"):
        field, value = {
            "snapshot-schema": ("database_schema_version", 17),
            "snapshot-mode": ("mode", "none"),
            "snapshot-unavailable": ("snapshot_status", "creating"),
            "snapshot-id": ("snapshot_id", ""),
        }[damage]
        acceptance[field] = value
    else:
        release["transaction_committed"] = mode == "fail-forward"
        if mode == "fail-forward":
            expected = "schema failure was not retained for fail-forward"
    errors = lifecycle.transition_errors(before, after, mode=mode)
    assert any(expected in error for error in errors), (
        f"{mode} accepted {damage} without its causal rejection: {errors}"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "complete"},
        {"transaction_committed": True},
        {"rollback_completed": ["cpu"]},
        {"release_lifecycle": "ROLLED_BACK"},
    ],
    ids=["not-failed", "committed", "rollback-completed", "rollback-lifecycle"],
)
def test_fail_forward_witness_rejects_commit_or_runtime_rollback(
    change: dict[str, Any],
) -> None:
    before, after = schema_documents("fail-forward")
    after["release_state"].update(change)
    expected = (
        "schema failure attempted a runtime rollback"
        if "rollback_completed" in change or "release_lifecycle" in change
        else "schema failure was not retained for fail-forward"
    )
    assert expected in lifecycle.transition_errors(before, after, mode="fail-forward")


@pytest.mark.parametrize("phase", ["failed", "partial-convergence"])
def test_fail_forward_preserves_the_advanced_schema_without_runtime_rollback(
    phase: str,
) -> None:
    before, after = schema_documents("fail-forward")
    after["release_state"]["phase"] = phase
    assert lifecycle.transition_errors(before, after, mode="fail-forward") == []


def test_unknown_lifecycle_mode_is_not_silently_treated_as_a_schema_upgrade() -> None:
    before, after = schema_documents("schema")
    with pytest.raises(ValueError, match="unknown lifecycle proof mode"):
        lifecycle.transition_errors(before, after, mode="unknown")


def lifecycle_cli(
    monkeypatch: pytest.MonkeyPatch, state: Path, output: Path, *arguments: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["boot-store", *arguments, "--state-dir", str(state), "--output", str(output)],
    )
    lifecycle.main()


def test_lifecycle_cli_captures_then_verifies_actual_observed_schema_and_history(
    lifecycle_io: LifecycleIO,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = lifecycle_io.site.source.parent
    before = state / "before.json"
    output = state / "proof.json"
    lifecycle_cli(monkeypatch, state, before, "capture")
    captured = before.read_bytes()
    assert json.loads(captured)["store"]["schema_version"] == 17
    lifecycle_io.version = 18
    lifecycle_cli(
        monkeypatch,
        state,
        output,
        "verify",
        "--before",
        str(before),
        "--mode",
        "schema",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == {"verdict": "PASS", "errors": []}
    assert report["report_type"] == "boot-store-lifecycle-proof"
    assert report["before_sha256"] == hashlib.sha256(captured).hexdigest()
    assert report["historical_records_checked"] == len(lifecycle_io.rows)
    assert report["after"]["store"]["schema_version"] == 18
    assert (
        report["scope"]
        == "observed lifecycle transition only; not the whole regional case"
    )
    assert before.read_bytes() == captured, "verify overwrote its original witness"
    assert lifecycle_io.sql[-1][1] == (
        ["workflow", "remote_command"],
        ["command", "history"],
    ), "verification must request exactly the captured record keys"
    assert output.stat().st_mode & 0o777 == 0o600


def test_lifecycle_cli_writes_a_failed_proof_and_exits_nonzero_on_history_loss(
    lifecycle_io: LifecycleIO,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = lifecycle_io.site.source.parent
    before, output = state / "before.json", state / "proof.json"
    lifecycle_cli(monkeypatch, state, before, "capture")
    lifecycle_io.version = 18
    lifecycle_io.rows = []
    with pytest.raises(SystemExit) as raised:
        lifecycle_cli(
            monkeypatch,
            state,
            output,
            "verify",
            "--before",
            str(before),
            "--mode",
            "schema",
        )
    assert raised.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"
    assert (
        "retained workflow/command history changed or disappeared" in report["errors"]
    )
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"


@pytest.mark.parametrize("kind", ["existing", "dangling-symlink"])
def test_capture_refuses_to_replace_a_prior_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    output = tmp_path / "before.json"
    if kind == "existing":
        output.write_text("original witness", encoding="utf-8")
    else:
        output.symlink_to(tmp_path / "absent.json")
    with pytest.raises(SystemExit) as raised:
        lifecycle_cli(monkeypatch, tmp_path, output, "capture")
    assert raised.value.code == 2
    assert "do not replace an old witness" in capsys.readouterr().err
    if kind == "existing":
        assert output.read_text(encoding="utf-8") == "original witness"
    else:
        assert output.is_symlink(), "capture replaced a protected output symlink"
        assert not (tmp_path / "absent.json").exists(), (
            "capture followed a dangling link"
        )


@pytest.mark.parametrize("missing", ["before", "mode"])
def test_verify_requires_both_predecessor_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    arguments = (
        ("--mode", "schema") if missing == "before" else ("--before", "before.json")
    )
    with pytest.raises(SystemExit) as raised:
        lifecycle_cli(
            monkeypatch, tmp_path, tmp_path / "proof.json", "verify", *arguments
        )
    assert raised.value.code == 2
    assert "requires before and mode" in capsys.readouterr().err
    assert not (tmp_path / "proof.json").exists(), "usage failure wrote a proof"


def test_verify_cannot_overwrite_its_before_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before = tmp_path / "before.json"
    before.write_text("original witness", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        lifecycle_cli(
            monkeypatch,
            tmp_path,
            before,
            "verify",
            "--before",
            str(before),
            "--mode",
            "schema",
        )
    assert raised.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert before.read_text(encoding="utf-8") == "original witness"


@pytest.mark.parametrize("archive", [None, "../foreign", "owned-archive"])
def test_reinstall_verification_requires_a_local_completed_uninstall_archive(
    lifecycle_io: LifecycleIO, monkeypatch: pytest.MonkeyPatch, archive: str | None
) -> None:
    state = lifecycle_io.site.source.parent
    before, output = state / "before.json", state / "proof.json"
    lifecycle_cli(monkeypatch, state, before, "capture")
    write_json_atomic(
        state / INSTALLATION_FILE,
        {"installation_id": "new", "retained_uninstall": {"archive": archive}},
    )
    if archive == "owned-archive":
        write_json_atomic(
            state / archive / "uninstall/state.json", {"phase": "STARTED"}
        )
        expected: type[Exception] = BootstrapError
        match = "bound completed uninstall"
    else:
        expected, match = ValueError, "installation archive is missing"
    with pytest.raises(expected, match=match):
        lifecycle_cli(
            monkeypatch,
            state,
            output,
            "verify",
            "--before",
            str(before),
            "--mode",
            "reinstall",
        )
    assert not output.exists(), (
        "an invalid retained handoff produced an acceptance proof"
    )


class ConfigIO:
    """Represent CLI responses and observed rollouts without starting a driver."""

    def __init__(self, path: Path) -> None:
        self.site = load_site(path)
        self.root = path.parent
        self.before = initialize_desired_admin_config(self.root)
        self.desired = self.before.patched(
            {
                "capacity": {
                    "controlWorkerReplicas": self.before.capacity.control_worker_replicas
                    - 1
                }
            }
        )
        self.runtime = self.before
        self.phase = "initial"
        self.generations = dict.fromkeys(CPU_ROLES, 1)
        self.calls: list[str] = []
        self.responses: dict[str, dict[str, Any]] = {}
        self.prefix: dict[str, str] = {}
        self.stamp_all_roles = True
        self.exit_codes: dict[str, int] = {}
        self.persist = {"apply": True, "restore": True}
        self.pending_at: set[str] = set()
        self.cpu_damage: dict[str, tuple[str, str, object]] = {}
        self.gpu_damage: set[str] = set()
        self.audit = self.root / "admin-config/owned-audit.json"
        self.audit_content: dict[str, Any] = {"status": "APPLIED"}

    def command(
        self, arguments: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        assert arguments[:4] == [
            sys.executable,
            "-m",
            "gpu_fault.admin.cli",
            "config",
        ], "the witness bypassed the public administrator config entrypoint"
        assert arguments[4:8] == [
            "--state-dir",
            str(self.root),
            "--reference",
            "CHG-OWNED",
        ], "the command lost its state directory or approval reference"
        descriptor = int(options["env"][SITE_OPERATION_LOCK_FD_ENV])
        assert options["pass_fds"] == (descriptor,), (
            "the child lost the actual site lock"
        )
        assert os.fstat(descriptor).st_ino > 0, (
            "the inherited lock descriptor is closed"
        )
        assert options["capture_output"] is True
        target = load_admin_config_file(
            self.root / "admin-config.yaml", base=self.before
        )
        if "--dry-run" in arguments:
            event = "dry-run"
        elif target == self.before:
            event = "restore"
        elif "apply" not in self.calls:
            event = "apply"
        else:
            event = "noop"
        self.calls.append(event)
        self.phase = event
        if event in self.pending_at:
            write_json_atomic(
                self.root / "admin-config/pending.json", {"status": "example-in-flight"}
            )
        code = self.exit_codes.get(event, 0)
        if code:
            return subprocess.CompletedProcess(arguments, code, "", "owned failure")
        if event in {"apply", "restore"}:
            for role, digest in target.role_sha256().items():
                if self.runtime.role_sha256()[role] != digest or self.stamp_all_roles:
                    # The product re-stamps gpu-fault.io/admin-config-sha256 on every
                    # CPU Deployment after each apply, and Kubernetes bumps a
                    # Deployment's generation on annotation changes (live 2026-09-20:
                    # api-ha/spool generation +1 per apply, no ReplicaSet, no Pod).
                    self.generations[role] += 1
            self.runtime = target
            if self.persist[event]:
                persist_desired_admin_config(
                    self.root, config=target, source="example-owned-config-command"
                )
            if event == "apply":
                write_json_atomic(self.audit, self.audit_content)
        default = {
            "status": {
                "dry-run": "DRY_RUN",
                "apply": "APPLIED",
                "noop": "NOOP",
                "restore": "APPLIED",
            }[event],
            "audit": str(self.audit),
        }
        return subprocess.CompletedProcess(
            arguments,
            code,
            self.prefix.get(event, "") + json.dumps(self.responses.get(event, default)),
            "",
        )

    def observation(self, site: RenderedSite) -> dict[str, Any]:
        assert site.source == self.site.source, "CPU observation escaped the owned site"
        result = {
            name: {
                "uid": role,
                "generation": self.generations[role],
                "replicas": self.runtime.capacity.control_worker_replicas
                if role == "worker"
                else 1,
                "template_sha256": self.runtime.role_sha256()[role],
            }
            for role, name in CPU_ROLES.items()
        }
        if self.phase in self.cpu_damage:
            role, field, value = self.cpu_damage[self.phase]
            result[CPU_ROLES[role]][field] = value
        return result

    def deployments(self, path: Path) -> dict[str, Any]:
        assert path == self.site.source, "GPU observation escaped the owned site"
        return {
            "gpu": {"gpu-a": {"executor": 2 if self.phase in self.gpu_damage else 1}}
        }

    def run(self, reference: str = "CHG-OWNED") -> dict[str, Any]:
        return admin_config.public_config_roundtrip(
            self.root, desired=self.desired, reference=reference
        )


@pytest.fixture
def config_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigIO:
    value = ConfigIO(site_file(tmp_path))
    monkeypatch.setattr(admin_config, "run_driver", value.command)
    monkeypatch.setattr(admin_config, "cpu_observation", value.observation)
    monkeypatch.setattr(admin_config, "deployment_generations", value.deployments)
    return value


@pytest.mark.parametrize(
    "damage", ["none", "membership-id", "membership-arn", "empty", "namespace"]
)
def test_admin_target_validation_binds_all_gpu_members_and_control_plane_fields(
    config_io: ConfigIO, damage: str
) -> None:
    value = copy.deepcopy(config_io.site.release_config)
    if damage == "membership-id":
        value["clusters"][0]["cluster_id"] = "foreign"
    elif damage == "membership-arn":
        value["clusters"][0]["eks_cluster_arn"] = value["cpu_eks_arn"]
    elif damage == "empty":
        value["clusters"] = []
    elif damage == "namespace":
        value["namespace"] = "foreign"
    path = config_io.root / "candidate.json"
    write_json_atomic(path, value)
    if damage == "none":
        admin_config.validate_admin_target(config_io.root, path)
    else:
        with pytest.raises(ValueError, match="differs"):
            admin_config.validate_admin_target(config_io.root, path)
    assert config_io.calls == [], "target validation started an administrator command"


def test_public_config_drill_reads_the_report_printed_after_the_child_release(
    config_io: ConfigIO,
) -> None:
    # Live 2026-09-20 (BOOT-020 a2): the apply and restore invocations run the
    # automatic release as a child that inherits stdout, so its JSON summary and
    # progress lines precede the command's own APPLIED report.
    summary = json.dumps({"release_id": "0e813e5eaf80", "status": "verified"}, indent=2)
    config_io.prefix["apply"] = (
        "release-deploy: rolling control-worker\n" + summary + "\n"
    )
    config_io.prefix["restore"] = summary + "\n"

    result = config_io.run()

    assert result["applied_config_sha256"] == config_io.desired.sha256()
    assert result["restored_config_sha256"] == config_io.before.sha256()
    assert config_io.calls == ["dry-run", "apply", "noop", "restore"]


@pytest.mark.parametrize("damage", ["inventory", "uid"])
def test_changed_roles_rejects_replacement_instead_of_counting_it_as_rollout(
    damage: str,
) -> None:
    before = {"worker": {"uid": "original", "generation": 1}}
    after = copy.deepcopy(before)
    if damage == "inventory":
        after["ingress"] = {"uid": "other", "generation": 1}
        match = "role inventory"
    else:
        after["worker"]["uid"] = "replacement"
        match = "replaced a CPU Deployment"
    with pytest.raises(ValueError, match=match):
        admin_config.changed_roles(before, after)


@pytest.mark.parametrize("damage", ["reference", "pending", "no-change", "aurora"])
def test_public_config_refuses_unowned_or_non_cpu_changes_before_editing(
    config_io: ConfigIO, damage: str
) -> None:
    editable = config_io.root / "admin-config.yaml"
    before = editable.read_bytes()
    reference = "CHG-OWNED"
    if damage == "reference":
        reference, match = "  ", "requires a change reference"
    elif damage == "pending":
        write_json_atomic(
            config_io.root / "admin-config/pending.json", {"phase": "APPLYING"}
        )
        match = "take over a pending apply"
    elif damage == "no-change":
        config_io.desired = config_io.before
        match = "CPU-only change"
    else:
        config_io.desired = config_io.desired.patched({"aurora": {"maxAcu": 64}})
        match = "CPU-only change"
    with pytest.raises(ValueError, match=match):
        config_io.run(reference)
    assert editable.read_bytes() == before, (
        "rejected admission overwrote the editable config"
    )
    assert config_io.calls == [], "rejected admission invoked a public command"


@pytest.mark.parametrize("damage", ["exit", "status", "pending"])
def test_unapplied_config_failure_does_not_issue_a_restore_or_erase_pending_evidence(
    config_io: ConfigIO, damage: str
) -> None:
    if damage == "exit":
        config_io.exit_codes["dry-run"] = 7
        match = "command failed"
    elif damage == "status":
        config_io.responses["dry-run"] = {"status": "NOOP"}
        match = "dry-run changed"
    else:
        config_io.pending_at.add("dry-run")
        match = "dry-run changed"
    with pytest.raises(ValueError, match=match):
        config_io.run()
    assert config_io.calls == ["dry-run"], (
        "unapplied failure attempted an automatic restore"
    )
    assert load_desired_admin_config(config_io.root) == config_io.before
    assert (
        load_admin_config_file(config_io.root / "admin-config.yaml")
        == config_io.desired
    )
    assert (config_io.root / "admin-config/pending.json").exists() == (
        damage == "pending"
    )


@pytest.mark.parametrize("damage", ["status", "persisted-config"])
def test_apply_requires_a_committed_requested_configuration(
    config_io: ConfigIO, damage: str
) -> None:
    if damage == "status":
        config_io.responses["apply"] = {"status": "PENDING"}
        config_io.pending_at.add("apply")
    else:
        config_io.persist["apply"] = False
    with pytest.raises(ValueError, match="did not commit"):
        config_io.run()
    assert config_io.calls == (
        ["dry-run", "apply"] if damage == "status" else ["dry-run", "apply", "restore"]
    ), "only an acknowledged application can enter the witness's restore path"
    if damage == "status":
        assert (config_io.root / "admin-config/pending.json").is_file(), (
            "an unconfirmed application lost its reconciliation evidence"
        )
    else:
        assert load_desired_admin_config(config_io.root) == config_io.before


@pytest.mark.parametrize("damage", ["cpu", "gpu", "noop-status", "noop-cpu"])
def test_public_config_checks_the_changed_role_set_and_a_strict_repeat_noop(
    config_io: ConfigIO, damage: str
) -> None:
    if damage == "cpu":
        config_io.cpu_damage["apply"] = ("ingress", "template_sha256", "foreign")
    elif damage == "gpu":
        config_io.gpu_damage.add("apply")
    elif damage == "noop-status":
        config_io.responses["noop"] = {"status": "APPLIED"}
    else:
        config_io.cpu_damage["noop"] = ("worker", "generation", 99)
    match = "outside its declared scope" if damage in {"cpu", "gpu"} else "strict NOOP"
    with pytest.raises(ValueError, match=match):
        config_io.run()
    assert config_io.calls[-1] == "restore", (
        "acknowledged config change was not restored"
    )
    assert ("noop" in config_io.calls) == (damage not in {"cpu", "gpu"})
    assert load_desired_admin_config(config_io.root) == config_io.before
    assert config_io.runtime == config_io.before


def test_public_config_drill_reads_replicas_template_and_pods_not_generation(
    config_io: ConfigIO,
) -> None:
    # Live 2026-09-20 (BOOT-020 a3): every apply re-stamped api-ha and spool with
    # the admin-config digest annotation, Kubernetes bumped their generation, and
    # the drill called that "rolled a role outside its declared scope".
    config_io.run()  # stamp_all_roles is on: generation moves on every role
    config_io.calls.clear()
    config_io.cpu_damage["apply"] = ("spool", "replicas", 2)
    with pytest.raises(
        ValueError, match="outside its declared scope.*spool"
    ) as failure:
        config_io.run()
    assert "expected=" in str(failure.value), "the scope error names the plan"


@pytest.mark.parametrize("damage", ["outside-state", "empty-audit", "pending"])
def test_public_config_audit_must_be_local_complete_and_not_pending(
    config_io: ConfigIO, damage: str
) -> None:
    if damage == "outside-state":
        config_io.audit = config_io.root / "foreign-audit.json"
        match = "audit is not bound"
    elif damage == "empty-audit":
        config_io.audit_content = {}
        match = "lacks its completed audit"
    else:
        config_io.pending_at.add("noop")
        match = "lacks its completed audit"
    with pytest.raises(ValueError, match=match):
        config_io.run()
    assert config_io.calls == ["dry-run", "apply", "noop", "restore"]
    assert load_desired_admin_config(config_io.root) == config_io.before
    assert config_io.audit.exists(), "failed proof erased the CLI's audit evidence"
    if damage == "pending":
        assert (config_io.root / "admin-config/pending.json").exists(), (
            "the witness must not erase a newly observed pending transaction"
        )


@pytest.mark.parametrize(
    "damage", ["status", "persisted-config", "template", "replicas", "gpu"]
)
def test_public_config_cannot_report_success_until_restore_is_observed(
    config_io: ConfigIO, damage: str
) -> None:
    if damage == "status":
        config_io.responses["restore"] = {"status": "PENDING"}
    elif damage == "persisted-config":
        config_io.persist["restore"] = False
    elif damage == "template":
        config_io.cpu_damage["restore"] = ("worker", "template_sha256", "foreign")
    elif damage == "replicas":
        config_io.cpu_damage["restore"] = ("worker", "replicas", 99)
    else:
        config_io.gpu_damage.add("restore")
    match = (
        "restore did not complete"
        if damage in {"status", "persisted-config"}
        else "restore the original runtime scope"
    )
    with pytest.raises(ValueError, match=match):
        config_io.run()
    assert config_io.calls == ["dry-run", "apply", "noop", "restore"]
    assert config_io.audit.is_file(), (
        "restore rejection discarded the application audit"
    )
    assert (
        load_admin_config_file(config_io.root / "admin-config.yaml") == config_io.before
    )


def test_public_config_roundtrip_keeps_untouched_roles_and_restores_the_original_config(
    config_io: ConfigIO,
) -> None:
    result = config_io.run()
    assert result == {
        "entrypoint": "gpu-fault-admin config",
        "applied_config_sha256": config_io.desired.sha256(),
        "affected_deployments": [CPU_ROLES["worker"]],
        "dry_run": True,
        "repeat_noop": True,
        "audit_present": True,
        "restored_config_sha256": config_io.before.sha256(),
    }
    assert config_io.calls == ["dry-run", "apply", "noop", "restore"]
    # Two applies, two admin-config stamps on every role: generation is not a
    # rollout signal, so untouched roles bump too and the drill must not count it.
    assert config_io.generations == {"worker": 3, "ingress": 3, "spool": 3}
    assert load_desired_admin_config(config_io.root) == config_io.before
    assert config_io.runtime == config_io.before
    assert not (config_io.root / "admin-config/pending.json").exists(), (
        "a completed roundtrip left a pending transaction"
    )


class ObservedEpochRelease(SimpleNamespace):
    def _load_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.observed_state)


class EpochIO:
    def __init__(self, root: Path) -> None:
        disposable, protected = root / "disposable", root / "protected"
        disposable.mkdir()
        protected.mkdir()
        self.site = site_file(disposable)
        self.protected = site_file(protected)
        document = yaml.safe_load(self.protected.read_text(encoding="utf-8"))
        prefix = "arn:aws:eks:us-east-1:123456789012:cluster/"
        document["spec"]["cpu"]["eksArn"] = prefix + "protected-cpu"
        document["spec"]["clusters"][0]["eksClusterArn"] = prefix + "protected-gpu"
        self.protected.write_text(yaml.safe_dump(document), encoding="utf-8")
        self.identities = {
            self.site: {"release_id": "disposable-release", "cluster_id": "gpu-a"},
            self.protected: {"release_id": "protected-release", "cluster_id": "gpu-a"},
        }
        self.previous = root / "boot023.json"
        write_json_atomic(
            self.previous,
            {
                "case_id": "GF-REGIONAL-BOOT-023",
                "verdict": "PASS",
                "status": "COMPLETED",
                **self.identities[self.protected],
            },
        )
        self.state = release_state_fixture()
        self.state.update(
            phase="complete",
            transaction_committed=True,
            release_id="disposable-release",
        )
        self.events: list[str] = []
        self.status_code = 0
        self.status_report = {
            "status": "HEALTHY",
            "checks": {"cpu": "READY", "gpu": "READY"},
        }
        self.publication = {
            "publication": {"map_sha256": "a" * 64, "registry_generation": 4},
            "deployments": {CPU_ROLES["worker"]: {"uid": "worker", "generation": 3}},
        }
        self.during_publication: Callable[[], None] | None = None

    def fixture(self, path: Path, cluster_id: str) -> SimpleNamespace:
        assert path in self.identities and cluster_id == "gpu-a", (
            "epoch observation escaped its two owned targets"
        )

        def identity() -> dict[str, str]:
            self.events.append("identity:" + path.parent.name)
            return dict(self.identities[path])

        return SimpleNamespace(regional=SimpleNamespace(evidence_identity=identity))

    def status(self, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
        assert args == ("status", "--state-dir", str(self.site.parent), "--full")
        assert timeout == 1800, "read-only status lost its bounded preflight budget"
        self.events.append("status")
        return subprocess.CompletedProcess(
            args,
            self.status_code,
            "owned status observation\n" + json.dumps(self.status_report, indent=2),
            "",
        )

    def release(
        self, config: ReleaseConfig, runner: rollout.Runner
    ) -> ObservedEpochRelease:
        assert config.namespace == load_site(self.site).release_config["namespace"]
        self.events.append("release-read")
        value = ObservedEpochRelease(**vars(release_diff_fixture(probes=self.events)))
        value.observed_state = self.state
        return value

    def membership(self, site: RenderedSite) -> dict[str, Any]:
        assert site.source == self.site, (
            "membership observation selected the protected fleet"
        )
        self.events.append("membership")
        if self.during_publication is not None:
            self.during_publication()
        return copy.deepcopy(self.publication)

    def produce(self) -> dict[str, Any]:
        return epoch.produce_epoch_receipt(self.site, self.protected, self.previous)

    def validate(self, receipt: dict[str, Any], *, now: datetime) -> None:
        epoch.validate_epoch_receipt(
            receipt,
            site_path=self.site,
            protected_path=self.protected,
            previous_path=self.previous,
            now=now,
        )


@pytest.fixture
def epoch_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EpochIO:
    value = EpochIO(tmp_path)
    monkeypatch.setattr(epoch, "SiteFixture", value.fixture)
    monkeypatch.setattr(epoch, "admin_command", value.status)
    monkeypatch.setattr(rollout, "RegionalRelease", value.release)
    monkeypatch.setattr(epoch, "membership_observation", value.membership)
    return value


def test_epoch_receipt_binds_both_targets_and_preserves_the_original_predecessor(
    epoch_io: EpochIO,
) -> None:
    predecessor = epoch_io.previous.read_bytes()
    report = epoch_io.produce()
    assert report["case_id"] == "GF-REGIONAL-BOOT-024"
    assert report["report_type"] == "epoch-prerequisite" and report["status"] == "READY"
    assert "verdict" not in report, (
        "preflight readiness must not become a BOOT-024 PASS"
    )
    assert report["target"] == epoch_io.identities[epoch_io.site]
    assert report["protected_target"] == epoch_io.identities[epoch_io.protected]
    assert report["scope"] == epoch.isolated_epoch_scope(
        load_site(epoch_io.site).release_config,
        load_site(epoch_io.protected).release_config,
    )
    assert (
        report["site_sha256"] == hashlib.sha256(epoch_io.site.read_bytes()).hexdigest()
    )
    assert (
        report["protected_site_sha256"]
        == hashlib.sha256(epoch_io.protected.read_bytes()).hexdigest()
    )
    assert (
        report["sequence_predecessor_sha256"] == hashlib.sha256(predecessor).hexdigest()
    )
    assert report["sequence_predecessor_case_id"] == "GF-REGIONAL-BOOT-023"
    assert (
        report["status_report_sha256"]
        == hashlib.sha256(
            json.dumps(epoch_io.status_report, sort_keys=True).encode()
        ).hexdigest()
    )
    assert report["membership_publication"] == epoch_io.publication
    assert epoch_io.previous.read_bytes() == predecessor, (
        "epoch admission relabelled or rewrote the protected predecessor"
    )
    assert "read-only-probe" in epoch_io.events, (
        "the real release classifier was bypassed"
    )
    assert epoch_io.events[-2:] == ["identity:disposable", "identity:protected"], (
        "preflight did not recheck both identities after observing membership"
    )
    epoch_io.validate(report, now=datetime.fromisoformat(report["observed_at"]))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("predecessor", "protected-site BOOT-023"),
        ("status", "not healthy"),
        ("release-diff", "committed NOOP baseline"),
        ("uncommitted", "committed NOOP baseline"),
        ("phase", "committed NOOP baseline"),
        ("release-id", "committed NOOP baseline"),
    ],
)
def test_epoch_preflight_refuses_bad_sequence_health_or_committed_baseline(
    epoch_io: EpochIO, damage: str, message: str
) -> None:
    if damage == "predecessor":
        value = json.loads(epoch_io.previous.read_text(encoding="utf-8"))
        value["release_id"] = epoch_io.identities[epoch_io.site]["release_id"]
        write_json_atomic(epoch_io.previous, value)
    elif damage == "status":
        epoch_io.status_code = 7
    else:
        field, value = {
            "release-diff": ("wheel_sha256", "f" * 64),
            "uncommitted": ("transaction_committed", False),
            "phase": ("phase", "failed"),
            "release-id": ("release_id", "other-release"),
        }[damage]
        epoch_io.state[field] = value
    with pytest.raises(ValueError, match=message):
        epoch_io.produce()
    assert "membership" not in epoch_io.events, (
        "a refused baseline reached later admission work"
    )
    if damage == "predecessor":
        assert "status" not in epoch_io.events, (
            "a foreign predecessor authorized target I/O"
        )


@pytest.mark.parametrize(
    "damage", ["site-file", "protected-file", "site-identity", "protected-identity"]
)
def test_epoch_preflight_detects_file_or_observed_identity_drift_during_admission(
    epoch_io: EpochIO, damage: str
) -> None:
    def drift() -> None:
        path = epoch_io.protected if damage.startswith("protected") else epoch_io.site
        if damage.endswith("file"):
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            epoch_io.identities[path]["release_id"] = "replacement-release"

    epoch_io.during_publication = drift
    with pytest.raises(ValueError, match="identity changed during preflight"):
        epoch_io.produce()
    assert "membership" in epoch_io.events, (
        "the test did not reach the final identity recheck"
    )


@pytest.mark.parametrize("age", [0, 900])
def test_epoch_validation_accepts_both_ends_of_its_readiness_window(
    epoch_io: EpochIO, age: int
) -> None:
    report = epoch_io.produce()
    observed = datetime.fromisoformat(report["observed_at"])
    epoch_io.validate(report, now=observed + timedelta(seconds=age))


@pytest.mark.parametrize(
    "damage",
    [
        "naive-time",
        "future",
        "expired",
        "schema",
        "case",
        "type",
        "status",
        "site-hash",
        "protected-hash",
        "predecessor-hash",
        "scope",
    ],
)
def test_epoch_validation_rejects_stale_or_rebound_readiness(
    epoch_io: EpochIO, damage: str
) -> None:
    report = epoch_io.produce()
    observed = datetime.fromisoformat(report["observed_at"])
    now = observed
    if damage == "naive-time":
        report["observed_at"] = observed.replace(tzinfo=None).isoformat()
    elif damage == "future":
        now = observed - timedelta(seconds=1)
    elif damage == "expired":
        now = observed + timedelta(seconds=901)
    else:
        field, value = {
            "schema": ("schema_version", 2),
            "case": ("case_id", "GF-REGIONAL-BOOT-023"),
            "type": ("report_type", "case-result"),
            "status": ("status", "PASS"),
            "site-hash": ("site_sha256", "f" * 64),
            "protected-hash": ("protected_site_sha256", "f" * 64),
            "predecessor-hash": ("sequence_predecessor_sha256", "f" * 64),
            "scope": ("scope", {}),
        }[damage]
        report[field] = value
    with pytest.raises(ValueError, match="stale or belongs to another target"):
        epoch_io.validate(report, now=now)


def test_epoch_cli_produces_and_checks_the_same_read_only_receipt(
    epoch_io: EpochIO,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = epoch_io.site.parent / "epoch.json"
    arguments = [
        "epoch-preflight",
        "--site",
        str(epoch_io.site),
        "--protected-site",
        str(epoch_io.protected),
        "--predecessor-evidence",
        str(epoch_io.previous),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    epoch.main()
    assert json.loads(capsys.readouterr().out) == {
        "case_id": epoch.CASE_ID,
        "prerequisite": "READY",
        "output": str(output),
    }
    before = output.read_bytes()
    calls = list(epoch_io.events)
    monkeypatch.setattr(sys, "argv", [*arguments, "--check"])
    epoch.main()
    assert json.loads(capsys.readouterr().out) == {
        "case_id": epoch.CASE_ID,
        "prerequisite": "READY",
    }
    assert output.read_bytes() == before, (
        "check rewrote or refreshed an existing receipt"
    )
    assert epoch_io.events == calls, (
        "receipt checking performed a fresh target observation"
    )
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("module", [lifecycle, epoch], ids=["store-lifecycle", "epoch"])
def test_lifecycle_script_entrypoints_enforce_their_required_target_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], module: Any
) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(module.__file__), run_name="__main__")
    assert raised.value.code == 2
    assert "required" in capsys.readouterr().err


def test_public_config_drill_uses_the_state_dirs_own_deploy_host_cli(
    tmp_path: Path,
) -> None:
    # Live 2026-09-20: a managed state dir carries its own deployer venv and the
    # checkout's module CLI refuses to act on it (rc 2); the drill must call the
    # binary that owns the dir. Without one, the module form stays (the witness
    # above pins that path).
    from scripts.e2e.regional import boot020_admin_config as admin_config

    assert admin_config.admin_config_command(tmp_path, "CHG-1", "--dry-run")[:4] == [
        sys.executable,
        "-m",
        "gpu_fault.admin.cli",
        "config",
    ]
    own = tmp_path / "deployer-venv" / "bin" / "gpu-fault-admin"
    own.parent.mkdir(parents=True)
    own.write_text("#!/bin/sh\n", encoding="utf-8")
    assert admin_config.admin_config_command(tmp_path, "CHG-1", "--dry-run") == [
        str(own),
        "config",
        "--state-dir",
        str(tmp_path),
        "--reference",
        "CHG-1",
        "--dry-run",
    ]


def test_inadmissible_desired_config_is_refused_before_the_editable_file_changes(
    config_io: ConfigIO,
) -> None:
    # Live 2026-09-20 (BOOT-020 a1): the drill wrote controlWorkerReplicas 7 into
    # admin-config.yaml, the administrator's dry-run refused it ("connection
    # ceiling 1276 exceeds the validated fleet budget 1200"), nothing was applied,
    # and the edit stayed behind -- unparseable by the guard that refused it. The
    # drill must admit its own desired config before editing anything.
    editable = config_io.root / "admin-config.yaml"
    before = editable.read_bytes()
    from dataclasses import replace

    config_io.desired = replace(
        config_io.desired,
        capacity=replace(config_io.desired.capacity, control_worker_replicas=64),
    )
    with pytest.raises(ValueError, match="not admissible"):
        config_io.run()
    assert editable.read_bytes() == before, (
        "an inadmissible drill must not edit the site config"
    )
    assert config_io.calls == [], (
        "an inadmissible drill must not reach the administrator"
    )
