#!/usr/bin/env python3
"""GF-REGIONAL-PREEMPT-036 acceptance runner: the dispatcher closes a stuck workflow.

Three production incidents each held a release for hours because the record that
blocked it had no close path: a workflow BLOCKED at compile time with no
executable owner, a FAILED workflow whose remote command stayed WAITING for
thirteen hours, and a re-planned-away generation that kept being dispatched. The
first fix was an operator command with a mode per shape; since 2026-09-08 the
dispatcher closes all three itself on ``WorkflowDispatcher.sweep_stuck_records``
before every scan, by Store predicate and never by age. This case is the
acceptance evidence for that sweep.

It is fully isolated and touches no cluster. Per shape it provisions a throwaway
store (a PostgreSQL 16 container when docker is available, otherwise a SQLite
file, recorded either way), seeds the stuck shape through Store APIs only, and
drives the product's own dispatcher -- built from its public constructors, with
no adapters -- against that store for the passes the shape needs, then one more
to show the sweep is idempotent. The release preflight verdict is measured the
same way the release engine measures it, by running the shipped
``workflow_safety`` and ``remote_command_stats`` probe sources.

Every store URL is one this process created. An ambient ``GPU_FAULT_STORE_URL``
is never used and a URL the provisioner does not own is refused, because a
runner that seeds stuck workflows must not be able to point at a real database.

Evidence: ``<run-dir>/cases/GF-REGIONAL-PREEMPT-036/GF-REGIONAL-PREEMPT-036.json``
with digests only -- no URLs, no credentials. Exit code is non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from gpu_fault.store import PostgresStore, SqliteStore  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    utc_now,
    write_json_atomic,
)
from scripts.e2e.regional.preempt036_verdicts import (  # noqa: E402
    CASE_ID,
    SHAPES,
    SWEEPER_EXECUTOR_ID,
    ShapeSeed,
    all_errors,
    case_verdict,
    command_errors,
    command_snapshot,
    event_errors,
    held_errors,
    incident_errors,
    incident_snapshot,
    open_command_errors,
    record_errors,
    rerun_errors,
    safety_errors,
    seed_shape,
    sweep,
    workflow_snapshot,
)

CASE_TITLE = (
    "卡死工作流由 dispatcher 自动关闭：compile-blocked、orphaned-commands、"
    "retired-generation 三种形态各自在清扫中收敛"
)
POSTGRES_IMAGE = "postgres:16"
POSTGRES_READY_ATTEMPTS = 60
# ``ApplicationContext.from_environment`` refuses to build an active control
# plane without these. The values are inert: no operation is allowed except
# evidence collection, and nothing in this case dispatches anything.
CONTEXT_ENVIRONMENT = {
    "GPU_FAULT_EXECUTOR_MODE": "active",
    "GPU_FAULT_ALLOW_SINGLE_CLUSTER": "true",
    "GPU_FAULT_ALLOWED_OPERATIONS": "FREEZE_EVIDENCE",
    "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
    "GPU_FAULT_EXECUTION_TOKEN": "0" * 32,
    "AWS_DEFAULT_REGION": "us-west-2",
}
_DATABASE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class RunnerError(RuntimeError):
    """A failure of the harness itself, not a verdict about the system."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class StoreProvisioner:
    """Hands out isolated stores and refuses any URL it did not create."""

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        self._owned: dict[str, str] = {}

    def _register(self, name: str, url: str) -> str:
        self._owned[url] = name
        return url

    def owns(self, url: str) -> bool:
        return url in self._owned

    def require_owned(self, url: str) -> str:
        if not self.owns(url):
            raise RunnerError(
                "refusing to use a store this run did not create; "
                f"owned stores: {sorted(self._owned.values())}"
            )
        return url

    def create(self, name: str) -> str:
        raise NotImplementedError

    def open(self, url: str) -> Any:
        raise NotImplementedError


class SqliteProvisioner(StoreProvisioner):
    def __init__(self, directory: Path, reason: str) -> None:
        super().__init__("sqlite", reason)
        self.directory = directory

    def create(self, name: str) -> str:
        path = self.directory / f"{name}-{uuid4().hex[:12]}.db"
        return self._register(name, f"sqlite:///{path}")

    def open(self, url: str) -> Any:
        self.require_owned(url)
        # The Store's own constructor is the schema ensure path; this runner
        # never writes DDL.
        return SqliteStore(url.removeprefix("sqlite:///"))


class PostgresProvisioner(StoreProvisioner):
    def __init__(self, base_url: str, container: str) -> None:
        super().__init__("postgres", f"docker {POSTGRES_IMAGE} container started")
        self.base_url = base_url
        self.container = container

    def create(self, name: str) -> str:
        import psycopg
        from psycopg import sql

        database = f"gpu_fault_p036_{name.replace('-', '_')}_{uuid4().hex[:8]}"
        if _DATABASE_NAME.fullmatch(database) is None:
            raise RunnerError(f"refusing an unexpected database name: {database}")
        with psycopg.connect(self.base_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                )
        head, _, _ = self.base_url.rpartition("/")
        return self._register(name, f"{head}/{database}")

    def open(self, url: str) -> Any:
        self.require_owned(url)
        # ``initialize_schema=True`` is the deployed schema ensure job's own
        # entry point. Reused rather than reimplemented: production DDL has
        # exactly one source.
        return PostgresStore(url, initialize_schema=True)


@contextmanager
def _provision(backend: str, workdir: Path) -> Iterator[StoreProvisioner]:
    """A PostgreSQL 16 container when one can be started, a SQLite file otherwise."""

    workdir.mkdir(parents=True, exist_ok=True)
    if backend == "sqlite":
        yield SqliteProvisioner(workdir, "requested with --backend sqlite")
        return
    docker = shutil.which("docker")
    if docker is None:
        if backend == "postgres":
            raise RunnerError("--backend postgres needs docker on PATH")
        yield SqliteProvisioner(workdir, "docker is not on PATH")
        return
    container = f"gpu-fault-preempt036-postgres-{os.getpid()}-{uuid4().hex[:8]}"
    started = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "-d",
            "--name",
            container,
            "-p",
            "127.0.0.1::5432",
            "-e",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            POSTGRES_IMAGE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode:
        detail = (started.stderr or started.stdout).strip()
        if backend == "postgres":
            raise RunnerError(
                f"--backend postgres could not start a container: {detail}"
            )
        yield SqliteProvisioner(workdir, f"docker run failed: {detail[:200]}")
        return
    try:
        for _attempt in range(POSTGRES_READY_ATTEMPTS):
            ready = subprocess.run(
                [docker, "exec", container, "pg_isready", "-U", "postgres"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RunnerError("throwaway PostgreSQL 16 never became ready")
        port = subprocess.run(
            [
                docker,
                "inspect",
                "-f",
                '{{(index (index .NetworkSettings.Ports "5432/tcp") 0).HostPort}}',
                container,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        yield PostgresProvisioner(
            f"postgresql://postgres@127.0.0.1:{port}/postgres",
            container,
        )
    finally:
        subprocess.run(
            [docker, "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _child_environment(provisioner: StoreProvisioner, url: str) -> dict[str, str]:
    provisioner.require_owned(url)
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_FAULT_STORE_URL": url,
        **CONTEXT_ENVIRONMENT,
    }


def _run_probe(provisioner: StoreProvisioner, url: str, source: str) -> dict[str, Any]:
    """Execute a shipped probe source against the isolated store, as the engine does."""

    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=_child_environment(provisioner, url),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RunnerError(
            f"shipped probe exited {completed.returncode}: "
            f"{completed.stderr.strip()[:600]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(
            f"shipped probe printed non-JSON: {completed.stdout[:200]!r}"
        ) from exc
    if not isinstance(value, dict):
        raise RunnerError("shipped probe printed a non-object result")
    return value


def _probe_source(name: str) -> str:
    """The shipped probe text, from the release engine's own registry."""

    from gpu_fault_release.regional_release_probes import probe_source

    return str(probe_source(name))


def _snapshots(store: Any, seed: ShapeSeed) -> dict[str, Mapping[str, Any]]:
    return {
        "workflows": workflow_snapshot(store, seed.workflow_ids),
        "incidents": incident_snapshot(store, seed.incident_ids),
        "commands": command_snapshot(store, seed.workflow_ids),
    }


def run_shape(
    shape: str,
    provisioner: StoreProvisioner,
    *,
    safety_probe: str,
    stats_probe: str,
) -> dict[str, Any]:
    """Seed one shape, sweep it, and return its evidence, verdict included."""

    url = provisioner.create(shape)
    store = provisioner.open(url)
    seed: ShapeSeed = seed_shape(shape, store)
    stages: dict[str, list[str]] = {}

    safety_before = _run_probe(provisioner, url, safety_probe)
    stats_before = _run_probe(provisioner, url, stats_probe)
    before = _snapshots(store, seed)

    held_per_pass = sweep(store, passes=seed.passes)
    settled = _snapshots(store, seed)
    stages["held"] = held_errors(held_per_pass, seed)
    stages["records"] = record_errors(before["workflows"], settled["workflows"], seed)
    stages["incidents"] = incident_errors(
        before["incidents"], settled["incidents"], seed
    )
    stages["audit_events"] = event_errors(settled["workflows"], seed)
    stages["commands"] = command_errors(before["commands"], settled["commands"], seed)

    safety_after = _run_probe(provisioner, url, safety_probe)
    stats_after = _run_probe(provisioner, url, stats_probe)
    stages["release_preflight"] = safety_errors(safety_before, safety_after, seed)
    stages["open_remote_commands"] = open_command_errors(
        stats_before, stats_after, seed
    )

    # One more pass than the shape needs: the sweep runs every tick in
    # production, so a shape it keeps re-writing would grow the audit forever.
    rerun_held = sweep(store, passes=1)
    rerun = _snapshots(store, seed)
    stages["rerun"] = rerun_errors(settled, rerun) + held_errors(
        [rerun_held[0]], ShapeSeed(**{**seed.__dict__, "passes": 1})
    )

    return {
        "shape": shape,
        "verdict": case_verdict(stages),
        "errors": all_errors(stages),
        "stages": {name: list(errors) for name, errors in stages.items()},
        "store_identity_sha256": _sha256(url),
        "sweeper_executor_id": SWEEPER_EXECUTOR_ID,
        "passes": seed.passes,
        "held_per_pass": [list(held) for held in held_per_pass],
        "seeded_workflow_ids": list(seed.workflow_ids),
        "closed_workflow_ids": list(seed.closed_ids),
        "untouched_workflow_ids": list(seed.untouched_ids),
        "statuses_before": {
            request_id: value["status"]
            for request_id, value in before["workflows"].items()
        },
        "statuses_after": {
            request_id: value["status"]
            for request_id, value in settled["workflows"].items()
        },
        "release_preflight": {
            "blockers_before": list(safety_before.get("blockers") or []),
            "blockers_after": list(safety_after.get("blockers") or []),
            "compile_blocked_before": list(safety_before.get("compile_blocked") or []),
            "compile_blocked_after": list(safety_after.get("compile_blocked") or []),
            "resolved_blocked": list(safety_before.get("resolved_blocked") or []),
            "open_remote_commands_before": stats_before.get("by_status"),
            "open_remote_commands_after": stats_after.get("by_status"),
        },
        # Every seeded record is still there after the sweep, by construction of
        # the snapshot (a missing row raises); recorded so the evidence says so.
        "records_deleted": len(before["workflows"]) - len(settled["workflows"]),
        "audit_events": {
            request_id: value["events"]
            for request_id, value in settled["workflows"].items()
        },
        "incident_reasons": {
            incident_id: value["reasons"]
            for incident_id, value in settled["incidents"].items()
        },
    }


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GF-REGIONAL-PREEMPT-036: prove the dispatcher's sweep closes each "
            "stuck workflow shape, in an isolated store, with no cluster access"
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="acceptance run directory; evidence is written under cases/<case id>/",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "postgres", "sqlite"),
        default="auto",
        help=(
            "auto (default) uses a throwaway PostgreSQL 16 container and falls "
            "back to SQLite when docker cannot start one"
        ),
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=SHAPES,
        default=[],
        help="stuck-workflow shapes to run; every shape by default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arguments().parse_args(argv)
    shapes = tuple(arguments.shape) or SHAPES
    run_dir: Path = arguments.run_dir
    evidence_path = run_dir / "cases" / CASE_ID / f"{CASE_ID}.json"
    started_at = utc_now()
    workdir = run_dir / "work" / CASE_ID
    document: dict[str, Any] = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "title": CASE_TITLE,
        "verdict": "FAIL",
        "started_at": started_at,
        "completed_at": None,
        "shapes_requested": list(shapes),
        "store_backend": None,
        "store_backend_reason": None,
        "shapes": {},
        "errors": [],
    }
    try:
        with _provision(arguments.backend, workdir) as provisioner:
            document["store_backend"] = provisioner.backend
            document["store_backend_reason"] = provisioner.reason
            safety_probe = _probe_source("workflow_safety")
            stats_probe = _probe_source("remote_command_stats")
            document["release_probe_sha256"] = {
                "workflow_safety": _sha256(safety_probe),
                "remote_command_stats": _sha256(stats_probe),
            }
            for shape in shapes:
                document["shapes"][shape] = run_shape(
                    shape,
                    provisioner,
                    safety_probe=safety_probe,
                    stats_probe=stats_probe,
                )
                document["completed_at"] = utc_now()
                write_json_atomic(evidence_path, document)
    except Exception as exc:  # noqa: BLE001 -- a harness failure is a FAIL, recorded
        document["errors"].append(f"{type(exc).__name__}: {exc}")
        document["completed_at"] = utc_now()
        write_json_atomic(evidence_path, document)
        print(json.dumps({"case_id": CASE_ID, "verdict": "FAIL"}, sort_keys=True))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    document["errors"] = [
        f"{shape}: {error}"
        for shape, value in sorted(document["shapes"].items())
        for error in value["errors"]
    ]
    document["verdict"] = (
        "PASS"
        if not document["errors"]
        and all(value["verdict"] == "PASS" for value in document["shapes"].values())
        else "FAIL"
    )
    document["completed_at"] = utc_now()
    write_json_atomic(evidence_path, document)
    print(
        json.dumps(
            {
                "case_id": CASE_ID,
                "verdict": document["verdict"],
                "store_backend": document["store_backend"],
                "evidence": str(evidence_path),
            },
            sort_keys=True,
        )
    )
    for error in document["errors"]:
        print(error, file=sys.stderr)
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
