#!/usr/bin/env python3
"""GF-REGIONAL-PREEMPT-036 acceptance runner: an operator closes a stuck workflow.

Three production incidents each held a release for hours because the record that
blocked it had no close path: a workflow BLOCKED at compile time with no
executable owner, a FAILED workflow whose remote command stayed WAITING for
thirteen hours, and a re-planned-away generation that kept being dispatched. The
fixes shipped as ``gpu-fault-admin workflow-reconcile --mode
{compile-blocked,orphaned-commands,retired-generation} --plan/--apply``. This
case is the acceptance evidence for that operator path.

It is fully isolated and touches no cluster. Per mode it provisions a throwaway
store (a PostgreSQL 16 container when docker is available, otherwise a SQLite
file, recorded either way), seeds the stuck shape through Store APIs only, and
then drives the *shipped* reconcile entry points: the admin layer sends each
mode's module source plus a stdin/stdout driver into the CPU ingress Pod, and
this runner executes that exact text in a subprocess bound to the isolated
store. The release preflight verdict is measured the same way, by running the
shipped ``workflow_safety`` and ``remote_command_stats`` probe sources.

Every store URL is one this process created. An ambient ``GPU_FAULT_STORE_URL``
is never used and a URL the provisioner does not own is refused, because a
runner that seeds stuck workflows must not be able to point at a real database.

Evidence: ``<run-dir>/cases/GF-REGIONAL-PREEMPT-036/GF-REGIONAL-PREEMPT-036.json``
with digests only -- no URLs, no credentials. Exit code is non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from gpu_fault.admin.command_log import command_log  # noqa: E402
from gpu_fault.admin.operator_identity import (  # noqa: E402
    local_operator_identity,
    resolve_operator_identity,
)
from gpu_fault.admin.workflow_reconcile import (  # noqa: E402
    compile_blocked_script,
    orphaned_commands_script,
    retired_generation_script,
)
from gpu_fault.store import PostgresStore, SqliteStore  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    utc_now,
    write_json_atomic,
)
from scripts.e2e.regional.preempt036_verdicts import (  # noqa: E402
    CASE_ID,
    COMPILE_BLOCKED_MODE,
    DRIFT_REFUSALS,
    INELIGIBLE_REFUSALS,
    MODES,
    ORPHANED_COMMANDS_MODE,
    REFERENCE,
    RERUN_REFUSED,
    RETIRED_GENERATION_MODE,
    ModeSeed,
    all_errors,
    apply_errors,
    case_verdict,
    command_errors,
    command_log_errors,
    command_snapshot,
    event_errors,
    open_command_errors,
    plan_errors,
    record_errors,
    refusal_errors,
    rerun_errors,
    rerun_event_errors,
    safety_errors,
    seed_mode,
    workflow_snapshot,
)

CASE_TITLE = (
    "卡死工作流由运维审计关闭：compile-blocked、orphaned-commands、"
    "retired-generation 三种 workflow-reconcile 模式各自 plan→apply"
)
PROBE_DIR = ROOT / "deploy" / "control-plane" / "regional"
PROBE_REGISTRY = PROBE_DIR / "regional_release_probes.py"
POSTGRES_IMAGE = "postgres:16"
POSTGRES_READY_ATTEMPTS = 60
TAMPERED_DIGEST = "0" * 64
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
SCRIPT_BUILDERS = {
    COMPILE_BLOCKED_MODE: compile_blocked_script,
    ORPHANED_COMMANDS_MODE: orphaned_commands_script,
    RETIRED_GENERATION_MODE: retired_generation_script,
}
SCRIPT_MODULES = {
    COMPILE_BLOCKED_MODE: "gpu_fault.admin.compile_blocked",
    ORPHANED_COMMANDS_MODE: "gpu_fault.admin.orphaned_commands",
    RETIRED_GENERATION_MODE: "gpu_fault.retired_generation",
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

        database = f"gpu_fault_p035_{name.replace('-', '_')}_{uuid4().hex[:8]}"
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
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_FAULT_STORE_URL": url,
        **CONTEXT_ENVIRONMENT,
    }
    return environment


def _run_source(
    provisioner: StoreProvisioner,
    url: str,
    source: str,
    payload: Mapping[str, Any] | None,
) -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        input=None if payload is None else json.dumps(payload, separators=(",", ":")),
        env=_child_environment(provisioner, url),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _run_json(
    provisioner: StoreProvisioner,
    url: str,
    source: str,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    code, out, err = _run_source(provisioner, url, source, payload)
    if code:
        raise RunnerError(f"shipped source exited {code}: {err.strip()[:600]}")
    try:
        value = json.loads(out)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"shipped source printed non-JSON: {out[:200]!r}") from exc
    if not isinstance(value, dict):
        raise RunnerError("shipped source printed a non-object result")
    return value


def _expect_refusal(
    provisioner: StoreProvisioner,
    url: str,
    source: str,
    payload: Mapping[str, Any],
) -> str:
    code, out, err = _run_source(provisioner, url, source, payload)
    if code == 0:
        raise RunnerError(f"an apply that had to be refused succeeded: {out[:300]!r}")
    return err


def _probe_source(name: str) -> str:
    spec = importlib.util.spec_from_file_location(
        "gpu_fault_preempt036_probe_registry",
        PROBE_REGISTRY,
    )
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot load the probe registry at {PROBE_REGISTRY}")
    module = importlib.util.module_from_spec(spec)
    # The probe registry lives under deploy/, which the deploy-layout gate
    # requires to stay free of Python cache artifacts; load it without
    # writing bytecode next to it.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return str(module.probe_source(name))


def _module_source(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RunnerError(f"cannot locate {module_name}")
    return Path(spec.origin).read_text(encoding="utf-8")


def _shipped_evidence(mode: str, script: str) -> dict[str, Any]:
    """That what this runner executed is the text the admin layer ships."""

    source = _module_source(SCRIPT_MODULES[mode])
    return {
        "script_sha256": _sha256(script),
        "module_sha256": _sha256(source),
        "starts_with_module_source": script.startswith(source),
        "driver_sha256": _sha256(script[len(source) :]),
    }


def _safety(provisioner: StoreProvisioner, url: str, source: str) -> dict[str, Any]:
    return _run_json(provisioner, url, source, None)


def admin_plan_digest(plan: Mapping[str, Any]) -> str:
    """The admin-side digest of a plan document, as the CLI's apply payload carries."""

    return _sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")))


def apply_payload(
    workflow_ids: Sequence[str],
    *,
    plan_sha256: str,
    actor: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """What ``gpu-fault-admin workflow-reconcile --apply`` sends into the Pod (I1)."""

    return {
        "mode": "apply",
        "workflow_ids": list(workflow_ids),
        "plan_sha256": plan_sha256,
        "reference": REFERENCE,
        "actor": actor,
        "admin_plan_sha256": admin_plan_digest(plan),
    }


def run_mode(
    mode: str,
    provisioner: StoreProvisioner,
    *,
    safety_probe: str,
    stats_probe: str,
    actor: str,
    state_dir: Path,
) -> dict[str, Any]:
    """Drive one mode end to end and return its evidence, verdict included."""

    url = provisioner.create(mode)
    store = provisioner.open(url)
    seed: ModeSeed = seed_mode(mode, store)
    script = SCRIPT_BUILDERS[mode]()
    stages: dict[str, list[str]] = {}

    safety_before = _safety(provisioner, url, safety_probe)
    stats_before = _run_json(provisioner, url, stats_probe, None)

    full_plan = _run_json(
        provisioner,
        url,
        script,
        {"mode": "plan", "workflow_ids": list(seed.workflow_ids)},
    )
    stages["plan"] = plan_errors(full_plan, seed)

    ineligible_output = _expect_refusal(
        provisioner,
        url,
        script,
        apply_payload(
            seed.workflow_ids,
            plan_sha256=str(full_plan["plan_sha256"]),
            actor=actor,
            plan=full_plan,
        ),
    )
    stages["refuse_ineligible"] = refusal_errors(
        "ineligible",
        ineligible_output,
        INELIGIBLE_REFUSALS[mode],
    ) + [
        error
        for request_id in seed.refused_ids
        for error in refusal_errors(
            request_id,
            ineligible_output,
            seed.refusal_substrings[request_id],
        )
    ]

    approved = _run_json(
        provisioner,
        url,
        script,
        {"mode": "plan", "workflow_ids": list(seed.actionable_ids)},
    )
    approved_digest = str(approved["plan_sha256"])
    tampered_output = _expect_refusal(
        provisioner,
        url,
        script,
        apply_payload(
            seed.actionable_ids,
            plan_sha256=TAMPERED_DIGEST,
            actor=actor,
            plan=approved,
        ),
    )
    stages["refuse_tampered_plan"] = refusal_errors(
        "tampered plan",
        tampered_output,
        DRIFT_REFUSALS[mode],
    )

    commands_before = command_snapshot(store, seed.workflow_ids)
    approved_payload = apply_payload(
        seed.actionable_ids, plan_sha256=approved_digest, actor=actor, plan=approved
    )
    # The admin CLI wraps every mutating command in ``command_log`` (I3); the
    # same context is opened here around the apply, into this run's own state
    # directory, so the evidence shows where the console record lands.
    with command_log(
        state_dir, command=f"workflow-reconcile --mode {mode} --apply", kind="mutating"
    ) as log_path:
        result = _run_json(provisioner, url, script, approved_payload)
    stages["command_log"] = command_log_errors(log_path, state_dir=state_dir)
    stages["apply"] = apply_errors(result, seed, approved_plan_sha256=approved_digest)
    records_after = workflow_snapshot(store, seed.statuses_after)
    stages["records"] = record_errors(records_after, seed)
    stages["operator_events"] = event_errors(
        records_after,
        seed,
        actor=actor,
        admin_plan_sha256=str(approved_payload["admin_plan_sha256"]),
        approved_plan_sha256=approved_digest,
    )
    commands_after = command_snapshot(store, seed.workflow_ids)
    stages["commands"] = command_errors(commands_before, commands_after, seed)

    safety_after = _safety(provisioner, url, safety_probe)
    stats_after = _run_json(provisioner, url, stats_probe, None)
    stages["release_preflight"] = safety_errors(safety_before, safety_after, seed)
    stages["open_remote_commands"] = open_command_errors(
        stats_before,
        stats_after,
        seed,
    )

    rerun_plan = _run_json(
        provisioner,
        url,
        script,
        {"mode": "plan", "workflow_ids": list(seed.actionable_ids)},
    )
    rerun_payload = apply_payload(
        seed.actionable_ids,
        plan_sha256=str(rerun_plan["plan_sha256"]),
        actor=actor,
        plan=rerun_plan,
    )
    rerun_output = ""
    rerun_result: dict[str, Any] | None = None
    if seed.rerun_contract == RERUN_REFUSED:
        rerun_output = _expect_refusal(provisioner, url, script, rerun_payload)
    else:
        rerun_result = _run_json(provisioner, url, script, rerun_payload)
    stages["rerun"] = rerun_errors(
        seed,
        plan=rerun_plan,
        result=rerun_result,
        output=rerun_output,
    )
    records_after_rerun = workflow_snapshot(store, seed.statuses_after)
    stages["untouched_by_rerun"] = [
        *record_errors(records_after_rerun, seed),
        *rerun_event_errors(records_after, records_after_rerun, seed),
    ]

    return {
        "mode": mode,
        "verdict": case_verdict(stages),
        "errors": all_errors(stages),
        "stages": {name: list(errors) for name, errors in stages.items()},
        "store_identity_sha256": _sha256(url),
        "shipped_source": _shipped_evidence(mode, script),
        "seeded_workflow_ids": list(seed.workflow_ids),
        "applied_workflow_ids": list(seed.actionable_ids),
        "refused_workflow_ids": list(seed.refused_ids),
        "plan_sha256": {
            "full": str(full_plan["plan_sha256"]),
            "approved": approved_digest,
            "settled": str(result.get("settled_plan_sha256") or ""),
            "rerun": str(rerun_plan["plan_sha256"]),
        },
        "release_preflight": {
            "blockers_before": list(safety_before.get("blockers") or []),
            "blockers_after": list(safety_after.get("blockers") or []),
            "resolved_blocked": list(safety_before.get("resolved_blocked") or []),
            "open_remote_commands_before": stats_before.get("by_status"),
            "open_remote_commands_after": stats_after.get("by_status"),
        },
        "records_deleted": result.get("records_deleted"),
        "rerun_contract": seed.rerun_contract,
        "actor_sha256": _sha256(actor),
        "admin_plan_sha256": approved_payload["admin_plan_sha256"],
        "operator_events": {
            request_id: (records_after.get(request_id) or {}).get("events")
            for request_id in seed.workflow_ids
        },
        "command_log_sha256": (
            _sha256(Path(log_path).read_text(encoding="utf-8", errors="replace"))
            if log_path is not None and Path(log_path).is_file()
            else None
        ),
    }


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GF-REGIONAL-PREEMPT-036: prove workflow-reconcile closes each stuck "
            "workflow shape, in an isolated store, with no cluster access"
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
        "--mode",
        action="append",
        choices=MODES,
        default=[],
        help="reconcile modes to run; every mode by default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arguments().parse_args(argv)
    modes = tuple(arguments.mode) or MODES
    run_dir: Path = arguments.run_dir
    evidence_path = run_dir / "cases" / CASE_ID / f"{CASE_ID}.json"
    started_at = utc_now()
    workdir = run_dir / "work" / CASE_ID
    document: dict[str, Any] = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "title": CASE_TITLE,
        "verdict": "FAIL",
        "started_at": started_at,
        "completed_at": None,
        "reference": REFERENCE,
        "reconcile_modes": list(modes),
        "store_backend": None,
        "store_backend_reason": None,
        "modes": {},
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
            # The identity the admin CLI would resolve: the STS caller ARN when
            # the AWS CLI answers, else user@host -- never ``unknown-identity``
            # while there is a local identity to fall back to (I1). Only its
            # digest is recorded.
            actor = resolve_operator_identity(fallback=local_operator_identity())
            document["actor_sha256"] = _sha256(actor)
            state_dir = workdir / "admin-state"
            state_dir.mkdir(parents=True, exist_ok=True)
            for mode in modes:
                document["modes"][mode] = run_mode(
                    mode,
                    provisioner,
                    safety_probe=safety_probe,
                    stats_probe=stats_probe,
                    actor=actor,
                    state_dir=state_dir,
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
        f"{mode}: {error}"
        for mode, value in sorted(document["modes"].items())
        for error in value["errors"]
    ]
    document["verdict"] = (
        "PASS"
        if not document["errors"]
        and all(value["verdict"] == "PASS" for value in document["modes"].values())
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
