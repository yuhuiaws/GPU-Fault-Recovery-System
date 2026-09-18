"""Inert startup, explicit ARM, schema preparation, and isolated runtime probe."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from scripts.e2e.regional.probes.notify008_identity import runtime_identity
from scripts.e2e.regional.probes.notify008_postgres import (
    PostgresFactory,
    connect,
    database_identity,
    prepare_schema,
    reject_ambient_database_configuration,
)
from scripts.e2e.regional.probes.notify008_process import Budget, exercise
from scripts.e2e.regional.probes.notify008_protocol import (
    CASE_ID,
    SCOPE,
    ProbeError,
    Target,
    digest,
    report_errors,
)

CASE_ROOT = Path("/case")
CONTROL = Path("/control")
WORK = Path("/work")
UID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SAFE_ENVIRONMENT = {
    "AWS_CONFIG_FILE": "/dev/null",
    "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
    "AWS_EC2_METADATA_DISABLED": "true",
    "KUBECONFIG": "/dev/null",
    "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
}
PROBE_COMMANDS = ("idle", "inspect", "arm", "prepare", "run", "stop")
FAILURE_STAGES = frozenset(
    (
        *PROBE_COMMANDS,
        "context",
        "environment",
        "configuration",
        "pod-identity",
        "runtime-identity",
        "arm-guard",
        "database-connect",
        "database-identity",
        "schema-prepare",
        "exercise",
        "report-validation",
        "unknown",
    )
)
ERROR_TYPES = frozenset(
    {
        "ProbeError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "KeyError",
        "OSError",
        "FileNotFoundError",
        "FileExistsError",
        "PermissionError",
        "TimeoutError",
        "JSONDecodeError",
        "OperationalError",
        "InterfaceError",
        "ImportError",
        "ModuleNotFoundError",
        "RegionalCommandFailed",
        "RegionalCommandTimeout",
        "RegionalFixtureError",
        "ProcessSupervisionLost",
        "KeyboardInterrupt",
        "SystemExit",
        "Exception",
    }
)
FAILURE_REASONS = {
    "ambient database configuration is forbidden in the sandbox": "ambient-database-configuration",
    "sandbox inherited a cloud, credential or application environment": "ambient-authority",
    **{
        f"sandbox environment differs at {name}": "environment-mismatch"
        for name in SAFE_ENVIRONMENT
    },
    "sandbox Pod UID is unavailable": "pod-identity-unavailable",
    "sandbox configuration is oversized": "configuration-oversized",
    "sandbox configuration shape differs": "configuration-shape",
    "shipped probe bundle changed": "bundle-mismatch",
    "sandbox execution identity differs": "execution-identity-mismatch",
    "sandbox runtime differs from the approved deployed component": "runtime-identity-mismatch",
    "sandbox control path is a symlink": "control-path-symlink",
    "a stopped sandbox cannot be rearmed": "already-stopped",
    "sandbox ARM marker differs": "arm-marker-mismatch",
    "sandbox is not armed for this run": "not-armed",
    "isolated process budget expired": "process-budget-expired",
    "separate schema preparation has not completed": "schema-not-prepared",
    "sandbox PostgreSQL must be version 16": "database-version-mismatch",
    "database is not the fixed Unix-socket sandbox": "database-identity-mismatch",
    "NOTIFY008 refuses Aurora or an unknown backend": "database-backend-refused",
    "sandbox database owner differs": "database-owner-mismatch",
    "schema preparation refuses a nonempty database": "database-not-empty",
}
REASON_CODES = frozenset(
    (*FAILURE_REASONS.values(), "report-contract-failed", "unclassified-error")
)


def safe_error_type(error: BaseException) -> str:
    name = type(error).__name__
    return name if name in ERROR_TYPES else "Exception"


def safe_failure_fields(value: Any) -> dict[str, str]:
    """Accept only fixed diagnostic vocabulary, never exception text or logs."""
    value = value if isinstance(value, dict) else {}
    result = {}
    for key, allowed, fallback in (
        ("command", PROBE_COMMANDS, "unknown"),
        ("stage", FAILURE_STAGES, "unknown"),
        ("error_type", ERROR_TYPES, "Exception"),
        ("reason", REASON_CODES, "unclassified-error"),
    ):
        item = value.get(key)
        result[key] = item if isinstance(item, str) and item in allowed else fallback
    return result


def failure_fields(error: BaseException, command: str, stage: str) -> dict[str, str]:
    # Code identities expose the failing guard without traceback paths or locals.
    stages = {
        safe_environment.__code__: "environment",
        load_config.__code__: "configuration",
        pod_uid.__code__: "pod-identity",
        runtime_identity.__code__: "runtime-identity",
        require_armed.__code__: "arm-guard",
        connect.__code__: "database-connect",
        database_identity.__code__: "database-identity",
        prepare_schema.__code__: "schema-prepare",
        exercise.__code__: "exercise",
    }
    frame = error.__traceback__
    while frame is not None:
        stage = stages.get(frame.tb_frame.f_code, stage)
        frame = frame.tb_next
    message = error.args[0] if type(error) is ProbeError and error.args else None
    reason = (
        FAILURE_REASONS.get(message, "unclassified-error")
        if isinstance(message, str)
        else "unclassified-error"
    )
    return safe_failure_fields(
        {
            "command": command,
            "stage": stage,
            "error_type": safe_error_type(error),
            "reason": reason,
        }
    )


def safe_environment() -> None:
    reject_ambient_database_configuration()
    for name, value in SAFE_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise ProbeError(f"sandbox environment differs at {name}")
    if any(
        name.startswith(("AWS_", "GPU_FAULT_", "KUBE"))
        and name not in SAFE_ENVIRONMENT
        and name
        not in {
            "KUBERNETES_SERVICE_HOST",
            "KUBERNETES_SERVICE_PORT",
            "KUBERNETES_SERVICE_PORT_HTTPS",
            "KUBERNETES_PORT",
            "KUBERNETES_PORT_443_TCP",
            "KUBERNETES_PORT_443_TCP_ADDR",
            "KUBERNETES_PORT_443_TCP_PORT",
            "KUBERNETES_PORT_443_TCP_PROTO",
        }
        for name in os.environ
    ):
        raise ProbeError(
            "sandbox inherited a cloud, credential or application environment"
        )


def pod_uid() -> str:
    value = os.environ.get("NOTIFY008_POD_UID", "")
    if UID.fullmatch(value) is None:
        raise ProbeError("sandbox Pod UID is unavailable")
    return value


def marker(name: str) -> Path:
    return CONTROL / f"{name}-{pod_uid()}"


def write_private(path: Path, value: str) -> None:
    if path.is_symlink():
        raise ProbeError("sandbox control path is a symlink")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config() -> tuple[Target, dict[str, Any]]:
    path = CASE_ROOT / "config.json"
    if path.stat().st_size > 16384:
        raise ProbeError("sandbox configuration is oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "target",
        "namespace_uid",
        "source_sha256",
    }:
        raise ProbeError("sandbox configuration shape differs")
    target = Target(**value["target"])
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (CASE_ROOT / "scripts/e2e/regional/probes").glob("notify008_*.py")
    }
    sources["postgres-start"] = (CASE_ROOT / "postgres-start").read_text(
        encoding="utf-8"
    )
    if digest(sources) != value["source_sha256"]:
        raise ProbeError("shipped probe bundle changed")
    return target, value


def checked_context(arguments: argparse.Namespace) -> tuple[Target, dict[str, Any]]:
    safe_environment()
    target, config = load_config()
    if (
        arguments.expected_pod_uid != pod_uid()
        or arguments.expected_namespace_uid != config["namespace_uid"]
        or not arguments.expected_namespace_uid
        or arguments.bundle_sha256 != config["source_sha256"]
        or os.environ.get("NOTIFY008_RUN_ID") != target.run_id
    ):
        raise ProbeError("sandbox execution identity differs")
    identity = runtime_identity()
    if identity != {
        "distribution": "gpu-fault-control-plane",
        "version": target.runtime_version,
        "module_digest": target.runtime_module_digest,
    }:
        raise ProbeError("sandbox runtime differs from the approved deployed component")
    return target, identity


def inspect_context(target: Target, identity: dict[str, Any]) -> dict[str, Any]:
    arm = marker("arm")
    armed = arm.is_file() and not arm.is_symlink() and arm.read_text() == target.run_id
    return {
        "pod_uid": pod_uid(),
        "run_id": target.run_id,
        "runtime_identity": identity,
        "armed": armed,
        "stop_requested": marker("stop").exists(),
    }


def idle() -> None:
    safe_environment()
    pod_uid()
    seconds = int(os.environ.get("NOTIFY008_SECONDS", "0"))
    budget = Budget(seconds)
    while not marker("stop").exists():
        remaining = budget.remaining()
        time.sleep(min(0.2, remaining))


def arm(target: Target) -> None:
    if marker("stop").exists():
        raise ProbeError("a stopped sandbox cannot be rearmed")
    path = marker("arm")
    if path.exists():
        if path.is_symlink() or path.read_text() != target.run_id:
            raise ProbeError("sandbox ARM marker differs")
        return
    write_private(path, target.run_id)


def require_armed(target: Target) -> None:
    if (
        marker("stop").exists()
        or not marker("arm").is_file()
        or marker("arm").is_symlink()
        or marker("arm").read_text() != target.run_id
    ):
        raise ProbeError("sandbox is not armed for this run")


def prepare(target: Target) -> dict[str, Any]:
    import psycopg

    require_armed(target)
    with (WORK / "prepare-started").open("x"):
        pass
    budget = Budget(60)
    while True:
        require_armed(target)
        try:
            with connect() as connection:
                database_identity(connection)
            break
        except psycopg.OperationalError:
            time.sleep(min(0.2, budget.remaining()))
    facts = prepare_schema(target.run_id)
    write_private(WORK / "prepared.json", json.dumps(facts, sort_keys=True))
    return facts


def run(target: Target, identity: dict[str, Any]) -> dict[str, Any]:
    require_armed(target)
    if not (WORK / "prepared.json").is_file():
        raise ProbeError("separate schema preparation has not completed")
    root = WORK / "runtime"
    root.mkdir(mode=0o700, exist_ok=False)
    with connect() as connection:
        facts = database_identity(connection)
    result = {
        "case_id": CASE_ID,
        "run_id": target.run_id,
        "validation_scope": SCOPE,
        "pod_uid": pod_uid(),
        "runtime_identity": identity,
        "exactly_once_proven": False,
        **facts,
        **exercise(
            PostgresFactory(target.run_id),
            root,
            target.run_id,
            seconds=target.seconds - 60,
        ),
    }
    errors = report_errors(result, run_id=target.run_id)
    result.update(verdict="FAIL" if errors else "PASS", errors=errors)
    if errors:
        result.update(
            command="run",
            stage="report-validation",
            error_type="ProbeError",
            reason="report-contract-failed",
        )
    write_private(WORK / "result.json", json.dumps(result, sort_keys=True))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=PROBE_COMMANDS)
    value.add_argument("--expected-pod-uid", default="")
    value.add_argument("--expected-namespace-uid", default="")
    value.add_argument("--bundle-sha256", default="")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    os.umask(0o077)
    stage = "idle" if arguments.command == "idle" else "context"
    try:
        if arguments.command == "idle":
            idle()
            return 0
        target, identity = checked_context(arguments)
        stage = arguments.command
        if arguments.command == "arm":
            arm(target)
        elif arguments.command == "stop":
            write_private(marker("stop"), target.run_id)
        elif arguments.command == "prepare":
            print(json.dumps(prepare(target), sort_keys=True))
            return 0
        elif arguments.command == "run":
            result = run(target, identity)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["verdict"] == "PASS" else 1
        print(json.dumps(inspect_context(target, identity), sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "case_id": CASE_ID,
                    "verdict": "FAIL",
                    **failure_fields(exc, arguments.command, stage),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
