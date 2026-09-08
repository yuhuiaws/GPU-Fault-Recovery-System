from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.container_env_snapshot import (
    ROLE_DEPLOYMENTS,
    container_env_differences,
    load_container_env_snapshot,
    pod_container_env,
)
from gpu_fault.env import env_bool
from gpu_fault.env_validation import (
    environment_value_kinds,
    invalid_gpu_fault_environment_values,
    unknown_gpu_fault_environment,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)

SENSITIVE_ENV = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY)")
UVICORN_WORKERS = re.compile(r"--workers\s+(\d+)")
GRACEFUL_SECONDS = re.compile(r"--timeout-graceful-shutdown\s+(\d+)")
SLEEP_SECONDS = re.compile(r"^sleep\s+(\d+)$")
PROCESSOR_POOL_ENV = {
    "GPU_FAULT_PROCESSOR_WORKERS",
    "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
    "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
    "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
    "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
}
# The validators that encode the *current* release's contract. A verbatim
# rollback render carries the previous release's env, so these are the wrong
# rules for it and snapshot mode skips them by name (see validate()).
CURRENT_CONTRACT_VALIDATORS = (
    "validate_environment_inventory",
    "validate_role",
    "validate_shutdown",
    "validate_operation_allowlists",
    "validate_connections",
)


def manifest_paths(values: list[str]) -> list[Path]:
    if not values:
        values = ["deploy/control-plane/regional/generated"]
    result = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            result.extend(sorted(path.glob("*.yaml")))
        else:
            result.append(path)
    return result


def documents(paths: list[Path]) -> list[dict]:
    result = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"manifest does not exist: {path}")
        for item in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if isinstance(item, dict):
                result.append(item)
    return result


def container(deployment: dict) -> dict:
    values = deployment["spec"]["template"]["spec"]["containers"]
    if len(values) != 1:
        raise ValueError(f"{deployment['metadata']['name']} must have one container")
    return values[0]


def resolve_environment(
    deployment: dict,
    config_maps: dict[str, dict[str, str]],
) -> tuple[dict[str, str | None], list[str]]:
    name = deployment["metadata"]["name"]
    values: dict[str, str | None] = {}
    errors = []
    item = container(deployment)
    for source in item.get("envFrom", []):
        reference = source.get("configMapRef")
        if not reference:
            continue
        config_name = reference.get("name")
        if config_name not in config_maps:
            errors.append(f"{name}: envFrom ConfigMap {config_name} is missing")
            continue
        for key, value in config_maps[config_name].items():
            if key in values:
                errors.append(f"{name}: duplicate envFrom key {key}")
            values[key] = value
    for entry in item.get("env", []):
        key = entry.get("name")
        if not key:
            errors.append(f"{name}: env entry has no name")
            continue
        has_value = "value" in entry
        has_source = "valueFrom" in entry
        if has_value == has_source:
            errors.append(
                f"{name}: {key} must define exactly one of value or valueFrom"
            )
            continue
        if key in values:
            errors.append(f"{name}: explicit env {key} duplicates envFrom")
        if has_value:
            value = str(entry["value"])
            if SENSITIVE_ENV.search(key):
                errors.append(f"{name}: sensitive env {key} must use valueFrom")
            values[key] = value
            continue
        source = entry["valueFrom"]
        reference = source.get("configMapKeyRef")
        if reference and reference.get("name") in config_maps:
            values[key] = config_maps[reference["name"]].get(reference.get("key"))
        else:
            values[key] = "<valueFrom>"
    return values, errors


def integer(
    values: dict[str, str | None],
    name: str,
    default: int,
) -> int:
    raw = values.get(name)
    if raw in {None, "<valueFrom>"}:
        return default
    return int(raw)


def validate_shutdown(
    deployment: dict,
    values: dict[str, str | None],
) -> list[str]:
    name = deployment["metadata"]["name"]
    pod = deployment["spec"]["template"]["spec"]
    item = container(deployment)
    termination = int(pod.get("terminationGracePeriodSeconds", 30))
    command = " ".join(item.get("args") or [])
    graceful_match = GRACEFUL_SECONDS.search(command)
    graceful = int(graceful_match.group(1)) if graceful_match else 0
    lifecycle = item.get("lifecycle", {})
    pre_stop = lifecycle.get("preStop", {}).get("exec", {}).get("command", [])
    sleep_match = SLEEP_SECONDS.fullmatch(str(pre_stop[-1])) if pre_stop else None
    pre_stop_seconds = int(sleep_match.group(1)) if sleep_match else 0
    lifespan = integer(
        values,
        "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS",
        10,
    )
    margin = integer(values, "GPU_FAULT_SHUTDOWN_MARGIN_SECONDS", 20)
    required = pre_stop_seconds + max(graceful, lifespan) + margin
    errors = []
    if termination < required:
        errors.append(
            f"{name}: terminationGracePeriodSeconds={termination} "
            f"is less than preStop({pre_stop_seconds}) + "
            f"max(uvicorn={graceful}, lifespan={lifespan}) + "
            f"margin({margin}) = {required}"
        )
    request_max = integer(
        values,
        "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS",
        0,
    )
    exit_grace = integer(values, "GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS", 0)
    if (
        values.get("GPU_FAULT_SERVICE_ROLE") in {"worker", "spool-worker"}
        and request_max
        and lifespan < request_max + exit_grace + 5
    ):
        errors.append(
            f"{name}: lifespan shutdown budget {lifespan}s cannot "
            f"cover request {request_max}s + exit grace "
            f"{exit_grace}s + 5s"
        )
    return errors


def validate_role(
    deployment: dict,
    values: dict[str, str | None],
) -> list[str]:
    name = deployment["metadata"]["name"]
    role = values.get("GPU_FAULT_SERVICE_ROLE")
    errors = []
    command = " ".join(container(deployment).get("args") or [])
    if "uvicorn " in command:
        if "--no-proxy-headers" not in command:
            errors.append(f"{name}: uvicorn must explicitly disable proxy headers")
        if (
            "--proxy-headers" in command
            or "--forwarded-allow-ips" in command
            or values.get("FORWARDED_ALLOW_IPS") not in {None, ""}
        ):
            errors.append(f"{name}: proxy-derived client identity is forbidden")
    if role == "ingress":
        leaked = sorted(PROCESSOR_POOL_ENV.intersection(values))
        if leaked:
            errors.append(
                f"{name}: ingress contains processor pool config " + ", ".join(leaked)
            )
    if role == "worker" and not PROCESSOR_POOL_ENV.issubset(values):
        errors.append(f"{name}: worker is missing processor pool configuration")
    # The switches are read the way the processes read them, so a manifest the
    # lint accepts is one the worker will not refuse at start-up.
    configured = {key: value for key, value in values.items() if value is not None}
    try:
        spool_enabled = env_bool("GPU_FAULT_TELEMETRY_SPOOL", False, environ=configured)
        registry_enabled = env_bool(
            "GPU_FAULT_ENABLE_AGENT_REGISTRY", False, environ=configured
        )
    except ValueError as error:
        errors.append(f"{name}: {error}")
        return errors
    if (
        role == "spool-worker"
        and int(deployment["spec"].get("replicas", 0)) > 0
        and not spool_enabled
    ):
        errors.append(
            f"{name}: enabled spool worker must set GPU_FAULT_TELEMETRY_SPOOL=true"
        )
    if registry_enabled:
        for pin in (
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256",
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST",
        ):
            if values.get(pin) in {None, ""}:
                errors.append(f"{name}: enabled agent registry requires {pin}")
    return errors


def validate_operation_allowlists(
    environments: dict[str, dict[str, str | None]],
) -> list[str]:
    control = set()
    node = set()
    for values in environments.values():
        raw_control = values.get("GPU_FAULT_ALLOWED_OPERATIONS")
        raw_node = values.get("GPU_FAULT_NODE_ALLOWED_OPERATIONS")
        if raw_control not in {None, "", "<valueFrom>"}:
            control.update(
                item.strip() for item in str(raw_control).split(",") if item.strip()
            )
        if raw_node not in {None, "", "<valueFrom>"}:
            node.update(
                item.strip() for item in str(raw_node).split(",") if item.strip()
            )
    errors = []
    valid = {item.value for item in WorkflowOperation}
    unknown = sorted((control | node) - valid)
    if unknown:
        errors.append(
            "operation allowlists contain unknown values: " + ", ".join(unknown)
        )
    if not node:
        return errors
    node_adapter = {
        item.value for item in operations_for_adapter(OperationAdapter.NODE_ACTION)
    }
    expected = control & node_adapter
    missing = sorted(expected - node)
    extra = sorted(node - control)
    unsupported = sorted(node - node_adapter)
    if missing:
        errors.append(
            "node allowlist is missing control-plane NodeAction "
            "operations: " + ", ".join(missing)
        )
    if extra:
        errors.append(
            "node allowlist contains operations disabled by the "
            "control plane: " + ", ".join(extra)
        )
    if unsupported:
        errors.append(
            "node allowlist contains non-NodeAction operations: "
            + ", ".join(unsupported)
        )
    return errors


def validate_connections(
    deployments: list[dict],
    environments: dict[str, dict[str, str | None]],
) -> list[str]:
    total = 0
    budget = 1200
    details = []
    for deployment in deployments:
        name = deployment["metadata"]["name"]
        replicas = int(deployment["spec"].get("replicas", 1))
        values = environments[name]
        command = " ".join(container(deployment).get("args") or [])
        match = UVICORN_WORKERS.search(command)
        workers = int(match.group(1)) if match else 1
        pool = integer(values, "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", 8)
        listeners = (
            workers * replicas
            if values.get("GPU_FAULT_SERVICE_ROLE") in {"worker", "spool-worker"}
            else 0
        )
        subtotal = pool * workers * replicas + listeners
        total += subtotal
        details.append(f"{name}={pool}*{workers}*{replicas}+{listeners}")
        budget = integer(
            values,
            "GPU_FAULT_POSTGRES_FLEET_CONNECTION_BUDGET",
            budget,
        )
    if total > budget:
        return [
            f"PostgreSQL connection ceiling {total} exceeds budget "
            f"{budget}: " + ", ".join(details)
        ]
    return []


def validate_environment_inventory(
    deployment_name: str,
    values: dict[str, str | None],
) -> list[str]:
    """Hold a manifest to the same contract the Pod enforces at start-up.

    `validate_gpu_fault_environment` makes a process refuse unknown
    ``GPU_FAULT_*`` names and unusable values, so a manifest carrying either
    is a rollout that crash-loops. Catching it here means the release fails
    at render time, with the file name, instead of minutes into the rollout.
    Values resolved from Secrets (``<valueFrom>``) are not literal and are not
    judged. Blank booleans are refused even though the runtime reads blank as
    "unset": a manifest key that says nothing while looking like a setting is
    exactly the ambiguity the boolean rework removed everywhere else.
    """

    literal = {
        key: value
        for key, value in values.items()
        if value is not None and value != "<valueFrom>"
    }
    errors = [
        f"{deployment_name}: {name} is not in the runtime inventory; the Pod would "
        "refuse to start (regenerate the inventory or drop the key)"
        for name in unknown_gpu_fault_environment(literal)
    ]
    kinds = environment_value_kinds()
    for name, value in sorted(literal.items()):
        kind = kinds.get(name)
        if kind is not None and kind[0] == "boolean" and not value.strip():
            errors.append(
                f"{deployment_name}: {name} is a blank boolean; delete the key or "
                "set one of 0/1/true/false/yes/no/on/off"
            )
    errors.extend(
        f"{deployment_name}: {problem}"
        for problem in invalid_gpu_fault_environment_values(literal)
    )
    return errors


def validate_against_container_env_snapshot(
    deployments: list[dict[str, Any]],
    snapshot_path: str,
) -> list[str]:
    """Judge a verbatim rollback render by the snapshot it was rendered from.

    On rollback the renderer copies the previous release's container
    ``env``/``envFrom`` into the rendered Deployments, so the current release's
    inventory, role, shutdown, allowlist and connection rules would reject a
    correct render (a name the current code dropped is "unknown" to it). The
    previous release already ran those validators on exactly these lists when
    it deployed; what can go wrong now is the copy, so that is what is checked:
    the three role Deployments are rendered, the snapshot and the render name
    the same Deployments and containers, and every container's lists equal the
    snapshot's (env order-insensitive by name, envFrom in order -- see
    ``gpu_fault.container_env_snapshot``). ``resolve_environment`` is skipped
    too: it needs the previous release's ConfigMaps, which the engine restores
    outside this render. A malformed or unreadable snapshot raises.
    """

    snapshot = load_container_env_snapshot(snapshot_path)
    rendered = {
        deployment["metadata"]["name"]: pod_container_env(deployment)
        for deployment in deployments
    }
    errors = [
        f"role Deployment {name} is not rendered"
        for name in ROLE_DEPLOYMENTS
        if name not in rendered
    ]
    errors.extend(
        container_env_differences(snapshot, rendered, actual_label="rendered")
    )
    return errors


def validate(
    paths: list[Path],
    *,
    node_operations: str | None = None,
    container_env_snapshot: str | None = None,
) -> dict:
    items = documents(paths)
    config_maps = {
        item["metadata"]["name"]: item.get("data", {})
        for item in items
        if item.get("kind") == "ConfigMap"
    }
    deployments = [item for item in items if item.get("kind") == "Deployment"]
    if container_env_snapshot is not None:
        snapshot_errors = validate_against_container_env_snapshot(
            deployments, container_env_snapshot
        )
        return {
            "valid": not snapshot_errors,
            "deployments": len(deployments),
            "config_maps": len(config_maps),
            "errors": snapshot_errors,
            "container_env_snapshot": container_env_snapshot,
            "skipped": list(CURRENT_CONTRACT_VALIDATORS),
        }
    environments = {}
    errors = []
    for deployment in deployments:
        values, env_errors = resolve_environment(deployment, config_maps)
        environments[deployment["metadata"]["name"]] = values
        errors.extend(env_errors)
        errors.extend(
            validate_environment_inventory(deployment["metadata"]["name"], values)
        )
        errors.extend(validate_shutdown(deployment, values))
        errors.extend(validate_role(deployment, values))
    errors.extend(validate_connections(deployments, environments))
    if node_operations:
        environments["node-agent"] = {
            "GPU_FAULT_NODE_ALLOWED_OPERATIONS": node_operations
        }
    errors.extend(validate_operation_allowlists(environments))
    return {
        "valid": not errors,
        "deployments": len(deployments),
        "config_maps": len(config_maps),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="gpu-fault-config")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("validate")
    command.add_argument("manifest", nargs="*")
    command.add_argument("--node-operations")
    command.add_argument(
        "--container-env-snapshot",
        metavar="FILE",
        help=(
            "rollback only: the previous release's container env snapshot the "
            "render was substituted from; compare against it and skip the "
            "current-release contract validators"
        ),
    )
    command.add_argument("--json", action="store_true")
    readiness = subcommands.add_parser("collector-readiness")
    readiness.add_argument("--url", required=True)
    readiness.add_argument("--cluster-id", required=True)
    readiness.add_argument(
        "--execution-token",
        default=os.getenv("GPU_FAULT_EXECUTION_TOKEN", ""),
    )
    args = parser.parse_args()
    if args.command == "collector-readiness":
        return _collector_readiness_command(args)
    try:
        report = validate(
            manifest_paths(args.manifest),
            node_operations=args.node_operations,
            container_env_snapshot=args.container_env_snapshot,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {
            "valid": False,
            "deployments": 0,
            "config_maps": 0,
            "errors": [str(exc)],
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    if report.get("skipped"):
        print(
            "configuration validators skipped: "
            + ", ".join(report["skipped"])
            + " (container env snapshot mode: the render carries the previous "
            "release's env/envFrom verbatim, which that release validated when "
            "it deployed; the render is judged against the snapshot instead)"
        )
    if report["valid"]:
        print(
            "configuration valid: "
            f"{report['deployments']} deployment(s), "
            f"{report['config_maps']} ConfigMap(s)"
        )
    else:
        for error in report["errors"]:
            print(f"configuration invalid: {error}", file=sys.stderr)
    return 0 if report["valid"] else 1


def _collector_readiness_command(args) -> int:
    if not args.execution_token:
        print(
            "collector readiness requires an execution token",
            file=sys.stderr,
        )
        return 2
    url = args.url.rstrip("/") + "/v1/collector-readiness/" + args.cluster_id
    request = urllib_request.Request(
        url,
        headers={"X-GPU-Fault-Execution-Token": args.execution_token},
    )
    try:
        with urllib_request.urlopen(request, timeout=30) as response:
            report = json.loads(response.read())
    except Exception as exc:
        print(f"collector readiness failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
