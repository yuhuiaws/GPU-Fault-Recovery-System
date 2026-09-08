from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.command_log import child_failure

ARN_PATTERN = re.compile(
    r"^arn:(?P<partition>[^:]+):(?P<service>[^:]+):"
    r"(?P<region>[^:]*):(?P<account>[^:]*):(?P<resource>.+)$"
)
SAFE_NAME = re.compile(r"[^A-Za-z0-9-]+")
DEFAULT_RUNTIME_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"
DEFAULT_NODE_INSTALLER_IMAGE = "public.ecr.aws/amazonlinux/amazonlinux:2023"
DEFAULT_DCGM_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04"
NODE_ALLOWED_OPERATIONS = (
    "COLLECT_HUNG_TRIAGE,COLLECT_DIAGNOSTIC_BUNDLE,"
    "RUN_DCGM_DIAGNOSTIC,QUIESCE_GPU_SERVICES,"
    "VERIFY_NO_GPU_CLIENTS,TRIGGER_HEALTH_SNAPSHOT,"
    "RESET_GPU,RESET_ALL_GPUS_NVSWITCHES,"
    "RESTORE_GPU_SERVICES,RESTART_FABRIC_MANAGER,"
    "REMEDIATE_EFA_DRIVER"
)
SITE_TAG_KEY = "gpu-fault:site-id"
BOOTSTRAP_STATE_VERSION = 3
NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS = frozenset(
    {
        "GPU_FAULT_PYTHON_STACK_TOOL",
        "GPU_FAULT_QUIESCE_RESTORE_COMMAND",
    }
)


class BootstrapError(RuntimeError):
    pass


class BootstrapMutationRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class Arn:
    partition: str
    service: str
    region: str
    account: str
    resource: str

    @classmethod
    def parse(cls, value: str) -> Arn:
        match = ARN_PATTERN.fullmatch(value.strip())
        if match is None:
            raise BootstrapError(f"invalid AWS ARN: {value}")
        return cls(**match.groupdict())

    @property
    def resource_name(self) -> str:
        kind, separator, name = self.resource.partition("/")
        if not separator or kind != "cluster" or not name:
            raise BootstrapError(f"ARN does not identify a cluster: {self.resource}")
        return name


@dataclass(frozen=True)
class ClusterIdentity:
    input_arn: str
    role: str
    region: str
    account_id: str
    hyperpod_arn: str
    hyperpod_name: str
    eks_arn: str
    eks_name: str
    vpc_id: str
    subnet_ids: tuple[str, ...]
    node_recovery: str
    context: str
    subnet_cidrs: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapRequest:
    cpu_cluster_arn: str
    gpu_cluster_arns: tuple[str, ...]
    repository_root: Path
    state_dir: Path
    # SES sends from this address to this address; there is no separate sender,
    # recipient list or subject prefix (the site.yaml fields the release engine
    # reads are filled from it).
    alert_email: str | None = None
    staging_only_release: bool = False
    impact_base: str = "origin/main"
    # Amazon Managed Grafana dashboards (``gpu_fault.admin.grafana``): an
    # explicit workspace id is operator input; ``grafana_viewer`` is the IAM
    # Identity Center user granted VIEWER on the workspace after the import.
    grafana_workspace_id: str | None = None
    grafana_viewer: str | None = None


@dataclass(frozen=True)
class BootstrapResult:
    site_file: Path
    state_file: Path
    pending_gpu_cluster_arns: tuple[str, ...] = ()


class CommandRunner:
    def __init__(self) -> None:
        self._print_lock = threading.Lock()

    def run(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = True,
        sensitive: bool = False,
        mutate: bool = False,
    ) -> str:
        shown = "<sensitive command>" if sensitive else " ".join(arguments)
        with self._print_lock:
            print(f"+ {shown}", file=sys.stderr, flush=True)
        completed = subprocess.run(
            list(arguments),
            input=input_text,
            text=True,
            capture_output=capture,
            check=False,
            env=dict(env) if env is not None else None,
            cwd=cwd,
        )
        if completed.returncode:
            # The whole captured stderr stays in the message: callers match
            # ``NoSuchEntity``/``NotFound`` in it to tell "absent" from "broken".
            raise child_failure(
                BootstrapError,
                arguments,
                completed.returncode,
                detail=completed.stderr.strip() if capture else "",
                sensitive=sensitive,
            )
        return completed.stdout.strip() if capture else ""

    def aws_json(
        self,
        region: str,
        *arguments: str,
        mutate: bool = False,
        sensitive: bool = False,
    ) -> dict[str, Any]:
        raw = self.run(
            [
                "aws",
                *arguments,
                "--region",
                region,
                "--output",
                "json",
            ],
            mutate=mutate,
            sensitive=sensitive,
        )
        return cast(dict[str, Any], json.loads(raw))

    def aws_text(
        self,
        region: str,
        *arguments: str,
        mutate: bool = False,
        sensitive: bool = False,
    ) -> str:
        return self.run(
            [
                "aws",
                *arguments,
                "--region",
                region,
                "--output",
                "text",
            ],
            mutate=mutate,
            sensitive=sensitive,
        )


class ReadOnlyProbeRunner(CommandRunner):
    def __init__(self, delegate: CommandRunner) -> None:
        super().__init__()
        self._delegate = delegate

    def run(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = True,
        sensitive: bool = False,
        mutate: bool = False,
    ) -> str:
        if mutate:
            raise BootstrapMutationRequired(arguments[0])
        return self._delegate.run(
            arguments,
            input_text=input_text,
            env=env,
            cwd=cwd,
            capture=capture,
            sensitive=sensitive,
        )


def compute_agent_config_digest(
    runner: CommandRunner,
    *,
    repository_root: Path,
    runtime_profile_version: str,
) -> str:
    installer = repository_root / "deploy/node/install-gpu-fault-collector.sh"
    raw_environment = runner.run(
        [
            "bash",
            str(installer),
            "--print-config-digest-environment",
        ],
        cwd=repository_root,
    )
    try:
        installer_environment = json.loads(raw_environment)
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            "node installer config digest environment is invalid JSON"
        ) from exc
    if not isinstance(installer_environment, dict) or set(installer_environment) != set(
        NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS
    ):
        raise BootstrapError(
            "node installer config digest environment has unexpected keys"
        )
    if any(
        not isinstance(value, str) or not value or "\n" in value or "\r" in value
        for value in installer_environment.values()
    ):
        raise BootstrapError(
            "node installer config digest environment has invalid values"
        )
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository_root / "src"),
        "GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION": runtime_profile_version,
        "GPU_FAULT_NODE_ACTION_KEY_VERSION": "2",
        "GPU_FAULT_NODE_ALLOWED_OPERATIONS": NODE_ALLOWED_OPERATIONS,
        "GPU_FAULT_NODE_ALLOW_GPU_RESET": "true",
        "GPU_FAULT_NODE_ALLOW_FABRIC_RESET": "true",
        "GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE": "true",
        "GPU_FAULT_NODE_ALLOW_FABRIC_MANAGER_RESTART": "true",
        "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION": "true",
        **installer_environment,
    }
    digest = runner.run(
        [
            sys.executable,
            "-c",
            "from gpu_fault.node_agent import print_config_digest; "
            "print_config_digest()",
        ],
        env=environment,
        cwd=repository_root,
    )
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise BootstrapError("computed Agent config digest is invalid")
    return digest


class BootstrapState:
    def __init__(self, path: Path, *, site_id: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path.is_file():
            self.value = json.loads(path.read_text(encoding="utf-8"))
            if self.value.get("site_id") != site_id:
                raise BootstrapError("bootstrap state belongs to another site")
            version = int(self.value.get("schema_version") or 1)
            if version > BOOTSTRAP_STATE_VERSION:
                raise BootstrapError("bootstrap state was written by a newer version")
            if version < BOOTSTRAP_STATE_VERSION:
                self.value["schema_version"] = BOOTSTRAP_STATE_VERSION
                self.value["completed_tasks"] = []
                self._write()
        else:
            self.value = {
                "schema_version": BOOTSTRAP_STATE_VERSION,
                "site_id": site_id,
                "phase": "new",
                "resources": {},
                "completed_tasks": [],
            }

    def record(self, name: str, value: Any) -> None:
        with self._lock:
            self.value["resources"][name] = value
            self._write()

    def complete(self, task: str) -> None:
        with self._lock:
            completed = set(self.value["completed_tasks"])
            completed.add(task)
            self.value["completed_tasks"] = sorted(completed)
            self._write()

    def bind_inputs(
        self,
        digest: str,
        task_digests: Mapping[str, str] | None = None,
    ) -> None:
        with self._lock:
            if task_digests is None:
                if self.value.get("input_sha256") != digest:
                    self.value["completed_tasks"] = []
            else:
                previous = self.value.get("task_input_sha256")
                previous = previous if isinstance(previous, dict) else {}
                completed = {
                    task
                    for task in self.value["completed_tasks"]
                    if previous.get(task) == task_digests.get(task)
                }
                self.value["completed_tasks"] = sorted(completed)
                self.value["task_input_sha256"] = dict(sorted(task_digests.items()))
            self.value["input_sha256"] = digest
            self._write()

    def phase(self, value: str) -> None:
        with self._lock:
            self.value["phase"] = value
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.value, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, self.path)


def safe_name(value: str, *, maximum: int = 63) -> str:
    normalized = SAFE_NAME.sub("-", value).strip("-").lower()
    if not normalized:
        raise BootstrapError(f"cannot derive a resource name from {value!r}")
    if len(normalized) <= maximum:
        return normalized
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:8]
    return normalized[: maximum - 9].rstrip("-") + "-" + digest


def tag_map(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if not isinstance(value, list):
        return {}
    return {
        str(item.get("Key") or item.get("key") or ""): str(
            item.get("Value") or item.get("value") or ""
        )
        for item in value
        if isinstance(item, dict)
    }


def assert_site_tag(
    tags: object,
    *,
    site_id: str,
    description: str,
    allow_missing: bool = False,
) -> bool:
    owner = tag_map(tags).get(SITE_TAG_KEY)
    if owner == site_id:
        return True
    if owner is None and allow_missing:
        return False
    if owner is None:
        raise BootstrapError(
            f"{description} exists without {SITE_TAG_KEY}; refusing to share it"
        )
    raise BootstrapError(f"{description} belongs to site {owner!r}, not {site_id!r}")


def run_parallel(
    tasks: Mapping[str, Callable[[], Any]],
    *,
    state: BootstrapState,
    revalidate: frozenset[str] = frozenset(),
    probes: Mapping[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    completed = set(state.value["completed_tasks"])
    resources = state.value["resources"]
    results = {
        name: resources[name]
        for name in tasks
        if name in completed and name in resources and name not in revalidate
    }
    probe_tasks = probes or {}

    def reconcile(name: str, ensure: Callable[[], Any]) -> Any:
        probe = probe_tasks.get(name)
        if name in completed and name in revalidate and probe is not None:
            try:
                probe()
                return resources[name]
            except BootstrapMutationRequired:
                pass
        return ensure()

    pending = {
        name: (lambda name=name, function=function: reconcile(name, function))
        for name, function in tasks.items()
        if name not in results
    }
    if not pending:
        return results
    with ThreadPoolExecutor(max_workers=min(8, len(pending))) as executor:
        futures = {
            executor.submit(function): name for name, function in pending.items()
        }
        failures: list[tuple[str, Exception]] = []
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                failures.append((name, exc))
                continue
            state.record(name, results[name])
            state.complete(name)
    if failures:
        failed_names = ", ".join(sorted(name for name, _exc in failures))
        first_name, first_error = failures[0]
        first_error.add_note(
            f"parallel bootstrap task(s) failed: {failed_names}; "
            f"first observed task: {first_name}"
        )
        raise first_error
    return results


def describe_or_absent(
    runner: CommandRunner,
    region: str,
    *arguments: str,
    not_found: Sequence[str],
) -> dict[str, Any] | None:
    """One ``aws`` describe that answers both "is it there?" and "what is it?".

    Only an error naming one of the ``not_found`` codes means absent; anything
    else (AccessDenied, a throttled call, a broken CLI) is re-raised. Reading a
    failed describe as "absent" -- the ``subprocess.run(...).returncode == 0``
    pattern this replaces -- sends the run into a ``create-*`` call that then
    fails on the resource that was there all along, or worse, succeeds twice.
    """

    try:
        return runner.aws_json(region, *arguments)
    except BootstrapError as exc:
        message = str(exc)
        if any(code in message for code in not_found):
            return None
        raise


def write_secret(path: Path, value: str) -> None:
    """Write a secret file exactly once, at 0600, without a clobber window.

    The contract is write-once: an existing file is left untouched (the fleet
    master key is reused across runs and regenerating it would invalidate the
    fleet). ``O_CREAT | O_EXCL`` makes that atomic -- there is no gap between a
    ``path.exists()`` check and the write for an attacker to plant a symlink
    into, and the 0600 mode is set as the file is created rather than widened
    from the umask default afterwards. A symlink already sitting at the path is
    refused rather than followed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if os.path.islink(path):
            raise BootstrapError(f"refusing to write a secret over a symlink: {path}")
        os.chmod(path, 0o600)
        return
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
    finally:
        os.chmod(path, 0o600)


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def kubectl_apply(
    runner: CommandRunner,
    kubeconfig: Path,
    manifest: str,
    *,
    context: str | None = None,
) -> None:
    arguments = ["kubectl", "--kubeconfig", str(kubeconfig)]
    if context:
        arguments.extend(["--context", context])
    arguments.extend(["apply", "-f", "-"])
    runner.run(
        arguments,
        input_text=manifest,
        mutate=True,
        capture=False,
    )


def ensure_namespace(
    runner: CommandRunner,
    *,
    kubeconfig: Path,
    namespace: str,
    context: str | None = None,
) -> None:
    arguments = ["kubectl", "--kubeconfig", str(kubeconfig)]
    if context:
        arguments.extend(["--context", context])
    arguments.extend(
        [
            "create",
            "namespace",
            namespace,
            "--dry-run=client",
            "-o",
            "yaml",
        ]
    )
    manifest = runner.run(arguments)
    kubectl_apply(runner, kubeconfig, manifest, context=context)
