from __future__ import annotations

import ast
import re
import hashlib
import json
import os
import shutil
import subprocess
import tomllib
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/gpu_fault"


@dataclass(frozen=True)
class Component:
    distribution: str
    roots: tuple[str, ...]
    scripts: dict[str, str]
    entry_points: dict[str, dict[str, str]]
    include_globs: tuple[str, ...] = ()
    data_globs: tuple[str, ...] = (
        "env-inventory.json",
        "nvidia-*.yaml",
    )


COMPONENTS = {
    "control_plane": Component(
        distribution="gpu-fault-control-plane",
        roots=(
            "gpu_fault.api",
            "gpu_fault.app",
            "gpu_fault.aurora_credential_refresh",
            "gpu_fault.config_cli",
            "gpu_fault.fleet_cli",
            "gpu_fault.hyperpod_cli",
            "gpu_fault.store_migrate",
            "gpu_fault.training_submit_cli",
            "gpu_fault.workload_annotate_cli",
        ),
        scripts={
            "gpu-fault-api": "gpu_fault.api:run",
            "gpu-fault-aurora-credential-refresh": (
                "gpu_fault.aurora_credential_refresh:main"
            ),
            "gpu-fault-config": "gpu_fault.config_cli:main",
            "gpu-fault-fleet": "gpu_fault.fleet_cli:main",
            "gpu-fault-hyperpod": "gpu_fault.hyperpod_cli:main",
            "gpu-fault-store-migrate": "gpu_fault.store_migrate:main",
            "gpu-fault-workload-annotate": ("gpu_fault.workload_annotate_cli:main"),
            "gpu-training-submit": "gpu_fault.training_submit_cli:main",
        },
        entry_points={
            "gpu_fault.collector_sinks": {
                "http": "gpu_fault.collectors.sinks:HttpEventSink",
                "sqs": "gpu_fault.collectors.sinks:SqsEventSink",
            },
            "gpu_fault.workflow_adapters": {
                "control-plane-evidence": (
                    "gpu_fault.adapters.evidence:ControlPlaneEvidenceAdapter"
                ),
                "gpu-validation": (
                    "gpu_fault.adapters.gpu_validation:GpuValidationAdapter"
                ),
                "hyperpod": (
                    "gpu_fault.adapters.hyperpod.lifecycle:HyperPodLifecycleStepAdapter"
                ),
                "kubernetes": (
                    "gpu_fault.adapters.kubernetes.adapter:KubernetesWorkflowAdapter"
                ),
                "managed-recovery": (
                    "gpu_fault.adapters.managed_recovery:ManagedRecoveryObserverAdapter"
                ),
                "node-action": (
                    "gpu_fault.adapters.node_action.adapter:NodeActionWorkflowAdapter"
                ),
                "support": (
                    "gpu_fault.adapters.support_escalation:SupportEscalationAdapter"
                ),
            },
            "gpu_fault.notification_builders": {
                "dcgm-diagnostic": (
                    "gpu_fault.notifications.dcgm_diagnostic:DcgmDiagnosticEmailBuilder"
                ),
                "efa-rdma": (
                    "gpu_fault.notifications.efa_rdma:EfaRdmaEventEmailBuilder"
                ),
                "hardware-escalation": (
                    "gpu_fault.notifications.hardware_escalation:"
                    "HardwareEscalationEmailBuilder"
                ),
                "hardware-inventory": (
                    "gpu_fault.notifications.hardware_inventory:"
                    "HardwareInventoryEmailBuilder"
                ),
                "host-resource": (
                    "gpu_fault.notifications.host_resource:"
                    "HostResourceEventEmailBuilder"
                ),
                "hyperpod-advisory": (
                    "gpu_fault.notifications.hyperpod_advisory:"
                    "HyperPodAdvisoryEmailBuilder"
                ),
                "not-applicable": (
                    "gpu_fault.notifications.not_applicable:NotApplicableEmailBuilder"
                ),
                "nvlink74-mechanical": (
                    "gpu_fault.notifications.nvlink74_mechanical:"
                    "Nvlink74MechanicalEmailBuilder"
                ),
                "nvlink74-support": (
                    "gpu_fault.notifications.nvlink74_support:"
                    "Nvlink74SupportEmailBuilder"
                ),
                "restart-guard": (
                    "gpu_fault.notifications.restart_guard:RestartGuardEmailBuilder"
                ),
                "sxid-event": (
                    "gpu_fault.notifications.sxid_event:SxidEventEmailBuilder"
                ),
                "warm-spare": (
                    "gpu_fault.notifications.warm_spare:"
                    "WarmSpareReplacementEmailBuilder"
                ),
                "xid-investigatory": (
                    "gpu_fault.notifications.xid_investigatory:"
                    "XidInvestigatoryEmailBuilder"
                ),
            },
        },
        # `workflow_reconcile` has no importer inside the control plane: the
        # administrator's `workflow-reconcile --plan/--apply` runs it in the Pod
        # through a program the deploy host execs, so the module has to be in
        # this wheel even though nothing here reaches it by import. The deploy
        # host ships only the wrapper, which is what keeps the Store mutation on
        # the control plane.
        include_globs=("store/postgres/ddl*.py", "workflow_reconcile.py"),
    ),
    "executor": Component(
        distribution="gpu-fault-cluster-executor",
        roots=(
            "gpu_fault.cluster_executor",
            "gpu_fault.collectors_cli",
            "gpu_fault.completion_controller",
            "gpu_fault.node_installer_reconciler",
        ),
        scripts={
            "gpu-fault-cluster-executor": "gpu_fault.cluster_executor:main",
            "gpu-fault-cluster-executor-readiness": (
                "gpu_fault.cluster_executor:readiness_probe"
            ),
            "gpu-fault-collector": "gpu_fault.collectors_cli:main",
            "gpu-fault-completion-watcher": ("gpu_fault.completion_controller:main"),
            "gpu-fault-node-installer-reconciler": (
                "gpu_fault.node_installer_reconciler:main"
            ),
        },
        entry_points={
            "gpu_fault.collector_sinks": {
                "http": "gpu_fault.collectors.sinks:HttpEventSink",
                "sqs": "gpu_fault.collectors.sinks:SqsEventSink",
            }
        },
        include_globs=("store/postgres/ddl*.py",),
    ),
    "node_runtime": Component(
        distribution="gpu-fault-node-runtime",
        roots=(
            "gpu_fault.collectors_cli",
            "gpu_fault.node_agent",
            "gpu_fault.node_agent.app",
            "gpu_fault.node_agent.config",
            "gpu_fault.node_agent.quiesce",
        ),
        scripts={
            "gpu-fault-agent-config-digest": (
                "gpu_fault.node_agent:print_config_digest"
            ),
            "gpu-fault-collector": "gpu_fault.collectors_cli:main",
            "gpu-fault-node-agent": "gpu_fault.node_agent:run",
            "gpu-fault-restore-gpu-services": (
                "gpu_fault.node_agent:restore_gpu_services"
            ),
        },
        entry_points={
            "gpu_fault.collector_sinks": {
                "http": "gpu_fault.collectors.sinks:HttpEventSink",
                "sqs": "gpu_fault.collectors.sinks:SqsEventSink",
            }
        },
    ),
}
APPLICATION_COMPONENT_NAMES = tuple(COMPONENTS)


def component_definition(name: str) -> Component:
    component = COMPONENTS.get(name)
    if component is not None:
        return component
    if name != "deploy_host":
        raise KeyError(name)
    if __package__:
        from scripts.deploy_host_component import DEPLOY_HOST_COMPONENT_SPEC
    else:
        from deploy_host_component import DEPLOY_HOST_COMPONENT_SPEC
    return Component(**DEPLOY_HOST_COMPONENT_SPEC)


def _module_map() -> dict[str, Path]:
    result = {}
    for path in SOURCE.rglob("*.py"):
        relative = path.relative_to(SOURCE)
        if path.name == "__init__.py":
            parts = relative.parts[:-1]
        else:
            parts = (*relative.parts[:-1], path.stem)
        name = ".".join(("gpu_fault", *parts))
        result[name] = path
    return result


MODULES = _module_map()


def _resolved_relative(module: str, level: int, imported: str | None) -> str:
    package = (
        module if MODULES[module].name == "__init__.py" else module.rpartition(".")[0]
    )
    parts = package.split(".")
    if level > len(parts):
        return ""
    base = parts[: len(parts) - level + 1]
    if imported:
        base.extend(imported.split("."))
    return ".".join(base)


def _lazy_exports(module: str) -> dict[str, str]:
    tree = ast.parse(MODULES[module].read_text(encoding="utf-8"))
    exports: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_EXPORTS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, (ast.Tuple, ast.List))
                and value.elts
                and isinstance(value.elts[0], ast.Constant)
                and isinstance(value.elts[0].value, str)
                and value.elts[0].value in MODULES
            ):
                exports[key.value] = value.elts[0].value
    return exports


# ``"gpu_fault.collectors.gpu.dcgm:build_from_environment"`` -- the shape a
# registry, an entry point or ``_load_factory`` uses to name a callable without
# importing it. The module half is a dependency the import walk cannot see.
_FACTORY_REFERENCE = re.compile(
    r"^(gpu_fault(?:\.[A-Za-z_][A-Za-z0-9_]*)+):[A-Za-z_]\w*$"
)


def referenced_modules(tree: ast.AST) -> set[str]:
    """Modules named by string, not import: factory references and
    ``import_module("gpu_fault....")`` literals.

    The node-runtime wheel built on 2026-09-07 lost every collector the new
    ``collector_registry`` names this way (dcgm, nvidia_smi, host, kernel,
    fabric manager, node logs) and the node's collector services died with
    ModuleNotFoundError; wheel contents have to follow these strings too.
    """

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = _FACTORY_REFERENCE.match(node.value)
            if match and match.group(1) in MODULES:
                found.add(match.group(1))
        elif isinstance(node, ast.Call):
            callee = node.func
            name = (
                callee.attr
                if isinstance(callee, ast.Attribute)
                else callee.id
                if isinstance(callee, ast.Name)
                else ""
            )
            if name == "import_module" and node.args:
                first = node.args[0]
                if (
                    isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value in MODULES
                ):
                    found.add(first.value)
    return found


def _local_imports(module: str) -> set[str]:
    tree = ast.parse(MODULES[module].read_text(encoding="utf-8"))
    found: set[str] = set(referenced_modules(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gpu_fault" or alias.name.startswith("gpu_fault."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported = (
                _resolved_relative(module, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            if imported == "gpu_fault" or imported.startswith("gpu_fault."):
                found.add(imported)
                lazy = _lazy_exports(imported) if imported in MODULES else {}
                for alias in node.names:
                    candidate = f"{imported}.{alias.name}"
                    if candidate in MODULES:
                        found.add(candidate)
                    elif alias.name in lazy:
                        found.add(lazy[alias.name])
    return {name for name in found if name in MODULES}


def entrypoint_modules(component: Component) -> set[str]:
    values = [
        *component.scripts.values(),
        *(
            value
            for entries in component.entry_points.values()
            for value in entries.values()
        ),
    ]
    return {
        value.partition(":")[0]
        for value in values
        if value.partition(":")[0] in MODULES
    }


def extra_modules(component: Component) -> set[str]:
    paths = {
        path
        for pattern in component.include_globs
        for path in SOURCE.glob(pattern)
        if path.is_file()
    }
    reverse = {path: module for module, path in MODULES.items()}
    return {reverse[path] for path in paths}


def dependency_closure(roots: Iterable[str]) -> set[str]:
    missing = sorted(set(roots) - set(MODULES))
    if missing:
        raise RuntimeError("unknown component root module(s): " + ", ".join(missing))
    selected: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in selected:
            continue
        selected.add(module)
        pending.extend(sorted(_local_imports(module) - selected))
        parts = module.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent in MODULES and parent not in selected:
                pending.append(parent)
    selected.add("gpu_fault")
    return selected


def component_modules(name: str) -> set[str]:
    component = component_definition(name)
    return dependency_closure(
        {
            *component.roots,
            *entrypoint_modules(component),
            *extra_modules(component),
        }
    )


def component_data_files(name: str) -> tuple[Path, ...]:
    component = component_definition(name)
    data = SOURCE / "data"
    return tuple(
        sorted(
            {
                path
                for pattern in component.data_globs
                for path in data.glob(pattern)
                if path.is_file()
            }
        )
    )


def _copy_modules(
    modules: set[str],
    destination: Path,
    *,
    data_files: tuple[Path, ...],
) -> None:
    package = destination / "src/gpu_fault"
    for module in sorted(modules):
        source = MODULES[module]
        relative = source.relative_to(SOURCE)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o644)
    for source in data_files:
        target = package / "data" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o644)


def _toml_array(values: Iterable[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _write_project(
    destination: Path,
    component: Component,
) -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = root["project"]
    lines = [
        "[build-system]",
        f"requires = {_toml_array(root['build-system']['requires'])}",
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f"name = {json.dumps(component.distribution)}",
        f"version = {json.dumps(project['version'])}",
        f"description = {json.dumps(project['description'])}",
        f"license = {json.dumps(project['license'])}",
        f"license-files = {_toml_array(project['license-files'])}",
        f"requires-python = {json.dumps(project['requires-python'])}",
        f"dependencies = {_toml_array(project['dependencies'])}",
        "",
        "[project.optional-dependencies]",
    ]
    for name, values in project["optional-dependencies"].items():
        if name == "dev":
            continue
        lines.append(f"{name} = {_toml_array(values)}")
    lines.extend(["", "[project.scripts]"])
    for name, value in component.scripts.items():
        lines.append(f"{json.dumps(name)} = {json.dumps(value)}")
    for group, entries in component.entry_points.items():
        lines.extend(["", f"[project.entry-points.{json.dumps(group)}]"])
        for name, value in entries.items():
            lines.append(f"{json.dumps(name)} = {json.dumps(value)}")
    lines.extend(
        [
            "",
            "[tool.setuptools]",
            'package-dir = {"" = "src"}',
            "",
            "[tool.setuptools.packages.find]",
            'where = ["src"]',
            "",
            "[tool.setuptools.package-data]",
            'gpu_fault = ["data/*.yaml", "data/*.json"]',
            "",
        ]
    )
    (destination / "pyproject.toml").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    license_path = destination / "LICENSE"
    shutil.copy2(ROOT / "LICENSE", license_path)
    license_path.chmod(0o644)


def package_digest(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".json"}:
            continue
        relative = path.relative_to(package).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def component_source_digest(
    name: str,
    *,
    source_overrides: Mapping[str, bytes] | None = None,
) -> str:
    selected = component_modules(name)
    digest = hashlib.sha256()
    paths = [MODULES[module] for module in selected]
    paths.extend(component_data_files(name))
    for path in sorted(paths, key=lambda item: item.relative_to(SOURCE).as_posix()):
        relative = path.relative_to(SOURCE).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        content = (
            source_overrides.get(relative, path.read_bytes())
            if source_overrides is not None
            else path.read_bytes()
        )
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=8)
def validated_build_python(python: str) -> str:
    script = (
        "import importlib.metadata as m;"
        "expected={'build':'1.6.0','setuptools':'84.0.0'};"
        "actual={name:m.version(name) for name in expected};"
        "assert actual == expected, (actual, expected)"
    )
    subprocess.run([python, "-c", script], check=True)
    return python


def build_component(
    *,
    python: str,
    name: str,
    build_root: Path,
    output: Path,
) -> tuple[Path, str, set[str]]:
    python = validated_build_python(python)
    component = component_definition(name)
    selected = component_modules(name)
    project = build_root / name
    shutil.rmtree(project, ignore_errors=True)
    project.mkdir(parents=True)
    _copy_modules(
        selected,
        project,
        data_files=component_data_files(name),
    )
    _write_project(project, component)
    before = set(output.glob("*.whl"))
    previous_umask = os.umask(0o022)
    try:
        completed = subprocess.run(
            [
                python,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
            ],
            cwd=project,
            env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"},
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        os.umask(previous_umask)
    if completed.returncode:
        raise RuntimeError(
            f"{name} wheel build failed with status {completed.returncode}:\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    wheels = sorted(set(output.glob("*.whl")) - before)
    if len(wheels) != 1:
        raise RuntimeError(f"{name} build produced {len(wheels)} wheels")
    repack_wheel_stored(wheels[0])
    digest = package_digest(project / "src/gpu_fault")
    return wheels[0], digest, selected


def repack_wheel_stored(wheel: Path) -> Path:
    """Rewrite ``wheel`` in place with every entry stored, not deflated.

    A wheel is a zip with per-file deflate. Everything downstream compresses
    the *whole file* again -- xz into the control-plane and executor wheel
    ConfigMaps, gzip into the node installer bundle -- and compressing
    already-deflated members gains nothing: the deflated control-plane wheel
    was 1,106,539 bytes, xz took it to 1,079,868, and both exceed the 1 MiB a
    ConfigMap can hold. Stored entries let the outer xz see the Python source
    itself (643,872 bytes for the same wheel). Entry order, timestamps and
    permission bits are preserved so the wheel stays reproducible, and pip
    installs stored wheels exactly like deflated ones.
    """

    staging = wheel.with_name(wheel.name + ".stored")
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED) as target,
    ):
        for info in source.infolist():
            entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = info.external_attr
            entry.create_system = info.create_system
            target.writestr(entry, source.read(info.filename))
    staging.replace(wheel)
    return wheel
