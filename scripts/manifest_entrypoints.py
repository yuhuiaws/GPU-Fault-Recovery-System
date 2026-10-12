"""Python the shipped manifests reach by name, and the component wheel that owes it.

A component wheel carries the import closure of its roots, console scripts
and entry points (``scripts/component_wheels.py``). The walk follows
``import`` statements and the ``"module:attr"`` strings a registry names; it
cannot see a module that a *manifest* names. The control-plane roles are
launched with ``python -m gpu_fault.app.serve``: nothing imports a ``-m``
launcher, so the closure left it out, the release gate compared source
closures with themselves and passed, and every api-ha Pod of the first
release to carry that command crash-looped on
``ModuleNotFoundError: No module named 'gpu_fault.app.serve'`` -- the
checkout ran the tests, the installed wheel could not start a Pod.

The same hole exists for every other way a container names Python without
importing it: ``python -c "import gpu_fault..."`` one-liners, the ASGI factory
``gpu_fault.app:create_app`` the launcher is handed, and console scripts
whose ``[project.scripts]`` entry was never declared. This module lists each
such reference per component wheel from the manifests themselves (container
commands and args, exec probes, lifecycle hooks), so that
``component_wheels.build_component`` refuses to package a wheel that cannot
start its own Pods, ``tests/test_component_wheel_closures.py`` checks the
source closure and ``tests/test_artifact_consistency.py`` checks the built
wheel. Third-party modules (``python -m uvicorn``) are the dependency lock's
business and are not recorded.
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

ROOT = Path(__file__).resolve().parents[1]

#: Component wheel -> the manifests whose containers run from its venv. A
#: workload that is not listed is not checked; the PATH each manifest sets
#: (``/opt/gpu-fault/<component>/bin``) says which wheel it installs from.
MANIFEST_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "control_plane": (
        "deploy/control-plane/base/control-plane-deployment.yaml",
        "deploy/control-plane/regional/generated/*.yaml",
        "deploy/control-plane/regional/aurora-credential-refresh.yaml",
        "deploy/migrations/*.yaml",
    ),
    "executor": (
        "deploy/dataplane/cluster-action-executor.yaml",
        "deploy/dataplane/completion-watcher.yaml",
        "deploy/dataplane/kubernetes-node-resource-collector.yaml",
        "deploy/dataplane/node-installer-reconciler.yaml",
    ),
}
#: The packages a wheel can be asked for; anything else is a dependency.
LOCAL_PACKAGES = ("gpu_fault", "gpu_fault_release")
#: ``kind`` of a :class:`ManifestEntrypoint` whose ``target`` names a module
#: (``import_name`` only when the source tree has a module of that name: a
#: ``from pkg import name`` may import an attribute).
MODULE_KINDS = frozenset({"module", "import", "import_name", "factory"})
PROBE_FIELDS = ("startupProbe", "readinessProbe", "livenessProbe")
HOOK_FIELDS = ("postStart", "preStop")

_PYTHON = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_SHELLS = frozenset({"sh", "bash", "dash", "ash"})
_MODULE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_FACTORY = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):[A-Za-z_]\w*$")
_CONSOLE_SCRIPT = re.compile(r"^gpu-(?:fault|training)-[a-z0-9-]+$")
_IMPORT_LINE = re.compile(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.]*)")


@dataclass(frozen=True, order=True)
class ManifestEntrypoint:
    """One name a manifest asks a component venv to resolve.

    ``kind`` is ``module`` (``python -m target``), ``import`` (a module a
    ``python -c`` script imports), ``import_name`` (``from pkg import name``:
    a submodule when the tree has one, otherwise an attribute), ``factory``
    (the module half of a ``module:attr`` argument) or ``console_script``
    (``target`` is the command name).
    """

    manifest: str
    container: str
    kind: str
    target: str


def is_local_module(name: str) -> bool:
    return any(
        name == package or name.startswith(f"{package}.") for package in LOCAL_PACKAGES
    )


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _mappings(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _mappings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _mappings(item)


def _exec_command(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    handler = value.get("exec")
    return _strings(handler.get("command")) if isinstance(handler, dict) else []


def container_commands(document: object) -> Iterator[tuple[str, list[str]]]:
    """``(label, argv)`` for every process a Pod template in ``document`` starts:
    each container's and init container's command plus args, its exec probes
    and its lifecycle hooks."""

    for mapping in _mappings(document):
        for field in ("containers", "initContainers"):
            containers = mapping.get(field)
            if not isinstance(containers, list):
                continue
            for container in containers:
                if not isinstance(container, dict):
                    continue
                name = str(container.get("name", "?"))
                yield (
                    name,
                    [
                        *_strings(container.get("command")),
                        *_strings(container.get("args")),
                    ],
                )
                for probe in PROBE_FIELDS:
                    yield f"{name}/{probe}", _exec_command(container.get(probe))
                lifecycle = container.get("lifecycle")
                if isinstance(lifecycle, dict):
                    for hook in HOOK_FIELDS:
                        yield (
                            f"{name}/lifecycle.{hook}",
                            _exec_command(lifecycle.get(hook)),
                        )


def shell_words(argv: list[str]) -> list[str]:
    """The words a shell would run: ``sh -c SCRIPT`` becomes SCRIPT's words
    (quotes honoured, ``;``/``&&``/``|`` split off), anything else is itself."""

    if not argv or Path(argv[0]).name not in _SHELLS:
        return argv
    rest = argv[1:]
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if "c" in flag[1:]:
            break
    else:
        return []
    if not rest:
        return []
    lexer = shlex.shlex(rest[0], posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return rest[0].split()


def _imported_modules(script: str) -> set[tuple[str, str]]:
    """``(kind, name)`` for every module a ``python -c`` script imports."""

    try:
        tree = ast.parse(script)
    except SyntaxError:
        return {("import", name) for name in _IMPORT_LINE.findall(script)}
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(("import", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(("import", node.module))
            found.update(
                ("import_name", f"{node.module}.{alias.name}") for alias in node.names
            )
    return found


def _interpreter_targets(arguments: list[str]) -> Iterator[tuple[str, str]]:
    """What a ``python`` invocation runs by name, read like the interpreter
    reads its command line: option letters may be grouped (``-Im``), ``-X``
    and ``-W`` take a value, the first bare word is a script path."""

    index = 0
    while index < len(arguments):
        argument = arguments[index]
        index += 1
        if not argument.startswith("-") or argument == "-" or argument.startswith("--"):
            return
        letters = argument[1:]
        for position, letter in enumerate(letters):
            value = letters[position + 1 :]
            if letter in "mc":
                if not value:
                    if index >= len(arguments):
                        return
                    value = arguments[index]
                if letter == "m" and _MODULE.match(value):
                    yield "module", value
                elif letter == "c":
                    yield from _imported_modules(value)
                return
            if letter in "XW":
                if not value:
                    index += 1
                break


def python_references(words: list[str]) -> Iterator[tuple[str, str]]:
    """``(kind, target)`` for every local module or console script in ``words``."""

    for position, word in enumerate(words):
        if _PYTHON.match(Path(word).name):
            for kind, target in _interpreter_targets(words[position + 1 :]):
                if is_local_module(target):
                    yield kind, target
        elif _CONSOLE_SCRIPT.match(word):
            yield "console_script", word
        else:
            factory = _FACTORY.match(word)
            if factory and is_local_module(factory.group(1)):
                yield "factory", factory.group(1)


def manifest_paths(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"no manifest matches {pattern} under {root}")
        paths.extend(matches)
    return paths


def manifest_entrypoints(
    root: Path = ROOT,
    components: Mapping[str, tuple[str, ...]] = MANIFEST_COMPONENTS,
) -> dict[str, tuple[ManifestEntrypoint, ...]]:
    """Every name the manifests of each component ask its wheel to resolve."""

    found: dict[str, set[ManifestEntrypoint]] = {}
    for component, patterns in components.items():
        entries = found.setdefault(component, set())
        for path in manifest_paths(root, patterns):
            relative = path.relative_to(root).as_posix()
            for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
                for label, argv in container_commands(document):
                    for kind, target in python_references(shell_words(argv)):
                        entries.add(ManifestEntrypoint(relative, label, kind, target))
    return {component: tuple(sorted(entries)) for component, entries in found.items()}


def component_entrypoint_failures(
    component: str,
    *,
    modules: Collection[str],
    scripts: Collection[str],
    module_paths: Mapping[str, Path],
    root: Path = ROOT,
    components: Mapping[str, tuple[str, ...]] = MANIFEST_COMPONENTS,
) -> list[str]:
    """Why ``component``'s wheel could not start every container its manifests
    describe; empty when it can.

    ``modules`` is the module set the wheel ships (the source closure before a
    build, the wheel listing after), ``scripts`` its console script names and
    ``module_paths`` the source tree (``component_wheels.MODULES``), which
    tells a module that does not exist from one that exists but is not
    shipped, and a package (whose ``-m`` needs ``__main__``) from a module.
    """

    if component not in components:
        return []
    failures: list[str] = []
    entries = manifest_entrypoints(root, {component: components[component]})
    for entry in entries[component]:
        where = f"{entry.manifest} [{entry.container}]"
        if entry.kind == "console_script":
            if entry.target not in scripts:
                failures.append(
                    f"{where}: console script {entry.target} is not a "
                    f"[project.scripts] entry of the {component} wheel"
                )
            continue
        if entry.kind == "import_name" and entry.target not in module_paths:
            continue  # ``from pkg import name`` took an attribute, not a module
        required = [entry.target]
        if entry.kind == "module" and entry.target in module_paths:
            if module_paths[entry.target].name == "__init__.py":
                required.append(f"{entry.target}.__main__")
        for module in required:
            if module not in module_paths:
                failures.append(
                    f"{where}: {entry.kind} {module} does not exist under src/"
                )
            elif module not in modules:
                failures.append(
                    f"{where}: {entry.kind} {module} is outside the {component} "
                    "wheel's module closure; nothing imports it, so list it in the "
                    "component's roots in scripts/component_wheels.py"
                )
    return failures
