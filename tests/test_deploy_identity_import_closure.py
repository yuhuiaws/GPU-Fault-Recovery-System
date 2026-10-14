"""Every repository file an identity input imports or runs is an identity input too.

A deploy compares the application and deploy-host identities with the last
successful deploy (``scripts/staging_deploy.classify_source_deploy``): a change
that moves neither is ``QUALITY_ONLY``, applies nothing, and the site keeps the
copy in its snapshot. The identities hash explicit file lists
(``scripts/deploy_source_identity.py``, ``scripts/deploy_host_identity.py``,
``config/release-identity.yaml``), so a module split out of a listed script was
hashed by nothing: ``release_attestation.py`` (imported by ``release_deploy.py``),
``staging_deploy_inputs.py`` and ``staging_source_snapshot.py``
(``staging_deploy.py``), the unit and coverage gate modules behind ``ci_gate.py``
and the two ``tools/`` modules they import, and ``manifest_entrypoints.py``
(``component_wheels.py``). A change to one of them alone moved no identity.

A script an input runs is the same dependency. ``staging_deploy.py`` runs
``restore_ci_candidate.py``, ``select-affected-tests.py`` and
``verify-release-attestation.py``; the release engine runs
``deploy/dataplane/tools/verify_dataplane_executor.py`` and
``deploy/observability/install-amp-monitoring.sh`` from the snapshot; the gate
runners run the check scripts, some of them through ``make`` targets.

The walk follows every ``import`` of a Python file -- at module level or inside
a function, in either branch of an ``if __package__`` -- to the repository
modules outside ``src`` (``scripts``, ``tools``, ``tests`` and a script's
sibling modules). It follows every run of a repository script: a path a Python
file names in a command vector (a list or tuple display) or in a path join, a
path a shell script names outside a comment, and every ``make`` target either
names, through the ``Makefile`` recipes of all the targets it reaches (so the
``Makefile`` is an input itself). Tests a gate runs are its subject, not its
machinery, and are not followed. It starts from every hashed file and every
``src`` module (those ship in the component wheels) and requires each file it
reaches to be hashed as well; ``src`` modules themselves are bound by the
component wheels' import closures (``tests/test_component_wheel_closures.py``).
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import yaml

from scripts import deploy_host_identity, deploy_source_identity, manifest_entrypoints
from scripts.release_identity import build_release_identity, file_set_identity

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
#: Repository packages outside ``src`` that an identity input can import by name.
PACKAGES = frozenset({"scripts", "tools", "tests"})
#: The exemptions: package markers an import passes through. Each holds only its
#: docstring, so importing it runs nothing (checked below), and it needs no
#: identity.
PACKAGE_MARKERS = frozenset({"scripts/__init__.py", "tools/__init__.py"})
#: A repository script a command can run.
RUNNABLE = re.compile(r"(?<![\w/.-])((?:scripts|tools|deploy)/[\w./-]+\.(?:py|sh))")
MAKE_RULE = re.compile(r"^([\w.%-]+(?:[ \t]+[\w.%-]+)*)[ \t]*:(?![=:])(.*)$")
SHELL_MAKE = re.compile(r"(?:\$\(MAKE\)|(?<![\w-])make)[ \t]+([\w.-]+)")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _file_sets(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        files = value.get("files")
        if isinstance(files, dict):
            yield from files
        for item in value.values():
            yield from _file_sets(item)


def hashed_files() -> set[str]:
    """Every file one of the identities hashes, relative to the repository."""

    return {
        *_file_sets(build_release_identity(ROOT)),
        *_file_sets(
            file_set_identity(
                ROOT, deploy_source_identity.APPLICATION_IDENTITY_PROTOCOL_INPUTS
            )
        ),
        *_file_sets(
            file_set_identity(
                ROOT, deploy_source_identity.DEPLOY_HOST_ORCHESTRATION_INPUTS
            )
        ),
        *deploy_host_identity.BUNDLE_INPUTS,
    }


def _callee(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else ""


def _module_file(base: Path, dotted: str) -> Path | None:
    target = base.joinpath(*dotted.split("."))
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def repository_imports(path: Path) -> set[Path]:
    """The repository modules outside ``src`` that running ``path`` can import."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    script = not path.is_relative_to(ROOT / "src")
    found: set[Path] = set()

    def resolve(dotted: str) -> None:
        parts = dotted.split(".")
        if parts[0] in PACKAGES:
            # Importing a.b.c runs the package markers it passes through too.
            for end in range(1, len(parts) + 1):
                module = _module_file(ROOT, ".".join(parts[:end]))
                if module is not None:
                    found.add(module)
        elif script and len(parts) == 1:
            # A script run by path imports its siblings by bare name.
            sibling = _module_file(path.parent, dotted)
            if sibling is not None:
                found.add(sibling)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolve(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = path.parents[node.level - 1].relative_to(ROOT).parts
                module = ".".join((*package, *([module] if module else [])))
            resolve(module)
            for alias in node.names:
                resolve(f"{module}.{alias.name}")
        elif (
            isinstance(node, ast.Call)
            and _callee(node) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            resolve(node.args[0].value)
    found.discard(path)
    return found


def _runnable(text: str) -> set[Path]:
    return {ROOT / name for name in RUNNABLE.findall(text) if (ROOT / name).is_file()}


@functools.cache
def makefile_rules() -> dict[str, tuple[str, ...]]:
    """Recipe lines by target; a prerequisite reads as ``$(MAKE) <target>``."""

    rules: dict[str, list[str]] = {}
    current: list[str] = []
    text = MAKEFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    for line in text.splitlines():
        if line.startswith("\t"):
            for target in current:
                rules[target].append(line[1:])
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rule = MAKE_RULE.match(line)
        current = []
        if rule is not None and not rule.group(1).startswith("."):
            current = rule.group(1).split()
            prerequisites = rule.group(2).split("#")[0].split()
            for target in current:
                rules.setdefault(target, []).extend(
                    f"$(MAKE) {name}" for name in prerequisites
                )
    return {target: tuple(lines) for target, lines in rules.items()}


def make_runs(targets: Iterable[str]) -> set[Path]:
    """The repository scripts the recipes of ``targets`` run, and of every
    target those recipes reach."""

    rules = makefile_rules()
    found: set[Path] = set()
    seen: set[str] = set()
    pending = list(targets)
    while pending:
        target = pending.pop()
        if target in seen or target not in rules:
            continue
        seen.add(target)
        for line in rules[target]:
            found |= _runnable(line)
            if "$(MAKE)" in line or line.lstrip("@-+ ").startswith("make "):
                pending.extend(
                    word for word in re.findall(r"[\w.-]+", line) if word in rules
                )
    return found


def selector_make_targets() -> frozenset[str]:
    """The make targets ``select-affected-tests.py --execute`` takes from the
    impact matrix: the checks of every rule and its PostgreSQL command."""

    matrix = yaml.safe_load(
        (ROOT / "testcases/change-impact.yaml").read_text(encoding="utf-8")
    )
    targets = set(matrix["full"]["checks"])
    for rule in matrix["rules"]:
        targets.update(rule.get("checks") or ())
    command = matrix["full"]["postgres_command"]
    assert command[0] == "make", f"the PostgreSQL command is no make target: {command}"
    targets.update(command[1:2])
    return frozenset(targets)


#: Make targets a file picks from data at runtime, which no static walk can read
#: off its source, by the file that runs them.
DATA_DRIVEN_MAKE_TARGETS: dict[str, Callable[[], frozenset[str]]] = {
    "scripts/select-affected-tests.py": selector_make_targets
}


def _make_vectors(tree: ast.AST) -> Iterator[ast.List | ast.Tuple]:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.List, ast.Tuple))
            and node.elts
            and isinstance(node.elts[0], ast.Constant)
            and node.elts[0].value == "make"
        ):
            yield node


def data_driven_make(path: Path) -> bool:
    """Whether a make command in ``path`` takes a target that is no literal (an
    ``f"VAR=..."`` word is a variable, not a target)."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        not isinstance(element, (ast.Constant, ast.JoinedStr))
        for vector in _make_vectors(tree)
        for element in vector.elts[1:]
    )


def _path_join(node: ast.BinOp) -> str:
    parts: list[str] = []
    current: ast.expr = node
    while (
        isinstance(current, ast.BinOp)
        and isinstance(current.op, ast.Div)
        and isinstance(current.right, ast.Constant)
        and isinstance(current.right.value, str)
    ):
        parts.append(current.right.value)
        current = current.left
    return "/".join(reversed(parts))


def repository_runs(path: Path) -> set[Path]:
    """The repository scripts that running ``path`` can run, by path or ``make``."""

    found: set[Path] = set()
    targets: set[str] = set()
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)):
                for element in node.elts:
                    if isinstance(element, ast.Constant) and isinstance(
                        element.value, str
                    ):
                        found |= _runnable(element.value)
            elif isinstance(node, ast.BinOp):
                found |= _runnable(_path_join(node))
        for vector in _make_vectors(tree):
            targets.update(
                element.value
                for element in vector.elts[1:]
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
        data_driven = DATA_DRIVEN_MAKE_TARGETS.get(_relative(path))
        if data_driven is not None:
            targets |= data_driven()
    elif path.suffix == ".sh":
        text = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        found |= _runnable(text)
        targets.update(SHELL_MAKE.findall(text))
    if targets & set(makefile_rules()):
        found |= make_runs(targets)
        found.add(MAKEFILE)
    found.discard(path)
    return {item for item in found if not _relative(item).startswith("tests/")}


@functools.cache
def reached_files() -> dict[str, frozenset[str]]:
    """Each repository file that a hashed file or a ``src`` module imports or
    runs, transitively, with the files found reaching it."""

    pending = [
        ROOT / name for name in sorted(hashed_files()) if name.endswith((".py", ".sh"))
    ]
    pending.extend(sorted((ROOT / "src").rglob("*.py")))
    reached: dict[str, set[str]] = {}
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or path == MAKEFILE:
            continue
        seen.add(path)
        dependencies = repository_runs(path)
        if path.suffix == ".py":
            dependencies |= repository_imports(path)
        for dependency in sorted(dependencies):
            reached.setdefault(_relative(dependency), set()).add(_relative(path))
            pending.append(dependency)
    return {name: frozenset(by) for name, by in reached.items()}


def unhashed(hashed: set[str]) -> dict[str, frozenset[str]]:
    return {
        name: by
        for name, by in reached_files().items()
        if name not in hashed and name not in PACKAGE_MARKERS
    }


def test_every_file_an_identity_input_imports_or_runs_is_an_identity_input() -> None:
    missing = unhashed(hashed_files())

    assert missing == {}, (
        "an identity input imports or runs files no identity hashes, so a change to "
        "one of them alone deploys nothing; list each in the identity input of the "
        "file that reaches it (scripts/deploy_source_identity.py, "
        "scripts/deploy_host_identity.py, config/release-identity.yaml): "
        + "; ".join(
            f"{name} <- {', '.join(sorted(by)[:3])}"
            for name, by in sorted(missing.items())
        )
    )


def test_the_exempt_package_markers_run_nothing_and_are_still_imported() -> None:
    hashed = hashed_files()

    for marker in sorted(PACKAGE_MARKERS):
        tree = ast.parse((ROOT / marker).read_text(encoding="utf-8"))
        assert len(tree.body) == 1 and ast.get_docstring(tree) is not None, (
            f"{marker} now runs code when an identity input imports it: hash it "
            "with its importers instead of exempting it"
        )
        assert marker in reached_files(), f"no identity input imports {marker} any more"
        assert marker not in hashed, f"{marker} is hashed: drop its exemption"


def test_the_walk_follows_lazy_relative_and_script_mode_imports() -> None:
    wheels = repository_imports(ROOT / "scripts/component_wheels.py")

    # Imported inside functions, as scripts.<name> and as a bare sibling.
    assert ROOT / "scripts/manifest_entrypoints.py" in wheels
    assert ROOT / "scripts/deploy_host_component.py" in wheels
    assert ROOT / "scripts/ci_unit_gate.py" in repository_imports(
        ROOT / "scripts/ci_gate.py"
    )
    assert ROOT / "tools/case_scheduler.py" in repository_imports(
        ROOT / "tools/run_fault_test_cases.py"
    )
    # The split module batch 90 found outside every identity is reported with
    # the scripts that import it.
    missing = unhashed(hashed_files() - {"scripts/release_attestation.py"})
    assert set(missing) == {"scripts/release_attestation.py"}
    assert "scripts/release_deploy.py" in missing["scripts/release_attestation.py"]


def test_the_walk_follows_runs_by_path_by_shell_and_through_make() -> None:
    staging = repository_runs(ROOT / "scripts/staging_deploy.py")

    for script in (
        "scripts/check-public-release.py",
        "scripts/restore_ci_candidate.py",
        "scripts/select-affected-tests.py",
        "scripts/verify-release-attestation.py",
    ):
        assert ROOT / script in staging, f"staging_deploy.py runs {script}"
    # A path join, and a module constant that a command uses later.
    assert ROOT / "deploy/dataplane/tools/verify_dataplane_executor.py" in (
        repository_runs(ROOT / "src/gpu_fault_release/regional_release_validation.py")
    )
    assert ROOT / "deploy/observability/install-amp-monitoring.sh" in (
        repository_runs(
            ROOT / "src/gpu_fault_release/regional_dataplane_observability.py"
        )
    )
    # make test-postgres-stress -> $(MAKE) $(if ...,test-postgres-parallel,...)
    gates = repository_runs(ROOT / "scripts/run_release_gates.py")
    assert ROOT / "scripts/run_postgres_shards.py" in gates
    assert MAKEFILE in gates
    assert ROOT / "scripts/generate-cleanup-inventory.py" in repository_runs(
        ROOT / "deploy/control-plane/tools/update-deployment-contracts.sh"
    )
    # A comment runs nothing; a set of file names is no command; a test that a
    # gate runs is its subject, not its machinery.
    assert ROOT / "deploy/hyperpod/deploy.sh" not in repository_runs(
        ROOT / "deploy/control-plane/tools/prune-wheel-configmaps.sh"
    )
    assert ROOT / "tools/scenario_coverage.py" not in repository_runs(
        ROOT / "scripts/ci_coverage_gate.py"
    )
    assert all(not _relative(path).startswith("tests/") for path in gates), (
        "a test a gate runs is no deploy-host input"
    )


def test_make_targets_picked_from_data_are_followed_explicitly() -> None:
    """The walk reads targets off source; one picked from data needs an entry."""

    reached_python = [
        ROOT / name
        for name in (*reached_files(), *hashed_files())
        if name.endswith(".py") and (ROOT / name).is_file()
    ]
    unlisted = sorted(
        {
            _relative(path)
            for path in reached_python
            if data_driven_make(path)
            and _relative(path) not in DATA_DRIVEN_MAKE_TARGETS
        }
    )

    assert unlisted == [], (
        f"these files run make targets they read from data: {unlisted}; name "
        "the targets in DATA_DRIVEN_MAKE_TARGETS"
    )
    for name, targets in DATA_DRIVEN_MAKE_TARGETS.items():
        assert data_driven_make(ROOT / name), f"{name} no longer picks make targets"
        missing = targets() - set(makefile_rules())
        assert missing == set(), f"{name} runs targets the Makefile lacks: {missing}"


def test_modules_split_from_deploy_scripts_are_hashed_with_their_importer() -> None:
    orchestration = file_set_identity(
        ROOT, deploy_source_identity.DEPLOY_HOST_ORCHESTRATION_INPUTS
    )["files"]
    protocol = file_set_identity(
        ROOT, deploy_source_identity.APPLICATION_IDENTITY_PROTOCOL_INPUTS
    )["files"]

    for module in (
        "scripts/release_attestation.py",
        "scripts/staging_deploy_inputs.py",
        "scripts/staging_source_snapshot.py",
        "scripts/ci_unit_gate.py",
        "scripts/ci_coverage_config.py",
        "scripts/ci_coverage_floors.py",
        "scripts/ci_coverage_gate.py",
        "scripts/ci_pytest_evidence.py",
        "tools/coverage_objectives.py",
        "tools/pytest_result_identity.py",
    ):
        assert module in orchestration, f"{module} must move the deploy-host identity"
    assert "scripts/manifest_entrypoints.py" in protocol, (
        "the wheel builder's manifest check must move the application identity"
    )
    # The deploy-host bundle build calls the same check, which names no
    # deploy-host manifest and so can neither change nor refuse that bundle.
    assert "deploy_host" not in manifest_entrypoints.MANIFEST_COMPONENTS, (
        "manifest_entrypoints.py now checks deploy-host manifests: make it a "
        "deploy-host bundle input (scripts/deploy_host_identity.py BUNDLE_INPUTS)"
    )


def test_what_the_deploy_and_release_paths_run_moves_the_deploy_host_identity() -> None:
    orchestration = file_set_identity(
        ROOT, deploy_source_identity.DEPLOY_HOST_ORCHESTRATION_INPUTS
    )["files"]
    checks = sorted(_relative(path) for path in ROOT.glob("scripts/check-*.py"))

    assert len(checks) > 10, checks
    for script in (
        "Makefile",
        "deploy/dataplane/tools/verify_dataplane_executor.py",
        "deploy/observability/install-amp-monitoring.sh",
        "scripts/restore_ci_candidate.py",
        "scripts/run_postgres_shards.py",
        "scripts/select-affected-tests.py",
        "scripts/verify-release-attestation.py",
        *checks,
    ):
        assert script in orchestration, f"{script} must move the deploy-host identity"
