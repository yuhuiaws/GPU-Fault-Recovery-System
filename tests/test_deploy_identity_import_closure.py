"""Every repository module an identity input imports is an identity input too.

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

The walk follows every ``import`` of a hashed Python file -- at module level or
inside a function, in either branch of an ``if __package__`` -- to the
repository modules outside ``src`` (``scripts``, ``tools``, ``tests`` and a
script's sibling modules) and requires each to be hashed as well. ``src``
modules are bound by the component wheels' import closures instead
(``tests/test_component_wheel_closures.py``).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from scripts import deploy_host_identity, deploy_source_identity, manifest_entrypoints
from scripts.release_identity import build_release_identity, file_set_identity

ROOT = Path(__file__).resolve().parents[1]
#: Repository packages outside ``src`` that an identity input can import by name.
PACKAGES = frozenset({"scripts", "tools", "tests"})
#: The exemptions: package markers an import passes through. Each holds only its
#: docstring, so importing it runs nothing (checked below), and it needs no
#: identity.
PACKAGE_MARKERS = frozenset({"scripts/__init__.py", "tools/__init__.py"})


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


def imported_modules(hashed: set[str]) -> dict[str, str]:
    """Each repository module the hashed files reach, transitively, with the
    first file found importing it."""

    importers: dict[str, str] = {}
    pending = [ROOT / name for name in sorted(hashed) if name.endswith(".py")]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in sorted(repository_imports(path)):
            importers.setdefault(_relative(module), _relative(path))
            pending.append(module)
    return importers


def unhashed_imports(hashed: set[str]) -> dict[str, str]:
    return {
        module: importer
        for module, importer in imported_modules(hashed).items()
        if module not in hashed and module not in PACKAGE_MARKERS
    }


def test_every_module_an_identity_input_imports_is_an_identity_input() -> None:
    missing = unhashed_imports(hashed_files())

    assert missing == {}, (
        "an identity input imports modules no identity hashes, so a change to one "
        "of them alone deploys nothing; list each in the identity input of its "
        "importer (scripts/deploy_source_identity.py, "
        "scripts/deploy_host_identity.py, config/release-identity.yaml): "
        + "; ".join(f"{module} <- {importer}" for module, importer in missing.items())
    )


def test_the_exempt_package_markers_run_nothing_and_are_still_imported() -> None:
    hashed = hashed_files()
    imported = imported_modules(hashed)

    for marker in sorted(PACKAGE_MARKERS):
        tree = ast.parse((ROOT / marker).read_text(encoding="utf-8"))
        assert len(tree.body) == 1 and ast.get_docstring(tree) is not None, (
            f"{marker} now runs code when an identity input imports it: hash it "
            "with its importers instead of exempting it"
        )
        assert marker in imported, f"no identity input imports {marker} any more"
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
    # the script that imports it.
    hashed = hashed_files() - {"scripts/release_attestation.py"}
    assert unhashed_imports(hashed) == {
        "scripts/release_attestation.py": "scripts/release_deploy.py"
    }


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
