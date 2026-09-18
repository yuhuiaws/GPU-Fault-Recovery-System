"""Reviewed requirements, independent of how many acceptance cases already exist."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from gpu_fault.channel_registry import CHANNEL_REGISTRY
from gpu_fault.operation_registry import OPERATION_REGISTRY
from tools.run_fault_test_cases import UniqueKeyLoader

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Family = Literal[
    "fault-policy",
    "operation",
    "ingestion",
    "protocol",
    "security",
    "lifecycle",
    "availability",
    "recovery",
]


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    nodeid: Text
    expected: int = Field(default=1, ge=1)


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    id: Annotated[str, StringConstraints(pattern=r"^GF-REQ-[A-Z0-9-]+$")]
    family: Family
    title: Text
    source: list[Text] = Field(min_length=1)
    covers: list[Text] = Field(min_length=1)
    conditions: list[
        Literal["normal", "repeat", "timeout", "restart", "failure", "safety"]
    ] = Field(min_length=1)
    assertions: list[Text] = Field(min_length=1)
    cases: list[Text] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    runners: list[Text] = Field(default_factory=list)
    level: Literal["component", "regional"] = "component"
    critical: bool = False
    gap: Text | None = None

    @model_validator(mode="after")
    def unique_bindings(self) -> Requirement:
        for name in (
            "source",
            "covers",
            "conditions",
            "assertions",
            "cases",
            "runners",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{self.id} has duplicate {name}")
        if len(self.checks) != len({check.nodeid for check in self.checks}):
            raise ValueError(f"{self.id} has duplicate test selectors")
        if not self.gap and (
            not self.cases
            or not self.checks
            or (self.level == "regional" and not self.runners)
        ):
            raise ValueError(
                f"{self.id} needs implementation bindings or an explicit gap"
            )
        return self


class RequirementsDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int = Field(ge=1, le=1)
    requirements: list[Requirement] = Field(min_length=1)


def local_file(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"requirement reference leaves repository: {reference}")
    path = root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(
            f"requirement reference is not a local regular file: {reference}"
        )
    return path


def validate_check(root: Path, check: Check) -> None:
    filename, separator, selector = check.nodeid.partition("::")
    if (
        not separator
        or not filename.startswith("tests/")
        or not filename.endswith(".py")
    ):
        raise ValueError(f"requirement test selector is invalid: {check.nodeid}")
    path = local_file(root, filename)
    # Public test modules often re-export split cases. Imports are references,
    # not execution proof; the bound pytest collection must confirm exact IDs.
    name = selector.partition("[")[0].split("::")[0]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError(
            f"requirement test source cannot be parsed: {filename}"
        ) from exc
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    names.update(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    if name not in names:
        raise ValueError(f"requirement references an unknown test: {check.nodeid}")
    if "[" in selector and check.expected != 1:
        raise ValueError("an exact parameterized test must expect exactly one result")


def required_registry_keys() -> set[str]:
    return {f"operation:{operation}" for operation in OPERATION_REGISTRY} | {
        f"channel:{path}" for path in CHANNEL_REGISTRY
    }


def load_requirements(
    path: Path,
    *,
    root: Path,
    catalog: Mapping[str, Mapping[str, Any]],
    inventory: set[str] | None = None,
) -> tuple[Requirement, ...]:
    document = RequirementsDocument.model_validate(
        yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    )
    identifiers: set[str] = set()
    descriptions: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    coverage_keys: set[str] = set()
    for requirement in document.requirements:
        if requirement.id in identifiers:
            raise ValueError(f"duplicate requirement: {requirement.id}")
        identifiers.add(requirement.id)
        fingerprint = (
            tuple(sorted(requirement.covers)),
            tuple(sorted(requirement.assertions)),
        )
        if fingerprint in descriptions:
            raise ValueError(f"duplicate requirement scope: {requirement.id}")
        descriptions.add(fingerprint)
        coverage_keys.update(requirement.covers)
        for reference in (*requirement.source, *requirement.runners):
            local_file(root, reference)
        for runner in requirement.runners:
            if not runner.startswith(("scripts/e2e/", "scripts/perf/", "tools/")):
                raise ValueError(
                    f"requirement runner is outside acceptance tools: {runner}"
                )
        for check in requirement.checks:
            validate_check(root, check)
        for case_id in requirement.cases:
            case = catalog.get(case_id)
            if case is None:
                raise ValueError(f"requirement references an unknown case: {case_id}")
            evidence = case.get("evidence") or {}
            if case_id == "GF-REGIONAL-DESTR-004" or (
                evidence.get("verdict") == "SUPERSEDED"
            ):
                raise ValueError(
                    f"a retired case cannot cover a requirement: {case_id}"
                )
    expected = required_registry_keys() if inventory is None else inventory
    actual = {
        key for key in coverage_keys if key.startswith(("operation:", "channel:"))
    }
    if expected != actual:
        raise ValueError(
            "requirements do not match registries: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return tuple(document.requirements)
