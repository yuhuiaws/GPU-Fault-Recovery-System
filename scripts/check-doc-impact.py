"""Require documentation review for documentation-sensitive changes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT / "docs" / "code-doc-contracts.yaml"
MIN_REASON_LENGTH = 12
IMPACT_FIELD = "Documentation-Impact"
REASON_FIELD = "Documentation-Impact-Reason"


@dataclass(frozen=True)
class CoverageRule:
    root: str
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class Contract:
    identifier: str
    description: str
    paths: tuple[str, ...]
    documents: tuple[str, ...]

    def matches(self, path: str) -> bool:
        return any(glob_matches(pattern, path) for pattern in self.paths)


@dataclass(frozen=True)
class ContractSet:
    coverage: tuple[CoverageRule, ...]
    contracts: tuple[Contract, ...]


@dataclass(frozen=True)
class Acknowledgement:
    impact: str | None
    reason: str | None
    failures: tuple[str, ...] = ()


def _relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be repository-relative: {value}")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(_relative_path(item, field=f"{field}[]") for item in value)


def load_contracts(path: Path) -> ContractSet:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("contract file must contain a mapping")
    if raw.get("version") != 1:
        raise ValueError("contract file version must be 1")

    coverage_raw = raw.get("coverage")
    if not isinstance(coverage_raw, list) or not coverage_raw:
        raise ValueError("coverage must be a non-empty list")
    coverage: list[CoverageRule] = []
    for index, item in enumerate(coverage_raw):
        if not isinstance(item, dict):
            raise ValueError(f"coverage[{index}] must be a mapping")
        root = _relative_path(item.get("root"), field=f"coverage[{index}].root")
        extensions = _string_list(
            item.get("extensions"),
            field=f"coverage[{index}].extensions",
        )
        if any(not extension.startswith(".") for extension in extensions):
            raise ValueError(f"coverage[{index}].extensions must start with '.'")
        coverage.append(CoverageRule(root=root, extensions=extensions))

    contracts_raw = raw.get("contracts")
    if not isinstance(contracts_raw, list) or not contracts_raw:
        raise ValueError("contracts must be a non-empty list")
    contracts: list[Contract] = []
    identifiers: set[str] = set()
    for index, item in enumerate(contracts_raw):
        if not isinstance(item, dict):
            raise ValueError(f"contracts[{index}] must be a mapping")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*",
            identifier,
        ):
            raise ValueError(f"contracts[{index}].id must be a lowercase identifier")
        if identifier in identifiers:
            raise ValueError(f"duplicate contract id: {identifier}")
        identifiers.add(identifier)
        description = item.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"contracts[{index}].description must be a string")
        contracts.append(
            Contract(
                identifier=identifier,
                description=description,
                paths=_string_list(
                    item.get("paths"),
                    field=f"contracts[{index}].paths",
                ),
                documents=_string_list(
                    item.get("documents"),
                    field=f"contracts[{index}].documents",
                ),
            )
        )
    return ContractSet(tuple(coverage), tuple(contracts))


def _glob_regex(pattern: str) -> re.Pattern[str]:
    pieces = ["^"]
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            pieces.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            pieces.append(".*")
            index += 2
        elif pattern[index] == "*":
            pieces.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            pieces.append("[^/]")
            index += 1
        else:
            pieces.append(re.escape(pattern[index]))
            index += 1
    pieces.append("$")
    return re.compile("".join(pieces))


def glob_matches(pattern: str, path: str) -> bool:
    return bool(_glob_regex(pattern).fullmatch(path))


def validate_contracts(root: Path, contract_set: ContractSet) -> list[str]:
    failures: list[str] = []
    for contract in contract_set.contracts:
        for document in contract.documents:
            target = root / document
            if not target.is_file():
                failures.append(
                    f"{contract.identifier}: document does not exist: {document}"
                )

    all_files = [
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ]
    for contract in contract_set.contracts:
        if not any(contract.matches(path) for path in all_files):
            failures.append(
                f"{contract.identifier}: no repository file matches "
                f"{list(contract.paths)}"
            )

    for rule in contract_set.coverage:
        coverage_root = root / rule.root
        if not coverage_root.is_dir():
            failures.append(f"coverage root does not exist: {rule.root}")
            continue
        for path in coverage_root.rglob("*"):
            if not path.is_file() or path.suffix not in rule.extensions:
                continue
            relative = path.relative_to(root).as_posix()
            if not any(
                contract.matches(relative) for contract in contract_set.contracts
            ):
                failures.append(f"documentation contract does not cover: {relative}")
    return failures


def _git(
    root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def _is_git_repository(root: Path) -> bool:
    result = _git(root, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def changed_files_from_git(
    root: Path,
    *,
    base: str | None,
    head: str,
) -> tuple[list[str], str | None]:
    if not _is_git_repository(root):
        if base:
            raise ValueError("cannot evaluate --base outside a Git repository")
        return [], "Git metadata unavailable; only contract structure was checked"

    if base:
        result = _git(
            root,
            "diff",
            "--name-only",
            "--diff-filter=ACMRD",
            f"{base}...{head}",
            "--",
        )
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or "git diff failed")
        return result.stdout.splitlines(), None

    result = _git(
        root,
        "diff",
        "--name-only",
        "--diff-filter=ACMRD",
        "HEAD",
        "--",
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "git diff HEAD failed")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    if untracked.returncode != 0:
        raise ValueError(untracked.stderr.strip() or "git ls-files failed")
    return [*result.stdout.splitlines(), *untracked.stdout.splitlines()], None


def _field_values(body: str, name: str) -> list[str]:
    pattern = re.compile(rf"(?im)^[ \t]*{re.escape(name)}[ \t]*:[ \t]*(.*?)[ \t]*$")
    return [match.group(1).strip() for match in pattern.finditer(body)]


def acknowledgement_from_body(body: str) -> Acknowledgement:
    failures: list[str] = []
    impact_values = _field_values(body, IMPACT_FIELD)
    reason_values = _field_values(body, REASON_FIELD)
    if len(impact_values) > 1:
        failures.append(f"{IMPACT_FIELD} must appear at most once")
    if len(reason_values) > 1:
        failures.append(f"{REASON_FIELD} must appear at most once")
    return Acknowledgement(
        impact=impact_values[0].lower() if len(impact_values) == 1 else None,
        reason=reason_values[0] if len(reason_values) == 1 else None,
        failures=tuple(failures),
    )


def acknowledgement_from_event(path: Path | None) -> Acknowledgement:
    if path is None or not path.is_file():
        return Acknowledgement(None, None)
    try:
        event: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Acknowledgement(
            None,
            None,
            (f"cannot read GitHub event: {exc}",),
        )
    pull_request = event.get("pull_request") if isinstance(event, dict) else None
    body = pull_request.get("body") if isinstance(pull_request, dict) else None
    return acknowledgement_from_body(body or "")


def acknowledgement_from_environment(
    fallback: Acknowledgement,
) -> Acknowledgement:
    impact = os.getenv("GPU_FAULT_DOC_IMPACT")
    reason = os.getenv("GPU_FAULT_DOC_IMPACT_REASON")
    if impact is None and reason is None:
        return fallback
    return Acknowledgement(
        impact.strip().lower() if impact is not None else None,
        reason,
    )


def valid_no_docs_acknowledgement(
    acknowledgement: Acknowledgement,
) -> tuple[bool, list[str]]:
    failures = list(acknowledgement.failures)
    if acknowledgement.impact != "none":
        failures.append(f"set '{IMPACT_FIELD}: none' when no related document changed")
    reason = (acknowledgement.reason or "").strip()
    if len(reason) < MIN_REASON_LENGTH:
        failures.append(
            f"{REASON_FIELD} must explain the exemption "
            f"in at least {MIN_REASON_LENGTH} characters"
        )
    placeholders = {"required", "todo", "n/a", "none", "填写", "待填写"}
    if reason.lower() in placeholders:
        failures.append(f"{REASON_FIELD} cannot be a placeholder")
    return not failures, failures


def check_changed_files(
    changed_files: list[str],
    contract_set: ContractSet,
    acknowledgement: Acknowledgement,
) -> tuple[list[str], list[str]]:
    normalized = sorted(
        {
            _relative_path(path, field="changed file")
            for path in changed_files
            if path.strip()
        }
    )
    failures: list[str] = []
    satisfied: list[str] = []
    unsatisfied: list[tuple[Contract, list[str]]] = []
    changed = set(normalized)

    for contract in contract_set.contracts:
        matched = [path for path in normalized if contract.matches(path)]
        if not matched:
            continue
        if changed.intersection(contract.documents):
            satisfied.append(contract.identifier)
        else:
            unsatisfied.append((contract, matched))

    if not unsatisfied:
        return failures, satisfied

    accepted, acknowledgement_failures = valid_no_docs_acknowledgement(acknowledgement)
    if accepted:
        satisfied.extend(
            f"{contract.identifier} (explicit no-docs acknowledgement)"
            for contract, _ in unsatisfied
        )
        return failures, satisfied

    failures.extend(acknowledgement_failures)
    for contract, matched in unsatisfied:
        failures.append(
            f"{contract.identifier}: changed {', '.join(matched)}; "
            "update one of " + ", ".join(contract.documents)
        )
    return failures, satisfied


def _event_path(argument: Path | None) -> Path | None:
    if argument is not None:
        return argument
    value = os.getenv("GITHUB_EVENT_PATH")
    return Path(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--changed-files-from", type=Path)
    parser.add_argument("--event-path", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--acknowledge-no-docs", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args()

    root = args.root.resolve()
    contracts_path = (
        args.contracts.resolve()
        if args.contracts is not None
        else root / "docs" / "code-doc-contracts.yaml"
    )
    try:
        contract_set = load_contracts(contracts_path)
    except ValueError as exc:
        print(f"documentation impact check failed: {exc}")
        return 1

    failures = validate_contracts(root, contract_set)
    if failures:
        for failure in failures:
            print(f"documentation impact check failed: {failure}")
        return 1
    if args.validate_only:
        print("documentation impact check passed: contract structure is valid")
        return 0

    explicit_changes = list(args.changed_file)
    if args.changed_files_from is not None:
        try:
            explicit_changes.extend(
                args.changed_files_from.read_text(encoding="utf-8").splitlines()
            )
        except OSError as exc:
            print(
                "documentation impact check failed: "
                f"cannot read changed-files list: {exc}"
            )
            return 1
    if explicit_changes and args.base:
        print(
            "documentation impact check failed: "
            "--changed-file/--changed-files-from cannot be combined with --base"
        )
        return 1

    note: str | None = None
    try:
        if explicit_changes:
            changed_files = explicit_changes
        else:
            changed_files, note = changed_files_from_git(
                root,
                base=args.base,
                head=args.head,
            )
    except (OSError, ValueError) as exc:
        print(f"documentation impact check failed: {exc}")
        return 1

    acknowledgement = acknowledgement_from_environment(
        acknowledgement_from_event(_event_path(args.event_path))
    )
    if args.acknowledge_no_docs:
        acknowledgement = Acknowledgement("none", args.reason)
    try:
        failures, satisfied = check_changed_files(
            changed_files,
            contract_set,
            acknowledgement,
        )
    except ValueError as exc:
        print(f"documentation impact check failed: {exc}")
        return 1
    if failures:
        for failure in failures:
            print(f"documentation impact check failed: {failure}")
        return 1

    detail = f"; {note}" if note else ""
    if satisfied:
        print("documentation impact check passed: " + ", ".join(satisfied) + detail)
    else:
        print(
            "documentation impact check passed: "
            f"no documentation-sensitive changes{detail}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
