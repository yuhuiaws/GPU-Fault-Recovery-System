from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped,unused-ignore]


ROOT = Path(__file__).resolve().parents[3]
ORDER_PATH = ROOT / "testcases" / "regional-execution-order.yaml"
CATALOG_PATH = ROOT / "testcases" / "fault-scenarios.yaml"
# The interpreters a catalog ``command`` may name when it is really just a
# pytest invocation in disguise (``python3 -m pytest ...``).
_PYTHON_INTERPRETERS = frozenset({"python", "python3"})


class RegionalCaseContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegionalCaseMetadata:
    case_id: str
    title: str
    category: str
    level: str
    risk: str
    automation: str
    procedure: str
    predecessor: str | None

    @property
    def confirmation(self) -> str:
        short = self.case_id.removeprefix("GF-REGIONAL-").replace("-", "")
        return f"{short}_EXECUTE"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegionalCaseContractError(f"{path} is not a YAML mapping")
    return cast(dict[str, Any], value)


@lru_cache(maxsize=1)
def expanded_order() -> tuple[tuple[str, ...], dict[str, str], frozenset[str]]:
    """Expand the order file once: ordered ids, explicit predecessors, DO_NOT_RUN.

    A ``case:`` entry may carry ``predecessor: GF-REGIONAL-XXX-NNN`` when the
    evidence it needs is not the case that happens to sit in front of it (the
    warm-spare and destructive chains fan out from a few anchor cases, and the
    order notes have always said so in prose). ``range`` entries cannot carry
    one: a range is by definition a serial run.
    """
    order = _load_yaml(ORDER_PATH)
    result: list[str] = []
    explicit: dict[str, str] = {}
    phases = order.get("phases")
    if not isinstance(phases, list):
        raise RegionalCaseContractError("regional execution order has no phases")
    for phase in sorted(phases, key=lambda item: int(item["sequence"])):
        for entry in phase["entries"]:
            if "case" in entry:
                case_id = str(entry["case"])
                result.append(case_id)
                predecessor = entry.get("predecessor")
                if predecessor is not None:
                    if not isinstance(predecessor, str) or not predecessor.startswith(
                        "GF-REGIONAL-"
                    ):
                        raise RegionalCaseContractError(
                            f"{case_id} predecessor must be a GF-REGIONAL case id"
                        )
                    explicit[case_id] = predecessor
                continue
            if "predecessor" in entry:
                raise RegionalCaseContractError(
                    "regional execution order range entries cannot carry a predecessor"
                )
            item = entry["range"]
            result.extend(
                f"GF-REGIONAL-{item['prefix']}-{number:03d}"
                for number in range(int(item["start"]), int(item["end"]) + 1)
            )
    if len(result) != len(set(result)):
        raise RegionalCaseContractError("regional execution order has duplicate cases")
    retired = frozenset(str(entry["case"]) for entry in (order.get("do_not_run") or []))
    overlap = retired & set(result)
    if overlap:
        raise RegionalCaseContractError(
            f"regional execution order lists DO_NOT_RUN cases in a phase: {sorted(overlap)}"
        )
    position = {case_id: index for index, case_id in enumerate(result)}
    for case_id, predecessor in explicit.items():
        if predecessor not in position:
            raise RegionalCaseContractError(
                f"{case_id} predecessor {predecessor} is not in the formal "
                "regional execution order"
            )
        if position[predecessor] >= position[case_id]:
            raise RegionalCaseContractError(
                f"{case_id} predecessor {predecessor} does not run before it"
            )
    return tuple(result), explicit, retired


def ordered_case_ids() -> tuple[str, ...]:
    return expanded_order()[0]


def explicit_predecessors() -> dict[str, str]:
    """The ``predecessor:`` overrides the order file declares, by case id."""
    return dict(expanded_order()[1])


def do_not_run_case_ids() -> frozenset[str]:
    return expanded_order()[2]


@lru_cache(maxsize=1)
def _catalog_cases() -> dict[str, dict[str, Any]]:
    catalog = _load_yaml(CATALOG_PATH)
    cases = catalog.get("test_cases")
    if not isinstance(cases, list):
        raise RegionalCaseContractError("fault scenario catalog has no test_cases")
    result = {
        str(item["id"]): cast(dict[str, Any], item)
        for item in cases
        if isinstance(item, dict) and str(item.get("id", "")).startswith("GF-REGIONAL-")
    }
    return result


def is_pytest_wrapper(case: dict[str, Any]) -> bool:
    """True when running the case only runs pytest, so it leaves no case evidence.

    ``automation: pytest`` cases and ``automation: command`` cases whose command
    is a bare ``python3 -m pytest ...`` are both executed by
    ``tools/run_fault_test_cases.py`` and report into ``artifacts/fault``; neither
    ever writes ``cases/<id>/<id>.json`` under an acceptance run directory. A
    live case placed behind one of them in the order could therefore never
    satisfy its formal predecessor, so the positional fallback skips them.
    """
    automation = case.get("automation")
    if automation == "pytest":
        return True
    if automation != "command":
        return False
    command = case.get("command")
    if not isinstance(command, list) or len(command) < 3:
        return False
    return (
        Path(str(command[0])).name in _PYTHON_INTERPRETERS
        and command[1] == "-m"
        and command[2] == "pytest"
    )


def pytest_wrapper_case_ids() -> tuple[str, ...]:
    return tuple(
        case_id for case_id, case in _catalog_cases().items() if is_pytest_wrapper(case)
    )


def formal_predecessor(case_id: str) -> str | None:
    """The case whose PASS evidence ``case_id`` must read before it may run.

    An explicit ``predecessor:`` in the order file wins. Otherwise the answer is
    positional: the nearest earlier case in the formal order that can actually
    produce ``cases/<id>/<id>.json`` evidence, i.e. not a pytest wrapper.
    """
    ordered, explicit, retired = expanded_order()
    if case_id in retired:
        raise RegionalCaseContractError(
            f"{case_id} is DO_NOT_RUN in the formal regional execution order"
        )
    try:
        position = ordered.index(case_id)
    except ValueError as exc:
        raise RegionalCaseContractError(
            f"{case_id} is not in the formal regional execution order"
        ) from exc
    if case_id in explicit:
        return explicit[case_id]
    catalog = _catalog_cases()
    for candidate in reversed(ordered[:position]):
        case = catalog.get(candidate)
        if case is not None and is_pytest_wrapper(case):
            continue
        return candidate
    return None


def case_metadata(case_id: str) -> RegionalCaseMetadata:
    try:
        value = _catalog_cases()[case_id]
    except KeyError as exc:
        raise RegionalCaseContractError(
            f"{case_id} is not in the regional case catalog"
        ) from exc
    procedure = value.get("procedure")
    if not isinstance(procedure, str) or not procedure:
        raise RegionalCaseContractError(f"{case_id} has no procedure")
    return RegionalCaseMetadata(
        case_id=case_id,
        title=str(value["title"]),
        category=str(value["category"]),
        level=str(value["level"]),
        risk=str(value["risk"]),
        automation=str(value["automation"]),
        procedure=procedure,
        predecessor=formal_predecessor(case_id),
    )


def case_evidence_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / case_id / f"{case_id}.json"


def predecessor_path(
    run_dir: Path,
    case_id: str,
    explicit_path: str | Path | None,
) -> tuple[str | None, Path | None]:
    predecessor = formal_predecessor(case_id)
    if predecessor is None:
        return None, None
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
    else:
        path = case_evidence_path(run_dir, predecessor).resolve()
    return predecessor, path
