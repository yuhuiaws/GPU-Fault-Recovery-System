from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parents[3]
ORDER_PATH = ROOT / "testcases" / "regional-execution-order.yaml"
CATALOG_PATH = ROOT / "testcases" / "fault-scenarios.yaml"


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
def ordered_case_ids() -> tuple[str, ...]:
    order = _load_yaml(ORDER_PATH)
    result: list[str] = []
    phases = order.get("phases")
    if not isinstance(phases, list):
        raise RegionalCaseContractError("regional execution order has no phases")
    for phase in sorted(phases, key=lambda item: int(item["sequence"])):
        for entry in phase["entries"]:
            if "case" in entry:
                result.append(str(entry["case"]))
                continue
            item = entry["range"]
            result.extend(
                f"GF-REGIONAL-{item['prefix']}-{number:03d}"
                for number in range(int(item["start"]), int(item["end"]) + 1)
            )
    if len(result) != len(set(result)):
        raise RegionalCaseContractError("regional execution order has duplicate cases")
    return tuple(result)


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


def formal_predecessor(case_id: str) -> str | None:
    ordered = ordered_case_ids()
    try:
        position = ordered.index(case_id)
    except ValueError as exc:
        raise RegionalCaseContractError(
            f"{case_id} is not in the formal regional execution order"
        ) from exc
    return ordered[position - 1] if position else None


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
