from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import re
from typing import Any


EXECUTION_SCOPE_ENV = "GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE"
SELECTION_REFERENCE_ENV = "GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE"
FORMAL_SCOPE = "formal"
SELECTIVE_SCOPE = "selective"
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


@dataclass(frozen=True)
class AcceptanceScope:
    mode: str
    selection_reference: str | None

    @property
    def selective(self) -> bool:
        return self.mode == SELECTIVE_SCOPE

    def plan_fields(self) -> dict[str, str | None]:
        return {
            "execution_scope": self.mode,
            "selection_reference": self.selection_reference,
        }

    def result_fields(self) -> dict[str, str | bool | None]:
        return {
            **self.plan_fields(),
            "formal_sequence_satisfied": not self.selective,
        }


def current_acceptance_scope(
    environment: Mapping[str, str] | None = None,
) -> AcceptanceScope:
    values = environment if environment is not None else os.environ
    mode = values.get(EXECUTION_SCOPE_ENV, FORMAL_SCOPE).strip().lower()
    reference = values.get(SELECTION_REFERENCE_ENV, "").strip()
    if mode not in {FORMAL_SCOPE, SELECTIVE_SCOPE}:
        raise RuntimeError(
            f"{EXECUTION_SCOPE_ENV} must be {FORMAL_SCOPE!r} or {SELECTIVE_SCOPE!r}"
        )
    if mode == FORMAL_SCOPE:
        if reference:
            raise RuntimeError(
                f"{SELECTION_REFERENCE_ENV} is only valid for selective execution"
            )
        return AcceptanceScope(mode=mode, selection_reference=None)
    if not reference:
        raise RuntimeError(
            f"{SELECTION_REFERENCE_ENV} is required for selective execution"
        )
    if _REFERENCE_PATTERN.fullmatch(reference) is None:
        raise RuntimeError(
            f"{SELECTION_REFERENCE_ENV} must be a 1-128 character audit reference"
        )
    return AcceptanceScope(mode=mode, selection_reference=reference)


def scoped_case_evidence(value: Any) -> Any:
    if not isinstance(value, dict) or "case_id" not in value or "verdict" not in value:
        return value
    scope = current_acceptance_scope()
    if not scope.selective:
        return value
    document = dict(value)
    for key, expected in scope.result_fields().items():
        if key in document and document[key] != expected:
            raise RuntimeError(f"case evidence scope drifted at {key}")
        document[key] = expected
    return document
