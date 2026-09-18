from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from tools.regional_acceptance_plan import (
    RegionalAcceptancePlan,
    compile_regional_acceptance_plan,
)

A = "GF-REGIONAL-BOOT-011"
B = "GF-REGIONAL-CAP-001"
RETIRED = "GF-REGIONAL-BOOT-006"


def catalog_case(case_id: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "title": "Local compiler fixture",
        "category": "regional",
        "level": "unit",
        "risk": "non-destructive",
        "automation": "manual",
        "problem": "Inspect a local prerequisite.",
        "injection": "No environment mutation.",
        "expected": ["The prerequisite is satisfied."],
        "procedure": "Review existing local evidence.",
    }


class PlanInputs:
    def __init__(self, root: Path) -> None:
        self.order_path = root / "order.yaml"
        self.catalog_path = root / "catalog.yaml"
        self.override_path = root / "override.yaml"
        self.order: Any = {
            "schema_version": 1,
            "phases": [
                {
                    "sequence": 0,
                    "name": "Local",
                    "maintenance_window": "offline",
                    "entries": [{"case": A}, {"case": B}],
                }
            ],
            "do_not_run": [{"case": RETIRED, "reason": "Removed fixture."}],
        }
        self.catalog: Any = {
            "schema_version": 1,
            "test_cases": [catalog_case(A), catalog_case(B), catalog_case(RETIRED)],
        }
        self.write()

    def write(self) -> None:
        self.order_path.write_text(yaml.safe_dump(self.order, sort_keys=False))
        self.catalog_path.write_text(yaml.safe_dump(self.catalog, sort_keys=False))

    def compile(self, **kwargs: Any) -> RegionalAcceptancePlan:
        return compile_regional_acceptance_plan(
            order_path=self.order_path, catalog_path=self.catalog_path, **kwargs
        )

    def override(self, constraints: dict[str, Any], **updates: Any) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "reviewed": True,
            "reviewed_by": "unit-reviewer",
            "reviewed_at": "2026-09-12T00:00:00Z",
            "order_sha256": hashlib.sha256(self.order_path.read_bytes()).hexdigest(),
            "catalog_sha256": hashlib.sha256(
                self.catalog_path.read_bytes()
            ).hexdigest(),
            "cases": constraints,
            **updates,
        }
        self.override_path.write_text(yaml.safe_dump(value, sort_keys=False))
        return value
