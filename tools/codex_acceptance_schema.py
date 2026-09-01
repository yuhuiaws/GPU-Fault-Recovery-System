from __future__ import annotations

from collections.abc import Collection, Sequence


def _string_array_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": {
            "type": "string",
            "minLength": 1,
        },
    }


def build_analysis_schema(
    case_ids: Sequence[str],
    *,
    schema_version: int,
    acceptance_statuses: Sequence[str],
    result_fields: Collection[str],
    test_step_fields: Collection[str],
    step_id_pattern: str,
    test_step_kinds: Sequence[str],
    test_step_statuses: Sequence[str],
) -> dict[str, object]:
    evidence_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source", "observation"],
        "properties": {
            "source": {"type": "string", "minLength": 1},
            "observation": {"type": "string", "minLength": 1},
        },
    }
    test_step_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(test_step_fields),
        "properties": {
            "step_id": {
                "type": "string",
                "pattern": step_id_pattern,
            },
            "kind": {
                "type": "string",
                "enum": list(test_step_kinds),
            },
            "description": {"type": "string", "minLength": 1},
            "depends_on": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "pattern": step_id_pattern,
                },
            },
            "expected": {"type": "string", "minLength": 1},
            "executor_ref": {"type": "string", "minLength": 1},
            "status": {
                "type": "string",
                "enum": list(test_step_statuses),
            },
            "evidence": {
                "type": "array",
                "items": evidence_schema,
            },
        },
    }
    result_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(result_fields),
        "properties": {
            "case_id": {
                "type": "string",
                "enum": list(case_ids),
            },
            "status": {
                "type": "string",
                "enum": list(acceptance_statuses),
            },
            "summary": {"type": "string", "minLength": 1},
            "failure_details": _string_array_schema(),
            "reproduction": _string_array_schema(),
            "evidence": {
                "type": "array",
                "items": evidence_schema,
            },
            "test_process": {
                "type": "array",
                "minItems": 1,
                "items": test_step_schema,
            },
            "affected_dependents": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": list(case_ids),
                },
            },
            "blockers": _string_array_schema(),
            "human_actions": _string_array_schema(),
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "results"],
        "properties": {
            "schema_version": {"type": "integer", "const": schema_version},
            "results": {
                "type": "array",
                "minItems": len(case_ids),
                "maxItems": len(case_ids),
                "items": result_schema,
            },
        },
    }
