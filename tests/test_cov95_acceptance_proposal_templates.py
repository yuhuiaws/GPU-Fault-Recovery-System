from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests._cov95_acceptance_plan import A, B, PlanInputs
from tools.regional_acceptance_plan import (
    untrusted_dependency_proposal_to_override_template,
)


def proposal(executor: str = "codex-read-only") -> dict[str, Any]:
    return {
        "trusted": False,
        "results": [
            {
                "case_id": B,
                "executor": executor,
                "depends_on": [A],
                "locks": [{"resource": "unit-evidence", "mode": "shared"}],
            }
        ],
    }


@pytest.mark.parametrize(
    "executor",
    [
        "codex-read-only",
        "codex-manual",
        "human",
        "command",
        "pytest",
        "controlled-live-executor",
        "do-not-run",
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_proposal_template_never_promotes_trust_or_copies_a_mutating_executor(
    tmp_path: Path, executor: str, wrapped: bool
) -> None:
    inputs = PlanInputs(tmp_path)
    value = proposal(executor)
    if wrapped:
        value = {"trusted": False, "proposal": value}
    template = untrusted_dependency_proposal_to_override_template(
        value, order_path=inputs.order_path, catalog_path=inputs.catalog_path
    )
    assert template["reviewed"] is False
    assert template["reviewed_by"] == template["reviewed_at"] == ""
    assert (
        template["order_sha256"]
        == hashlib.sha256(inputs.order_path.read_bytes()).hexdigest()
    )
    constraints = template["cases"][B]
    assert constraints["depends_on"] == [A]
    assert constraints["locks"] == [{"resource": "unit-evidence", "mode": "shared"}]
    if executor in {"codex-read-only", "codex-manual"}:
        assert constraints["executor"] == "codex-manual"
    else:
        assert "executor" not in constraints
    inputs.override(
        template["cases"],
        **{key: value for key, value in template.items() if key != "cases"},
    )
    with pytest.raises(ValueError, match="reviewed: true"):
        inputs.compile(override_path=inputs.override_path)


@pytest.mark.parametrize(
    "fault", ["root-trust", "nested-trust", "duplicate", "executor", "unserializable"]
)
def test_invalid_untrusted_proposal_does_not_yield_an_override(
    tmp_path: Path, fault: str
) -> None:
    inputs = PlanInputs(tmp_path)
    value = proposal()
    if fault == "root-trust":
        value["trusted"] = True
    elif fault == "nested-trust":
        value = {"trusted": False, "proposal": {"trusted": True}}
    elif fault == "duplicate":
        value["results"].append(dict(value["results"][0]))
    elif fault == "executor":
        value["results"][0]["executor"] = "unrestricted-shell"
    else:
        value["untrusted-extra"] = object()
    with pytest.raises(ValueError):
        untrusted_dependency_proposal_to_override_template(
            value, order_path=inputs.order_path, catalog_path=inputs.catalog_path
        )
