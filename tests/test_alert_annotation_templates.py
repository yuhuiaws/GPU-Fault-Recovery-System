"""Alert annotation templates render sensibly in the Alertmanager email.

Two production emails (2026-09-10) showed what an unformatted template does:

* ``{{ $value }}`` of an ``increase(...[15m])`` rendered as
  ``1.0092621468926555 fault-layer event(s)`` because ``increase`` extrapolates
  to a non-integer; every ``$value`` must therefore be piped through a
  formatter (``printf "%.0f"`` for counts, ``printf "%.1f"`` for ratios,
  ``humanizeDuration`` for seconds).
* ``GpuFaultPeriodicServiceErrors`` is ``A or B`` where only ``B`` carries a
  ``job`` label, so ``{{ $labels.job }}`` rendered empty on the ``A`` branch.
  A label read by an annotation must be carried by every branch of the
  expression or be wrapped in an ``{{ if $labels.X }}`` guard.

The branch check is textual and deliberately shallow: the expression is
normalised to one line and split on the ``or`` / ``unless`` keywords, the
label set of a branch is the union of its ``by (...)`` clauses, and a branch
without any ``by (...)`` (a plain selector such as ``up{...} == 0``, which keeps
every series label) is treated as carrying every label. It does not parse
PromQL, so it cannot see through ``without (...)``, ``on (...) group_left``
label copies, or an ``or`` nested inside a function call. The real promtool
run (``tests/test_alert_rules_promtool.py``) parses the template syntax; this
file only proves the values and labels the templates read exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
CHECKER = lazy_script_module(ROOT / "scripts/check-alert-rules.py")

TEMPLATE_ACTION = re.compile(r"\{\{-?(.*?)-?\}\}", re.DOTALL)
LABEL_REFERENCE = re.compile(r"\$labels\.([A-Za-z_][A-Za-z0-9_]*)")
GUARDED_LABEL = re.compile(
    r"\{\{-?\s*(?:if|with)\s+\$labels\.([A-Za-z_][A-Za-z0-9_]*)\s*-?\}\}"
)
BY_CLAUSE = re.compile(r"\bby\s*\(([^)]*)\)")
BRANCH_SPLIT = re.compile(r"\s+(?:or|unless)\s+")


def _rules() -> list[tuple[str, str, dict[str, object]]]:
    """``(file, alert, rule)`` for every alert in both repository rule files."""

    rows: list[tuple[str, str, dict[str, object]]] = []
    for name, document in CHECKER.rule_documents(CHECKER.RULE_FILES):
        for group in document["groups"]:
            for rule in group["rules"]:
                if "alert" in rule:
                    rows.append((name, str(rule["alert"]), rule))
    return rows


def _annotation_text(rule: dict[str, object]) -> str:
    annotations = rule.get("annotations") or {}
    assert isinstance(annotations, dict), (
        f"annotations must be a mapping: {annotations!r}"
    )
    return "\n".join(str(value) for value in annotations.values())


def _branch_labels(branch: str) -> set[str] | None:
    """Labels a branch can carry; ``None`` when it keeps every series label."""

    clauses = BY_CLAUSE.findall(branch)
    if not clauses:
        return None
    labels: set[str] = set()
    for clause in clauses:
        labels.update(part.strip() for part in clause.split(",") if part.strip())
    return labels


def test_every_value_reference_is_piped_through_a_formatter() -> None:
    bare: list[str] = []
    for name, alert, rule in _rules():
        for action in TEMPLATE_ACTION.findall(_annotation_text(rule)):
            if "$value" in action and "|" not in action:
                bare.append(f"{name}:{alert}: {{{{{action}}}}}")

    assert bare == [], (
        "an unformatted $value renders as an extrapolated float in the email "
        "(pipe it through printf/humanize/humanizeDuration): " + "; ".join(bare)
    )


def test_labels_read_by_annotations_exist_on_every_expression_branch() -> None:
    defects: list[str] = []
    for name, alert, rule in _rules():
        text = _annotation_text(rule)
        referenced = set(LABEL_REFERENCE.findall(text))
        if not referenced:
            continue
        guarded = set(GUARDED_LABEL.findall(text))
        expression = " ".join(str(rule["expr"]).split())
        branches = BRANCH_SPLIT.split(expression)
        for label in sorted(referenced - guarded):
            for index, branch in enumerate(branches):
                carried = _branch_labels(branch)
                if carried is not None and label not in carried:
                    defects.append(
                        f"{name}:{alert} reads $labels.{label} but branch "
                        f"{index + 1} of its expression does not aggregate by it: "
                        f"{branch}"
                    )

    assert defects == [], (
        "a label missing from one branch renders empty in the email; guard it "
        "with {{ if $labels.X }} or carry it on every branch: " + "; ".join(defects)
    )


def test_the_periodic_service_alert_names_the_branch_that_fired() -> None:
    """The lease branch has no ``job`` label, so the text must not assume one."""

    rule = next(
        rule for _, alert, rule in _rules() if alert == "GpuFaultPeriodicServiceErrors"
    )
    description = str(rule["annotations"]["description"])  # type: ignore[index]

    assert "{{ if $labels.job }}" in description
    assert "{{ else }}" in description and "{{ end }}" in description
    assert (
        "task lease" in description.split("{{ else }}", 1)[1].split("{{ end }}", 1)[0]
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("up{job='x'} == 0", [None]),
        ("max by (a, b) (m) > 0 or max by (a) (n) > 0", [{"a", "b"}, {"a"}]),
        ("absent(m) or min by (region) (m) > 1", [None, {"region"}]),
    ],
)
def test_the_textual_branch_split_matches_its_documented_limits(
    expression: str, expected: list[set[str] | None]
) -> None:
    assert [_branch_labels(b) for b in BRANCH_SPLIT.split(expression)] == expected
