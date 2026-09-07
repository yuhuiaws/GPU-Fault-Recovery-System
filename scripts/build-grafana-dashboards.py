"""Render the Grafana dashboards for the GPU fault control plane.

The panel table lives in ``grafana_dashboard_catalog.py``; this script turns it
into ``deploy/observability/dashboards/<uid>.json`` and, on the way, reads
``deploy/observability/amp-rules.yaml`` so that every alert becomes part of the
panel that plots its metric families:

* an alert whose expression ends in ``> N`` / ``>= N`` / ``< N`` contributes a
  threshold step at ``N`` (red for ``critical``, orange for ``warning``), so
  the line an operator sees is the line the rule fires on;
* every alert contributes its name, severity, ``for`` duration and runbook
  anchor to the panel description, so the panel leads to the triage card.

Output is deterministic (sorted keys, ids in table order) and ``--check``
fails when a checked-in file differs from what the table renders, which is how
``make grafana-dashboards-check`` keeps the JSON and the rules in one commit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from grafana_dashboard_catalog import DASHBOARDS, Dashboard, Panel

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "deploy" / "observability" / "dashboards"
AMP_RULES = ROOT / "deploy" / "observability" / "amp-rules.yaml"
# Every module that renders a ``/metrics`` family the control-plane scrape job
# can see. ``collector_metrics.py`` is the one an earlier inventory forgot.
EXPORTER_SOURCES = (
    "src/gpu_fault/app/metrics.py",
    "src/gpu_fault/app/metrics_sections.py",
    "src/gpu_fault/app/builtin_metric_contributors.py",
    "src/gpu_fault/app/collector_metrics.py",
    "src/gpu_fault/completion_metrics_server.py",
)
DATASOURCE = {"type": "prometheus", "uid": "gpu-fault-amp"}
SCHEMA_VERSION = 39  # Grafana 10.4
TAGS = ["gpu-fault"]
RUNBOOK_DOCUMENT = "docs/管理员日常运维.md"
SEVERITY_COLOR = {"critical": "red", "warning": "orange"}
OK_COLOR = "green"
PANEL_HEIGHT = 8
PANELS_PER_LINE = 4
GRID_WIDTH = 24
# Adjacent string literals split long metric names across lines; joining them
# recovers the whole name before the regexes run.
LITERAL_CONTINUATION = re.compile(r"[\"']\s*\n\s*f?[\"']")
METRIC_NAME = re.compile(r"\bgpu_fault_[a-z0-9_]+")
CLUSTER_LABELLED = re.compile(
    r"(gpu_fault_[a-z0-9_]+)(?:\{\{cluster_id=|\{\{\{labels\}\}\})"
)
MULTI_SERIES_TYPE = re.compile(r"# TYPE (gpu_fault_[a-z0-9_]+) (?:summary|histogram)")
MULTI_SERIES_SUFFIXES = ("_sum", "_count", "_max", "_bucket")
# ``f"gpu_fault_workflow_dispatch_{name}"`` over a tuple of suffix literals:
# the prefix is joined with every quoted identifier in the same file.
FORMATTED_PREFIX = re.compile(r"\b(gpu_fault_[a-z0-9_]+_)\{")
IDENTIFIER_LITERAL = re.compile(r"\"([a-z][a-z0-9_]*)\"")
RULE_METRIC = re.compile(r"\b(?:gpu_fault|otelcol)_[a-z0-9_]+|\bup(?=\{)")
JOB_SELECTOR = re.compile(r'job="[^"]+"')
TRAILING_COMPARISON = re.compile(r"(>=|>|<)\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$")
VARIABLE_QUERIES = {
    "control_plane_cluster": (
        "Control plane",
        'label_values(up{job="gpu-fault-control-plane"}, control_plane_cluster)',
    ),
    "region": (
        "Region",
        'label_values(up{job="gpu-fault-control-plane",'
        'control_plane_cluster=~"$control_plane_cluster"}, region)',
    ),
    "cluster_id": (
        "Cluster",
        "label_values(gpu_fault_processor_cluster_queue_depth, cluster_id)",
    ),
}


@dataclass(frozen=True)
class AlertRule:
    name: str
    group: str
    expr: str
    duration: str | None
    severity: str
    runbook: str
    families: frozenset[str]
    jobs: frozenset[str]
    threshold: tuple[str, float] | None

    def covers(self, panel: Panel) -> bool:
        """Whether ``panel`` plots every family and job selector this rule reads."""
        text = " ".join(target.expr for target in panel.targets)
        return self.families <= frozenset(RULE_METRIC.findall(text)) and all(
            job in text for job in self.jobs
        )

    def describe(self) -> str:
        parts = [self.severity]
        if self.duration:
            parts.append(f"for {self.duration}")
        if self.threshold is not None:
            operator, value = self.threshold
            parts.append(f"{operator} {value:g}")
        return f"Alert {self.name} ({', '.join(parts)}): {self.runbook}"


def _normalised_source(relative: str) -> str:
    return LITERAL_CONTINUATION.sub("", (ROOT / relative).read_text(encoding="utf-8"))


def exporter_metric_families() -> frozenset[str]:
    """Every ``gpu_fault_*`` series name the exporters can render.

    Summary and histogram families are declared once and rendered with the
    ``_sum``/``_count``/``_max``/``_bucket`` suffixes, so those are added for
    every family whose ``# TYPE`` says so. A name assembled from an f-string
    prefix and a table of suffix literals is expanded from that same file.
    """
    names: set[str] = set()
    for relative in EXPORTER_SOURCES:
        text = _normalised_source(relative)
        names.update(METRIC_NAME.findall(text))
        for family in MULTI_SERIES_TYPE.findall(text):
            names.update(family + suffix for suffix in MULTI_SERIES_SUFFIXES)
        prefixes = set(FORMATTED_PREFIX.findall(text))
        if prefixes:
            suffixes = set(IDENTIFIER_LITERAL.findall(text))
            names.update(prefix + suffix for prefix in prefixes for suffix in suffixes)
    return frozenset(names)


def cluster_labelled_families() -> frozenset[str]:
    """Families rendered with a ``cluster_id`` label, read from the exporters."""
    names: set[str] = set()
    for relative in EXPORTER_SOURCES:
        names.update(CLUSTER_LABELLED.findall(_normalised_source(relative)))
    return frozenset(names)


def parse_threshold(expr: str) -> tuple[str, float] | None:
    """``(operator, value)`` when the rule ends in a numeric comparison.

    ``==`` is a state test rather than a threshold, and a rule that ends in a
    join (``and on (...) (...)``) has no single number to draw.
    """
    match = TRAILING_COMPARISON.search(expr)
    if match is None:
        return None
    return match.group(1), float(match.group(2))


def load_alert_rules(path: Path = AMP_RULES) -> list[AlertRule]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules: list[AlertRule] = []
    for group in payload["groups"]:
        for rule in group["rules"]:
            expr = " ".join(str(rule["expr"]).split())
            rules.append(
                AlertRule(
                    name=str(rule["alert"]),
                    group=str(group["name"]),
                    expr=expr,
                    duration=str(rule["for"]) if rule.get("for") else None,
                    severity=str(rule["labels"]["severity"]),
                    runbook=str(rule["annotations"]["runbook_url"]),
                    families=frozenset(RULE_METRIC.findall(expr)),
                    jobs=frozenset(JOB_SELECTOR.findall(expr)),
                    threshold=parse_threshold(expr),
                )
            )
    return rules


def _severity_rank(color: str) -> int:
    return 1 if color == SEVERITY_COLOR["critical"] else 0


def threshold_steps(rules: Iterable[AlertRule]) -> list[dict[str, Any]]:
    """Grafana threshold steps for the rules attached to one panel.

    A step colours everything at or above its value, so a ``>`` rule puts its
    severity colour at the value on a green base, while a ``<`` rule colours
    the base and turns green at the value. Two rules at one value keep the
    more severe colour.
    """
    base = OK_COLOR
    by_value: dict[float, str] = {}
    for rule in rules:
        if rule.threshold is None:
            continue
        operator, value = rule.threshold
        color = SEVERITY_COLOR[rule.severity]
        if operator == "<":
            base = color if _severity_rank(color) >= _severity_rank(base) else base
            by_value.setdefault(value, OK_COLOR)
            continue
        current = by_value.get(value)
        if current is None or _severity_rank(color) > _severity_rank(current):
            by_value[value] = color
    steps: list[dict[str, Any]] = [{"color": base, "value": None}]
    steps.extend(
        {"color": by_value[value], "value": value} for value in sorted(by_value)
    )
    return steps


def _description(panel: Panel, rules: Sequence[AlertRule]) -> str:
    lines = [panel.description] if panel.description else []
    lines.extend(rule.describe() for rule in rules)
    return "\n\n".join(lines)


def _targets(panel: Panel, instant: bool) -> list[dict[str, Any]]:
    return [
        {
            "datasource": DATASOURCE,
            "expr": target.expr,
            "instant": instant,
            "legendFormat": target.legend or "__auto",
            "range": not instant,
            "refId": chr(ord("A") + index),
        }
        for index, target in enumerate(panel.targets)
    ]


def render_panel(
    panel: Panel,
    rules: Sequence[AlertRule],
    panel_id: int,
    grid: dict[str, int],
) -> dict[str, Any]:
    attached = [rule for rule in rules if rule.covers(panel)]
    steps = threshold_steps(attached)
    has_threshold = len(steps) > 1
    rendered: dict[str, Any] = {
        "datasource": DATASOURCE,
        "description": _description(panel, attached),
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "thresholds": {"mode": "absolute", "steps": steps},
                "unit": panel.unit,
            },
            "overrides": [],
        },
        "gridPos": grid,
        "id": panel_id,
        "title": panel.title,
        "type": panel.kind,
    }
    if panel.kind == "stat":
        rendered["targets"] = _targets(panel, instant=True)
        rendered["options"] = {
            "colorMode": "value" if has_threshold else "none",
            "graphMode": "none",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "value_and_name",
        }
        return rendered
    rendered["targets"] = _targets(panel, instant=False)
    rendered["fieldConfig"]["defaults"]["custom"] = {
        "fillOpacity": 10,
        "lineWidth": 1,
        "showPoints": "never",
        "spanNulls": False,
        "thresholdsStyle": {"mode": "line" if has_threshold else "off"},
    }
    rendered["options"] = {
        "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        "tooltip": {"mode": "multi", "sort": "desc"},
    }
    return rendered


def _line_widths(count: int) -> list[int]:
    """Column widths that spread ``count`` panels over as few lines as needed."""
    lines = max(1, math.ceil(count / PANELS_PER_LINE))
    per_line = math.ceil(count / lines)
    widths: list[int] = []
    remaining = count
    while remaining > 0:
        on_this_line = min(per_line, remaining)
        widths.extend([GRID_WIDTH // on_this_line] * on_this_line)
        remaining -= on_this_line
    return widths


def render_panels(
    dashboard: Dashboard, rules: Sequence[AlertRule]
) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    next_id = 1
    y = 0
    for row in dashboard.rows:
        panels.append(
            {
                "collapsed": False,
                "gridPos": {"h": 1, "w": GRID_WIDTH, "x": 0, "y": y},
                "id": next_id,
                "panels": [],
                "title": row.title,
                "type": "row",
            }
        )
        next_id += 1
        y += 1
        x = 0
        for panel, width in zip(row.panels, _line_widths(len(row.panels))):
            if x + width > GRID_WIDTH:
                x = 0
                y += PANEL_HEIGHT
            grid = {"h": PANEL_HEIGHT, "w": width, "x": x, "y": y}
            panels.append(render_panel(panel, rules, next_id, grid))
            next_id += 1
            x += width
        y += PANEL_HEIGHT
    return panels


def _variable(name: str) -> dict[str, Any]:
    label, query = VARIABLE_QUERIES[name]
    return {
        "allValue": ".*",
        "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
        "datasource": DATASOURCE,
        "definition": query,
        "hide": 0,
        "includeAll": True,
        "label": label,
        "multi": True,
        "name": name,
        "options": [],
        "query": query,
        "refresh": 2,
        "regex": "",
        "sort": 1,
        "type": "query",
    }


def render_variables(dashboard: Dashboard) -> list[dict[str, Any]]:
    """Only the variables the dashboard's queries reference, in dependency order."""
    text = " ".join(
        target.expr
        for row in dashboard.rows
        for panel in row.panels
        for target in panel.targets
    )
    return [_variable(name) for name in VARIABLE_QUERIES if f"${name}" in text]


def render_dashboard(
    dashboard: Dashboard, rules: Sequence[AlertRule]
) -> dict[str, Any]:
    return {
        "annotations": {"list": []},
        "description": (
            f"Alert group {dashboard.group}; every panel carries the thresholds "
            f"and runbook anchors of the alerts it plots (see {RUNBOOK_DOCUMENT} §8)."
            if dashboard.group
            else f"Closed-loop essentials across every alert group (see {RUNBOOK_DOCUMENT} §8)."
        ),
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": render_panels(dashboard, rules),
        "refresh": "1m",
        "schemaVersion": SCHEMA_VERSION,
        "tags": TAGS,
        "templating": {"list": render_variables(dashboard)},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": dashboard.title,
        "uid": dashboard.uid,
        "version": 1,
        "weekStart": "",
    }


def render_dashboard_files() -> dict[str, str]:
    """``{file name: JSON text}`` for every dashboard in the catalog."""
    rules = load_alert_rules()
    return {
        f"{dashboard.uid}.json": json.dumps(
            render_dashboard(dashboard, rules),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
        for dashboard in DASHBOARDS
    }


def uncovered_alerts(rules: Sequence[AlertRule]) -> list[str]:
    panels = [
        panel
        for dashboard in DASHBOARDS
        for row in dashboard.rows
        for panel in row.panels
    ]
    return [
        rule.name for rule in rules if not any(rule.covers(panel) for panel in panels)
    ]


def check(rendered: dict[str, str], output_dir: Path) -> list[str]:
    problems: list[str] = []
    for name, text in rendered.items():
        path = output_dir / name
        if not path.is_file():
            problems.append(f"missing: {path}")
        elif path.read_text(encoding="utf-8") != text:
            problems.append(f"stale: {path}")
    for path in sorted(output_dir.glob("*.json")) if output_dir.is_dir() else []:
        if path.name not in rendered:
            problems.append(f"not generated by the catalog: {path}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a checked-in dashboard differs from the catalog",
    )
    args = parser.parse_args(argv)
    uncovered = uncovered_alerts(load_alert_rules())
    if uncovered:
        print("alerts without a panel: " + ", ".join(uncovered), file=sys.stderr)
        return 1
    rendered = render_dashboard_files()
    output_dir = OUTPUT_DIR
    if args.check:
        problems = check(rendered, output_dir)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            print(
                "Grafana dashboards are stale; run "
                "python scripts/build-grafana-dashboards.py",
                file=sys.stderr,
            )
            return 1
        print(f"Grafana dashboards are current ({len(rendered)} files).")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (output_dir / name).write_text(text, encoding="utf-8")
        print(f"Wrote {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
