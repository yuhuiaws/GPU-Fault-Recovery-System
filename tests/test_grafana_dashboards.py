"""Grafana dashboards are generated from the AMP alert rules and checked in.

``scripts/build-grafana-dashboards.py`` renders one dashboard per alert group
plus an overview into ``deploy/observability/dashboards/``. The threshold lines
on the panels are read out of ``amp-rules.yaml``, so a threshold that moves in
the rule moves on the panel in the same commit. These tests hold the contract
between the three: every checked-in file is exactly what the generator renders,
every alert has a panel, every metric a panel plots is one the exporters emit
and ADOT keeps, and every panel talks to the AMP data source the provisioner
creates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = lazy_script_module(ROOT / "scripts/build-grafana-dashboards.py")
ALERTING = lazy_script_module(ROOT / "scripts/verify-regional-alerting.py")
DASHBOARDS = ROOT / "deploy/observability/dashboards"
AMP_RULES = ROOT / "deploy/observability/amp-rules.yaml"
ADOT = ROOT / "deploy/observability/adot-control-plane.yaml"
DATASOURCE = {"type": "prometheus", "uid": "gpu-fault-amp"}
CLUSTER_VARIABLE_QUERY = (
    "label_values(gpu_fault_processor_cluster_queue_depth, cluster_id)"
)
CLUSTER_SELECTOR = 'cluster_id=~"$cluster_id"'
SEVERITY_COLOR = {"critical": "red", "warning": "orange"}
# The test's own reading of a rule, kept independent of the generator's parser
# so a bug there cannot be mirrored here.
RULE_METRIC = re.compile(r"\b(?:gpu_fault|otelcol)_[a-z0-9_]+|\bup(?=\{)")
JOB_SELECTOR = re.compile(r'job="[^"]+"')
TRAILING_COMPARISON = re.compile(r"(>=|>|<)\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$")
GPU_FAULT_METRIC = re.compile(r"\bgpu_fault_[a-z0-9_]+")
SLUG = re.compile(r"^gpu-fault-[a-z0-9-]+$")


def checked_in_dashboards() -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(DASHBOARDS.glob("*.json"))
    }


def panels_of(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    return [panel for panel in dashboard["panels"] if panel["type"] != "row"]


def all_panels() -> list[tuple[str, dict[str, Any]]]:
    return [
        (uid, panel)
        for uid, dashboard in checked_in_dashboards().items()
        for panel in panels_of(dashboard)
    ]


def panel_exprs(panel: dict[str, Any]) -> list[str]:
    return [str(target["expr"]) for target in panel["targets"]]


def amp_rules() -> list[dict[str, Any]]:
    payload = yaml.safe_load(AMP_RULES.read_text(encoding="utf-8"))
    return [rule for group in payload["groups"] for rule in group["rules"]]


def amp_groups() -> list[str]:
    payload = yaml.safe_load(AMP_RULES.read_text(encoding="utf-8"))
    return [group["name"] for group in payload["groups"]]


def flat(expr: str) -> str:
    return " ".join(str(expr).split())


def covering_panels(rule: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Panels whose queries select every metric family and job the alert does."""
    expr = flat(rule["expr"])
    families = set(RULE_METRIC.findall(expr))
    jobs = set(JOB_SELECTOR.findall(expr))
    covering = []
    for uid, panel in all_panels():
        text = " ".join(panel_exprs(panel))
        if families <= set(RULE_METRIC.findall(text)) and all(
            job in text for job in jobs
        ):
            covering.append((uid, panel))
    return covering


def threshold_steps(panel: dict[str, Any]) -> list[dict[str, Any]]:
    return list(panel["fieldConfig"]["defaults"]["thresholds"]["steps"])


def test_checked_in_dashboards_equal_generator_output() -> None:
    rendered = MODULE.render_dashboard_files()

    assert sorted(rendered) == sorted(path.name for path in DASHBOARDS.glob("*.json"))
    for name, text in rendered.items():
        assert (DASHBOARDS / name).read_text(encoding="utf-8") == text, (
            f"{name} is stale; run scripts/build-grafana-dashboards.py"
        )


def test_check_mode_reports_a_stale_file(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(MODULE, "OUTPUT_DIR", tmp_path)
    assert MODULE.main(["--check"]) != 0, "an empty output directory must fail --check"

    assert MODULE.main([]) == 0
    assert MODULE.main(["--check"]) == 0

    stale = next(tmp_path.glob("*.json"))
    stale.write_text("{}\n", encoding="utf-8")
    assert MODULE.main(["--check"]) != 0, "a stale file must fail --check"


def test_one_dashboard_per_alert_group_plus_overview() -> None:
    uids = set(checked_in_dashboards())

    assert uids == {"gpu-fault-overview", *amp_groups()}


def test_dashboard_identity_fields() -> None:
    dashboards = checked_in_dashboards()

    for uid, dashboard in dashboards.items():
        assert SLUG.match(uid), f"dashboard uid is not a slug: {uid}"
        assert dashboard["uid"] == uid, f"{uid}: file name and uid differ"
        assert dashboard["editable"] is True, f"{uid}: not editable"
        assert dashboard["tags"] == ["gpu-fault"], f"{uid}: tags"
        assert dashboard["schemaVersion"] == 39, f"{uid}: schemaVersion for 10.4"
        assert dashboard["title"].startswith("GPU Fault"), f"{uid}: title"
    titles = [dashboard["title"] for dashboard in dashboards.values()]
    assert len(set(titles)) == len(titles), "dashboard titles must be unique"


def test_every_panel_and_target_uses_the_amp_datasource() -> None:
    panels = all_panels()

    assert panels, "no panels were generated"
    for uid, panel in panels:
        assert panel["datasource"] == DATASOURCE, f"{uid}/{panel['title']}"
        assert panel["targets"], f"{uid}/{panel['title']} has no query"
        for target in panel["targets"]:
            assert target["datasource"] == DATASOURCE, f"{uid}/{panel['title']}"
    for uid, dashboard in checked_in_dashboards().items():
        for variable in dashboard["templating"]["list"]:
            assert variable["datasource"] == DATASOURCE, f"{uid}/${variable['name']}"


def test_every_plotted_gpu_fault_metric_is_emitted_by_an_exporter() -> None:
    families = MODULE.exporter_metric_families()
    assert "gpu_fault_processor_queue_depth" in families, "inventory is empty"
    assert "gpu_fault_collector_silent_nodes" in families, (
        "collector_metrics.py must be part of the inventory"
    )

    unknown = {
        f"{uid}/{panel['title']}: {name}"
        for uid, panel in all_panels()
        for name in GPU_FAULT_METRIC.findall(" ".join(panel_exprs(panel)))
        if name not in families
    }
    assert unknown == set(), sorted(unknown)


def test_every_plotted_metric_survives_the_adot_keep_filter() -> None:
    documents = yaml.safe_load_all(ADOT.read_text(encoding="utf-8"))
    collector = next(
        document
        for document in documents
        if document and document.get("kind") == "ConfigMap"
    )
    config = yaml.safe_load(collector["data"]["collector.yaml"])
    scrapes = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    probes = [
        {"alert": f"{uid}/{panel['title']}", "expr": " ".join(panel_exprs(panel))}
        for uid, panel in all_panels()
    ]

    assert ALERTING.keep_filter_defects(scrapes, probes) == []


def test_every_alert_is_covered_by_a_panel_that_names_it() -> None:
    for rule in amp_rules():
        covering = covering_panels(rule)
        assert covering, f"no panel plots the metrics of {rule['alert']}"
        anchor = str(rule["annotations"]["runbook_url"])
        for uid, panel in covering:
            description = str(panel.get("description", ""))
            assert rule["alert"] in description, (
                f"{uid}/{panel['title']} covers {rule['alert']} but does not name it"
            )
            assert anchor in description, (
                f"{uid}/{panel['title']} does not carry the runbook anchor of "
                f"{rule['alert']}"
            )
            if rule.get("for"):
                assert f"for {rule['for']}" in description, (
                    f"{uid}/{panel['title']} does not carry the for duration of "
                    f"{rule['alert']}"
                )


def test_alert_thresholds_are_threshold_steps_on_the_covering_panels() -> None:
    thresholded = 0
    for rule in amp_rules():
        match = TRAILING_COMPARISON.search(flat(rule["expr"]))
        if match is None:
            continue
        thresholded += 1
        value = float(match.group(2))
        color = SEVERITY_COLOR[rule["labels"]["severity"]]
        for uid, panel in covering_panels(rule):
            steps = threshold_steps(panel)
            values = {step["value"] for step in steps}
            colors = {step["color"] for step in steps}
            assert value in values, (
                f"{uid}/{panel['title']} lacks the {value} threshold of {rule['alert']}"
            )
            assert color in colors, (
                f"{uid}/{panel['title']} lacks the {color} step of {rule['alert']}"
            )
    assert thresholded >= 30, f"only {thresholded} rules carry a numeric threshold"


def test_cluster_variable_on_every_dashboard_with_cluster_series() -> None:
    cluster_families = MODULE.cluster_labelled_families()
    assert "gpu_fault_processor_cluster_queue_depth" in cluster_families
    assert "gpu_fault_collector_silent_nodes" in cluster_families
    assert "gpu_fault_processor_queue_depth" not in cluster_families

    for uid, dashboard in checked_in_dashboards().items():
        variables = {
            variable["name"]: variable for variable in dashboard["templating"]["list"]
        }
        exprs = [expr for panel in panels_of(dashboard) for expr in panel_exprs(panel)]
        uses_cluster = any(
            name in cluster_families
            for expr in exprs
            for name in GPU_FAULT_METRIC.findall(expr)
        )
        if uses_cluster:
            variable = variables.get("cluster_id")
            assert variable is not None, (
                f"{uid} plots per-cluster series without $cluster_id"
            )
            assert variable["query"] == CLUSTER_VARIABLE_QUERY, uid
            assert variable["multi"] is True and variable["includeAll"] is True, uid
        for expr in exprs:
            if any(name in cluster_families for name in GPU_FAULT_METRIC.findall(expr)):
                assert CLUSTER_SELECTOR in expr, f"{uid}: {expr}"
        for name in ("control_plane_cluster", "region"):
            if any(f"${name}" in expr for expr in exprs):
                assert name in variables, f"{uid} references ${name} without a variable"


def test_dashboards_contain_no_absolute_urls() -> None:
    for path in sorted(DASHBOARDS.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, path.name


def test_overview_plots_the_closed_loop_essentials() -> None:
    overview = checked_in_dashboards()["gpu-fault-overview"]
    text = " ".join(
        expr for panel in panels_of(overview) for expr in panel_exprs(panel)
    )

    for name in (
        "gpu_fault_incidents_by_state",
        "gpu_fault_workflow_total",
        "gpu_fault_processor_queue_depth",
        "gpu_fault_processor_cluster_queue_depth",
        "gpu_fault_processor_cluster_queue_oldest_age_seconds",
        "gpu_fault_remediation_budget_cluster_active_claims",
        "gpu_fault_remediation_budget_cluster_limit",
        "gpu_fault_remediation_budget_cluster_waiting_workflows",
        "gpu_fault_store_io_in_flight",
        "gpu_fault_store_io_max_in_flight",
        "gpu_fault_notification_outbox_depth",
        "gpu_fault_orphan_workflows",
        "gpu_fault_capacity_largest_cluster_node_count",
        "gpu_fault_capacity_managed_node_count",
        'count by (alertname) (ALERTS{alertstate="firing"})',
    ):
        assert name in text, f"overview is missing {name}"


def test_panel_ids_are_unique_and_grid_positions_are_set() -> None:
    for uid, dashboard in checked_in_dashboards().items():
        ids = [panel["id"] for panel in dashboard["panels"]]
        assert len(set(ids)) == len(ids), f"{uid}: duplicate panel ids"
        for panel in dashboard["panels"]:
            position = panel["gridPos"]
            assert set(position) == {"h", "w", "x", "y"}, f"{uid}/{panel['title']}"
            assert 0 <= position["x"] and position["x"] + position["w"] <= 24, (
                f"{uid}/{panel['title']} overflows the 24-column grid"
            )
