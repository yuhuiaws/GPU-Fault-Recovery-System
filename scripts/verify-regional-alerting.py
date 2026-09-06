#!/usr/bin/env python3
"""Verify the regional AMP alerting path.

The static checks prove that every compatibility PrometheusRule alert is
present in AMP, that every metric referenced by AMP survives ADOT's keep
filter, and that every AMP alert names a runbook section that exists. Optional
decoded AWS payloads additionally prove that the live AMP namespace has all
required groups and that Alertmanager has an SNS receiver.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

EXPECTED_GROUPS = {
    "gpu-fault-collector-health",
    "gpu-fault-control-plane-capacity",
    "gpu-fault-orchestration-invariants",
    "gpu-fault-policy-coverage",
    "gpu-fault-recovery-outcome",
    "gpu-fault-remote-command",
    "gpu-fault-telemetry-pipeline",
}
RUNBOOK_DOCUMENT = "docs/管理员日常运维.md"
# The triage cards live under one section and are titled with the bare alert
# name, so the anchor is derivable from the alert. Any other `###` heading in
# the document is a normal numbered subsection and is not an alert card.
RUNBOOK_HEADING = re.compile(r"^### (Gpu[A-Za-z0-9]+)\s*$", re.MULTILINE)

# Every metric family these modules publish is read out of the Store, so each
# control-plane replica publishes the same cluster-level value. The list is
# derived from their source rather than restated here, so a new family is
# covered the moment it is added. `collector_metrics.py` was missing for a while
# and that is exactly how `GpuFaultCollectorSilent` shipped grouping on a label
# its metric never carries. The other two `/metrics` producers --
# `metrics.py` and `metrics_sections.py` -- are per-replica (queue depths, lane
# occupancy, pool waits) and deliberately stay out of scope: for those, which
# Pod is affected is the point.
STORE_DERIVED_CONTRIBUTORS = (
    "src/gpu_fault/app/builtin_metric_contributors.py",
    "src/gpu_fault/app/collector_metrics.py",
)
# Adjacent string literals are how the long HELP lines are written, and some
# split inside a metric name. Collapsing the joint recovers the whole name.
LITERAL_CONTINUATION = re.compile(r'"\s*\n\s*f?"')
METRIC_NAME = re.compile(r"\bgpu_fault_[a-z0-9_]+")
# What an AMP rule may select. `otelcol_` is the collector's own pipeline
# telemetry, which travels the same keep filters as everything else and is just
# as unalertable when one of them drops it; a name-scoped check that only knew
# about `gpu_fault_` would have declared the new rules verified without looking.
AMP_METRIC = re.compile(r"\b(?:gpu_fault|otelcol)_[a-z0-9_]+")
# `_max` is the summary suffix `_append_summary` adds, not a metric of its own.
SERIES_SUFFIXES = ("_bucket", "_count", "_sum", "_max")
AGGREGATION = re.compile(
    r"\b(?:sum|max|min|avg|count|group|stddev|stdvar|topk|bottomk|quantile)\s*"
    r"(?:by|without)?\s*(?:\([^()]*\)\s*)?\("
)
BY_CLAUSE = re.compile(r"\b(by|without)\s*\(([^()]*)\)")
# amp-alertmanager.yaml groups on these, and ADOT relabels them per series, so an
# aggregation that does not carry them through collapses distinct control planes
# into one notification group.
GROUPING_LABELS = ("control_plane_cluster", "region")
# Added by the collector, not by the control plane. Left in place they join every
# alert's series identity, so an ADOT image bump resolves and re-fires every
# alert and invalidates any silence an operator has in place.
COLLECTOR_SCOPE_LABELS = ("otel_scope_name", "otel_scope_version")


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def markdown_anchor(heading: str) -> str:
    """Render the GitHub anchor a Markdown heading produces."""
    slug = heading.strip().lower().replace(" ", "-")
    return "".join(
        character for character in slug if character.isalnum() or character == "-"
    )


def runbook_defects(root: Path, rules: list[dict[str, Any]]) -> list[str]:
    """Require a two-way match between AMP alerts and runbook triage cards.

    An alert without a runbook is a page nobody knows how to action, and a
    runbook card without an alert is prose that silently rots because no
    on-call path ever reaches it. Both directions have to be checked here:
    the annotation itself is untyped free text, so nothing else in the
    repository notices when the two drift.
    """
    document = root / RUNBOOK_DOCUMENT
    text = document.read_text(encoding="utf-8")
    documented = set(RUNBOOK_HEADING.findall(text))

    defects: list[str] = []
    alerted: set[str] = set()
    for rule in rules:
        alert = rule["alert"]
        alerted.add(alert)
        url = (rule.get("annotations") or {}).get("runbook_url")
        if not url:
            defects.append(f"alert has no annotations.runbook_url: {alert}")
            continue
        expected = f"{RUNBOOK_DOCUMENT}#{markdown_anchor(alert)}"
        if url != expected:
            defects.append(
                f"alert {alert} runbook_url is {url!r}, expected {expected!r}"
            )
        elif alert not in documented:
            defects.append(
                f"alert {alert} points at a missing runbook section: {expected}"
            )
    for orphan in sorted(documented - alerted):
        defects.append(
            f"{RUNBOOK_DOCUMENT} documents {orphan} but no AMP alert fires it"
        )
    return defects


def annotation_defects(rules: list[dict[str, Any]]) -> list[str]:
    """Require every alert to carry both a summary and a description.

    ``runbook_url`` was already mandatory, but it only tells the on-call
    operator where to read; it does not travel in the notification body. The
    SNS message an administrator actually receives is built from the summary
    and the description, so an alert with only a summary pages someone with a
    one-line title and a repository path. Fifteen rules were in exactly that
    state, which is why this is enforced rather than left to review.

    A description that merely restates the summary is rejected too: it passes a
    presence check while leaving the payload as uninformative as it was.
    """
    defects: list[str] = []
    for rule in rules:
        alert = rule["alert"]
        annotations = rule.get("annotations") or {}
        summary = (annotations.get("summary") or "").strip()
        description = (annotations.get("description") or "").strip()
        if not summary:
            defects.append(f"alert has no annotations.summary: {alert}")
        if not description:
            defects.append(f"alert has no annotations.description: {alert}")
        elif description == summary:
            defects.append(
                f"alert {alert} description only repeats its summary; it should say "
                "what to check and what the boundary is"
            )
    return defects


def store_derived_families(root: Path) -> set[str]:
    """The metric families /metrics computes from the Store, read from source."""
    families: set[str] = set()
    for source in STORE_DERIVED_CONTRIBUTORS:
        text = (root / source).read_text(encoding="utf-8")
        families.update(METRIC_NAME.findall(LITERAL_CONTINUATION.sub("", text)))
    return families


def _series_family(metric: str) -> str:
    for suffix in SERIES_SUFFIXES:
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


def aggregation_defects(root: Path, rules: list[dict[str, Any]]) -> list[str]:
    """Require every Store-derived alert to aggregate, and to keep its grouping.

    A Store-derived family is a cluster-level fact that every control-plane
    replica publishes identically, so an unaggregated threshold on one turns a
    single condition into one alert entry per replica -- and the count grows with
    the Deployment. `GpuFaultRecoveryBlockedBacklog` shipped that way and sent
    six identical entries for one backlog of four workflows.

    The second half is why a bare `by` list is not enough on its own: dropping
    ``control_plane_cluster``/``region`` from the aggregation removes the labels
    Alertmanager groups on, which trades duplicate entries for merged ones.

    Rules on per-replica families -- a queue depth, a spool error, a collector
    heartbeat -- are deliberately out of scope: for those, which Pod is affected
    is the point.
    """
    families = store_derived_families(root)
    defects: list[str] = []
    for rule in rules:
        alert = rule["alert"]
        expr = " ".join(str(rule.get("expr", "")).split())
        derived = sorted(
            {
                metric
                for metric in METRIC_NAME.findall(expr)
                if _series_family(metric) in families
            }
        )
        if not derived:
            continue
        if not AGGREGATION.search(expr):
            defects.append(
                f"alert {alert} thresholds the Store-derived series "
                f"{', '.join(derived)} without aggregating them, so it fires "
                "once per control-plane replica"
            )
        elif not BY_CLAUSE.search(expr):
            # A bare `min(x)`/`max(x)` collapses to a single label-less series,
            # which is the mirror image of the duplicate-entry defect: instead of
            # one notification per replica, every control plane and region lands
            # in one group and only the first one is ever named. It aggregates,
            # so the check above is satisfied, and it carries no `by` clause, so
            # the loop below never looks at it.
            defects.append(
                f"alert {alert} aggregates the Store-derived series "
                f"{', '.join(derived)} with no `by` clause, so it drops the "
                "Alertmanager grouping labels: " + ", ".join(GROUPING_LABELS)
            )
        for keyword, labels in BY_CLAUSE.findall(expr):
            retained = {label.strip() for label in labels.split(",") if label.strip()}
            missing = [
                label
                for label in GROUPING_LABELS
                if (label not in retained) is (keyword == "by")
            ]
            if missing:
                defects.append(
                    f"alert {alert} aggregates with {keyword} ({labels.strip()}) and "
                    f"loses the Alertmanager grouping labels: {', '.join(missing)}"
                )
    return defects


def collector_label_defects(scrape: dict[str, Any]) -> list[str]:
    """Require the collector's own scope labels to be dropped before AMP.

    Alert identity is the full label set, so a label that changes when the
    collector is upgraded -- and nothing else about the fleet does -- resolves
    every firing alert and re-fires it as a new one. That both re-notifies and
    drops the silences an operator put in place mid-incident. No rule in
    amp-rules.yaml selects on them.
    """
    dropped: set[str] = set()
    for rule in scrape.get("metric_relabel_configs", []):
        if rule.get("action") != "labeldrop":
            continue
        pattern = re.compile(f"^(?:{rule['regex']})$")
        dropped.update(
            label for label in COLLECTOR_SCOPE_LABELS if pattern.match(label)
        )
    missing = [label for label in COLLECTOR_SCOPE_LABELS if label not in dropped]
    if not missing:
        return []
    return [
        "ADOT does not labeldrop the collector scope labels, so they enter alert "
        "identity: " + ", ".join(missing)
    ]


def keep_filter_defects(
    scrapes: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> list[str]:
    """Require every metric an AMP rule selects to survive ADOT's keep filters.

    The keep lists are the only thing that reaches AMP, so a family missing from
    all of them makes its alert dead configuration no matter how correct the
    expression is. A metric reaches AMP if *any* scrape job keeps it, which is
    why the filters are unioned rather than checked job by job: the
    self-telemetry job keeps `otelcol_*`, which the control-plane job drops, and
    vice versa for `gpu_fault_*`.
    """
    keeps = [
        rule["regex"]
        for scrape in scrapes
        for rule in scrape["metric_relabel_configs"]
        if rule["action"] == "keep"
    ]
    pattern = re.compile("^(?:" + "|".join(f"(?:{keep})" for keep in keeps) + ")$")

    defects: list[str] = []
    for rule in rules:
        for metric in AMP_METRIC.findall(str(rule["expr"])):
            if not pattern.match(metric.removesuffix("_bucket")):
                defects.append(
                    f"AMP rule metric is dropped by ADOT: {rule['alert']} uses {metric}"
                )
    return defects


def _static_defects(root: Path) -> list[str]:
    defects: list[str] = []
    amp = _yaml(root / "deploy/observability/amp-rules.yaml")
    amp_rules = [rule for group in amp["groups"] for rule in group["rules"]]
    amp_alerts = {rule["alert"] for rule in amp_rules}
    defects.extend(runbook_defects(root, amp_rules))
    defects.extend(annotation_defects(amp_rules))
    defects.extend(aggregation_defects(root, amp_rules))

    prometheus_rule_alerts: set[str] = set()
    path = root / "deploy/control-plane/regional/processor-alerts.yaml"
    for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
        if not document:
            continue
        spec = document.get("spec") or document
        for group in spec.get("groups", []):
            for rule in group.get("rules", []):
                prometheus_rule_alerts.add(rule["alert"])

    for alert in sorted(prometheus_rule_alerts - amp_alerts):
        defects.append(f"alert only exists in the unloaded PrometheusRule: {alert}")

    documents = list(
        yaml.safe_load_all(
            (root / "deploy/observability/adot-control-plane.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    collector = next(
        document
        for document in documents
        if document and document.get("kind") == "ConfigMap"
    )
    config = yaml.safe_load(collector["data"]["collector.yaml"])
    # Every scrape job, not just the first: the self-telemetry job carries the
    # only evidence that remote_write is delivering anything, and its keep filter
    # and its label drops are as load-bearing as the control-plane job's.
    scrapes = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    for scrape in scrapes:
        defects.extend(collector_label_defects(scrape))
    defects.extend(keep_filter_defects(scrapes, amp_rules))
    return defects


def _live_rule_defects(path: Path) -> tuple[list[str], set[str]]:
    payload = _yaml(path)
    groups = {group["name"] for group in (payload or {}).get("groups", [])}
    missing = EXPECTED_GROUPS - groups
    defects = (
        ["live AMP namespace is missing groups: " + ", ".join(sorted(missing))]
        if missing
        else []
    )
    return defects, groups


def _alertmanager_defects(path: Path) -> tuple[list[str], list[str]]:
    outer = _yaml(path) or {}
    config: Any = outer.get("alertmanager_config", outer)
    if isinstance(config, str):
        config = yaml.safe_load(config) or {}

    receivers: list[str] = []
    defects: list[str] = []
    for receiver in config.get("receivers", []):
        sns_configs = receiver.get("sns_configs") or []
        for sns in sns_configs:
            topic = str(sns.get("topic_arn", ""))
            if (
                not topic.startswith("arn:")
                or ":sns:" not in topic
                or "REPLACE_WITH_" in topic
            ):
                defects.append(
                    "Alertmanager SNS receiver has an invalid topic ARN: "
                    f"{receiver.get('name', '<unnamed>')}"
                )
                continue
            receivers.append(str(receiver.get("name", "<unnamed>")))
    if not receivers:
        defects.append("live Alertmanager has no valid SNS receiver")
    route_receiver = str((config.get("route") or {}).get("receiver", ""))
    if route_receiver not in receivers:
        defects.append(
            "Alertmanager root route does not select a valid SNS "
            f"receiver: {route_receiver or '<unset>'}"
        )
    return defects, receivers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--live-rules", type=Path)
    parser.add_argument("--live-alertmanager", type=Path)
    args = parser.parse_args()

    if bool(args.live_rules) != bool(args.live_alertmanager):
        parser.error("--live-rules and --live-alertmanager must be provided together")

    defects = _static_defects(args.repo_root)
    print(f"UNREACHABLE_ALERT_DEFECTS= {len(defects)}")

    if args.live_rules:
        live_defects, groups = _live_rule_defects(args.live_rules)
        defects.extend(live_defects)
        print("AMP_RULE_GROUPS=", ",".join(sorted(groups)))

        manager_defects, receivers = _alertmanager_defects(args.live_alertmanager)
        defects.extend(manager_defects)
        print(
            "ALERTMANAGER_SNS_RECEIVERS=",
            ",".join(sorted(receivers)),
        )

    for defect in defects:
        print(f"FAIL: {defect}")
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
