#!/usr/bin/env python3
"""Verify the regional AMP alerting path.

The static checks prove that every compatibility PrometheusRule alert is
present in AMP and that every metric referenced by AMP survives ADOT's keep
filter. Optional decoded AWS payloads additionally prove that the live AMP
namespace has all required groups and that Alertmanager has an SNS receiver.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


EXPECTED_GROUPS = {
    "gpu-fault-collector-health",
    "gpu-fault-control-plane-capacity",
    "gpu-fault-orchestration-invariants",
    "gpu-fault-policy-coverage",
    "gpu-fault-remote-command",
}


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _static_defects(root: Path) -> list[str]:
    defects: list[str] = []
    amp = _yaml(root / "deploy/observability/amp-rules.yaml")
    amp_alerts = {rule["alert"] for group in amp["groups"] for rule in group["rules"]}

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
    scrape = config["receivers"]["prometheus"]["config"]["scrape_configs"][0]
    keep = next(
        rule for rule in scrape["metric_relabel_configs"] if rule["action"] == "keep"
    )["regex"]
    pattern = re.compile(f"^(?:{keep})$")

    for group in amp["groups"]:
        for rule in group["rules"]:
            for metric in re.findall(r"\bgpu_fault_[a-z0-9_]+", rule["expr"]):
                base_metric = metric.removesuffix("_bucket")
                if not pattern.match(base_metric):
                    defects.append(
                        "AMP rule metric is dropped by ADOT: "
                        f"{rule['alert']} uses {metric}"
                    )
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
