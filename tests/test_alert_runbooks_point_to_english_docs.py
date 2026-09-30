"""Every alert an operator receives must lead to English documentation.

``deploy/observability/amp-rules.yaml`` is the only thing an on-call operator
sees before they open a runbook: the ``runbook_url`` annotation travels in the
Alertmanager payload and into the Grafana panel links rendered from the same
file. Those references used to point at the Chinese operations manual; the
English edition under ``docs/en/`` is now the operator-facing target, and the
anchors are the alert names lowercased, exactly as GitHub slugs the ``###``
runbook cards. The two files are maintained by different people, so the
contract is checked here from the repository checkout rather than assumed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from gpu_fault_release.regional_dataplane_observability import (
    DATAPLANE_COLLECTOR_ALERT,
    DATAPLANE_COLLECTOR_RUNBOOK_URL,
)

ROOT = Path(__file__).resolve().parents[1]
AMP_RULES = ROOT / "deploy/observability/amp-rules.yaml"
DASHBOARDS = ROOT / "deploy/observability/dashboards"
ENGLISH_DOCS = "docs/en/"
ENGLISH_RUNBOOK = "docs/en/administrator-operations.md"
CJK = re.compile(r"[一-鿿]")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})")


def github_anchor(heading: str) -> str:
    """GitHub's heading slug: lowercase, spaces to hyphens, punctuation dropped."""
    slug = heading.strip().lower().replace(" ", "-")
    return "".join(
        character
        for character in slug
        if character.isalnum() or character in ("-", "_")
    )


def heading_anchors(document: Path) -> set[str]:
    anchors: set[str] = set()
    in_fence = False
    for line in document.read_text(encoding="utf-8").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match:
            anchors.add(github_anchor(match.group("text")))
    return anchors


def runbook_urls() -> dict[str, str]:
    payload = yaml.safe_load(AMP_RULES.read_text(encoding="utf-8"))
    urls: dict[str, str] = {}
    for group in payload["groups"]:
        for rule in group["rules"]:
            annotations = rule.get("annotations") or {}
            urls[str(rule["alert"])] = str(annotations.get("runbook_url") or "")
    return urls


def test_every_runbook_url_points_at_the_english_documentation() -> None:
    offenders = {
        alert: url
        for alert, url in runbook_urls().items()
        if not url.startswith(ENGLISH_DOCS) or CJK.search(url)
    }

    assert offenders == {}, (
        "runbook_url must point at docs/en/ (the Chinese manual is no longer the "
        f"operator-facing target): {offenders}"
    )


def test_every_runbook_anchor_is_a_heading_of_the_english_file() -> None:
    """Fails with the missing file or anchors named until the English edition lands."""
    missing_files: set[str] = set()
    missing_anchors: dict[str, str] = {}
    anchors_by_file: dict[str, set[str]] = {}
    for alert, url in sorted(runbook_urls().items()):
        path_text, _, anchor = url.partition("#")
        document = ROOT / path_text
        if not document.is_file():
            missing_files.add(path_text)
            continue
        available = anchors_by_file.setdefault(path_text, heading_anchors(document))
        if not anchor or anchor not in available:
            missing_anchors[alert] = url

    assert not missing_files, (
        "runbook_url targets do not exist in this checkout (the English edition "
        f"has not been merged yet): {sorted(missing_files)}"
    )
    assert missing_anchors == {}, (
        "these alerts name an anchor that is not a heading of the English file; "
        "each `### <AlertName>` runbook card must exist byte-identical to the "
        f"alert name: {json.dumps(missing_anchors, indent=2, sort_keys=True)}"
    )


def test_runbook_anchor_is_the_lowercased_alert_name() -> None:
    mismatched = {
        alert: url
        for alert, url in runbook_urls().items()
        if url != f"{ENGLISH_RUNBOOK}#{github_anchor(alert)}"
    }

    assert mismatched == {}, (
        f"runbook_url must be {ENGLISH_RUNBOOK}#<alert name lowercased>: {mismatched}"
    )


def test_rendered_dataplane_rule_points_at_the_english_documentation() -> None:
    expected = f"{ENGLISH_RUNBOOK}#{github_anchor(DATAPLANE_COLLECTOR_ALERT)}"

    assert DATAPLANE_COLLECTOR_RUNBOOK_URL == expected, (
        "the per-cluster data-plane rule is rendered at deploy time and never "
        f"passes through amp-rules.yaml: {DATAPLANE_COLLECTOR_RUNBOOK_URL!r}"
    )


def test_grafana_dashboards_reference_only_english_documentation() -> None:
    offenders: dict[str, list[str]] = {}
    for dashboard in sorted(DASHBOARDS.glob("*.json")):
        text = dashboard.read_text(encoding="utf-8")
        chinese_paths = sorted(set(re.findall(r"docs/[^\s\"']*[一-鿿][^\s\"']*", text)))
        if chinese_paths or CJK.search(text):
            offenders[dashboard.name] = chinese_paths or ["<CJK text in dashboard>"]

    assert offenders == {}, (
        "dashboard descriptions and links are operator-facing and must cite the "
        f"English documentation: {offenders}"
    )
