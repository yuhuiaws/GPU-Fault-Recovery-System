"""Nested Alertmanager routes are validated at every depth (M-28).

The routing tree resolves a page on the matching leaf route's receiver, so a
misrouted nested route drops or misdirects the page exactly like a bad root
route. The verifier must therefore recurse into ``route["routes"]`` rather than
only inspecting the root.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = lazy_script_module(ROOT / "scripts/verify-regional-alerting.py")

VALID_TOPIC = "arn:aws:sns:us-west-2:000000000000:gpu-fault-alerts"


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "alertmanager.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _base(route: dict) -> dict:
    return {
        "receivers": [
            {"name": "sns-primary", "sns_configs": [{"topic_arn": VALID_TOPIC}]}
        ],
        "route": route,
    }


def test_valid_nested_routes_pass(tmp_path: Path) -> None:
    payload = _base(
        {
            "receiver": "sns-primary",
            "routes": [
                # Inherits the parent receiver by omitting its own.
                {"match": {"severity": "warning"}},
                # Declares a valid receiver.
                {"match": {"severity": "critical"}, "receiver": "sns-primary"},
            ],
        }
    )

    defects, receivers = MODULE.alertmanager_defects(_write(tmp_path, payload))

    assert receivers == ["sns-primary"]
    assert defects == []


def test_nested_route_with_unknown_receiver_is_rejected(tmp_path: Path) -> None:
    payload = _base(
        {
            "receiver": "sns-primary",
            "routes": [
                {"match": {"severity": "warning"}, "receiver": "sns-primary"},
                {"match": {"severity": "critical"}, "receiver": "typo-receiver"},
            ],
        }
    )

    defects, _receivers = MODULE.alertmanager_defects(_write(tmp_path, payload))

    assert any("typo-receiver" in defect for defect in defects), defects
    assert any("routes[1]" in defect for defect in defects), defects


def test_deeply_nested_route_with_unknown_receiver_is_rejected(tmp_path: Path) -> None:
    payload = _base(
        {
            "receiver": "sns-primary",
            "routes": [
                {
                    "match": {"team": "gpu"},
                    "routes": [
                        {"match": {"severity": "page"}, "receiver": "ghost-receiver"}
                    ],
                }
            ],
        }
    )

    defects, _receivers = MODULE.alertmanager_defects(_write(tmp_path, payload))

    assert any("ghost-receiver" in defect for defect in defects), defects
