"""``gpu-fault-admin status`` names an open remediation storm breaker."""

from __future__ import annotations

from gpu_fault_release.regional_admin_checks_coverage import remediation_storm_lines
from gpu_fault_release.regional_admin_commands import _control_api_findings


def _report(storm: dict | None) -> dict:
    return {
        "clusters": {
            "gpu-a": {"remediation_storm": storm},
            "gpu-b": {"remediation_storm": {"state": "CLOSED"}},
        }
    }


OPEN = {
    "state": "TRIPPED",
    "tripped_at": "2026-10-29T12:00:00+00:00",
    "trip_nodes": 3,
    "trip_domains": 3,
    "trip_threshold": 3,
}


def test_an_open_breaker_is_one_line_with_its_exit() -> None:
    (line,) = remediation_storm_lines(_report(OPEN))

    assert line.startswith(
        "remediation storm breaker gpu-a: TRIPPED since 2026-10-29T12:00:00+00:00 "
        "(3 nodes, threshold 3, in 600 s across 3 domains)"
    ), line
    assert (
        "remediation-storm release --state-dir <state-dir> --cluster-id gpu-a" in line
    )


def test_closed_or_absent_breakers_print_nothing() -> None:
    assert remediation_storm_lines(_report(None)) == []
    assert remediation_storm_lines({"clusters": None}) == []


def test_the_breaker_line_leads_the_control_api_findings() -> None:
    health = {"checks": [{"name": "control_api", "details": _report(OPEN)}]}

    findings = _control_api_findings(health)

    assert findings and findings[0].startswith("remediation storm breaker gpu-a")
