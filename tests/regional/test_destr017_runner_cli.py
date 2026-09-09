"""GF-REGIONAL-DESTR-017 runner surface: plan identity, derived run identity,
the command line and the on-node holder path.

Split from ``test_destr017_out_of_band_reboot_fence.py`` (the fence-proof
contract tests) so each file stays within the review size limit; these need
only the runner module and a few constants, not the synthetic snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as destr017
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

NODE = "node-b"
GENERATION = 7
BOOT_BEFORE = "11111111-1111-4111-8111-111111111111"


def test_plan_identity_digest_is_stable_across_key_order() -> None:
    preflight = {
        "release_id": "rel",
        "node": {"uid": "u1", "boot_id": BOOT_BEFORE},
        "store": {
            "agent": {"generation": GENERATION, "incarnation_id": "inc-a"},
            "profile": {"profile_version": "p1"},
        },
        "runtime_identity": {"x": 1},
    }
    identity = destr017.plan_identity(preflight, node=NODE)
    shuffled = dict(reversed(list(identity.items())))
    assert destr017.identity_digest(identity) == destr017.identity_digest(shuffled)
    assert identity["agent_generation"] == GENERATION
    assert identity["node_boot_id"] == BOOT_BEFORE
    assert identity["node"] == NODE


def test_the_plan_identity_pins_the_generation_the_fence_will_compare() -> None:
    """A run whose plan was built against another Agent generation is a
    different case: the node rebooted between plan and execute."""

    base = {
        "release_id": "rel",
        "node": {"uid": "u1", "boot_id": BOOT_BEFORE},
        "store": {
            "agent": {"generation": GENERATION, "incarnation_id": "inc-a"},
            "profile": {"profile_version": "p1"},
        },
        "runtime_identity": {"x": 1},
    }
    drifted = json.loads(json.dumps(base))
    drifted["store"]["agent"]["generation"] = GENERATION + 1
    first = destr017.identity_digest(destr017.plan_identity(base, node=NODE))
    second = destr017.identity_digest(destr017.plan_identity(drifted, node=NODE))
    assert first != second


def test_derived_identity_is_deterministic_per_run_and_attempt(tmp_path: Path) -> None:
    first = destr017.derived_identity(tmp_path, 1)
    assert first == destr017.derived_identity(tmp_path, 1)
    assert first != destr017.derived_identity(tmp_path, 2)
    assert first.startswith("destr017-"), first
    assert first.endswith("-a1"), first


def test_parser_accepts_the_documented_arguments() -> None:
    parser = destr017.parser()
    arguments = parser.parse_args(["--run-dir", "/tmp/run"])
    assert arguments.execute is False and arguments.plan is False
    assert arguments.reboot_delay_seconds == destr017.DEFAULT_REBOOT_DELAY_SECONDS
    assert arguments.max_hold_seconds == destr017.DEFAULT_MAX_HOLD_SECONDS
    arguments = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            "DESTR017_EXECUTE",
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
            "--pci-bdf",
            "0000:0a:00.0",
            "--host-probe-image",
            "img",
            "--reboot-delay-seconds",
            "45",
            "--max-hold-seconds",
            "900",
        ]
    )
    assert arguments.node == NODE
    assert arguments.pci_bdf == "0000:0a:00.0"
    assert arguments.reboot_delay_seconds == 45
    help_text = parser.format_help()
    for option in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert option in help_text, option


def test_configure_bounds_the_reboot_delay_and_the_device_hold() -> None:
    parser = destr017.parser()
    for flag, value in (
        ("--reboot-delay-seconds", "10"),
        ("--reboot-delay-seconds", "900"),
        ("--max-hold-seconds", "30"),
        ("--max-hold-seconds", "7200"),
    ):
        arguments = parser.parse_args(
            [
                "--run-dir",
                "/tmp/run",
                "--node",
                NODE,
                "--host-probe-image",
                "img",
                flag,
                value,
            ]
        )
        with pytest.raises(RegionalFixtureError):
            destr017.configure(arguments)


def test_the_holder_is_armed_with_the_host_path_of_the_probe_script() -> None:
    """The arm unit runs on the node; /host is the Pod's mount, not the host's.
    Live, the unit died with "can't open file '/host/run/...'" and nothing held."""
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/e2e/regional/run_destr017_out_of_band_reboot_fence.py"
    ).read_text(encoding="utf-8")
    assert "/host/run/gpu-fault-host-probe-" not in source, "pod-side path leaked"
    assert "run.fence.host_script" in source, "arm-holder must receive host_script"
