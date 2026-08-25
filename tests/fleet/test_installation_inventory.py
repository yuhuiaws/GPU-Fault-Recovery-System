from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.installation_inventory import (
    installed_unit_report,
    read_installed_systemd_units,
)
from tests._builders import copy_model
from tests.fleet._support import heartbeat, registry, signed

UNITS = ["gpu-fault-kernel-collector.service", "gpu-fault-node-agent.service"]


def test_installed_units_file_is_authoritative_with_discovery_fallback(
    tmp_path,
) -> None:
    inventory = tmp_path / "installed-units.txt"
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    (systemd / "gpu-fault-discovered.service").write_text("", encoding="utf-8")

    assert read_installed_systemd_units(inventory, systemd) == [
        "gpu-fault-discovered.service"
    ]

    inventory.write_text("gpu-fault-node-agent.service\n", encoding="utf-8")
    assert read_installed_systemd_units(inventory, systemd) == [
        "gpu-fault-node-agent.service"
    ]


def test_fleet_persists_full_installed_unit_inventory() -> None:
    fleet = registry()
    value = heartbeat(
        "node-a", installed_unit_report=installed_unit_report(UNITS, include_units=True)
    )

    record = fleet.register(signed(value))

    assert record.installed_unit_inventory is not None
    assert record.installed_unit_inventory.units == sorted(UNITS)


def test_digest_only_heartbeat_preserves_stored_inventory() -> None:
    fleet = registry()
    full = installed_unit_report(UNITS, include_units=True)
    first = fleet.register(signed(heartbeat("node-a", installed_unit_report=full)))
    digest_only = installed_unit_report(UNITS, include_units=False)
    second = fleet.register(
        signed(
            heartbeat(
                "node-a",
                observed_at=first.last_seen_at + timedelta(seconds=1),
                installed_unit_report=digest_only,
            )
        )
    )

    assert second.installed_unit_inventory == (first.installed_unit_inventory)
    assert second.generation == first.generation


def test_changed_digest_requires_full_inventory() -> None:
    fleet = registry()
    first = fleet.register(
        signed(
            heartbeat(
                "node-a",
                installed_unit_report=installed_unit_report(UNITS, include_units=True),
            )
        )
    )
    changed = [*UNITS, "gpu-fault-dcgm-exporter.service"]

    with pytest.raises(ValueError, match="full installed unit inventory is required"):
        fleet.register(
            signed(
                heartbeat(
                    "node-a",
                    observed_at=first.last_seen_at + timedelta(seconds=1),
                    installed_unit_report=(
                        installed_unit_report(changed, include_units=False)
                    ),
                )
            )
        )


def test_legacy_heartbeat_preserves_new_inventory() -> None:
    fleet = registry()
    first = fleet.register(
        signed(
            heartbeat(
                "node-a",
                installed_unit_report=installed_unit_report(UNITS, include_units=True),
            )
        )
    )
    legacy = copy_model(
        heartbeat("node-a", observed_at=first.last_seen_at + timedelta(seconds=1)),
        installed_unit_report=None,
    )

    second = fleet.register(signed(legacy))

    assert second.installed_unit_inventory == (first.installed_unit_inventory)
