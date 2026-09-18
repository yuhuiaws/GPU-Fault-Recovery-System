"""AUTH015's private snapshots, parallel scans and ordered compensation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.identity_acceptance_common import run_cleanup_steps
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture


@dataclass(repr=False)
class Auth015Context:
    master_sha256: str
    regional: RegionalLiveFixture
    original_gpu: dict[str, Any]
    original_cpu: dict[str, Any]
    before_keys: dict[str, str]
    before_cpu_keys: dict[str, str]
    gpu_node_names: list[str]
    before_agents: dict[str, Any]
    probes: list[HostProbeFixture]


def scan_auth015_hosts(
    probes: list[HostProbeFixture], master_sha256: str, *, create: bool
) -> dict[str, Any]:
    def scan(probe: HostProbeFixture) -> tuple[str, dict[str, Any]]:
        if create:
            probe.create()
        return probe.settings.node, probe.execute("--master-sha256", master_sha256)

    with ThreadPoolExecutor(max_workers=len(probes)) as pool:
        return dict(pool.map(scan, probes))


def cleanup_auth015(
    baseline: Auth015Context,
    *,
    rotation_started: bool,
    expected_gpu: dict[str, Any] | None,
    expected_cpu: dict[str, Any] | None,
    restore: Callable[[str, dict[str, Any], dict[str, Any] | None], None],
) -> tuple[dict[str, Any], list[str]]:
    steps: list[tuple[str, Callable[[], Any]]] = []
    if rotation_started:
        steps.extend(
            [
                (
                    "restore_gpu_secret",
                    lambda: restore("gpu", baseline.original_gpu, expected_gpu),
                ),
                (
                    "restore_cpu_secret",
                    lambda: restore("cpu", baseline.original_cpu, expected_cpu),
                ),
            ]
        )
    for probe in baseline.probes:
        steps.append((f"cleanup_probe:{probe.settings.node}", probe.cleanup))
    cleanup, errors = run_cleanup_steps(steps)
    return {
        probe.settings.node: cleanup.get(f"cleanup_probe:{probe.settings.node}")
        or {"cleanup_error": True}
        for probe in baseline.probes
    }, errors
