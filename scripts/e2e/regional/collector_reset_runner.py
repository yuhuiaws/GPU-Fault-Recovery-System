"""Owned reset sampling and per-SXID physical proof for Collector acceptance."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.collector_acceptance_fixture import CollectorAcceptanceFixture
from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.collector_case_cleanup import CaseCleanup
from scripts.e2e.regional.collector_reset_evidence import full_fabric_reset_errors
from scripts.e2e.regional.collector_sxid_evidence import (
    FULL_RESET_VARIANTS,
    full_reset_variant_errors,
    stable_inventory_errors,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

FULL_RESET_COOLDOWN_SECONDS = 60
FULL_RESET_STABLE_SAMPLES = 3
FULL_RESET_STABLE_INTERVAL_SECONDS = 5


class NodeTarget(Protocol):
    @property
    def node(self) -> str: ...


def stop_sampler(host: HostProbeFixture, run_id: str, errors: list[str]) -> None:
    """Stop the owned sampler without hiding the failure that initiated cleanup."""

    try:
        host.execute("stop-reset-sampler", "--run-id", run_id, timeout=60)
    except Exception as exc:
        errors.append(f"reset sampler stop failed: {type(exc).__name__}: {exc}")


def wait_full_reset_stability(
    regional: RegionalLiveFixture,
    settings: NodeTarget,
    reset_host: HostProbeFixture,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    """A settled workflow alone does not authorize a second full-fabric reset."""

    require_action_time(
        FULL_RESET_COOLDOWN_SECONDS
        + FULL_RESET_STABLE_INTERVAL_SECONDS * (FULL_RESET_STABLE_SAMPLES - 1)
        + 180
    )
    time.sleep(FULL_RESET_COOLDOWN_SECONDS)
    samples: list[dict[str, Any]] = []
    for index in range(FULL_RESET_STABLE_SAMPLES):
        if index:
            time.sleep(FULL_RESET_STABLE_INTERVAL_SECONDS)
        host = reset_host.execute("snapshot")
        node = regional.node_snapshot(settings.node)
        errors = stable_inventory_errors(baseline, host, node)
        if (
            samples and node["uid"] != samples[0]["node"]["uid"]
        ) or regional.business_workloads(settings.node):
            errors.append("node identity/workload changed between SXID reset variants")
        if errors:
            raise RegionalFixtureError("; ".join(errors))
        samples.append({"host": host, "node": node})
    return samples


def run_full_reset_variant(
    settings: NodeTarget,
    reset_host: HostProbeFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
    *,
    baseline: dict[str, Any],
    audit_before: dict[str, Any],
    sxid: int,
    classification: str,
    cleanup: CaseCleanup,
) -> dict[str, Any]:
    if (sxid, classification) not in FULL_RESET_VARIANTS:
        raise RegionalFixtureError("full-fabric SXID variant is not allowlisted")
    bdf = str(baseline["gpu_inventory"][0]["pci_bdf"])
    errors: list[str] = []
    marker = f"c014-full-{sxid}-{int(time.time())}-a{attempt}"
    run_id = f"c014-{sxid}-{attempt}"
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    write_json_atomic(case_dir / "reset-audit-before.json", audit_before)
    require_action_time(180)
    reset_host.execute(
        "start-reset-sampler",
        "--run-id",
        run_id,
        "--probe-script",
        reset_host.host_script,
        timeout=60,
    )
    state: dict[str, Any] = {}
    try:
        injected_at = datetime.now(timezone.utc)
        cleanup.register_seed(collector, marker, quiesce_host=reset_host)
        require_action_time(180)
        collector.execute(
            "append-sxid",
            "--sxid",
            str(sxid),
            "--marker",
            marker,
            "--pci-bdf",
            bdf,
            "--classification",
            classification,
            "--message",
            "NVSWITCH_NON_CORRECTABLE",
            "--include-switch",
        )
        state = collector.wait_marker(
            marker,
            case_dir=case_dir / "positive",
            timeout_seconds=1800,
            terminal_workflow=True,
            observed_after=injected_at,
        )
        cleanup.register_state(collector, state)
        after = reset_host.execute(
            "snapshot",
            "--since-epoch",
            str(injected_at.timestamp()),
            "--pci-bdf",
            bdf.rsplit(".", 1)[0],
            "--run-id",
            run_id,
            timeout=180,
        )
        write_json_atomic(case_dir / "host-after-positive.json", after)
        audit_after = collector.execute("reset-audit")
        write_json_atomic(case_dir / "reset-audit-after.json", audit_after)
        errors.extend(
            full_fabric_reset_errors(
                baseline,
                after,
                state,
                audit_before=audit_before,
                audit_after=audit_after,
                node=settings.node,
            )
        )
        errors.extend(
            full_reset_variant_errors(state, sxid=sxid, classification=classification)
        )
    finally:
        stop_sampler(reset_host, run_id, errors)
    restore = cleanup.restore(
        collector,
        state,
        profile_version=profile_version,
        reason=f"COLLECT-014 SXID{sxid} validated cleanup",
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "sxid": sxid,
        "classification": classification,
        "state": state,
        "restore_workflows": restore,
    }
