"""GF-GENERAL-LIVE-013 ``pci_missing``: the live hp 2026-10-27 oracle defects (b203).

Three runner oracles failed a chain the product got right, and each is driven
here through the phase-aware fakes on the live timeline: the unreadable-root
window grading ``_rdma``'s ``rdma_link_down`` of the real RDMA tree as an
inventory count; the fresh-inventory read anchored on the runner's clock and
then on a ``efa_inventory_mismatch=0`` sample the product never records after a
reboot (no latest-value row, health summaries are never raw evidence) -- now
the latest ``efa_inventory_active_count`` after the RESTART_NODE completion,
the new boot's first inventory batch and the chain's judged reading; and the
HyperPod receipt matched by instance id where ``BatchRebootClusterNodes`` names
the ``NodeLogicalId``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_pci as mod
from scripts.e2e.regional import general_live_inventory as inventory
from scripts.e2e.regional.general_live_source_replay import host_source_replay
from tests.regional._general_live_efa_pci_support import (
    DEVICES,
    EXPECTED,
    HP_LABELS,
    INSTANCE,
    LOGICAL_ID,
    OTHER_LOGICAL_ID,
    _base_doc,
    _baseline_batch,
    _execution,
    _healthy_batch,
    _hp_event,
    _hp_inventory,
    _judged,
    _r3_settled,
    _rdma_batch,
    _settled_doc,
    _workflow,
    _world,
)


def test_unreadable_root_records_the_real_trees_rdma_counters_without_grading_them(
    tmp_path: Path,
) -> None:
    negative = _base_doc(
        batches=[
            _rdma_batch("2026-10-24T00:00:15+00:00"),
            _rdma_batch("2026-10-24T00:00:35+00:00"),
        ]
    )
    world = _world(tmp_path, negative=negative)
    stages = mod.unreadable_root(world.ctx, host_source_replay(world.ctx))["stages"]
    assert stages["negative_no_count"] == [], stages["negative_no_count"]
    assert stages["negative_unknown"] == [], stages["negative_unknown"]
    oracle = json.loads(
        (world.ctx.case_dir / "oracle.jsonl").read_text().splitlines()[-1]
    )
    assert (oracle["counts_reported"], oracle["rdma_samples_recorded"]) == (
        0,
        2 * len(DEVICES),
    ), oracle
    assert inventory.EFA_INVENTORY_METRICS == inventory.EFA_METRICS - {
        "rdma_link_down"
    }, inventory.EFA_INVENTORY_METRICS


def test_the_live_hp_negative_window_passes_end_to_end(tmp_path: Path) -> None:
    negative = _base_doc(batches=[_rdma_batch("2026-10-24T00:00:15+00:00")])
    result = mod.pci_missing(_world(tmp_path, negative=negative).ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_fresh_inventory_is_anchored_on_the_products_restart_node_completion(
    tmp_path: Path,
) -> None:
    # The new boot's first batch landed after RESTART_NODE SUCCEEDED (00:20:00,
    # the execution's updated_at) and before the runner finished its post-reboot
    # bookkeeping (its clock here is the real one): the live hp timeline.
    settled = _settled_doc(fresh=False)
    settled["batches"].append(_healthy_batch("2026-10-24T00:21:00+00:00"))
    world = _world(tmp_path, settled=settled)
    result = mod.pci_missing(world.ctx)
    assert result["stages"]["fresh_inventory"] == [], result["stages"]
    assert result["verdict"] == "PASS", result["errors"]
    anchor = json.loads(
        (world.ctx.case_dir / "fresh-inventory-anchor.json").read_text()
    )
    assert anchor["restart_node_completed_at"] == "2026-10-24T00:20:00+00:00", anchor
    assert mod.parse_time(anchor["new_boot_observed_at"]) is not None, anchor


def test_a_count_older_than_the_restart_node_completion_is_not_fresh(
    tmp_path: Path,
) -> None:
    # A healthy batch after the reboot was requested (00:10) but before
    # RESTART_NODE completed (00:20) is the new boot's first batch -- it reads
    # the shortfall gauges -- yet not the restored count the gate reads from.
    settled = _settled_doc(fresh=False)
    settled["batches"].append(_healthy_batch("2026-10-24T00:19:00+00:00"))
    result = mod.pci_missing(_world(tmp_path, settled=settled).ctx)
    errors = result["stages"]["fresh_inventory"]
    assert len(errors) == 1, errors
    assert (
        "efa_inventory_active_count=3 after the reboot completed at 2026-10-24T00:20:00+00:00"
        in errors[0]
    ), errors


def test_reboot_completed_at_reads_the_succeeded_restart_node_execution() -> None:
    assert mod.reboot_completed_at(_settled_doc(), ["inc-efa"]) == datetime(
        2026, 10, 24, 0, 20, tzinfo=timezone.utc
    )
    assert mod.reboot_requested_at(_settled_doc(), ["inc-efa"]) == datetime(
        2026, 10, 24, 0, 10, tzinfo=timezone.utc
    )
    running = _settled_doc(workflows=[_workflow("RUNNING")])
    assert mod.reboot_completed_at(running, ["inc-efa"]) is None
    assert mod.reboot_requested_at(running, ["inc-efa"]) is None
    assert mod.reboot_completed_at(_settled_doc(), []) is None


def test_the_live_hp_r3_timeline_passes_on_the_products_post_reboot_record(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path, settled=_r3_settled(baseline=_baseline_batch()))
    result = mod.pci_missing(world.ctx)
    assert result["stages"]["fresh_inventory"] == [], result["stages"]
    assert result["verdict"] == "PASS", result["errors"]
    anchor = json.loads(
        (world.ctx.case_dir / "fresh-inventory-anchor.json").read_text()
    )
    assert anchor["restart_node_requested_at"] == "2026-10-24T00:10:00+00:00", anchor
    assert anchor["restart_node_completed_at"] == "2026-10-24T00:20:00+00:00", anchor
    assert anchor["fresh_inventory_polls"] == 0, anchor
    readings = anchor["post_reboot_readings"]
    assert set(readings) == set(mod.POST_REBOOT_SHORTFALL_METRICS), readings
    assert all(
        item["value"] == 0 and item["observed_at"] == "2026-10-24T00:15:00+00:00"
        for item in readings.values()
    ), readings
    assert anchor["judged_reading"]["value"] == EXPECTED, anchor


def test_fresh_inventory_names_a_missing_first_batch_of_the_new_boot(
    tmp_path: Path,
) -> None:
    result = mod.pci_missing(_world(tmp_path, settled=_r3_settled(baseline=None)).ctx)
    errors = result["stages"]["fresh_inventory"]
    assert len(errors) == 2, errors
    assert all(
        "after the reboot was requested at 2026-10-24T00:10:00+00:00: the new "
        "boot's first inventory batch is missing" in error
        for error in errors
    ), errors
    assert {error.split(" ")[1] for error in errors} == set(
        mod.POST_REBOOT_SHORTFALL_METRICS
    ), errors


def test_fresh_inventory_names_a_shortfall_in_the_new_boots_first_batch(
    tmp_path: Path,
) -> None:
    short = _baseline_batch(
        failure_mode="PCI_DEVICE_MISSING", active=EXPECTED - 1, streak=1
    )
    settled = _r3_settled(baseline=short, latest_active=EXPECTED - 1)
    result = mod.pci_missing(_world(tmp_path, settled=settled).ctx)
    errors = result["stages"]["fresh_inventory"]
    assert len(errors) == 2, errors
    assert f"efa_inventory_active_count={EXPECTED}" in errors[0], errors
    assert "(latest 2)" in errors[0], errors
    assert (
        "the latest efa_inventory_missing_count after the reboot was requested at "
        "2026-10-24T00:10:00+00:00 is 1 (observed 2026-10-24T00:15:00+00:00), not 0"
    ) in errors[1], errors


@pytest.mark.parametrize(
    ("details", "needle"),
    [
        ({}, "without details.inventory_evidence"),
        (
            _judged(EXPECTED - 1),
            "judged efa_inventory_active_count=2, not the restored 3",
        ),
        (_judged(observed_at="2026-10-24T00:19:00+00:00"), "before the restore at"),
    ],
)
def test_fresh_inventory_requires_the_chains_judged_reading(
    tmp_path: Path, details: dict[str, Any], needle: str
) -> None:
    workflow = _workflow(
        "SUCCEEDED",
        executions=tuple(
            _execution(
                operation,
                **({"details": details} if operation == "VALIDATE_FABRIC" else {}),
            )
            for operation in (
                "MARK_UNSCHEDULABLE",
                "QUARANTINE",
                "RESTART_NODE",
                "VALIDATE_GPU",
                "VALIDATE_HOST",
                "VALIDATE_FABRIC",
                "RESTORE_SCHEDULING",
            )
        ),
    )
    settled = _r3_settled(baseline=_baseline_batch(), workflows=[workflow])
    result = mod.pci_missing(_world(tmp_path, settled=settled).ctx)
    errors = result["stages"]["fresh_inventory"]
    assert len(errors) == 1 and needle in errors[0], errors
    assert result["verdict"] == "FAIL", result["stages"]


def test_fresh_inventory_waits_one_summary_phase_for_the_latest_count(
    tmp_path: Path,
) -> None:
    # The chain settled before the first health summary after the reboot: the
    # latest-value row still carries the pre-reboot reading, the next summary
    # advances it.
    stale = _r3_settled(
        baseline=_baseline_batch(), latest_at="2026-10-24T00:05:00+00:00"
    )
    world = _world(
        tmp_path, settled=stale, settled_later=_r3_settled(baseline=_baseline_batch())
    )
    result = mod.pci_missing(world.ctx)
    assert result["stages"]["fresh_inventory"] == [], result["stages"]
    assert result["verdict"] == "PASS", result["errors"]
    anchor = json.loads(
        (world.ctx.case_dir / "fresh-inventory-anchor.json").read_text()
    )
    assert anchor["fresh_inventory_polls"] >= 1, anchor


def test_the_hyperpod_receipt_is_matched_by_the_nodes_logical_id(
    tmp_path: Path,
) -> None:
    world = _world(
        tmp_path, labels=HP_LABELS, run_output=_hp_inventory, events=(_hp_event(),)
    )
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert (result["provider"], result["node_logical_id"]) == ("hp", LOGICAL_ID)
    wait = world.regional.provider_waits[0]
    assert wait["event_names"] == {"BatchRebootClusterNodes", "RebootClusterNodes"}
    assert wait["event_source"] == "sagemaker.amazonaws.com", wait
    assert result["provider_events"][0]["request_node_logical_ids"] == [LOGICAL_ID]
    before = json.loads((world.ctx.case_dir / "node-before.json").read_text())
    assert before["hyperpod_binding"] == {
        "node_logical_id": LOGICAL_ID,
        "error": None,
        "cluster": "hp-cluster",
    }, before
    assert before["node"]["node_logical_id"] == LOGICAL_ID, before


@pytest.mark.parametrize(
    ("events", "run_output", "needle"),
    [
        ((_hp_event(OTHER_LOGICAL_ID),), _hp_inventory, "another target"),
        (
            (_hp_event(),),
            lambda command: json.dumps({"ClusterNodeSummaries": []}),
            "NodeLogicalId is unresolved (0 HyperPod nodes",
        ),
        (
            (_hp_event(LOGICAL_ID, OTHER_LOGICAL_ID),),
            _hp_inventory,
            "together with another target",
        ),
        ((), _hp_inventory, f"or its NodeLogicalId {LOGICAL_ID} within"),
    ],
)
def test_hyperpod_receipt_fail_branches(
    tmp_path: Path,
    events: tuple[dict[str, Any], ...],
    run_output: Callable[[list[str]], str],
    needle: str,
) -> None:
    world = _world(tmp_path, labels=HP_LABELS, run_output=run_output, events=events)
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "FAIL", result["stages"]
    receipt = result["stages"]["provider_receipt"]
    assert any(needle in error for error in receipt), receipt
    others = {
        name: errors
        for name, errors in result["stages"].items()
        if errors and name != "provider_receipt"
    }
    assert others == {}, others


def test_hyperpod_logical_id_names_a_failed_or_ambiguous_binding(
    tmp_path: Path,
) -> None:
    def refused(command: list[str]) -> str:
        raise RuntimeError("aws refused")

    def duplicated(command: list[str]) -> str:
        return json.dumps(
            {
                "ClusterNodeSummaries": [
                    {"InstanceId": INSTANCE, "NodeLogicalId": "a"},
                    {"InstanceId": INSTANCE, "NodeLogicalId": "b"},
                ],
                "NextToken": "more",
            }
        )

    world = _world(tmp_path, labels=HP_LABELS, run_output=refused)
    binding = mod.hyperpod_logical_id(world.ctx, INSTANCE)
    assert binding == {
        "node_logical_id": None,
        "error": "RuntimeError: aws refused",
        "cluster": "hp-cluster",
    }, binding
    world = _world(tmp_path / "dup", labels=HP_LABELS, run_output=duplicated)
    binding = mod.hyperpod_logical_id(world.ctx, INSTANCE)
    assert binding["error"] == (
        f"2 HyperPod nodes of hp-cluster bind {INSTANCE} (inventory incomplete)"
    ), binding
    assert mod.hyperpod_logical_id(world.ctx, None)["node_logical_id"] is None
