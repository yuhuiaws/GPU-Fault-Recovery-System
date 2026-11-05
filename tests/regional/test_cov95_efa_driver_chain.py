"""GF-GENERAL-LIVE-013 EFA driver negatives: the waits and readers at their edges.

``test_general_live_efa_driver`` grades both variants on the settled live
shapes. These cases script the product's records one store read at a time, so
each wait is pinned to the record it keeps waiting on -- a chain not yet on the
record, a remediation still running, a successor still executing, the node on
its old boot, a new boot whose inventory is still short, an incident the
successor's restore has not closed yet -- and each reader to the reading that
makes it name a problem: a probe that cannot list the functions, a hold
receipt without a baseline count, a release before any target was chosen, a
detection without a confirmed mismatch sample, an oracle reading that lost
more than the one function, a trap that has not written its record yet, a
post-boot sample whose time cannot be read.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_driver as mod
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeTransportError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_poll_clock,
    store_script,
)
from tests.regional.test_general_live_efa_driver import (
    BASE_BOOT,
    BDF,
    NEW_BOOT,
    NODE,
    REPAIRED,
    SPECS,
    TRAP_FIRED_TEXT,
    _doc,
    _functions,
    _node_snapshot,
    _summary,
)


def summary(bound: int, **changes: Any) -> dict[str, Any]:
    return {**_summary(bound_after=bound), **changes}


def probe(**script: Any) -> FakeProbe:
    value: dict[str, Any] = {
        "functions": {"functions": _functions(), "bound_count": 16},
        "summary": [summary(16), summary(15)],
        "hold-unbind": {
            "pci_bdf": BDF,
            "driver_override": "gfrs-live-hold",
            "bound": False,
            "baseline_bound_count": 16,
            "boot_id": BASE_BOOT,
        },
        "trap-unbind": {
            "pci_bdf": BDF,
            "bound": False,
            "grace_seconds": 5,
            "baseline_bound_count": 16,
            "boot_id": BASE_BOOT,
        },
        "release": {"clean": True, "bound": True, "pci_bdf": BDF},
    }
    value.update(script)
    return FakeProbe(value)


class World(FakeRegional):
    """The store read by read: ``docs`` in order (the last repeats); the Node
    stays on ``BASE_BOOT`` until read ``new_boot_from`` and is then on
    ``NEW_BOOT`` (``None``: the default node, already on ``NEW_BOOT``)."""

    def __init__(
        self,
        docs: list[dict[str, Any]],
        *,
        new_boot_from: int | None = None,
        node: dict[str, Any] | None = None,
    ) -> None:
        self.docs = docs
        self.reads = 0
        self.new_boot_from = new_boot_from
        super().__init__(
            cluster_id="gpu-eks-cluster",
            cpu_python=store_script({"activity": self.read}),
            node_snapshot=node or _node_snapshot(),
        )

    def read(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        self.reads += 1
        return json.loads(json.dumps(self.docs[min(self.reads, len(self.docs)) - 1]))

    def node_snapshot(self, node: str) -> dict[str, Any]:
        value = super().node_snapshot(node)
        if self.new_boot_from is not None and self.reads < self.new_boot_from:
            value["boot_id"] = BASE_BOOT
        return value


def context(
    variant: str, regional: FakeRegional, fake: FakeProbe, tmp_path: Path
) -> VariantContext:
    spec = SPECS[variant]
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"efa-driver@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=120,
    )


def timeline(ctx: VariantContext, name: str) -> list[bool]:
    document = json.loads(
        (ctx.case_dir / f"{name}-timeline.json").read_text(encoding="utf-8")
    )
    return [entry["accepted"] for entry in document["entries"]]


# --------------------------------------------------------------------------- #
# the fixture's own readings
# --------------------------------------------------------------------------- #
def test_a_survey_without_a_bound_count_records_no_baseline(tmp_path: Path) -> None:
    fake = probe(summary={"boot_id": BASE_BOOT, "bound_count": None})
    ctx = context("bind_failure", World([_doc()]), fake, tmp_path)
    fixture = mod.EfaDriverFixture(ctx)

    value = fixture.survey()

    assert value == {"boot_id": BASE_BOOT, "bound_count": None}
    assert (fixture.boot_id, fixture.baseline_bound) == (BASE_BOOT, None)
    assert json.loads(
        (ctx.case_dir / "efa-driver-survey.json").read_text(encoding="utf-8")
    ) == {"boot_id": BASE_BOOT, "bound_count": None}


def test_a_probe_that_cannot_list_the_functions_leaves_the_fixture_not_ready(
    tmp_path: Path,
) -> None:
    listing = {"error": "PermissionError: /sys/bus/pci/devices", "functions": []}
    ctx = context("bind_failure", World([_doc()]), probe(functions=listing), tmp_path)

    with pytest.raises(VariantBlocked, match="could not enumerate functions") as info:
        mod.EfaDriverFixture(ctx).target_bdf()
    assert info.value.reason == "FIXTURE_NOT_READY"
    assert "PermissionError: /sys/bus/pci/devices" in info.value.detail


def test_a_hold_receipt_without_a_baseline_count_keeps_the_surveyed_one(
    tmp_path: Path,
) -> None:
    receipt = {"pci_bdf": BDF, "bound": False, "boot_id": "boot-armed"}
    ctx = context(
        "bind_failure", World([_doc()]), probe(**{"hold-unbind": receipt}), tmp_path
    )
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()

    value = fixture.hold_unbind(restore_seconds=600)

    assert value == receipt
    assert (fixture.baseline_bound, fixture.boot_id) == (16, "boot-armed")
    assert fixture.armed_at is not None, "the hold arms the fixture"


def test_a_release_before_any_target_was_chosen_touches_nothing(tmp_path: Path) -> None:
    fake = probe()
    ctx = context("bind_failure", World([_doc()]), fake, tmp_path)
    mod.EfaDriverFixture(ctx)

    results = ctx.cleanup.run()

    (item,) = [entry for entry in results if entry["name"] == "efa-driver"]
    assert item["ok"] is True, item
    assert item["result"] == {"released": False, "reason": "no target was selected"}
    assert fake.calls == [], "nothing was selected, so the probe is never asked"


# --------------------------------------------------------------------------- #
# bind_failure: every wait polls through the record it is waiting on
# --------------------------------------------------------------------------- #
def test_bind_failure_waits_through_a_forming_chain_and_a_slow_exit(
    tmp_path: Path,
) -> None:
    passing = _doc()
    no_expected_count = {
        **passing,
        "metric_latest": [
            {"name": "efa_inventory_active_count", "value": 16},
            {"name": "efa_inventory_expected_count", "value": "16"},
        ],
    }
    docs = [
        no_expected_count,  # _expected_efa_count: no numeric expected count
        passing,  # detection: the finding is on the record
        {**passing, "workflows": []},  # rung: no chain yet
        _doc(record_status="RUNNING", reboot=False),  # rung: still remediating
        passing,  # rung: FAILED with its reboot successor
        _doc(successor_status="RUNNING"),  # settled: the successor still runs
        passing,  # settled: everything ended, the node on its old boot
        _doc(recovered_active=15),  # settled: new boot, inventory still short
        _doc(source_state="QUARANTINED"),  # settled: healthy, source still open
        passing,  # source closed by the successor's restore
    ]
    world = World(docs, new_boot_from=8)
    ctx = context("bind_failure", world, probe(), tmp_path)

    result = mod.bind_failure(ctx)

    assert result["verdict"] == "PASS", result["errors"]
    assert result["expected_count"] == 16, (
        "without a numeric expected count the surveyed bound count stands in"
    )
    assert world.reads == len(docs)
    assert result["boot_id_after"] == NEW_BOOT
    assert timeline(ctx, "bind-failure-rung") == [False, False, True]
    assert timeline(ctx, "efa-driver-settled") == [False, False, False, True]
    assert timeline(ctx, "efa-driver-source-closed") == [True]


def test_bind_failure_names_an_unconfirmed_mismatch_and_a_wider_loss(
    tmp_path: Path,
) -> None:
    passing = _doc()
    unconfirmed = json.loads(json.dumps(passing))
    for batch in unconfirmed["batches"]:
        batch["samples"] = [
            item
            for item in batch["samples"]
            if item["name"] != "efa_inventory_mismatch"
        ]
    unconfirmed["batches"][-1]["samples"].append(
        {"name": "efa_inventory_mismatch", "device": "node", "value": 0, "labels": {}}
    )
    fake = probe(summary=[summary(16), summary(10)])
    ctx = context(
        "bind_failure", World([passing, unconfirmed, passing]), fake, tmp_path
    )

    result = mod.bind_failure(ctx)

    assert result["verdict"] == "FAIL", result
    assert result["stages"]["driver_unbound"] == [
        "no efa_inventory_mismatch=1 sample reached the product"
    ]
    assert (
        result["stages"]["no_global_unload"]
        == [
            "only 10 EFA functions bound at 2026-10-24T00:04:00+00:00 (expected >= 15; "
            "a global unload or extra loss)"
        ]
        * 2
    )


# --------------------------------------------------------------------------- #
# post_repair_failure
# --------------------------------------------------------------------------- #
def test_post_repair_without_a_finding_is_inconclusive_and_keeps_the_oracle(
    tmp_path: Path,
) -> None:
    quiet = {**_doc(), "incidents": [], "workflows": []}
    fake = probe(summary=[summary(16), summary(15), summary(15)])
    ctx = context("post_repair_failure", World([quiet]), fake, tmp_path)
    store_poll_clock(ctx)

    with pytest.raises(VariantInconclusive, match=f"unbound function {BDF}"):
        mod.post_repair_failure(ctx)

    assert [call[0] for call in fake.calls] == [
        "functions",
        "summary",
        "trap-unbind",
        "summary",
        "summary",
    ], "the oracle reads the function once more before giving up"
    assert timeline(ctx, "post-repair-detection")[-1] is False


def test_the_trap_read_keeps_polling_until_the_trap_wrote_its_record(
    tmp_path: Path,
) -> None:
    doc = _doc(
        remediation_status="SUCCEEDED",
        node_results=REPAIRED,
        post_repair=("VALIDATE_FABRIC", "FAILED"),
    )
    fake = probe(
        summary=[
            summary(16),
            summary(15),
            summary(15),
            summary(15, trap_fired_at=TRAP_FIRED_TEXT),
        ]
    )
    ctx = context("post_repair_failure", World([doc]), fake, tmp_path)

    result = mod.post_repair_failure(ctx)

    assert result["verdict"] == "PASS", result["errors"]
    assert result["trap"]["fired"] is True
    assert timeline(ctx, "post-repair-trap") == [False, True]


# --------------------------------------------------------------------------- #
# the release after the product's reboot reads the new boot's inventory
# --------------------------------------------------------------------------- #
def reboot_release(
    tmp_path: Path, node: dict[str, Any], doc: dict[str, Any]
) -> Callable[[], dict[str, Any]]:
    def lost(_arguments: tuple[str, ...]) -> dict[str, Any]:
        raise HostProbeTransportError("owned host probe Pod has terminated")

    fake = probe(release=lost)
    ctx = context("bind_failure", World([doc], node=node), fake, tmp_path)
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()
    fixture.expected_count = 16

    def run() -> dict[str, Any]:
        (item,) = [
            entry for entry in ctx.cleanup.run() if entry["name"] == "efa-driver"
        ]
        return item

    return run


def test_without_a_ready_time_every_post_boot_sample_counts(tmp_path: Path) -> None:
    run = reboot_release(tmp_path, _node_snapshot(ready_since=""), _doc())

    item = run()

    assert item["ok"] is True, item
    after = item["result"]["node_after"]
    assert after["ready_since"] == ""
    assert after["inventory"]["active"]["value"] == 16
    assert after["errors"] == []


def test_a_post_boot_sample_whose_time_cannot_be_read_never_counts(
    tmp_path: Path,
) -> None:
    doc = _doc()
    doc["batches"].append(
        {
            "observed_at": "not-a-time",
            "batch_id": "host-garbled",
            "edge_filter_reasons": [],
            "collection_errors": [],
            "samples": [
                {"name": "efa_inventory_mismatch", "device": "node", "value": 1},
                {"name": "efa_inventory_active_count", "device": "node", "value": 15},
            ],
        }
    )
    naive_ready = _node_snapshot(ready_since="2026-10-23T23:50:00")
    run = reboot_release(tmp_path, naive_ready, doc)

    item = run()

    assert item["ok"] is True, item
    inventory = item["result"]["node_after"]["inventory"]
    assert inventory["active"]["batch_id"] == "host-9", (
        "the garbled batch sorts last but is not a reading of the new boot"
    )
    assert inventory["mismatch"]["value"] == 0
