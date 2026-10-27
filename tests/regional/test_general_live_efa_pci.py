"""GF-GENERAL-LIVE-013 ``pci_missing``: registry, plan shape, grading, real collector.

The variant is driven end to end over phase-aware fakes: the source-replay
probe answers the unreadable-root build, the real-root copy without one BDF,
the RDMA tree without that device, the mixed window, the close, the re-create
after the reboot and the snapshot on the new boot; the store answers the
negative-stage, detection and settled documents by what the probe has been
asked so far; the node snapshot flips to a new boot id once the positive
window is closed. The real host collector is also run over the probe-built
fixtures to pin the product semantics the grading rests on: one dropped real
function is PCI_DEVICE_MISSING confirmed on the second tick, an unreadable
root is UNKNOWN on every tick and resets the confirmation streak.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors.host.observation import efa_inventory_counts
from scripts.e2e.regional import general_live_efa_pci as mod
from scripts.e2e.regional import general_live_inventory as inventory
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import (
    coverage_rows,
    load_registry,
    validate_spec,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    host_source_replay,
)
from scripts.e2e.regional.probes import general_live_source_builders as builders
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.collectors import _cov95_runtime_collect as cov_support
from tests.collectors import _cov95_runtime_collect_host as cov_host
from tests.regional._general_live_efa_pci_support import (
    BDFS,
    DEVICES,
    DROPPED_BDF,
    DROPPED_DEVICE,
    EXPECTED,
    INSTANCE,
    NODE,
    OTHER_INSTANCE,
    PEER,
    PEER_NODE,
    SPEC,
    _base_doc,
    _calls,
    _command,
    _confirmed_batches,
    _efa_batch,
    _event,
    _execution,
    _flags,
    _incident,
    _inventory_doc,
    _node,
    _settled_doc,
    _unknown_batch,
    _workflow,
    _world,
)

isolated_runtime = cov_support.isolated_runtime
host_case = cov_host.host_case


# --------------------------------------------------------------------------- #
# registry and plan shape
# --------------------------------------------------------------------------- #
def test_pci_missing_is_registered_as_the_reboot_variant() -> None:
    registry = load_registry()
    spec = registry["GF-GENERAL-LIVE-013.pci_missing"]
    validate_spec(spec)
    assert spec is SPEC and spec.implemented, spec.describe()
    assert spec.lanes == ("eks", "hp") and spec.proof_mode == "live-replay"
    assert spec.approvals == (HOST_SOURCE_REPLAY, inventory.NODE_REBOOT)
    assert (spec.flags, spec.risk, spec.expected_outcome) == ("X", "H", "PASS")
    assert spec.confirmation == "GENERALLIVE013_EXECUTE"
    plan = spec.describe()
    assert plan["approvals"] == ["HOST_SOURCE_REPLAY", "NODE_REBOOT"], plan
    for phrase in ("unreadable PCI root", "one real EFA function absent"):
        assert phrase in plan["injection"], plan["injection"]
    for phrase in (
        "PCI_DEVICE_MISSING",
        "exactly one RESTART_NODE",
        "new boot id",
        "no REPLACE_NODE",
        "--node-b untouched",
    ):
        assert phrase in plan["expected"], plan["expected"]
    assert "validated restore" in plan["cleanup"], plan["cleanup"]
    assert 60 <= spec.max_minutes <= 180, spec.max_minutes


def test_013_denominator_is_complete_across_both_modules() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-013"])
    # 013 is spread over the inventory module, this module (pci_missing) and
    # general_live_efa_driver (bind_failure / post_repair_failure): check the registry.
    from scripts.e2e.regional.general_live_registry import load_registry

    registered = {
        key.split(".", 1)[1]
        for key in load_registry()
        if key.startswith("GF-GENERAL-LIVE-013.")
    }
    assert registered == expected, registered ^ expected
    assert "pci_missing" not in {item.variant for item in inventory.VARIANTS}
    assert mod.POSITIVE_WINDOW_SECONDS <= SPEC.max_minutes * 60, (
        "the positive stage's window must fit the variant's max_minutes"
    )


# --------------------------------------------------------------------------- #
# the PASS flow
# --------------------------------------------------------------------------- #
def test_pci_missing_passes_end_to_end(tmp_path: Path) -> None:
    world = _world(tmp_path)
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert (
        result["proof_scope"] == "live-replay" and not result["physical_fault_injected"]
    )
    assert (result["device"], result["bdf"]) == (DROPPED_DEVICE, DROPPED_BDF)
    assert (result["boot_id_before"], result["boot_id_after"]) == ("boot-1", "boot-2")
    assert result["instance_id"] == INSTANCE and result["expected_count"] == EXPECTED
    for stage in (
        "negative_unknown",
        "negative_no_count",
        "negative_no_finding",
        "negative_no_reboot",
        "negative_not_isolated",
        "failure_mode",
        "confirmed_sample",
        "finding",
        "reboot_planned",
        "rebooted",
        "post_reboot_clean",
        "real_reads",
        "chain",
        "reboot_receipts",
        "single_reboot",
        "node_restored",
        "fresh_inventory",
        "provider_receipt",
        "peer_untouched",
    ):
        assert result["stages"][stage] == [], (stage, result["stages"][stage])
    # The probe was driven in the reviewed order: unreadable root first (closed),
    # then the real root without the BDF plus the RDMA tree without the device in
    # one mixed window, closed the moment the finding existed, then re-created on
    # the new boot and asked for the residue and the real tree.
    commands = [call[0] for call in world.probe.calls]
    assert commands.index("close-window") < commands.index("build-infiniband-fixture")
    # The unreadable-root close removed the node's state dir with the builders
    # in it: they ship again before the missing-function build (skipped live on
    # HyperPod 2026-10-27, "install general_live_source_builders.py before
    # building a fixture", INCONCLUSIVE).
    first_close = commands.index("close-window")
    second_build = commands.index("build-pci-fixture", first_close)
    assert "install-source" in commands[first_close:second_build], commands
    assert world.probe.wipes[0] == ("close-window", world.ctx.run_id), world.probe.wipes
    builds = _calls(world.probe, "build-pci-fixture")
    assert _flags(builds[0]).get("--unreadable") == [""], builds[0]
    assert _flags(builds[1])["--source"] == [mod.PCI_ROOT], builds[1]
    assert _flags(builds[1])["--drop"] == [DROPPED_BDF], builds[1]
    tree = _calls(world.probe, "build-infiniband-fixture")[0]
    assert _flags(tree)["--drop"] == [DROPPED_DEVICE], tree
    opens = _calls(world.probe, "open-window")
    assert _flags(opens[0])["--mode"] == ["pci-root"], opens[0]
    positive = _flags(opens[1])
    assert positive["--mode"] == ["mixed"] and positive.get("--pci-root") == [""], (
        opens[1]
    )
    assert positive["--tree"] == [f"infiniband:{inventory.INFINIBAND_ROOT}"], opens[1]
    assert int(positive["--restore-seconds"][0]) == inventory.DETECTION_BUDGET + 300
    assert len(_calls(world.probe, "close-window")) == 2, commands
    assert (world.probe.cleanups, world.probe.creates) == (1, 1), (
        "the probe Pod is re-created once, after the reboot"
    )
    second_close = commands.index("close-window", commands.index("close-window") + 1)
    assert commands.index("snapshot") > second_close, commands
    assert "read-source" in commands, commands
    # The provider receipt was looked up on the EC2 API for this lane.
    wait = world.regional.provider_waits[0]
    assert wait["event_names"] == {"RebootInstances"}, wait
    assert wait["event_source"] == "ec2.amazonaws.com", wait
    assert wait["timeout_seconds"] == mod.PROVIDER_RECEIPT_BUDGET, wait
    assert world.regional.peer_reads, "the pair node's incidents were read"
    assert result["provider_events"][0]["request_node_ids"] == [INSTANCE]
    # Evidence of every step was written; the oracle names the dropped function.
    names = {path.name for path in world.ctx.case_dir.iterdir()}
    for name in (
        "unreadable-root-fixture.json",
        "unreadable-root-close.json",
        "pci-fixture.json",
        "infiniband-fixture.json",
        "source-window-open.json",
        "source-window-close.json",
        "node-settled.json",
        "probe-recreate.json",
        "post-reboot-snapshot.json",
        "post-reboot-read-back.json",
        "node-after.json",
        "provider-events.json",
        "oracle.jsonl",
    ):
        assert name in names, (name, sorted(names))
    oracle = [
        json.loads(line)
        for line in (world.ctx.case_dir / "oracle.jsonl").read_text().splitlines()
    ]
    assert any(item.get("dropped_bdf") == DROPPED_BDF for item in oracle), oracle
    assert any(item.get("unreadable") is True for item in oracle), oracle
    # Cleanup: the hand-back is registered and skipped on a node nothing holds.
    assert "validated-restore" in world.ctx.cleanup.pending()
    results = {item["name"]: item for item in world.ctx.cleanup.run()}
    assert results["validated-restore"]["result"]["skipped"] is True, results
    assert results["source-replay"]["ok"], results


def test_pci_missing_runs_without_a_pair_node(tmp_path: Path) -> None:
    world = _world(tmp_path, peer="")
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["peer_untouched"] == [] and not world.regional.peer_reads


# --------------------------------------------------------------------------- #
# the negative stage
# --------------------------------------------------------------------------- #
def test_a_finding_from_an_unreadable_root_fails_and_skips_the_reboot_stage(
    tmp_path: Path,
) -> None:
    negative = _base_doc(
        batches=[_unknown_batch("2026-10-24T00:00:15+00:00"), *_confirmed_batches()],
        incidents=[_incident("ACTION_PENDING")],
        workflows=[_workflow("FAILED")],
    )
    world = _world(tmp_path, negative=negative, detection=negative)
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["negative_no_finding"], result["stages"]
    assert result["stages"]["negative_no_count"], result["stages"]
    assert any(
        ("RESTART_NODE" in error for error in result["stages"]["negative_no_reboot"])
    ), result["stages"]["negative_no_reboot"]
    assert result["stages"]["positive_skipped"], result["stages"]
    assert result["negative_settled"] is True
    assert not _calls(world.probe, "build-infiniband-fixture"), (
        "the missing-function stage never ran on top of a wrong negative"
    )
    assert "validated-restore" in world.ctx.cleanup.pending()


def test_unreadable_root_needs_the_unknown_marker_and_no_count(tmp_path: Path) -> None:
    counted = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:00:15+00:00",
                failure_mode="PCI_DEVICE_MISSING",
                active=0,
                mismatch=0,
                streak=1,
            )
        ]
    )
    world = _world(tmp_path, negative=counted)
    replay = host_source_replay(world.ctx)
    negative = mod.unreadable_root(world.ctx, replay)
    stages = negative["stages"]
    assert any((mod.UNKNOWN_MARKER in error for error in stages["negative_unknown"])), (
        stages["negative_unknown"]
    )
    assert any(
        "efa_inventory_active_count=0" in error for error in stages["negative_no_count"]
    ), stages["negative_no_count"]
    assert stages["negative_no_finding"] == [] and stages["negative_no_reboot"] == []
    assert _flags(_calls(world.probe, "open-window")[0])["--mode"] == ["pci-root"]
    assert _calls(world.probe, "close-window"), "the unreadable window is closed"


# --------------------------------------------------------------------------- #
# FAIL branches of the positive stage
# --------------------------------------------------------------------------- #
def _second_reboot() -> list[dict[str, Any]]:
    return [
        _workflow("SUCCEEDED"),
        _workflow(
            "SUCCEEDED",
            request_id="wf-escalation",
            operations=("RESTART_NODE", "VALIDATE_FABRIC", "RESTORE_SCHEDULING"),
        ),
    ]


@pytest.mark.parametrize(
    ("stage", "world_kwargs", "needle"),
    [
        (
            "failure_mode",
            {"settled": _settled_doc(), "detection_mode": "LINK_INACTIVE"},
            "failure_mode",
        ),
        (
            "single_reboot",
            {"settled": _settled_doc(workflows=_second_reboot())},
            "2 RESTART_NODE",
        ),
        (
            "single_reboot",
            {"settled": _settled_doc(commands=[_command(), _command("REPLACE_NODE")])},
            "REPLACE_NODE",
        ),
        (
            "reboot_receipts",
            {"settled": _settled_doc(commands=[_command(node_ids=(PEER,))])},
            "scope",
        ),
        (
            "reboot_receipts",
            {
                "settled": _settled_doc(
                    workflows=[
                        _workflow(
                            "SUCCEEDED",
                            executions=tuple(
                                _execution(op)
                                for op in (
                                    "MARK_UNSCHEDULABLE",
                                    "QUARANTINE",
                                    "VALIDATE_FABRIC",
                                )
                            ),
                        )
                    ]
                )
            },
            "0 SUCCEEDED RESTART_NODE",
        ),
        ("chain", {"settled": _settled_doc(incident_state="ESCALATED")}, "ESCALATED"),
        (
            "fresh_inventory",
            {"settled": _settled_doc(fresh=False)},
            "efa_inventory_active_count",
        ),
        ("node_restored", {"node_after": {"unschedulable": True}}, "isolated"),
        ("node_restored", {"allocatable_after": "31"}, "efa_allocatable"),
        ("provider_receipt", {"events": (_event(OTHER_INSTANCE),)}, "another target"),
        ("provider_receipt", {"events": ()}, "no accepted"),
        (
            "provider_receipt",
            {
                "events": (
                    _event(),
                    _event(api_error_code="Client.UnauthorizedOperation"),
                )
            },
            None,
        ),
        (
            "peer_untouched",
            {"peer_after": {**PEER_NODE, "boot_id": "peer-boot-2"}},
            "rebooted",
        ),
        (
            "post_reboot_clean",
            {
                "probe_extra": {
                    "snapshot": {
                        "dropin": True,
                        "unit_pci_root": "/run/gfrs-live/x/pci",
                    }
                }
            },
            "dropin",
        ),
        (
            "real_reads",
            {"probe_extra": {"read-source": {"same": False}}},
            "real RDMA tree",
        ),
    ],
)
def test_positive_stage_fail_branches(
    tmp_path: Path, stage: str, world_kwargs: dict[str, Any], needle: str | None
) -> None:
    kwargs = dict(world_kwargs)
    detection_mode = kwargs.pop("detection_mode", None)
    if detection_mode:
        kwargs["detection"] = _base_doc(
            incidents=[_incident("ACTION_PENDING")],
            workflows=[_workflow("RUNNING")],
            batches=_confirmed_batches(detection_mode),
        )
    world = _world(tmp_path, **kwargs)
    result = mod.pci_missing(world.ctx)
    if needle is None:
        # A refused provider call beside the accepted one is not a second reboot.
        assert result["stages"][stage] == [], result["stages"][stage]
        assert result["verdict"] == "PASS", result["errors"]
        return
    assert result["verdict"] == "FAIL", result["stages"]
    assert any(needle in error for error in result["stages"][stage]), (
        stage,
        result["stages"][stage],
    )
    others = {
        name: errors
        for name, errors in result["stages"].items()
        if errors and name != stage
    }
    assert not others or stage in {"single_reboot", "failure_mode"}, others


def test_no_reboot_within_budget_is_a_fail_without_recreating_the_probe(
    tmp_path: Path,
) -> None:
    settled = _settled_doc(
        incident_state="ACTION_PENDING",
        workflows=[_workflow("BLOCKED", executions=())],
        commands=[],
        fresh=False,
    )
    world = _world(tmp_path, reboots=False, settled=settled)
    step = iter(range(0, 10**7, 120))
    origin = datetime(2026, 10, 24, tzinfo=timezone.utc)
    world.ctx.clock = lambda: origin + timedelta(seconds=next(step))
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "FAIL", result["errors"]
    assert any(
        ("no reboot within" in error for error in result["stages"]["rebooted"])
    ), result["stages"]["rebooted"]
    assert "post_reboot_clean" not in result["stages"], (
        "no residue check without a reboot"
    )
    assert (world.probe.cleanups, world.probe.creates) == (0, 0)
    assert result["boot_id_after"] is None
    assert result["stages"]["reboot_receipts"], result["stages"]


def test_a_confirmed_mismatch_without_an_incident_is_a_product_fail(
    tmp_path: Path,
) -> None:
    detection = _base_doc(batches=_confirmed_batches())
    world = _world(tmp_path, detection=detection, reboots=False)
    result = mod.pci_missing(world.ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["finding"] == [
        "the confirmed PCI_DEVICE_MISSING mismatch minted no incident"
    ]
    assert "reboot_planned" not in result["stages"]
    assert len(_calls(world.probe, "close-window")) == 2, "the window was closed anyway"


# --------------------------------------------------------------------------- #
# INCONCLUSIVE and BLOCKED
# --------------------------------------------------------------------------- #
def test_no_missing_sample_is_inconclusive_after_closing_the_window(
    tmp_path: Path,
) -> None:
    detection = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:00:15+00:00",
                failure_mode="HEALTHY",
                active=EXPECTED,
                mismatch=0,
                streak=0,
            )
        ]
    )
    world = _world(tmp_path, detection=detection, reboots=False)
    with pytest.raises(VariantInconclusive, match="efa_inventory_missing_count"):
        mod.pci_missing(world.ctx)
    assert len(_calls(world.probe, "close-window")) == 2, world.probe.calls


def test_a_replaced_node_during_the_reboot_wait_is_inconclusive(tmp_path: Path) -> None:
    world = _world(tmp_path, replaced=True)
    with pytest.raises(VariantInconclusive, match="Spot reclaim"):
        mod.pci_missing(world.ctx)


def test_blocked_without_an_active_efa_function(tmp_path: Path) -> None:
    world = _world(tmp_path, probe_extra={"inventory": _inventory_doc(efa=False)})
    with pytest.raises(VariantBlocked) as blocked:
        mod.pci_missing(world.ctx)
    assert blocked.value.reason == "CAPABILITY_ABSENT"
    assert not _calls(world.probe, "open-window"), "nothing was opened"


def test_blocked_when_the_real_root_does_not_match_the_expectation(
    tmp_path: Path,
) -> None:
    world = _world(
        tmp_path,
        probe_extra={
            "build-pci-fixture": {"path": "/run/x/pci", "efa_devices": 7, "dropped": []}
        },
    )
    with pytest.raises(VariantBlocked) as blocked:
        mod.pci_missing(world.ctx)
    assert blocked.value.reason == "FIXTURE_NOT_READY"
    assert "7 EFA functions" in blocked.value.detail, blocked.value.detail
    assert len(_calls(world.probe, "open-window")) == 1, (
        "only the negative window was opened"
    )


# --------------------------------------------------------------------------- #
# cleanup hand-back and the graders
# --------------------------------------------------------------------------- #
def test_hand_back_restores_only_an_isolated_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restores: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod,
        "restore_incidents",
        lambda ctx, incidents, *, reason: restores.append(
            {"incidents": incidents, "reason": reason}
        )
        or [{"status": "SUCCEEDED"}],
    )
    world = _world(tmp_path)
    assert mod.hand_back(world.ctx, [_incident("ACTION_PENDING")])["skipped"] is True
    assert restores == []
    world.regional.reboots = True
    world.regional.node_after = {"unschedulable": True}
    world.probe.calls.extend([("close-window",), ("close-window",)])
    value = mod.hand_back(world.ctx, [_incident("ACTION_PENDING")])
    assert value["restored"] == [{"status": "SUCCEEDED"}], value
    assert "pci_missing" in restores[0]["reason"], restores


def test_residue_and_receipt_graders() -> None:
    clean = {
        "window_open": False,
        "dropin": False,
        "state_dir": False,
        "deadman": False,
        "unit_active_state": "active",
        "unit_pci_root": "/sys/bus/pci/devices",
    }
    assert mod.residue_errors(clean) == []
    dirty = {**clean, "deadman": True, "unit_pci_root": "/run/gfrs-live/s/fixtures/pci"}
    errors = mod.residue_errors(dirty)
    assert len(errors) == 2 and "deadman" in errors[0] and "fixture" in errors[1], (
        errors
    )
    assert mod.residue_errors({**clean, "unit_active_state": "failed"}), "unit down"
    observed = _settled_doc()
    assert mod.reboot_receipt_errors(observed, ["inc-efa"], NODE) == []
    assert mod.single_reboot_errors(observed) == []
    missing_adapter = _settled_doc(
        workflows=[
            _workflow(
                "SUCCEEDED",
                executions=(_execution("RESTART_NODE", adapter_operation_id=None),),
            )
        ]
    )
    assert any(
        (
            "adapter operation" in error
            for error in mod.reboot_receipt_errors(missing_adapter, ["inc-efa"], NODE)
        )
    ), mod.reboot_receipt_errors(missing_adapter, ["inc-efa"], NODE)
    no_command = _settled_doc(commands=[])
    assert any(
        (
            "remote command" in error
            for error in mod.reboot_receipt_errors(no_command, ["inc-efa"], NODE)
        )
    ), mod.reboot_receipt_errors(no_command, ["inc-efa"], NODE)
    confirmed = _base_doc(batches=_confirmed_batches())
    assert mod.confirmed_sample_errors(confirmed, expected=EXPECTED) == []
    assert mod.confirmed_sample_errors(confirmed, expected=EXPECTED + 1), (
        "a wrong expectation is named"
    )
    assert mod.confirmed_sample_errors(_base_doc(), expected=EXPECTED) == [
        "no efa_inventory_mismatch=1 sample reached the product"
    ]
    before = {**_node("boot-1"), "efa_allocatable": "32"}
    after = {**_node("boot-2"), "efa_allocatable": "32"}
    assert mod.node_restored_errors(before, after, _node("boot-2")) == []
    again = mod.node_restored_errors(
        before, {**after, "boot_id": "boot-3"}, _node("boot-2")
    )
    assert again and "changed again" in again[0], again
    assert mod.efa_functions(_inventory_doc()) == list(zip(DEVICES, BDFS))
    inactive = _inventory_doc()
    inactive["devices"][DEVICES[0]]["ports"]["1"] = {
        "state": "1: DOWN",
        "phys_state": "3: Disabled",
    }
    inactive["devices"][DEVICES[1]]["pci"] = None
    assert mod.efa_functions(inactive) == [(DEVICES[2], BDFS[2])]


# --------------------------------------------------------------------------- #
# the real host collector over the probe-built fixtures
# --------------------------------------------------------------------------- #
def _fake_sysfs(tmp_path: Path) -> tuple[Path, Path]:
    """A node's ``/sys``: EFA functions with their RDMA devices, two other devices."""

    sys_root = tmp_path / "sys"
    devices = sys_root / "devices" / "pci0000:50"
    pci_root = sys_root / "bus" / "pci" / "devices"
    drivers = sys_root / "bus" / "pci" / "drivers"
    ib_root = sys_root / "class" / "infiniband"
    (drivers / "efa").mkdir(parents=True)
    (drivers / "ena").mkdir(parents=True)
    (sys_root / "class" / "net").mkdir(parents=True)
    pci_root.mkdir(parents=True)
    ib_root.mkdir(parents=True)
    for bdf, vendor, device, driver in (
        ("0000:00:00.0", "0x10de", "0x2330", None),
        ("0000:00:05.0", "0x1d0f", "0xec20", "ena"),
    ):
        function = devices / bdf
        function.mkdir(parents=True)
        (function / "vendor").write_text(vendor + "\n")
        (function / "device").write_text(device + "\n")
        if driver:
            (function / "driver").symlink_to(drivers / driver)
        (pci_root / bdf).symlink_to(function)
    for name, bdf in zip(DEVICES, BDFS):
        function = devices / bdf
        function.mkdir(parents=True)
        (function / "vendor").write_text("0x1d0f\n")
        (function / "device").write_text("0xefa2\n")
        (function / "uevent").write_text(f"DRIVER=efa\nPCI_SLOT_NAME={bdf}\n")
        (function / "driver").symlink_to(drivers / "efa")
        (pci_root / bdf).symlink_to(function)
        rdma = ib_root / name
        (rdma / "ports" / "1").mkdir(parents=True)
        (rdma / "ports" / "1" / "state").write_text("4: ACTIVE\n")
        (rdma / "ports" / "1" / "phys_state").write_text("5: LinkUp\n")
        (rdma / "device").symlink_to(function)
    return ib_root, pci_root


def _probe_build(capsys: pytest.CaptureFixture[str], *arguments: str) -> dict[str, Any]:
    code = probe.main([*arguments, "--run-id", "gl013-pci-missing-a1-deadbeef"])
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0, receipt
    return receipt


def _view(batch: Any) -> dict[str, Any]:
    return {
        "observed_at": batch.observed_at.isoformat(),
        "batch_id": batch.batch_id,
        "edge_filter_reasons": [],
        "collection_errors": list(batch.collection_errors),
        "samples": [
            {
                "name": s.name,
                "device": s.device,
                "value": s.value,
                "labels": dict(s.labels),
            }
            for s in batch.samples
        ],
    }


def _values(batch: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item["value"] for item in batch["samples"]}


def test_the_real_collector_confirms_one_dropped_function_as_pci_device_missing(
    host_case: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    ib_root, pci_root = _fake_sysfs(tmp_path)
    # The probe's read-only inventory (its ``inventory`` command runs ethtool
    # through shutil.which, which host_case stubs): the device -> BDF mapping
    # the runner picks the dropped function from.
    listed = builders.rdma_inventory(ib_root, tmp_path / "sys" / "class" / "net")
    assert mod.efa_functions(listed) == list(zip(DEVICES, BDFS)), listed["devices"]
    pci = _probe_build(
        capsys, "build-pci-fixture", "--source", str(pci_root), "--drop", DROPPED_BDF
    )
    assert (pci["efa_devices"], pci["dropped"]) == (EXPECTED - 1, [DROPPED_BDF]), pci
    assert pci["efa_functions"] == list(BDFS[:-1]) and pci["unreadable"] is False
    tree = _probe_build(
        capsys,
        "build-infiniband-fixture",
        "--source",
        str(ib_root),
        "--drop",
        DROPPED_DEVICE,
    )
    assert tree["dropped"] == [DROPPED_DEVICE] and DROPPED_DEVICE not in tree["devices"]
    collector = host_case.build(
        "_efa_inventory",
        infiniband_root=tree["path"],
        pci_devices_root=pci["path"],
        expected_efa_device_count=EXPECTED,
        inventory_mismatch_consecutive_samples=2,
    )
    batches = []
    for _ in range(2):
        batches.append(_view(collector.collect_once()))
        host_case.clock.sleep(15)
    detection = _base_doc(batches=batches)
    assert inventory.efa_failure_mode_errors(detection, mod.PCI_DEVICE_MISSING) == []
    assert mod.confirmed_sample_errors(detection, expected=EXPECTED) == [], batches
    first, second = (_values(batch) for batch in batches)
    assert (first["efa_inventory_mismatch"], second["efa_inventory_mismatch"]) == (0, 1)
    assert second["efa_inventory_missing_count"] == 1
    assert second["efa_inventory_discovered_count"] == EXPECTED - 1, second
    assert batches[1]["collection_errors"] == []


def test_the_real_collector_reads_an_unreadable_root_as_unknown_and_resets_the_streak(
    host_case: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    ib_root, pci_root = _fake_sysfs(tmp_path)
    tree = _probe_build(
        capsys,
        "build-infiniband-fixture",
        "--source",
        str(ib_root),
        "--drop",
        DROPPED_DEVICE,
    )
    dropped = ("build-pci-fixture", "--source", str(pci_root), "--drop", DROPPED_BDF)
    pci = _probe_build(capsys, *dropped)
    collector = host_case.build(
        "_efa_inventory",
        infiniband_root=tree["path"],
        pci_devices_root=pci["path"],
        expected_efa_device_count=EXPECTED,
        inventory_mismatch_consecutive_samples=2,
    )

    def tick() -> dict[str, Any]:
        batch = _view(collector.collect_once())
        host_case.clock.sleep(15)
        return batch

    first = tick()  # one missing function: streak 1, not confirmed
    unreadable = _probe_build(capsys, "build-pci-fixture", "--unreadable")
    assert unreadable["path"] == pci["path"] and unreadable["unreadable"] is True
    assert Path(pci["path"]).is_file(), "the unreadable root is a regular file"
    with pytest.raises(OSError, match="Not a directory"):
        efa_inventory_counts(Path(tree["path"]), Path(pci["path"]))
    second = tick()  # the read fails: UNKNOWN, no count, the streak is reset
    third_fixture = _probe_build(capsys, *dropped)
    assert Path(third_fixture["path"]).is_dir(), Path(third_fixture["path"])
    third = tick()  # missing again: the streak restarts at 1, not confirmed
    fourth = tick()  # the second consecutive missing tick confirms
    assert _values(first)["efa_inventory_mismatch"] == 0, first
    assert second["samples"] == [], second
    errors = inventory.efa_collection_errors(_base_doc(batches=[second]))
    assert errors and mod.UNKNOWN_MARKER in errors[0], errors
    assert "Not a directory" in errors[0], errors
    assert _values(third)["efa_inventory_mismatch"] == 0, (
        "a failed read never counts toward the confirmation streak"
    )
    assert _values(fourth)["efa_inventory_mismatch"] == 1, fourth
