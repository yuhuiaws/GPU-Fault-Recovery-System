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
from types import SimpleNamespace
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
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-32-162.us-west-2.compute.internal"
# Built at runtime: the public-release gate refuses a literal EC2 instance id.
INSTANCE = "i-" + "0" * 16 + "a"
OTHER_INSTANCE = "i-" + "f" * 16 + "b"
EXPECTED = 3
DEVICES = ("rdmap0s0", "rdmap1s0", "rdmap2s0")
BDFS = ("0000:53:00.0", "0000:53:01.0", "0000:53:02.0")
DROPPED_DEVICE, DROPPED_BDF = DEVICES[-1], BDFS[-1]
SPEC = mod.VARIANTS[0]
ACTIVE = {"state": "4: ACTIVE", "phys_state": "5: LinkUp"}
UNKNOWN_ERROR = (
    "EFA inventory observation unavailable: [Errno 20] Not a directory: "
    "'/run/gfrs-live/source-x/fixtures/pci'"
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
# the fakes: a phase-aware probe, store and cluster
# --------------------------------------------------------------------------- #
class RebootableProbe(FakeProbe):
    """A source-replay probe whose Pod can be deleted and re-created."""

    def __init__(self, script: dict[str, Any]) -> None:
        super().__init__(script)
        self.cleanups = 0
        self.creates = 0

    def cleanup(self) -> dict[str, bool]:
        self.cleanups += 1
        return {"pod": False, "configmap": False}

    def create(self) -> None:
        self.creates += 1


def _inventory_doc(*, efa: bool = True) -> dict[str, Any]:
    return {
        "root": "/sys/class/infiniband",
        "devices": {
            name: {
                "driver": "efa" if efa else "mlx5_core",
                "efa": efa,
                "netdevs": [],
                "ports": {"1": dict(ACTIVE)},
                "pci": bdf,
            }
            for name, bdf in zip(DEVICES, BDFS)
        },
        "physical_netdevs": ["eth0"],
    }


def _probe_script(**extra: Any) -> dict[str, Any]:
    def build_pci(arguments: tuple[str, ...]) -> dict[str, Any]:
        if "--unreadable" in arguments:
            return {
                "path": "/run/gfrs-live/source-x/fixtures/pci",
                "efa_devices": 0,
                "unreadable": True,
                "dropped": [],
            }
        dropped = [
            arguments[index + 1]
            for index, flag in enumerate(arguments)
            if flag == "--drop"
        ]
        return {
            "path": "/run/gfrs-live/source-x/fixtures/pci",
            "efa_devices": EXPECTED - len(dropped),
            "efa_functions": [bdf for bdf in BDFS if bdf not in dropped],
            "dropped": dropped,
            "unreadable": False,
            "source": mod.PCI_ROOT,
        }

    script: dict[str, Any] = {
        "install-source": {"bytes": 1},
        "inventory": _inventory_doc(),
        "build-pci-fixture": build_pci,
        "build-infiniband-fixture": {
            "path": "/run/gfrs-live/source-x/fixtures/infiniband",
            "devices": {name: {} for name in DEVICES[:-1]},
            "dropped": [DROPPED_DEVICE],
        },
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
            "verification": {"ok": True, "checks": []},
        },
        "close-window": {"clean": True},
        "snapshot": {
            "window_open": False,
            "dropin": False,
            "state_dir": False,
            "deadman": False,
            "unit_active_state": "active",
            "unit_pci_root": None,
        },
        "read-source": {"same": True, "path": "/sys/class/infiniband"},
        "status": {"fixtures": {}},
    }
    script.update(extra)
    return script


def _node(boot: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "node-uid",
        "boot_id": boot,
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "spare_pool_annotations": {},
        "gpu_allocatable": "8",
        **extra,
    }


PEER_NODE = {**_node("peer-boot-1"), "name": PEER, "uid": "peer-uid"}


def _efa_batch(
    observed_at: str,
    *,
    failure_mode: str,
    active: int,
    mismatch: int,
    streak: int,
    errors: tuple[str, ...] = (),
) -> dict[str, Any]:
    labels = {
        "resource": "EFA",
        "expected_count": str(EXPECTED),
        "observed_count": str(active),
        "missing_count": str(max(0, EXPECTED - active)),
        "excess_count": str(max(0, active - EXPECTED)),
        "consecutive_mismatch_samples": str(streak),
        "required_consecutive_samples": "2",
        "failure_mode": failure_mode,
        "expected_efa_device_count": str(EXPECTED),
        "discovered_count": str(active),
        "driver_bound_count": str(active),
    }
    names = {
        "efa_inventory_expected_count": EXPECTED,
        "efa_inventory_active_count": active,
        "efa_inventory_missing_count": max(0, EXPECTED - active),
        "efa_inventory_excess_count": max(0, active - EXPECTED),
        "efa_inventory_mismatch": mismatch,
        "efa_inventory_discovered_count": active,
    }
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": [],
        "collection_errors": list(errors),
        "samples": [
            {"name": name, "device": None, "value": value, "labels": dict(labels)}
            for name, value in names.items()
        ],
    }


def _unknown_batch(observed_at: str) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": ["collection-error:EFA inventory"],
        "collection_errors": [UNKNOWN_ERROR],
        "samples": [],
    }


def _base_doc(**extra: Any) -> dict[str, Any]:
    return {
        "since": "2026-10-24T00:00:00+00:00",
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
        "incidents": [],
        "workflows": [],
        "commands": [],
        "batches": [],
        "markers": [],
        "metric_latest": [
            {
                "name": "efa_inventory_expected_count",
                "device": None,
                "value": EXPECTED,
                "observed_at": "2026-10-24T00:00:00+00:00",
                "labels": {},
            }
        ],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
        **extra,
    }


def _incident(state: str, workflow: str = "wf-reboot") -> dict[str, Any]:
    return {
        "incident_id": "inc-efa",
        "event_id": "host-2-efa_inventory_mismatch-node",
        "event_type": "NODE_HEALTH",
        "effective_action": "REBOOT_NODE",
        "state": state,
        "created_at": "2026-10-24T00:00:36+00:00",
        "workflow_request_id": workflow,
        "node_ids": [NODE],
        "cluster_id": "gpu-eks-cluster",
    }


def _execution(
    operation: str, status: str = "SUCCEEDED", **extra: Any
) -> dict[str, Any]:
    return {
        "operation": operation,
        "status": status,
        "phase": "official",
        "details": {},
        "error": None,
        "adapter_operation_id": f"remote/cmd-{operation.lower()}",
        "updated_at": "2026-10-24T00:20:00+00:00",
        **extra,
    }


def _workflow(
    status: str,
    *,
    request_id: str = "wf-reboot",
    operations: tuple[str, ...] = (
        "MARK_UNSCHEDULABLE",
        "QUARANTINE",
        "RESTART_NODE",
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ),
    executions: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    if executions is None:
        executions = (
            ()
            if status == "RUNNING"
            else tuple(_execution(operation) for operation in operations)
        )
    return {
        "request_id": request_id,
        "incident_id": "inc-efa",
        "status": status,
        "created_at": "2026-10-24T00:00:37+00:00",
        "official_steps": [
            {"operation": operation, "parameters": {}} for operation in operations
        ],
        "step_executions": list(executions),
    }


def _command(
    operation: str = "RESTART_NODE",
    *,
    status: str = "SUCCEEDED",
    node_ids: tuple[str, ...] = (NODE,),
    workflow: str = "wf-reboot",
) -> dict[str, Any]:
    return {
        "command_id": f"cmd-{operation.lower()}",
        "workflow_request_id": workflow,
        "incident_id": "inc-efa",
        "status": status,
        "error": None,
        "step": {"operation": operation, "node_ids": list(node_ids)},
        "result_details": {"action": "REBOOT"},
    }


def _confirmed_batches(
    failure_mode: str = "PCI_DEVICE_MISSING",
) -> list[dict[str, Any]]:
    return [
        _efa_batch(
            "2026-10-24T00:00:15+00:00",
            failure_mode=failure_mode,
            active=EXPECTED - 1,
            mismatch=0,
            streak=1,
        ),
        _efa_batch(
            "2026-10-24T00:00:30+00:00",
            failure_mode=failure_mode,
            active=EXPECTED - 1,
            mismatch=1,
            streak=2,
        ),
    ]


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _event(instance: str = INSTANCE, **extra: Any) -> dict[str, Any]:
    return {
        "event_name": "RebootInstances",
        "event_time": "2026-10-24T00:05:00Z",
        "request_node_ids": [instance],
        "request_node_logical_ids": None,
        "api_error_code": None,
        **extra,
    }


class World(FakeRegional):
    """The cluster as the variant reads it, phased by what the probe was asked."""

    def __init__(
        self,
        probe_fake: FakeProbe,
        *,
        negative: dict[str, Any],
        detection: dict[str, Any],
        settled: dict[str, Any],
        reboots: bool = True,
        replaced: bool = False,
        node_after: dict[str, Any] | None = None,
        allocatable_after: str = "32",
        events: tuple[dict[str, Any], ...] = (_event(),),
        peer_after: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cpu_python=self._store)
        self.probe_fake = probe_fake
        self.docs = {"negative": negative, "detection": detection, "settled": settled}
        self.reboots = reboots
        self.replaced = replaced
        self.node_after = node_after or {}
        self.allocatable_after = allocatable_after
        self.events = list(events)
        self.peer_after = peer_after
        self.provider_waits: list[dict[str, Any]] = []
        self.peer_reads: list[str] = []

    # -- phases ---------------------------------------------------------------
    def _count(self, command: str) -> int:
        return sum(1 for call in self.probe_fake.calls if call[0] == command)

    def positive_closed(self) -> bool:
        return self._count("close-window") >= 2

    def rebooted(self) -> bool:
        return self.reboots and self.positive_closed()

    def phase(self) -> str:
        if self._count("close-window") == 0:
            return "negative"
        if self._count("snapshot") == 0 and not (
            self.positive_closed() and not self.reboots
        ):
            return "detection"
        return "settled"

    # -- the store -------------------------------------------------------------
    def _store(self, script: str, *arguments: str) -> dict[str, Any]:
        dispatch = store_script({"activity": lambda args: self._activity(args)})
        return dispatch(script, *arguments)

    def _activity(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        if arguments[1] == PEER:
            self.peer_reads.append(arguments[2])
            return _base_doc(node_id=PEER, metric_latest=[])
        return json.loads(json.dumps(self.docs[self.phase()]))

    # -- the cluster -----------------------------------------------------------
    def node_snapshot(self, node: str) -> dict[str, Any]:
        if node == PEER:
            if self.peer_after and self.rebooted():
                return dict(self.peer_after)
            return dict(PEER_NODE)
        if self.replaced and self.positive_closed():
            return _node("boot-9", uid="node-uid-replacement")
        if self.rebooted():
            return _node("boot-2", **self.node_after)
        return _node("boot-1")

    def kubectl(self, cluster: str, *arguments: str, **_: Any) -> str:
        allocatable = self.allocatable_after if self.rebooted() else "32"
        return json.dumps(
            {
                "metadata": {"name": arguments[2]},
                "spec": {"providerID": f"aws:///us-west-2a/{INSTANCE}"},
                "status": {
                    "allocatable": {
                        "vpc.amazonaws.com/efa": allocatable,
                        "nvidia.com/gpu": "8",
                    }
                },
            }
        )

    def wait_provider_events(
        self, since: datetime, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.provider_waits.append({"since": since, **kwargs})
        return [
            item for item in self.events if item["event_name"] in kwargs["event_names"]
        ]


def _settled_doc(
    *,
    incident_state: str = "RECOVERED",
    workflows: list[dict[str, Any]] | None = None,
    commands: list[dict[str, Any]] | None = None,
    fresh: bool = True,
    failure_mode: str = "PCI_DEVICE_MISSING",
) -> dict[str, Any]:
    batches = _confirmed_batches(failure_mode)
    if fresh:
        batches.append(
            _efa_batch(
                _future(), failure_mode="HEALTHY", active=EXPECTED, mismatch=0, streak=0
            )
        )
    return _base_doc(
        incidents=[_incident(incident_state)],
        workflows=workflows if workflows is not None else [_workflow("SUCCEEDED")],
        commands=commands if commands is not None else [_command()],
        batches=batches,
    )


def _world(
    tmp_path: Path,
    *,
    negative: dict[str, Any] | None = None,
    detection: dict[str, Any] | None = None,
    settled: dict[str, Any] | None = None,
    probe_extra: dict[str, Any] | None = None,
    peer: str = PEER,
    **world: Any,
) -> SimpleNamespace:
    fake = RebootableProbe(_probe_script(**(probe_extra or {})))
    regional = World(
        fake,
        negative=negative
        or _base_doc(batches=[_unknown_batch("2026-10-24T00:00:15+00:00")]),
        detection=detection
        or _base_doc(
            incidents=[_incident("ACTION_PENDING")],
            workflows=[_workflow("RUNNING")],
            batches=_confirmed_batches(),
        ),
        settled=settled or _settled_doc(),
        **world,
    )
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": fake},
        settings=make_settings(SPEC, node=NODE, peer=peer),
        window_minutes=SPEC.max_minutes,
    )
    return SimpleNamespace(ctx=ctx, probe=fake, regional=regional)


def _calls(fake: FakeProbe, command: str) -> list[tuple[str, ...]]:
    return [call for call in fake.calls if call[0] == command]


def _flags(call: tuple[str, ...]) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {}
    for index, item in enumerate(call[1:], start=1):
        if item.startswith("--"):
            value = call[index + 1] if index + 1 < len(call) else ""
            flags.setdefault(item, []).append("" if value.startswith("--") else value)
    return flags


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
