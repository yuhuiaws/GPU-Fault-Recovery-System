"""Fakes and record builders of the GF-GENERAL-LIVE-013 ``pci_missing`` tests.

A phase-aware source-replay probe whose Pod can be deleted and re-created, the
store documents of the negative, detection and settled phases as the variant
reads them, the cluster (node snapshots, kubectl, the provider receipt) and
the builders of the product records they carry. Shared by
``test_general_live_efa_pci.py`` (registry, flow, graders, real collector) and
``test_general_live_efa_pci_reboot_record.py`` (the live hp 2026-10-27 oracle
defects).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.e2e.regional import general_live_efa_pci as mod
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    StatefulSourceProbe,
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


# --------------------------------------------------------------------------- #
# the fakes: a phase-aware probe, store and cluster
# --------------------------------------------------------------------------- #
class RebootableProbe(StatefulSourceProbe):
    """A source-replay probe whose Pod can be deleted and re-created.

    It keeps the node's state dir per run like the live probe: the close
    removes the installed builders, the new boot removes everything.
    """

    def __init__(self, script: dict[str, Any]) -> None:
        super().__init__(script)
        self.cleanups = 0
        self.creates = 0

    def cleanup(self) -> dict[str, bool]:
        self.cleanups += 1
        return {"pod": False, "configmap": False}

    def create(self) -> None:
        self.creates += 1
        super().create()


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


def _judged(value: int = EXPECTED, observed_at: str | None = None) -> dict[str, Any]:
    """``details`` of a VALIDATE_FABRIC that judged the inventory on ``value``
    (``adapters/gpu_validation`` ``inventory_evidence``, batch 175)."""

    return {
        "validation": "VALIDATE_FABRIC",
        "inventory_evidence": {
            NODE: {
                "efa_inventory_active_count": {
                    "value": value,
                    "expected": EXPECTED,
                    "observed_at": observed_at or _future(),
                }
            }
        },
    }


def _execution(
    operation: str, status: str = "SUCCEEDED", **extra: Any
) -> dict[str, Any]:
    details = _judged() if operation == "VALIDATE_FABRIC" else {}
    return {
        "operation": operation,
        "status": status,
        "phase": "official",
        "details": details,
        "error": None,
        "adapter_operation_id": f"remote/cmd-{operation.lower()}",
        "started_at": "2026-10-24T00:10:00+00:00",
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
        labels: dict[str, str] | None = None,
        run_output: Callable[[list[str]], str] | None = None,
        settled_later: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            cpu_python=self._store,
            run_output=run_output,
            node_metadata={"name": NODE, "labels": labels} if labels else None,
        )
        self.probe_fake = probe_fake
        self.docs = {"negative": negative, "detection": detection, "settled": settled}
        # The store after the settle read: what the fresh-inventory wait polls.
        self.settled_later = settled_later
        self.settled_reads = 0
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
        phase = self.phase()
        if phase == "settled":
            self.settled_reads += 1
            if self.settled_later is not None and self.settled_reads > 1:
                return json.loads(json.dumps(self.settled_later))
        return json.loads(json.dumps(self.docs[phase]))

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
# live hp 2026-10-27: the records of the three oracle defects (b203)
# --------------------------------------------------------------------------- #
# Synthetic HyperPod identities: the public-release gate refuses literal ones.
LOGICAL_ID = "11111111-2222-4333-8444-555555555555"
OTHER_LOGICAL_ID = "66666666-7777-4888-9999-000000000000"
HP_LABELS = {"sagemaker.amazonaws.com/compute-type": "hyperpod"}


def _rdma_batch(observed_at: str) -> dict[str, Any]:
    """The negative window as the live node delivered it: the inventory UNKNOWN,
    ``_rdma``'s per-port readings of the real RDMA tree still in the batch."""

    batch = _unknown_batch(observed_at)
    batch["samples"] = [
        {
            "name": "rdma_link_down",
            "device": device,
            "value": 0,
            "labels": {"port": "1"},
        }
        for device in DEVICES
    ]
    return batch


def _healthy_batch(observed_at: str) -> dict[str, Any]:
    return _efa_batch(
        observed_at, failure_mode="HEALTHY", active=EXPECTED, mismatch=0, streak=0
    )


def _hp_inventory(command: list[str]) -> str:
    if "list-cluster-nodes" not in command:
        return "null"
    assert "--include-node-logical-ids" in command, command
    assert command[command.index("--cluster-name") + 1] == "hp-cluster", command
    return json.dumps(
        {
            "ClusterNodeSummaries": [
                {"InstanceId": OTHER_INSTANCE, "NodeLogicalId": OTHER_LOGICAL_ID},
                {"InstanceId": INSTANCE, "NodeLogicalId": LOGICAL_ID},
            ]
        }
    )


def _hp_event(*logical: str, **extra: Any) -> dict[str, Any]:
    return _event(
        event_name="BatchRebootClusterNodes",
        request_node_ids=None,
        request_node_logical_ids=list(logical or (LOGICAL_ID,)),
        **extra,
    )


def _r3_settled(
    *,
    baseline: dict[str, Any] | None,
    latest_active: int = EXPECTED,
    latest_at: str = "2026-10-24T00:25:00+00:00",
    workflows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The store as live hp 2026-10-27 r3 left it: the new boot's first batch
    before RESTART_NODE completed (00:20), then health summaries only -- never
    raw evidence -- advancing the latest-value rows."""

    settled = _settled_doc(fresh=False, workflows=workflows)
    if baseline is not None:
        settled["batches"].append(baseline)
    settled["metric_latest"].append(
        {
            "name": "efa_inventory_active_count",
            "device": None,
            "value": latest_active,
            "observed_at": latest_at,
            "labels": {},
        }
    )
    return settled


def _baseline_batch(
    observed_at: str = "2026-10-24T00:15:00+00:00", **counts: Any
) -> dict[str, Any]:
    batch = _efa_batch(
        observed_at,
        **{
            "failure_mode": "HEALTHY",
            "active": EXPECTED,
            "mismatch": 0,
            "streak": 0,
            **counts,
        },
    )
    batch["edge_filter_reasons"] = ["baseline"]
    return batch
