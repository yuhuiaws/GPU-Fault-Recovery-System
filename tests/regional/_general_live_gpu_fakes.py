"""Fakes for the lane-R DCGM / GPU-inventory replay category tests.

No cluster, no node, no systemd: ``dcgm_probe`` scripts the DCGM replay
controller's subcommands (survey, install-source, deploy/arm/advance/disarm,
repoint/restore, audit-read, stop, close-run, cleanup-leftover) and records
every call; ``gpu_cpu_python``
answers the two read-only store probes the categories use -- the GPU telemetry
state (``general_live_gpu_store.GPU_STATE_PROBE``) and the node activity
(``general_live_store.ACTIVITY_PROBE``) -- from scripted documents so the
verdict helpers run over realistic records. These sit beside the shared
``_general_live_fakes`` (reused for ``FakeRegional`` / ``make_context``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from scripts.e2e.regional.general_live_gpu_generation import (
    PENDING_FLAG_VERDICT_PROBE,
    rejudging_answer,
)
from scripts.e2e.regional.general_live_gpu_store import GPU_STATE_PROBE
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from tests.regional._general_live_fakes import FakeProbe


def survey(
    *,
    gpus: Sequence[str] = (
        "GPU-00000000-0000-4000-8000-000000000000",
        "GPU-11111111-1111-4111-8111-111111111111",
    ),
    model: str = "NVIDIA H200",
    fields: Sequence[str] = (
        "ecc_sbe_volatile_total",
        "ecc_dbe_volatile_total",
        "ecc_dbe_aggregate_total",
        "retired_pages_dbe_total",
        "row_remap_pending",
        "row_remap_failure",
        "pcie_replay_total",
        "nvlink_crc_flit_error_total",
        "nvlink_replay_error_total",
        "nvlink_recovery_error_total",
        "clock_throttle_reasons",
        "fabric_manager_status",
    ),
    prior_boots: Sequence[str] = ("22222222-2222-4222-8222-222222222222",),
    residue_clean: bool = True,
) -> dict[str, Any]:
    entries = {
        gpu: {
            "gpu_index": str(index),
            "pci_bdf": f"00000000:{index + 1:02d}:00.0",
            "model": model,
            "values": {
                name: (3.0 if name == "fabric_manager_status" else 0.0)
                for name in fields
            },
            "metric_names": {name: [name] for name in fields},
        }
        for index, gpu in enumerate(gpus)
    }
    return {
        "boot_id": "33333333-3333-4333-8333-333333333333",
        "prior_boot_ids": list(prior_boots),
        "collector_env": {
            "GPU_FAULT_METRICS_MODE": "dcgm",
            "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
        },
        "metrics_unit": {"ActiveState": "active", "MainPID": "123"},
        "gpu_inventory": [
            {
                "index": str(i),
                "uuid": gpu,
                "pci_bus_id": entries[gpu]["pci_bdf"],
                "name": model,
            }
            for i, gpu in enumerate(gpus)
        ],
        "exporter": {"sample_count": len(gpus) * len(fields), "gpus": entries},
        "residue": {
            "run_dirs": [],
            "metrics_dropins": [],
            "units": [],
            "runs": {},
            "foreign_dropins": [],
            "foreign_run_dirs": [],
            "port": 9412,
            "port_answers": False,
        }
        if residue_clean
        # An older probe's flat answer: a drop-in that is not ours.
        else {"run_dirs": [], "metrics_dropins": ["x.conf"], "units": []},
        "tools": {"systemd-run": True, "systemctl": True, "nvidia-smi": True},
        "venv_python": True,
    }


def audit_records(
    *served: dict[str, Any], aborted: str | None = None
) -> list[dict[str, Any]]:
    """Replay journal records: one armed read per served map."""

    records = []
    for index, values in enumerate(served):
        records.append(
            {
                "read": index + 1,
                "armed": True,
                "proof_scope": "live-replay",
                "physical_fault_injected": False,
                "served": values,
                "aborted": aborted,
            }
        )
    return records


def dcgm_probe(
    survey_value: dict[str, Any] | None = None,
    *,
    audit: Sequence[dict[str, Any]] | None = None,
    inventory_audit: Sequence[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> FakeProbe:
    gpus = list((survey_value or survey())["exporter"]["gpus"])
    served = (
        list(audit)
        if audit is not None
        else audit_records({f"{gpus[0]}/ecc_sbe_volatile_total": 1.0})
    )

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        journal = _flag(arguments, "--journal", "replay.jsonl")
        offset = int(_flag(arguments, "--offset", "0"))
        source = list(inventory_audit or []) if journal == "inventory.jsonl" else served
        chunk = source[offset:]
        return {
            "records": list(chunk),
            "next_offset": offset + len(chunk),
            "total": len(source),
        }

    script: dict[str, Any] = {
        "survey": survey_value or survey(),
        "install-source": {"ok": True},
        "deploy-proxy": {
            "status": {"armed": False},
            "endpoint": "http://127.0.0.1:9412/metrics",
        },
        "repoint-collector": {"replay_endpoint": "http://127.0.0.1:9412/metrics"},
        "arm": {"armed": True, "status": {"armed": True}},
        "advance": {"advanced": True},
        "disarm": {"disarmed": True},
        "status": {"status": {"armed": True}},
        "collector-restart": {"restarted": True},
        "restore-collector": {
            "restored_endpoint": "http://127.0.0.1:9400/metrics",
            "original_endpoint": "http://127.0.0.1:9400/metrics",
        },
        "stop-proxy": {"run_dir_removed": True, "unit_active": False},
        "close-run": {
            "clean": True,
            "leftover": [],
            "errors": [],
            "deadman_kept": False,
            "disarm": {"disarmed": True, "http_status": 200},
            "restore": {
                "restored_endpoint": "http://127.0.0.1:9400/metrics",
                "original_endpoint": "http://127.0.0.1:9400/metrics",
                "dropin_removed": True,
            },
            "stop": {"run_dir_removed": True, "unit_active": False},
        },
        "cleanup-leftover": {
            "reclaimed": True,
            "clean": True,
            "leftover": [],
            "errors": [],
        },
        "audit-read": audit_read,
    }
    script.update(extra or {})
    return FakeProbe(script)


def _flag(arguments: tuple[str, ...], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def gpu_cpu_python(
    *, gpu_state: Any = None, activity: Any = None, generation: Any = None
) -> Callable[..., dict[str, Any]]:
    """A ``cpu_python`` dispatcher for the GPU-state, activity and
    product-generation probes.

    Each of ``gpu_state`` / ``activity`` / ``generation`` is a document, a
    callable taking the probe arguments, or a list consumed one call at a time
    (the last repeats). ``generation`` is the control plane's answer to
    ``general_live_gpu_generation.PENDING_FLAG_VERDICT_PROBE``; by default a
    product whose post-reset VALIDATE_GPU re-judges the pending flag on newer
    data (``rejudging_answer``), so the reset variants run their full oracle.
    """

    iterators: dict[str, Any] = {}

    def dispatch(script: str, *arguments: str) -> dict[str, Any]:
        name = {
            GPU_STATE_PROBE: "gpu_state",
            ACTIVITY_PROBE: "activity",
            PENDING_FLAG_VERDICT_PROBE: "generation",
        }.get(script, "?")
        value = {
            "gpu_state": gpu_state,
            "activity": activity,
            "generation": generation,
        }.get(name)
        if value is None and name == "generation":
            return rejudging_answer(arguments[0] if arguments else "row_remap_pending")
        if value is None:
            return _empty(name)
        if callable(value):
            return value(arguments)
        if isinstance(value, list):
            iterator = iterators.setdefault(name, iter(value))
            try:
                return dict(next(iterator))
            except StopIteration:
                return dict(value[-1]) if value else _empty(name)
        return dict(value)

    return dispatch


def _empty(name: str) -> dict[str, Any]:
    if name == "gpu_state":
        return {
            "latest": [],
            "active_findings": [],
            "findings": [],
            "xid_events": [],
            "batches": [],
            "inventory_records": [],
            "inventory_snapshot": None,
            "collector_statuses": [],
            "agent": None,
            "capabilities": {},
        }
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def gpu_incident(
    incident_id: str,
    finding_id: str,
    *,
    state: str = "ACTION_PENDING",
    action: str = "RUN_DIAGNOSTICS",
    workflow: str = "wf-1",
    gpus: Sequence[str] = (),
) -> dict[str, Any]:
    """A NODE_HEALTH incident minted by a GPU finding (event id ``gpu-<fid>``)."""

    return {
        "incident_id": incident_id,
        "event_id": f"gpu-{finding_id}",
        "event_type": "NODE_HEALTH",
        "state": state,
        "effective_action": action,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow,
        "node_ids": ["ip-10-0-43-5.us-west-2.compute.internal"],
    }


def gpu_workflow(
    request_id: str,
    incident_id: str,
    operations: Sequence[str],
    *,
    status: str = "RUNNING",
    executions: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": status,
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [{"operation": op, "parameters": {}} for op in operations],
        "step_executions": list(executions),
    }
