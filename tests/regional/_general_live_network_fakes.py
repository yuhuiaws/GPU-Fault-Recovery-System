"""Fakes shared by the GF-GENERAL-LIVE-008/009 network tests.

The node survey, the scripted net/node/window probes (``FakeProbe``) and the
test interfaces of ``test_general_live_network.py``, plus the records of the
009 ``counter_reset`` variant (``test_general_live_counter_reset.py``): the
product's stored edges around a one-process recreate of the drop pair, the
netdev oracle that watches both bursts and the index change, and what the net
probe's ``recreate-veth`` prints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional import general_live_network as net
from tests.regional._general_live_fakes import FakeProbe

SPECS = {spec.variant: spec for spec in net.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
IFA = "gfrsaaaaaaa"
IFB = "gfrsaaaaaab"
IFC = "gfrsaaaaaac"
IFD = "gfrsaaaaaad"


def survey(required: str = "") -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {name: True for name in ("ip", "tc", "systemd-run")},
        "modules": {
            "veth": {"loaded": True, "available": True},
            "dummy": {"loaded": True, "available": True},
            "sch_netem": {"loaded": True, "available": True},
        },
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
            "GPU_FAULT_REQUIRED_INTERFACES": required,
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def netdev(
    *,
    has_device: bool = True,
    default_route: bool = False,
    policy_tables: tuple[str, ...] = (),
    rdma: bool = False,
    driver: str | None = "ena",
    addresses: tuple[str, ...] = ("10.0.43.5/24",),
) -> dict[str, Any]:
    return {
        "has_device": has_device,
        "operstate": "up",
        "driver": driver if has_device else None,
        "rdma": rdma,
        "addresses": list(addresses),
        "default_route": default_route,
        "policy_tables": list(policy_tables),
    }


#: A node with a dedicated data NIC beside the management interface: the
#: spec's site FAIL applies when the required list is empty.
INVENTORY_WITH_DATA_NIC = {
    "interfaces": {
        "enp71s0": netdev(default_route=True),
        "enp72s0": netdev(driver="mlx5_core", rdma=True, addresses=("10.1.0.5/24",)),
        "eni9f0a1b2c3d4": netdev(has_device=False, addresses=()),
    },
    "default_route_devices": ["enp71s0"],
    "policy_tables": {},
}
#: This site's p5en: the primary ENA (default route), the VPC CNI's secondary
#: ENIs (one policy table each), host-side pod veths; EFA-only rdmap devices
#: have no netdev. Nothing to declare.
INVENTORY_WITHOUT_DATA_NIC = {
    "interfaces": {
        "ens5": netdev(default_route=True),
        "ens6": netdev(policy_tables=("2",), addresses=("10.0.44.7/24",)),
        "ens7": netdev(policy_tables=("3",), addresses=("10.0.45.9/24",)),
        "eni9f0a1b2c3d4": netdev(has_device=False, addresses=()),
        "enib1c2d3e4f5a": netdev(has_device=False, addresses=()),
    },
    "default_route_devices": ["ens5"],
    "policy_tables": {"2": ["ens6"], "3": ["ens7"]},
}


def net_probe(down: str | None = None, **extra: Any) -> FakeProbe:
    operstate = {IFA: "up", IFB: "up", IFC: "up", IFD: "up"}
    if down:
        operstate[down] = "down"
    script = {
        "inventory": INVENTORY_WITH_DATA_NIC,
        "create-veth": [
            {"a": IFA, "b": IFB, "link_a": {"operstate": "up"}},
            {"a": IFC, "b": IFD, "link_a": {"operstate": "up"}},
        ],
        "create-dummy": {"name": "gfrsaaaaaad", "link": {"operstate": "unknown"}},
        "set-link": lambda args: {
            "name": args[2],
            "state": args[-1],
            "link": {"operstate": args[-1]},
        },
        "delete-interface": {"deleted": IFA, "present": False},
        "recreate-veth": recreate_output(),
        "start-drops": {"unit": "gfrs-x-drops"},
        "stop-drops": {"stopped": "gfrs-x-drops"},
        "reading": {
            "interfaces": {
                name: {"operstate": state} for name, state in operstate.items()
            }
        },
        "cleanup": {"clean": True},
    }
    script.update(extra)
    return FakeProbe(script)


def node_probe(
    survey_value: dict[str, Any], oracle: dict[str, Any] | None = None
) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_value,
            "oracle-start": {"unit": "oracle"},
            "oracle-stop": {},
            "oracle-read": oracle or {"records": [], "next_offset": 0, "total": 0},
        }
    )


def window_probe() -> FakeProbe:
    return FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )


def probes(node: FakeProbe, netp: FakeProbe, win: FakeProbe) -> dict[str, FakeProbe]:
    return {f"node@{NODE}": node, f"net@{NODE}": netp, f"window@{NODE}": win}


RESET_ERROR = (
    f"_network: HostObservationError: {IFA}: rx_dropped reset (400 -> 0); delta unknown"
)
#: The variant's own clock (``general_live_network_reset.utc_now``): the first
#: burst starts at R0, the recreate is asked at R0+120 s, later readings 60 s
#: apart. The oracle's node clock and the product's ``observed_at`` are on it.
R0 = datetime(2026, 10, 25, 21, 53, tzinfo=timezone.utc)
R0_EPOCH = R0.timestamp()
#: The probe dates the recreate on the node's clock.
RECREATED_AT = 125.0


def reset_at_text(seconds: float) -> str:
    return (R0 + timedelta(seconds=seconds)).isoformat()


def reset_batch(
    seconds: float,
    samples: list[dict[str, Any]],
    *,
    errors: tuple[str, ...] = (),
    reasons: tuple[str, ...] = ("threshold:network_drops_delta",),
) -> dict[str, Any]:
    return {
        "observed_at": reset_at_text(seconds),
        "batch_id": f"host-{NODE}-{int(seconds)}",
        "edge_filter_reasons": list(reasons),
        "collection_errors": list(errors),
        "samples": samples,
    }


def drops_sample(value: float, device: str = IFA) -> dict[str, Any]:
    return {
        "name": "network_drops_delta",
        "device": device,
        "value": value,
        "labels": {},
    }


def link_down_sample(value: float, device: str) -> dict[str, Any]:
    return {"name": "network_link_down", "device": device, "value": value, "labels": {}}


def reset_status(seconds: float) -> dict[str, Any]:
    """The HOST_TELEMETRY status row: upserted for every accepted batch."""

    return {
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
        "collector": "HOST_TELEMETRY",
        "observed_at": reset_at_text(seconds),
        "ingested_at": reset_at_text(seconds + 1),
        "last_success_at": reset_at_text(seconds),
        "last_error_at": None,
        "batch_id": f"host-{NODE}-{int(seconds)}",
        "sample_count": 101,
        "errors": [],
    }


def reset_doc(
    *,
    marked: bool = True,
    reset_tick_reading: bool = False,
    burst: bool = True,
    burst_zero: bool = True,
    ramp: bool = True,
    zero_after: bool = True,
    status_at: float | None = None,
    incidents: list[dict[str, Any]] | None = None,
    workflows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The product's stored edges around the recreate.

    The first burst's threshold edge (R0+20 s) and recovered edge (R0+50 s);
    ``marked``: the reset tick (R0+130 s) carries the ``_network`` reset error
    and no drops reading for the interface (batch 173), ``reset_tick_reading``
    puts one in; ``ramp``: the second burst's edge from the new baseline
    (R0+200 s) and its recovered edge (R0+230 s). The status row sits on the
    newest batch unless ``status_at`` says otherwise.
    """

    batches = []
    if burst:
        batches.append(reset_batch(20, [drops_sample(400.0), drops_sample(400.0, IFB)]))
        if burst_zero:
            batches.append(
                reset_batch(
                    50,
                    [drops_sample(0.0), drops_sample(0.0, IFB)],
                    reasons=("recovered",),
                )
            )
    if marked:
        tick = [drops_sample(0.0)] if reset_tick_reading else []
        batches.append(
            reset_batch(
                130, tick, errors=(RESET_ERROR,), reasons=("collection-error:_network",)
            )
        )
    if ramp:
        batches.append(
            reset_batch(200, [drops_sample(150.0), drops_sample(150.0, IFB)])
        )
        if zero_after:
            batches.append(
                reset_batch(
                    230,
                    [drops_sample(0.0), drops_sample(0.0, IFB)],
                    reasons=("recovered",),
                )
            )
    newest = max([20.0, *(130.0 if marked else 20.0,), *(200.0 if ramp else 20.0,)])
    return {
        "incidents": list(incidents or []),
        "workflows": list(workflows or []),
        "batches": batches,
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [
            reset_status(newest if status_at is None else status_at)
        ],
    }


def reset_oracle(*, rise: bool = True, recreate: bool = True) -> list[dict[str, Any]]:
    """The netdev oracle, 5-s samples over 300 s: the pair's drops rise to 400
    before the recreate (node clock R0+125 s), the index changes and the
    counters restart, the second burst counts 150 from R0+180 s."""

    rows = []
    for index in range(61):
        moment = R0_EPOCH + 5 * index
        if moment < R0_EPOCH + RECREATED_AT:
            drops = min(400, 20 * index) if rise else 0
            ifindex = 31
        else:
            ramp = int(moment - (R0_EPOCH + 180)) * 10
            drops = max(0, min(150, ramp)) if rise else 0
            ifindex = 33 if recreate else 31
        rows.append(
            {
                "t": moment,
                "at": datetime.fromtimestamp(moment, timezone.utc).isoformat(),
                "netdev": {
                    IFA: {
                        "present": True,
                        "ifindex": ifindex,
                        "operstate": "up",
                        "rx_errors": 0,
                        "tx_errors": 0,
                        "rx_dropped": 0,
                        "tx_dropped": drops,
                    },
                    IFB: {
                        "present": True,
                        "ifindex": ifindex - 1,
                        "operstate": "up",
                        "rx_errors": 0,
                        "tx_errors": 0,
                        "rx_dropped": drops,
                        "tx_dropped": 0,
                    },
                },
            }
        )
    done = {"records": [], "next_offset": len(rows), "total": len(rows)}
    return [{"records": rows, "next_offset": len(rows), "total": len(rows)}, done]


def recreate_output(*, new_index: str = "33") -> dict[str, Any]:
    """What ``recreate-veth`` prints: the pair before and after, the gap."""

    return {
        "a": IFA,
        "b": IFB,
        "deadman": "gfrs-aaaaaa-ndeadman-interfaces",
        "before": {
            "a": {"present": True, "ifindex": "31", "operstate": "up"},
            "b": {"present": True, "ifindex": "30", "operstate": "up"},
        },
        "link_a": {"present": True, "ifindex": new_index, "operstate": "up"},
        "link_b": {
            "present": True,
            "ifindex": str(int(new_index) - 1),
            "operstate": "up",
        },
        "delete_ms": 3.1,
        "gap_ms": 2.4,
        "recreated_at": reset_at_text(RECREATED_AT),
        "management_digest": "d" * 64,
    }
