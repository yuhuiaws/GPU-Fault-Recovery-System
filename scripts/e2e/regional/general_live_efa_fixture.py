"""The reusable EFA source fixture of GF-GENERAL-LIVE-010 / 011 ``broken_pair``.

What a category needs to replace one EFA reading inside the host collector
unit and to judge the product's records afterwards, with no verdict of its
own: the product's counter and metric names (pinned against
``collectors/host/network.py`` and ``host_health.py`` in the tests), target
selection over the node's real inventory (``inventory`` probe command),
:class:`EfaSourceFixture` (bind a counter/state file, attach a netdev, carry an
``ethtool -S`` plan, open/step/close one window, read the real value back), and
the pure matchers over the activity document (collection errors, zero and
negative readings, the NOT_SUPPORTED marker, the VALIDATE_FABRIC execution,
the wrapper's audit journal). ``general_live_efa.py`` builds the 010 variants
on it; ``general_live_nccl_pair.py`` breaks the traffic pair with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_source_replay import (
    INFINIBAND_TREE,
    HostSourceReplay,
    command_plan,
    host_source_replay,
)
from scripts.e2e.regional.general_live_verdicts import (
    final_execution,
    parse_time,
    samples,
    workflows_of,
)

CASE = "GF-GENERAL-LIVE-010"
INFINIBAND = "/sys/class/infiniband"
PROOF_SCOPE = "live-replay"
#: The port counter files ``_rdma`` sums into ``rdma_errors_delta`` and reports
#: as ``rdma_<counter>_delta`` (collectors/host/network.py; pinned in the tests).
RDMA_COUNTERS = (
    "symbol_error",
    "link_error_recovery",
    "link_downed",
    "port_rcv_errors",
    "port_xmit_discards",
    "port_rcv_remote_physical_errors",
    "port_xmit_constraint_errors",
    "port_rcv_constraint_errors",
)
RDMA_HW_COUNTERS = (
    "local_ack_timeout_err",
    "out_of_sequence",
    "packet_seq_err",
    "poll_cq_err",
    "rnr_nak_retry_err",
    "rx_drops",
    "rx_overrun_errors",
    "send_queue_overflow",
    "tx_drops",
)
#: Which of those files a variant binds, in preference order: the EFA driver
#: exposes ``rx_drops``; the rest are mlx5/IB names a port may or may not have.
COUNTER_PREFERENCE = (
    ("hw_counters", "rx_drops"),
    ("hw_counters", "tx_drops"),
    ("counters", "port_rcv_errors"),
    *(("hw_counters", name) for name in RDMA_HW_COUNTERS),
    *(("counters", name) for name in RDMA_COUNTERS),
)
#: The traffic pair ``rdma_traffic.port_traffic`` reads (011 broken_pair).
TRAFFIC_PAIRS = (("rx_bytes", "tx_bytes"), ("send_bytes", "recv_bytes"))
#: ``ethtool -S`` name token -> the delta it feeds (``_ETHTOOL_GROUPS`` of the
#: product, error groups only; pinned in the tests).
ETHTOOL_GROUPS = {
    "rnr": "efa_rnr_errors_delta",
    "retry": "efa_retry_errors_delta",
    "cq_err": "efa_cq_errors_delta",
}
#: Test-origin counter names (``gfrs`` prefix) carrying exactly one token each,
#: appended by the wrapper when the real listing has no counter of the group.
REPLAY_COUNTERS = {
    "rnr": "gfrs_rnr_err",
    "retry": "gfrs_retry_err",
    "cq_err": "gfrs_cq_err",
}
UNSUPPORTED_METRIC = "efa_ethtool_counters_unsupported"
ERRORS_METRIC = "rdma_errors_delta"
LINK_METRIC = "rdma_link_down"
INVENTORY_METRIC = "efa_inventory_mismatch"
LINK_DOWN_TEXT = "1: DOWN"
FAILED_STDERR = "Cannot get device statistics: Input/output error"
STEP = 5000
RAMP_STEP = 1000
DETECTION_BUDGET = 420
VALIDATION_BUDGET = 600
#: Skew allowed between the runner's clock and the collector's ``observed_at``.
CLOCK_TOLERANCE_SECONDS = 5
PLAN_SECONDS = 3 * 3600
NO_EFA = "no EFA RDMA device (driver efa) in the node's /sys/class/infiniband"


# --------------------------------------------------------------------------- #
# target selection (pure, over the inventory document)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EfaTarget:
    """One EFA port's reading the variant replaces, named as the node has it."""

    device: str
    port: str
    group: str | None = None
    counter: str | None = None
    real_value: str | None = None
    netdev: str | None = None
    #: ``real``: the EFA function owns the netdev; ``fixture``: attached by the tree.
    netdev_source: str = "none"
    ethtool_names: tuple[str, ...] = ()

    @property
    def port_device(self) -> str:
        """The ``device`` label of the port's samples (``<dev>/<port>``)."""

        return f"{self.device}/{self.port}"

    def dest(self, name: str) -> str:
        return f"{INFINIBAND}/{self.device}/ports/{self.port}/{name}"

    @property
    def counter_dest(self) -> str:
        return self.dest(f"{self.group}/{self.counter}")

    @property
    def counter_metric(self) -> str:
        return f"rdma_{self.counter}_delta"

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "port": self.port,
            "group": self.group,
            "counter": self.counter,
            "real_value": self.real_value,
            "netdev": self.netdev,
            "netdev_source": self.netdev_source,
            "ethtool_counters": len(self.ethtool_names),
        }


def efa_devices(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The devices the product's ``is_efa_device`` accepts, by name."""

    return {
        name: device
        for name, device in sorted((inventory.get("devices") or {}).items())
        if device.get("efa")
    }


def _digits(value: Any) -> bool:
    return isinstance(value, str) and value.isascii() and value.isdigit()


def counter_target(inventory: Mapping[str, Any]) -> EfaTarget:
    """The first EFA port with a counter file ``_rdma`` sums, by preference."""

    devices = efa_devices(inventory)
    if not devices:
        raise VariantNotRun("CAPABILITY_ABSENT", NO_EFA)
    for name, device in devices.items():
        for port, entry in sorted((device.get("ports") or {}).items()):
            for group, counter in COUNTER_PREFERENCE:
                value = (entry.get(group) or {}).get(counter)
                if _digits(value):
                    return EfaTarget(name, port, group, counter, value)
    raise VariantNotRun(
        "CAPABILITY_ABSENT",
        "no EFA port exposes a numeric counter file the host collector sums into "
        f"{ERRORS_METRIC} ({', '.join(f'{g}/{c}' for g, c in COUNTER_PREFERENCE[:3])}, ...)",
    )


def state_target(inventory: Mapping[str, Any]) -> EfaTarget:
    """The first EFA port whose ``state`` and ``phys_state`` files read."""

    devices = efa_devices(inventory)
    if not devices:
        raise VariantNotRun("CAPABILITY_ABSENT", NO_EFA)
    for name, device in devices.items():
        for port, entry in sorted((device.get("ports") or {}).items()):
            if entry.get("state") and entry.get("phys_state"):
                return EfaTarget(
                    name, port, None, None, entry["state"], None, "none", ()
                )
    raise VariantNotRun("CAPABILITY_ABSENT", "no EFA port has readable state files")


def pair_target(inventory: Mapping[str, Any]) -> EfaTarget:
    """The first EFA port carrying a complete traffic pair; ``counter`` is its TX half."""

    devices = efa_devices(inventory)
    if not devices:
        raise VariantNotRun("CAPABILITY_ABSENT", NO_EFA)
    for name, device in devices.items():
        for port, entry in sorted((device.get("ports") or {}).items()):
            counters = entry.get("hw_counters") or {}
            for rx, tx in TRAFFIC_PAIRS:
                if _digits(counters.get(rx)) and _digits(counters.get(tx)):
                    return EfaTarget(name, port, "hw_counters", tx, counters[tx])
    raise VariantNotRun(
        "CAPABILITY_ABSENT", "no EFA port exposes a complete rx/tx traffic pair"
    )


def netdev_target(inventory: Mapping[str, Any]) -> EfaTarget:
    """An EFA device with a netdev for ``ethtool -S``: its own, else one attached.

    The host collector runs ``ethtool`` only on ``device/net/*`` of an EFA
    function. When no function owns a netdev, the node's first physical netdev
    is attached to the first EFA device by the fixture tree (``netdev_source``
    ``fixture``) so the reading exists to replay; the netdev is never touched.
    """

    devices = efa_devices(inventory)
    if not devices:
        raise VariantNotRun("CAPABILITY_ABSENT", NO_EFA)
    ethtool = inventory.get("ethtool") or {}
    if not ethtool.get("path"):
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "ethtool is not installed on the node; the host collector skips the "
            "EFA netdev counters without it",
        )
    statistics = ethtool.get("statistics") or {}

    def names(interface: str) -> tuple[str, ...]:
        return tuple(
            str(item)
            for item in (statistics.get(interface) or {}).get("counters") or []
        )

    for name, device in devices.items():
        owned = list(device.get("netdevs") or [])
        if owned:
            return EfaTarget(
                name,
                "1",
                netdev=owned[0],
                netdev_source="real",
                ethtool_names=names(owned[0]),
            )
    physical = [str(item) for item in inventory.get("physical_netdevs") or []]
    if not physical:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "no EFA function owns a netdev and the node has no physical netdev to attach",
        )
    first = next(iter(devices))
    return EfaTarget(
        first,
        "1",
        netdev=physical[0],
        netdev_source="fixture",
        ethtool_names=names(physical[0]),
    )


def real_counter(token: str, names: Iterable[str]) -> str | None:
    """A real ``ethtool -S`` counter the product files under ``token`` alone."""

    others = [item for item in ETHTOOL_GROUPS if item != token]
    for name in names:
        lowered = name.lower()
        if token in lowered and not any(other in lowered for other in others):
            return name
    return None


# --------------------------------------------------------------------------- #
# the reusable fixture
# --------------------------------------------------------------------------- #
class EfaSourceFixture:
    """One node's EFA readings replaced inside the host collector unit.

    ``survey`` reads the real inventory; ``bind`` installs a counter/state file
    and schedules its bind; ``attach_netdev`` builds the tree that gives an EFA
    function a netdev; ``install_ethtool``/``replan_ethtool`` carry a reviewed
    ``ethtool -S`` plan; ``open`` opens the one window with everything scheduled;
    ``step`` moves a bound value while the window is open; ``close`` closes it
    and ``restored_errors`` proves every bound source reads real again.
    """

    def __init__(
        self, ctx: VariantContext, replay: HostSourceReplay | None = None
    ) -> None:
        self.ctx = ctx
        self.replay = replay or host_source_replay(ctx)
        self.inventory: dict[str, Any] = {}
        self.binds: list[tuple[str, str]] = []
        self.values: dict[str, str] = {}
        self.trees: list[tuple[str, str]] = []
        self.wrappers: list[str] = []
        self.opened: dict[str, Any] = {}
        self.closed_at: datetime | None = None

    def survey(self) -> dict[str, Any]:
        self.inventory = self.replay.inventory()
        self.ctx.evidence.write("efa-inventory", self.inventory)
        return self.inventory

    def bind(self, name: str, dest: str, value: str) -> str:
        self.replay.install(name, f"{value}\n".encode("utf-8"))
        self.binds.append((name, dest))
        self.values[name] = str(value)
        return dest

    def attach_netdev(self, target: EfaTarget) -> dict[str, Any]:
        if target.netdev is None:
            raise ValueError("the target names no netdev to attach")
        built = self.replay.build_infiniband(netdev=[(target.device, target.netdev)])
        self.trees.append((INFINIBAND_TREE, INFINIBAND))
        self.ctx.evidence.write("efa-netdev-tree", built)
        return built

    def install_ethtool(self, rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        receipt = self.replay.install_wrapper(
            "ethtool", command_plan("ethtool", rules, not_after_seconds=PLAN_SECONDS)
        )
        if "ethtool" not in self.wrappers:
            self.wrappers.append("ethtool")
        return receipt

    def replan_ethtool(self, rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self.replay.replan(
            "ethtool", command_plan("ethtool", rules, not_after_seconds=PLAN_SECONDS)
        )

    def open(self) -> dict[str, Any]:
        restore = min(5400, max(60, self.ctx.spec.max_minutes * 60 + 300))
        self.opened = self.replay.open_window(
            binds=self.binds,
            trees=self.trees,
            wrappers=self.wrappers,
            restore_seconds=restore,
        )
        self.ctx.evidence.write("source-window", self.opened)
        return self.opened

    def step(self, name: str, value: int | str) -> dict[str, Any]:
        self.values[name] = str(value)
        return self.replay.set_value(name, value)

    def audit(self) -> dict[str, Any]:
        journal = self.replay.audit("ethtool")
        self.ctx.evidence.write("ethtool-audit", journal)
        return journal

    def close(self) -> dict[str, Any]:
        receipt = self.replay.close()
        self.closed_at = self.ctx.clock()
        self.ctx.evidence.write("source-window-close", receipt)
        if not receipt.get("clean", True):
            raise RuntimeError(f"the source window did not close clean: {receipt}")
        return receipt

    def restored_errors(self) -> list[str]:
        """Every bound source reads the same real value in the unit and on the host."""

        errors: list[str] = []
        views = []
        for name, dest in self.binds:
            view = self.replay.read_back(dest)
            views.append(view)
            unit, host = view.get("unit") or {}, view.get("host") or {}
            if not (unit.get("readable") and host.get("readable")):
                errors.append(f"{dest}: not readable after the window closed")
            elif not view.get("same"):
                errors.append(
                    f"{dest}: the unit still reads another value than the host"
                )
            elif unit.get("value") == self.values.get(name):
                errors.append(f"{dest}: the unit still reads the fixture value")
        self.ctx.evidence.write("source-read-back", {"views": views, "errors": errors})
        return errors


# --------------------------------------------------------------------------- #
# verdict pieces (pure, over the activity document)
# --------------------------------------------------------------------------- #
def collection_errors(document: Mapping[str, Any], *needles: str) -> list[str]:
    """The batches' collection errors that contain every needle."""

    return [
        str(error)
        for batch in document.get("batches") or []
        for error in batch.get("collection_errors") or []
        if all(needle in str(error) for needle in needles)
    ]


def metric_values(document: Mapping[str, Any], metric: str, device: str) -> list[float]:
    return [
        float(item["value"])
        for item in samples(dict(document), metric, device=device)
        if isinstance(item.get("value"), (int, float))
    ]


def _since(sample: Mapping[str, Any], moment: datetime | None) -> bool:
    if moment is None:
        return True
    observed = parse_time(sample.get("observed_at"))
    return observed is not None and observed >= moment - timedelta(
        seconds=CLOCK_TOLERANCE_SECONDS
    )


def zero_readings(
    document: Mapping[str, Any],
    metric: str,
    device: str,
    *,
    after: datetime | None = None,
) -> list[dict[str, Any]]:
    """Samples of ``metric`` on ``device`` that read exactly 0 (after ``after``)."""

    return [
        item
        for item in samples(dict(document), metric, device=device)
        if item.get("value") == 0 and _since(item, after)
    ]


def negative_readings(
    document: Mapping[str, Any], metrics: Iterable[str], device: str
) -> list[str]:
    return [
        f"{metric} read {item['value']} on {device}"
        for metric in metrics
        for item in samples(dict(document), metric, device=device)
        if isinstance(item.get("value"), (int, float)) and item["value"] < 0
    ]


def reset_tick_readings(
    document: Mapping[str, Any],
    needles: Iterable[str],
    metrics: Iterable[str],
    device: str,
) -> list[str]:
    """Readings of ``metrics`` on ``device`` shipped in a reset tick.

    A reset tick is a batch whose collection errors carry every needle (the
    reader, the device, ``reset``, ``delta unknown``); the product's rule for
    it is *no delta at all* for the counter that reset and for the aggregate it
    feeds (``collectors/host/network.py`` ``_monotonic_delta``), so any sample
    of those metrics on the device in that very batch is the old clamp -- a 0
    that clears the signal, or a delta measured against a stale baseline.
    """

    wanted = list(needles)
    found = []
    for batch in document.get("batches") or []:
        errors = [str(error) for error in batch.get("collection_errors") or []]
        if not any(all(needle in error for needle in wanted) for error in errors):
            continue
        for sample in batch.get("samples") or []:
            if sample.get("name") in set(metrics) and sample.get("device") == device:
                found.append(
                    f"{sample.get('name')}={sample.get('value')} shipped for {device} "
                    f"in the reset tick batch {batch.get('batch_id')} "
                    f"(observed {batch.get('observed_at')})"
                )
    return found


def unsupported_markers(
    document: Mapping[str, Any], interface: str, metric: str, reason: str
) -> list[dict[str, Any]]:
    """``efa_ethtool_counters_unsupported`` samples naming ``metric`` for ``reason``."""

    return [
        item
        for item in samples(dict(document), UNSUPPORTED_METRIC, device=interface)
        if (item.get("labels") or {}).get("reason") == reason
        and metric in str((item.get("labels") or {}).get("metrics", "")).split(",")
    ]


def validation_errors(
    document: Mapping[str, Any],
    incident: Mapping[str, Any],
    metric: str,
    *,
    operation: str = "VALIDATE_FABRIC",
) -> list[str]:
    """``operation`` ran for the incident; a failure names the held metric."""

    workflows = workflows_of(dict(document), str(incident.get("incident_id")))
    if not workflows:
        return [f"incident {incident.get('incident_id')} has no workflow"]
    execution = final_execution(workflows[0], operation)
    if execution is None:
        return [f"{operation} was not executed for {incident.get('incident_id')}"]
    status = execution.get("status")
    if status == "FAILED":
        details = execution.get("details") or {}
        named = (
            any(
                reason in {metric, f"host_diagnostic:{metric}"}
                or str(reason).startswith(f"{metric}:")
                for reasons in (details.get("node_failures") or {}).values()
                if isinstance(reasons, list)
                for reason in reasons
            )
            or details.get("host_diagnostic_failure") is True
        )
        return [] if named else [f"{operation} failed without naming {metric}"]
    if status != "SUCCEEDED":
        return [f"{operation} ended {status}"]
    return []


def served_errors(
    journal: Mapping[str, Any], interface: str, *, returncode: int | None = None
) -> list[str]:
    """The wrapper's journal shows the product asked ``-S interface`` and a rule fired."""

    entries = [
        item
        for item in journal.get("entries") or []
        if list(item.get("argv") or []) == ["-S", interface]
    ]
    if not entries:
        return [f"the product never ran ethtool -S {interface} through the wrapper"]
    applied = [item for item in entries if item.get("applied_rules")]
    if not applied:
        return [f"no plan rule was applied to ethtool -S {interface}"]
    if returncode is not None and not any(
        item.get("served_returncode") == returncode for item in applied
    ):
        return [f"ethtool -S {interface} was never served exit {returncode}"]
    return []


__all__ = [
    "CASE",
    "CLOCK_TOLERANCE_SECONDS",
    "COUNTER_PREFERENCE",
    "DETECTION_BUDGET",
    "ERRORS_METRIC",
    "ETHTOOL_GROUPS",
    "FAILED_STDERR",
    "INFINIBAND",
    "INVENTORY_METRIC",
    "LINK_DOWN_TEXT",
    "LINK_METRIC",
    "PROOF_SCOPE",
    "RAMP_STEP",
    "RDMA_COUNTERS",
    "RDMA_HW_COUNTERS",
    "REPLAY_COUNTERS",
    "STEP",
    "TRAFFIC_PAIRS",
    "UNSUPPORTED_METRIC",
    "VALIDATION_BUDGET",
    "EfaSourceFixture",
    "EfaTarget",
    "collection_errors",
    "counter_target",
    "efa_devices",
    "metric_values",
    "negative_readings",
    "netdev_target",
    "pair_target",
    "real_counter",
    "reset_tick_readings",
    "served_errors",
    "state_target",
    "unsupported_markers",
    "validation_errors",
    "zero_readings",
]
