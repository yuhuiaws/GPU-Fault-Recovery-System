"""GF-GENERAL-LIVE-010 EFA/RDMA counter variants over fakes, pinned to the product.

The runner's counter names, ethtool tokens and metric names are pinned against
the product's readers (``collectors/host/network.py`` run over a fake sysfs tree
and a stub ``ethtool``) and its rule table (``host_health.py``); the variants
are driven through a scripted source-replay probe and scripted product records.
Nothing here touches a node.
"""

from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors import HostTelemetryCollector
from gpu_fault.collectors.host.rdma_traffic import TRAFFIC_COUNTERS
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy, efa_inventory_rule
from gpu_fault.models import RecoveryAction, Severity
from scripts.e2e.regional import general_live_efa as efa
from scripts.e2e.regional import general_live_efa_fixture as fx
from scripts.e2e.regional import general_live_source_replay as replay
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_registry import build_registry, coverage_rows
from tests.collectors._support import NOW, RecordingSink, context
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in efa.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
DEV, PORT = "rdmap79s0", "1"
PORT_DEVICE = f"{DEV}/{PORT}"
IFACE = "ens5"
COUNTER_DEST = f"/sys/class/infiniband/{DEV}/ports/{PORT}/hw_counters/rx_drops"
STATE_DEST = f"/sys/class/infiniband/{DEV}/ports/{PORT}/state"
SURVEY = {
    "tools": {},
    "modules": {},
    "backing": [],
    "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    "collector_env": {
        "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
    },
}


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def inventory(
    *,
    efa: bool = True,
    owned_netdev: bool = True,
    physical: tuple[str, ...] = (IFACE,),
    ethtool: tuple[str, ...] = (),
    ethtool_path: str | None = "/usr/sbin/ethtool",
    counters: bool = True,
    pair: bool = True,
) -> dict[str, Any]:
    hw = {"tx_pkts": "3"}
    if counters:
        hw["rx_drops"] = "7"
    if pair:
        hw.update(rx_bytes="100", tx_bytes="200")

    def device(netdevs: list[str]) -> dict[str, Any]:
        return {
            "driver": "efa" if efa else "mlx5_core",
            "uevent_drivers": ["efa"] if efa else ["mlx5_core"],
            "efa": efa,
            "netdevs": netdevs,
            "ports": {
                PORT: {
                    "state": "4: ACTIVE",
                    "phys_state": "5: LinkUp",
                    "counters": {"port_rcv_errors": "0"} if counters else {},
                    "hw_counters": dict(hw),
                }
            },
        }

    return {
        "root": "/sys/class/infiniband",
        "devices": {
            DEV: device([IFACE] if owned_netdev else []),
            "rdmap80s0": device([]),
        },
        "physical_netdevs": list(physical),
        "ethtool": {
            "path": ethtool_path,
            "statistics": {
                IFACE: {
                    "returncode": 0,
                    "counters": list(ethtool),
                    "truncated": False,
                    "stderr": "",
                }
            },
        },
    }


def served(
    interface: str = IFACE, *, returncode: int = 0, applied: tuple[int, ...] = (0,)
) -> dict[str, Any]:
    return {
        "argv": ["-S", interface],
        "applied_rules": list(applied),
        "served_returncode": returncode,
        "real": "/usr/sbin/ethtool",
    }


def source_probe(
    inv: dict[str, Any],
    *,
    audit: list[dict[str, Any]] | None = None,
    read_back: str = "7",
    **extra: Any,
) -> FakeProbe:
    def read_source(arguments: tuple[str, ...]) -> dict[str, Any]:
        path = arguments[arguments.index("--path") + 1]
        view = {"readable": True, "value": read_back, "device": "21"}
        return {
            "path": path,
            "unit": view,
            "host": dict(view),
            "same": True,
            "window_open": False,
        }

    script: dict[str, Any] = {
        "inventory": inv,
        "install-source": {"bytes": 2},
        "install-wrapper": {"shim": "/run/gfrs-live/source-x/bin/ethtool"},
        "build-infiniband-fixture": {"devices": {DEV: {"netdevs": [IFACE]}}},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
            "verification": {"ok": True},
        },
        "set-value": lambda arguments: {
            "value": arguments[arguments.index("--value") + 1]
        },
        "audit-read": {"entries": [served()] if audit is None else audit, "calls": {}},
        "status": {"fixtures": {}},
        "close-window": {"clean": True},
        "read-source": read_source,
    }
    script.update(extra)
    return FakeProbe(script)


def future(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def sample(
    name: str, value: float, device: str | None, labels: dict[str, str] | None = None
) -> dict[str, Any]:
    return {"name": name, "device": device, "value": value, "labels": labels or {}}


def batch(
    samples: list[dict[str, Any]],
    *,
    errors: tuple[str, ...] = (),
    observed_at: str | None = None,
) -> dict[str, Any]:
    at = observed_at or future(60)
    return {
        "observed_at": at,
        "batch_id": f"host-{NODE}-{at}",
        "edge_filter_reasons": [],
        "collection_errors": list(errors),
        "samples": samples,
    }


def doc(**parts: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
        "metric_latest": [],
    }
    base.update(parts)
    return base


def diagnostic(
    metric: str,
    device: str,
    *,
    validation: str = "SUCCEEDED",
    details: dict[str, Any] | None = None,
    state: str = "RECOVERED",
    reasons: tuple[str, ...] = (),
    workflow_status: str = "SUCCEEDED",
    tag: str = "1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One diagnostic-only incident and its FREEZE_EVIDENCE + VALIDATE_FABRIC workflow."""

    inc = incident(
        f"inc-{tag}", metric, device=device, state=state, workflow=f"wf-{tag}"
    )
    inc["reasons"] = list(reasons)
    wf = workflow(
        f"wf-{tag}",
        f"inc-{tag}",
        ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"],
        status=workflow_status,
        executions=[
            {
                "step_index": 1,
                "operation": "VALIDATE_FABRIC",
                "status": validation,
                "details": details or {},
                "updated_at": future(5),
            }
        ],
    )
    return inc, wf


def run(
    variant: str,
    tmp_path: Path,
    *,
    inv: dict[str, Any] | None = None,
    records: Any = None,
    source: FakeProbe | None = None,
) -> tuple[dict[str, Any], FakeProbe]:
    spec = SPECS[variant]
    source = source or source_probe(inv if inv is not None else inventory())
    regional = FakeRegional(
        cpu_python=store_script({"activity": records if records is not None else doc()})
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={
            f"node@{NODE}": FakeProbe({"survey": SURVEY}),
            f"source-replay@{NODE}": source,
        },
    )
    result = spec.execute(ctx)
    ctx.cleanup.run()
    return result, source


def calls(probe: FakeProbe, command: str) -> list[tuple[str, ...]]:
    return [call for call in probe.calls if call[0] == command]


def installed(probe: FakeProbe, name: str) -> list[bytes]:
    return [
        base64.b64decode(call[call.index("--b64") + 1])
        for call in calls(probe, "install-source")
        if call[call.index("--name") + 1] == name
    ]


def plans(probe: FakeProbe) -> list[dict[str, Any]]:
    return [json.loads(item) for item in installed(probe, "plan-ethtool.json")]


def window_flags(probe: FakeProbe) -> dict[str, list[str]]:
    opened = calls(probe, "open-window")[0]
    flags: dict[str, list[str]] = {}
    for index, item in enumerate(opened):
        if item.startswith("--") and index + 1 < len(opened):
            flags.setdefault(item, []).append(opened[index + 1])
    return flags


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_covers_every_010_variant() -> None:
    build_registry(efa.VARIANTS, rows=coverage_rows(), categories=[10])
    for spec in efa.VARIANTS:
        assert spec.lanes == ("eks", "hp"), f"{spec.key} runs on both clusters"
        if spec.variant == "non_efa":
            assert spec.deferred.startswith("CAPABILITY_ABSENT"), spec.deferred
            continue
        assert spec.implemented and spec.proof_mode == "live-replay", spec.key
        assert spec.approvals == ("HOST_SOURCE_REPLAY",), (
            f"{spec.key} needs the host source replay token"
        )

        assert spec.flags == "S", spec.key
    # Batch 173 closed the sysfs/netdev counter-reset gap: counter_reset reads
    # the shared reset rule on the product's records and expects PASS like the
    # rest; no 010 variant names a product gap any more.
    assert all(spec.expected_outcome == "PASS" for spec in efa.VARIANTS), (
        "every 010 variant is expected to PASS"
    )
    assert all(spec.product_gap == "" for spec in efa.VARIANTS), (
        "no 010 variant carries a product gap"
    )
    assert "delta unknown" in SPECS["counter_reset"].expected


# --------------------------------------------------------------------------- #
# pins against the product
# --------------------------------------------------------------------------- #
def test_metric_rules_are_the_products() -> None:
    for metric in (fx.ERRORS_METRIC, fx.LINK_METRIC, *fx.ETHTOOL_GROUPS.values()):
        threshold, _category, severity, action, _reason = NodeHealthPolicy.METRIC_RULES[
            metric
        ]
        assert threshold == 1.0 and severity is Severity.WARNING, metric
        assert action is RecoveryAction.RUN_DIAGNOSTICS, f"{metric} is diagnostic only"
        assert metric in NodeHealthPolicy.LATEST_METRICS_REQUIRED, (
            f"VALIDATE_FABRIC reads the latest {metric}"
        )
    assert fx.UNSUPPORTED_METRIC not in NodeHealthPolicy.METRIC_RULES, (
        "a NOT_SUPPORTED marker never mints a finding"
    )
    assert fx.INVENTORY_METRIC in NodeHealthPolicy.METRIC_RULES
    link = efa_inventory_rule(
        HostMetricSample(
            name=fx.INVENTORY_METRIC, value=1, labels={"failure_mode": "LINK_INACTIVE"}
        )
    )
    assert link[3] is RecoveryAction.RUN_DIAGNOSTICS, (
        "LINK_INACTIVE diagnoses, no reboot"
    )
    assert {name for pair in fx.TRAFFIC_PAIRS for name in pair} <= set(TRAFFIC_COUNTERS)


def _efa_tree(base: Path, *, netdev: str | None = None) -> tuple[Path, Path]:
    """``(infiniband root, port dir)`` of one EFA device laid out like sysfs."""

    root = base / "sys" / "class" / "infiniband"
    root.mkdir(parents=True)
    pci = base / "sys" / "devices" / "pci0000:53" / "0000:53:00.0"
    device = pci / "infiniband" / DEV
    port = device / "ports" / PORT
    (port / "counters").mkdir(parents=True)
    (port / "hw_counters").mkdir(parents=True)
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    (pci / "uevent").write_text("DRIVER=efa\n")
    efa_driver = base / "sys" / "bus" / "pci" / "drivers" / "efa"
    efa_driver.mkdir(parents=True)
    (pci / "driver").symlink_to(efa_driver)
    if netdev:
        (pci / "net" / netdev).mkdir(parents=True)
    (device / "device").symlink_to(pci)
    (root / DEV).symlink_to(device)
    return root, port


def test_sysfs_counter_names_are_the_ones_the_product_sums(tmp_path: Path) -> None:
    root, port = _efa_tree(tmp_path)
    names = [("counters", name) for name in fx.RDMA_COUNTERS] + [
        ("hw_counters", name) for name in fx.RDMA_HW_COUNTERS
    ]
    for group, name in names:
        (port / group / name).write_text("10\n")
    (port / "hw_counters" / "gfrs_bogus").write_text("10\n")
    for name in ("rx_bytes", "tx_bytes"):  # the traffic pair _rdma also needs
        (port / "hw_counters" / name).write_text("500\n")
    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", infiniband_root=str(root)
    )
    collector.collect_rdma_samples(NOW)
    for group, name in names:
        (port / group / name).write_text("11\n")
    (port / "hw_counters" / "gfrs_bogus").write_text("99\n")
    samples = collector.collect_rdma_samples(NOW + timedelta(seconds=15))
    by_name = {item.name: item for item in samples}
    for _group, name in names:
        delta = by_name.get(f"rdma_{name}_delta")
        assert delta is not None and delta.value == 1 and delta.device == PORT_DEVICE, (
            f"the product reads {name} into rdma_{name}_delta"
        )
    assert "rdma_gfrs_bogus_delta" not in by_name, "unknown files are not summed"
    assert by_name[fx.ERRORS_METRIC].value == len(names), (
        "rdma_errors_delta is the sum over exactly these counters"
    )
    assert by_name[fx.LINK_METRIC].value == 0
    assert [group for group, name in fx.COUNTER_PREFERENCE[:1]] == ["hw_counters"]
    assert all(
        (
            group == "counters"
            and name in fx.RDMA_COUNTERS
            or (group == "hw_counters" and name in fx.RDMA_HW_COUNTERS)
            for group, name in fx.COUNTER_PREFERENCE
        )
    ), "the preference list names only files the product sums"


def _ethtool_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listings: list[str]
) -> tuple[HostTelemetryCollector, RecordingSink]:
    root, _port = _efa_tree(tmp_path, netdev=IFACE)
    outputs = iter(listings)

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv == ["ethtool", "-S", IFACE], argv
        return subprocess.CompletedProcess(argv, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(
        "gpu_fault.collectors.host.network.shutil.which",
        lambda command: "/usr/sbin/ethtool" if command == "ethtool" else None,
    )
    times = iter(NOW + timedelta(seconds=15 * index) for index in range(10))
    sink = RecordingSink()
    collector = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        infiniband_root=str(root),
        runner=runner,
        now=lambda: next(times),
        health_summary_seconds=300,
    )
    for name in HostTelemetryCollector.CONTRIBUTORS:
        if name != "_efa_network":
            monkeypatch.setattr(collector, name, lambda _observed_at: [])
    return collector, sink


def test_ethtool_tokens_and_replay_counters_match_the_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def listing(value: int) -> str:
        return "NIC statistics:\n" + "".join(
            f"     {name}: {value}\n" for name in fx.REPLAY_COUNTERS.values()
        )

    collector, sink = _ethtool_collector(
        tmp_path, monkeypatch, [listing(10), listing(15)]
    )
    collector.collect_once()
    collector.collect_once()
    _path, payload = sink.requests[-1]
    deltas = {
        (item["name"], item["device"]): item["value"]
        for item in payload["samples"]
        if item["name"] in fx.ETHTOOL_GROUPS.values()
    }
    assert deltas == {(metric, IFACE): 5.0 for metric in fx.ETHTOOL_GROUPS.values()}, (
        "each test-origin counter feeds exactly its own group"
    )
    named = {
        metric
        for item in payload["samples"]
        if item["name"] == fx.UNSUPPORTED_METRIC
        for metric in item["labels"]["metrics"].split(",")
    }
    assert named.isdisjoint(fx.ETHTOOL_GROUPS.values()), (
        "every error group has a counter: only the congestion groups are NOT_SUPPORTED"
    )
    assert payload["collection_errors"] == []


def test_missing_group_is_the_products_not_supported_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = "     rnr_nak_err: 3\n     gfrs_retry_err: 4\n"
    collector, sink = _ethtool_collector(tmp_path, monkeypatch, [listing])
    collector.collect_once()
    _path, payload = sink.requests[-1]
    markers = [
        item for item in payload["samples"] if item["name"] == fx.UNSUPPORTED_METRIC
    ]
    assert len(markers) == 1 and markers[0]["device"] == IFACE, payload["samples"]
    assert markers[0]["labels"]["reason"] == "MISSING_COUNTER"
    named = set(markers[0]["labels"]["metrics"].split(","))
    assert "efa_cq_errors_delta" in named, "the missing group is named"
    assert named.isdisjoint({"efa_rnr_errors_delta", "efa_retry_errors_delta"}), (
        "the served groups are not missing"
    )
    assert fx.unsupported_markers(
        doc(batches=[batch(markers)]), IFACE, "efa_cq_errors_delta", "MISSING_COUNTER"
    ), "the runner's matcher reads the product's marker"
    assert fx.real_counter("rnr", ["rnr_nak_err", "gfrs_retry_err"]) == "rnr_nak_err"


def test_a_withdrawn_counter_is_delivered_by_the_products_edge_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``missing_counter`` reads the marker from delivered batches: the tick on
    which a served counter disappears must be an edge (live 2026-10-25: it was
    not, the marker named the counter on every tick and never left the node)."""

    served = "NIC statistics:\n     gfrs_rnr_err: 0\n     queue_0_tx_cnt: 7\n"
    withdrawn = "NIC statistics:\n     queue_0_tx_cnt: 7\n"
    collector, sink = _ethtool_collector(
        tmp_path, monkeypatch, [served, served, withdrawn, withdrawn]
    )
    for _ in range(4):
        collector.collect_once()

    reasons = [payload["edge_filter_reasons"] for _, payload in sink.requests]
    assert len(reasons) == 2 and reasons[0] == ["baseline"], reasons
    assert any(reason.startswith("not-supported:") for reason in reasons[1]), (
        f"the withdrawal tick is not an edge: {reasons}"
    )
    delivered = doc(
        batches=[
            {
                **batch(payload["samples"]),
                "edge_filter_reasons": payload["edge_filter_reasons"],
            }
            for _, payload in sink.requests
        ]
    )
    assert fx.unsupported_markers(
        delivered, IFACE, "efa_rnr_errors_delta", "MISSING_COUNTER"
    ), (
        "the runner's matcher reads the withdrawn counter's marker from the delivered batch"
    )
    assert fx.metric_values(delivered, "efa_rnr_errors_delta", IFACE) == [], (
        "the withdrawn counter never ships a zero delta"
    )
    readings = fx.device_readings(delivered, IFACE, ["efa_rnr_errors_delta"])
    named = [set(reading["markers"][0]["metrics"].split(",")) for reading in readings]
    assert named[1] - named[0] == {"efa_rnr_errors_delta"} and named[0] < named[1], (
        f"the withdrawn counter is the only change between the markers: {readings}"
    )


# --------------------------------------------------------------------------- #
# target selection
# --------------------------------------------------------------------------- #
def test_counter_target_prefers_rx_drops_then_falls_back() -> None:
    target = fx.counter_target(inventory())
    assert (target.device, target.port, target.group, target.counter) == (
        DEV,
        PORT,
        "hw_counters",
        "rx_drops",
    )
    assert target.counter_dest == COUNTER_DEST and target.real_value == "7"
    assert (
        target.port_device == PORT_DEVICE
        and target.counter_metric == "rdma_rx_drops_delta"
    )
    inv = inventory()
    del inv["devices"][DEV]["ports"][PORT]["hw_counters"]["rx_drops"]
    fallback = fx.counter_target(inv)
    assert (fallback.group, fallback.counter) == ("counters", "port_rcv_errors")
    with pytest.raises(VariantNotRun, match="CAPABILITY_ABSENT.*no EFA RDMA device"):
        fx.counter_target(inventory(efa=False))
    with pytest.raises(VariantNotRun, match="no EFA port exposes a numeric counter"):
        fx.counter_target(inventory(counters=False))


def test_state_and_pair_targets() -> None:
    state = fx.state_target(inventory())
    assert state.dest("state") == STATE_DEST and state.real_value == "4: ACTIVE"
    pair = fx.pair_target(inventory())
    assert (
        pair.counter_dest.endswith("/hw_counters/tx_bytes") and pair.real_value == "200"
    )
    with pytest.raises(VariantNotRun, match="complete rx/tx traffic pair"):
        fx.pair_target(inventory(pair=False))


def test_netdev_target_uses_a_real_netdev_or_attaches_a_physical_one() -> None:
    real = fx.netdev_target(inventory(ethtool=("queue_0_tx_cnt",)))
    assert (real.device, real.netdev, real.netdev_source) == (DEV, IFACE, "real")
    assert real.ethtool_names == ("queue_0_tx_cnt",)
    attached = fx.netdev_target(inventory(owned_netdev=False))
    assert (attached.netdev, attached.netdev_source) == (IFACE, "fixture")
    with pytest.raises(VariantNotRun, match="ethtool is not installed"):
        fx.netdev_target(inventory(ethtool_path=None))
    with pytest.raises(VariantNotRun, match="no physical netdev to attach"):
        fx.netdev_target(inventory(owned_netdev=False, physical=()))


def test_real_counter_takes_a_single_token_name_only() -> None:
    names = ["queue_0_tx_cnt", "rnr_retry_err", "rnr_nak_err", "cq_err_cnt"]
    assert fx.real_counter("rnr", names) == "rnr_nak_err", (
        "rnr_retry_err carries two tokens"
    )
    assert fx.real_counter("retry", names) is None
    assert fx.real_counter("cq_err", names) == "cq_err_cnt"
    for token, name in fx.REPLAY_COUNTERS.items():
        assert fx.real_counter(token, [name]) == name, (
            "each test-origin name is unambiguous"
        )


# --------------------------------------------------------------------------- #
# verdict pieces
# --------------------------------------------------------------------------- #
def test_validation_errors_read_the_fabric_step() -> None:
    inc, wf = diagnostic(fx.ERRORS_METRIC, PORT_DEVICE)
    assert (
        fx.validation_errors(
            doc(incidents=[inc], workflows=[wf]), inc, fx.ERRORS_METRIC
        )
        == []
    )
    inc, wf = diagnostic(
        fx.LINK_METRIC,
        PORT_DEVICE,
        validation="FAILED",
        details={"node_failures": {NODE: ["rdma_link_down"]}},
    )
    assert (
        fx.validation_errors(doc(incidents=[inc], workflows=[wf]), inc, fx.LINK_METRIC)
        == []
    )
    inc, wf = diagnostic(
        fx.ERRORS_METRIC,
        PORT_DEVICE,
        validation="FAILED",
        details={"node_failures": {NODE: [f"rdma_errors_delta:{PORT_DEVICE}"]}},
    )
    assert (
        fx.validation_errors(
            doc(incidents=[inc], workflows=[wf]), inc, fx.ERRORS_METRIC
        )
        == []
    )
    inc, wf = diagnostic(
        fx.ERRORS_METRIC,
        PORT_DEVICE,
        validation="FAILED",
        details={"node_failures": {NODE: ["gpu"]}},
    )
    assert fx.validation_errors(
        doc(incidents=[inc], workflows=[wf]), inc, fx.ERRORS_METRIC
    ) == ["VALIDATE_FABRIC failed without naming rdma_errors_delta"]
    assert fx.validation_errors(doc(incidents=[inc]), inc, fx.ERRORS_METRIC) == [
        "incident inc-1 has no workflow"
    ]
    inc, wf = diagnostic(fx.ERRORS_METRIC, PORT_DEVICE, validation="RUNNING")
    wf["step_executions"] = []
    assert fx.validation_errors(
        doc(incidents=[inc], workflows=[wf]), inc, fx.ERRORS_METRIC
    ) == ["VALIDATE_FABRIC was not executed for inc-1"]


def test_served_errors_read_the_wrapper_journal() -> None:
    assert fx.served_errors({"entries": [served()]}, IFACE) == []
    assert fx.served_errors({"entries": []}, IFACE) == [
        f"the product never ran ethtool -S {IFACE} through the wrapper"
    ]
    assert fx.served_errors({"entries": [served(applied=())]}, IFACE) == [
        f"no plan rule was applied to ethtool -S {IFACE}"
    ]
    assert fx.served_errors(
        {"entries": [served(returncode=0)]}, IFACE, returncode=1
    ) == [f"ethtool -S {IFACE} was never served exit 1"]


def test_readings_and_collection_error_matchers() -> None:
    document = doc(
        batches=[
            batch(
                [
                    sample(fx.ERRORS_METRIC, 0, PORT_DEVICE),
                    sample(fx.ERRORS_METRIC, -3, PORT_DEVICE),
                    sample(fx.ERRORS_METRIC, 0, "other/1"),
                ],
                errors=(
                    f"_rdma: HostObservationError: {PORT_DEVICE}: rx_drops reset (7 -> 1); delta unknown",
                ),
                observed_at=future(30),
            ),
            batch(
                [sample(fx.ERRORS_METRIC, 0, PORT_DEVICE)],
                observed_at="2026-10-24T00:00:00+00:00",
            ),
        ]
    )
    now = datetime.now(timezone.utc)
    assert len(fx.zero_readings(document, fx.ERRORS_METRIC, PORT_DEVICE)) == 2
    assert (
        len(fx.zero_readings(document, fx.ERRORS_METRIC, PORT_DEVICE, after=now)) == 1
    ), "only the reading after the moment counts"
    assert fx.negative_readings(document, (fx.ERRORS_METRIC,), PORT_DEVICE) == [
        f"{fx.ERRORS_METRIC} read -3 on {PORT_DEVICE}"
    ]
    assert len(fx.collection_errors(document, PORT_DEVICE, "reset")) == 1
    assert fx.collection_errors(document, "ethtool") == []
    assert fx.metric_values(document, fx.ERRORS_METRIC, PORT_DEVICE) == [0.0, 0.0, -3.0]


def test_device_readings_record_what_each_batch_carried() -> None:
    marker = {"reason": "MISSING_COUNTER", "metrics": "efa_rnr_errors_delta"}
    first = batch(
        [
            sample("efa_rnr_errors_delta", 0, IFACE),
            sample("efa_rnr_errors_delta", 4, "eth9"),
            sample(fx.UNSUPPORTED_METRIC, 1, "eth9", marker),
        ],
        errors=(f"_efa_network: HostObservationError: {IFACE}: boom", "other: x"),
        observed_at=future(10),
    )
    first["edge_filter_reasons"] = ["baseline"]
    second = batch(
        [sample(fx.UNSUPPORTED_METRIC, 1, IFACE, marker)], observed_at=future(20)
    )
    readings = fx.device_readings(
        doc(batches=[first, second]), IFACE, ["efa_rnr_errors_delta"]
    )
    assert [reading["deltas"] for reading in readings] == [
        {"efa_rnr_errors_delta": 0},
        {},
    ], "only the netdev's own deltas, other netdevs are not its evidence"
    assert [reading["markers"] for reading in readings] == [[], [marker]], readings
    assert readings[0]["collection_errors"] == [
        f"_efa_network: HostObservationError: {IFACE}: boom"
    ]
    assert readings[0]["edge_filter_reasons"] == ["baseline"]
    assert readings[0]["batch_id"] == first["batch_id"]
    assert fx.readings_summary(readings) == (
        "2 batch(es); markers: MISSING_COUNTER[efa_rnr_errors_delta]; "
        "deltas: efa_rnr_errors_delta=0"
    )
    assert fx.readings_summary([]) == "0 batch(es); markers: none; deltas: none"


# --------------------------------------------------------------------------- #
# sysfs variants
# --------------------------------------------------------------------------- #
def _rdma_error_doc(*, grown: bool = True) -> dict[str, Any]:
    inc, wf = diagnostic(fx.ERRORS_METRIC, PORT_DEVICE)
    value = 5000 if grown else 1
    return doc(
        incidents=[inc],
        workflows=[wf],
        batches=[
            batch(
                [
                    sample(fx.ERRORS_METRIC, value, PORT_DEVICE),
                    sample("rdma_rx_drops_delta", value, PORT_DEVICE),
                ]
            ),
            batch([sample(fx.ERRORS_METRIC, 0, PORT_DEVICE)], observed_at=future(90)),
        ],
    )


def test_rdma_error_steps_a_bound_real_counter(tmp_path: Path) -> None:
    result, source = run("rdma_error", tmp_path, records=_rdma_error_doc())
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay"
    assert result["target"]["counter"] == "rx_drops" and result["fixture_value"] == 5007
    assert installed(source, "efa-counter") == [b"7\n"], (
        "the bind starts at the real value"
    )
    assert window_flags(source)["--bind"] == [f"efa-counter:{COUNTER_DEST}"]
    assert window_flags(source)["--mode"] == ["bind"]
    stepped = calls(source, "set-value")
    assert len(stepped) == 1 and stepped[0][stepped[0].index("--value") + 1] == "5007"
    order = [call[0] for call in source.calls]
    assert order.index("close-window") < order.index("read-source"), (
        "the real value is read back after the window closed"
    )
    assert calls(source, "read-source")[0][2] == COUNTER_DEST


def test_rdma_error_fails_without_a_finding_or_without_growth(tmp_path: Path) -> None:
    result, _source = run("rdma_error", tmp_path, records=doc())
    assert result["verdict"] == "FAIL" and any("chain:" in e for e in result["errors"])
    result, _source = run("rdma_error", tmp_path, records=_rdma_error_doc(grown=False))
    assert result["verdict"] == "FAIL"
    assert any("counter_grew" in error for error in result["errors"]), result["errors"]


def test_rdma_error_fails_when_the_unit_still_reads_the_fixture(tmp_path: Path) -> None:
    source = source_probe(inventory(), read_back="5007")
    result, _source = run(
        "rdma_error", tmp_path, records=_rdma_error_doc(), source=source
    )
    assert result["verdict"] == "FAIL"
    assert any(
        (
            "real_restored" in error and "fixture value" in error
            for error in result["errors"]
        )
    ), "the read-back flags a unit that still reads the fixture"


def test_variants_are_not_run_without_an_efa_device(tmp_path: Path) -> None:
    for variant in (
        "rdma_error",
        "link_inactive",
        "rnr",
        "ethtool_failed",
        "counter_reset",
    ):
        with pytest.raises(VariantNotRun, match="CAPABILITY_ABSENT"):
            run(variant, tmp_path, inv=inventory(efa=False))


def _link_doc(
    *,
    validation: str = "FAILED",
    failure_mode: str = "LINK_INACTIVE",
    judged_after_restore: bool = False,
    late_inactive: bool = False,
    stale_gate: bool = False,
) -> dict[str, Any]:
    """The held tick (before the restore), the link chain judged on it, the
    LINK_INACTIVE co-finding concluded on the restored port, fresh zeros."""

    inc, wf = diagnostic(
        fx.LINK_METRIC,
        PORT_DEVICE,
        validation=validation,
        details={"node_failures": {NODE: ["rdma_link_down"]}},
        reasons=(
            "RDMA/EFA/InfiniBand port is not active",
            "diagnostic inconclusive: validation failed",
        ),
        workflow_status="FAILED" if validation == "FAILED" else "SUCCEEDED",
    )
    if judged_after_restore:
        wf["step_executions"][0]["updated_at"] = future(EXPOSURE + 20)
    inv_inc = incident(
        "inc-inv", fx.INVENTORY_METRIC, state="RECOVERED", workflow="wf-inv"
    )
    inv_wf = workflow(
        "wf-inv",
        "inc-inv",
        ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"],
        status="SUCCEEDED",
        executions=[
            {
                "operation": "VALIDATE_FABRIC",
                "status": "SUCCEEDED",
                "details": {
                    "node_pending": {
                        NODE: {"missing_inventory_action_completion": ["x"]}
                    }
                }
                if stale_gate
                else {"inventory_evidence": {NODE: {}}},
            }
        ],
    )
    batches = [
        batch(
            [
                sample(fx.LINK_METRIC, 1, PORT_DEVICE),
                sample(fx.INVENTORY_METRIC, 1, None, {"failure_mode": failure_mode}),
                sample(fx.MISSING_METRIC, 1, None, {"failure_mode": failure_mode}),
            ],
            observed_at=future(10),
        ),
        batch([sample(fx.LINK_METRIC, 0, PORT_DEVICE)], observed_at=future(120)),
        batch([sample(fx.LINK_METRIC, 0, PORT_DEVICE)], observed_at=future(150)),
    ]
    if late_inactive:
        batches.append(
            batch(
                [sample(fx.MISSING_METRIC, 1, None)], observed_at=future(EXPOSURE + 30)
            )
        )
    return doc(incidents=[inc, inv_inc], workflows=[wf, inv_wf], batches=batches)


EXPOSURE = replay.bounded_exposure_seconds(15)


def test_link_inactive_binds_the_state_file_for_a_bounded_exposure(
    tmp_path: Path,
) -> None:
    source = source_probe(
        inventory(),
        read_back="4: ACTIVE",
        **{"close-window": {"clean": True, "already_closed": True}},
    )
    result, source = run("link_inactive", tmp_path, records=_link_doc(), source=source)
    assert result["verdict"] == "PASS", result["errors"]
    assert installed(source, "efa-state") == [b"1: DOWN\n"]
    flags = window_flags(source)
    assert flags["--bind"] == [f"efa-state:{STATE_DEST}"]
    assert flags["--restore-seconds"] == [str(EXPOSURE)] and EXPOSURE == 26, (
        "the window's deadman is the restore: after the second tick, before a third"
    )
    assert result["closed_by"] == "deadman" and result["exposure_seconds"] == EXPOSURE
    assert result["stages"]["recovery"] == [] and result["stages"]["settled"] == []
    assert result["stages"]["co_finding_diagnosed"] == [], result["stages"]
    spec = SPECS["link_inactive"]
    assert (
        spec.approvals == ("HOST_SOURCE_REPLAY",)
        and "bounded exposure" in spec.injection
    )


def test_link_inactive_fails_on_the_held_port_judged_clean_or_a_wrong_mode(
    tmp_path: Path,
) -> None:
    result, source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(validation="SUCCEEDED"),
        source=source_probe(inventory(), read_back="4: ACTIVE"),
    )
    assert result["verdict"] == "FAIL"
    assert result["closed_by"] == "runner", "a late deadman: the runner closed"
    assert any(
        "succeeded while the fault was still held" in e for e in result["errors"]
    ), result["errors"]
    result, _source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(failure_mode="PCI_DEVICE_MISSING"),
        source=source_probe(inventory(), read_back="4: ACTIVE"),
    )
    assert result["verdict"] == "FAIL"
    assert any("inventory_grading" in e for e in result["errors"]), result["errors"]
    result, _source = run(
        "link_inactive",
        tmp_path,
        records=_link_doc(stale_gate=True),
        source=source_probe(inventory(), read_back="4: ACTIVE"),
    )
    assert result["verdict"] == "FAIL"
    assert any("missing_inventory_action_completion" in e for e in result["errors"]), (
        result["errors"]
    )


def test_link_inactive_is_inconclusive_when_the_bound_decided_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="after the bounded exposure ended"):
        run(
            "link_inactive",
            tmp_path,
            records=_link_doc(validation="SUCCEEDED", judged_after_restore=True),
            source=source_probe(inventory(), read_back="4: ACTIVE"),
        )
    with pytest.raises(VariantInconclusive, match="exposure bound did not hold"):
        run(
            "link_inactive",
            tmp_path,
            records=_link_doc(late_inactive=True),
            source=source_probe(inventory(), read_back="4: ACTIVE"),
        )
    with pytest.raises(VariantInconclusive, match="no sample"):
        run(
            "link_inactive",
            tmp_path,
            records=doc(),
            source=source_probe(inventory(), read_back="4: ACTIVE"),
        )


# --------------------------------------------------------------------------- #
# ethtool variants
# --------------------------------------------------------------------------- #
def _ethtool_doc(metric: str, value: float = 5000) -> dict[str, Any]:
    inc, wf = diagnostic(metric, IFACE)
    return doc(
        incidents=[inc], workflows=[wf], batches=[batch([sample(metric, value, IFACE)])]
    )


@pytest.mark.parametrize(
    ("variant", "token"), [("rnr", "rnr"), ("retry", "retry"), ("cq_error", "cq_err")]
)
def test_ethtool_deltas_use_the_reviewed_wrapper(
    tmp_path: Path, variant: str, token: str
) -> None:
    metric = fx.ETHTOOL_GROUPS[token]
    result, source = run(variant, tmp_path, records=_ethtool_doc(metric))
    assert result["verdict"] == "PASS", result["errors"]
    assert result["counter"] == fx.REPLAY_COUNTERS[token] and result["metric"] == metric
    installed_plans = plans(source)
    assert [plan["rules"][0] for plan in installed_plans] == [
        {
            "query": "stats",
            "interface": IFACE,
            "action": "set_counter",
            "counter": fx.REPLAY_COUNTERS[token],
            "value": "0",
        },
        {
            "query": "stats",
            "interface": IFACE,
            "action": "set_counter",
            "counter": fx.REPLAY_COUNTERS[token],
            "value": "5000",
        },
    ], "baseline first, the step as a re-plan"
    assert window_flags(source)["--wrapper"] == ["ethtool"]
    assert window_flags(source)["--mode"] == ["command"]
    assert calls(source, "audit-read")[0][2] == "ethtool"
    if token != "cq_err":
        assert any(("congestion" in note for note in result["notes"])), (
            "RNR/retry notes name congestion"
        )


def test_ethtool_delta_prefers_a_real_counter_and_attaches_a_netdev_when_none_is_owned(
    tmp_path: Path,
) -> None:
    inv = inventory(owned_netdev=False, ethtool=("queue_0_tx_cnt", "rnr_nak_err"))
    result, source = run(
        "rnr", tmp_path, inv=inv, records=_ethtool_doc("efa_rnr_errors_delta")
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert (
        result["counter"] == "rnr_nak_err"
        and result["target"]["netdev_source"] == "fixture"
    )
    built = calls(source, "build-infiniband-fixture")[0]
    assert built[built.index("--netdev") + 1] == f"{DEV}:{IFACE}"
    flags = window_flags(source)
    assert flags["--tree"] == ["infiniband:/sys/class/infiniband"] and flags[
        "--wrapper"
    ] == ["ethtool"]
    assert flags["--mode"] == ["mixed"]


def test_ethtool_delta_fails_on_a_hardware_step_or_an_unserved_wrapper(
    tmp_path: Path,
) -> None:
    document = _ethtool_doc("efa_rnr_errors_delta")
    document["workflows"][0]["official_steps"].append(
        {"operation": "RESTART_NODE", "parameters": {}}
    )
    result, _source = run("rnr", tmp_path, records=document)
    assert result["verdict"] == "FAIL" and any(
        "RESTART_NODE" in e for e in result["errors"]
    )
    result, _source = run(
        "rnr",
        tmp_path,
        records=_ethtool_doc("efa_rnr_errors_delta"),
        source=source_probe(inventory(), audit=[]),
    )
    assert result["verdict"] == "FAIL" and any("served" in e for e in result["errors"])


def _failed_doc(*, zero: bool = False, no_statistics: bool = False) -> dict[str, Any]:
    samples = []
    if zero:
        samples.append(sample("efa_rnr_errors_delta", 0, IFACE))
    if no_statistics:
        samples.append(
            sample(
                fx.UNSUPPORTED_METRIC,
                3,
                IFACE,
                {"reason": "NO_STATISTICS", "metrics": ""},
            )
        )
    return doc(
        batches=[
            batch(
                samples,
                errors=(
                    f"_efa_network: HostObservationError: {IFACE}: ethtool -S exited 1: "
                    f"{fx.FAILED_STDERR}",
                ),
            )
        ]
    )


def test_ethtool_failed_is_unknown_never_zero(tmp_path: Path) -> None:
    source = source_probe(inventory(), audit=[served(returncode=1)])
    result, source = run(
        "ethtool_failed", tmp_path, records=_failed_doc(), source=source
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert plans(source)[0]["rules"] == [
        {
            "query": "stats",
            "interface": IFACE,
            "action": "exit",
            "exit_code": 1,
            "stderr": fx.FAILED_STDERR,
        }
    ]
    for records, stage in (
        (_failed_doc(zero=True), "no_zero_reading"),
        (_failed_doc(no_statistics=True), "not_no_statistics"),
        (doc(), "unknown_recorded"),
    ):
        result, _source = run(
            "ethtool_failed",
            tmp_path,
            records=records,
            source=source_probe(inventory(), audit=[served(returncode=1)]),
        )
        assert result["verdict"] == "FAIL" and any(
            stage in e for e in result["errors"]
        ), (stage, result["errors"])


def _missing_docs(*, marker: bool = True) -> list[dict[str, Any]]:
    present = doc(batches=[batch([sample("efa_rnr_errors_delta", 0, IFACE)])])
    gone = doc(
        batches=[
            batch(
                [
                    sample(
                        fx.UNSUPPORTED_METRIC,
                        2,
                        IFACE,
                        {
                            "reason": "MISSING_COUNTER",
                            "metrics": "efa_retry_errors_delta,efa_rnr_errors_delta",
                        },
                    )
                ]
                if marker
                else []
            )
        ]
    )
    return [present, gone]


def test_missing_counter_withdraws_a_real_counter_and_reads_not_supported(
    tmp_path: Path,
) -> None:
    inv = inventory(ethtool=("rnr_nak_err", "queue_0_tx_cnt"))
    result, source = run("missing_counter", tmp_path, inv=inv, records=_missing_docs())
    assert result["verdict"] == "PASS", result["errors"]
    assert (
        result["counter"] == "rnr_nak_err"
        and result["metric"] == "efa_rnr_errors_delta"
    )
    assert [plan["rules"][0]["action"] for plan in plans(source)] == [
        "set_counter",
        "delete_counter",
    ]
    case_dir = tmp_path / "cases" / SPECS["missing_counter"].key
    present = json.loads((case_dir / "counter-present-readings.json").read_text())
    missing = json.loads((case_dir / "counter-missing-readings.json").read_text())
    assert [reading["deltas"] for reading in present] == [{"efa_rnr_errors_delta": 0}]
    assert [reading["markers"] for reading in present] == [[]]
    assert [reading["markers"] for reading in missing] == [
        [
            {
                "reason": "MISSING_COUNTER",
                "metrics": "efa_retry_errors_delta,efa_rnr_errors_delta",
            }
        ]
    ], "the evidence names what each batch carried for the netdev"
    result, source = run("missing_counter", tmp_path, records=_missing_docs())
    assert result["verdict"] == "PASS", result["errors"]
    assert [plan["rules"][0]["action"] for plan in plans(source)] == [
        "set_counter",
        "passthrough",
    ], "a test-origin counter is withdrawn by passing the real listing through"
    result, _source = run(
        "missing_counter", tmp_path, records=_missing_docs(marker=False)
    )
    assert result["verdict"] == "FAIL" and result["errors"] == [
        "not_supported_marker: no efa_ethtool_counters_unsupported MISSING_COUNTER "
        f"marker names efa_rnr_errors_delta on {IFACE} since the withdrawal "
        "(1 batch(es); markers: none; deltas: none)"
    ], "a failure names what the batches since the withdrawal carried"


# --------------------------------------------------------------------------- #
# counter reset
# --------------------------------------------------------------------------- #
SYSFS_RESET_ERROR = (
    f"_rdma: HostObservationError: {PORT_DEVICE}: rx_drops reset (5007 -> 1); "
    "delta unknown"
)
ETHTOOL_RESET_ERROR = (
    f"_efa_network: HostObservationError: {IFACE}: gfrs_rnr_err reset (5000 -> 1); "
    "delta unknown"
)


def _reset_doc(
    *,
    fixed: bool,
    with_ethtool: bool = True,
    reset_tick_reading: bool = False,
    gapped: bool = False,
    reminted: bool = False,
) -> dict[str, Any]:
    """The product's records of a ramp -> reset -> ramp on both readers.

    ``fixed``: the reset tick is a batch carrying only the two readers'
    collection errors (batch 173); the ramp batches before and after carry
    the deltas. Otherwise the old ``_rdma`` clamp: no sysfs error, a
    ``rdma_errors_delta=0`` reading at the reset and the fault minted again.
    ``reset_tick_reading`` puts a port delta into the fixed reset tick batch --
    a delta measured against a stale baseline, which the rule forbids.
    ``gapped``: the ramp after the reset was slower than the tick (live
    2026-10-25): a 0 reading between two steps on both readers, each step a
    new episode and its own incident -- the product's contract, a PASS.
    ``reminted``: a second incident per reader although the readings show one
    unbroken episode -- a fault minted without a reading, the FAIL.
    """

    inc, wf = diagnostic(fx.ERRORS_METRIC, PORT_DEVICE, tag="r")
    incidents, workflows = [inc], [wf]
    if with_ethtool:
        einc, ewf = diagnostic("efa_rnr_errors_delta", IFACE, tag="e")
        incidents.append(einc)
        workflows.append(ewf)
    ramp = [
        sample(fx.ERRORS_METRIC, 1000, PORT_DEVICE),
        sample("rdma_rx_drops_delta", 1000, PORT_DEVICE),
        sample("efa_rnr_errors_delta", 1000, IFACE),
    ]
    zeros = [
        sample(fx.ERRORS_METRIC, 0, PORT_DEVICE),
        sample("efa_rnr_errors_delta", 0, IFACE),
    ]
    batches = [batch(list(ramp), observed_at=future(30))]
    if fixed:
        errors = [SYSFS_RESET_ERROR] + ([ETHTOOL_RESET_ERROR] if with_ethtool else [])
        reset_tick = (
            [sample("rdma_rx_drops_delta", 5, PORT_DEVICE)]
            if reset_tick_reading
            else []
        )
        batches.append(batch(reset_tick, errors=tuple(errors), observed_at=future(60)))
    else:
        second, second_wf = diagnostic(fx.ERRORS_METRIC, PORT_DEVICE, tag="r2")
        incidents.append(second)
        workflows.append(second_wf)
        batches.append(
            batch(
                [sample(fx.ERRORS_METRIC, 0, PORT_DEVICE)],
                errors=(ETHTOOL_RESET_ERROR,) if with_ethtool else (),
                observed_at=future(60),
            )
        )
    batches.append(batch(list(ramp), observed_at=future(90)))
    if gapped or reminted:
        for reader, metric, device in (
            ("r2", fx.ERRORS_METRIC, PORT_DEVICE),
            ("e2", "efa_rnr_errors_delta", IFACE),
        ):
            if device == IFACE and not with_ethtool:
                continue
            more, more_wf = diagnostic(metric, device, tag=reader)
            incidents.append(more)
            workflows.append(more_wf)
        if gapped:
            batches.append(batch(list(zeros), observed_at=future(105)))
        batches.append(batch(list(ramp), observed_at=future(120)))
    return doc(incidents=incidents, workflows=workflows, batches=batches)


def test_counter_reset_passes_on_the_reset_rule_and_fails_on_the_old_clamp(
    tmp_path: Path,
) -> None:
    result, source = run("counter_reset", tmp_path, records=_reset_doc(fixed=True))
    assert result["verdict"] == "PASS", result["errors"]
    values = [call[call.index("--value") + 1] for call in calls(source, "set-value")]
    assert "1" in values and values.index("1") > 0, (
        "the counter is reset to 1 after ramping"
    )
    ramp = [int(value) for value in values[: values.index("1")]]
    assert ramp == sorted(ramp) and ramp[0] == 7 + fx.RAMP_STEP, (
        "the ramp climbs from the real value"
    )
    assert all((int(value) >= 1 for value in values)), (
        "the ramp never writes a value below 1"
    )
    case_dir = tmp_path / "cases" / SPECS["counter_reset"].key
    readings = json.loads((case_dir / "reset-readings.json").read_text())
    assert [reading["deltas"] for reading in readings["sysfs"]] == [
        {fx.ERRORS_METRIC: 1000, "rdma_rx_drops_delta": 1000},
        {},
        {fx.ERRORS_METRIC: 1000, "rdma_rx_drops_delta": 1000},
    ], "the evidence shows every batch's port readings"
    assert readings["sysfs"][1]["collection_errors"] == [SYSFS_RESET_ERROR]
    assert [reading["deltas"] for reading in readings["ethtool"]] == [
        {"efa_rnr_errors_delta": 1000},
        {},
        {"efa_rnr_errors_delta": 1000},
    ]
    after_ramp = json.loads((case_dir / "reset-after-ramp.json").read_text())
    steps = [step["sysfs"] for step in after_ramp["steps"]]
    assert steps == sorted(steps) and steps[0] == 1 + fx.RAMP_STEP, (
        "every step is recorded with its time"
    )
    assert result["zero_readings_between_steps"] == {"sysfs": 0, "ethtool": 0}
    # The old sysfs clamp: no _rdma reset error, a 0 reading, the fault re-minted.
    result, _source = run("counter_reset", tmp_path, records=_reset_doc(fixed=False))
    assert result["verdict"] == "FAIL", result
    stages = result["stages"]
    assert stages["sysfs_reset_unknown"] == [
        f"no _rdma collection error names the {PORT_DEVICE} rx_drops reset with "
        "'delta unknown': the sysfs reader did not mark the reset UNKNOWN"
    ], stages
    assert stages["sysfs_not_reminted"] == [], (
        "the clamp's 0 is a reading in the records: the second incident matches a "
        "second episode, the FAIL is the missing UNKNOWN"
    )
    assert (
        stages["ethtool_reset_unknown"] == [] and stages["ethtool_not_reminted"] == []
    )
    assert stages["no_decrement"] == [] and stages["fault_seen"] == []


def test_counter_reset_judges_incidents_against_the_readings_episodes(
    tmp_path: Path,
) -> None:
    """A ramp slower than the tick ships 0 readings between steps (live
    2026-10-25: 7 zeros, 4 incidents per reader flagged as "re-minted" while
    the reset tick was UNKNOWN on both readers): each is a reading that ends
    the episode by the product's contract, so the incident count is judged
    against the episodes the readings show, and a 0 after the reset is no
    clamp unless it sits in the reset tick's batch."""

    result, _source = run(
        "counter_reset", tmp_path, records=_reset_doc(fixed=True, gapped=True)
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert result["zero_readings_between_steps"] == {"sysfs": 1, "ethtool": 1}
    assert any("judged against the episodes" in note for note in result["notes"]), (
        result["notes"]
    )
    result, _source = run(
        "counter_reset", tmp_path, records=_reset_doc(fixed=True, reminted=True)
    )
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["sysfs_not_reminted"] == [
        f"the reset re-minted the fault: 2 {fx.ERRORS_METRIC} incidents on "
        f"{PORT_DEVICE} for 1 episode(s) in the product's own readings"
    ]
    assert result["stages"]["ethtool_not_reminted"] == [
        f"the reset re-minted the fault: 2 efa_rnr_errors_delta incidents on {IFACE} "
        "for 1 episode(s) in the product's own readings"
    ]
    assert result["stages"]["sysfs_reset_unknown"] == [], (
        "the reset tick itself was UNKNOWN: only the re-mint fails"
    )


def test_episodes_count_threshold_crossings_of_the_readings() -> None:
    def document(*values: float) -> dict[str, Any]:
        return doc(
            batches=[
                batch(
                    [sample(fx.ERRORS_METRIC, value, PORT_DEVICE)],
                    observed_at=future(index),
                )
                for index, value in enumerate(values)
            ]
        )

    assert fx.episodes(document(), fx.ERRORS_METRIC, PORT_DEVICE) == 0
    assert fx.episodes(document(1000, 2000, 1000), fx.ERRORS_METRIC, PORT_DEVICE) == 1
    assert (
        fx.episodes(
            document(0, 1000, 0, 0, 1000, 1000, 0, 1), fx.ERRORS_METRIC, PORT_DEVICE
        )
        == 3
    )
    assert fx.episodes(document(0, 0.5, 0), fx.ERRORS_METRIC, PORT_DEVICE) == 0, (
        "below the WARNING threshold nothing opens"
    )
    assert fx.episodes(document(1000), fx.ERRORS_METRIC, "other/1") == 0


def test_reset_ramp_after_the_reset_does_not_read_the_store(tmp_path: Path) -> None:
    """``poll=False``: the after-reset ramp accepts nothing, so each step costs the
    two probe round trips only; the one read afterwards grades the whole window."""

    reads: list[str] = []
    document = _reset_doc(fixed=True)

    def records(arguments: tuple[str, ...]) -> dict[str, Any]:
        reads.append(arguments[2])
        return dict(document)

    result, _source = run("counter_reset", tmp_path, records=records)
    assert result["verdict"] == "PASS", result["errors"]
    before = json.loads(
        (
            tmp_path / "cases" / SPECS["counter_reset"].key / "reset-before-ramp.json"
        ).read_text()
    )
    after = json.loads(
        (
            tmp_path / "cases" / SPECS["counter_reset"].key / "reset-after-ramp.json"
        ).read_text()
    )
    assert len(before["steps"]) >= 1 and len(after["steps"]) >= 1
    # one read per before-ramp poll (the first step mints), one after the settle,
    # the settled wait, and none during the after-reset ramp
    assert len(reads) <= len(before["steps"]) + 3, reads


def test_counter_reset_fails_on_a_delta_shipped_in_the_reset_tick(
    tmp_path: Path,
) -> None:
    result, _source = run(
        "counter_reset",
        tmp_path,
        records=_reset_doc(fixed=True, reset_tick_reading=True),
    )
    assert result["verdict"] == "FAIL", result
    assert any(
        "rdma_rx_drops_delta=5" in error and "reset tick batch" in error
        for error in result["stages"]["sysfs_reset_unknown"]
    ), result["stages"]


def test_counter_reset_needs_the_error_to_name_the_counter(tmp_path: Path) -> None:
    records = _reset_doc(fixed=True)
    for item in records["batches"]:
        item["collection_errors"] = [
            error.replace("rx_drops reset", "error counter total reset")
            for error in item["collection_errors"]
        ]
    result, _source = run("counter_reset", tmp_path, records=records)
    assert result["verdict"] == "FAIL", result
    assert any(
        "rx_drops reset with 'delta unknown'" in error
        for error in result["stages"]["sysfs_reset_unknown"]
    ), result["stages"]


def test_counter_reset_skips_the_ethtool_half_without_a_netdev(tmp_path: Path) -> None:
    inv = inventory(ethtool_path=None)
    result, source = run(
        "counter_reset",
        tmp_path,
        inv=inv,
        records=_reset_doc(fixed=True, with_ethtool=False),
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert "ethtool_reset_unknown" not in result["stages"]
    assert calls(source, "install-wrapper") == [] and "skipped" in result["notes"][1]
