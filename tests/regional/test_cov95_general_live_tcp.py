"""GF-GENERAL-LIVE TCP-only loss: fixture refusals and the window's record reads."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional import general_live_tcp as tcp
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

SPEC = next(spec for spec in net.VARIANTS if spec.variant == "tcp_only")
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
IFA = "gfrsaaaaaaa"
SINCE = datetime(2026, 11, 5, 4, 0, tzinfo=timezone.utc)
UNTIL = datetime(2026, 11, 5, 4, 10, tzinfo=timezone.utc)
TIMING = {"fixture_deadman": 900}


def fixture_context(
    tmp_path: Path, setup: dict[str, Any], *, node: dict[str, Any] | None = None
) -> tuple[Any, FakeProbe]:
    probe = FakeProbe({"tcp-setup": setup, "cleanup": {"clean": True}})
    regional = FakeRegional(node_snapshot=node) if node else FakeRegional()
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"net@{NODE}": probe},
        settings=make_settings(SPEC, node=NODE),
    )
    return ctx, probe


def test_a_refused_tcp_fixture_is_not_run_with_the_probes_reason(
    tmp_path: Path,
) -> None:
    ctx, probe = fixture_context(
        tmp_path, {"refused": {"reason": "CAPABILITY_ABSENT", "detail": "no sch_netem"}}
    )

    with pytest.raises(VariantNotRun, match="no sch_netem") as caught:
        tcp.create_fixture(ctx, TIMING)

    assert caught.value.reason == "CAPABILITY_ABSENT"
    assert ctx.cleanup.pending() == ["net-tcp"], "the cleanup is registered first"
    assert probe.calls[0][:3] == ("tcp-setup", "--deadman-seconds", "900")


def test_a_fixture_without_a_root_netem_qdisc_is_a_fixture_error(
    tmp_path: Path,
) -> None:
    ctx, _ = fixture_context(
        tmp_path, {"interface": IFA, "qdisc": {"kind": "tbf", "root": True}}
    )

    with pytest.raises(RegionalFixtureError, match="no root netem qdisc"):
        tcp.create_fixture(ctx, TIMING)


def batch(observed_at: Any, *samples: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": extra.pop("batch_id", "host-1"),
        "samples": list(samples),
        **extra,
    }


def test_product_retransmits_count_only_positive_samples_inside_the_flow() -> None:
    sample = {"name": tcp.TCP_METRIC, "device": None}
    document = {
        "batches": [
            batch("2026-11-05T04:05:00", {**sample, "value": 4}),
            batch("not a time", {**sample, "value": 9}),
            batch("2026-11-05T04:20:00+00:00", {**sample, "value": 7}),
            batch("2026-11-05T04:06:00+00:00", {**sample, "value": 0}),
        ]
    }

    found = tcp.product_retransmits(document, SINCE, UNTIL)

    assert [item["value"] for item in found] == [4]


def test_host_delivery_inside_ignores_other_collectors() -> None:
    inside = "2026-11-05T04:05:00+00:00"
    document = {
        "collector_statuses": [
            {"collector": "GPU_DCGM", "batch_id": "host-2", "observed_at": inside},
            {
                "collector": tcp.HOST_COLLECTOR,
                "batch_id": "host-3",
                "observed_at": inside,
            },
        ]
    }

    assert tcp.host_delivery_inside(document, SINCE, UNTIL) == {
        "collector": tcp.HOST_COLLECTOR,
        "batch_id": "host-3",
        "observed_at": inside,
    }


def test_veth_latest_keeps_the_test_interfaces_counters_inside_the_window() -> None:
    inside = "2026-11-05T04:05:00+00:00"
    document = {
        "metric_latest": [
            {"name": "network_link_down", "device": IFA, "observed_at": inside},
            {"name": "network_drops_delta", "device": "enp71s0", "observed_at": inside},
            {"name": "network_drops_delta", "device": IFA, "observed_at": inside},
            {"name": "network_errors_delta", "device": IFA, "observed_at": "late"},
        ]
    }

    assert tcp.veth_latest_inside(document, IFA, SINCE, UNTIL) == [
        {
            "name": "network_drops_delta",
            "device": IFA,
            "observed_at": inside,
            "source": "metric_latest",
        }
    ]


def test_forbidden_edges_ignore_batches_before_the_window() -> None:
    edge = sorted(tcp.FORBIDDEN_EDGES)[0]
    document = {
        "batches": [
            batch("2026-11-05T03:55:00+00:00", edge_filter_reasons=[edge]),
            batch(
                "2026-11-05T04:02:00+00:00",
                batch_id="host-9",
                edge_filter_reasons=[edge],
            ),
        ]
    }

    assert tcp.forbidden_edges(document, SINCE) == [
        f"batch host-9 (2026-11-05T04:02:00+00:00) was delivered on {[edge]}: the "
        "collector took the TCP loss for a NIC/fabric fault"
    ]


def test_a_node_isolated_after_the_window_is_named(tmp_path: Path) -> None:
    node = {"name": NODE, "ready": "True", "unschedulable": True, "taints": []}
    ctx, _ = fixture_context(tmp_path, {}, node=node)

    stages = tcp.product_stages(ctx, {"incidents": [], "batches": []}, IFA, since=SINCE)

    assert stages["no_isolation"] == [f"{NODE} is isolated after a TCP-only window"]
