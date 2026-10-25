"""GF-GENERAL-LIVE-009 tcp_only: wiring, approvals, oracle math and verdicts.

The variant is driven through scripted probes (``FakeProbe``) and scripted
product records: the temporary package step runs only when ``tc`` or
``sch_netem`` is missing and only behind ``TEMP_TEST_PACKAGE_INSTALL`` (without
it the old CAPABILITY_ABSENT path stays), a refused dnf transaction is
NOT_RUN(FIXTURE_NOT_READY), only the host-side veth is declared required, and
the verdict needs the oracle's retransmissions with flat NIC counters plus a
*witness* that the product accepted a host batch inside the flow: the host
collector status row or the veth's latest counter records moving inside the
window (a summary-only batch leaves no raw evidence), or a stored batch that
happens to carry the metric -- each confirmed by the oracle's RetransSegs rise
over the collector interval that ended there.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_fixture as fx
from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional import general_live_tcp as tcp
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    store_script,
    workflow,
)

SPEC = next(spec for spec in net.VARIANTS if spec.variant == "tcp_only")
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
IFA = "gfrsaaaaaaa"
IFB = "gfrsaaaaaab"
REAL = "enp71s0"
TC_NEVRA = "iproute-tc-5.10.0-2.amzn2023.0.5.x86_64"
BOTH = ("KERNEL_MODULE_LOAD", "TEMP_TEST_PACKAGE_INSTALL")
#: The wall clock the variant reads: the flow starts at T0 and stops at T0+600 s;
#: the oracle's node clock and the product's ``observed_at`` are on it too.
T0 = datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)
T0_EPOCH = T0.timestamp()
#: 5-s oracle samples over the flow window.
ORACLE_SAMPLES = 61


def survey(*, tc: bool = True, loaded: bool = True, available: bool = True) -> dict:
    return {
        "cpu_count": 8,
        "tools": {
            "ip": True,
            "systemd-run": True,
            "nsenter": True,
            "tc": tc,
            "dnf": True,
            "rpm": True,
        },
        "modules": {
            "veth": {"loaded": True, "available": True},
            "sch_netem": {"loaded": loaded, "available": available},
        },
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
            "GPU_FAULT_REQUIRED_INTERFACES": "",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def records(
    *,
    retrans_step: int = 10,
    real_errors_step: int = 0,
    veth_drop_step: int = 0,
    flat_after: int | None = None,
) -> list[dict[str, Any]]:
    """The node oracle: RetransSegs rising ``retrans_step`` per 5-s sample
    (flat from sample ``flat_after`` on), the veth and the real interface read
    every sample."""

    result = []
    for index in range(ORACLE_SAMPLES):
        rising = index if flat_after is None else min(index, flat_after)
        result.append(
            {
                "t": T0_EPOCH + 5 * index,
                "tcp_retransmits": 100 + retrans_step * rising,
                "netdev": {
                    IFA: {
                        "present": True,
                        "rx_errors": 0,
                        "tx_errors": 0,
                        "rx_dropped": 0,
                        "tx_dropped": veth_drop_step * index,
                    },
                    REAL: {
                        "present": True,
                        "rx_errors": real_errors_step * index,
                        "tx_errors": 0,
                        "rx_dropped": 7 * index,  # real drops are not the oracle's
                        "tx_dropped": 0,
                    },
                },
            }
        )
    return result


def reading(t: float, drops: int, received: int | None) -> dict[str, Any]:
    return {
        "t": t,
        "retrans_segs": 100,
        "qdisc": {"kind": "netem", "root": True, "drops": drops},
        "receiver": None if received is None else {"received_bytes": received},
    }


BEFORE = reading(T0_EPOCH, 0, None)
AFTER = reading(T0_EPOCH + 300, 40, 5_000_000)


@pytest.fixture(autouse=True)
def flow_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two readings per execute (the flow's start and stop); a test that runs
    # the variant twice gets the same pair again.
    moments = itertools.cycle([T0, T0 + timedelta(seconds=600)])
    monkeypatch.setattr(tcp, "utc_now", lambda: next(moments))


def at(offset: float) -> str:
    return (T0 + timedelta(seconds=offset)).isoformat()


def status_row(offset: float, *, prefix: str = "host-") -> dict[str, Any]:
    """The HOST_TELEMETRY collector status: upserted for every accepted batch."""

    return {
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
        "collector": "HOST_TELEMETRY",
        "observed_at": at(offset),
        "ingested_at": at(offset + 1),
        "last_success_at": at(offset),
        "last_error_at": None,
        "batch_id": f"{prefix}{NODE}-{int(T0_EPOCH + offset) * 1_000_000}",
        "sample_count": 101,
        "errors": [],
    }


def latest_row(name: str, value: float, offset: float) -> dict[str, Any]:
    return {
        "name": name,
        "device": IFA,
        "value": value,
        "observed_at": at(offset),
        "labels": {},
    }


def evidence_batch(
    offset: float,
    retrans: float,
    *,
    veth_errors: float = 0.0,
    reasons: Sequence[str] = ("collection-error:_bmc",),
) -> dict[str, Any]:
    """A stored host batch (delivered on an edge) carrying the TCP metric."""

    return {
        "observed_at": at(offset),
        "batch_id": f"host-{NODE}-{int(T0_EPOCH + offset) * 1_000_000}",
        "edge_filter_reasons": list(reasons),
        "collection_errors": [],
        "samples": [
            {
                "name": "tcp_retransmits_delta",
                "device": None,
                "value": retrans,
                "labels": {},
            },
            {
                "name": "network_errors_delta",
                "device": IFA,
                "value": veth_errors,
                "labels": {},
            },
            {"name": "network_drops_delta", "device": IFA, "value": 0.0, "labels": {}},
        ],
    }


def activity(
    *,
    status_offset: float | None = 90.0,
    status_prefix: str = "host-",
    latest_offset: float | None = 90.0,
    latest_values: tuple[float, float] = (0.0, 0.0),
    batches: Sequence[dict[str, Any]] = (),
    incidents: Sequence[dict[str, Any]] = (),
    workflows: Sequence[dict[str, Any]] = (),
) -> Any:
    """The product's records: by default the live shape -- no raw evidence,
    the status row and the veth's latest records moved 90 s into the flow."""

    def document(_arguments: Any) -> dict[str, Any]:
        return {
            "since": T0.isoformat(),
            # Minted inside the read: the live probe lists the node's incidents
            # created since ``since`` and the verdicts take only those.
            "incidents": [{**item, "created_at": at(100)} for item in incidents],
            "workflows": list(workflows),
            "batches": list(batches),
            "markers": [],
            "commands": [],
            "health_signals": {},
            "notifications": [],
            "metric_latest": []
            if latest_offset is None
            else [
                latest_row("network_errors_delta", latest_values[0], latest_offset),
                latest_row("network_drops_delta", latest_values[1], latest_offset),
            ],
            "collector_statuses": []
            if status_offset is None
            else [status_row(status_offset, prefix=status_prefix)],
        }

    return document


class Probes:
    def __init__(
        self,
        node_survey: dict[str, Any],
        *,
        oracle: list[dict[str, Any]] | None = None,
        after: dict[str, Any] = AFTER,
        ensure: dict[str, Any] | None = None,
    ) -> None:
        rows = records() if oracle is None else oracle
        self.node = FakeProbe(
            {
                "survey": node_survey,
                "oracle-start": {"unit": "oracle"},
                "oracle-read": {
                    "records": rows,
                    "next_offset": len(rows),
                    "total": len(rows),
                },
                "oracle-stop": {},
            }
        )
        self.net = FakeProbe(
            {
                "tcp-setup": {
                    "interface": IFA,
                    "peer": IFB,
                    "netns": "gfrsaaaaaans",
                    "qdisc": {"kind": "netem", "root": True, "loss": "5%"},
                    "real_interfaces": [REAL],
                    "refused": None,
                },
                "tcp-start": {"units": {}},
                "tcp-reading": [BEFORE, after],
                "tcp-stop": {"stopped": []},
                "cleanup": {"clean": True},
            }
        )
        self.window = FakeProbe(
            {
                "open-window": {"after": {"ActiveState": "active"}},
                "close-window": {"after": {"ActiveState": "active"}},
            }
        )
        self.deps = FakeProbe(
            {
                "ensure": ensure
                or {
                    "refused": None,
                    "tools": {"tc": True},
                    "modules": {"sch_netem": {"loaded": True, "builtin": False}},
                    "installed": [TC_NEVRA],
                    "modules_loaded_by_run": ["sch_netem"],
                    "transaction": [],
                },
                "cleanup": {"clean": True},
            }
        )

    def mapping(self) -> dict[str, FakeProbe]:
        return {
            f"node@{NODE}": self.node,
            f"net@{NODE}": self.net,
            f"window@{NODE}": self.window,
            f"deps@{NODE}": self.deps,
        }


def context(
    tmp_path: Path,
    probes: Probes,
    *,
    approvals: tuple[str, ...] = ("KERNEL_MODULE_LOAD",),
    document: Any = None,
) -> fx.VariantContext:
    regional = FakeRegional(
        cpu_python=store_script({"activity": document or activity()})
    )
    settings = dataclasses.replace(make_settings(SPEC, node=NODE), approvals=approvals)
    return make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes.mapping(),
        settings=settings,
    )


def subcommands(probe: FakeProbe) -> list[str]:
    return [call[0] for call in probe.calls]


def sources(result: dict[str, Any]) -> list[str]:
    return [item["source"] for item in result["witnesses"]["confirmed"]]


# --------------------------------------------------------------------------- #
# the temporary dependency step: only when missing, only when approved
# --------------------------------------------------------------------------- #
def test_without_the_approval_a_missing_tc_stays_capability_absent(
    tmp_path: Path,
) -> None:
    probes = Probes(survey(tc=False))
    ctx = context(tmp_path, probes)
    with pytest.raises(VariantNotRun) as raised:
        SPEC.execute(ctx)
    assert raised.value.reason == "CAPABILITY_ABSENT"
    assert raised.value.detail.startswith("the node lacks tc"), (
        "raised.value.detail.startswith('the node lacks tc')"
    )
    assert probes.deps.calls == [] and probes.net.calls == []


def test_with_the_approval_a_missing_tc_is_installed_for_the_run(
    tmp_path: Path,
) -> None:
    probes = Probes(survey(tc=False, loaded=False))
    ctx = context(tmp_path, probes, approvals=BOTH)
    result = SPEC.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    ensure = probes.deps.calls[0]
    assert ensure[:5] == ("ensure", "--tool", "tc", "--module", "sch_netem")
    assert "--allow-install" in ensure and "--allow-module-load" in ensure
    assert ensure[ensure.index("--deadman-seconds") + 1] == "3600"
    assert result["dependencies"]["installed"] == [TC_NEVRA]
    # The package cleanup is registered first, so it runs last (LIFO), after
    # the fixture is gone and the module is no longer in use.
    assert ctx.cleanup.pending() == ["deps", "net-tcp", "oracle-tcp"]
    ctx.cleanup.run()
    assert ctx.cleanup.failed == [], ctx.cleanup.results
    assert ("cleanup", "--kind", "tcp", "--run-id", ctx.run_id) in probes.net.calls
    assert probes.deps.calls[-1] == ("cleanup", "--run-id", ctx.run_id)


def test_tools_already_on_the_node_touch_no_package(tmp_path: Path) -> None:
    probes = Probes(survey())
    result = SPEC.execute(context(tmp_path, probes, approvals=BOTH))
    assert result["verdict"] == "PASS", result["errors"]
    assert probes.deps.calls == [], "nothing missing, so no dependency probe at all"
    assert result["dependencies"]["changed"] is False


def test_a_missing_module_is_loaded_without_installing(tmp_path: Path) -> None:
    probes = Probes(survey(loaded=False))
    SPEC.execute(context(tmp_path, probes))
    ensure = probes.deps.calls[0]
    assert "--allow-module-load" in ensure and "--allow-install" not in ensure


def test_a_refused_install_is_fixture_not_ready(tmp_path: Path) -> None:
    detail = "dnf would also installing dependencies libbpf-1.0-1.amzn2023.x86_64"
    probes = Probes(
        survey(tc=False),
        ensure={"refused": {"reason": "FIXTURE_NOT_READY", "detail": detail}},
    )
    ctx = context(tmp_path, probes, approvals=BOTH)
    with pytest.raises(VariantNotRun) as raised:
        SPEC.execute(ctx)
    assert (raised.value.reason, raised.value.detail) == ("FIXTURE_NOT_READY", detail)
    assert probes.net.calls == [], "no fixture once the dependency step refused"
    assert ctx.cleanup.pending() == ["deps"], "a half-done install is still undone"


def test_an_unknown_refusal_reason_is_fixture_not_ready(tmp_path: Path) -> None:
    probes = Probes(survey(tc=False), ensure={"refused": {"reason": "WHATEVER"}})
    with pytest.raises(VariantNotRun) as raised:
        SPEC.execute(context(tmp_path, probes, approvals=BOTH))
    assert raised.value.reason == "FIXTURE_NOT_READY"


def test_a_dirty_dependency_cleanup_fails_the_cleanup_stack(tmp_path: Path) -> None:
    probes = Probes(survey(tc=False))
    probes.deps.script["cleanup"] = {
        "clean": False,
        "residue": {"packages": [TC_NEVRA]},
    }
    ctx = context(tmp_path, probes, approvals=BOTH)
    SPEC.execute(ctx)
    ctx.cleanup.run()
    assert [item["name"] for item in ctx.cleanup.failed] == ["deps"]


# --------------------------------------------------------------------------- #
# the fixture and the window
# --------------------------------------------------------------------------- #
def test_only_the_host_side_veth_is_declared_required(tmp_path: Path) -> None:
    probes = Probes(survey())
    SPEC.execute(context(tmp_path, probes))
    opened = next(call for call in probes.window.calls if call[0] == "open-window")
    declared = [
        opened[i + 1] for i, item in enumerate(opened) if item == "--append-env"
    ]
    assert declared == [f"GPU_FAULT_REQUIRED_INTERFACES={IFA}"]
    restore = opened[opened.index("--restore-seconds") + 1]
    setup = probes.net.calls[0]
    assert setup[0] == "tcp-setup" and int(setup[2]) > int(restore), (
        "the fixture deadman outlives the window, so a required veth never vanishes "
        "while it is declared"
    )


def test_the_oracle_samples_the_test_veth_and_every_real_interface(
    tmp_path: Path,
) -> None:
    probes = Probes(survey())
    SPEC.execute(context(tmp_path, probes))
    started = next(call for call in probes.node.calls if call[0] == "oracle-start")
    netdevs = [started[i + 1] for i, item in enumerate(started) if item == "--netdev"]
    assert netdevs == [IFA, REAL] and "--snmp" in started


def test_a_fixture_on_a_non_test_interface_is_refused(tmp_path: Path) -> None:
    probes = Probes(survey())
    probes.net.script["tcp-setup"] = {
        **probes.net.script["tcp-setup"],
        "interface": REAL,
    }
    with pytest.raises(tcp.RegionalFixtureError, match="not a gfrs test interface"):
        SPEC.execute(context(tmp_path, probes))


# --------------------------------------------------------------------------- #
# the stimulus oracle
# --------------------------------------------------------------------------- #
def test_tcp_oracle_math() -> None:
    good = tcp.tcp_oracle(
        records(), interface=IFA, real_interfaces=[REAL], before=BEFORE, after=AFTER
    )
    assert good["errors"] == [], good["errors"]
    assert (good["retrans_delta"], good["netem_drops"], good["received_bytes"]) == (
        600.0,
        40.0,
        5_000_000.0,
    )
    assert good["real_interfaces"][REAL] == {"rx_errors": 0.0, "tx_errors": 0.0}
    flat = tcp.tcp_oracle(
        records(retrans_step=0),
        interface=IFA,
        real_interfaces=[REAL],
        before=BEFORE,
        after=AFTER,
    )
    assert any("rose by 0.0 (< 20)" in error for error in flat["errors"]), flat
    reset = records()
    reset[5]["tcp_retransmits"] = 0
    backwards = tcp.tcp_oracle(
        reset, interface=IFA, real_interfaces=[REAL], before=BEFORE, after=AFTER
    )
    assert any("went backwards" in error for error in backwards["errors"]), (
        "any(('went backwards' in error for error in backwards['errors']))"
    )
    # Samples outside the flow window never count.
    late = records() + [{"t": T0_EPOCH + 9999.0, "tcp_retransmits": 1, "netdev": {}}]
    assert (
        tcp.tcp_oracle(
            late, interface=IFA, real_interfaces=[REAL], before=BEFORE, after=AFTER
        )["errors"]
        == []
    )


def test_tcp_oracle_needs_flat_nic_counters_and_a_working_flow() -> None:
    def errors(rows: list[dict[str, Any]], **kw: Any) -> list[str]:
        return tcp.tcp_oracle(
            rows,
            interface=IFA,
            real_interfaces=kw.get("real", [REAL]),
            before=BEFORE,
            after=kw.get("after", AFTER),
        )["errors"]

    assert any(
        "real interface enp71s0" in e for e in errors(records(real_errors_step=1))
    ), "any(('real interface enp71s0' in e for e in errors(records(real_errors"
    assert any("netem must not move" in e for e in errors(records(veth_drop_step=1))), (
        "any(('netem must not move' in e for e in errors(records(veth_drop_step"
    )
    assert any(
        "was not read throughout" in e for e in errors(records(), real=["ens9"])
    ), "any(('was not read throughout' in e for e in errors(records(), real=['"
    assert any(
        "netem dropped 0" in e
        for e in errors(records(), after=reading(T0_EPOCH + 300, 0, 9))
    ), "any(('netem dropped 0' in e for e in errors(records(), after=reading(1"
    assert any(
        "no TCP data reached" in e
        for e in errors(records(), after=reading(T0_EPOCH + 300, 9, 0))
    ), "any(('no TCP data reached' in e for e in errors(records(), after=readi"
    assert any("0 sample(s)" in e for e in errors([])), (
        "any(('0 sample(s)' in e for e in errors([])))"
    )


def test_oracle_retransmits_reads_the_collector_interval_ending_at_a_batch() -> None:
    rows = records()
    moment = T0 + timedelta(seconds=90)
    # [90 - 15 - 5, 90 + 5] holds the samples at 70..95: 10 per 5 s -> 50.
    assert tcp.oracle_retransmits(rows, moment, 15) == 50.0
    assert tcp.oracle_retransmits(records(flat_after=12), moment, 15) == 0.0
    assert tcp.oracle_retransmits(rows[:2], moment, 15) is None, "no samples there"


def test_inconclusive_when_a_real_interface_counts_errors(tmp_path: Path) -> None:
    probes = Probes(survey(), oracle=records(real_errors_step=2))
    with pytest.raises(VariantInconclusive, match="real interface enp71s0"):
        SPEC.execute(context(tmp_path, probes))


def test_inconclusive_when_netem_drops_nothing(tmp_path: Path) -> None:
    probes = Probes(survey(), after=reading(T0_EPOCH + 300, 0, 5_000_000))
    with pytest.raises(VariantInconclusive, match="netem dropped"):
        SPEC.execute(context(tmp_path, probes))


def test_inconclusive_when_the_flow_produced_no_retransmits(tmp_path: Path) -> None:
    probes = Probes(survey(), oracle=records(retrans_step=0))
    with pytest.raises(VariantInconclusive, match="rose by 0.0 .* ineffective"):
        SPEC.execute(context(tmp_path, probes))
    # The stimulus failed, so the product's records were never judged.
    assert "tcp-window.json" not in {path.name for path in tmp_path.rglob("*.json")}


# --------------------------------------------------------------------------- #
# the witness: the product accepted a host batch inside the flow
# --------------------------------------------------------------------------- #
def test_passes_on_the_collector_status_witness_alone(tmp_path: Path) -> None:
    document = activity(latest_offset=None)
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "PASS", result["errors"]
    assert sources(result) == ["collector_status"]
    witness = result["witnesses"]["confirmed"][0]
    assert witness["observed_at"] == at(90) and witness["oracle_retransmits"] == 50.0
    assert witness["batch_id"].startswith(f"host-{NODE}-")
    assert result["product_retransmits"] == [], "no raw evidence was needed"


def test_passes_on_the_veth_latest_records_alone(tmp_path: Path) -> None:
    document = activity(status_offset=None)
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "PASS", result["errors"]
    assert sources(result) == ["metric_latest", "metric_latest"]
    assert {item["name"] for item in result["witnesses"]["confirmed"]} == {
        "network_errors_delta",
        "network_drops_delta",
    }


def test_passes_on_a_stored_batch_that_carries_the_metric(tmp_path: Path) -> None:
    document = activity(
        status_offset=None, latest_offset=None, batches=[evidence_batch(90, 35.0)]
    )
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "PASS", result["errors"]
    assert sources(result) == ["evidence"]
    assert [item["value"] for item in result["product_retransmits"]] == [35.0]


def test_the_window_evidence_names_the_witnesses(tmp_path: Path) -> None:
    ctx = context(tmp_path, Probes(survey()))
    result = SPEC.execute(ctx)
    assert sources(result) == ["collector_status", "metric_latest", "metric_latest"]
    window = json.loads((ctx.case_dir / "tcp-window.json").read_text())
    assert (window["since"], window["stopped"]) == (at(15), at(600))
    assert window["batches"] == [] and window["witnesses"] == result["witnesses"]
    assert window["host_collector"][0]["last_success_at"] == at(90)
    assert [item["name"] for item in window["metric_latest"]] == [
        "network_errors_delta",
        "network_drops_delta",
    ]
    assert any(
        "summary-only batch leaves no raw evidence" in n for n in result["notes"]
    ), result["notes"]


@pytest.mark.parametrize(
    "document,needle",
    [
        pytest.param(
            activity(status_offset=5.0, latest_offset=-100.0),
            "host collector last success " + at(5),
            id="before-one-interval",
        ),
        pytest.param(
            activity(status_offset=900.0, latest_offset=900.0),
            "proves nothing",
            id="after-the-stop",
        ),
        pytest.param(
            activity(status_prefix="k8s-efa-", latest_offset=None),
            "host collector last success none",
            id="allocatable-reader-not-the-host-collector",
        ),
        pytest.param(
            activity(status_offset=None, latest_offset=None),
            "latest gfrsaaaaaaa counter record observed never",
            id="nothing-at-all",
        ),
    ],
)
def test_no_witness_inside_the_window_is_inconclusive(
    tmp_path: Path, document: Any, needle: str
) -> None:
    with pytest.raises(VariantInconclusive, match="silent channel") as caught:
        SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert needle in str(caught.value), str(caught.value)


def test_a_witness_the_oracle_cannot_confirm_is_inconclusive(tmp_path: Path) -> None:
    # RetransSegs rose 120 in the first minute (the stimulus oracle is content)
    # and not at all over the interval ending at the status row's batch.
    probes = Probes(survey(), oracle=records(flat_after=12))
    with pytest.raises(VariantInconclusive, match="silent channel") as caught:
        SPEC.execute(context(tmp_path, probes, document=activity(latest_offset=None)))
    assert "1 candidate(s) the oracle could not confirm" in str(caught.value)


def test_a_stored_batch_with_zero_retransmits_is_no_witness(tmp_path: Path) -> None:
    document = activity(
        status_offset=None, latest_offset=None, batches=[evidence_batch(90, 0.0)]
    )
    with pytest.raises(VariantInconclusive, match="proves nothing"):
        SPEC.execute(context(tmp_path, Probes(survey()), document=document))


# --------------------------------------------------------------------------- #
# the negative: anything the product minted from the loss fails
# --------------------------------------------------------------------------- #
def test_a_nic_finding_from_tcp_loss_fails(tmp_path: Path) -> None:
    found = incident("inc-e", "network_errors_delta", device=IFA, workflow="wf-e")
    steps = workflow("wf-e", "inc-e", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"])
    document = activity(incidents=[found], workflows=[steps])
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("no_nic_finding:") for error in result["errors"]), (
        "any((error.startswith('no_nic_finding:') for error in result['errors']"
    )


def test_an_rdma_finding_from_tcp_loss_fails(tmp_path: Path) -> None:
    found = incident("inc-r", "rdma_errors_delta", device="rdmap0s29", workflow="wf-r")
    steps = workflow("wf-r", "inc-r", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"])
    document = activity(incidents=[found], workflows=[steps])
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("no_rdma_finding:") for error in result["errors"]), (
        result["errors"]
    )


def test_a_finding_fails_even_without_a_witness(tmp_path: Path) -> None:
    found = incident("inc-d", "network_drops_delta", device=IFA, workflow="wf-d")
    steps = workflow("wf-d", "inc-d", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"])
    document = activity(
        status_offset=None, latest_offset=None, incidents=[found], workflows=[steps]
    )
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "FAIL", "a minted finding is never INCONCLUSIVE"


def test_a_hardware_plan_fails(tmp_path: Path) -> None:
    found = incident("inc-x", "gpu_temperature", workflow="wf-x")
    steps = workflow("wf-x", "inc-x", ["FREEZE_EVIDENCE", "RESET_GPU"])
    document = activity(incidents=[found], workflows=[steps])
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("hardware:") for error in result["errors"]), (
        "any((error.startswith('hardware:') for error in result['errors']))"
    )


def test_nonzero_product_counters_on_the_test_veth_fail(tmp_path: Path) -> None:
    stored = activity(batches=[evidence_batch(90, 35.0, veth_errors=3.0)])
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=stored))
    assert result["verdict"] == "FAIL"
    assert any(
        error.startswith("test_interface_zero:")
        and "read network_errors_delta=3.0" in error
        for error in result["errors"]
    ), result["errors"]
    latest = activity(latest_values=(0.0, 2.0))
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=latest))
    assert result["verdict"] == "FAIL"
    assert any(
        error.startswith("test_interface_zero:")
        and "latest network_drops_delta" in error
        for error in result["errors"]
    ), result["errors"]


def test_a_stored_batch_on_a_nic_or_fabric_edge_fails(tmp_path: Path) -> None:
    for reason in ("threshold:network_drops_delta", "threshold:efa_rnr_errors_delta"):
        document = activity(batches=[evidence_batch(90, 35.0, reasons=(reason,))])
        result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
        assert result["verdict"] == "FAIL", reason
        assert any(
            error.startswith("no_fabric_edge:") and reason in error
            for error in result["errors"]
        ), result["errors"]
    # The edges a quiet node delivers on are no fault: baseline, summary, a
    # collection error elsewhere, a recovered reason.
    for reason in ("baseline", "health-summary", "collection-error:_bmc", "recovered"):
        document = activity(batches=[evidence_batch(90, 35.0, reasons=(reason,))])
        result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
        assert result["verdict"] == "PASS", (reason, result["errors"])


def test_timing_keeps_the_deadmen_in_order(tmp_path: Path) -> None:
    ctx = context(tmp_path, Probes(survey()))
    ctx.survey = survey()
    timing = tcp.tcp_timing(ctx)
    assert timing == {
        "flow_wait": 510,
        "flow_seconds": 690,
        "oracle_seconds": 750,
        "restore_seconds": 1455,
        "fixture_deadman": 2400,
        "deps_deadman": 3600,
    }
    assert timing["restore_seconds"] + 600 <= timing["fixture_deadman"]
    install = tcp.DEPS_TIMEOUT
    assert timing["fixture_deadman"] + install < timing["deps_deadman"] <= 5400


# --------------------------------------------------------------------------- #
# registry and plan
# --------------------------------------------------------------------------- #
def test_the_approval_token_and_the_spec() -> None:
    meaning = reg.APPROVALS["TEMP_TEST_PACKAGE_INSTALL"]
    for needle in ("dnf", "iproute-tc", "kernel-modules-extra-$(uname -r)", "deadman"):
        assert needle in meaning, needle
    assert SPEC.approvals == ("KERNEL_MODULE_LOAD",)
    assert SPEC.optional_approvals == ("TEMP_TEST_PACKAGE_INSTALL",)
    assert SPEC.describe()["optional_approvals"] == ["TEMP_TEST_PACKAGE_INSTALL"]
    assert SPEC.proof_mode == reg.NATIVE and SPEC.implemented
    for needle in ("collector status", "latest counter records", "RetransSegs rise"):
        assert needle in SPEC.expected, needle


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"optional_approvals": ("NOPE",)}, "unknown approvals"),
        (
            {
                "approvals": ("KERNEL_MODULE_LOAD",),
                "optional_approvals": ("KERNEL_MODULE_LOAD",),
            },
            "both required and optional",
        ),
    ],
)
def test_validate_spec_checks_optional_approvals(overrides, match) -> None:
    with pytest.raises(reg.RegistryError, match=match):
        reg.validate_spec(dataclasses.replace(SPEC, **overrides))


def test_plan_binds_the_optional_approval(tmp_path: Path) -> None:
    from scripts.e2e.regional import run_general_live as runner

    settings = dataclasses.replace(make_settings(SPEC, node=NODE), approvals=BOTH)
    details = runner.plan_details(SPEC, settings, {"errors": []})
    assert details["approvals_required"] == ["KERNEL_MODULE_LOAD"]
    assert details["approvals_optional"] == ["TEMP_TEST_PACKAGE_INSTALL"]
    assert details["approvals_given"] == list(BOTH)
    assert set(details["approval_meanings"]) == set(BOTH)


def test_an_unapproved_optional_token_does_not_block_the_plan() -> None:
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "preflight": {
                    "agent": {"lifecycle_state": "ACTIVE"},
                    "open_incidents": [],
                    "active_workflows": [],
                    "collector_statuses": [
                        {
                            "collector": channel,
                            "last_success_at": datetime.now(timezone.utc).isoformat(),
                            "last_error_at": None,
                        }
                        for channel in (
                            "HOST_TELEMETRY",
                            "GPU_INVENTORY",
                            "GPU_METRICS",
                        )
                    ],
                }
            }
        )
    )
    settings = make_settings(SPEC, node=NODE)
    preflight = fx.read_only_preflight(settings, SPEC, regional)
    assert preflight["planned_verdict"] is None and preflight["errors"] == []
    assert preflight["missing_approvals"] == []
