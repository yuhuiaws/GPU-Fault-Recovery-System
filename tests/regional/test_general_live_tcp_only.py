"""GF-GENERAL-LIVE-009 tcp_only: wiring, approvals, oracle math and verdicts.

The variant is driven through scripted probes (``FakeProbe``) and scripted
product records: the temporary package step runs only when ``tc`` or
``sch_netem`` is missing and only behind ``TEMP_TEST_PACKAGE_INSTALL`` (without
it the old CAPABILITY_ABSENT path stays), a refused dnf transaction is
NOT_RUN(FIXTURE_NOT_READY), only the host-side veth is declared required, and
the verdict needs the oracle's retransmissions with flat NIC counters plus a
product batch that actually carried them.
"""

from __future__ import annotations

import dataclasses
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
    *, retrans_step: int = 50, real_errors_step: int = 0, veth_drop_step: int = 0
) -> list[dict[str, Any]]:
    result = []
    for index in range(13):
        result.append(
            {
                "t": 1000.0 + 25 * index,
                "tcp_retransmits": 100 + retrans_step * index,
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


BEFORE = reading(1000.0, 0, None)
AFTER = reading(1300.0, 40, 5_000_000)
#: The wall clock the variant reads: the flow starts at T0 and stops at T0+600 s.
T0 = datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def flow_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    moments = iter([T0, T0 + timedelta(seconds=600)])
    monkeypatch.setattr(
        tcp, "utc_now", lambda: next(moments, T0 + timedelta(seconds=600))
    )


def activity(
    *,
    retrans: float = 35.0,
    veth_errors: float = 0.0,
    offset: float = 90.0,
    incidents: Sequence[dict[str, Any]] = (),
    workflows: Sequence[dict[str, Any]] = (),
) -> Any:
    def document(_arguments: Any) -> dict[str, Any]:
        observed = (T0 + timedelta(seconds=offset)).isoformat()
        return {
            "incidents": list(incidents),
            "workflows": list(workflows),
            "batches": [
                {
                    "observed_at": observed,
                    "batch_id": "host-1",
                    "edge_filter_reasons": ["health-summary"],
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
                        {
                            "name": "network_drops_delta",
                            "device": IFA,
                            "value": 0.0,
                            "labels": {},
                        },
                    ],
                }
            ],
            "markers": [],
            "commands": [],
            "health_signals": {},
            "notifications": [],
            "collector_statuses": [],
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
# oracle and verdicts
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
        records(retrans_step=1),
        interface=IFA,
        real_interfaces=[REAL],
        before=BEFORE,
        after=AFTER,
    )
    assert any("rose by 12.0 (< 20)" in error for error in flat["errors"]), flat
    reset = records()
    reset[5]["tcp_retransmits"] = 0
    backwards = tcp.tcp_oracle(
        reset, interface=IFA, real_interfaces=[REAL], before=BEFORE, after=AFTER
    )
    assert any("went backwards" in error for error in backwards["errors"]), (
        "any(('went backwards' in error for error in backwards['errors']))"
    )
    # Samples outside the flow window never count.
    late = records() + [{"t": 9999.0, "tcp_retransmits": 1, "netdev": {}}]
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
        "netem dropped 0" in e for e in errors(records(), after=reading(1300.0, 0, 9))
    ), "any(('netem dropped 0' in e for e in errors(records(), after=reading(1"
    assert any(
        "no TCP data reached" in e
        for e in errors(records(), after=reading(1300.0, 9, 0))
    ), "any(('no TCP data reached' in e for e in errors(records(), after=readi"
    assert any("0 sample(s)" in e for e in errors([])), (
        "any(('0 sample(s)' in e for e in errors([])))"
    )


def test_inconclusive_when_a_real_interface_counts_errors(tmp_path: Path) -> None:
    probes = Probes(survey(), oracle=records(real_errors_step=2))
    with pytest.raises(VariantInconclusive, match="real interface enp71s0"):
        SPEC.execute(context(tmp_path, probes))


def test_inconclusive_when_netem_drops_nothing(tmp_path: Path) -> None:
    probes = Probes(survey(), after=reading(1300.0, 0, 5_000_000))
    with pytest.raises(VariantInconclusive, match="netem dropped"):
        SPEC.execute(context(tmp_path, probes))


def test_a_nic_finding_from_tcp_loss_fails(tmp_path: Path) -> None:
    found = incident("inc-e", "network_errors_delta", device=IFA, workflow="wf-e")
    steps = workflow("wf-e", "inc-e", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"])
    probes = Probes(survey())
    document = activity(incidents=[found], workflows=[steps])
    result = SPEC.execute(context(tmp_path, probes, document=document))
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("no_nic_finding:") for error in result["errors"]), (
        "any((error.startswith('no_nic_finding:') for error in result['errors']"
    )


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
    document = activity(veth_errors=3.0)
    result = SPEC.execute(context(tmp_path, Probes(survey()), document=document))
    assert result["verdict"] == "FAIL"
    assert any(
        error.startswith("test_interface_zero:") for error in result["errors"]
    ), "any((error.startswith('test_interface_zero:') for error in result['err"


@pytest.mark.parametrize(
    "retrans,offset",
    [(0.0, 90.0), (35.0, 5.0), (35.0, 900.0)],
    ids=["zero", "before-one-interval", "after-the-stop"],
)
def test_no_product_batch_with_retransmits_is_inconclusive(
    tmp_path: Path, retrans: float, offset: float
) -> None:
    probes = Probes(survey())
    document = activity(retrans=retrans, offset=offset)
    with pytest.raises(VariantInconclusive, match="proves nothing"):
        SPEC.execute(context(tmp_path, probes, document=document))


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
