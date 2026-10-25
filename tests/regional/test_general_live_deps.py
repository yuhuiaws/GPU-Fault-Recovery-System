"""The shared temporary-package step (``general_live_deps.ensure_dependencies``).

One helper for 009 ``tcp_only``, the 007 SMART variants and 004
``probe_error``: a missing tool stays NOT_RUN(CAPABILITY_ABSENT) with the token
hint unless ``TEMP_TEST_PACKAGE_INSTALL`` is given, the probe is asked for
exactly the reviewed tools/modules with its cleanup registered first (LIFO: the
package goes after the fixture), a refusal is the probe's NOT_RUN, the survey
reflects the installed tool afterwards, and nothing is touched when the node
has everything.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_deps as deps
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.probes import general_live_deps_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
SMART_NEVRA = "smartmontools-1:7.2-11.amzn2023.0.1.x86_64"
SELINUX_NEVRA = "smartmontools-selinux-1:7.2-11.amzn2023.0.1.noarch"
#: The probe's receipt: the closure in install order, dnf's table as printed.
SMART_INSTALLED = [SELINUX_NEVRA, SMART_NEVRA]
SMART_TRANSACTION = [
    {"name": "smartmontools", "section": "Installing"},
    {"name": "smartmontools-selinux", "section": "Installing dependencies"},
]
SPEC = VariantSpec(
    case_id="GF-GENERAL-LIVE-007",
    variant="health_failed",
    proof_mode=NATIVE,
    lanes=("eks",),
    risk="L",
    injection="x",
    expected="y",
    cleanup="z",
    max_minutes=16,
    execute=lambda ctx: {},
)


def survey(**tools: bool) -> dict[str, Any]:
    return {
        "tools": {"systemd-run": True, "dnf": True, "rpm": True, **tools},
        "modules": {"sch_netem": {"loaded": False, "available": True}},
    }


def ensured(**overrides: Any) -> dict[str, Any]:
    value = {
        "refused": None,
        "tools": {"smartctl": True},
        "modules": {},
        "installed": SMART_INSTALLED,
        "modules_loaded_by_run": [],
        "transaction": SMART_TRANSACTION,
        "deadman": "gfrs-abcdef-pdeadman",
    }
    value.update(overrides)
    return value


def context(
    tmp_path: Path, deps_probe: FakeProbe, approvals: tuple[str, ...] = ()
) -> Any:
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    settings = dataclasses.replace(make_settings(SPEC, node=NODE), approvals=approvals)
    return make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"deps@{NODE}": deps_probe},
        settings=settings,
    )


def test_without_the_token_a_missing_tool_is_capability_absent_with_the_hint(
    tmp_path: Path,
) -> None:
    deps_probe = FakeProbe({"ensure": ensured()})
    ctx = context(tmp_path, deps_probe)
    value = survey(smartctl=False)
    with pytest.raises(VariantNotRun) as raised:
        deps.ensure_dependencies(ctx, value, tools=("smartctl",), deadman_seconds=2400)
    assert raised.value.reason == "CAPABILITY_ABSENT"
    assert raised.value.detail == (
        "the node lacks smartctl (TEMP_TEST_PACKAGE_INSTALL installs smartmontools "
        "with its reviewed dependency smartmontools-selinux for one run)"
    ), "the hint names the closure: what the token would put on the node"
    assert deps_probe.calls == [] and ctx.cleanup.pending() == []
    assert value["tools"]["smartctl"] is False, "the survey is not touched"


def test_with_the_token_the_probe_installs_and_the_cleanup_is_registered_first(
    tmp_path: Path,
) -> None:
    deps_probe = FakeProbe({"ensure": ensured(), "cleanup": {"clean": True}})
    ctx = context(tmp_path, deps_probe, approvals=(deps.PACKAGE_APPROVAL,))
    value = survey(smartctl=False)
    ctx.cleanup.push("earlier", lambda: None)
    result = deps.ensure_dependencies(
        ctx, value, tools=("smartctl",), deadman_seconds=2400, evidence="smart-deps"
    )
    assert deps_probe.calls == [
        (
            "ensure",
            "--tool",
            "smartctl",
            "--deadman-seconds",
            "2400",
            "--allow-install",
            "--run-id",
            ctx.run_id,
        )
    ]
    assert deps_probe.timeouts == [("ensure", deps.DEPS_TIMEOUT)]
    assert result["changed"] is True and result["installed"] == SMART_INSTALLED
    assert deps.dependency_summary(result) == {
        "changed": True,
        "installed": SMART_INSTALLED,
        "modules_loaded_by_run": [],
        "transaction": SMART_TRANSACTION,
    }
    assert value["tools"]["smartctl"] is True, "the survey now says the node has it"
    assert (ctx.case_dir / "smart-deps.json").is_file(), "the receipt is evidence"
    # Registered on the stack right before the call: LIFO runs it after every
    # fixture registered later, and after a half-finished install too.
    assert ctx.cleanup.pending() == ["earlier", "deps"]
    ctx.cleanup.push("fixture", lambda: None)
    assert ctx.cleanup.run() and ctx.cleanup.failed == [], ctx.cleanup.results
    assert deps_probe.calls[-1] == ("cleanup", "--run-id", ctx.run_id)
    assert deps_probe.timeouts[-1] == ("cleanup", deps.DEPS_CLEANUP_TIMEOUT)


def test_a_node_with_everything_touches_nothing(tmp_path: Path) -> None:
    deps_probe = FakeProbe({"ensure": ensured()})
    ctx = context(tmp_path, deps_probe, approvals=(deps.PACKAGE_APPROVAL,))
    result = deps.ensure_dependencies(
        ctx, survey(smartctl=True), tools=("smartctl",), deadman_seconds=2400
    )
    assert result == {"changed": False, "installed": [], "modules_loaded_by_run": []}
    assert deps_probe.calls == [] and ctx.cleanup.pending() == []


def test_a_refused_transaction_is_the_probes_not_run_and_keeps_the_cleanup(
    tmp_path: Path,
) -> None:
    refused = {
        "reason": "FIXTURE_NOT_READY",
        "detail": "dnf would also install the dependencies mailx-12.5-30.amzn2023.x86_64",
    }
    deps_probe = FakeProbe({"ensure": ensured(refused=refused, installed=[])})
    ctx = context(tmp_path, deps_probe, approvals=(deps.PACKAGE_APPROVAL,))
    with pytest.raises(VariantNotRun) as raised:
        deps.ensure_dependencies(
            ctx, survey(smartctl=False), tools=("smartctl",), deadman_seconds=2400
        )
    assert (raised.value.reason, raised.value.detail) == (
        "FIXTURE_NOT_READY",
        refused["detail"],
    )
    assert ctx.cleanup.pending() == ["deps"], "a half-finished install is undone"


def test_an_unknown_refusal_reason_is_fixture_not_ready(tmp_path: Path) -> None:
    deps_probe = FakeProbe({"ensure": ensured(refused={"reason": "WHATEVER"})})
    ctx = context(tmp_path, deps_probe, approvals=(deps.PACKAGE_APPROVAL,))
    with pytest.raises(VariantNotRun) as raised:
        deps.ensure_dependencies(
            ctx, survey(smartctl=False), tools=("smartctl",), deadman_seconds=2400
        )
    assert raised.value.reason == "FIXTURE_NOT_READY"
    assert raised.value.detail == "the dependency step refused"


def test_a_tool_still_missing_after_the_step_is_a_fixture_error(tmp_path: Path) -> None:
    deps_probe = FakeProbe({"ensure": ensured(tools={"smartctl": False})})
    ctx = context(tmp_path, deps_probe, approvals=(deps.PACKAGE_APPROVAL,))
    with pytest.raises(RegionalFixtureError, match="smartctl still unusable"):
        deps.ensure_dependencies(
            ctx, survey(smartctl=False), tools=("smartctl",), deadman_seconds=2400
        )


def test_only_the_probes_allow_list_can_be_asked_for(tmp_path: Path) -> None:
    ctx = context(tmp_path, FakeProbe(), approvals=(deps.PACKAGE_APPROVAL,))
    with pytest.raises(RegionalFixtureError, match="no reviewed package for ipmitool"):
        deps.ensure_dependencies(
            ctx, survey(ipmitool=False), tools=("ipmitool",), deadman_seconds=2400
        )
    assert deps.TOOL_PACKAGES is probe.TOOL_PACKAGES, (
        "one allow-list: the probe's, shipped to the node"
    )
    assert deps.PACKAGE_CLOSURES is probe.PACKAGE_CLOSURES, (
        "and one set of reviewed closures: the probe's"
    )


@pytest.mark.parametrize(
    ("module", "approvals", "reason", "needle"),
    [
        (
            {"loaded": False, "available": True},
            (deps.PACKAGE_APPROVAL,),
            "NOT_AUTHORIZED",
            "needs KERNEL_MODULE_LOAD",
        ),
        (
            {"loaded": False, "available": False},
            (deps.MODULE_APPROVAL,),
            "CAPABILITY_ABSENT",
            "installs the running kernel's module package",
        ),
    ],
)
def test_module_loads_need_their_own_token_or_package(
    tmp_path: Path,
    module: dict[str, bool],
    approvals: tuple[str, ...],
    reason: str,
    needle: str,
) -> None:
    deps_probe = FakeProbe({"ensure": ensured()})
    ctx = context(tmp_path, deps_probe, approvals=approvals)
    value = survey(tc=True)
    value["modules"]["sch_netem"] = module
    with pytest.raises(VariantNotRun) as raised:
        deps.ensure_dependencies(
            ctx, value, tools=("tc",), modules=("sch_netem",), deadman_seconds=2400
        )
    assert raised.value.reason == reason and needle in raised.value.detail
    assert deps_probe.calls == []


def test_a_module_load_alone_asks_for_no_install(tmp_path: Path) -> None:
    answer = ensured(
        tools={"tc": True},
        modules={"sch_netem": {"loaded": True, "builtin": False}},
        installed=[],
        modules_loaded_by_run=["sch_netem"],
    )
    deps_probe = FakeProbe({"ensure": answer})
    ctx = context(tmp_path, deps_probe, approvals=(deps.MODULE_APPROVAL,))
    result = deps.ensure_dependencies(
        ctx, survey(tc=True), tools=("tc",), modules=("sch_netem",), deadman_seconds=900
    )
    call = deps_probe.calls[0]
    assert call[:5] == ("ensure", "--tool", "tc", "--module", "sch_netem")
    assert "--allow-module-load" in call and "--allow-install" not in call
    assert result["modules_loaded_by_run"] == ["sch_netem"]


def test_checked_cleanup_fails_the_stack_on_residue(tmp_path: Path) -> None:
    deps_probe = FakeProbe(
        {"cleanup": {"clean": False, "residue": {"packages": [SMART_NEVRA]}}}
    )
    ctx = context(tmp_path, deps_probe)
    with pytest.raises(RegionalFixtureError, match="deps cleanup left residue"):
        deps.checked_cleanup(ctx, "deps")


def test_the_package_deadman_fires_after_the_fixtures() -> None:
    assert deps.deps_deadman_seconds(2400) == 2400 + deps.DEPS_TIMEOUT + 300
    assert deps.deps_deadman_seconds(5000) == probe.MAX_DEADMAN_SECONDS
    assert deps.deps_deadman_seconds(960) > 960 + deps.DEPS_TIMEOUT, (
        "the install budget is covered even when the install took all of it"
    )
