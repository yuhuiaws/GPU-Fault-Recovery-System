"""Contract tests for the runner-side host source replay controller.

Nothing here runs kubectl or a node: ``HostSourceReplay`` is driven over a
``FakeProbe`` to prove what it sends (sha256 + base64 for every installed file,
the validated plan, the exact ``open-window`` flags per mode), that the extended
modes are gated on the ``HOST_SOURCE_REPLAY`` approval while the phase-1 binds
and the PCI-root repoint keep working without it, and that cleanup fails a
window that will not close clean.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from scripts.e2e.regional import general_live_source_replay as helper
from scripts.e2e.regional.general_live_registry import (
    APPROVALS,
    LIVE_REPLAY,
    VariantSpec,
    validate_spec,
)
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tools import host_command_replay as wrapper

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
TOKEN = helper.HOST_SOURCE_REPLAY


class _Regional:
    class settings:
        gpu_kubeconfig = Path("/tmp/kubeconfig")
        gpu_context = "ctx"
        namespace = "gpu-fault-system"


def _script(**extra: object) -> dict[str, object]:
    script: dict[str, object] = {
        "install-source": {"bytes": 1},
        "install-wrapper": {"shim": "/run/gfrs-live/source-x/bin/smartctl"},
        "build-infiniband-fixture": {"devices": {}},
        "build-nvidia-proc-fixture": {"gpus": []},
        "build-pci-fixture": {"efa_devices": 15},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
        },
        "set-value": {"value": "1"},
        "status": {"fixtures": {}},
        "verify": {"ok": True},
        "audit-read": {"entries": []},
        "close-window": {"clean": True},
    }
    script.update(extra)
    return script


def _controller(tmp_path: Path, approvals: tuple[str, ...] = (TOKEN,), **extra: object):
    fake = FakeProbe(_script(**extra))
    controller = helper.HostSourceReplay(
        _Regional(),
        node=NODE,
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-007.health_failed",
        run_id="gl007-health-failed-a1-deadbeef",
        case_dir=tmp_path / "case",
        host=fake,
        approvals=approvals,
    )
    return controller, fake


def _flags(call: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(call[1::2], call[2::2]))


# --------------------------------------------------------------------------- #
# the token
# --------------------------------------------------------------------------- #
def test_registry_carries_the_host_source_replay_token() -> None:
    meaning = APPROVALS[TOKEN]
    for phrase in (
        "gpu-fault-host-collector.service only",
        "/proc/meminfo",
        "/sys/class/infiniband",
        "/proc/driver/nvidia",
        "smartctl, ipmitool, ethtool",
        "nvidia-smi",
        "no host binary is replaced",
        "Node Agent is never touched",
        "deadman",
    ):
        assert phrase in meaning, f"the token's description names {phrase!r}"
    spec = VariantSpec(
        case_id="GF-GENERAL-LIVE-007",
        variant="health_failed",
        proof_mode=LIVE_REPLAY,
        lanes=("eks",),
        risk="L",
        injection="smartctl -H -j replay",
        expected="smart_health_failed",
        cleanup="window closed",
        max_minutes=20,
        approvals=(TOKEN,),
        execute=lambda ctx: {},
    )
    validate_spec(spec)


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #
def test_command_plan_builds_a_validated_plan() -> None:
    plan = helper.command_plan(
        "smartctl",
        [
            helper.rule(
                "health",
                "set_json",
                device="/dev/nvme0",
                field="smart_status.passed",
                value=False,
            )
        ],
        not_after_seconds=1200,
        now=1_000_000.0,
    )
    assert plan["tool"] == "smartctl" and plan["not_after"] == 1_001_200.0, plan
    assert plan["rules"][0]["field"] == "smart_status.passed", plan
    assert wrapper.validate_plan("smartctl", plan) == plan, (
        "the plan is already in validated shape"
    )
    with pytest.raises(helper.HostSourceReplayError, match="plan rejected"):
        helper.command_plan(
            "smartctl",
            [helper.rule("health", "set_json", field="model_name", value="x")],
        )
    with pytest.raises(
        helper.HostSourceReplayError, match="never in a replay allow-list"
    ):
        helper.command_plan("py-spy", [helper.rule("dump", "exit", exit_code=1)])
    with pytest.raises(helper.HostSourceReplayError, match="not_after_seconds"):
        helper.command_plan(
            "smartctl", [helper.rule("scan", "passthrough")], not_after_seconds=10
        )
    assert helper.TOOLS == ("smartctl", "ipmitool", "ethtool", "nvidia-smi")
    assert helper.TREE_DESTS == ("/sys/class/infiniband", "/proc/driver/nvidia")


# --------------------------------------------------------------------------- #
# the controller
# --------------------------------------------------------------------------- #
def test_install_wrapper_ships_tool_and_plan_with_sha256(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path)
    plan = helper.command_plan(
        "smartctl", [helper.rule("scan", "exit", exit_code=2)], now=1.0
    )
    receipts = controller.install_wrapper("smartctl", plan)
    installs = [call for call in fake.calls if call[0] == "install-source"]
    assert [_flags(call)["--name"] for call in installs] == [
        "host_command_replay.py",
        "plan-smartctl.json",
    ]
    tool_bytes = helper.WRAPPER_TOOL.read_bytes()
    assert _flags(installs[0])["--sha256"] == hashlib.sha256(tool_bytes).hexdigest(), (
        "the wrapper file travels with its sha256"
    )
    assert base64.b64decode(_flags(installs[0])["--b64"]) == tool_bytes
    sent_plan = json.loads(base64.b64decode(_flags(installs[1])["--b64"]))
    assert (
        sent_plan == plan
        and _flags(installs[1])["--sha256"]
        == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    )
    wrap = next(call for call in fake.calls if call[0] == "install-wrapper")
    assert wrap[1:] == ("--tool", "smartctl", "--run-id", controller.run_id), wrap
    assert receipts["wrapper"]["shim"].endswith("/bin/smartctl"), receipts
    controller.install_wrapper(
        "ipmitool",
        helper.command_plan(
            "ipmitool",
            [helper.rule("sensor", "replace_output", stdout="x | 1 | u | cr\n")],
            now=1.0,
        ),
        allow_missing_real=True,
    )
    installs = [call for call in fake.calls if call[0] == "install-source"]
    assert [_flags(call)["--name"] for call in installs] == [
        "host_command_replay.py",
        "plan-smartctl.json",
        "plan-ipmitool.json",
    ], "the tool file is shipped once per run"
    wrap = [call for call in fake.calls if call[0] == "install-wrapper"][-1]
    assert "--allow-missing-real" in wrap, wrap
    controller.replan(
        "smartctl",
        helper.command_plan("smartctl", [helper.rule("scan", "passthrough")], now=1.0),
    )
    assert [
        _flags(call)["--name"] for call in fake.calls if call[0] == "install-source"
    ][-1] == "plan-smartctl.json"
    with pytest.raises(helper.HostSourceReplayError, match="plan rejected"):
        controller.install_wrapper(
            "smartctl", {"tool": "smartctl", "not_after": 5.0, "rules": []}
        )


def test_open_window_sends_every_mode_flag(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path)
    controller.install("meminfo", b"MemTotal: 1 kB\n")
    controller.install_wrapper(
        "ethtool",
        helper.command_plan(
            "ethtool", [helper.rule("stats", "exit", exit_code=75)], now=1.0
        ),
    )
    controller.build_infiniband(
        drop=["rdmap3s0"],
        duplicate=[("rdmap9s0", "rdmap0s0")],
        port_state=[("rdmap1s0", "1", "1: DOWN")],
        phys_state=[("rdmap1s0", "1", "3: Disabled")],
        driver=[("rdmap2s0", "none")],
        counters=[("rdmap0s0", "1", "hw_counters/rx_drops", "5000")],
        remove_counters=[("rdmap0s0", "1", "hw_counters/tx_pkts")],
        no_ports=["rdmap4s0"],
    )
    built = next(call for call in fake.calls if call[0] == "build-infiniband-fixture")
    assert built[1:] == (
        "--source",
        "/sys/class/infiniband",
        "--drop",
        "rdmap3s0",
        "--duplicate",
        "rdmap9s0=rdmap0s0",
        "--port-state",
        "rdmap1s0:1:1: DOWN",
        "--phys-state",
        "rdmap1s0:1:3: Disabled",
        "--driver",
        "rdmap2s0:none",
        "--set",
        "rdmap0s0:1:hw_counters/rx_drops=5000",
        "--remove-counter",
        "rdmap0s0:1:hw_counters/tx_pkts",
        "--no-ports",
        "rdmap4s0",
        "--run-id",
        controller.run_id,
    ), built
    controller.build_nvidia_proc(omit_version=True, gpu_count=7)
    nv = next(call for call in fake.calls if call[0] == "build-nvidia-proc-fixture")
    assert nv[1:] == (
        "--source",
        "/proc/driver/nvidia",
        "--omit-version",
        "--gpu-count",
        "7",
        "--run-id",
        controller.run_id,
    )
    receipt = controller.open_window(
        binds=[("meminfo", "/proc/meminfo")],
        trees=[(helper.INFINIBAND_TREE, "/sys/class/infiniband")],
        wrappers=["ethtool"],
        inaccessible=["tool:smartctl", "/proc/driver/nvidia/gpus"],
        env={"GPU_FAULT_EXPECTED_GPU_COUNT": "7"},
        restore_seconds=1200,
    )
    assert receipt["proof_scope"] == "live-replay"
    opened = next(call for call in fake.calls if call[0] == "open-window")
    assert opened[1:] == (
        "--mode",
        "mixed",
        "--restore-seconds",
        "1200",
        "--bind",
        "meminfo:/proc/meminfo",
        "--tree",
        "infiniband:/sys/class/infiniband",
        "--wrapper",
        "ethtool",
        "--inaccessible",
        "tool:smartctl",
        "--inaccessible",
        "/proc/driver/nvidia/gpus",
        "--env",
        "GPU_FAULT_EXPECTED_GPU_COUNT=7",
        "--run-id",
        controller.run_id,
    ), opened
    assert ("open-window", 240) in fake.timeouts, (
        "a verified open gets the long exec timeout"
    )
    controller.set_tree_value(
        helper.INFINIBAND_TREE, "rdmap0s0/ports/1/hw_counters/rx_drops", 9000
    )
    stepped = [call for call in fake.calls if call[0] == "set-value"][-1]
    assert stepped[1:] == (
        "--name",
        "infiniband",
        "--path",
        "rdmap0s0/ports/1/hw_counters/rx_drops",
        "--value",
        "9000",
        "--run-id",
        controller.run_id,
    )
    controller.audit("ethtool", tail=20)
    audit = next(call for call in fake.calls if call[0] == "audit-read")
    assert audit[1:] == (
        "--tool",
        "ethtool",
        "--tail",
        "20",
        "--run-id",
        controller.run_id,
    )
    controller.verify()
    receipts = controller.cleanup()
    assert receipts["close"]["clean"] is True and not any(
        receipts["probe_residuals"].values()
    )


def test_open_command_and_open_tree_shapes(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path)
    controller.install_wrapper(
        "smartctl",
        helper.command_plan("smartctl", [helper.rule("scan", "passthrough")], now=1.0),
    )
    controller.open_command(["smartctl"], restore_seconds=900)
    opened = [call for call in fake.calls if call[0] == "open-window"][-1]
    assert opened[1:] == (
        "--mode",
        "command",
        "--restore-seconds",
        "900",
        "--wrapper",
        "smartctl",
        "--run-id",
        controller.run_id,
    ), opened
    controller.close()
    controller.open_tree(
        [(helper.NVIDIA_PROC_TREE, "/proc/driver/nvidia")],
        restore_seconds=600,
        verify=False,
    )
    opened = [call for call in fake.calls if call[0] == "open-window"][-1]
    assert opened[1:] == (
        "--mode",
        "tree",
        "--restore-seconds",
        "600",
        "--tree",
        "nvidia-proc:/proc/driver/nvidia",
        "--skip-verify",
        "--run-id",
        controller.run_id,
    ), opened
    with pytest.raises(helper.HostSourceReplayError, match="not allow-listed"):
        controller.open_window(binds=[("x", "/etc/passwd")])
    with pytest.raises(helper.HostSourceReplayError, match="not allow-listed"):
        controller.open_window(trees=[("x", "/sys/class/net")])
    with pytest.raises(
        helper.HostSourceReplayError, match="never in a replay allow-list"
    ):
        controller.open_window(wrappers=["py-spy"])
    with pytest.raises(helper.HostSourceReplayError, match="not an overridable"):
        controller.open_window(env={"GPU_FAULT_CLUSTER_TOKEN": "x"})


def test_phase_one_modes_need_no_token_but_extended_modes_do(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path, approvals=())
    assert controller.approved is False
    controller.install("proc-stat", b"cpu 1 0 1 1 0 0 0 0 0 0\n")
    controller.open_bind([("proc-stat", "/proc/stat")], restore_seconds=600)
    opened = [call for call in fake.calls if call[0] == "open-window"][-1]
    assert opened[1:] == (
        "--mode",
        "bind",
        "--restore-seconds",
        "600",
        "--bind",
        "proc-stat:/proc/stat",
        "--run-id",
        controller.run_id,
    ), opened
    controller.close()
    controller.install("rx-errors", b"abc")
    controller.open_bind(
        [("rx-errors", "/sys/class/net/gfrs1a2b/statistics/rx_errors")]
    )
    controller.set_value("rx-errors", 5000)
    controller.close()
    controller.build_pci(efa_count=15)
    controller.open_pci_root(restore_seconds=600)
    opened = [call for call in fake.calls if call[0] == "open-window"][-1]
    assert opened[1:] == (
        "--mode",
        "pci-root",
        "--restore-seconds",
        "600",
        "--pci-root",
        "--run-id",
        controller.run_id,
    ), opened
    controller.close()
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.open_bind([("meminfo", "/proc/meminfo")])
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.install_wrapper(
            "smartctl",
            helper.command_plan(
                "smartctl", [helper.rule("scan", "passthrough")], now=1.0
            ),
        )
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.build_infiniband()
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.build_nvidia_proc()
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.open_window(inaccessible=["/proc/meminfo"])
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.open_window(env={"GPU_FAULT_EXPECTED_GPU_COUNT": "7"})
    assert not any(call[0] == "install-wrapper" for call in fake.calls), (
        "a refused request never reaches the node"
    )


def test_cleanup_fails_a_window_that_will_not_close_clean(tmp_path: Path) -> None:
    controller, _fake = _controller(
        tmp_path, **{"close-window": {"clean": False, "residue": {"dropin": True}}}
    )
    controller.install("meminfo", b"x")
    controller.open_bind([("meminfo", "/proc/meminfo")])
    with pytest.raises(RuntimeError, match="did not close clean"):
        controller.cleanup()
    evidence = controller.evidence(tool="smartctl")
    assert (
        evidence["proof_scope"] == "live-replay"
        and evidence["physical_fault_injected"] is False
    ), evidence
    assert evidence["approved"] is True and evidence["tool"] == "smartctl", evidence


def test_host_source_replay_from_a_variant_context(tmp_path: Path) -> None:
    spec = VariantSpec(
        case_id="GF-GENERAL-LIVE-007",
        variant="health_failed",
        proof_mode=LIVE_REPLAY,
        lanes=("eks",),
        risk="L",
        injection="smartctl -H -j replay",
        expected="smart_health_failed",
        cleanup="window closed",
        max_minutes=20,
        approvals=(TOKEN,),
        execute=lambda ctx: {},
    )
    fake = FakeProbe(_script())
    regional = FakeRegional()
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": fake},
        settings=settings,
    )
    replay = helper.host_source_replay(ctx)
    assert replay.approved is True and replay.host is fake, (
        "the context's approvals and probe are used"
    )
    assert "source-replay" in ctx.cleanup.pending(), (
        "the window sits on the variant's LIFO cleanup stack"
    )
    unapproved = make_context(
        spec,
        tmp_path=tmp_path / "b",
        regional=regional,
        probes={f"source-replay@{NODE}": FakeProbe(_script())},
        settings=dataclasses.replace(settings, approvals=()),
    )
    assert helper.host_source_replay(unapproved).approved is False


# --------------------------------------------------------------------------- #
# the kernel collector's command window (KERNEL_COLLECTOR_SOURCE_REPLAY, 032)
# --------------------------------------------------------------------------- #
KERNEL_TOKEN = helper.KERNEL_COLLECTOR_SOURCE_REPLAY
VERSION_RULE = helper.rule("driver_version", "exit", exit_code=1, stderr="withheld\n")


def _kernel_controller(tmp_path: Path, approvals: tuple[str, ...], **extra: object):
    fake = FakeProbe(_script(**extra))
    controller = helper.HostSourceReplay(
        _Regional(),
        node=NODE,
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-032.missing_version",
        run_id="gl032-missing-version-a1-deadbeef",
        case_dir=tmp_path / "case",
        host=fake,
        approvals=approvals,
        unit=helper.KERNEL_UNIT,
    )
    return controller, fake


def test_registry_carries_the_kernel_collector_token() -> None:
    meaning = APPROVALS[KERNEL_TOKEN]
    for phrase in (
        "gpu-fault-kernel-collector.service",
        "driver-version discovery query only",
        "--query-gpu=driver_version",
        "GPU_FAULT_DRIVER_BRANCH is unset",
        "no file bind",
        "deadman",
        "Node Agent are never touched",
    ):
        assert phrase in meaning, f"the token's description names {phrase!r}"
    assert helper.UNIT_TOKENS == {
        helper.HOST_UNIT: TOKEN,
        helper.KERNEL_UNIT: KERNEL_TOKEN,
    }
    assert helper.UNIT_HANDLES[helper.KERNEL_UNIT] == "kernel-collector-replay"


def test_kernel_collector_window_needs_its_own_token_and_carries_one_query(
    tmp_path: Path,
) -> None:
    refused, fake = _kernel_controller(tmp_path, approvals=(TOKEN,))
    assert refused.unit == helper.KERNEL_UNIT and refused.token == KERNEL_TOKEN
    assert refused.approved is False, "HOST_SOURCE_REPLAY never unlocks the kernel unit"
    plan = helper.command_plan("nvidia-smi", [VERSION_RULE], now=1.0)
    with pytest.raises(helper.HostSourceReplayError, match=KERNEL_TOKEN):
        refused.install_wrapper("nvidia-smi", plan)
    with pytest.raises(helper.HostSourceReplayError, match=KERNEL_TOKEN):
        refused.open_command(["nvidia-smi"])
    assert fake.calls == [], "a refused request never reaches the node"

    controller, fake = _kernel_controller(tmp_path, approvals=(KERNEL_TOKEN,))
    assert controller.approved is True
    controller.install_wrapper("nvidia-smi", plan)
    installed = next(call for call in fake.calls if call[0] == "install-wrapper")
    assert installed[1:] == (
        "--tool",
        "nvidia-smi",
        "--unit",
        helper.KERNEL_UNIT,
        "--run-id",
        controller.run_id,
    ), installed
    controller.open_command(["nvidia-smi"], restore_seconds=1800)
    opened = next(call for call in fake.calls if call[0] == "open-window")
    assert opened[1:] == (
        "--mode",
        "command",
        "--restore-seconds",
        "1800",
        "--unit",
        helper.KERNEL_UNIT,
        "--wrapper",
        "nvidia-smi",
        "--run-id",
        controller.run_id,
    ), opened
    with pytest.raises(helper.HostSourceReplayError, match="may rewrite only"):
        controller.install_wrapper(
            "nvidia-smi",
            helper.command_plan(
                "nvidia-smi",
                [
                    VERSION_RULE,
                    helper.rule(
                        "utilization",
                        "drop_row",
                        uuid="GPU-11111111-2222-3333-4444-555555555555",
                    ),
                ],
                now=1.0,
            ),
        )
    with pytest.raises(helper.HostSourceReplayError, match="may not carry a smartctl"):
        controller.install_wrapper(
            "smartctl",
            helper.command_plan(
                "smartctl", [helper.rule("scan", "exit", exit_code=2)], now=1.0
            ),
        )
    for refusal in (
        lambda: controller.open_window(binds=[("meminfo", "/proc/meminfo")]),
        lambda: controller.open_window(
            wrappers=["nvidia-smi"], env={"GPU_FAULT_EXPECTED_GPU_COUNT": "7"}
        ),
        lambda: controller.open_window(
            wrappers=["nvidia-smi"], inaccessible=["/proc/meminfo"]
        ),
        lambda: controller.open_pci_root(),
        lambda: controller.build_meminfo(SwapTotal=1),
        lambda: controller.build_nvidia_proc(),
        lambda: controller.build_infiniband(),
        lambda: controller.build_pci(efa_count=15),
        lambda: controller.read_back("/proc/driver/nvidia/version"),
        lambda: controller.set_value("meminfo", 1),
    ):
        with pytest.raises(
            helper.HostSourceReplayError, match="command-output replay only"
        ):
            refusal()
    from datetime import datetime, timezone

    since = datetime(2026, 10, 25, 8, tzinfo=timezone.utc)
    controller.journal(since=since, tail=50)
    journal = next(call for call in fake.calls if call[0] == "journal-read")
    assert journal[1:] == (
        "--unit",
        helper.KERNEL_UNIT,
        "--since",
        since.isoformat(),
        "--tail",
        "50",
        "--run-id",
        controller.run_id,
    ), journal
    evidence = controller.evidence(tool="nvidia-smi")
    assert evidence["unit"] == helper.KERNEL_UNIT and evidence["approved"] is True
    assert evidence["proof_scope"] == "live-replay"


def test_kernel_collector_token_does_not_unlock_the_host_collector(
    tmp_path: Path,
) -> None:
    controller, fake = _controller(tmp_path, approvals=(KERNEL_TOKEN,))
    assert controller.unit == helper.HOST_UNIT and controller.approved is False
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.install_wrapper(
            "smartctl",
            helper.command_plan(
                "smartctl", [helper.rule("scan", "passthrough")], now=1.0
            ),
        )
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        controller.open_bind([("meminfo", "/proc/meminfo")])
    assert not any(
        call[0] in {"install-wrapper", "open-window"} for call in fake.calls
    ), "a refused request never reaches the node"
    with pytest.raises(helper.HostSourceReplayError, match="Node Agent"):
        helper.HostSourceReplay(
            _Regional(),
            node=NODE,
            image="img@sha256:" + "0" * 64,
            case_id="x",
            run_id="gl032-x-a1-deadbeef",
            case_dir=tmp_path,
            host=fake,
            unit=wrapper.NODE_AGENT_UNIT,
        )


def test_kernel_collector_source_replay_from_a_variant_context(tmp_path: Path) -> None:
    spec = VariantSpec(
        case_id="GF-GENERAL-LIVE-032",
        variant="missing_version",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="M",
        injection="driver_version withheld",
        expected="BLOCKED_MISSING_EVIDENCE",
        cleanup="window closed",
        max_minutes=30,
        approvals=(KERNEL_TOKEN, "KERNEL_SOURCE_REPLAY"),
        execute=lambda ctx: {},
    )
    fake = FakeProbe(_script())
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(),
        probes={f"kernel-collector-replay@{NODE}": fake},
        settings=settings,
    )
    replay = helper.kernel_collector_source_replay(ctx)
    assert replay.unit == helper.KERNEL_UNIT and replay.host is fake
    assert replay.approved is True
    assert "kernel-collector-replay" in ctx.cleanup.pending()
    assert "source-replay" not in ctx.cleanup.pending()


# --------------------------------------------------------------------------- #
# read-only inventory / read-back and the netdev tree mutation (010, 011)
# --------------------------------------------------------------------------- #
def test_inventory_and_read_back_need_no_token(tmp_path: Path) -> None:
    controller, fake = _controller(
        tmp_path,
        approvals=(),
        inventory={"devices": {"rdmap79s0": {"efa": True}}},
        **{
            "read-source": {
                "path": "/sys/class/infiniband/rdmap79s0/ports/1/state",
                "same": True,
            }
        },
    )
    assert controller.inventory()["devices"] == {"rdmap79s0": {"efa": True}}
    dest = "/sys/class/infiniband/rdmap79s0/ports/1/state"
    assert controller.read_back(dest)["same"] is True
    assert fake.calls[0][:3] == ("inventory", "--source", "/sys/class/infiniband")
    assert fake.calls[1][:3] == ("read-source", "--path", dest)
    with pytest.raises(helper.HostSourceReplayError, match="not allow-listed"):
        controller.read_back("/etc/passwd")


def test_build_infiniband_attaches_a_netdev(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path)
    controller.build_infiniband(netdev=[("rdmap79s0", "ens5")], drop=["rdmap80s0"])
    call = fake.calls[-1]
    assert call[0] == "build-infiniband-fixture"
    assert "--netdev" in call and call[call.index("--netdev") + 1] == "rdmap79s0:ens5"
    assert call[call.index("--drop") + 1] == "rdmap80s0"


def test_build_meminfo_sends_allow_listed_fields_only(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path)
    controller.build_meminfo(SwapTotal=8388608, SwapFree=1048576)
    built = next(call for call in fake.calls if call[0] == "build-meminfo-fixture")
    assert built[1:] == (
        "--source",
        "/proc/meminfo",
        "--set",
        "SwapTotal=8388608",
        "--set",
        "SwapFree=1048576",
        "--run-id",
        controller.run_id,
    ), built
    with pytest.raises(helper.HostSourceReplayError, match="meminfo fields"):
        controller.build_meminfo(MemTotal=1)
    with pytest.raises(helper.HostSourceReplayError, match="meminfo fields"):
        controller.build_meminfo()
    unapproved, _fake = _controller(tmp_path / "b", approvals=())
    with pytest.raises(helper.HostSourceReplayError, match=TOKEN):
        unapproved.build_meminfo(SwapTotal=1)
    assert helper.MEMINFO_FIXTURE == "meminfo"
    assert helper.MEMINFO_FIELDS == ("MemFree", "MemAvailable", "SwapTotal", "SwapFree")


def test_build_commands_ship_the_builders_once_before_the_first_build(
    tmp_path: Path,
) -> None:
    controller, fake = _controller(tmp_path)
    controller.inventory()
    controller.read_back("/proc/meminfo")
    controller.build_meminfo(SwapTotal=8388608, SwapFree=1048576)
    controller.build_infiniband(drop=["rdmap80s0"])
    controller.build_pci(efa_count=2)
    assert [call[0] for call in fake.calls] == [
        "inventory",
        "read-source",
        "install-source",
        "build-meminfo-fixture",
        "build-infiniband-fixture",
        "build-pci-fixture",
    ], "read-only commands need no builders; one install precedes the first build"
    shipped = fake.calls[2]
    content = helper.BUILDERS_MODULE.read_bytes()
    assert shipped[shipped.index("--name") + 1] == probe.BUILDERS_NAME, shipped
    assert (
        shipped[shipped.index("--sha256") + 1] == hashlib.sha256(content).hexdigest()
    ), "the builders are sha256-checked like the wrapper"
    assert base64.b64decode(shipped[shipped.index("--b64") + 1]) == content, (
        "the checkout's module is what ships"
    )
    assert helper.BUILDERS_MODULE.name == probe.BUILDERS_NAME, helper.BUILDERS_MODULE


# --------------------------------------------------------------------------- #
# PCI root shapes, snapshot and re-create after a reboot (013 pci_missing)
# --------------------------------------------------------------------------- #
def test_build_pci_sends_one_reviewed_shape_per_call(tmp_path: Path) -> None:
    controller, fake = _controller(tmp_path, approvals=())
    controller.build_pci(efa_count=16, malformed=True)
    controller.build_pci(
        source="/sys/bus/pci/devices", drop=["0000:53:00.0", "0000:53:01.0"]
    )
    controller.build_pci(unreadable=True)
    builds = [call for call in fake.calls if call[0] == "build-pci-fixture"]
    assert builds[0][1:5] == ("--efa-count", "16", "--malformed", "--run-id"), builds
    assert builds[1][1:7] == (
        "--source",
        "/sys/bus/pci/devices",
        "--drop",
        "0000:53:00.0",
        "--drop",
        "0000:53:01.0",
    ), builds[1]
    assert builds[2][1:3] == ("--unreadable", "--run-id"), builds[2]
    installs = [call for call in fake.calls if call[0] == "install-source"]
    assert len(installs) == 1, "the builders ship once before the first build"
    for kwargs in (
        {},
        {"efa_count": 1, "unreadable": True},
        {"source": "/x", "unreadable": True},
    ):
        with pytest.raises(helper.HostSourceReplayError, match="exactly one"):
            controller.build_pci(**kwargs)
    # The PCI-root repoint stays a phase-1 mode: no token is needed to open it.
    controller.open_pci_root(restore_seconds=600)
    assert [call for call in fake.calls if call[0] == "open-window"], fake.calls


def test_snapshot_and_recreate_after_a_reboot(tmp_path: Path) -> None:
    class RebootedProbe(FakeProbe):
        def __init__(self, script: dict[str, object]) -> None:
            super().__init__(script)
            self.events: list[str] = []

        def cleanup(self) -> dict[str, bool]:
            self.events.append("cleanup")
            return {"pod": False, "configmap": False}

        def create(self) -> None:
            self.events.append("create")

    fake = RebootedProbe(
        _script(
            snapshot={
                "window_open": False,
                "dropin": False,
                "state_dir": False,
                "deadman": False,
                "unit_active_state": "active",
                "unit_pci_root": None,
            }
        )
    )
    controller = helper.HostSourceReplay(
        _Regional(),
        node=NODE,
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-013.pci_missing",
        run_id="gl013-pci-missing-a1-deadbeef",
        case_dir=tmp_path / "case",
        host=fake,
        approvals=(TOKEN,),
    )
    controller.build_pci(unreadable=True)
    controller.open_pci_root(restore_seconds=600)
    receipt = controller.recreate()
    assert receipt["recreated"] is True and fake.events == ["cleanup", "create"]
    snapshot = controller.snapshot()
    assert snapshot["dropin"] is False and snapshot["unit_pci_root"] is None
    # Nothing is open on the new boot: cleanup closes no window, and the
    # builders -- gone with /run -- are shipped again before the next build.
    controller.cleanup()
    assert "close-window" not in [call[0] for call in fake.calls], fake.calls
    controller.build_pci(efa_count=16)
    installs = [call for call in fake.calls if call[0] == "install-source"]
    assert len(installs) == 2, "the builders are shipped again after the reboot"
