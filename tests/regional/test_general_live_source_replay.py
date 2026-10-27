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
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    StatefulSourceProbe,
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
    # The read-only oracle commands live in the builders file (the probe is one
    # file under the size limit), so the builders are shipped first, once.
    assert fake.calls[0][0] == "install-source", fake.calls[0]
    assert fake.calls[1][:3] == ("inventory", "--source", "/sys/class/infiniband")
    assert fake.calls[2][:3] == ("read-source", "--path", dest)
    controller.status()
    assert [call[0] for call in fake.calls].count("install-source") == 1
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
        "install-source",
        "inventory",
        "read-source",
        "build-meminfo-fixture",
        "build-infiniband-fixture",
        "build-pci-fixture",
    ], (
        "one install of the builders precedes the first command that needs them "
        "(inventory lives there; read-source and the window never need it)"
    )
    shipped = fake.calls[0]
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


# --------------------------------------------------------------------------- #
# the state dir leaves with the window (013 pci_missing, HyperPod 2026-10-27)
# --------------------------------------------------------------------------- #
def _stateful(tmp_path: Path, **extra: object):
    fake = StatefulSourceProbe(_script(**extra))
    controller = helper.HostSourceReplay(
        _Regional(),
        node=NODE,
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-013.pci_missing",
        run_id="gl013-pci-missing-a6-471be317",
        case_dir=tmp_path / "case",
        host=fake,
        approvals=(TOKEN,),
    )
    return controller, fake


def test_the_stateful_fake_refuses_what_the_live_probe_refuses() -> None:
    """The fake is only worth driving if it refuses a build without the builders
    (per run id, like ``fixtures_dir(run_id)``) and forgets them on a close."""

    fake = StatefulSourceProbe(_script())
    build = ("build-pci-fixture", "--unreadable", "--run-id", "gl013-x")
    with pytest.raises(HostProbeError, match="install general_live_source_builders"):
        fake.execute(*build)
    with pytest.raises(HostProbeError, match="install host_command_replay.py"):
        fake.execute("install-wrapper", "--tool", "smartctl", "--run-id", "gl013-x")
    fake.execute(
        "install-source",
        "--name",
        probe.BUILDERS_NAME,
        "--sha256",
        "0" * 64,
        "--b64",
        "",
        "--run-id",
        "gl013-x",
    )
    assert fake.execute(*build) == {"efa_devices": 15}
    with pytest.raises(HostProbeError, match="install general_live_source_builders"):
        fake.execute("build-pci-fixture", "--unreadable", "--run-id", "gl013-x-r2")
    fake.execute("close-window", "--run-id", "gl013-x")
    assert fake.wipes == [("close-window", "gl013-x")]
    with pytest.raises(HostProbeError, match="install general_live_source_builders"):
        fake.execute(*build)


def test_builders_ship_again_after_the_close_removed_the_state_dir(
    tmp_path: Path,
) -> None:
    """013 ``pci_missing``'s order on a node-faithful probe (HyperPod 2026-10-27).

    ``inventory`` ships the builders once and the unreadable-root build rides on
    them; ``close-window`` removes ``state_dir(run_id)`` -- fixtures, wrapper,
    builders -- on the node; the missing-function build must ship them again.
    Red before the fix: the once-per-run flag survived the close, the install
    was skipped and the probe refused ``build-pci-fixture`` ("install
    general_live_source_builders.py before building a fixture",
    ``load_builders_module``), ending the variant INCONCLUSIVE.
    """

    controller, fake = _stateful(tmp_path)
    controller.inventory(source="/sys/class/infiniband")
    controller.build_pci(unreadable=True)
    controller.open_pci_root(restore_seconds=600)
    controller.close()
    controller.build_pci(source="/sys/bus/pci/devices", drop=["0000:53:00.0"])
    controller.build_infiniband(source="/sys/class/infiniband", drop=["rdmap0s0"])
    assert [call[0] for call in fake.calls] == [
        "install-source",
        "inventory",
        "build-pci-fixture",
        "open-window",
        "close-window",
        "install-source",
        "build-pci-fixture",
        "build-infiniband-fixture",
    ], (
        "the builders ship once per node state: before the first command that "
        "needs them and again after the close removed them -- not per build"
    )
    digest = hashlib.sha256(helper.BUILDERS_MODULE.read_bytes()).hexdigest()
    for call in (call for call in fake.calls if call[0] == "install-source"):
        flags = _flags(call)
        assert flags["--name"] == probe.BUILDERS_NAME, call
        assert flags["--sha256"] == digest, "sha256-checked on every install"
        assert flags["--run-id"] == controller.run_id, (
            "install and build name the same run: the probe keeps both under "
            "fixtures_dir(run_id)"
        )
    assert fake.wipes == [("close-window", controller.run_id)]


def test_await_ticks_close_and_an_already_closed_window_forget_the_installed_files(
    tmp_path: Path,
) -> None:
    """``await-ticks --close`` restores on the node (013 ``link_inactive``); a
    window the deadman closed meanwhile answers ``already_closed``. Both removed
    the state dir, so the next build ships the builders again."""

    for receipt in (
        {"completed": True, "ticks": [1, 2], "close": {"clean": True}},
        {"already_closed": True, "completed": False, "ticks": []},
    ):
        controller, fake = _stateful(tmp_path, **{"await-ticks": receipt})
        controller.build_infiniband(source="/sys/class/infiniband", drop=["rdmap0s0"])
        controller.open_tree(
            [(helper.INFINIBAND_TREE, "/sys/class/infiniband")], restore_seconds=300
        )
        controller.await_ticks(2, interval=15, timeout=120)
        controller.build_infiniband(source="/sys/class/infiniband", drop=["rdmap1s0"])
        installs = [call for call in fake.calls if call[0] == "install-source"]
        assert len(installs) == 2, (receipt, [call[0] for call in fake.calls])
        assert fake.wipes == [("await-ticks", controller.run_id)], receipt
        controller.cleanup()
        assert "close-window" not in [call[0] for call in fake.calls], (
            "the window was already closed on the node; cleanup closes nothing"
        )


def test_the_wrapper_ships_again_after_a_close(tmp_path: Path) -> None:
    """A second command window after a close: the wrapper left with the state dir
    ("install host_command_replay.py before wrapping a tool" otherwise)."""

    controller, fake = _stateful(tmp_path)
    plan = helper.command_plan(
        "smartctl", [helper.rule("scan", "exit", exit_code=2)], now=1.0
    )
    controller.install_wrapper("smartctl", plan)
    controller.open_command(["smartctl"], restore_seconds=600)
    controller.close()
    controller.install_wrapper("smartctl", plan)
    controller.open_command(["smartctl"], restore_seconds=600)
    names = [
        _flags(call)["--name"] for call in fake.calls if call[0] == "install-source"
    ]
    assert names == [
        probe.WRAPPER_NAME,
        "plan-smartctl.json",
        probe.WRAPPER_NAME,
        "plan-smartctl.json",
    ], names
    assert [call[0] for call in fake.calls].count("install-wrapper") == 2


class _Cadence:
    """The cadence surface ``exposure_fallback_seconds``/``observed_exposure`` read."""

    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.written: dict[str, object] = {}
        self.evidence = self

    def write(self, name: str, document: object) -> None:
        self.written[name] = document

    def setting(self, key: str, default: int) -> int:
        value = self.env.get(key)
        return int(value) if value not in (None, "") else default

    @property
    def host_interval(self) -> int:
        return self.setting("GPU_FAULT_HOST_INTERVAL_SECONDS", 15)


def test_exposure_fallback_follows_the_collectors_cadence() -> None:
    """The deadman of a bounded exposure is the fallback: after the collector's
    restart phase (startup spread + start-up jitter) and the required ticks,
    each with its own duration, plus the probe's close. A fixed 1.7-interval
    bound from the restart is gone (live 2026-10-26: first tick 13 s in)."""

    ctx = _Cadence({})
    assert helper.confirmation_ticks(ctx) == 2, "two consecutive samples by default"
    assert (
        helper.exposure_fallback_seconds(ctx, 2) == 15 + 20 + 2 * (15 + 10) + 20 == 105
    )
    ctx = _Cadence(
        {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "30",
            "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES": "3",
            "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS": "5",
            "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS": "0",
        }
    )
    assert helper.confirmation_ticks(ctx) == 3
    assert helper.exposure_fallback_seconds(ctx, 3) == 5 + 0 + 3 * (30 + 10) + 20 == 145
    assert (
        helper.exposure_fallback_seconds(
            _Cadence({"GPU_FAULT_HOST_INTERVAL_SECONDS": "3600"}), probe.MAX_TICKS
        )
        == probe.MAX_RESTORE_SECONDS
    ), "never past the deadman's ceiling"
    assert (
        helper.confirmation_ticks(
            _Cadence({"GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES": "99"})
        )
        == probe.MAX_TICKS
    )
    assert not hasattr(helper, "bounded_exposure_seconds"), (
        "the fixed bound from the restart is retired"
    )


def _ticks(count: int) -> list[dict[str, object]]:
    return [
        {
            "pings_seen": 17,
            "first_ping_at": f"2026-10-26T03:34:{40 + 15 * index:02d}+00:00",
            "last_ping_at": f"2026-10-26T03:34:{43 + 15 * index:02d}+00:00",
        }
        for index in range(count)
    ]


def test_observed_exposure_closes_on_the_probes_tick_count(tmp_path: Path) -> None:
    """``await-ticks --close`` on the node is the restore; the record carries the
    ticks the probe saw, when the real source was read again and who closed."""

    close = {
        "clean": True,
        "closed_at": "2026-10-26T03:35:10+00:00",
        "restore": {"ok": True, "checks": [{"kind": "source", "ok": True}]},
        "after": {"ActiveState": "active"},
    }
    controller, fake = _controller(
        tmp_path,
        **{
            "await-ticks": {
                "completed": True,
                "unit_active_at": "2026-10-26T03:34:30+00:00",
                "ticks": _ticks(2),
                "close": close,
            }
        },
    )
    controller.open_tree([("infiniband", "/sys/class/infiniband")], restore_seconds=105)
    ctx = _Cadence({})
    record = helper.observed_exposure(ctx, controller, ticks=2, name="link-inactive")
    assert record["unit_active_at"] == "2026-10-26T03:34:30+00:00", (
        "the receipt's activation stamp rides along (the ticks are counted after it)"
    )
    call = next(call for call in fake.calls if call[0] == "await-ticks")
    assert call[:8] == (
        "await-ticks",
        "--count",
        "2",
        "--interval",
        "15",
        "--timeout",
        "105",
        "--close",
    ), call
    assert dict(fake.timeouts)["await-ticks"] == 105 + 120, (
        "the close's own time on top"
    )
    assert record["closed_by"] == "probe" and record["completed"] is True
    assert record["closed_at"] == "2026-10-26T03:35:10+00:00"
    assert record["ticks_observed"] == 2 and record["ticks_required"] == 2
    assert record["restore"] == close["restore"] and record["clean"] is True
    assert ctx.written["link-inactive-exposure"]["receipt"]["completed"] is True
    receipts = controller.cleanup()
    assert "close" not in receipts, "the window is closed; cleanup closes nothing twice"
    # The fallback deadman got there first: nothing to wait for, nothing to close.
    controller, fake = _controller(
        tmp_path,
        **{"await-ticks": {"already_closed": True, "completed": False, "ticks": []}},
    )
    controller.open_tree([("infiniband", "/sys/class/infiniband")], restore_seconds=105)
    record = helper.observed_exposure(ctx, controller, ticks=2, name="link-inactive")
    assert record["closed_by"] == "deadman" and record["completed"] is False
    assert record["clean"] is True and record["ticks_observed"] == 0
    assert "close" not in controller.cleanup()
    # A close that left residue fails the variant, like cleanup does.
    controller, fake = _controller(
        tmp_path,
        **{
            "await-ticks": {
                "completed": True,
                "ticks": _ticks(2),
                "close": {"clean": False},
            }
        },
    )
    controller.open_tree([("infiniband", "/sys/class/infiniband")], restore_seconds=105)
    with pytest.raises(helper.HostSourceReplayError, match="did not close clean"):
        helper.observed_exposure(ctx, controller, ticks=2, name="link-inactive")


def test_exposure_reading_gaps_name_the_missing_reading() -> None:
    """The live eks 013 run of 2026-10-26: the probe's tick 1 (18:59:03.1Z) was
    the restarted unit's READY=1 stamp -- no batch of it exists -- and the one
    shortfall batch (observed 18:59:13.83Z, the tick's start, before its pings
    13.98-14.16Z) belongs to tick 2; the close at 18:59:22.6Z came before the
    second confirming read. The lines say so; a batch without the shortfall, a
    shortfall observed after the close and another device's sample are told
    apart."""

    def batch(observed_at: str, value: float, device: str | None = None) -> dict:
        return {
            "observed_at": observed_at,
            "samples": [
                {
                    "name": "efa_inventory_missing_count",
                    "device": device,
                    "value": value,
                }
            ],
        }

    def tick(first: str, last: str, pings: int = 2) -> dict:
        return {"pings_seen": pings, "first_ping_at": first, "last_ping_at": last}

    live = {
        "closed_at": "2026-10-26T18:59:22.611068+00:00",
        "ticks": [
            tick(
                "2026-10-26T18:59:03.103206+00:00",
                "2026-10-26T18:59:03.103206+00:00",
                1,
            ),
            tick(
                "2026-10-26T18:59:13.978503+00:00", "2026-10-26T18:59:14.157629+00:00"
            ),
        ],
    }
    assert helper.exposure_reading_gaps(
        live,
        [batch("2026-10-26T18:59:13.829889+00:00", 1)],
        metric="efa_inventory_missing_count",
        required=2,
        tolerance_seconds=5,
    ) == [
        "tick 1 ended 2026-10-26T18:59:03.103206+00:00: no batch of it reached the "
        "product -- not a collector read, or its batch was never delivered",
        "1 of the 2 confirming efa_inventory_missing_count readings never happened "
        "before the close at 2026-10-26T18:59:22.611068+00:00",
    ]
    # The fixed probe's receipt: the activation named, real ticks only.
    fixed = {
        "closed_at": "2026-10-26T18:59:36.900000+00:00",
        "unit_active_at": "2026-10-26T18:59:03.103236+00:00",
        "ticks": [
            tick(
                "2026-10-26T18:59:13.978503+00:00", "2026-10-26T18:59:14.157629+00:00"
            ),
            tick(
                "2026-10-26T18:59:28.900000+00:00", "2026-10-26T18:59:29.100000+00:00"
            ),
        ],
    }
    first, second = (
        batch("2026-10-26T18:59:13.829889+00:00", 1),
        batch("2026-10-26T18:59:28.800000+00:00", 0),
    )
    assert helper.exposure_reading_gaps(
        fixed, [first, second], metric="efa_inventory_missing_count", required=2
    ) == [
        "tick 2 ended 2026-10-26T18:59:29.100000+00:00 (26.0 s after the unit became "
        "active): its batch carries no efa_inventory_missing_count >= 1 -- the "
        "collector read no shortfall on it",
        "1 of the 2 confirming efa_inventory_missing_count readings never happened "
        "before the close at 2026-10-26T18:59:36.900000+00:00",
    ]
    second["samples"][0]["value"] = 1
    late = batch("2026-10-26T18:59:44+00:00", 1)
    assert helper.exposure_reading_gaps(
        fixed,
        [first, second, late],
        metric="efa_inventory_missing_count",
        required=2,
        tolerance_seconds=5,
    ) == [
        "1 efa_inventory_missing_count reading(s) observed after the close at "
        "2026-10-26T18:59:36.900000+00:00 (first 2026-10-26T18:59:44+00:00): read after "
        "the restore, outside the bound"
    ], "both confirming readings happened; only the late one is named"
    assert (
        helper.exposure_reading_gaps(
            fixed, [first, second], metric="efa_inventory_missing_count", required=2
        )
        == []
    )
    # Another device's sample is not this port's reading.
    port = [
        batch("2026-10-26T18:59:13.829889+00:00", 1, "rdmap110s0/1"),
        batch("2026-10-26T18:59:28.800000+00:00", 1, "rdmap111s0/1"),
    ]
    lines = helper.exposure_reading_gaps(
        fixed,
        port,
        metric="efa_inventory_missing_count",
        required=2,
        device="rdmap110s0/1",
    )
    assert len(lines) == 2 and lines[0].startswith("tick 2 ended "), lines
    assert "read no shortfall on it" in lines[0], lines
    assert helper.exposure_reading_gaps({}, [], metric="x", required=2) == [
        "2 of the 2 confirming x readings never happened before the close"
    ], "a deadman close without a receipt still says what is missing"
