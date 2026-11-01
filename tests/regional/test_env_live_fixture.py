"""Pure parts of env_live_fixture (device selection, plans, preflight) with fakes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional import env_live_fixture as fixture
from scripts.e2e.regional.regional_commands import RegionalFixtureError

DEVICES = [
    {
        "gpu_uuid": "GPU-aaaa",
        "gpu_index": "0",
        "product": "NVIDIA H200",
        "limits": {
            "gpu_tlimit_c": 59.0,
            "gpu_slowdown_tlimit_c": -2.0,
            "gpu_shutdown_tlimit_c": -8.0,
        },
    },
    {
        "gpu_uuid": "GPU-bbbb",
        "gpu_index": "1",
        "product": "NVIDIA H200",
        "limits": {"gpu_tlimit_c": 60.0, "gpu_slowdown_tlimit_c": -2.0},
    },
]
REAL = "http://127.0.0.1:9400/metrics"
PROXY = "http://127.0.0.1:9411/metrics"
DROPIN = (
    "/run/systemd/system/gpu-fault-metrics-collector.service.d/"
    "gpu-fault-env-replay-0123456789abcdef.conf"
)
#: The probe's proven repoint receipt (live shape, 2026-10-26): the unit
#: restarted, its process reads the proxy, the proxy counted the first read.
PROVEN_REPOINT: dict[str, Any] = {
    "repointed": True,
    "run_id": "r1",
    "unit": "gpu-fault-metrics-collector.service",
    "original_endpoint": REAL,
    "proxy_endpoint": PROXY,
    "dropin": DROPIN,
    "process_environment": {fixture.ENDPOINT_ENV: PROXY},
    "first_read": {
        "observed": True,
        "reads_before": 0,
        "reads_after": 1,
        "waited_seconds": 18.1,
        "budget_seconds": 65,
    },
    "after": {
        "ActiveState": "active",
        "MainPID": "3798726",
        "InvocationID": "52ac4fabb52d4b77beb6d0e928da732e",
        "ExecMainStartTimestamp": "Mon 2026-10-26 12:50:14 UTC",
        "DropInPaths": DROPIN,
    },
}


def test_select_target_gpu_prefers_requested_then_lowest_index() -> None:
    assert fixture.select_target_gpu(DEVICES, "GPU-bbbb")["gpu_uuid"] == "GPU-bbbb"
    assert fixture.select_target_gpu(DEVICES, "")["gpu_uuid"] == "GPU-aaaa"
    with pytest.raises(RegionalFixtureError, match="not on the node"):
        fixture.select_target_gpu(DEVICES, "GPU-zzzz")
    with pytest.raises(RegionalFixtureError, match="no GPU device limits"):
        fixture.select_target_gpu([], "")
    with pytest.raises(RegionalFixtureError, match="report no UUID"):
        fixture.select_target_gpu([{"gpu_index": "0", "limits": {}}], "")


def test_build_device_plans_keyed_by_uuid() -> None:
    built = fixture.build_device_plans(
        DEVICES, thresholds=GpuMetricsThresholds(), confirmation_samples=3
    )
    assert set(built) == {"GPU-aaaa", "GPU-bbbb"}
    report = fixture.device_plan_report(built["GPU-aaaa"])
    assert report["runnable_variants"]
    assert "core_tlimit" in report["decisions"]


def test_replay_tool_sha256_matches_the_repo_file() -> None:
    expected = hashlib.sha256(fixture.REPLAY_TOOL.read_bytes()).hexdigest()
    assert fixture.replay_tool_sha256() == expected


def test_write_plan_file_round_trips(tmp_path: Path) -> None:
    plan = {"steps": [{"reads": 1, "values": {}}]}
    path = fixture.write_plan_file(tmp_path, fixture.PLAN_FILE, plan)
    assert json.loads(path.read_text()) == plan


class FakeRegional:
    """A read-only stand-in for RegionalLiveFixture used by the preflight."""

    def __init__(self, *, device_xml: str, clean: bool = True) -> None:
        self.device_xml = device_xml
        self.clean = clean
        self.settings = type(
            "S",
            (),
            {
                "gpu_kubeconfig": None,
                "gpu_context": "",
                "namespace": "ns",
                "cluster_id": "c",
            },
        )()

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {
            "name": node,
            "boot_id": "boot-1",
            "ready": "True",
            "ownership_annotations": {}
            if self.clean
            else {"gpu-fault.io/incident-id": "i"},
        }

    def store_snapshot(self, **_options: Any) -> dict[str, Any]:
        return {
            "agent": {"lifecycle_state": "ACTIVE" if self.clean else "REVOKED"},
            "collector_statuses": [{"collector": "GPU_METRICS", "sample_count": 5}],
            "incident": None,
        }

    def business_workloads(self, node: str) -> list[dict[str, str]]:
        return [] if self.clean else [{"name": "biz"}]

    def evidence_identity(self, **_: Any) -> dict[str, str]:
        return {"release_id": "rel-1", "cluster_id": "c"}

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {}

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if "exec" in args:
            return self.device_xml
        # pod lookup for the exporter
        return json.dumps(
            {"items": [{"metadata": {"name": "gpu-fault-dcgm-exporter-x"}}]}
        )


H200_XML = """<nvidia_smi_log><gpu><product_name>NVIDIA H200</product_name>
<minor_number>0</minor_number><uuid>GPU-aaaa</uuid>
<pci><pci_bus_id>00000000:59:00.0</pci_bus_id></pci>
<temperature><gpu_temp>28 C</gpu_temp><gpu_temp_tlimit>59 C</gpu_temp_tlimit>
<gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>
<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>
<gpu_temp_max_gpu_tlimit_threshold>0 C</gpu_temp_max_gpu_tlimit_threshold>
<gpu_temp_max_mem_tlimit_threshold>0 C</gpu_temp_max_mem_tlimit_threshold></temperature></gpu></nvidia_smi_log>"""


def test_read_device_limits_parses_with_the_product_parser() -> None:
    regional = FakeRegional(device_xml=H200_XML)
    devices, source = fixture.read_device_limits(regional, "node-a")
    assert devices and devices[0]["gpu_uuid"] == "GPU-aaaa"
    assert devices[0]["limits"]["gpu_slowdown_tlimit_c"] == -2.0
    assert source == "exporter-pod"


def _settings() -> fixture.EnvLiveSettings:
    regional = type("R", (), {"environment": lambda self: {}})()
    return fixture.EnvLiveSettings(
        regional=regional,
        case_id="GF-ENV-LIVE-004",
        confirmation="ENVLIVE004_EXECUTE",
        node="node-a",
        host_probe_image="x@sha256:" + "0" * 64,
        gpu_uuid="GPU-aaaa",
        predecessor_path=None,
        predecessor_id=None,
        site_file=None,
    )


def test_read_only_preflight_clean_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fixture, "RegionalLiveFixture", lambda s: FakeRegional(device_xml=H200_XML)
    )
    result = fixture.read_only_preflight(_settings(), tmp_path)
    assert result["errors"] == []
    assert result["target_gpu"]["gpu_uuid"] == "GPU-aaaa"
    assert "core_tlimit" in result["device_plans"]["GPU-aaaa"]["decisions"]


def test_read_only_preflight_dirty_node_collects_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fixture,
        "RegionalLiveFixture",
        lambda s: FakeRegional(device_xml=H200_XML, clean=False),
    )
    result = fixture.read_only_preflight(_settings(), tmp_path)
    text = "\n".join(result["errors"])
    assert "workflow ownership" in text and "business workload" in text
    assert "Node Agent is not ACTIVE" in text


class FakeHostProbe:
    """Records the controller's host-probe argv instead of touching a node."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.calls: list[tuple[str, ...]] = []
        self.created = False
        self.cleaned = False

    def create(self) -> None:
        self.created = True

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append(arguments)
        if arguments[0] == "install-source":
            return {"name": arguments[arguments.index("--name") + 1]}
        if arguments[0] == "deploy-proxy":
            return {"status": {"armed": False}}
        if arguments[0] == "repoint-collector":
            # The proven receipt: the controller refuses anything less.
            return dict(PROVEN_REPOINT)
        if arguments[0] == "status":
            return {"status": {"armed": False, "reads": 1}, "effective_endpoint": PROXY}
        if arguments[0] == "snapshot":
            return {
                "metrics_unit": dict(PROVEN_REPOINT["after"]),
                "effective_endpoint": PROXY,
                "open_windows": ["0123456789abcdef"],
                "leftover_units": [],
            }
        if arguments[0] == "restore-collector":
            return {"restored_endpoint": "http://127.0.0.1:9400/metrics"}
        return {"ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.cleaned = True
        return {"pod": False, "configmap": False}


def _controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> fixture.EnvReplayController:
    monkeypatch.setattr(fixture, "HostProbeFixture", FakeHostProbe)
    monkeypatch.setattr(fixture, "HostProbeSettings", lambda **kwargs: kwargs)
    # The controller's action-time guard needs no window outside action_window.
    monkeypatch.setattr(fixture, "require_action_time", lambda seconds=1: None)
    regional = FakeRegional(device_xml=H200_XML)
    return fixture.EnvReplayController(
        regional,
        node="node-a",
        image="x@sha256:" + "0" * 64,
        case_id="GF-ENV-LIVE-004",
        run_id="r1",
        case_dir=tmp_path,
    )


def test_controller_ships_the_audited_tool_and_plan_then_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(monkeypatch, tmp_path)
    controller.create()
    plan_path = fixture.write_plan_file(
        tmp_path, fixture.PLAN_FILE, {"steps": [{"reads": 1, "values": {}}]}
    )
    controller.deploy_proxy(plan_path, "GPU-aaaa")
    controller.repoint()
    controller.arm()
    host = controller.host
    verbs = [call[0] for call in host.calls]
    # Both sources are installed before the proxy is deployed.
    assert verbs.index("install-source") < verbs.index("deploy-proxy")
    assert verbs.count("install-source") == 2
    assert "repoint-collector" in verbs and "arm" in verbs
    # The installed tool is the audited repo file (sha passed on the argv).
    install = next(call for call in host.calls if call[0] == "install-source")
    assert "--sha256" in install


def test_controller_cleanup_restores_endpoint_before_stopping_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(monkeypatch, tmp_path)
    controller.create()
    controller.repoint()  # marks the collector as repointed, so cleanup restores it
    receipts = controller.cleanup()
    verbs = [call[0] for call in controller.host.calls]
    assert verbs.index("restore-collector") < verbs.index("stop-proxy")
    assert receipts["probe_residuals"] == {"pod": False, "configmap": False}
    assert controller.host.cleaned, controller.host.calls


def test_controller_fails_the_fixture_when_the_probe_rolled_the_repoint_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The probe restarted the unit, read its process environment, waited for
    # the first pass-through read and judged the repoint ineffective: it has
    # rolled the drop-in back and answers ``repointed: false`` with the reason.
    # That is a fixture error here -- raised with the reason, never a replay
    # budget to wait out -- and the unit is still marked for the restore.
    controller = _controller(monkeypatch, tmp_path)
    controller.create()
    host = controller.host
    original = host.execute

    def execute(*arguments: str, timeout: int = 180) -> dict[str, Any]:
        if arguments[0] == "repoint-collector":
            host.calls.append(arguments)
            return {
                "repointed": False,
                "reason": "the repointed collector made no pass-through read of "
                "the proxy inside the first-read budget",
                "process_environment": {
                    fixture.ENDPOINT_ENV: "http://127.0.0.1:9411/metrics"
                },
                "first_read": {
                    "observed": False,
                    "reads_before": 0,
                    "reads_after": 0,
                    "waited_seconds": 65.0,
                    "budget_seconds": 65,
                },
                "rollback": {
                    "restarted": True,
                    "restored_endpoint": "http://127.0.0.1:9400/metrics",
                },
            }
        return original(*arguments, timeout=timeout)

    monkeypatch.setattr(host, "execute", execute)
    with pytest.raises(RegionalFixtureError) as info:
        controller.repoint()
    message = str(info.value)
    assert "did not take effect on node-a" in message
    assert "no pass-through read of the proxy" in message
    assert "restarted=True" in message and "9400/metrics" in message
    receipts = controller.cleanup()
    verbs = [call[0] for call in host.calls]
    assert verbs.index("restore-collector") < verbs.index("stop-proxy")
    assert "restore" in receipts and "stop_proxy" in receipts


def test_controller_refuses_an_unproven_repoint_receipt_and_still_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An older probe copy answers without ``repointed``/``first_read``: that is
    # no proof, nothing may be armed on it, and the unit is still restored.
    controller = _controller(monkeypatch, tmp_path)
    controller.create()
    host = controller.host
    original = host.execute

    def execute(*arguments: str, timeout: int = 180) -> dict[str, Any]:
        if arguments[0] == "repoint-collector":
            host.calls.append(arguments)
            return {"ok": True}
        return original(*arguments, timeout=timeout)

    monkeypatch.setattr(host, "execute", execute)
    with pytest.raises(fixture.RepointNotProven, match="no first-read proof"):
        controller.repoint()
    assert controller.proof == {}
    receipts = controller.cleanup()
    verbs = [call[0] for call in host.calls]
    assert verbs.index("restore-collector") < verbs.index("stop-proxy")
    assert "restore" in receipts and "stop_proxy" in receipts


def test_controller_keeps_the_proof_and_snapshot_reads_the_unit_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(monkeypatch, tmp_path)
    controller.create()
    receipt = controller.repoint()
    assert receipt["repointed"] is True
    assert controller.proof == {
        "process_endpoint": PROXY,
        "first_read": PROVEN_REPOINT["first_read"],
    }
    snapshot = controller.snapshot()
    assert snapshot["effective_endpoint"] == PROXY
    assert (
        snapshot["metrics_unit"]["InvocationID"] == "52ac4fabb52d4b77beb6d0e928da732e"
    )
    assert controller.host.calls[-1] == ("snapshot",), controller.host.calls[-1]


def test_repoint_proof_is_one_rule_for_the_runner_and_the_024_variants() -> None:
    from scripts.e2e.regional import general_live_env_replay as er

    assert er.repoint_proof is fixture.repoint_proof
    assert issubclass(fixture.RepointNotProven, RegionalFixtureError), (
        "the 024 variants keep catching RegionalFixtureError"
    )
    assert issubclass(fixture.NodeNotClean, fixture.EnvLiveNotShown), (
        "the runner maps every not-shown fixture failure to INCONCLUSIVE"
    )
    with pytest.raises(fixture.RepointNotProven, match="did not take effect: gone"):
        fixture.repoint_proof({"repointed": False, "reason": "gone"}, PROXY)
    assert fixture.repoint_proof(dict(PROVEN_REPOINT), PROXY) == {
        "process_endpoint": PROXY,
        "first_read": PROVEN_REPOINT["first_read"],
    }


def test_node_leftovers_names_each_leftover_and_passes_a_clean_node() -> None:
    clean = {
        "open_windows": [],
        "leftover_units": [],
        "metrics_unit": {
            "ActiveState": "active",
            # The site's own drop-in is not a replay leftover.
            "DropInPaths": (
                "/etc/systemd/system/gpu-fault-metrics-collector.service.d/10-site.conf"
            ),
        },
        "effective_endpoint": REAL,
        "baseline_endpoint": REAL,
    }
    assert fixture.node_leftovers(clean) == []
    assert fixture.node_leftovers(None) == []
    dirty = {
        "open_windows": ["239299cded976dc8", "63f7c3094292bf79"],
        "leftover_units": [
            {
                "unit": "gpu-fault-env-replay-63f7c3094292bf79.service",
                "load": "loaded",
                "active": "failed",
                "sub": "failed",
            },
            {
                "unit": "gpu-fault-env-collector-restore-239299cded976dc8.timer",
                "load": "loaded",
                "active": "active",
                "sub": "waiting",
            },
        ],
        "metrics_unit": {
            "ActiveState": "active",
            "DropInPaths": f"/x/10-site.conf {DROPIN}",
        },
        "effective_endpoint": PROXY,
        "collector_env": {fixture.ENDPOINT_ENV: REAL},
    }
    reasons = fixture.node_leftovers(dirty)
    assert len(reasons) == 4, reasons
    assert reasons[0] == (
        "a previous replay window is still open on the node: "
        "239299cded976dc8, 63f7c3094292bf79"
    )
    assert reasons[1] == (
        "a repoint drop-in of a previous run is still loaded on "
        f"gpu-fault-metrics-collector.service: {DROPIN}"
    )
    assert reasons[2] == (
        "transient units of a previous run are still loaded: "
        "gpu-fault-env-collector-restore-239299cded976dc8.timer (active/waiting), "
        "gpu-fault-env-replay-63f7c3094292bf79.service (failed/failed)"
    )
    assert reasons[3] == (
        f"the collector process reads {PROXY!r}, not its configured endpoint {REAL!r}"
    )
    # Without a configured endpoint to compare against, a proxy endpoint is
    # still a leftover; the real exporter is not.
    assert fixture.node_leftovers({"effective_endpoint": PROXY}) == [
        f"the collector process still reads a replay proxy ({PROXY})"
    ]
    assert fixture.node_leftovers({"effective_endpoint": REAL}) == []


def test_require_proxy_read_before_arm_refuses_reverted_armed_or_reset() -> None:
    proof = fixture.repoint_proof(dict(PROVEN_REPOINT), PROXY)
    good = {"status": {"armed": False, "reads": 2}, "effective_endpoint": PROXY}
    assert fixture.require_proxy_read_before_arm(good, proof, proxy_endpoint=PROXY) == {
        "effective_endpoint": PROXY,
        "reads": 2,
        "first_read_reads_after": 1,
        "armed": False,
    }
    with pytest.raises(fixture.RepointNotProven, match="reverted to 'http://127"):
        fixture.require_proxy_read_before_arm(
            {**good, "effective_endpoint": REAL}, proof, proxy_endpoint=PROXY
        )
    with pytest.raises(fixture.RepointNotProven, match="armed=True"):
        fixture.require_proxy_read_before_arm(
            {**good, "status": {"armed": True, "reads": 2}}, proof, proxy_endpoint=PROXY
        )
    with pytest.raises(fixture.RepointNotProven, match="went back from 1"):
        fixture.require_proxy_read_before_arm(
            {**good, "status": {"armed": False, "reads": 0}},
            proof,
            proxy_endpoint=PROXY,
        )
    # A status without the endpoint (an older probe) is not a reversal.
    older = {"status": {"armed": False, "reads": 1}}
    assert (
        fixture.require_proxy_read_before_arm(older, proof, proxy_endpoint=PROXY)[
            "effective_endpoint"
        ]
        is None
    )


def test_control_plane_reads_use_the_general_live_probes() -> None:
    """The node's GPU records and NODE_HEALTH activity, not the XID snapshot.

    ``store_snapshot`` is keyed on an XID event; a GPU environment finding's
    incident is ``inc-gpu-<finding id>``, which it never returned (live
    GF-ENV-LIVE-004 "no workflow was observed").
    """

    import datetime as dt

    from scripts.e2e.regional.general_live_gpu_store import GPU_STATE_PROBE
    from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE

    calls: list[tuple[str, tuple[str, ...]]] = []

    class Recording(FakeRegional):
        def cpu_python(self, script: str, *arguments: str, **_: Any) -> Any:
            calls.append((script, arguments))
            return {"probe": "gpu" if script is GPU_STATE_PROBE else "activity"}

        def store_snapshot(self, **_options: Any) -> dict[str, Any]:
            raise AssertionError("the XID-keyed snapshot must not be read")

    regional = Recording(device_xml=H200_XML)
    since = dt.datetime(2031, 1, 1, tzinfo=dt.timezone.utc)
    names = ("gpu_tlimit_c", "gpu_temperature_c")
    assert fixture.gpu_records(regional, "node-a", since, names) == {"probe": "gpu"}
    assert fixture.node_records(regional, "node-a", since) == {"probe": "activity"}
    (gpu_script, gpu_args), (activity_script, activity_args) = calls
    assert gpu_script is GPU_STATE_PROBE and activity_script is ACTIVITY_PROBE
    assert gpu_args == (
        "c",
        "node-a",
        since.isoformat(),
        "gpu_temperature_c,gpu_tlimit_c",
    )
    assert activity_args[:3] == ("c", "node-a", since.isoformat())
    assert activity_args[-1] == "GPU_METRICS"
    # The finding view carries what the judges grade on.
    for key in ("confirmation_count", "threshold_source", "workload_state"):
        assert f'"{key}"' in GPU_STATE_PROBE, key


def test_node_state_projects_one_bounded_node_read() -> None:
    seen: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    class Nodes(FakeRegional):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            seen.append((args, kwargs))
            return json.dumps(
                {
                    "metadata": {"name": "node-a", "uid": "u", "annotations": {}},
                    "spec": {
                        "unschedulable": True,
                        "taints": [
                            {"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}
                        ],
                    },
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            )

    node = fixture.node_state(Nodes(device_xml=H200_XML), "node-a")
    assert node["unschedulable"] is True
    assert [item["key"] for item in node["taints"]] == ["gpu-fault.io/quarantined"]
    assert seen[0][0] == ("get", "node", "node-a", "-o", "json")
    assert seen[0][1]["timeout"] == fixture.NODE_READ_TIMEOUT_SECONDS


def test_controller_accepts_an_injected_created_host(tmp_path: Path) -> None:
    class Host:
        created = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def create(self) -> None:
            raise AssertionError("an already created host is not re-created")

        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            self.calls.append(arguments)
            return {"status": {"armed": False}}

        def cleanup(self) -> dict[str, bool]:
            return {"pod": False, "configmap": False}

    host = Host()
    controller = fixture.EnvReplayController(
        FakeRegional(device_xml=H200_XML),
        node="node-a",
        image="x@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-024.thermal_warning",
        run_id="r1",
        case_dir=tmp_path,
        active_deadline_seconds=2400,
        host=host,
    )
    controller.create()
    assert controller.host is host
    assert controller.status(port=9411) == {"status": {"armed": False}}
    assert host.calls == [("status", "--run-id", "r1", "--port", "9411")]


# --------------------------------------------------------------------------- #
# Host-mode nodes (plain EKS): no exporter Pod, the host probe reads the limits
# --------------------------------------------------------------------------- #
class HostModeRegional(FakeRegional):
    """A node whose exporter is the host systemd container: no exporter Pod."""

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if "exec" in args:
            raise AssertionError("nothing to exec into on a host-mode node")
        return json.dumps({"items": []})


HOST_PREFLIGHT = {
    "device_limits": [
        {
            "gpu_uuid": "GPU-aaaa",
            "gpu_index": "0",
            "pci_bdf": "59:00",
            "product": "NVIDIA H200",
            "limits": {
                "gpu_slowdown_tlimit_c": -2.0,
                "gpu_shutdown_tlimit_c": -8.0,
                "gpu_max_operating_tlimit_c": 0.0,
            },
        }
    ],
    "current_tlimit_readings": {"GPU-aaaa": 59.0},
    "collector_env": {"GPU_FAULT_METRICS_MODE": "dcgm"},
}


def test_read_device_limits_falls_back_to_the_host_probe_without_a_pod() -> None:
    calls: list[str] = []

    def host_read() -> dict[str, Any]:
        calls.append("preflight")
        return dict(HOST_PREFLIGHT)

    devices, source = fixture.read_device_limits(
        HostModeRegional(device_xml=""), "node-a", host_read=host_read
    )
    assert source == "host-probe" and calls == ["preflight"]
    assert devices[0]["gpu_uuid"] == "GPU-aaaa"
    # The exporter's T.Limit reading is merged in as the XML read merges it.
    assert devices[0]["limits"]["gpu_tlimit_c"] == 59.0
    assert devices[0]["limits"]["gpu_slowdown_tlimit_c"] == -2.0
    with pytest.raises(RegionalFixtureError, match="no host probe read"):
        fixture.read_device_limits(HostModeRegional(device_xml=""), "node-a")
    with pytest.raises(RegionalFixtureError, match="read no GPU device limits"):
        fixture.host_probe_devices({"device_limits": []})


def test_read_device_limits_prefers_the_exporter_pod_when_present() -> None:
    def host_read() -> dict[str, Any]:
        raise AssertionError("the Pod read must win where the exporter is a DaemonSet")

    devices, source = fixture.read_device_limits(
        FakeRegional(device_xml=H200_XML), "node-a", host_read=host_read
    )
    assert source == "exporter-pod" and devices[0]["limits"]["gpu_tlimit_c"] == 59.0


class RecordingHostProbe:
    """A host probe that answers ``preflight`` and journals create/cleanup."""

    instances: list["RecordingHostProbe"] = []

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.calls: list[str] = []
        RecordingHostProbe.instances.append(self)

    def create(self) -> None:
        self.calls.append("create")

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append(arguments[0])
        assert arguments == ("preflight",), arguments
        return dict(HOST_PREFLIGHT)

    def cleanup(self) -> dict[str, bool]:
        self.calls.append("cleanup")
        return {"pod": False, "configmap": False}


def test_read_only_preflight_on_a_host_mode_node_uses_a_transient_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RecordingHostProbe.instances = []
    monkeypatch.setattr(
        fixture, "RegionalLiveFixture", lambda s: HostModeRegional(device_xml="")
    )
    monkeypatch.setattr(fixture, "HostProbeFixture", RecordingHostProbe)
    monkeypatch.setattr(fixture, "HostProbeSettings", lambda **kwargs: kwargs)
    result = fixture.read_only_preflight(_settings(), tmp_path)
    assert result["errors"] == [], result["errors"]
    assert result["device_limits_source"] == "host-probe"
    assert result["target_gpu"]["gpu_uuid"] == "GPU-aaaa"
    assert "core_tlimit" in result["device_plans"]["GPU-aaaa"]["runnable_variants"]
    probe = RecordingHostProbe.instances[0]
    assert probe.calls == ["create", "preflight", "cleanup"]
    assert probe.settings["probe_script"] == fixture.PROBE_SCRIPT
    assert probe.settings["run_id"].startswith("gf-env-live-004-limits-"), (
        probe.settings
    )


def test_read_only_preflight_fails_when_the_limit_probe_leaves_residuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Leaky(RecordingHostProbe):
        def cleanup(self) -> dict[str, bool]:
            self.calls.append("cleanup")
            return {"pod": True, "configmap": False}

    monkeypatch.setattr(
        fixture, "RegionalLiveFixture", lambda s: HostModeRegional(device_xml="")
    )
    monkeypatch.setattr(fixture, "HostProbeFixture", Leaky)
    monkeypatch.setattr(fixture, "HostProbeSettings", lambda **kwargs: kwargs)
    result = fixture.read_only_preflight(_settings(), tmp_path)
    assert any("left residuals" in item for item in result["errors"]), result["errors"]


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-010: the control node's own preflight
# --------------------------------------------------------------------------- #
def _scope_settings(control_node: str) -> fixture.EnvLiveSettings:
    regional = type("R", (), {"environment": lambda self: {}})()
    return fixture.EnvLiveSettings(
        regional=regional,
        case_id="GF-ENV-LIVE-010",
        confirmation="ENVLIVE010_EXECUTE",
        node="node-a",
        host_probe_image="x@sha256:" + "0" * 64,
        gpu_uuid="GPU-aaaa",
        predecessor_path=None,
        predecessor_id=None,
        site_file=None,
        control_node=control_node,
    )


def test_control_node_preflight_requires_a_clean_distinct_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = fixture.control_node_preflight(
        FakeRegional(device_xml=H200_XML), _scope_settings("node-b")
    )
    assert clean["errors"] == [] and clean["name"] == "node-b"
    same = fixture.control_node_preflight(
        FakeRegional(device_xml=H200_XML), _scope_settings("node-a")
    )
    assert same["errors"] == ["the control node (--node-b) equals the target node"]
    dirty = fixture.control_node_preflight(
        FakeRegional(device_xml=H200_XML, clean=False), _scope_settings("node-b")
    )
    text = "\n".join(dirty["errors"])
    assert "control node node-b" in text
    assert "workflow ownership" in text and "business workload" in text
    assert "Node Agent is not ACTIVE" in text
    # The whole preflight carries the control node's errors and record.
    monkeypatch.setattr(
        fixture,
        "RegionalLiveFixture",
        lambda s: FakeRegional(device_xml=H200_XML, clean=False),
    )
    result = fixture.read_only_preflight(_scope_settings("node-b"), tmp_path)
    assert result["control_node"]["name"] == "node-b"
    assert any("control node node-b" in item for item in result["errors"]), result[
        "errors"
    ]
    monkeypatch.setattr(
        fixture, "RegionalLiveFixture", lambda s: FakeRegional(device_xml=H200_XML)
    )
    result = fixture.read_only_preflight(_scope_settings(""), tmp_path)
    assert result["control_node"] is None
    # Every other case ignores the profile's node-b, dirty or not.
    monkeypatch.setattr(
        fixture,
        "RegionalLiveFixture",
        lambda s: FakeRegional(device_xml=H200_XML, clean=False),
    )
    other = fixture.read_only_preflight(
        dataclasses.replace(_scope_settings("node-b"), case_id="GF-ENV-LIVE-004"),
        tmp_path,
    )
    assert other["control_node"] is None
    assert not any("control node" in item for item in other["errors"]), other["errors"]


def test_env_live_arguments_take_the_control_node_from_node_b() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    fixture.add_env_live_arguments(parser, confirmation="X")
    arguments = parser.parse_args(
        [
            "--run-dir",
            "/tmp/x",
            "--case",
            "GF-ENV-LIVE-010",
            "--node",
            "node-a",
            "--node-b",
            "node-b",
        ]
    )
    assert arguments.node_b == "node-b"
    assert fixture.EnvLiveSettings.__dataclass_fields__["control_node"].default == ""


# --------------------------------------------------------------------------- #
# The controller's 009 verbs and the resume-before-restore cleanup order
# --------------------------------------------------------------------------- #
def test_controller_pause_verbs_and_cleanup_resume_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _controller(monkeypatch, tmp_path)
    controller.create()
    controller.repoint()
    controller.pause_collector(after_reads=7, seconds=195, wait_seconds=150)
    # The 009 recount hold: a second, labelled pause one read later.
    controller.pause_collector(
        after_reads=8, seconds=240, wait_seconds=93, label="hold"
    )
    controller.pause_status()
    controller.restart_collector()
    receipts = controller.cleanup()
    host = controller.host
    verbs = [call[0] for call in host.calls]
    pauses = [call for call in host.calls if call[0] == "pause-collector"]
    pause, hold = pauses
    assert pause[pause.index("--after-reads") + 1] == "7"
    assert pause[pause.index("--seconds") + 1] == "195"
    assert pause[pause.index("--wait-seconds") + 1] == "150"
    assert "--label" not in pause
    assert hold[hold.index("--after-reads") + 1] == "8"
    assert hold[hold.index("--label") + 1] == "hold"
    assert "pause-status" in verbs and "restart-collector" in verbs
    # Every scheduled pause is resumed before the endpoint restore (the gap
    # pause as ``resume``, the hold as ``resume_hold``), then the proxy stops.
    resumes = [call for call in host.calls if call[0] == "resume-collector"]
    assert len(resumes) == 2
    assert [
        call[call.index("--label") + 1] for call in resumes if "--label" in call
    ] == ["hold"]
    assert max(i for i, verb in enumerate(verbs) if verb == "resume-collector") < (
        verbs.index("restore-collector")
    )
    assert verbs.index("restore-collector") < verbs.index("stop-proxy")
    assert "resume" in receipts and "resume_hold" in receipts and "restore" in receipts


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-011: the absolute shutdown limit is the one the product published
# --------------------------------------------------------------------------- #
#: The newest batch of the target GPU, as the GPU-state probe lists it: the H200
#: shutdown limit DCGM field 159 reported, published by the collector.
NEWEST = "2026-11-01T09:43:00+00:00"


def _product_latest(shutdown_at: str = NEWEST) -> dict[str, Any]:
    return {
        "latest": [
            {
                "gpu": "GPU-aaaa",
                "name": "gpu_temperature_c",
                "value": 28.0,
                "observed_at": NEWEST,
                "labels": {},
            },
            {
                "gpu": "GPU-aaaa",
                "name": "gpu_shutdown_temperature_c",
                "value": 95.0,
                "observed_at": shutdown_at,
                "labels": {"limit_source": "DCGM_FI_DEV_SHUTDOWN_TEMP"},
            },
        ]
    }


class ProductRegional(FakeRegional):
    """The exporter Pod's XML plus the product's own GPU-state records."""

    def __init__(self, *, state: Any, **options: Any) -> None:
        super().__init__(**options)
        self.state = state
        self.scripts: list[str] = []

    def cpu_python(self, script: str, *arguments: str, **_: Any) -> Any:
        self.scripts.append(script)
        return self.state(*arguments) if callable(self.state) else self.state


def _shutdown_settings() -> fixture.EnvLiveSettings:
    return dataclasses.replace(
        _settings(), case_id="GF-ENV-LIVE-011", confirmation="ENVLIVE011_EXECUTE"
    )


def test_the_shutdown_case_runs_on_the_limit_the_product_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional import run_env_live as runner

    regional = ProductRegional(device_xml=H200_XML, state=_product_latest())
    monkeypatch.setattr(fixture, "RegionalLiveFixture", lambda s: regional)

    result = fixture.read_only_preflight(_shutdown_settings(), tmp_path)

    assert result["errors"] == [], result["errors"]
    target = result["target_gpu"]
    assert target["limits"]["gpu_shutdown_temperature_c"] == 95.0
    assert target["limit_sources"] == {
        "gpu_shutdown_temperature_c": "product:DCGM_FI_DEV_SHUTDOWN_TEMP"
    }, "the precondition names the device source the product used"
    assert result["device_limits_source"] == "exporter-pod+product-records"
    assert regional.scripts == [fixture.GPU_STATE_PROBE], "one read-only store read"
    assert runner.replay_plans_for_case("GF-ENV-LIVE-011", result) == {
        "plans": {
            "shutdown": {"steps": [{"reads": 1, "values": {"gpu_temperature_c": 96.0}}]}
        },
        "not_run": {},
    }, "H200 is runnable: one DCGM_FI_DEV_GPU_TEMP read at 95 C + 1"


def test_a_limit_the_newest_batch_no_longer_carries_does_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional import run_env_live as runner

    regional = ProductRegional(
        device_xml=H200_XML, state=_product_latest("2026-11-01T09:30:00+00:00")
    )
    monkeypatch.setattr(fixture, "RegionalLiveFixture", lambda s: regional)

    result = fixture.read_only_preflight(_shutdown_settings(), tmp_path)

    assert "gpu_shutdown_temperature_c" not in result["target_gpu"]["limits"], (
        "gpu_metric_latest keeps a withheld limit's old value: it is not published"
    )
    replay = runner.replay_plans_for_case("GF-ENV-LIVE-011", result)
    assert replay["plans"] == {}
    assert "any device source" in replay["not_run"]["shutdown_absolute"], replay
    other = ProductRegional(device_xml=H200_XML, state=_product_latest())
    monkeypatch.setattr(fixture, "RegionalLiveFixture", lambda s: other)
    fixture.read_only_preflight(_settings(), tmp_path)
    assert other.scripts == [], "only 011 reads the product's published limits"


def test_the_published_limit_and_its_source_survive_the_product_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collector -> ingestion -> store -> the real GPU-state probe -> the plan."""

    import contextlib
    import io
    import subprocess
    import sys
    import types
    from datetime import datetime, timezone
    from unittest import mock

    from gpu_fault.app import ApplicationContext
    from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector
    from gpu_fault.collectors.gpu.discovery import query_nvidia_temperature_limits
    from gpu_fault.collectors.models import CollectorContext
    from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService
    from gpu_fault.store import InMemoryStore

    store = InMemoryStore()
    service = GpuMetricsService(store=store)

    class Sink:
        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            return service.ingest(GpuMetricBatch.model_validate(payload)).model_dump(
                mode="json"
            )

    def xml_runner(argv: Any, **_: Any) -> Any:
        return subprocess.CompletedProcess(argv, 0, stdout=H200_XML, stderr="")

    labels = 'gpu="0",UUID="GPU-aaaa",pci_bus_id="00000000:59:00.0"'
    DcgmMetricsCollector(
        Sink(), CollectorContext(cluster_id="c"), node_id="node-a"
    ).collect_text(
        f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 28\n"
        f"DCGM_FI_DEV_GPU_TEMP_LIMIT{{{labels}}} 59\n"
        f"DCGM_FI_DEV_SHUTDOWN_TEMP{{{labels}}} 95\n",
        observed_at=datetime.now(timezone.utc),
        extra_samples=query_nvidia_temperature_limits(xml_runner),
    )

    def probe(*arguments: str) -> Any:
        output = io.StringIO()
        with (
            mock.patch.object(
                ApplicationContext,
                "from_environment",
                lambda: types.SimpleNamespace(store=store),
            ),
            mock.patch.object(sys, "argv", ["gpu-state-probe", *arguments]),
            contextlib.redirect_stdout(output),
        ):
            exec(compile(fixture.GPU_STATE_PROBE, "<probe>", "exec"), {})
        return json.loads(output.getvalue())

    regional = ProductRegional(device_xml=H200_XML, state=probe)

    published = fixture.published_shutdown_limits(regional, "node-a")

    assert {
        gpu: (item["value"], item["source"]) for gpu, item in published.items()
    } == {"GPU-aaaa": (95.0, "DCGM_FI_DEV_SHUTDOWN_TEMP")}, published
