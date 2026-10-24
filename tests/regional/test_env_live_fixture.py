"""Pure parts of env_live_fixture (device selection, plans, preflight) with fakes."""

from __future__ import annotations

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
    devices = fixture.read_device_limits(regional, "node-a")
    assert devices and devices[0]["gpu_uuid"] == "GPU-aaaa"
    assert devices[0]["limits"]["gpu_slowdown_tlimit_c"] == -2.0


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


def test_node_activity_reads_the_store_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datetime as dt

    regional = FakeRegional(device_xml=H200_XML)
    out = fixture.node_activity(regional, "node-a", dt.datetime.now(dt.timezone.utc))
    assert out == {
        "agent": {"lifecycle_state": "ACTIVE"},
        "collector_statuses": [{"collector": "GPU_METRICS", "sample_count": 5}],
        "incident": None,
    }
