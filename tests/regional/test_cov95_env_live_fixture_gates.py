"""GF-ENV-LIVE shared fixture: the gates its preflight and cleanup hold.

``test_env_live_fixture`` covers the clean and the typical dirty node. These
cases pin the remaining refusals and receipts to the input that produces them:
the operator's flags turned into settings (an explicit predecessor, a probe
image without a digest), the environment the runner exports, a device read
whose XML lists GPUs the product parser yields nothing or no UUID for, a target
GPU whose T.Limit reading is too close to zero to arm, a control node that is
not Ready or cordoned, a cleanup whose every step fails and still runs the next
one, and a GPU-state read asked for an unusable filter.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import env_live_fixture as fixture
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.regional.test_env_live_fixture import (
    PROVEN_REPOINT,
    PROXY,
    FakeHostProbe,
    FakeRegional,
)

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "registry.example/probe@sha256:" + "0" * 64


def gpu_xml(*gpus: str) -> str:
    return "<nvidia_smi_log>" + "".join(gpus) + "</nvidia_smi_log>"


def gpu(
    index: str, uuid: str | None, *, tlimit: str | None, thresholds: bool = True
) -> str:
    temperature = "<gpu_temp>28 C</gpu_temp>"
    if tlimit is not None:
        temperature += f"<gpu_temp_tlimit>{tlimit}</gpu_temp_tlimit>"
    if thresholds:
        temperature += (
            "<gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>"
            "<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>"
            "<gpu_temp_max_gpu_tlimit_threshold>0 C</gpu_temp_max_gpu_tlimit_threshold>"
        )
    return (
        f"<gpu><product_name>NVIDIA H200</product_name>"
        f"<minor_number>{index}</minor_number>"
        + (f"<uuid>{uuid}</uuid>" if uuid else "")
        + f"<temperature>{temperature}</temperature></gpu>"
    )


def settings(**changes: Any) -> fixture.EnvLiveSettings:
    regional = type(
        "R", (), {"environment": lambda self: {"AWS_REGION": "us-west-2"}}
    )()
    value: dict[str, Any] = {
        "regional": regional,
        "case_id": "GF-ENV-LIVE-004",
        "confirmation": "ENVLIVE004_EXECUTE",
        "node": "node-a",
        "host_probe_image": IMAGE,
        "gpu_uuid": "GPU-aaaa",
        "predecessor_path": None,
        "predecessor_id": None,
        "site_file": None,
    }
    value.update(changes)
    return fixture.EnvLiveSettings(**value)


# --------------------------------------------------------------------------- #
# the script's own bootstrap
# --------------------------------------------------------------------------- #
def test_run_as_a_file_the_fixture_puts_the_repository_on_the_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "path", [item for item in sys.path if Path(item or ".").resolve() != ROOT]
    )

    namespace = runpy.run_path(
        str(ROOT / "scripts/e2e/regional/env_live_fixture.py"),
        run_name="env_live_fixture_as_a_file",
    )

    assert sys.path[0] == str(ROOT)
    assert namespace["ROOT"] == ROOT
    assert namespace["SAFE_TLIMIT_MARGIN_C"] == fixture.SAFE_TLIMIT_MARGIN_C


# --------------------------------------------------------------------------- #
# settings from the operator's flags
# --------------------------------------------------------------------------- #
def parsed(tmp_path: Path, *extra: str) -> argparse.Namespace:
    for name in ("cpu.config", "gpu.config"):
        (tmp_path / name).write_text("unit fake transport\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    fixture.add_env_live_arguments(parser, confirmation="ENVLIVE010_EXECUTE")
    return parser.parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--case",
            "GF-ENV-LIVE-010",
            "--cpu-kubeconfig",
            str(tmp_path / "cpu.config"),
            "--gpu-kubeconfig",
            str(tmp_path / "gpu.config"),
            "--gpu-context",
            "unit-gpu",
            "--cluster-id",
            "gpu-eks-cluster",
            "--region",
            "us-west-2",
            "--node",
            "node-a",
            *extra,
        ]
    )


@pytest.fixture
def no_site_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)


@pytest.mark.usefixtures("no_site_profile")
def test_configure_carries_the_operator_predecessor_and_the_control_node(
    tmp_path: Path,
) -> None:
    arguments = parsed(
        tmp_path,
        "--host-probe-image",
        IMAGE,
        "--gpu-uuid",
        "GPU-aaaa",
        "--node-b",
        "node-b",
        "--predecessor-evidence",
        str(tmp_path / "pred.json"),
        "--site-file",
        str(tmp_path / "site.yaml"),
        "--confirm-isolate",
        "ISOLATE",
        "--confirm-stop",
        "STOP",
    )

    value = fixture.configure(arguments, "ENVLIVE010_EXECUTE")

    assert (value.predecessor_id, value.predecessor_path) == (
        "operator-supplied",
        (tmp_path / "pred.json").resolve(),
    )
    assert value.site_file == (tmp_path / "site.yaml").resolve()
    assert (value.isolate_confirm, value.stop_confirm) == ("ISOLATE", "STOP")
    assert value.environment() == {
        **value.regional.environment(),
        "GPU_FAULT_ENV_LIVE_CASE": "GF-ENV-LIVE-010",
        "GPU_FAULT_ENV_LIVE_NODE": "node-a",
        "GPU_FAULT_HOST_PROBE_IMAGE": IMAGE,
        "GPU_FAULT_ENV_LIVE_GPU_UUID": "GPU-aaaa",
        "GPU_FAULT_ENV_LIVE_CONTROL_NODE": "node-b",
    }


@pytest.mark.usefixtures("no_site_profile")
def test_configure_without_a_predecessor_leaves_it_not_required(tmp_path: Path) -> None:
    value = fixture.configure(
        parsed(tmp_path, "--host-probe-image", IMAGE), "ENVLIVE010_EXECUTE"
    )

    assert (value.predecessor_id, value.predecessor_path, value.site_file) == (
        None,
        None,
        None,
    )
    environment = value.environment()
    assert "GPU_FAULT_ENV_LIVE_GPU_UUID" not in environment
    assert "GPU_FAULT_ENV_LIVE_CONTROL_NODE" not in environment
    assert environment["GPU_FAULT_ENV_LIVE_NODE"] == "node-a"


@pytest.mark.usefixtures("no_site_profile")
def test_configure_refuses_a_probe_image_without_a_digest(tmp_path: Path) -> None:
    arguments = parsed(tmp_path, "--host-probe-image", "registry.example/probe:v1")

    with pytest.raises(RegionalFixtureError, match="immutable digest"):
        fixture.configure(arguments, "ENVLIVE010_EXECUTE")


# --------------------------------------------------------------------------- #
# device limits
# --------------------------------------------------------------------------- #
class XmlRegional(FakeRegional):
    def __init__(self, xml: str, *, ready: str = "True") -> None:
        super().__init__(device_xml=xml)
        self.ready = ready

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {**super().node_snapshot(node), "ready": self.ready}


MIXED_XML = gpu_xml(
    gpu("0", "GPU-aaaa", tlimit="5 C"),
    gpu("1", None, tlimit="40 C"),
    gpu("2", "GPU-cccc", tlimit="61 C", thresholds=False),
    gpu("3", "GPU-dddd", tlimit=None),
)


def test_the_pod_read_keeps_only_gpus_the_product_parser_reads() -> None:
    devices, source = fixture.read_device_limits(XmlRegional(MIXED_XML), "node-a")

    assert source == "exporter-pod"
    assert [(item["gpu_index"], item["gpu_uuid"]) for item in devices] == [
        ("1", None),
        ("0", "GPU-aaaa"),
        ("3", "GPU-dddd"),
    ], "GPU-cccc reports a T.Limit reading but no limit the product parser reads"
    by_uuid = {item["gpu_uuid"]: item["limits"] for item in devices}
    assert by_uuid["GPU-aaaa"]["gpu_tlimit_c"] == 5.0
    assert "gpu_tlimit_c" not in by_uuid["GPU-dddd"], (
        "a GPU without a <gpu_temp_tlimit> reading gets none merged in"
    )
    assert "gpu_tlimit_c" not in by_uuid[None], (
        "a reading is merged only for a GPU the XML names by UUID"
    )


def test_the_host_read_skips_malformed_entries_and_keeps_a_recorded_reading() -> None:
    devices = fixture.host_probe_devices(
        {
            "device_limits": [
                "not-a-device",
                {
                    "gpu_uuid": "GPU-aaaa",
                    "gpu_index": "0",
                    "limits": {"gpu_tlimit_c": 57.0, "gpu_slowdown_tlimit_c": -2.0},
                },
                {"gpu_uuid": "GPU-bbbb", "gpu_index": "1", "limits": {}},
            ],
            "current_tlimit_readings": {"GPU-aaaa": 59.0},
        }
    )

    assert [item["gpu_uuid"] for item in devices] == ["GPU-aaaa", "GPU-bbbb"]
    assert devices[0]["limits"]["gpu_tlimit_c"] == 57.0, (
        "the parser's own reading wins over the exporter's"
    )
    assert devices[1]["limits"] == {}, "no reading recorded for GPU-bbbb"


def test_a_preflight_names_every_gate_the_node_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(EXECUTION_SCOPE_ENV, raising=False)
    monkeypatch.delenv(SELECTION_REFERENCE_ENV, raising=False)
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(
        json.dumps({"case_id": "operator-supplied", "verdict": "FAIL"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fixture, "RegionalLiveFixture", lambda _s: XmlRegional(MIXED_XML, ready="False")
    )
    value = settings(predecessor_path=predecessor, predecessor_id="operator-supplied")

    result = fixture.read_only_preflight(value, tmp_path)

    assert result["errors"] == [
        "operator-supplied predecessor evidence is not PASS",
        "node-a is not Ready",
        "target GPU T.Limit reading 5.0C is not safely above 0; refuse to arm",
    ]
    assert sorted(result["device_plans"]) == ["GPU-aaaa", "GPU-dddd"], (
        "a device without a UUID gets no plan"
    )
    assert result["target_gpu"]["gpu_uuid"] == "GPU-aaaa"
    assert (
        json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))["errors"]
        == result["errors"]
    )


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-010's control node
# --------------------------------------------------------------------------- #
class ControlRegional(FakeRegional):
    def node_snapshot(self, node: str) -> dict[str, Any]:
        snapshot = super().node_snapshot(node)
        if node == "node-b":
            snapshot.update(ready="Unknown", unschedulable=True)
        return snapshot


def test_a_control_node_that_is_not_ready_and_cordoned_is_refused() -> None:
    result = fixture.control_node_preflight(
        ControlRegional(device_xml=""),
        settings(case_id="GF-ENV-LIVE-010", control_node="node-b"),
    )

    assert result["errors"] == [
        "control node node-b is not Ready",
        "control node node-b is cordoned",
    ]
    assert result["agent_state"] == "ACTIVE"


# --------------------------------------------------------------------------- #
# cleanup keeps going past every failed step
# --------------------------------------------------------------------------- #
class FailingHost(FakeHostProbe):
    FAILING = {"resume-collector", "restore-collector", "stop-proxy"}

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        if arguments[0] in self.FAILING:
            self.calls.append(arguments)
            raise RegionalFixtureError(f"{arguments[0]} refused by the node")
        return super().execute(*arguments, timeout=timeout)


def test_cleanup_records_each_failed_step_and_still_deletes_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixture, "require_action_time", lambda seconds=1: None)
    host = FailingHost(settings=None)
    controller = fixture.EnvReplayController(
        FakeRegional(device_xml=""),
        node="node-a",
        image=IMAGE,
        case_id="GF-ENV-LIVE-009",
        run_id="r1",
        case_dir=tmp_path,
        host=host,
    )
    controller.repoint()
    assert controller.proof == {
        "process_endpoint": PROXY,
        "first_read": PROVEN_REPOINT["first_read"],
    }
    controller.pause_collector(after_reads=7, seconds=195, wait_seconds=150)

    receipts = controller.cleanup()

    assert receipts == {
        "resume_error": "RegionalFixtureError: resume-collector refused by the node",
        "restore_error": "RegionalFixtureError: restore-collector refused by the node",
        "stop_proxy_error": "RegionalFixtureError: stop-proxy refused by the node",
        "probe_residuals": {"pod": False, "configmap": False},
    }
    assert [call[0] for call in host.calls][-3:] == [
        "resume-collector",
        "restore-collector",
        "stop-proxy",
    ]
    assert host.cleaned is True


def test_cleanup_of_a_never_repointed_collector_only_stops_the_proxy(
    tmp_path: Path,
) -> None:
    host = FakeHostProbe(settings=None)
    controller = fixture.EnvReplayController(
        FakeRegional(device_xml=""),
        node="node-a",
        image=IMAGE,
        case_id="GF-ENV-LIVE-004",
        run_id="r1",
        case_dir=tmp_path,
        host=host,
    )

    receipts = controller.cleanup()

    assert receipts == {
        "stop_proxy": {"ok": True},
        "probe_residuals": {"pod": False, "configmap": False},
    }
    assert [call[0] for call in host.calls] == ["stop-proxy"], (
        "nothing was repointed or paused, so nothing is restored or resumed"
    )


# --------------------------------------------------------------------------- #
# GPU-state reads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("names", [["gpu_temperature_c", ""], ["a,b"]])
def test_gpu_records_refuses_an_empty_or_comma_filter(names: list[str]) -> None:
    calls: list[tuple[Any, ...]] = []

    class Recording(FakeRegional):
        def cpu_python(self, *arguments: Any, **options: Any) -> dict[str, Any]:
            calls.append(arguments)
            return {}

    with pytest.raises(ValueError, match="non-empty, no comma"):
        fixture.gpu_records(
            Recording(device_xml=""),
            "node-a",
            datetime(2026, 11, 1, tzinfo=timezone.utc),
            names,
        )
    assert calls == [], "a refused filter never reaches the control plane"
