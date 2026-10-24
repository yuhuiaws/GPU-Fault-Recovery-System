"""GF-GENERAL-LIVE-003 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_utilization as util
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    make_settings,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in util.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


class FakeJob:
    instances: list["FakeJob"] = []

    def __init__(self, ctx: Any, *, name: str, gpu_count: int = 8) -> None:
        self.name = name
        self.gpu_count = gpu_count
        FakeJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {"pods": []}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": f"{self.name}-a001",
            "gpu_uuids": ["GPU-1"],
            "pods": [],
        }

    def cleanup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJob.instances = []
    monkeypatch.setattr(util, "ManagedIdleJob", FakeJob)


def survey() -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            name: True
            for name in ("losetup", "mkfs.ext4", "mount", "fallocate", "systemd-run")
        },
        "modules": {},
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "ext4"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            }
        ],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def node_probe(oracle: dict[str, Any] | None = None) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey(),
            "oracle-start": {"unit": "oracle"},
            "oracle-stop": {},
            "oracle-read": oracle or {"records": [], "next_offset": 0, "total": 0},
            "memory-start": {"unit": "mem", "deadman": "d"},
            "memory-stop": {"memory": {}},
            "page-cache-start": {"unit": "pc", "file": "/x", "deadman": "d"},
            "page-cache-stop": {"removed": ["/x"]},
        }
    )


def storage_probe() -> FakeProbe:
    return FakeProbe(
        {
            "fs-create": {"name": "fs", "mountpoint": "/mnt/gfrs-live/x/fs"},
            "fs-fill": {"statvfs": {"used_percent": 92.0}},
            "fs-unfill": {"statvfs": {"used_percent": 20.0}},
            "fs-destroy": {"destroyed": True},
        }
    )


def window_probe() -> FakeProbe:
    return FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )


def probes(oracle: dict[str, Any] | None = None) -> dict[str, FakeProbe]:
    return {
        f"node@{NODE}": node_probe(oracle),
        f"storage@{NODE}": storage_probe(),
        f"window@{NODE}": window_probe(),
    }


def low_doc(
    metric: str,
    rule_id: str,
    validation: str,
    device: str = "node",
    *,
    probe: str | None = None,
    attempt: str | None = None,
) -> dict[str, Any]:
    inc = incident(
        "inc-u",
        metric,
        device=device,
        rule_id=rule_id,
        state="ACTION_PENDING",
        workflow="wf-u",
    )
    if attempt is not None:
        inc["attempt_id"] = attempt
    executions = [
        {
            "operation": validation,
            "status": "FAILED" if probe else "SUCCEEDED",
            "details": {"host_diagnostic_failure": bool(probe)},
        }
    ]
    wf = workflow(
        "wf-u",
        "inc-u",
        ["FREEZE_EVIDENCE", validation],
        status="FAILED" if probe else "SUCCEEDED",
        probe_metric=probe,
        executions=executions,
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                "2026-10-24T00:05:00+00:00",
                metric,
                2.0,
                device=None if device == "node" else device,
            )
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def empty() -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def test_gpu_idle_passes(tmp_path: Path) -> None:
    spec = SPECS["gpu_idle"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "host_gpu_utilization_percent",
                    "LOW_GPU_UTILIZATION",
                    "VALIDATE_GPU",
                    attempt="gl3-idle-1-a001",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_gpu_idle_fails_when_dcgm_diagnostic_runs(tmp_path: Path) -> None:
    spec = SPECS["gpu_idle"]
    settings = _settings(spec, tmp_path)
    doc = low_doc(
        "host_gpu_utilization_percent",
        "LOW_GPU_UTILIZATION",
        "VALIDATE_GPU",
        attempt="gl3-idle-1-a001",
    )
    doc["workflows"][0]["official_steps"].insert(
        1, {"operation": "RUN_DCGM_DIAGNOSTIC", "parameters": {}}
    )
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"


def test_rank_cpu_idle_passes(tmp_path: Path) -> None:
    spec = SPECS["rank_cpu_idle"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "training_rank_cpu_percent",
                    "LOW_CPU_UTILIZATION",
                    "VALIDATE_HOST",
                    attempt="gl3-rankcpu-1-a001",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_low_available_memory_requires_approval(tmp_path: Path) -> None:
    spec = SPECS["low_available_memory"]
    settings = _settings(spec, tmp_path, approvals=())
    regional = FakeRegional(cpu_python=store_script({"activity": empty()}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_low_available_memory_passes(tmp_path: Path) -> None:
    spec = SPECS["low_available_memory"]
    settings = _settings(spec, tmp_path, approvals=("HOST_MEMORY_PRESSURE",))
    oracle = {
        "records": [{"memory_available_percent": 3.0} for _ in range(40)],
        "next_offset": 40,
        "total": 40,
    }
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "memory_available_percent",
                    "memory",
                    "VALIDATE_HOST",
                    probe="memory_available_percent",
                )
            }
        )
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(oracle),
        settings=settings,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_high_page_cache_passes(tmp_path: Path) -> None:
    spec = SPECS["high_page_cache"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "page_cache_percent",
                    "page_cache",
                    "VALIDATE_HOST",
                    probe="page_cache_percent",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_scratch_full_passes(tmp_path: Path) -> None:
    spec = SPECS["scratch_full"]
    settings = _settings(spec, tmp_path)
    mount = util._fs_mount(
        make_context(
            spec,
            tmp_path=tmp_path,
            regional=FakeRegional(cpu_python=store_script({"activity": empty()})),
            probes=probes(),
            settings=settings,
        )
    )
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "local_filesystem_used_percent",
                    "local_filesystem",
                    "VALIDATE_HOST",
                    device=mount,
                    probe="local_filesystem_used_percent",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_scratch_full_inconclusive_below_90(tmp_path: Path) -> None:
    spec = SPECS["scratch_full"]
    settings = _settings(spec, tmp_path)
    store = storage_probe()
    store.script["fs-fill"] = {"statvfs": {"used_percent": 50.0}}
    regional = FakeRegional(cpu_python=store_script({"activity": empty()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={
            f"node@{NODE}": node_probe(),
            f"storage@{NODE}": store,
            f"window@{NODE}": window_probe(),
        },
        settings=settings,
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_idle_node_and_ambiguous_ownership_pass(tmp_path: Path) -> None:
    for variant in ("idle_node", "ambiguous_ownership"):
        spec = SPECS[variant]
        settings = _settings(spec, tmp_path)
        regional = FakeRegional(cpu_python=store_script({"activity": empty()}))
        ctx = make_context(
            spec,
            tmp_path=tmp_path,
            regional=regional,
            probes=probes(),
            settings=settings,
        )
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])


def test_attempt_change_attributes_to_b(tmp_path: Path) -> None:
    spec = SPECS["attempt_change"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "host_gpu_utilization_percent",
                    "LOW_GPU_UTILIZATION",
                    "VALIDATE_GPU",
                    attempt="gl3-attB-1-a001",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def _settings(spec: Any, tmp_path: Path, *, approvals: tuple[str, ...] | None = None):
    settings = make_settings(spec)
    updates: dict[str, Any] = {"site_file": tmp_path / "site.yaml"}
    (tmp_path / "site.yaml").write_text("site: test\n")
    if approvals is not None:
        updates["approvals"] = approvals
    return type(settings)(**{**settings.__dict__, **updates})
