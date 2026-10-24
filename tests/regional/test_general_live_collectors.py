"""GF-GENERAL-LIVE-025/031 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_collectors as col
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    workflow,
)

SPECS = {spec.variant: spec for spec in col.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey(
    *, exporters: int = 1, units: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"systemctl": True, "systemd-run": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "exporter_processes": [
            {"pid": 100 + i, "start_ticks": "5"} for i in range(exporters)
        ],
        "units": units
        or {
            "gpu-fault-host-collector.service": {
                "UnitFileState": "enabled",
                "ActiveState": "active",
            },
            "gpu-fault-log-collector.service": {
                "UnitFileState": "disabled",
                "ActiveState": "inactive",
            },
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def node_probe(survey_value: dict[str, Any], **extra: Any) -> FakeProbe:
    script = {
        "survey": survey_value,
        "exporter-pause": {"pid": 100, "stopped": True, "deadman": "d"},
        "exporter-resume": {"processes": [{"pid": 100, "resumed": True}]},
        "unit-pause": {"unit": "u", "stopped": True, "deadman": "d"},
        "unit-resume": {"unit": "u"},
        "log-collector-enable": {"after": {"ActiveState": "active"}},
        "log-collector-disable": {"restored": True},
    }
    script.update(extra)
    return FakeProbe(script)


def observation(channel: str, state: str) -> dict[str, Any]:
    return {
        "name": "collector_observation_valid",
        "device": channel,
        "value": 1 if state == "HEALTHY" else 0,
        "labels": {"state": state, "reason": "x"},
        "observed_at": "t",
    }


def dcgm_error_doc(
    *, state: str = "UNKNOWN", with_workflow: bool = True
) -> dict[str, Any]:
    inc = incident(
        "inc-dcgm",
        "dcgm_field_completeness",
        device="node",
        state="ACTION_PENDING",
        workflow="wf-dcgm",
    )
    workflows = []
    if with_workflow:
        workflows = [
            workflow(
                "wf-dcgm",
                "inc-dcgm",
                ["FREEZE_EVIDENCE", "TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_HOST"],
                status="SUCCEEDED",
                executions=[
                    {"operation": "VALIDATE_HOST", "status": "SUCCEEDED", "details": {}}
                ],
            )
        ]
    return {
        "incidents": [inc],
        "workflows": workflows,
        "batches": [],
        "metric_latest": [observation("GPU_METRICS", state)],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [
            {
                "collector": "GPU_METRICS",
                "last_error_at": "2026-10-24T00:05:00+00:00",
                "last_success_at": "2026-10-24T00:00:00+00:00",
                "errors": ["scrape failed"],
            }
        ],
    }


def fresh_success_doc(channel: str) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "metric_latest": [observation(channel, "HEALTHY")],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [
            {
                "collector": channel,
                "last_success_at": "2026-10-24T01:00:00+00:00",
                "last_error_at": None,
                "errors": [],
            }
        ],
        "agent": {"lifecycle_state": "ACTIVE"},
    }


def silence_doc(channel: str) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [
            workflow(
                "wf-s",
                "collector-silent",
                ["TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_HOST"],
                status="SUCCEEDED",
            )
        ],
        "batches": [],
        "metric_latest": [observation(channel, "UNKNOWN")],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [
            {
                "notification_id": "n1",
                "incident_id": "collector-silent",
                "category": "COLLECTOR",
                "deduplication_key": f"collector-silent/c/{NODE}/{channel}/1",
                "created_at": "t",
                "subject": "silent",
                "status": "SENT",
            }
        ],
        "collector_statuses": [
            {
                "collector": channel,
                "last_success_at": "2026-10-24T00:00:00+00:00",
                "last_error_at": None,
                "errors": [],
            }
        ],
        "agent": {"lifecycle_state": "ACTIVE"},
    }


def settings(spec: Any, tmp_path: Path, *, approvals: tuple[str, ...] | None = None):
    s = make_settings(spec)
    updates: dict[str, Any] = {"site_file": tmp_path / "site.yaml"}
    (tmp_path / "site.yaml").write_text("site: test\n")
    if approvals is not None:
        updates["approvals"] = approvals
    return type(s)(**{**s.__dict__, **updates})


RESUME_CALLS = {"exporter-resume", "unit-resume", "log-collector-disable"}


def context(
    spec: Any,
    tmp_path: Path,
    activity_docs: Any,
    *,
    survey_value: dict[str, Any] | None = None,
    approvals=None,
) -> Any:
    """Activity returns the injected document until the producer is resumed on
    the node probe, then the recovered document; this mirrors the real timeline
    without depending on how many times the module polls the store."""

    node = node_probe(survey_value or survey())
    injected = activity_docs if isinstance(activity_docs, dict) else activity_docs[0]
    recovered = activity_docs[-1] if isinstance(activity_docs, list) else activity_docs

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        resumed = any(call and call[0] in RESUME_CALLS for call in node.calls)
        return dict(recovered if resumed else injected)

    regional = FakeRegional(cpu_python=responder)
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings(spec, tmp_path, approvals=approvals),
        probes={f"node@{NODE}": node},
    )


def test_exporter_timeout_passes(tmp_path: Path) -> None:
    spec = SPECS["exporter_timeout"]
    docs = [dcgm_error_doc(), dcgm_error_doc(), fresh_success_doc("GPU_METRICS")]
    ctx = context(spec, tmp_path, docs)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_exporter_timeout_inconclusive_without_exporter(tmp_path: Path) -> None:
    spec = SPECS["exporter_timeout"]
    ctx = context(spec, tmp_path, dcgm_error_doc(), survey_value=survey(exporters=0))
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_exporter_timeout_fails_on_hardware_step(tmp_path: Path) -> None:
    spec = SPECS["exporter_timeout"]
    bad = dcgm_error_doc()
    bad["workflows"][0]["official_steps"].append(
        {"operation": "RESET_GPU", "parameters": {}}
    )
    ctx = context(spec, tmp_path, [bad, bad, fresh_success_doc("GPU_METRICS")])
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"


def test_sustained_unknown_and_post_refresh_pass(tmp_path: Path) -> None:
    for variant, docs in (
        ("sustained_unknown", dcgm_error_doc()),
        ("post_refresh_success", [dcgm_error_doc(), fresh_success_doc("GPU_METRICS")]),
    ):
        spec = SPECS[variant]
        # sustained_unknown ends by reading a fresh success too
        if variant == "sustained_unknown":
            docs = [
                dcgm_error_doc(),
                dcgm_error_doc(),
                fresh_success_doc("GPU_METRICS"),
            ]
        ctx = context(spec, tmp_path, docs)
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])


@pytest.mark.parametrize(
    "variant,channel",
    [
        ("inventory_silent", "GPU_INVENTORY"),
        ("metrics_silent", "GPU_METRICS"),
        ("host_silent", "HOST_TELEMETRY"),
    ],
)
def test_silence_variants_pass(tmp_path: Path, variant: str, channel: str) -> None:
    spec = SPECS[variant]
    ctx = context(
        spec,
        tmp_path,
        [silence_doc(channel), silence_doc(channel), fresh_success_doc(channel)],
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_logs_silent_requires_approval(tmp_path: Path) -> None:
    spec = SPECS["logs_silent"]
    ctx = context(spec, tmp_path, silence_doc("NODE_LOGS"), approvals=())
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_logs_silent_passes_with_approval(tmp_path: Path) -> None:
    spec = SPECS["logs_silent"]
    ctx = context(
        spec,
        tmp_path,
        [
            silence_doc("NODE_LOGS"),
            silence_doc("NODE_LOGS"),
            fresh_success_doc("NODE_LOGS"),
        ],
        approvals=("ENABLE_NODE_LOG_COLLECTOR",),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_disabled_unit_and_expired_agent_and_new_success(tmp_path: Path) -> None:
    spec = SPECS["disabled_unit"]
    doc = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "metric_latest": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }
    ctx = context(spec, tmp_path, doc)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]

    spec = SPECS["expired_agent"]
    ctx = context(spec, tmp_path, {**doc, "agent": {"lifecycle_state": "EXPIRED"}})
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]

    spec = SPECS["new_success"]
    docs = [
        {
            **silence_doc("HOST_TELEMETRY"),
            "collector_statuses": [
                {
                    "collector": "HOST_TELEMETRY",
                    "last_success_at": "2026-10-24T00:00:00+00:00",
                    "last_error_at": None,
                    "errors": [],
                }
            ],
        },
        silence_doc("HOST_TELEMETRY"),
        fresh_success_doc("HOST_TELEMETRY"),
    ]
    ctx = context(spec, tmp_path, docs)
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_capability_and_replay_variants_are_deferred() -> None:
    assert SPECS["inactive_cluster"].deferred.startswith("CAPABILITY_ABSENT"), (
        "SPECS['inactive_cluster'].deferred.startswith('CAPABILITY_ABSENT')"
    )
    assert SPECS["notification_failure"].deferred.startswith("CAPABILITY_ABSENT"), (
        "SPECS['notification_failure'].deferred.startswith('CAPABILITY_ABSENT')"
    )
    # The five DCGM field variants moved to the reviewed field replay
    # (general_live_gpu_fields); this module no longer carries them.
    for variant in (
        "missing_field",
        "one_gpu_missing",
        "conflicting_alias",
        "nan_negative",
        "duplicate_slot",
    ):
        assert variant not in SPECS, variant
