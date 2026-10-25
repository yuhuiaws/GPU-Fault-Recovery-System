"""GF-GENERAL-LIVE-012 GPU inventory read paths: injector/verdict contract.

The four read-path variants are driven over a source-replay ``FakeProbe`` (the
wrapper plan that is installed, the window flags, the journal that is read
back) and a ``FakeRegional`` that scripts the product's records. The product's
real host collector is also run through ``host_case`` so the labels and
collection-error texts the verdicts key on (``failure_mode``, ``discovered_count``,
"GPU inventory query failed", "could not be read ... unknown as well") are the
ones ``collectors/host/inventory.py`` really writes, and the REBOOT_NODE rule
the variants must never feed is pinned against ``host_health.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from scripts.e2e.regional import general_live_devices as mod
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import APPROVALS, coverage_rows
from tests.collectors import _cov95_runtime_collect as support
from tests.collectors import _cov95_runtime_collect_host as host_support
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)
from tools import host_command_replay as wrapper

isolated_runtime = support.isolated_runtime
host_case = host_support.host_case

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
SPECS = {spec.variant: spec for spec in mod.VARIANTS if spec.case_id == mod.CASE_012}
UUIDS = [f"GPU-0000000{index}-0000-0000-0000-000000000000" for index in range(8)]
EXPECTED = 8


# --------------------------------------------------------------------------- #
# scripted product records
# --------------------------------------------------------------------------- #
def _latest() -> list[dict[str, Any]]:
    return [
        {
            "name": "host_gpu_utilization_percent",
            "device": uuid,
            "value": 0.0,
            "observed_at": "2026-10-24T00:00:00+00:00",
            "labels": {},
        }
        for uuid in UUIDS
    ] + [
        {
            "name": "gpu_inventory_expected_count",
            "device": None,
            "value": EXPECTED,
            "observed_at": "2026-10-24T00:00:00+00:00",
            "labels": {},
        }
    ]


def _batch(
    observed_at: str, samples: list[dict[str, Any]], errors: list[str] = ()
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": ["baseline"],
        "collection_errors": list(errors),
        "samples": samples,
    }


def _inventory(
    observed: int, *, consecutive: int, expected: int = EXPECTED
) -> list[dict[str, Any]]:
    missing = max(0, expected - observed)
    labels = {
        "resource": "GPU",
        "expected_count": str(expected),
        "observed_count": str(observed),
        "missing_count": str(missing),
        "excess_count": str(max(0, observed - expected)),
        "consecutive_mismatch_samples": str(consecutive),
        "required_consecutive_samples": "2",
        "node_instance_type": "p5en.48xlarge",
        "expected_gpu_count": str(expected),
    }
    return [
        {"name": name, "device": None, "value": value, "labels": labels}
        for name, value in (
            ("gpu_inventory_expected_count", expected),
            ("gpu_inventory_active_count", observed),
            ("gpu_inventory_missing_count", missing),
            ("gpu_inventory_mismatch", 1 if consecutive >= 2 else 0),
        )
    ]


def _failed(mode: str, *, discovered: bool) -> list[dict[str, Any]]:
    labels = {"resource": "GPU", "expected_count": str(EXPECTED), "failure_mode": mode}
    if discovered:
        labels["discovered_count"] = str(EXPECTED)
    samples = [
        {
            "name": "gpu_inventory_expected_count",
            "device": None,
            "value": EXPECTED,
            "labels": labels,
        }
    ]
    if discovered:
        samples.append(
            {
                "name": "gpu_inventory_discovered_count",
                "device": None,
                "value": EXPECTED,
                "labels": labels,
            }
        )
    return samples


def _doc(batches: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "since": "2026-10-24T00:00:00+00:00",
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
        "batches": batches,
        "incidents": [],
        "workflows": [],
        "markers": [],
        "metric_latest": _latest(),
        "notifications": [],
        "collector_statuses": [],
        **extra,
    }


def _probe(rewrites: int, **extra: Any) -> FakeProbe:
    entries = [
        {"query": "utilization", "applied_rules": [0], "served_returncode": 0}
        for _ in range(rewrites)
    ] + [
        {"query": "utilization", "applied_rules": [], "served_returncode": 0},
        {"query": "driver_version", "applied_rules": [], "served_returncode": 0},
    ]
    script: dict[str, Any] = {
        "install-source": {"bytes": 1},
        "install-wrapper": {"shim": "/run/gfrs-live/source-x/bin/nvidia-smi"},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
            "verification": {
                "ok": True,
                "checks": [
                    {
                        "kind": "inaccessible",
                        "dest": "/proc/driver/nvidia/gpus",
                        "ok": True,
                    }
                ],
            },
        },
        "audit-read": {"entries": entries, "calls": {"nvidia-smi/0": rewrites}},
        "close-window": {"clean": True},
        "status": {"fixtures": {}},
    }
    script.update(extra)
    return FakeProbe(script)


def _ctx(
    variant: str,
    *,
    docs: list[dict[str, Any]],
    tmp_path: Path,
    probe: FakeProbe | None = None,
    node_snapshot: dict[str, Any] | None = None,
) -> Any:
    spec = SPECS[variant]
    node = FakeProbe({"survey": {"tools": {}, "residue": {}, "collector_env": {}}})
    regional = FakeRegional(
        cpu_python=store_script({"activity": docs}), node_snapshot=node_snapshot
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": probe or _probe(1), f"node@{NODE}": node},
        settings=make_settings(spec, node=NODE),
        window_minutes=90,
    )
    return ctx


def _plan_sent(probe: FakeProbe) -> dict[str, Any]:
    installs = [call for call in probe.calls if call[0] == "install-source"]
    flags = dict(zip(installs[-1][1::2], installs[-1][2::2]))
    assert flags["--name"] == "plan-nvidia-smi.json", installs
    return json.loads(base64.b64decode(flags["--b64"]))


def _opened(probe: FakeProbe) -> tuple[str, ...]:
    return next(call for call in probe.calls if call[0] == "open-window")


# --------------------------------------------------------------------------- #
# single_sample
# --------------------------------------------------------------------------- #
def _single_sample_docs() -> list[dict[str, Any]]:
    window = _doc(
        [
            _batch("2026-10-24T00:00:20+00:00", _inventory(7, consecutive=1)),
            _batch("2026-10-24T00:05:20+00:00", _inventory(8, consecutive=0)),
        ]
    )
    return [_doc([]), window]


def test_single_sample_drops_one_real_uuid_once_and_passes(tmp_path: Path) -> None:
    probe = _probe(1)
    ctx = _ctx(
        "single_sample", docs=_single_sample_docs(), tmp_path=tmp_path, probe=probe
    )
    result = mod.gpu_single_sample(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay" and result["dropped_uuid"] == UUIDS[0]
    plan = _plan_sent(probe)
    assert plan["rules"] == [
        {"query": "utilization", "action": "drop_row", "uuid": UUIDS[0], "max_calls": 1}
    ], "exactly one real UUID is dropped from exactly one query"
    assert wrapper.validate_plan("nvidia-smi", plan) == plan
    assert _opened(probe)[1:7] == (
        "--mode",
        "command",
        "--restore-seconds",
        str(SPECS["single_sample"].max_minutes * 60),
        "--wrapper",
        "nvidia-smi",
    ), _opened(probe)
    assert result["missing_samples"] == [
        {"observed_at": "2026-10-24T00:00:20+00:00", "value": 1, "consecutive": "1"}
    ]
    assert any(call[0] == "audit-read" for call in probe.calls), (
        "the journal is the oracle"
    )


def test_single_sample_is_inconclusive_when_the_drop_was_not_served(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        "single_sample", docs=_single_sample_docs(), tmp_path=tmp_path, probe=_probe(0)
    )
    with pytest.raises(VariantInconclusive, match="0 times"):
        mod.gpu_single_sample(ctx)


def test_single_sample_fails_when_the_product_confirmed_the_loss(
    tmp_path: Path,
) -> None:
    confirmed = _doc(
        [
            _batch("2026-10-24T00:00:20+00:00", _inventory(7, consecutive=1)),
            _batch("2026-10-24T00:00:35+00:00", _inventory(7, consecutive=2)),
            _batch("2026-10-24T00:05:20+00:00", _inventory(8, consecutive=0)),
        ]
    )
    ctx = _ctx("single_sample", docs=[_doc([]), confirmed], tmp_path=tmp_path)
    result = mod.gpu_single_sample(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("gpu_inventory_mismatch=1" in error for error in result["errors"]), (
        result
    )


def test_single_sample_blocks_without_a_real_uuid(tmp_path: Path) -> None:
    ctx = _ctx("single_sample", docs=[_doc([], metric_latest=[])], tmp_path=tmp_path)
    with pytest.raises(VariantBlocked, match="names no GPU UUID"):
        mod.gpu_single_sample(ctx)


# --------------------------------------------------------------------------- #
# query_failed / proc_unreadable
# --------------------------------------------------------------------------- #
def _failed_docs(
    mode: str, *, discovered: bool, recover: bool = True
) -> list[dict[str, Any]]:
    error = "_gpu_inventory: CollectorError: GPU inventory query failed: Failed to initialize NVML"
    if not discovered:
        error += (
            "; /proc/driver/nvidia/gpus could not be read (Permission denied), so the "
            "bound GPU count is unknown as well"
        )
    batches = [
        _batch(
            "2026-10-24T00:00:20+00:00",
            _failed(mode, discovered=discovered),
            errors=[
                "_gpu_utilization: CollectorError: GPU utilization query failed",
                error,
            ],
        )
    ]
    if recover:
        batches.append(
            _batch("2026-10-24T00:01:35+00:00", _inventory(8, consecutive=0))
        )
    return [_doc([]), _doc(batches)]


def test_query_failed_is_unknown_with_driver_context_not_zero(tmp_path: Path) -> None:
    probe = _probe(mod.FAILED_TICKS)
    docs = _failed_docs(mod.DRIVER_QUERY_FAILED, discovered=True)
    ctx = _ctx("query_failed", docs=docs, tmp_path=tmp_path, probe=probe)
    result = mod.gpu_query_failed(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    plan = _plan_sent(probe)
    assert plan["rules"] == [
        {
            "query": "utilization",
            "action": "replace_output",
            "stdout": mod.NVML_FAILURE_TEXT,
            "stderr": mod.NVML_FAILURE_TEXT,
            "exit_code": mod.NVML_FAILURE_EXIT,
            "max_calls": mod.FAILED_TICKS,
        }
    ]
    assert "--inaccessible" not in _opened(probe)
    assert result["failed_ticks_served"] == mod.FAILED_TICKS


def test_proc_unreadable_hides_the_procfs_and_needs_no_discovered_count(
    tmp_path: Path,
) -> None:
    probe = _probe(mod.FAILED_TICKS)
    docs = _failed_docs(mod.INVENTORY_UNAVAILABLE, discovered=False)
    ctx = _ctx("proc_unreadable", docs=docs, tmp_path=tmp_path, probe=probe)
    result = mod.gpu_proc_unreadable(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    opened = _opened(probe)
    assert ("--inaccessible", "/proc/driver/nvidia/gpus") == tuple(
        opened[opened.index("--inaccessible") : opened.index("--inaccessible") + 2]
    ), opened
    assert opened[1:3] == ("--mode", "mixed"), opened


def test_proc_unreadable_fails_when_a_count_was_fabricated(tmp_path: Path) -> None:
    docs = _failed_docs(mod.INVENTORY_UNAVAILABLE, discovered=True)
    ctx = _ctx("proc_unreadable", docs=docs, tmp_path=tmp_path, probe=_probe(3))
    result = mod.gpu_proc_unreadable(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("fabricated discovered_count" in error for error in result["errors"]), (
        result["errors"]
    )


def test_query_failed_fails_on_a_zero_count_or_a_reboot(tmp_path: Path) -> None:
    zero = _batch(
        "2026-10-24T00:00:35+00:00",
        _inventory(0, consecutive=2),
        errors=["_gpu_inventory: CollectorError: GPU inventory query failed: x"],
    )
    incidents = [
        {
            "incident_id": "inc-gpu",
            "event_id": "host-1-gpu_inventory_mismatch-node",
            "event_type": "NODE_HEALTH",
            "effective_action": "REBOOT_NODE",
            "state": "ACTION_PENDING",
            "created_at": "2026-10-24T00:00:36+00:00",
            "workflow_request_id": "wf-gpu",
            "node_ids": [NODE],
            "cluster_id": "gpu-eks-cluster",
        }
    ]
    workflows = [
        {
            "request_id": "wf-gpu",
            "incident_id": "inc-gpu",
            "status": "RUNNING",
            "created_at": "2026-10-24T00:00:36+00:00",
            "official_steps": [{"operation": "RESTART_NODE", "parameters": {}}],
            "step_executions": [],
        }
    ]
    first, window = _failed_docs(mod.DRIVER_QUERY_FAILED, discovered=True)
    window["batches"].insert(1, zero)
    window["incidents"], window["workflows"] = incidents, workflows
    ctx = _ctx("query_failed", docs=[first, window], tmp_path=tmp_path, probe=_probe(3))
    result = mod.gpu_query_failed(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert "gpu_inventory_mismatch=1" in joined and "active_count=0" in joined, joined
    assert "RESTART_NODE" in joined and "inc-gpu" in joined, joined


def test_failed_query_is_inconclusive_below_two_served_failures(tmp_path: Path) -> None:
    docs = _failed_docs(mod.DRIVER_QUERY_FAILED, discovered=True)
    ctx = _ctx("query_failed", docs=docs, tmp_path=tmp_path, probe=_probe(1))
    with pytest.raises(VariantInconclusive, match="1 times"):
        mod.gpu_query_failed(ctx)


# --------------------------------------------------------------------------- #
# expectation_wrong (expected FAIL while the product gap is present)
# --------------------------------------------------------------------------- #
def _wrong_docs(*, finding: bool) -> list[dict[str, Any]]:
    wrong = EXPECTED + 1
    batches = [
        _batch(
            "2026-10-24T00:00:20+00:00", _inventory(8, consecutive=1, expected=wrong)
        ),
        _batch(
            "2026-10-24T00:00:35+00:00", _inventory(8, consecutive=2, expected=wrong)
        ),
    ]
    extra: dict[str, Any] = {}
    if finding:
        extra["incidents"] = [
            {
                "incident_id": "inc-exp",
                "event_id": "host-2-gpu_inventory_mismatch-node",
                "event_type": "NODE_HEALTH",
                "effective_action": "REBOOT_NODE",
                "state": "ACTION_PENDING",
                "created_at": "2026-10-24T00:00:36+00:00",
                "workflow_request_id": "wf-exp",
                "node_ids": [NODE],
                "cluster_id": "gpu-eks-cluster",
            }
        ]
        extra["workflows"] = [
            {
                "request_id": "wf-exp",
                "incident_id": "inc-exp",
                "status": "SUCCEEDED",
                "created_at": "2026-10-24T00:00:36+00:00",
                "official_steps": [
                    {"operation": "MARK_UNSCHEDULABLE", "parameters": {}},
                    {"operation": "RESTART_NODE", "parameters": {}},
                    {"operation": "RESTORE_SCHEDULING", "parameters": {}},
                ],
                "step_executions": [],
            }
        ]
    return [_doc([]), _doc(batches, **extra)]


def test_expectation_wrong_overrides_the_recorded_count_plus_one(
    tmp_path: Path,
) -> None:
    probe = _probe(0)
    ctx = _ctx(
        "expectation_wrong",
        docs=_wrong_docs(finding=False),
        tmp_path=tmp_path,
        probe=probe,
    )
    result = mod.gpu_expectation_wrong(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    opened = _opened(probe)
    assert ("--env", f"GPU_FAULT_EXPECTED_GPU_COUNT={EXPECTED + 1}") == tuple(
        opened[opened.index("--env") : opened.index("--env") + 2]
    ), opened
    assert any(call[0] == "close-window" for call in probe.calls), (
        "the window is closed as soon as the product recorded the override"
    )
    assert result["recorded_expectations"][-1]["expected_count"] == str(EXPECTED + 1)


def test_expectation_wrong_fails_on_the_products_reboot_chain(tmp_path: Path) -> None:
    ctx = _ctx("expectation_wrong", docs=_wrong_docs(finding=True), tmp_path=tmp_path)
    result = mod.gpu_expectation_wrong(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert "inc-exp" in joined and "RESTART_NODE" in joined, joined
    spec = SPECS["expectation_wrong"]
    assert spec.expected_outcome == "FAIL" and "REBOOT_NODE" in spec.product_gap
    assert spec.approvals == (mod.HOST_SOURCE_REPLAY, mod.NODE_REBOOT), (
        "the product's chain reboots the node while the gap is present"
    )


def test_expectation_wrong_blocks_without_a_recorded_expectation(
    tmp_path: Path,
) -> None:
    latest = [
        item for item in _latest() if item["name"] != "gpu_inventory_expected_count"
    ]
    ctx = _ctx(
        "expectation_wrong", docs=[_doc([], metric_latest=latest)], tmp_path=tmp_path
    )
    with pytest.raises(VariantBlocked, match="no gpu_inventory_expected_count"):
        mod.gpu_expectation_wrong(ctx)


# --------------------------------------------------------------------------- #
# the product's own collector and rule, pinned
# --------------------------------------------------------------------------- #
def test_registry_shape_for_012() -> None:
    expected = set(coverage_rows()[mod.CASE_012])
    assert set(SPECS) == expected, "012 registers every coverage variant"
    implemented = {name for name, spec in SPECS.items() if spec.implemented}
    assert implemented == {
        "single_sample",
        "query_failed",
        "proc_unreadable",
        "expectation_wrong",
    }, implemented
    for name in ("single_sample", "query_failed", "proc_unreadable"):
        spec = SPECS[name]
        assert spec.lanes == ("eks", "hp") and spec.approvals == (
            mod.HOST_SOURCE_REPLAY,
        )
        assert spec.expected_outcome == "PASS"
    assert mod.NODE_REBOOT in APPROVALS and mod.HOST_SOURCE_REPLAY in APPROVALS
    for name in ("confirmed_missing", "reboot_verified", "post_reboot_failure"):
        assert SPECS[name].deferred.startswith("VENDOR_LAB_ONLY"), SPECS[name].deferred


def test_the_mismatch_sample_feeds_the_reboot_rule() -> None:
    threshold, category, severity, action, _reason = NodeHealthPolicy.METRIC_RULES[
        "gpu_inventory_mismatch"
    ]
    assert (threshold, severity, action) == (
        1.0,
        Severity.CRITICAL,
        RecoveryAction.REBOOT_NODE,
    )
    assert category.value == "GPU"
    assert (
        "gpu_inventory_mismatch" not in NodeHealthPolicy.METRIC_RULE_SUSTAIN_SECONDS
    ), (
        "no sustain: the second confirmed sample reboots, which is why one may never appear"
    )


def _batch_view(batch: Any) -> dict[str, Any]:
    return {
        "observed_at": batch.observed_at.isoformat(),
        "batch_id": batch.batch_id,
        "edge_filter_reasons": [],
        "collection_errors": list(batch.collection_errors),
        "samples": [
            {
                "name": sample.name,
                "device": sample.device,
                "value": sample.value,
                "labels": dict(sample.labels),
            }
            for sample in batch.samples
        ],
    }


@pytest.mark.parametrize("procfs", ["readable", "unreadable"])
def test_the_real_collector_output_satisfies_the_verdict_checks(
    host_case, procfs
) -> None:
    """``collectors/host/inventory.py`` writes what the stages look for."""

    gpus = host_case.roots.proc / "driver/nvidia/gpus"
    if procfs == "readable":
        for index in range(EXPECTED):
            (gpus / f"0000:{0x10 + index:02x}:00.0").mkdir(parents=True)
    else:
        host_support.write_file(gpus, "not a directory\n")
    collector = host_case.build(
        "_gpu_inventory",
        expected_gpu_count=EXPECTED,
        inventory_mismatch_consecutive_samples=2,
        node_instance_type="ml.p5en.48xlarge",
    )
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = (
        mod.NVML_FAILURE_TEXT,
        mod.NVML_FAILURE_TEXT,
        mod.NVML_FAILURE_EXIT,
    )
    batches = []
    for _ in range(3):
        batches.append(_batch_view(collector.collect_once()))
        host_case.clock.sleep(15)
    observed = {"batches": batches, "incidents": [], "workflows": []}
    unreadable = procfs == "unreadable"
    mode = mod.INVENTORY_UNAVAILABLE if unreadable else mod.DRIVER_QUERY_FAILED
    assert (
        mod.gpu_failure_mode_errors(
            observed, mode, expected=EXPECTED, discovered=not unreadable
        )
        == []
    ), batches
    needles = ("GPU inventory query failed",) + (
        ("could not be read", "unknown as well") if unreadable else ()
    )
    assert mod.collection_error_errors(observed, *needles) == [], batches
    assert mod.never_zero_errors(observed) == [], batches
    wrong_mode = mod.DRIVER_QUERY_FAILED if unreadable else mod.INVENTORY_UNAVAILABLE
    assert mod.gpu_failure_mode_errors(
        observed, wrong_mode, expected=EXPECTED, discovered=False
    ), f"the other failure mode must not match {batches}"


def test_a_single_dropped_row_is_not_confirmed_by_the_real_collector(host_case) -> None:
    gpus = host_case.roots.proc / "driver/nvidia/gpus"
    for index in range(EXPECTED):
        (gpus / f"0000:{0x10 + index:02x}:00.0").mkdir(parents=True)
    collector = host_case.build(
        "_gpu_inventory",
        expected_gpu_count=EXPECTED,
        inventory_mismatch_consecutive_samples=2,
    )
    argv = tuple(collector.GPU_QUERY_ARGV)
    full = "".join(f"{uuid}, 0\n" for uuid in UUIDS)
    dropped = "".join(f"{uuid}, 0\n" for uuid in UUIDS[1:])
    batches = []
    for stdout in (dropped, full, full):
        host_case.outputs[argv] = (stdout, "", 0)
        batches.append(_batch_view(collector.collect_once()))
        host_case.clock.sleep(15)
    observed = {"batches": batches, "incidents": [], "workflows": []}
    assert mod.never_zero_errors(observed) == [], batches
    first = {item["name"]: item for item in batches[0]["samples"]}
    assert first["gpu_inventory_missing_count"]["value"] == 1
    assert (
        first["gpu_inventory_mismatch"]["labels"]["consecutive_mismatch_samples"] == "1"
    )
