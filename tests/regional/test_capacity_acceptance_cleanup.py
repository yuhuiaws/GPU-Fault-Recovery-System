from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import capacity_acceptance_base as base


class CleanupHarness(base.CapHarnessBase):
    def __init__(self) -> None:
        self.run_id = "cap-review"
        self.resource_prefix = "gpu-fault-cap-review"
        self.secret_name = self.resource_prefix + "-registry"
        self.configmap_name = self.resource_prefix + "-scripts"
        self.processor_mode = "active-active"
        self.region = "us-west-2"
        self.active_probe = None
        self.calls: list[tuple[str, ...]] = []
        self.deleted: list[str] = []
        self.residual_pods: list[dict[str, Any]] = []
        self.refused_kind = ""

    def kubectl(
        self, *args: str, check: bool = True, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        refused = args[:2] == ("delete", self.refused_kind)
        if refused and check:
            raise base.CapError("delete was refused")
        if args[:2] in {("get", "pods"), ("get", "pod")}:
            selector = args[args.index("-l") + 1]
            matches = selector == "gpu-fault.io/capacity-probe=cap-review-cap001"
            items = self.residual_pods if matches else []
            output = (
                json.dumps({"items": items})
                if args[-1] == "json"
                else "\n".join("pod/" + item["metadata"]["name"] for item in items)
            )
        else:
            output = ""
        return subprocess.CompletedProcess(args, int(refused), output, "")

    def drop_database_fallback(self, database: str) -> None:
        self.deleted.append(database)


def probe() -> base.Probe:
    return base.Probe(
        case="CAP001",
        deployment="gpu-fault-cap-review-cap001",
        service="gpu-fault-cap-review-cap001",
        database="gpu_fault_cap_review_cap001",
        pod="probe-pod",
        local_port=1234,
        url="http://127.0.0.1:1234",
        port_forward=SimpleNamespace(poll=lambda: 0),  # type: ignore[arg-type]
    )


def test_probe_cleanup_checks_the_label_it_actually_created() -> None:
    harness = CleanupHarness()
    harness.residual_pods = [{"metadata": {"name": "probe-pod"}}]
    owned = probe()
    harness.active_probe = owned

    with pytest.raises(base.CapError, match="Pod|pod|residual"):
        harness.cleanup_probe(owned)

    assert harness.deleted == [], "a live probe must not lose its database"
    assert harness.active_probe is owned, "failed cleanup must remain retryable"


def test_failed_deployment_delete_is_not_silently_successful() -> None:
    harness = CleanupHarness()
    harness.refused_kind = "deployment"
    owned = probe()
    harness.active_probe = owned

    with pytest.raises(base.CapError, match="delete"):
        harness.cleanup_probe(owned)

    assert harness.active_probe is owned, "the pending cleanup must not be forgotten"


def test_common_cleanup_attempts_both_resources_and_reports_errors() -> None:
    harness = CleanupHarness()
    harness.refused_kind = "secret"

    with pytest.raises(base.CapError, match="secret"):
        harness.cleanup_common()

    assert ("delete", "configmap", harness.configmap_name, "--ignore-not-found") in (
        harness.calls
    ), "one refused deletion must not suppress the other cleanup"


def test_successful_cleanup_records_verified_absence() -> None:
    harness = CleanupHarness()
    owned = probe()
    harness.active_probe = owned

    result = harness.cleanup_probe(owned)

    assert result == {"database_dropped": True, "residual_probe_pods": []}, (
        "successful cleanup must contain the actual absence proof"
    )
    assert harness.deleted == [owned.database], "only the owned database is removed"
    assert harness.active_probe is None, "verified cleanup can release the tracker"


def test_partial_probe_creation_is_tracked_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CleanupHarness()
    cleaned: list[base.Probe] = []

    def apply(**kwargs: Any) -> None:
        raise OSError("Deployment creation acknowledgement lost")

    def cleanup(owned: base.Probe) -> dict[str, Any]:
        cleaned.append(owned)
        harness.active_probe = None
        return {"database_dropped": True, "residual_probe_pods": []}

    monkeypatch.setattr(harness, "_apply_probe_resources", apply)
    monkeypatch.setattr(harness, "cleanup_probe", cleanup)

    with pytest.raises(OSError, match="acknowledgement"):
        harness.deploy_probe("CAP001", {})

    assert len(cleaned) == 1, "partial creation must have one owned cleanup target"
    assert cleaned[0].deployment == harness.resource_prefix + "-cap001", (
        "the failed create and cleanup must use the same Deployment"
    )
    assert cleaned[0].database == "gpu_fault_cap_review_cap001", (
        "cleanup must use the exact normalized database passed to the probe"
    )


@pytest.mark.parametrize(
    "result", [{}, {"status": "FAIL"}, {"status": "PENDING"}, {"status": "PARTIAL"}]
)
def test_harness_rejects_a_missing_or_failed_case_result(
    result: dict[str, Any],
) -> None:
    assert (
        base.CapHarnessBase.verdict(
            result=result, error=None, cleanup_errors=[], production_unchanged=True
        )
        == "FAIL"
    ), "a returned dictionary is not itself a passing case result"


def test_capacity_rejects_unbounded_latency_factor(tmp_path: Path) -> None:
    with pytest.raises(base.CapError, match="b_latency_factor"):
        base.CapCoreHarness(
            site_path=tmp_path / "unused-site",
            run_dir=tmp_path,
            case_id="GF-REGIONAL-CAP-001",
            predecessor={"valid": True},
            b_latency_factor=float("inf"),
        )
