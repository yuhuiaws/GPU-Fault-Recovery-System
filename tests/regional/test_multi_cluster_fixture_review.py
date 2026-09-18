from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.e2e.regional import multi_cluster_fixture as multi


def pod_document() -> dict[str, Any]:
    return {
        "items": [
            {
                "metadata": {"name": "api", "uid": "api-uid"},
                "spec": {"containers": [{"name": "api"}, {"name": "sidecar"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {"name": name, "ready": True, "restartCount": 0}
                        for name in ("api", "sidecar")
                    ],
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "condition",
        "missing",
        "duplicate",
        "extra",
        "count",
        "count-bool",
        "uid",
        "init",
        "deleting",
    ],
)
def test_multicluster_snapshots_refuse_incomplete_control_plane_state(
    defect: str,
) -> None:
    document = pod_document()
    pod = document["items"][0]
    if defect == "empty":
        document["items"] = []
    elif defect == "condition":
        pod["status"]["conditions"] = []
    elif defect == "missing":
        pod["status"]["containerStatuses"].pop()
    elif defect == "duplicate":
        pod["status"]["containerStatuses"].append(pod["status"]["containerStatuses"][0])
    elif defect == "extra":
        pod["status"]["containerStatuses"].append({"name": "extra", "ready": True})
    elif defect in {"count", "count-bool"}:
        pod["status"]["containerStatuses"][0]["restartCount"] = (
            None if defect == "count" else False
        )
    elif defect == "uid":
        del pod["metadata"]["uid"]
    elif defect == "init":
        pod["spec"]["initContainers"] = [{"name": "init"}]
    else:
        pod["metadata"]["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    with pytest.raises(multi.RegionalFixtureError):
        multi.container_statuses(document)


@pytest.mark.parametrize(
    "defect", ["none", "empty", "removed", "added", "count", "not-ready", "uid"]
)
def test_multicluster_container_comparison_checks_complete_sets(defect: str) -> None:
    before = multi.container_statuses(pod_document())
    after = copy.deepcopy(before)
    if defect == "empty":
        after = {}
    elif defect == "removed":
        del after["api"]["containers"]["sidecar"]
    elif defect == "added":
        after["api"]["containers"]["new"] = {"restart_count": 0}
    elif defect == "count":
        after["api"]["containers"]["api"]["restart_count"] = "0"
    elif defect == "not-ready":
        after["api"]["ready"] = False
    elif defect == "uid":
        after["api"]["uid"] = "replacement"
    assert bool(multi.container_status_errors(before, after)) is (defect != "none")


def test_missing_metrics_are_unknown_not_zero() -> None:
    reading = multi.cluster_pressure_reading("", "cluster-a")
    assert reading["cluster_queue_depth"] is None
    assert reading["rejections"] == {name: None for name in multi.REJECTION_COUNTERS}


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1", "not-a-number"])
def test_invalid_pressure_metrics_are_not_skipped(value: str) -> None:
    with pytest.raises(multi.RegionalFixtureError):
        multi.cluster_pressure_reading(
            f'{multi.REJECTION_COUNTERS[0]}{{reason="test"}} {value}\n', "cluster-a"
        )


def test_metrics_parser_preserves_escaped_labels_and_sums_counter_series() -> None:
    reading = multi.cluster_pressure_reading(
        'gpu_fault_processor_cluster_queue_depth{cluster_id="cluster-a"} 2\n'
        'gpu_fault_store_io_rejections_total{reason="one\\\\two"} 3\n'
        'gpu_fault_store_io_rejections_total{reason="quote\\"value"} 4\n',
        "cluster-a",
    )
    assert reading["cluster_queue_depth"] == 2
    assert reading["rejections"]["gpu_fault_store_io_rejections_total"] == 7
