from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import multi_cluster_fixture as multi
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_multi_cluster_fixture_review import pod_document


def settings(tmp_path: Path, **changes: Any) -> multi.MultiClusterSettings:
    for name in ("cpu", "a", "b"):
        (tmp_path / name).write_text("synthetic config", encoding="ascii")
    values = {
        "cpu_kubeconfig": tmp_path / "cpu",
        "namespace": "gpu-system",
        "region": "us-west-2",
        "cluster_a": multi.ClusterTarget("a", tmp_path / "a", "context-a"),
        "cluster_b": multi.ClusterTarget("b", tmp_path / "b", "context-b"),
    }
    return multi.MultiClusterSettings(**(values | changes))


@pytest.mark.parametrize("field", ["namespace", "region"])
def test_multicluster_requires_explicit_scope(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match="must be explicit"):
        settings(tmp_path, **{field: " "})


@pytest.mark.parametrize(
    ("cluster", "context", "file", "message"),
    [
        ("", "context-b", "b", "identity"),
        ("b", "", "b", "identity"),
        ("a", "context-b", "b", "distinct cluster IDs"),
        ("b", "context-a", "a", "distinct GPU contexts"),
        ("b", "context-b", "absent", "does not exist"),
    ],
)
def test_multicluster_target_validation_precedes_fixture_creation(
    tmp_path: Path, cluster: str, context: str, file: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings(
            tmp_path, cluster_b=multi.ClusterTarget(cluster, tmp_path / file, context)
        )


def test_multicluster_constructs_only_the_selected_approved_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = settings(tmp_path)
    calls = []
    monkeypatch.setattr(
        multi, "RegionalLiveFixture", lambda value: calls.append(value) or value
    )
    region = config.regional(config.cluster_b)
    assert region.cluster_id == "b"
    assert region.gpu_context == "context-b"
    assert region.cpu_kubeconfig == tmp_path / "cpu"
    assert region.gpu_kubeconfig == tmp_path / "b"
    assert config.environment() == {
        "CPU_KUBECONFIG": str(tmp_path / "cpu"),
        "GPU_A_KUBECONFIG": str(tmp_path / "a"),
        "GPU_A_CONTEXT": "context-a",
        "GPU_A_CLUSTER_ID": "a",
        "GPU_B_KUBECONFIG": str(tmp_path / "b"),
        "GPU_B_CONTEXT": "context-b",
        "GPU_B_CLUSTER_ID": "b",
        "GPU_FAULT_NAMESPACE": "gpu-system",
        "AWS_REGION": "us-west-2",
    }
    with pytest.raises(ValueError, match="outside the approved"):
        config.regional(multi.ClusterTarget("b", tmp_path / "a", "context-b"))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "clusters",
    [
        None,
        [],
        ["a", "b"],
        [{"cluster_id": "a"}],
        [{"cluster_id": "a"}, {"cluster_id": "a"}],
    ],
)
def test_registry_snapshot_rejects_unknown_or_wrong_pair(clusters: Any) -> None:
    config = SimpleNamespace(
        cluster_a=SimpleNamespace(cluster_id="a"),
        cluster_b=SimpleNamespace(cluster_id="b"),
    )
    regional = SimpleNamespace(cpu_python=lambda *args: {"clusters": clusters})
    with pytest.raises(multi.RegionalFixtureError, match="different cluster pair"):
        multi.registration_snapshot(regional, config)


def test_registry_snapshot_preserves_physical_target_binding() -> None:
    clusters = [
        {"cluster_id": "b", "synthetic": False},
        {"cluster_id": "a", "synthetic": False},
    ]
    calls = []
    region = SimpleNamespace(
        cpu_python=lambda *args: calls.append(args) or {"clusters": clusters}
    )
    config = SimpleNamespace(
        cluster_a=SimpleNamespace(cluster_id="a"),
        cluster_b=SimpleNamespace(cluster_id="b"),
    )
    snapshot = multi.registration_snapshot(region, config)
    assert snapshot == clusters
    assert snapshot[0] is not clusters[0]
    assert calls[0][1:] == ("a", "b")


def test_control_plane_status_reader_covers_both_roles() -> None:
    calls = []

    def kubectl(*args: str) -> str:
        calls.append(args)
        doc = pod_document()
        doc["items"][0]["metadata"]["name"] = args[4].removeprefix("app=")
        return json.dumps(doc)

    result = multi.control_plane_container_statuses(SimpleNamespace(kubectl=kubectl))
    assert set(result) == set(multi.CONTROL_PLANE_APPS)
    assert all(result[app]["app"] == app for app in multi.CONTROL_PLANE_APPS), (
        "CPU container snapshots lost their role attribution"
    )
    assert len(calls) == 2


@pytest.mark.parametrize("defect", ["disappeared", "appeared", "oom", "negative"])
def test_container_comparison_detects_restart_scope_changes(defect: str) -> None:
    before = multi.container_statuses(pod_document())
    after = copy.deepcopy(before)
    if defect == "disappeared":
        after = {"other": copy.deepcopy(before["api"])}
        expected = "api: Pod disappeared"
    elif defect == "appeared":
        after["other"] = copy.deepcopy(before["api"])
        expected = "other: Pod appeared"
    elif defect == "oom":
        after["api"]["containers"]["api"]["last_terminated_reason"] = "OOMKilled"
        expected = "api/api: OOMKilled"
    else:
        after["api"]["containers"]["api"]["restart_count"] = -1
        expected = "api/api: restartCount is unknown"
    assert expected in multi.container_status_errors(before, after)


def test_snapshot_rejects_duplicate_pod_names_even_with_distinct_uids() -> None:
    document = pod_document()
    duplicate = copy.deepcopy(document["items"][0])
    duplicate["metadata"]["uid"] = "another-uid"
    document["items"].append(duplicate)
    with pytest.raises(multi.RegionalFixtureError, match="duplicated"):
        multi.container_statuses(document)


@pytest.mark.parametrize("count", [-1, None])
def test_snapshot_never_coerces_an_unknown_restart_count(count: Any) -> None:
    document = pod_document()
    document["items"][0]["status"]["containerStatuses"][0]["restartCount"] = count
    with pytest.raises(multi.RegionalFixtureError, match="restart count is unknown"):
        multi.container_statuses(document)


def test_control_plane_pressure_reads_the_requested_cluster_only() -> None:
    text = (
        'gpu_fault_processor_cluster_queue_depth{cluster_id="a"} 1\n'
        'gpu_fault_processor_cluster_queue_depth{cluster_id="b"} 7\n'
        "other_metric 8\n"
    )
    calls = []
    region = SimpleNamespace(
        cpu_python=lambda script: calls.append(script) or {"text": text}
    )
    reading = multi.control_plane_pressure(region, "b")
    assert reading["cluster_queue_depth"] == 7
    assert all(value is None for value in reading["rejections"].values()), (
        "missing rejection metrics must remain unknown"
    )
    assert calls == [multi.METRICS_PROBE]


@pytest.mark.parametrize(
    "lines",
    [
        '{cluster_id="a"} NaN',
        '{cluster_id="a"} -1',
        '{cluster_id="a"} 1\n'
        + 'gpu_fault_processor_cluster_queue_depth{cluster_id="a"} 2',
    ],
)
def test_cluster_pressure_rejects_nonfinite_negative_or_duplicate_depth(
    lines: str,
) -> None:
    with pytest.raises(multi.RegionalFixtureError, match="invalid or duplicated"):
        multi.cluster_pressure_reading(
            "gpu_fault_processor_cluster_queue_depth" + lines + "\n", "a"
        )


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "one",
        "id",
        "synthetic",
        "disabled",
        "same-eks",
        "same-hyperpod",
        "missing-eks",
        "missing-hyperpod",
    ],
)
def test_two_logical_registrations_do_not_prove_two_physical_clusters(
    defect: str,
) -> None:
    registrations = [
        {
            "cluster_id": name,
            "enabled": True,
            "synthetic": False,
            "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/" + name,
            "hyperpod_cluster_name": "hp-" + name,
        }
        for name in ("a", "b")
    ]
    if defect == "one":
        registrations.pop()
    elif defect == "id":
        registrations[1]["cluster_id"] = "a"
    elif defect in {"synthetic", "disabled"}:
        registrations[1]["synthetic" if defect == "synthetic" else "enabled"] = (
            defect == "synthetic"
        )
    elif defect.startswith("same-"):
        key = "eks_cluster_arn" if defect == "same-eks" else "hyperpod_cluster_name"
        registrations[1][key] = registrations[0][key]
    elif defect.startswith("missing-"):
        key = "eks_cluster_arn" if defect == "missing-eks" else "hyperpod_cluster_name"
        registrations[1][key] = ""
    assert multi.registrations_are_distinct_physical_clusters(registrations) is (
        defect == "none"
    )
