"""NOTIFY-005 holds every GPU of its nodes, because only held GPUs are graded.

The case proves that one node's eight low-utilisation findings leave as one
mail. Its fixture used to request one GPU and rely on the node's seven
unallocated GPUs being graded as well; the product now grades only the GPUs a
managed attempt holds (``gpu_fault/host_low_utilization.py``), so the fixture
requests all eight. Its CPU burners hold no compute context, so the per-GPU CPU
rule has nothing to grade and the runner checks that no ``LOW_CPU_UTILIZATION``
mail names the nodes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample
from gpu_fault.models import WorkloadState
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional.notify005_checks import (
    NODE_GPU_COUNT,
    low_utilization_manifest,
    phase_checks,
)
from tests._builders import (
    asgi_client,
    attempt_observation,
    container_observation,
    host_telemetry_batch,
)
from tests.host_health._support import RecordingNotifier

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
CLUSTER = "hp-cluster"
NODE = "hyperpod-i-00000000000000001"
NAME = "notify005-1n-1-1792356981"
WORKLOAD = f"gpu-fault-system/job/{NAME}"
GPUS = tuple(f"GPU-{index:08x}-965e-fc68-40e4-225aaa3315b6" for index in range(8))


def _containers(manifest):
    if manifest["kind"] == "Job":
        return manifest["spec"]["template"]["spec"]["containers"]
    return [
        container
        for spec in manifest["spec"]["pytorchReplicaSpecs"].values()
        for container in spec["template"]["spec"]["containers"]
    ]


@pytest.mark.parametrize("nodes", [("a",), ("a", "b", "c")], ids=["1n", "3n"])
def test_the_fixture_requests_every_gpu_of_each_selected_node(nodes):
    manifest = low_utilization_manifest(name=NAME, nodes=nodes, image="image")

    resources = [item["resources"] for item in _containers(manifest)]

    assert NODE_GPU_COUNT == 8
    assert resources and all(
        item["requests"]["nvidia.com/gpu"]
        == item["limits"]["nvidia.com/gpu"]
        == str(NODE_GPU_COUNT)
        for item in resources
    ), resources


def _post(context, offset):
    observed_at = NOW + timedelta(seconds=offset)
    context.store.save_attempt_observation(
        attempt_observation(
            NAME,
            f"{NAME}-a001",
            observed_at,
            cluster_id=CLUSTER,
            workload_ids=[WORKLOAD],
            containers=[
                # What the watcher reads in the burner's container: the eight
                # GPUs the device plugin gave it.
                container_observation(
                    "pod-0", f"{NAME}-x7k2p", 0, NODE, gpu_uuids=list(GPUS), gpu_count=8
                )
            ],
        )
    )
    batch = host_telemetry_batch(
        f"host-{NODE}-{offset}",
        observed_at,
        [
            # Eight busy cores of 96 and no compute process: the burners.
            HostMetricSample(name="cpu_usage_percent", value=8.6, unit="percent"),
            *(
                HostMetricSample(
                    name="host_gpu_utilization_percent",
                    value=0.0,
                    unit="percent",
                    device=uuid,
                )
                for uuid in GPUS
            ),
        ],
        cluster_id=CLUSTER,
        node_id=NODE,
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[WORKLOAD],
        received_at=observed_at,
    )

    async def run():
        async with asgi_client(context) as client:
            return await client.post(
                "/v1/collector-events/host-telemetry",
                json=batch.model_dump(mode="json"),
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    return response.json()


def test_one_phase_mails_one_message_naming_all_eight_gpus_and_no_cpu_rule(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(store=InMemoryStore(), notification_notifier=notifier)
    responses = [_post(context, offset) for offset in range(0, 391, 30)]
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=context.store)),
    )

    def cpu_python(script: str, *args: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["notification-query", *args])
        exec(compile(script, "<notification-query>", "exec"), {})
        return json.loads(capsys.readouterr().out)

    records, polls = runner.wait_low_utilization_notifications(
        SimpleNamespace(cpu_python=cpu_python),
        cluster_id=CLUSTER,
        needle=NAME,
        nodes=(NODE,),
        observed_after=datetime(1970, 1, 1, tzinfo=timezone.utc),
        deadline_seconds=0,
    )

    findings = [item for response in responses for item in response["findings"]]
    assert sorted(item["device"] for item in findings) == sorted(GPUS), findings
    assert {item["metric_name"] for item in findings} == {
        "host_gpu_utilization_percent"
    }
    assert len(notifier.notifications) == 1, notifier.notifications
    assert [
        (item["low_gpu_utilization"], item["low_cpu_utilization"]) for item in records
    ] == [(True, False)], records
    assert records[0]["gpu_devices"] == sorted(GPUS)
    assert polls[-1]["low_cpu_notification_count"] == 0, polls
    checks = phase_checks(
        {
            "nodes": [NODE],
            "pods": [{"node": NODE}],
            "metadata_errors": [],
            "notifications": records,
            "polls": polls,
        }
    )
    assert all(checks.values()), checks


def test_a_low_cpu_mail_for_the_phase_fails_the_cpu_rule_check() -> None:
    record = {
        "matched_nodes": ["node-a"],
        "status": "SENT",
        "low_gpu_utilization": True,
        "provider_message_id_present": True,
        "gpu_devices": ["GPU-a", "GPU-b"],
    }
    observation = {
        "nodes": ["node-a"],
        "pods": [{"node": "node-a"}],
        "metadata_errors": [],
        "notifications": [record],
    }

    quiet = phase_checks({**observation, "polls": [{"low_cpu_notification_count": 0}]})
    loud = phase_checks({**observation, "polls": [{"low_cpu_notification_count": 1}]})

    assert quiet["cpu_rule_silent"] is True, quiet
    assert loud["cpu_rule_silent"] is False, loud
    assert all(value for key, value in loud.items() if key != "cpu_rule_silent"), loud
