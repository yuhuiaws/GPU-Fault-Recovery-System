from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional.notify005_checks import node_errors, phase_checks

NODES = ("node-a", "node-b", "node-c")


def notification(node: str, status: str = "SENT") -> dict:
    return {
        "matched_nodes": [node],
        "status": status,
        "low_gpu_utilization": True,
        "provider_message_id_present": status == "SENT",
        "gpu_devices": ["GPU-a", "GPU-b"],
    }


@pytest.mark.parametrize(
    "records",
    [
        [],
        [notification("node-a", "PENDING")],
        [notification("other")],
        [notification("node-a"), notification("node-a")],
    ],
)
def test_phase_rejects_silence_unsent_wrong_node_and_duplicate(records) -> None:
    observation = {
        "nodes": ["node-a"],
        "pods": [{"node": "node-a"}],
        "metadata_errors": [],
        "notifications": records,
    }
    assert not all(phase_checks(observation).values()), observation


def test_node_requires_identity_readiness_and_eight_gpus() -> None:
    valid = {
        "uid": "u",
        "ready": "True",
        "unschedulable": False,
        "gpu_allocatable": "8",
        "ownership_annotations": {},
    }
    assert node_errors(valid) == []
    for key, value in (
        ("uid", None),
        ("ready", None),
        ("unschedulable", True),
        ("gpu_allocatable", "4"),
        ("ownership_annotations", {"owner": "other"}),
    ):
        assert node_errors({**valid, key: value}), {**valid, key: value}


def test_late_duplicate_is_observed_after_initial_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    rows = iter(
        [
            [notification("node-a")],
            [notification("node-a")],
            [notification("node-a"), notification("node-a")],
        ]
    )
    regional = SimpleNamespace(cpu_python=lambda *a: {"records": next(rows)})
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    records, polls = runner.wait_low_utilization_notifications(
        regional,
        cluster_id="c",
        needle="job",
        nodes=("node-a",),
        observed_after=datetime.now(timezone.utc),
        deadline_seconds=10,
        poll_seconds=1,
        settle_seconds=3,
    )
    assert len(records) == 2
    assert len(polls) == 3


@pytest.mark.parametrize(
    "first_sent,cleanup_fails", [(True, False), (False, False), (True, True)]
)
def test_two_phase_flow_stops_on_failure_and_keeps_cleanup_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_sent: bool,
    cleanup_fails: bool,
) -> None:
    created = []
    (tmp_path / "site").write_text("fixture-only\n")
    regional = SimpleNamespace(
        node_snapshot=lambda node: {
            "name": node,
            "uid": f"uid-{node}",
            "ready": "True",
            "unschedulable": False,
            "gpu_allocatable": "8",
            "ownership_annotations": {},
        },
        cpu_python=lambda *a: {"records": []},
        kubectl=lambda *a, **kw: "",
    )
    site = SimpleNamespace(
        regional=lambda _: regional,
        ready_pods=lambda *a: ["worker"],
        pod_json=lambda *a: {"duration_seconds": 1},
        site_file=tmp_path / "site",
    )

    class Fixture:
        resource = "job"

        def __init__(self, regional, settings, state_path=None):
            # The runner hands every managed workload an ownership state file.
            self.state_path = state_path
            self.settings = settings
            self.name = settings.job_id
            self.deleted = False
            created.append(self)

        def submit(self):
            return {}

        def workload(self):
            value = yaml.safe_load(self.settings.manifest.read_text())
            if value["kind"] == "PyTorchJob":
                for role, offset in (("Master", "0"), ("Worker", "1")):
                    metadata = value["spec"]["pytorchReplicaSpecs"][role]["template"][
                        "metadata"
                    ]
                    metadata["labels"]["gpu-fault.io/role"] = role.lower()
                    metadata["annotations"] = {
                        "gpu-fault.io/rank-offset": offset,
                        "gpu-fault.io/expected-critical-ranks": "3",
                    }
            return value

        def pods(self):
            return (
                []
                if self.deleted
                else [
                    {"node": node, "phase": "Running", "ready": True}
                    for node in NODES[: self.settings.expected_pods]
                ]
            )

        def delete(self):
            if cleanup_fails:
                raise RuntimeError("deletion refused")
            self.deleted = True

    monkeypatch.setattr(runner, "ManagedWorkloadFixture", Fixture)
    monkeypatch.setattr(runner, "foreign_gpu_reservations", lambda *a: [])
    monkeypatch.setattr(
        runner, "wait_low_utilization_latch_disarmed", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        runner,
        "wait_low_utilization_notifications",
        lambda *a, **kw: (
            [
                notification(node, "SENT" if first_sent else "PENDING")
                for node in kw["nodes"]
            ],
            [],
        ),
    )
    result = runner.run_notify005(
        site,
        SimpleNamespace(cluster_id="c"),
        nodes=NODES,
        case_dir=tmp_path,
        attempt=1,
        training_image="image@sha256:" + "a" * 64,
    )
    assert result["verdict"] == (
        "PASS" if first_sent and not cleanup_fails else "FAIL"
    ), result
    assert len(created) == (2 if first_sent and not cleanup_fails else 1)
    if cleanup_fails:
        assert result["cleanup_errors"]
        assert "deletion refused" in result["error"]
    else:
        assert all(fixture.deleted for fixture in created), {
            fixture.name: fixture.deleted for fixture in created
        }
