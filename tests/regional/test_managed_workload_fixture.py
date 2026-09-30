"""ManagedWorkloadFixture: the test workload the destructive cases submit."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault import training_submit_cli
from scripts.e2e.regional.managed_workload_fixture import (
    OWNER_LABEL,
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from tests.regional._workload_restart_support import (
    apply_restart_metadata,
    restart_state,
)

REGIONAL = Path(__file__).resolve().parents[2] / "scripts" / "e2e" / "regional"


def _regional(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


class Kubernetes:
    """An in-memory API at Fixture.run; no subprocess or kubeconfig is executed."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.deletes: list[tuple[str, str, dict[str, Any]]] = []
        self.created: list[dict[str, Any]] = []
        self.fail_create_ack = False
        self.fail_delete_ack = False
        self.fail_reads = False
        self.replace_before_delete = False

    def add(self, document: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(document)
        metadata = value["metadata"]
        metadata.setdefault("namespace", "gpu-fault-system")
        metadata.setdefault("uid", f"uid-{len(self.objects)}-{metadata['name']}")
        metadata.setdefault("resourceVersion", "1")
        self.objects[(value["kind"].lower(), metadata["name"])] = value
        return value

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("check", True) is True
        args = command[command.index("-n") + 2 :]
        output = ""
        if args[0] == "create":
            document = json.loads(kwargs["input_text"])
            key = (document["kind"].lower(), document["metadata"]["name"])
            if key in self.objects:
                raise RegionalFixtureError("AlreadyExists")
            self.created.append(copy.deepcopy(document))
            self.add(document)
            if self.fail_create_ack:
                raise RegionalFixtureError("create receipt lost")
        elif args[0] == "get":
            if self.fail_reads:
                raise RegionalFixtureError("API read failed")
            kinds = args[1].split(",")
            if kinds == ["node"]:
                output = json.dumps(
                    {
                        "items": [
                            {"metadata": {"name": "node-a"}, "status": {"images": []}}
                        ]
                    }
                )
            elif "-l" in args:
                label, value = args[args.index("-l") + 1].split("=", 1)
                output = json.dumps(
                    {
                        "items": [
                            obj
                            for (kind, _name), obj in self.objects.items()
                            if kind in kinds
                            and (obj["metadata"].get("labels") or {}).get(label)
                            == value
                        ]
                    }
                )
            else:
                document = self.objects.get((args[1], args[2]))
                output = json.dumps(document) if document is not None else ""
        elif args[0] == "delete":
            assert args[1] == "--raw", "name/label deletion bypasses UID fencing"
            plural, name = args[2].split("/")[-2:]
            kind = {
                "pods": "pod",
                "jobs": "job",
                "pytorchjobs": "pytorchjob",
                "services": "service",
            }[plural]
            options = json.loads(kwargs["input_text"])
            self.deletes.append((kind, name, options))
            current = self.objects[(kind, name)]
            if self.replace_before_delete:
                current["metadata"]["uid"] = "replacement"
            assert options["preconditions"] == {
                "uid": current["metadata"]["uid"],
                "resourceVersion": current["metadata"]["resourceVersion"],
            }, "API UID/resourceVersion precondition rejected the replacement"
            del self.objects[(kind, name)]
            if self.fail_delete_ack:
                raise RegionalFixtureError("delete receipt lost")
        elif args[0] != "wait":
            raise AssertionError(f"unexpected mocked command: {args}")
        return subprocess.CompletedProcess(command, 0, output, "")


def harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: str = "manifests/training/xid11-single-node-job.yaml",
) -> tuple[ManagedWorkloadFixture, Kubernetes, str]:
    site = tmp_path / "site.yaml"
    site.write_text("schemaVersion: 1\n", encoding="utf-8")
    regional = _regional(tmp_path)
    api = Kubernetes()
    monkeypatch.setattr(regional, "run", api.run)
    fixture = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=(REGIONAL / manifest),
            site_file=site,
            job_id="test-job",
            attempt_id="test-attempt",
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=1,
        ),
    )
    rendered = training_submit_cli.render_workload(
        fixture.settings.manifest,
        job_id="test-job",
        attempt_id="test-attempt",
        attempt_number=1,
        runtime_profile_version="profile-test",
        expected_critical_ranks=None,
        training_container=None,
        restart_budget=1,
        namespace=regional.settings.namespace,
    ).manifest
    return fixture, api, rendered


def test_unsubmitted_fixture_never_deletes_a_preexisting_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    existing = api.add(yaml.safe_load(rendered))
    fixture.delete()
    with pytest.raises(RegionalFixtureError, match="already exists"):
        fixture.submit_rendered(rendered)
    fixture.delete()
    assert api.deletes == []
    assert api.objects[(fixture.resource, fixture.name)] == existing


def test_submit_entrypoint_creates_and_cleans_owned_recovery_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        training_submit_cli,
        "load_site",
        lambda path: SimpleNamespace(
            release_config={"runtime_profile": {"version": "profile-test"}}
        ),
    )
    result = fixture.submit()
    assert result["create_only"] is True
    source = api.objects[(fixture.resource, fixture.name)]
    retry = copy.deepcopy(source)
    retry["metadata"].update({"name": fixture.name + "-r-123", "uid": "retry-uid"})
    state = restart_state(fixture, retry_parent_name=retry["metadata"]["name"])
    apply_restart_metadata(retry, state)
    api.add(retry)
    pod = api.add(
        {
            "kind": "Pod",
            "metadata": {
                "name": "training",
                "labels": retry["metadata"]["labels"],
                "ownerReferences": [
                    {
                        "apiVersion": retry["apiVersion"],
                        "kind": retry["kind"],
                        "name": retry["metadata"]["name"],
                        "uid": retry["metadata"]["uid"],
                        "controller": True,
                    }
                ],
            },
            "spec": {},
            "status": {},
        }
    )
    fixture.authorize_restart(state)
    fixture.delete()
    assert api.objects == {}
    assert api.deletes[-1][:2] == ("pod", "training")
    assert api.deletes[-1][2]["gracePeriodSeconds"] == 0
    assert all(
        options["propagationPolicy"] == "Orphan"
        for kind, _name, options in api.deletes
        if kind != "pod"
    )
    assert {
        options["preconditions"]["uid"] for _kind, _name, options in api.deletes
    } == {source["metadata"]["uid"], "retry-uid", pod["metadata"]["uid"]}


@pytest.mark.parametrize("defect", ["create-ack", "delete-ack"])
def test_fixture_cleanup_handles_lost_mutation_receipts(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    api.fail_create_ack = defect == "create-ack"
    api.fail_delete_ack = defect == "delete-ack"
    if api.fail_create_ack:
        with pytest.raises(RegionalFixtureError, match="receipt lost"):
            fixture.submit_rendered(rendered)
    else:
        fixture.submit_rendered(rendered)
    fixture.delete()
    assert api.objects == {}
    assert len(api.deletes) == 1


def test_cleanup_deletes_the_controllers_replica_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-09-24: 26 headless replica Services of deleted drill PyTorchJobs
    outlived their runs (the Orphan delete of the controller strips their
    ownerReferences) and blocked the uninstall as unregistered live resources."""

    fixture, api, rendered = harness(
        tmp_path, monkeypatch, "manifests/training/xid11-three-node-pytorchjob.yaml"
    )
    fixture.submit_rendered(rendered)
    job = next(obj for (kind, _n), obj in api.objects.items() if kind == "pytorchjob")
    for replica in ("master-0", "worker-0"):
        api.add(
            {
                "kind": "Service",
                "metadata": {
                    "name": f"{job['metadata']['name']}-{replica}",
                    "labels": {
                        "training.kubeflow.org/job-name": job["metadata"]["name"]
                    },
                    "ownerReferences": [
                        {
                            "apiVersion": "kubeflow.org/v1",
                            "kind": "PyTorchJob",
                            "name": job["metadata"]["name"],
                            "uid": job["metadata"]["uid"],
                        }
                    ],
                },
            }
        )
    api.add(
        {
            "kind": "Service",
            "metadata": {
                "name": "someone-elses-master-0",
                "labels": {"training.kubeflow.org/job-name": job["metadata"]["name"]},
                "ownerReferences": [
                    {
                        "apiVersion": "kubeflow.org/v1",
                        "kind": "PyTorchJob",
                        "name": "other",
                        "uid": "uid-other",
                    }
                ],
            },
        }
    )

    fixture.delete()

    assert set(api.objects) == {("service", "someone-elses-master-0")}, (
        "our controller's Services go; a Service another owner holds stays"
    )
    kinds = [kind for kind, _name, _options in api.deletes]
    assert kinds.index("pytorchjob") < kinds.index("service"), (
        "the controller is orphaned first, its Services are removed last"
    )


@pytest.mark.parametrize("defect", ["uid", "owner", "read", "delete-race"])
def test_cleanup_never_deletes_an_unknown_or_replaced_resource(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    current = api.objects[(fixture.resource, fixture.name)]
    if defect == "uid":
        current["metadata"]["uid"] = "replacement"
    elif defect == "owner":
        current["metadata"]["labels"][OWNER_LABEL] = "other-run"
    elif defect == "read":
        api.fail_reads = True
    else:
        api.replace_before_delete = True
    with pytest.raises((RegionalFixtureError, AssertionError)):
        fixture.delete()
    assert (fixture.resource, fixture.name) in api.objects
    if defect != "delete-race":
        assert api.deletes == []


@pytest.mark.parametrize(
    "defect", ["condition", "missing-container", "extra-container", "deleting", "none"]
)
def test_workload_readiness_requires_exact_running_pod_container_sets(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    pod = api.add(
        {
            "kind": "Pod",
            "metadata": {
                "name": "training",
                "labels": api.created[0]["metadata"]["labels"],
            },
            "spec": {"nodeName": "node-a", "containers": [{"name": "training"}]},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [{"name": "training", "ready": True}],
            },
        }
    )
    if defect == "condition":
        pod["status"]["conditions"] = []
    elif defect == "missing-container":
        pod["status"]["containerStatuses"] = []
    elif defect == "extra-container":
        pod["status"]["containerStatuses"].append({"name": "extra", "ready": True})
    elif defect == "deleting":
        pod["metadata"]["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    assert fixture.pods_healthy(fixture.pods()) is (defect == "none")


@pytest.mark.parametrize("defect", ["none", "create-ack", "uid", "read", "occupied"])
def test_prewarm_ownership_cleanup_and_ack_loss(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    prewarm = ImagePrewarmFixture(
        fixture.regional, case_id="case-test", run_id="run-test"
    )
    if defect == "occupied":
        api.add(prewarm.manifest("node-a", 0))
    api.fail_create_ack = defect == "create-ack"
    if defect in {"create-ack", "occupied"}:
        with pytest.raises(RegionalFixtureError):
            prewarm.create(["node-a"])
    else:
        prewarm.create(["node-a"])
    if defect == "uid":
        next(iter(api.objects.values()))["metadata"]["uid"] = "replacement"
    elif defect == "read":
        api.fail_reads = True
    if defect in {"uid", "read"}:
        with pytest.raises(RegionalFixtureError, match="cleanup is unproven"):
            prewarm.cleanup()
        assert api.deletes == []
    else:
        assert not any(prewarm.cleanup().values()), (
            "successful owned prewarm cleanup must report no residual resources"
        )
        assert len(api.objects) == (1 if defect == "occupied" else 0)
