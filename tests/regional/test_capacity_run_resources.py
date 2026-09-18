"""Run-labelled resource receipts through a fake Kubernetes API."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.perf.regional_capacity_resources import RunResources, run_manifest


class ResourceAPI:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.pending: dict[str, Any] | None = None
        self.create_outcome = "success"
        self.patch_ack_lost = False
        self.events: list[str] = []

    def __call__(self, *args: str, **kwargs: Any) -> str:
        assert kwargs.get("check", True), "resource API errors must be checked"
        if args[0] == "get":
            value = self.objects.get((args[1], args[2]))
            if value is None:
                return ""
            return json.dumps(
                value["metadata"] if args[-1] == "jsonpath={.metadata}" else value
            )
        if args[0] == "create":
            value = json.loads(kwargs["stdin"])
            key = (value["kind"].lower(), value["metadata"]["name"])
            value["metadata"].update(uid=f"uid-{key[0]}-{key[1]}", resourceVersion="1")
            self.events.append(f"create/{key[0]}")
            if self.create_outcome == "pending":
                self.pending = value
            else:
                self.objects[key] = value
            if self.create_outcome != "success":
                raise TimeoutError("create acknowledgement lost")
            return "created"
        if args[0] == "patch":
            value = self.objects[("configmap", args[2])]
            patches = json.loads(kwargs["stdin"])
            assert patches[:2] == [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": value["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": value["metadata"]["resourceVersion"],
                },
            ]
            self.events.append("patch/configmap")
            value["data"] = patches[2]["value"]
            value["metadata"]["resourceVersion"] = "2"
            if self.patch_ack_lost:
                raise TimeoutError("patch acknowledgement lost")
            return "patched"
        assert args[:2] == ("delete", "--raw"), (
            f"unexpected resource mutation: {args[:2]}"
        )
        options = json.loads(kwargs["stdin"])
        uid = options["preconditions"]["uid"]
        key = next(
            key
            for key, value in self.objects.items()
            if value["metadata"]["uid"] == uid
        )
        assert options["propagationPolicy"] == "Foreground"
        self.events.append(f"delete/{key[0]}")
        del self.objects[key]
        return "deleted"


def manifest(kind: str, name: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": kind,
        "metadata": {"name": name, "namespace": "perf-ns"},
    }
    if kind == "Job":
        value["spec"] = {"template": {"spec": {"containers": []}}}
    if kind == "ConfigMap":
        value["data"] = {"start_epoch": ""}
    return value


@pytest.mark.parametrize("outcome", ["success", "committed-ack-loss"])
def test_created_uid_receipts_survive_lost_ack_and_reload(
    tmp_path: Path, outcome: str
) -> None:
    api = ResourceAPI()
    api.create_outcome = outcome
    resources = RunResources(tmp_path, "run-a", "perf-ns", api)
    if outcome == "success":
        resources.create(manifest("Job", "load"))
    else:
        with pytest.raises(TimeoutError):
            resources.create(manifest("Job", "load"))
    receipt = json.loads((tmp_path / "capacity-resources.json").read_text())
    assert receipt["resources"]["job/load"]["uid"] == "uid-job-load"
    assert api.objects[("job", "load")]["spec"]["template"]["metadata"]["labels"] == {
        "gpu-fault.io/acceptance-run": "run-a"
    }
    RunResources(tmp_path, "run-a", "perf-ns", api).delete_all()
    assert api.objects == {}
    assert api.events == ["create/job", "delete/job"]


def test_unknown_create_uid_cannot_be_cleaned_by_an_empty_read(tmp_path: Path) -> None:
    api = ResourceAPI()
    api.create_outcome = "pending"
    resources = RunResources(tmp_path, "run-a", "perf-ns", api)
    with pytest.raises(TimeoutError):
        resources.create(manifest("Job", "load"))
    with pytest.raises(RuntimeError, match="acknowledgement is unresolved"):
        RunResources(tmp_path, "run-a", "perf-ns", api).delete_all()
    assert api.events == ["create/job"]
    assert api.pending is not None, "the simulated API create must still be pending"
    api.objects[("job", "load")] = api.pending
    RunResources(tmp_path, "run-a", "perf-ns", api).delete_all()
    assert api.objects == {}


def test_recreated_resource_with_same_run_label_is_not_deleted(tmp_path: Path) -> None:
    api = ResourceAPI()
    resources = RunResources(tmp_path, "run-a", "perf-ns", api)
    resources.create(manifest("Job", "load"))
    api.objects[("job", "load")]["metadata"]["uid"] = "replacement"
    with pytest.raises(RuntimeError, match="UID changed"):
        resources.delete_all()
    assert api.events == ["create/job"]


def test_all_claimants_stop_before_independent_resources(tmp_path: Path) -> None:
    api = ResourceAPI()
    resources = RunResources(tmp_path, "run-a", "perf-ns", api)
    for kind in ("ConfigMap", "Job", "Pod"):
        resources.create(manifest(kind, kind.lower()))
    resources.delete_all()
    assert api.events[3:] == ["delete/job", "delete/pod", "delete/configmap"]


@pytest.mark.parametrize("lost_ack", [False, True])
def test_start_gate_uses_uid_cas_and_confirms_lost_ack(
    tmp_path: Path, lost_ack: bool
) -> None:
    api = ResourceAPI()
    resources = RunResources(tmp_path, "run-a", "perf-ns", api)
    resources.create(manifest("ConfigMap", "start"))
    api.patch_ack_lost = lost_ack
    resources.update_data("start", {"start_epoch": "123.000000"})
    assert api.events == ["create/configmap", "patch/configmap"]
    api.objects[("configmap", "start")]["data"] = {"start_epoch": "operator-value"}
    with pytest.raises(RuntimeError, match="outside the run"):
        resources.update_data("start", {"start_epoch": "456.000000"})
    assert api.events == ["create/configmap", "patch/configmap"]


def test_labelled_ha_manifest_keeps_original_annotations() -> None:
    original = manifest("Pod", "ha-probe")
    original["metadata"]["annotations"] = {"ha-owner": "kept"}
    before = copy.deepcopy(original)
    value = run_manifest(original, "ha-run")
    assert value["metadata"]["labels"] == {"gpu-fault.io/acceptance-run": "ha-run"}
    assert value["metadata"]["annotations"] == {"ha-owner": "kept"}
    assert original == before
