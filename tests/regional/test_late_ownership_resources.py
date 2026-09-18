from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts.e2e.regional import late_ownership_resources as resources
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from tests.regional._late_ownership_support import scope


class Api:
    def __init__(self, binding):
        self.source = {
            "kind": "PyTorchJob",
            "metadata": {
                "name": binding.workload.name,
                "namespace": binding.workload.namespace,
                "uid": binding.workload.uid,
                "resourceVersion": "1",
            },
        }
        self.objects = {("pytorchjob", binding.workload.name): deepcopy(self.source)}
        self.calls = []
        self.mode = None

    def read(self, kind, name):
        return deepcopy(self.objects.get((kind, name)))

    def kubectl(self, side, *args, **kwargs):
        assert side == "gpu", "ownership mutation escaped the GPU-local API"
        self.calls.append((args, kwargs))
        if args[0] == "create":
            value = json.loads(kwargs["input_text"])
            value["metadata"].update(uid="created-uid", resourceVersion="2")
            self.objects[(value["kind"].lower(), value["metadata"]["name"])] = deepcopy(
                value
            )
            if self.mode == "lost-create":
                raise OSError("unacknowledged create")
            if self.mode == "bad-name":
                value["metadata"]["name"] = "other"
            elif self.mode == "bad-label":
                value["metadata"]["labels"] = {}
            elif self.mode == "bad-uid":
                value["metadata"]["uid"] = ""
            elif self.mode == "gone-after-create":
                self.objects.clear()
            return json.dumps(value)
        if args[0] == "patch":
            assert args[3] == "--type=json"
            key = (args[1], args[2])
            value = self.objects[key]
            patches = json.loads(args[5])
            assert patches[0] == {
                "op": "test",
                "path": "/metadata/uid",
                "value": value["metadata"]["uid"],
            }
            assert patches[1] == {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": value["metadata"]["resourceVersion"],
            }
            if self.mode == "lost-patch-before":
                raise OSError("unacknowledged patch")
            for item in patches[2:]:
                name = item["path"].split("/")[-1]
                if item["op"] == "test":
                    assert value["metadata"][name] == item["value"]
                elif item["op"] == "remove":
                    value["metadata"].pop(name)
                else:
                    value["metadata"][name] = item["value"]
            if self.mode == "lost-patch-after":
                raise OSError("unacknowledged patch")
            return "{}"
        assert args[0] == "wait"
        if self.mode == "gone-after-ready":
            self.objects.pop(("pod", args[2].removeprefix("pod/")))
        return ""

    def delete(self, resource):
        self.calls.append(("delete", resource))
        key = (resource["kind"].lower(), resource["metadata"]["name"])
        assert self.objects[key]["metadata"]["uid"] == resource["metadata"]["uid"]
        if self.mode != "delete-remains":
            self.objects.pop(key)


@pytest.fixture
def owned(monkeypatch, tmp_path):
    binding = scope()
    api = Api(binding)
    monkeypatch.setattr(
        resources,
        "read_resource",
        lambda regional, kind, name: regional.read(kind, name),
    )
    monkeypatch.setattr(
        resources, "delete_resource", lambda regional, value: regional.delete(value)
    )
    mutation = resources.OwnedMutation(
        api, binding, api.source, "approved-image@sha256:local"
    )
    mutation.journal_path = tmp_path / "ownership.json"
    return api, mutation


def test_real_uid_and_resource_version_patch_restores_only_the_owned_source(owned):
    api, mutation = owned
    mutation.change_owner()
    assert (
        mutation.source_object()["metadata"]["ownerReferences"]
        == mutation.injected_owners
    )
    journal = json.loads(mutation.journal_path.read_text())
    assert journal["scope_sha256"] == mutation.scope.digest()
    assert journal["resources"][0]["uid"] == "created-uid"
    assert journal["mutation_started"] is True
    create = next(kwargs for args, kwargs in api.calls if args[0] == "create")
    anchor = json.loads(create["input_text"])
    assert anchor["spec"]["suspend"] is True
    assert anchor["spec"]["template"]["spec"]["automountServiceAccountToken"] is False
    mutation.cleanup()
    assert mutation.source_object() == api.source
    assert len(api.objects) == 1
    assert json.loads(mutation.journal_path.read_text())["mutation_started"] is False
    mutation.cleanup()


def test_preexisting_original_owner_is_restored_exactly(owned):
    api, mutation = owned
    original = [{"uid": "original", "name": "parent", "controller": False}]
    api.source["metadata"]["ownerReferences"] = original
    mutation.source = deepcopy(api.source)
    api.objects[("pytorchjob", mutation.scope.workload.name)] = deepcopy(api.source)
    mutation.change_owner()
    mutation.cleanup()
    assert mutation.source_object()["metadata"]["ownerReferences"] == original


def test_late_sibling_is_real_cuda_and_uid_bound_not_a_model_counter(owned):
    api, mutation = owned
    uid = mutation.late_sibling()
    assert uid == "created-uid"
    pod = api.objects[("pod", mutation.resources[0]["name"])]
    assert pod["metadata"]["ownerReferences"][0]["uid"] == mutation.scope.workload.uid
    assert pod["metadata"]["ownerReferences"][0]["controller"] is True
    assert pod["spec"]["nodeName"] == mutation.scope.nodes[1].name
    assert pod["spec"]["automountServiceAccountToken"] is False
    assert pod["spec"]["activeDeadlineSeconds"] == 240
    container = pod["spec"]["containers"][0]
    assert container["image"] == mutation.image
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "torch.ones(1,device='cuda')" in container["command"][-1]
    assert "torch.cuda.synchronize()" in container["command"][-1]
    assert container["readinessProbe"]["exec"]["command"] == [
        "test",
        "-f",
        "/tmp/late-ownership-ready",
    ]
    mutation.cleanup()
    assert list(api.objects) == [("pytorchjob", mutation.scope.workload.name)]


@pytest.mark.parametrize("method", ["change_owner", "late_sibling"])
@pytest.mark.parametrize("defect", ["gone", "uid", "owner"])
def test_source_replacement_is_rejected_before_any_mutation(owned, method, defect):
    api, mutation = owned
    source = api.objects[("pytorchjob", mutation.scope.workload.name)]
    if defect == "gone":
        api.objects.clear()
    elif defect == "uid":
        source["metadata"]["uid"] = "replaced"
    else:
        source["metadata"]["ownerReferences"] = [{"uid": "replaced"}]
    with pytest.raises(BoundaryDenied):
        getattr(mutation, method)()
    assert api.calls == []


@pytest.mark.parametrize(
    "defect", ["bad-name", "bad-label", "bad-uid", "gone-after-create", "lost-create"]
)
def test_unacknowledged_creation_is_not_adopted_by_cleanup(owned, defect):
    api, mutation = owned
    api.mode = defect
    with pytest.raises((BoundaryDenied, OSError)):
        mutation.change_owner()
    if defect != "gone-after-create":
        with pytest.raises(BoundaryDenied):
            mutation.cleanup()
        assert not any(args == "delete" for args, _kwargs in api.calls), (
            "an unacknowledged resource must never be deleted by name"
        )
    else:
        mutation.cleanup()
    assert not any(args[0] == "patch" for args, _kwargs in api.calls), (
        "unacknowledged creation must not authorize a source-owner patch"
    )


def test_existing_named_resource_is_not_owned_by_this_run(owned):
    api, mutation = owned
    name = f"late-owner-{mutation.scope.challenge[:12]}"
    api.objects[("job", name)] = {"metadata": {"uid": "foreign"}}
    with pytest.raises(BoundaryDenied, match="already exists"):
        mutation.change_owner()
    assert api.calls == [] and mutation.resources == []


@pytest.mark.parametrize("when", ["before", "after"])
def test_lost_patch_ack_cleanup_uses_fresh_owner_state_and_never_replays_blindly(
    owned, when
):
    api, mutation = owned
    api.mode = f"lost-patch-{when}"
    with pytest.raises(OSError):
        mutation.change_owner()
    assert mutation.mutation_started, (
        "lost patch acknowledgement must retain durable mutation intent"
    )
    api.mode = None
    mutation.cleanup()
    assert mutation.source_object() == api.source and len(api.objects) == 1
    patches = [args for args, _kwargs in api.calls if args[0] == "patch"]
    assert len(patches) == (1 if when == "before" else 2)


@pytest.mark.parametrize("defect", ["uid", "owner", "label", "missing-uid"])
def test_replaced_owned_resource_cannot_be_deleted(owned, defect):
    api, mutation = owned
    mutation.late_sibling()
    entry = mutation.resources[0]
    value = api.objects[(entry["kind"], entry["name"])]
    if defect == "uid":
        value["metadata"]["uid"] = "replaced"
    elif defect == "owner":
        value["metadata"]["ownerReferences"] = []
    elif defect == "label":
        value["metadata"]["labels"][resources.OWNER_LABEL] = "other"
    else:
        entry["uid"] = None
    with pytest.raises(BoundaryDenied, match="UID"):
        mutation.cleanup()
    assert not any(args == "delete" for args, _kwargs in api.calls), (
        "cleanup must preserve a resource whose ownership changed"
    )


def test_replaced_injected_source_owner_is_not_overwritten(owned):
    api, mutation = owned
    mutation.change_owner()
    api.objects[("pytorchjob", mutation.scope.workload.name)]["metadata"][
        "ownerReferences"
    ] = [{"uid": "foreign"}]
    before = len(api.calls)
    with pytest.raises(BoundaryDenied, match="replaced"):
        mutation.cleanup()
    assert len(api.calls) == before


@pytest.mark.parametrize("mode", ["gone-after-ready", "delete-remains"])
def test_missing_client_or_cleanup_residual_prevents_completion(owned, mode):
    api, mutation = owned
    api.mode = mode
    if mode == "gone-after-ready":
        with pytest.raises(BoundaryDenied, match="disappeared"):
            mutation.late_sibling()
    else:
        mutation.late_sibling()
        with pytest.raises(BoundaryDenied, match="remains"):
            mutation.cleanup()


def test_unknown_creation_absence_is_not_a_delete_receipt(owned):
    api, mutation = owned
    mutation.resources.append({"kind": "job", "name": "unknown", "uid": None})
    with pytest.raises(BoundaryDenied, match="never acknowledged"):
        mutation.cleanup()
    assert api.calls == []
