"""Resource-boundary regressions using the existing fake API and real I/O helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote

import pytest

from scripts.e2e.regional import late_ownership_resources as resources
from scripts.e2e.regional import managed_workload_fixture as managed
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional.test_late_ownership_resources import owned as owned

Before = Callable[[tuple[str, ...]], None]
After = Callable[[tuple[str, ...], str], str]


@dataclass
class BoundaryApi:
    api: Any
    mutation: resources.OwnedMutation
    before: Before | None = None
    after: After | None = None
    ignore_patch: bool = False
    delete_timeout: str = ""
    keep_deleted: bool = False
    clock: float = 0
    events: list[tuple[tuple[str, ...], dict[str, Any]]] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def source(self) -> dict[str, Any]:
        value: dict[str, Any] = self.api.objects[
            "pytorchjob", self.mutation.scope.workload.name
        ]
        return value

    def journal(self) -> dict[str, Any]:
        path = self.mutation.journal_path
        assert path is not None, "the test requires the real mutation journal"
        value: dict[str, Any] = json.loads(path.read_text())
        return value

    def mutation_calls(self) -> list[tuple[str, ...]]:
        return [
            args for args, _ in self.events if args[0] in {"create", "patch", "delete"}
        ]


@pytest.fixture
def edge(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> BoundaryApi:
    api, mutation = request.getfixturevalue("owned")
    api.settings = SimpleNamespace(namespace=mutation.scope.workload.namespace)
    boundary = BoundaryApi(api, mutation)
    original = api.kubectl

    def kube(plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "gpu", "mutation must remain on the local GPU API"
        assert (
            kwargs["timeout"]
            == {"get": 60, "create": 60, "patch": 60, "wait": 40, "delete": 120}[
                args[0]
            ]
        ), "each request must retain its bounded timeout"
        boundary.events.append((args, kwargs))
        if boundary.before is not None:
            boundary.before(args)
        if args[0] == "get":
            value = api.read(args[1], args[2])
            result = "" if value is None else json.dumps(value)
        elif args[0] == "delete":
            assert args[1] == "--raw", "DELETE must carry explicit preconditions"
            parts = args[2].split("/")
            namespace, plural, name = parts[-3:]
            assert unquote(namespace) == mutation.scope.workload.namespace
            kind = next(k for k, (_, p) in managed.RESOURCE_APIS.items() if p == plural)
            key = kind, unquote(name)
            value = api.objects[key]
            options = json.loads(kwargs["input_text"])
            assert options["propagationPolicy"] == "Foreground"
            if kind == "pod":
                assert options["gracePeriodSeconds"] == 0
            if options["preconditions"] != {
                "uid": value["metadata"]["uid"],
                "resourceVersion": value["metadata"]["resourceVersion"],
            }:
                raise RegionalFixtureError("controlled DELETE UID/RV conflict")
            if boundary.delete_timeout == "before":
                raise TimeoutError("controlled DELETE timeout before commit")
            if not boundary.keep_deleted:
                del api.objects[key]
            if boundary.delete_timeout == "after":
                raise TimeoutError("controlled DELETE timeout after commit")
            result = "{}"
        elif args[0] == "patch":
            current = api.objects[args[1], args[2]]
            patch = json.loads(args[5])
            for operation in patch:
                if operation["op"] == "test":
                    name = operation["path"].removeprefix("/metadata/")
                    if current["metadata"].get(name) != operation["value"]:
                        raise RegionalFixtureError(
                            "controlled PATCH UID/RV/owner conflict"
                        )
            before = deepcopy(current)
            try:
                result = (
                    "{}" if boundary.ignore_patch else original(plane, *args, **kwargs)
                )
            finally:
                if current != before:
                    current["metadata"]["resourceVersion"] = str(
                        int(before["metadata"]["resourceVersion"]) + 1
                    )
        else:
            result = original(plane, *args, **kwargs)
        if boundary.after is not None:
            result = boundary.after(args, result)
        return str(result)

    monkeypatch.setattr(api, "kubectl", kube)
    monkeypatch.setattr(resources, "read_resource", managed.read_resource)
    monkeypatch.setattr(resources, "delete_resource", managed.delete_resource)
    monkeypatch.setattr(
        managed,
        "time",
        SimpleNamespace(monotonic=lambda: boundary.clock, sleep=boundary.sleep),
    )
    return boundary


def test_journal_intent_and_uid_ack_precede_callbacks_and_readback(
    edge: BoundaryApi,
) -> None:
    seen: list[str] = []

    def inspect(args: tuple[str, ...]) -> None:
        if args[0] == "create":
            journal = edge.journal()
            assert journal["resources"][-1]["uid"] is None
            assert journal["mutation_started"] is False
            assert journal["scope_sha256"] == edge.mutation.scope.digest()
            seen.append("create-intent")
        elif args[0] == "get" and args[1] == "job" and edge.api.read(args[1], args[2]):
            assert edge.journal()["resources"][-1]["uid"] == "created-uid"
            seen.append("durable-uid-before-readback")
        elif args[0] == "patch":
            journal = edge.journal()
            assert journal["mutation_started"] is True
            assert journal["injected_owners"] == edge.mutation.injected_owners
            seen.append("owner-intent")

    edge.before = inspect
    edge.mutation.change_owner()
    assert seen[:3] == ["create-intent", "durable-uid-before-readback", "owner-intent"]
    assert edge.journal()["mutation_started"] is True
    edge.before = None
    edge.mutation.cleanup()
    assert edge.journal()["mutation_started"] is False
    assert edge.source()["metadata"].get("ownerReferences") is None


def test_no_journal_path_still_uses_owned_conditional_cleanup(
    edge: BoundaryApi,
) -> None:
    edge.mutation.journal_path = None
    edge.mutation.change_owner()
    edge.mutation.cleanup()
    assert edge.source()["metadata"].get("ownerReferences") is None
    assert len(edge.api.objects) == 1


def test_successful_patch_ack_without_owner_readback_is_not_success(
    edge: BoundaryApi,
) -> None:
    edge.ignore_patch = True
    with pytest.raises(
        BoundaryDenied, match="controlled owner change was not observed"
    ):
        edge.mutation.change_owner()
    assert edge.mutation.mutation_started is True
    assert edge.journal()["mutation_started"] is True
    edge.ignore_patch = False
    edge.mutation.cleanup()
    assert len(edge.api.objects) == 1
    assert sum(args[0] == "patch" for args in edge.mutation_calls()) == 1


def test_restore_readback_failure_retains_anchor_and_retry_uses_fresh_version(
    edge: BoundaryApi,
) -> None:
    edge.mutation.change_owner()
    edge.ignore_patch = True
    before = edge.source()["metadata"]["resourceVersion"]
    with pytest.raises(BoundaryDenied, match="original source owner was not restored"):
        edge.mutation.cleanup()
    assert edge.mutation.mutation_started is True
    assert edge.journal()["mutation_started"] is True
    assert not any(args[0] == "delete" for args in edge.mutation_calls()), edge.events
    assert edge.source()["metadata"]["ownerReferences"] == edge.mutation.injected_owners
    edge.ignore_patch = False
    edge.mutation.cleanup()
    assert int(edge.source()["metadata"]["resourceVersion"]) > int(before)
    assert edge.journal()["mutation_started"] is False
    assert len(edge.api.objects) == 1


@pytest.mark.parametrize("phase", ["change", "restore"])
def test_concurrent_source_owner_version_change_is_not_overwritten(
    edge: BoundaryApi, phase: str
) -> None:
    if phase == "restore":
        edge.mutation.change_owner()
    foreign = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": "foreign",
            "uid": "foreign-uid",
        }
    ]

    def race(args: tuple[str, ...]) -> None:
        if args[0] == "patch":
            edge.source()["metadata"]["ownerReferences"] = deepcopy(foreign)
            edge.source()["metadata"]["resourceVersion"] = "91"

    edge.before = race
    with pytest.raises(RegionalFixtureError, match="PATCH UID/RV/owner conflict"):
        if phase == "change":
            edge.mutation.change_owner()
        else:
            edge.mutation.cleanup()
    assert edge.source()["metadata"]["ownerReferences"] == foreign
    assert not any(args[0] == "delete" for args in edge.mutation_calls()), edge.events


@pytest.mark.parametrize("field", ["uid", "resourceVersion"])
def test_delete_rechecks_exact_uid_and_latest_version(
    edge: BoundaryApi, field: str
) -> None:
    edge.mutation.late_sibling()
    entry = edge.mutation.resources[-1]

    def race(args: tuple[str, ...]) -> None:
        if args[0] == "delete":
            edge.api.objects[entry["kind"], entry["name"]]["metadata"][field] = (
                "foreign-91"
            )

    edge.before = race
    with pytest.raises(RegionalFixtureError, match="DELETE UID/RV conflict"):
        edge.mutation.cleanup()
    assert (
        edge.api.objects[entry["kind"], entry["name"]]["metadata"][field]
        == "foreign-91"
    )


@pytest.mark.parametrize("when", ["before", "after"])
def test_delete_timeout_needs_actual_absence_not_a_swallowed_error(
    edge: BoundaryApi, when: str
) -> None:
    edge.mutation.late_sibling()
    edge.delete_timeout = when
    if when == "before":
        with pytest.raises(TimeoutError):
            edge.mutation.cleanup()
        assert len(edge.api.objects) == 2
        edge.delete_timeout = ""
    edge.mutation.cleanup()
    assert len(edge.api.objects) == 1
    deletes = sum(args[0] == "delete" for args in edge.mutation_calls())
    edge.mutation.cleanup()
    assert sum(args[0] == "delete" for args in edge.mutation_calls()) == deletes


def test_delete_success_with_residual_is_bounded_and_not_completed(
    edge: BoundaryApi,
) -> None:
    edge.mutation.late_sibling()
    edge.keep_deleted = True
    with pytest.raises(RegionalFixtureError, match="deletion was not confirmed"):
        edge.mutation.cleanup()
    assert edge.clock == 120
    assert len(edge.api.objects) == 2
    edge.keep_deleted = False
    edge.mutation.cleanup()
    assert len(edge.api.objects) == 1


def test_wait_timeout_keeps_the_acknowledged_pod_cleanup_authority(
    edge: BoundaryApi,
) -> None:
    def unavailable(args: tuple[str, ...]) -> None:
        if args[0] == "wait":
            raise TimeoutError("controlled readiness timeout")

    edge.before = unavailable
    with pytest.raises(TimeoutError):
        edge.mutation.late_sibling()
    assert edge.journal()["resources"][-1]["uid"] == "created-uid"
    assert len(edge.api.objects) == 2
    edge.before = None
    edge.mutation.cleanup()
    assert len(edge.api.objects) == 1


@pytest.mark.parametrize(
    "field", ["uid", "resourceVersion", "namespace", "name", "kind"]
)
def test_readback_identity_uses_the_real_shared_validator(
    edge: BoundaryApi, field: str
) -> None:
    def corrupt(args: tuple[str, ...], raw: str) -> str:
        if args[:2] == ("get", "pod") and raw:
            value = json.loads(raw)
            if field == "kind":
                value["kind"] = "Job"
            else:
                value["metadata"][field] = (
                    None if field in {"uid", "resourceVersion"} else "foreign"
                )
            return json.dumps(value)
        return raw

    edge.after = corrupt
    with pytest.raises(RegionalFixtureError, match="identity"):
        edge.mutation.late_sibling()
    assert not any(args[0] == "wait" for args, _ in edge.events), edge.events
    assert edge.journal()["resources"][-1]["uid"] == "created-uid"


@pytest.mark.parametrize("field", ["namespace", "kind", "resourceVersion"])
def test_malformed_create_ack_cannot_be_repaired_by_readback(
    edge: BoundaryApi, field: str
) -> None:
    def bad_ack(args: tuple[str, ...], raw: str) -> str:
        if args[0] == "create":
            value = json.loads(raw)
            if field == "kind":
                value["kind"] = "Pod"
            else:
                value["metadata"][field] = (
                    "foreign-namespace" if field == "namespace" else None
                )
            return json.dumps(value)
        return raw

    edge.after = bad_ack
    with pytest.raises((BoundaryDenied, RegionalFixtureError)):
        edge.mutation.change_owner()
    assert not any(args[0] == "patch" for args in edge.mutation_calls()), edge.events


@pytest.mark.parametrize(
    "field",
    [
        "nodeName",
        "image",
        "command",
        "activeDeadlineSeconds",
        "automountServiceAccountToken",
    ],
)
def test_admitted_sibling_spec_drift_cannot_be_acknowledged_as_the_owned_holder(
    edge: BoundaryApi, field: str
) -> None:
    def drift(args: tuple[str, ...], raw: str) -> str:
        if args[0] == "create":
            value = json.loads(raw)
            spec = value["spec"]
            if field == "image":
                spec["containers"][0]["image"] = "foreign-image"
            elif field == "command":
                spec["containers"][0]["command"] = ["foreign-command"]
            else:
                spec[field] = {
                    "nodeName": "unapproved-node",
                    "activeDeadlineSeconds": 2400,
                    "automountServiceAccountToken": True,
                }[field]
            edge.api.objects["pod", value["metadata"]["name"]] = deepcopy(value)
            return json.dumps(value)
        return raw

    edge.after = drift
    with pytest.raises(BoundaryDenied):
        edge.mutation.late_sibling()
    assert not any(args[0] == "wait" for args, _ in edge.events), edge.events


def test_an_unsuspended_anchor_cannot_authorize_source_owner_mutation(
    edge: BoundaryApi,
) -> None:
    def unsuspend(args: tuple[str, ...], raw: str) -> str:
        if args[0] == "create":
            value = json.loads(raw)
            value["spec"]["suspend"] = False
            edge.api.objects["job", value["metadata"]["name"]] = deepcopy(value)
            return json.dumps(value)
        return raw

    edge.after = unsuspend
    with pytest.raises(BoundaryDenied):
        edge.mutation.change_owner()
    assert not any(args[0] == "patch" for args in edge.mutation_calls()), edge.events


def test_replaced_owned_spec_is_not_deleted_using_only_uid_and_label(
    edge: BoundaryApi,
) -> None:
    edge.mutation.late_sibling()
    entry = edge.mutation.resources[-1]
    current = edge.api.objects[entry["kind"], entry["name"]]
    current["spec"]["containers"][0]["image"] = "foreign-image"
    current["metadata"]["resourceVersion"] = "92"
    before = list(edge.mutation_calls())
    with pytest.raises(BoundaryDenied):
        edge.mutation.cleanup()
    assert edge.mutation_calls() == before
    assert edge.api.objects[entry["kind"], entry["name"]] == current


def test_failed_intent_write_cannot_reach_the_api(
    edge: BoundaryApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(_path: Path, _document: dict[str, Any]) -> None:
        raise OSError("controlled journal failure")

    monkeypatch.setattr(resources, "write_json_atomic", unavailable)
    with pytest.raises(OSError, match="journal failure"):
        edge.mutation.change_owner()
    assert edge.mutation_calls() == []
    assert edge.mutation.mutation_started is False


def admission(edge: BoundaryApi, change: Callable[[dict[str, Any]], None]) -> None:
    def transform(args: tuple[str, ...], raw: str) -> str:
        if args[0] != "create":
            return raw
        value: dict[str, Any] = json.loads(raw)
        change(value)
        edge.api.objects[value["kind"].lower(), value["metadata"]["name"]] = deepcopy(
            value
        )
        return json.dumps(value)

    edge.after = transform


@pytest.mark.parametrize("reconstruct", [False, True])
def test_valid_identity_unapproved_spec_has_only_unchanged_deletion_custody(
    edge: BoundaryApi, reconstruct: bool
) -> None:
    admission(edge, lambda value: value["spec"].update(suspend=False))
    with pytest.raises(BoundaryDenied, match="declared spec"):
        edge.mutation.change_owner()
    entry = edge.journal()["resources"][0]
    assert entry["uid"] == "created-uid"
    assert entry["approved"] is False
    assert len(entry["ack_sha256"]) == 64
    assert entry["expected"]["spec"]["suspend"] is True
    assert edge.mutation.mutation_started is False
    assert not any(args[0] in {"patch", "wait"} for args, _ in edge.events), edge.events
    current = edge.api.objects[entry["kind"], entry["name"]]
    current["metadata"].update(resourceVersion="93", generation=2)
    current["status"] = {"active": 0}
    target = edge.mutation
    if reconstruct:
        saved = edge.journal()
        assert saved["scope_sha256"] == target.scope.digest()
        target = resources.OwnedMutation(
            target.regional,
            target.scope,
            deepcopy(target.source),
            target.image,
            resources=deepcopy(saved["resources"]),
            mutation_started=saved["mutation_started"],
            injected_owners=saved["injected_owners"],
            journal_path=target.journal_path,
        )
    target.cleanup()
    assert len(edge.api.objects) == 1
    assert edge.source()["metadata"].get("ownerReferences") is None


def test_unapproved_resource_changed_since_ack_is_not_deleted(
    edge: BoundaryApi,
) -> None:
    admission(edge, lambda value: value["spec"].update(suspend=False))
    with pytest.raises(BoundaryDenied):
        edge.mutation.change_owner()
    entry = edge.mutation.resources[0]
    edge.api.objects[entry["kind"], entry["name"]]["spec"]["backoffLimit"] = 9
    with pytest.raises(BoundaryDenied, match="deletion-only"):
        edge.mutation.cleanup()
    assert not any(args[0] == "delete" for args in edge.mutation_calls()), edge.events


@pytest.mark.parametrize(
    "raw", ["[]", '{"metadata":{},"metadata":{}}', '{"metadata":NaN}', "not-json"]
)
def test_malformed_json_ack_never_establishes_uid_custody(
    edge: BoundaryApi, raw: str
) -> None:
    edge.after = lambda args, value: raw if args[0] == "create" else value
    with pytest.raises(BoundaryDenied, match="not acknowledged"):
        edge.mutation.change_owner()
    entry = edge.journal()["resources"][0]
    assert entry["uid"] is None and entry["ack_sha256"] is None
    assert entry["approved"] is False
    with pytest.raises(BoundaryDenied, match="UID"):
        edge.mutation.cleanup()
    assert not any(args[0] in {"patch", "delete"} for args in edge.mutation_calls()), (
        edge.events
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("apiVersion", "batch/v9"),
        ("uid", 7),
        ("resourceVersion", True),
        ("deletionTimestamp", "2033-01-01T00:00:00Z"),
    ],
)
def test_ack_gvk_uid_version_and_deletion_state_are_strict(
    edge: BoundaryApi, field: str, value: Any
) -> None:
    def change(args: tuple[str, ...], raw: str) -> str:
        if args[0] == "create":
            document = json.loads(raw)
            target = document if field == "apiVersion" else document["metadata"]
            target[field] = value
            return json.dumps(document)
        return raw

    edge.after = change
    with pytest.raises(BoundaryDenied, match="not acknowledged"):
        edge.mutation.change_owner()
    assert edge.journal()["resources"][0]["uid"] is None
    assert not any(args[0] == "patch" for args in edge.mutation_calls()), edge.events


def pod_defaults(spec: dict[str, Any]) -> None:
    spec.update(
        dnsPolicy="ClusterFirst",
        schedulerName="default-scheduler",
        terminationGracePeriodSeconds=30,
        enableServiceLinks=True,
        securityContext={},
        serviceAccountName="default",
        serviceAccount="default",
        hostNetwork=False,
        hostPID=False,
        hostIPC=False,
        shareProcessNamespace=False,
        preemptionPolicy="PreemptLowerPriority",
        priority=0,
        nodeSelector={},
        affinity={},
        schedulingGates=[],
        tolerations=[
            {
                "key": "node.kubernetes.io/" + key,
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300,
            }
            for key in ("not-ready", "unreachable")
        ],
    )
    for container in spec["containers"]:
        container.update(
            imagePullPolicy="IfNotPresent",
            terminationMessagePath="/dev/termination-log",
            terminationMessagePolicy="File",
            securityContext={},
            stdin=False,
            stdinOnce=False,
            tty=False,
        )


@pytest.mark.parametrize("reverse_tolerations", [False, True])
def test_known_pod_defaults_and_equivalent_quantities_preserve_declared_spec(
    edge: BoundaryApi, reverse_tolerations: bool
) -> None:
    def defaulted(value: dict[str, Any]) -> None:
        pod_defaults(value["spec"])
        if reverse_tolerations:
            value["spec"]["tolerations"].reverse()
        container = value["spec"]["containers"][0]
        container["readinessProbe"].update(
            initialDelaySeconds=0, timeoutSeconds=1, successThreshold=1
        )
        container["resources"]["requests"].update(cpu="0.1", memory="0.5Gi")
        container["resources"]["limits"].update(
            cpu="1000m", memory="2048Mi", **{"nvidia.com/gpu": 1}
        )
        value["metadata"]["ownerReferences"][0].pop("blockOwnerDeletion")

    admission(edge, defaulted)
    assert edge.mutation.late_sibling() == "created-uid"
    assert edge.journal()["resources"][0]["approved"] is True
    edge.mutation.cleanup()
    assert len(edge.api.objects) == 1


@pytest.mark.parametrize(
    "selector_key", ["controller-uid", "batch.kubernetes.io/controller-uid"]
)
def test_job_defaults_bind_generated_selectors_and_labels_to_the_ack_uid(
    edge: BoundaryApi, selector_key: str
) -> None:
    def defaulted(value: dict[str, Any]) -> None:
        name, uid = value["metadata"]["name"], value["metadata"]["uid"]
        spec = value["spec"]
        spec.update(
            parallelism=1,
            completions=1,
            manualSelector=False,
            completionMode="NonIndexed",
            podReplacementPolicy="TerminatingOrFailed",
            managedBy="kubernetes.io/job-controller",
            selector={"matchLabels": {selector_key: uid}, "matchExpressions": []},
        )
        spec["template"]["metadata"] = {
            "creationTimestamp": None,
            "annotations": {},
            "labels": {
                "controller-uid": uid,
                "batch.kubernetes.io/controller-uid": uid,
                "job-name": name,
                "batch.kubernetes.io/job-name": name,
            },
        }
        pod_defaults(spec["template"]["spec"])
        spec["template"]["spec"]["containers"][0]["resources"] = {
            "requests": {},
            "limits": {},
        }

    admission(edge, defaulted)
    edge.mutation.change_owner()
    assert edge.journal()["resources"][0]["approved"] is True
    edge.mutation.cleanup()
    assert len(edge.api.objects) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("securityContext", {"privileged": True}),
        ("hostNetwork", True),
        ("dnsPolicy", "Default"),
        ("priority", False),
        ("volumes", [{"name": "foreign", "hostPath": {"path": "/"}}]),
        ("tolerations", [{"operator": "Exists"}]),
    ],
)
def test_unknown_or_changed_pod_defaults_are_not_execution_authority(
    edge: BoundaryApi, field: str, value: Any
) -> None:
    admission(edge, lambda document: document["spec"].update({field: value}))
    with pytest.raises(BoundaryDenied, match="declared spec"):
        edge.mutation.late_sibling()
    assert edge.journal()["resources"][0]["approved"] is False
    assert not any(args[0] == "wait" for args, _ in edge.events), edge.events


@pytest.mark.parametrize("quantity", [True, None, "NaN", "not-a-quantity", "2"])
def test_gpu_quantity_must_remain_finite_typed_and_equal(
    edge: BoundaryApi, quantity: Any
) -> None:
    def changed(value: dict[str, Any]) -> None:
        value["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = (
            quantity
        )

    admission(edge, changed)
    with pytest.raises(BoundaryDenied, match="quantity|declared spec"):
        edge.mutation.late_sibling()
    assert edge.journal()["resources"][0]["approved"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("spec", None),
        ("containers", None),
        ("container", None),
        ("resources", []),
        ("requests", []),
        ("readinessProbe", []),
        ("owners", False),
        ("owners", ["invalid"]),
        ("annotation", "foreign"),
    ],
)
def test_malformed_valid_identity_ack_has_no_execution_authority(
    edge: BoundaryApi, field: str, value: Any
) -> None:
    def changed(document: dict[str, Any]) -> None:
        if field == "spec":
            document["spec"] = value
        elif field == "containers":
            document["spec"]["containers"] = value
        elif field == "container":
            document["spec"]["containers"][0] = value
        elif field == "requests":
            document["spec"]["containers"][0]["resources"]["requests"] = value
        elif field == "owners":
            document["metadata"]["ownerReferences"] = value
        elif field == "annotation":
            document["metadata"]["annotations"]["gpu-fault.io/late-ownership-spec"] = (
                value
            )
        else:
            document["spec"]["containers"][0][field] = value

    admission(edge, changed)
    with pytest.raises(BoundaryDenied):
        edge.mutation.late_sibling()
    assert edge.journal()["resources"][0]["uid"] == "created-uid"
    assert edge.journal()["resources"][0]["approved"] is False
    assert not any(args[0] == "wait" for args, _ in edge.events), edge.events


@pytest.mark.parametrize(
    "defect",
    ["template", "pod-spec", "metadata", "labels", "selector", "generated-label"],
)
def test_unrecognized_job_template_or_controller_defaults_are_rejected(
    edge: BoundaryApi, defect: str
) -> None:
    def changed(value: dict[str, Any]) -> None:
        spec = value["spec"]
        if defect == "template":
            spec["template"] = None
        elif defect == "pod-spec":
            spec["template"]["spec"] = None
        elif defect == "metadata":
            spec["template"]["metadata"] = []
        elif defect == "labels":
            spec["template"]["metadata"] = {"labels": []}
        elif defect == "selector":
            spec["selector"] = {
                "matchLabels": {"batch.kubernetes.io/controller-uid": "foreign"}
            }
        else:
            spec["template"]["metadata"] = {"labels": {"controller-uid": "foreign"}}

    admission(edge, changed)
    with pytest.raises(BoundaryDenied):
        edge.mutation.change_owner()
    assert edge.journal()["resources"][0]["approved"] is False
    assert not any(args[0] == "patch" for args in edge.mutation_calls()), edge.events


@pytest.mark.parametrize("phase", ["before-owner-patch", "cleanup"])
def test_full_source_spec_drift_cannot_be_hidden_by_same_uid_and_owners(
    edge: BoundaryApi, phase: str
) -> None:
    spec = {"pytorchReplicaSpecs": {"Worker": {"replicas": 1}}}
    edge.api.source["spec"] = deepcopy(spec)
    edge.mutation.source = deepcopy(edge.api.source)
    edge.source()["spec"] = deepcopy(spec)
    if phase == "cleanup":
        edge.mutation.change_owner()
        edge.source()["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = 2
        edge.source()["metadata"]["resourceVersion"] = "91"
        before = list(edge.mutation_calls())
        with pytest.raises(BoundaryDenied, match="declared state"):
            edge.mutation.cleanup()
        assert edge.mutation_calls() == before
    else:

        def drift(args: tuple[str, ...], raw: str) -> str:
            if args[0] == "create":
                edge.source()["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = 2
                edge.source()["metadata"]["resourceVersion"] = "91"
            return raw

        edge.after = drift
        with pytest.raises(BoundaryDenied, match="declared state"):
            edge.mutation.change_owner()
        assert not any(args[0] == "patch" for args in edge.mutation_calls()), (
            edge.events
        )


def test_source_owner_change_during_anchor_creation_is_rechecked(
    edge: BoundaryApi,
) -> None:
    foreign = [{"uid": "foreign", "kind": "Job", "name": "foreign"}]

    def drift(args: tuple[str, ...], raw: str) -> str:
        if args[0] == "create":
            edge.source()["metadata"].update(
                ownerReferences=foreign, resourceVersion="91"
            )
        return raw

    edge.after = drift
    with pytest.raises(BoundaryDenied, match="source owner drifted"):
        edge.mutation.change_owner()
    assert edge.source()["metadata"]["ownerReferences"] == foreign
    assert not any(args[0] == "patch" for args in edge.mutation_calls()), edge.events


@pytest.mark.parametrize("field", ["expected", "ack_sha256", "approved"])
def test_missing_or_corrupt_custody_is_not_repaired_by_a_get(
    edge: BoundaryApi, field: str
) -> None:
    edge.mutation.late_sibling()
    entry = edge.mutation.resources[0]
    if field == "expected":
        entry.pop(field)
    elif field == "ack_sha256":
        entry[field] = None
    else:
        entry[field] = "true"
    before = list(edge.mutation_calls())
    with pytest.raises(BoundaryDenied, match="custody record"):
        edge.mutation.cleanup()
    assert edge.mutation_calls() == before


def test_absence_does_not_upgrade_a_legacy_uid_only_record(edge: BoundaryApi) -> None:
    edge.mutation.resources.append(
        {"kind": "pod", "name": "old-holder", "uid": "unproved-uid", "owners": []}
    )
    before = list(edge.events)
    with pytest.raises(BoundaryDenied, match="custody record"):
        edge.mutation.cleanup()
    assert edge.events == before


@pytest.mark.parametrize("field", ["spec", "namespace", "name"])
def test_custody_intent_cannot_be_rebound_before_cleanup(
    edge: BoundaryApi, field: str
) -> None:
    edge.mutation.late_sibling()
    entry = edge.mutation.resources[0]
    if field == "spec":
        entry["expected"]["spec"]["nodeName"] = "unapproved-node"
    else:
        entry["expected"]["metadata"][field] = "foreign"
    before = list(edge.events)
    with pytest.raises(BoundaryDenied, match="custody record"):
        edge.mutation.cleanup()
    assert edge.events == before
