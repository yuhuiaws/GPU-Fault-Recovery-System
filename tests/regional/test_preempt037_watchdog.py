"""CPU-only watchdog ownership and dispatcher-window cleanup with fake APIs."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes import client

from scripts.e2e.regional import preempt037_watchdog as watchdog
from scripts.e2e.regional import run_preempt037_dispatcher_liveness as runner
from scripts.e2e.regional.probes import preempt037_dispatcher_watchdog as probe

IMAGE = "img@sha256:" + "a" * 64


def deployment(value: str | None, *, uid: str = "uid-1") -> Any:
    environment = [SimpleNamespace(name="OTHER", value="unchanged", value_from=None)]
    if value is not None:
        environment.append(
            SimpleNamespace(name=probe.VARIABLE, value=value, value_from=None)
        )
    return SimpleNamespace(
        metadata=SimpleNamespace(uid=uid, resource_version="version-1"),
        spec=SimpleNamespace(
            template=SimpleNamespace(
                spec=SimpleNamespace(
                    containers=[
                        SimpleNamespace(
                            name=probe.CONTAINER, env=environment, image=IMAGE
                        )
                    ]
                )
            )
        ),
    )


@pytest.mark.parametrize(
    "baseline", [{"present": True, "value": "true"}, {"present": False, "value": None}]
)
def test_watchdog_patch_binds_uid_version_and_only_its_variable(baseline: dict) -> None:
    baseline = {**baseline, "image": IMAGE}
    patches = probe.deployment_patch(
        deployment("false"), uid="uid-1", baseline=baseline
    )
    assert patches[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-1"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "version-1"},
    ]
    assert patches[2]["value"] == probe.VARIABLE
    assert patches[3]["value"] == "false"
    assert patches[-1]["op"] == ("replace" if baseline["present"] else "remove")
    assert patches[-1]["path"].startswith("/spec/template/spec/containers/0/env/1"), (
        f"watchdog patch escaped its owned environment variable: {patches[-1]}"
    )


def test_watchdog_refuses_replaced_deployment_or_unowned_value() -> None:
    baseline = {"present": True, "value": "true", "image": IMAGE}
    with pytest.raises(RuntimeError, match="identity"):
        probe.deployment_patch(
            deployment("false", uid="different"), uid="uid-1", baseline=baseline
        )
    with pytest.raises(RuntimeError, match="no longer owned"):
        probe.deployment_patch(
            deployment("operator-hold"), uid="uid-1", baseline=baseline
        )
    assert (
        probe.deployment_patch(deployment("true"), uid="uid-1", baseline=baseline) == []
    )
    changed_image = deployment("false")
    changed_image.spec.template.spec.containers[0].image = (
        "different@sha256:" + "b" * 64
    )
    with pytest.raises(RuntimeError, match="image changed"):
        probe.deployment_patch(changed_image, uid="uid-1", baseline=baseline)


def test_watchdog_rechecks_restoration_instead_of_trusting_patch_ack() -> None:
    snapshots = iter([deployment("false"), deployment("true")])
    patches = []
    api = SimpleNamespace(
        read_namespaced_deployment=lambda *a, **k: next(snapshots),
        patch_namespaced_deployment=lambda *a, **k: patches.append(a[2]),
    )
    probe.restore(
        api, "cpu-ns", "uid-1", {"present": True, "value": "true", "image": IMAGE}
    )
    assert len(patches) == 1


def test_kubernetes_sdk_sends_json_patch_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_client = client.ApiClient(
        configuration=client.Configuration(host="https://api.invalid")
    )
    sent = []

    def request(method: str, url: str, **kwargs: Any) -> None:
        sent.append((method, kwargs["headers"]["Content-Type"], kwargs["body"]))
        raise RuntimeError("fake transport stopped before network")

    monkeypatch.setattr(api_client.rest_client, "request", request)
    patch = probe.deployment_patch(
        deployment("false"),
        uid="uid-1",
        baseline={"present": False, "value": None, "image": IMAGE},
    )
    with pytest.raises(RuntimeError, match="fake transport"):
        client.AppsV1Api(api_client).patch_namespaced_deployment(
            "deployment", "ns", patch
        )
    assert sent == [("PATCH", "application/json-patch+json", patch)]
    api_client.close()


def test_watchdog_permissions_are_namespaced_and_target_one_deployment() -> None:
    fixture = watchdog.DispatcherWatchdog(
        SimpleNamespace(settings=SimpleNamespace(namespace="cpu-ns")),
        baseline={
            "uid": "uid-1",
            "image": "img@sha256:" + "a" * 64,
            "present": False,
            "value": None,
        },
        restore_at=10000,
    )
    manifests = fixture.manifests()
    role = next(item for item in manifests if item["kind"] == "Role")
    assert role["rules"] == [
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "resourceNames": ["gpu-fault-control-worker"],
            "verbs": ["get", "patch"],
        }
    ]
    assert all(item["metadata"]["namespace"] == "cpu-ns" for item in manifests), (
        "watchdog resources must remain in the bound CPU namespace"
    )
    job = manifests[-1]
    assert job["spec"]["template"]["spec"]["serviceAccountName"] == fixture.name
    assert job["spec"]["backoffLimit"] == 0
    assert "volumes" not in job["spec"]["template"]["spec"]
    assert job["spec"]["template"]["spec"]["containers"][0]["command"][0] == (
        "/opt/gpu-fault/control-plane/bin/python"
    )


@pytest.mark.parametrize("foreign", ["value", "uid", "image", None])
def test_dispatcher_restoration_patches_only_its_owned_window(
    foreign: str | None,
) -> None:
    baseline = {"uid": "uid-1", "image": IMAGE, "present": False, "value": None}
    patches: list[list[dict[str, Any]]] = []
    document = {
        "metadata": {
            "uid": "different" if foreign == "uid" else "uid-1",
            "resourceVersion": "version-1",
            "generation": 2,
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": probe.CONTAINER,
                            "image": "different" if foreign == "image" else IMAGE,
                            "env": [
                                {"name": "OTHER", "value": "preserve"},
                                {
                                    "name": probe.VARIABLE,
                                    "value": "operator-hold"
                                    if foreign == "value"
                                    else "false",
                                },
                            ],
                        }
                    ]
                }
            }
        },
    }

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        if args[0] == "get":
            return json.dumps(document)
        if args[0] == "patch":
            patches.append(json.loads(kwargs["input_text"]))
            return "patched"
        assert args[:2] == ("rollout", "status")
        return "restored"

    regional = SimpleNamespace(kubectl=kubectl)
    if foreign:
        with pytest.raises(runner.RegionalFixtureError, match="identity|owned"):
            runner.set_variable(regional, probe.VARIABLE + "-", baseline=baseline)
        assert not patches, f"foreign {foreign} state was patched: {patches}"
    else:
        assert (
            runner.set_variable(regional, probe.VARIABLE + "-", baseline=baseline)
            == "restored"
        )
        assert patches == [
            [
                {"op": "test", "path": "/metadata/uid", "value": "uid-1"},
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": "version-1",
                },
                {"op": "remove", "path": "/spec/template/spec/containers/0/env/1"},
            ]
        ]


def test_one_fresh_replica_cannot_mask_another_stalled_periodic_loop() -> None:
    verdicts = runner.verdicts
    assert not verdicts.periodic_alive(
        [f"{verdicts.PERIODIC_METRIC} 1000", f"{verdicts.PERIODIC_METRIC} 500"],
        now=1000,
        threshold_seconds=300,
    ), "a fresh replica must not conceal another replica's stalled periodic loop"


@pytest.mark.parametrize("failure", [None, "missing-ack", "create-ack-loss"])
def test_watchdog_admission_and_resource_cleanup_are_run_owned(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    resources: dict[str, dict[str, Any]] = {}
    deleted: list[str] = []
    baseline = {"uid": "uid-1", "image": IMAGE, "present": False, "value": None}

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        assert kwargs.get("check", True), "watchdog API failures must not be ignored"
        if args[0] == "create":
            item = json.loads(kwargs["input_text"])
            kind = item["kind"].lower()
            item["metadata"].update(uid=f"uid-{kind}", resourceVersion="version-1")
            resources[kind] = item
            if kind == "job" and failure == "create-ack-loss":
                raise RuntimeError("watchdog creation acknowledgement lost")
            return "created"
        if args[0] == "wait":
            return "Ready"
        if args[0] == "logs":
            return json.dumps(
                {
                    "state": "ARMED",
                    "restore_at": 1200.0,
                    "deployment_uid": "foreign"
                    if failure == "missing-ack"
                    else "uid-1",
                    "image": IMAGE,
                }
            )
        if args[0] == "get":
            return (
                json.dumps(resources[args[1]]["metadata"])
                if args[1] in resources
                else ""
            )
        if args[0] == "delete":
            assert args[1] == "--raw"
            options = json.loads(kwargs["input_text"])
            uid = options["preconditions"]["uid"]
            kind = next(
                kind
                for kind, item in resources.items()
                if item["metadata"]["uid"] == uid
            )
            assert options["preconditions"]["resourceVersion"] == "version-1"
            deleted.append(kind)
            del resources[kind]
            return "deleted"
        raise AssertionError(args)

    monkeypatch.setattr(watchdog.time, "time", lambda: 1000.0)
    fixture = watchdog.DispatcherWatchdog(
        SimpleNamespace(settings=SimpleNamespace(namespace="cpu-ns"), kubectl=kubectl),
        baseline=baseline,
        restore_at=1200.0,
    )
    if failure:
        with pytest.raises(RuntimeError, match="acknowledge"):
            fixture.arm()
    else:
        fixture.arm()
    assert set(resources) == {"job", "role", "rolebinding", "serviceaccount"}
    assert all(
        item["metadata"]["labels"]["gpu-fault.io/acceptance-run"] == fixture.run_id
        for item in resources.values()
    ), "every watchdog resource must carry this attempt's run identity"
    fixture.cleanup()
    assert deleted == ["job", "rolebinding", "role", "serviceaccount"]
    assert resources == {}


@pytest.mark.parametrize("failure", [None, "transient", "patch-ack-loss", "forbidden"])
def test_watchdog_process_arms_then_restores_with_bounded_api_retries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str | None,
) -> None:
    clock = [1000.0]
    value = ["true"]
    reads = [0]
    patches: list[Any] = []
    baseline = {"present": True, "value": "true", "image": IMAGE}

    def read(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["_request_timeout"] == (5, 10)
        reads[0] += 1
        if reads[0] == 2:
            if failure == "transient":
                raise client.exceptions.ApiException(status=503)
            if failure == "forbidden":
                raise client.exceptions.ApiException(status=403)
        return deployment(value[0])

    def patch(*args: Any, **kwargs: Any) -> None:
        patches.append(args[2])
        value[0] = "true"
        if failure == "patch-ack-loss":
            raise TimeoutError("patch acknowledgement lost")

    def sleep(seconds: float) -> None:
        if clock[0] < 1010:
            value[0] = "false"
        clock[0] += seconds

    monkeypatch.setenv("PREEMPT037_RESTORE_AT", "1010")
    monkeypatch.setenv("PREEMPT037_NAMESPACE", "cpu-ns")
    monkeypatch.setenv("PREEMPT037_DEPLOYMENT_UID", "uid-1")
    monkeypatch.setenv("PREEMPT037_BASELINE", json.dumps(baseline))
    monkeypatch.setattr(probe.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(
        probe.client,
        "AppsV1Api",
        lambda: SimpleNamespace(
            read_namespaced_deployment=read, patch_namespaced_deployment=patch
        ),
    )
    monkeypatch.setattr(probe.time, "time", lambda: clock[0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe.time, "sleep", sleep)
    if failure == "forbidden":
        with pytest.raises(RuntimeError, match="could not confirm"):
            probe.main()
        assert patches == []
    else:
        probe.main()
        assert len(patches) == 1
        assert value[0] == "true"
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[0] == {
        "state": "ARMED",
        "restore_at": 1010.0,
        "deployment_uid": "uid-1",
        "image": IMAGE,
    }
    assert [message["state"] for message in messages] == (
        ["ARMED"] if failure == "forbidden" else ["ARMED", "RESTORED"]
    )


@pytest.mark.parametrize("restore_fails", [False, True])
def test_window_disarms_watchdog_only_after_proven_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_fails: bool
) -> None:
    clock = [10000.0]
    value = [None]
    calls: list[str] = []
    baseline = {
        "present": False,
        "value": None,
        "uid": "uid-1",
        "image": "img@sha256:" + "a" * 64,
        "generation": 1,
    }

    class Watchdog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.name = "watchdog"
            self.run_id = "run-id"
            self.restore_at = kwargs["restore_at"]

        def arm(self) -> None:
            calls.append("arm")

        def cleanup(self) -> None:
            calls.append("disarm")

    def set_variable(
        regional: Any, assignment: str, *, baseline: dict[str, Any]
    ) -> str:
        assert baseline["uid"] == "uid-1"
        if assignment.endswith("=false"):
            calls.append("disable")
            value[0] = "false"
        else:
            calls.append("restore")
            if restore_fails:
                raise RuntimeError("restore failed")
            value[0] = None
        return "ready"

    def metrics(regional: Any) -> list[str]:
        stamp = 0 if value[0] == "false" else clock[0]
        return [
            f"{runner.verdicts.DISPATCH_METRIC} {stamp}\n"
            f"{runner.verdicts.PERIODIC_METRIC} {clock[0]}"
        ]

    monkeypatch.setattr(runner, "DispatcherWatchdog", Watchdog)
    monkeypatch.setattr(runner, "set_variable", set_variable)
    monkeypatch.setattr(
        runner,
        "deployment_variable",
        lambda r: {
            **copy.deepcopy(baseline),
            "present": value[0] is not None,
            "value": value[0],
        },
    )
    monkeypatch.setattr(runner, "worker_metrics", metrics)
    monkeypatch.setattr(
        runner,
        "replicas",
        lambda r: [{"pod": "worker", "values": {probe.VARIABLE: value[0] or "true"}}],
    )
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        runner,
        "utc_now",
        lambda: datetime.fromtimestamp(clock[0], timezone.utc).isoformat(),
    )
    regional = SimpleNamespace(
        cpu_python=lambda source: {"status_counts": {"SUCCEEDED": 600}}
    )
    deadline = datetime.fromtimestamp(clock[0] + 5000, timezone.utc)

    if restore_fails:
        with pytest.raises(RuntimeError, match="restore failed"):
            runner.execute(regional, tmp_path, deadline)
        assert calls == ["arm", "disable", "restore"]
    else:
        result = runner.execute(regional, tmp_path, deadline)
        assert result["verdict"] == "PASS"
        assert calls == ["arm", "disable", "restore", "disarm"]
    evidence = json.loads((tmp_path / "env-window-baseline.json").read_text())
    assert evidence["watchdog_retained"] is restore_fails
