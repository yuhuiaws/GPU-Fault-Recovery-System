from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.collector_device_plugin_fixture import (
    DevicePluginFixture,
    excluding_node_affinity,
)
from scripts.e2e.regional.probes import collector_plugin_watchdog as watchdog_probe
from scripts.e2e.regional.probes.collector_plugin_watchdog import (
    WINDOW_KEY,
    restoration_patch,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


class PluginApi:
    def __init__(self, pod_spec: dict[str, Any]) -> None:
        self.document = {
            "metadata": {
                "namespace": "kube-system",
                "name": "nvidia-device-plugin",
                "uid": "plugin-uid",
                "resourceVersion": "1",
            },
            "spec": {
                "template": {"spec": deepcopy(pod_spec)},
                "updateStrategy": {"type": "OnDelete"},
            },
            "status": {"desiredNumberScheduled": 3},
        }
        self.patches: list[list[dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.pods: list[dict[str, Any]] = []
        self.race = False
        self.resources: dict[str, dict[str, Any]] = {}

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-a", "cluster_id": "cluster-a"}

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "gpu"
        if args[:2] == ("get", "namespace"):
            return json.dumps({"metadata": {"uid": "namespace-a"}})
        if args[:2] == ("get", "deployment"):
            assert args[2] == "gpu-fault-cluster-executor"
            return json.dumps(
                {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "executor",
                                        "image": "executor@sha256:" + "b" * 64,
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        if args[:2] == ("get", "daemonset"):
            if kwargs.get("all_namespaces"):
                return json.dumps({"items": [self.document]})
            assert kwargs["namespace"] == "kube-system"
            return json.dumps(self.document)
        assert kwargs["namespace"] == "kube-system"
        if args[:2] == ("patch", "daemonset"):
            assert "--type=json" in args
            operations = json.loads(args[args.index("-p") + 1])
            self.patches.append(operations)
            if self.race:
                self.document["metadata"]["resourceVersion"] = "foreign-version"
            candidate = deepcopy(self.document)
            for operation in operations:
                keys = [
                    part.replace("~1", "/").replace("~0", "~")
                    for part in operation["path"].split("/")[1:]
                ]
                parent = candidate
                for key in keys[:-1]:
                    parent = parent[key]
                key = keys[-1]
                if operation["op"] == "remove":
                    del parent[key]
                elif operation["op"] == "test":
                    if parent.get(key) != operation["value"]:
                        raise RegionalFixtureError("JSON patch precondition refused")
                else:
                    assert operation["op"] in {"add", "replace"}
                    parent[key] = deepcopy(operation["value"])
            self.document = candidate
            self.document["metadata"]["resourceVersion"] = str(len(self.patches) + 1)
            return ""
        if args[:2] == ("get", "pod"):
            return json.dumps({"items": self.pods})
        if args[:2] == ("delete", "--raw"):
            name = args[2].rsplit("/", 1)[-1]
            if "/pods/" in args[2]:
                body = json.loads(kwargs["input_text"])
                pod = next(row for row in self.pods if row["metadata"]["name"] == name)
                assert body["preconditions"] == {
                    "uid": pod["metadata"]["uid"],
                    "resourceVersion": pod["metadata"]["resourceVersion"],
                }
                self.deleted.append(name)
            else:
                resource = next(
                    key
                    for key in self.resources
                    if args[2].endswith("/" + self.resources[key]["metadata"]["name"])
                    and ("/" + key + "s/") in args[2]
                )
                del self.resources[resource]
            return ""
        if args[0] == "create":
            resource = json.loads(kwargs["input_text"])
            kind = resource["kind"].lower()
            resource["metadata"]["uid"] = kind + "-uid"
            resource["metadata"]["resourceVersion"] = "1"
            self.resources[kind] = resource
            return json.dumps(resource)
        if args[0] == "get" and args[1] in {
            "serviceaccount",
            "role",
            "rolebinding",
            "job",
        }:
            resource = self.resources.get(args[1])
            if resource is None:
                return ""
            return json.dumps(
                resource["metadata"] if "jsonpath={.metadata}" in args else resource
            )
        if args[0] == "wait":
            return ""
        if args[0] == "logs":
            container = self.resources["job"]["spec"]["template"]["spec"]["containers"][
                0
            ]
            record = json.loads(container["env"][0]["value"])
            return json.dumps(
                {
                    "state": "ARMED",
                    "window_id": record["window_id"],
                    "daemonset_uid": record["daemonset_uid"],
                    "restore_at": record["restore_at"],
                }
            )
        raise AssertionError(f"unexpected mock Kubernetes call: {args}")


def fixture(api: PluginApi, case_dir: Path) -> DevicePluginFixture:
    return DevicePluginFixture(
        api,  # type: ignore[arg-type]
        token="nvidia-device-plugin",
        node="selected-node",
        resource="nvidia.com/gpu",
        case_dir=case_dir,
    )


@pytest.mark.parametrize(
    "pod_spec",
    [
        {},
        {"affinity": None},
        {"affinity": {}},
        {
            "affinity": {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": []
                }
            }
        },
        {
            "affinity": {
                "nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": []}
            }
        },
        {
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {"key": "gpu", "operator": "In", "values": ["h100"]}
                                ]
                            },
                            {
                                "matchFields": [
                                    {
                                        "key": "metadata.name",
                                        "operator": "In",
                                        "values": ["selected-node", "other-node"],
                                    }
                                ]
                            },
                        ]
                    }
                }
            }
        },
    ],
)
def test_affinity_roundtrip_preserves_absence_null_and_every_original_constraint(
    pod_spec: dict[str, Any], tmp_path: Path
) -> None:
    api = PluginApi(pod_spec)
    plugin = fixture(api, tmp_path)
    plugin.discover()
    original = deepcopy(api.document["spec"])
    plugin.exclude_node()
    changed = api.document["spec"]["template"]["spec"]["affinity"]
    terms = changed["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ]
    assert all(
        {"key": "metadata.name", "operator": "NotIn", "values": ["selected-node"]}
        in item["matchFields"]
        for item in terms
    ), f"selected-node exclusion is missing from an OR-affinity clause: {terms!r}"
    assert plugin.affinity == pod_spec.get("affinity")
    plugin.restore()
    assert api.document["spec"] == original
    assert len(api.patches) == 4
    assert api.resources == {}
    assert WINDOW_KEY not in api.document["metadata"]["annotations"]
    plugin.restore()
    assert len(api.patches) == 4


@pytest.mark.parametrize("terms", [[], [{}], [{"matchExpressions": []}]])
def test_exclusion_never_widens_an_empty_or_term(terms: list[dict[str, Any]]) -> None:
    with pytest.raises(RegionalFixtureError):
        excluding_node_affinity(
            {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": terms
                    }
                }
            },
            "selected-node",
        )


def test_rolling_update_refuses_before_any_plugin_mutation(tmp_path: Path) -> None:
    api = PluginApi({})
    api.document["spec"]["updateStrategy"]["type"] = "RollingUpdate"
    with pytest.raises(RegionalFixtureError, match="cluster-wide"):
        fixture(api, tmp_path).exclude_node()
    assert api.patches == [] and api.deleted == []


def test_restore_does_not_overwrite_foreign_affinity(tmp_path: Path) -> None:
    api = PluginApi({})
    plugin = fixture(api, tmp_path)
    plugin.exclude_node()
    foreign = {"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": []}}
    api.document["spec"]["template"]["spec"]["affinity"] = foreign
    with pytest.raises(RuntimeError, match="foreign affinity"):
        plugin.restore()
    assert len(api.patches) == 2
    assert api.document["spec"]["template"]["spec"]["affinity"] == foreign
    assert "job" in api.resources


@pytest.mark.parametrize("field", ["uid", "resourceVersion"])
def test_missing_or_replaced_daemonset_identity_refuses_mutation(
    field: str, tmp_path: Path
) -> None:
    api = PluginApi({})
    plugin = fixture(api, tmp_path)
    plugin.discover()
    api.document["metadata"][field] = ""
    with pytest.raises(RegionalFixtureError):
        plugin.exclude_node()
    assert api.patches == []


def test_resource_version_race_is_rejected_atomically(tmp_path: Path) -> None:
    api = PluginApi({})
    api.race = True
    with pytest.raises(RegionalFixtureError, match="precondition"):
        fixture(api, tmp_path).exclude_node()
    assert "affinity" not in api.document["spec"]["template"]["spec"]
    assert api.deleted == []


def test_only_target_node_pods_owned_by_the_same_daemonset_uid_are_deleted(
    tmp_path: Path,
) -> None:
    api = PluginApi({})
    for name, node, owner_uid in (
        ("target", "selected-node", "plugin-uid"),
        ("other-node", "other-node", "plugin-uid"),
        ("replaced-owner", "selected-node", "old-plugin-uid"),
    ):
        api.pods.append(
            {
                "metadata": {
                    "name": name,
                    "uid": name + "-uid",
                    "resourceVersion": "1",
                    "ownerReferences": [
                        {
                            "kind": "DaemonSet",
                            "name": "nvidia-device-plugin",
                            "uid": owner_uid,
                        }
                    ],
                },
                "spec": {"nodeName": node},
            }
        )
    fixture(api, tmp_path).exclude_node()
    assert api.deleted == ["target"]


def test_detached_watchdog_restores_exact_affinity_and_closes_delayed_writes(
    tmp_path: Path,
) -> None:
    api = PluginApi({})
    plugin = fixture(api, tmp_path)
    plugin.exclude_node()
    assert plugin.watchdog is not None
    patch = restoration_patch(deepcopy(api.document), plugin.watchdog.record)
    api.kubectl(
        "gpu",
        "patch",
        "daemonset",
        plugin.name,
        "--type=json",
        "-p",
        json.dumps(patch),
        namespace="kube-system",
    )
    assert "affinity" not in api.document["spec"]["template"]["spec"]
    with pytest.raises(RegionalFixtureError, match="closed"):
        plugin.watchdog.require_armed()
    plugin.restore()
    assert api.resources == {}
    assert WINDOW_KEY not in api.document["metadata"]["annotations"]


def test_failed_watchdog_ack_never_excludes_or_deletes_a_pod(tmp_path: Path) -> None:
    class NoAck(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[0] == "logs":
                return json.dumps({"state": "ARMED", "window_id": "another-run"})
            return super().kubectl(plane, *args, **kwargs)

    api = NoAck({})
    plugin = fixture(api, tmp_path)
    with pytest.raises(RegionalFixtureError, match="acknowledge"):
        plugin.exclude_node()
    assert "affinity" not in api.document["spec"]["template"]["spec"]
    assert api.deleted == []
    plugin.restore()
    assert api.resources == {}


def test_ondelete_drift_retains_watchdog_without_overwriting_strategy(
    tmp_path: Path,
) -> None:
    api = PluginApi({})
    plugin = fixture(api, tmp_path)
    plugin.exclude_node()
    api.document["spec"]["updateStrategy"]["type"] = "RollingUpdate"
    with pytest.raises(RuntimeError, match="OnDelete"):
        plugin.restore()
    assert "job" in api.resources
    assert api.document["spec"]["updateStrategy"]["type"] == "RollingUpdate"


def test_detached_worker_entrypoint_restores_after_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from kubernetes import client, config

    api = PluginApi({})
    plugin = fixture(api, tmp_path)
    plugin.exclude_node()
    assert plugin.watchdog is not None
    record = plugin.watchdog.record
    clock = [record["restore_at"] - 3]
    monkeypatch.setattr(
        watchdog_probe,
        "time",
        SimpleNamespace(
            time=lambda: clock[0],
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )

    class Apps:
        api_client = SimpleNamespace(sanitize_for_serialization=lambda value: value)

        def read_namespaced_daemon_set(
            self, name: str, namespace: str, **kwargs: Any
        ) -> dict[str, Any]:
            assert (name, namespace) == (plugin.name, plugin.namespace)
            return deepcopy(api.document)

        def patch_namespaced_daemon_set(
            self, name: str, namespace: str, patch: Any, **kwargs: Any
        ) -> None:
            api.kubectl(
                "gpu",
                "patch",
                "daemonset",
                name,
                "--type=json",
                "-p",
                json.dumps(patch),
                namespace=namespace,
            )

    monkeypatch.setattr(config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(client, "AppsV1Api", Apps)
    monkeypatch.setenv("COLLECTOR_PLUGIN_WINDOW", json.dumps(record))
    watchdog_probe.main()
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["state"] for row in receipts] == ["ARMED", "RESTORED"]
    assert "affinity" not in api.document["spec"]["template"]["spec"]
    plugin.restore()
    assert not api.resources, (
        f"restored plugin watchdog left owned resources: {list(api.resources)!r}"
    )


def test_resource_replacement_during_cleanup_cannot_be_deleted(tmp_path: Path) -> None:
    class ReplacementApi(PluginApi):
        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            if args[:2] == ("get", "job") and "jsonpath={.metadata}" in args:
                self.resources["job"]["metadata"]["uid"] = "foreign-job-uid"
            return super().kubectl(plane, *args, **kwargs)

    api = ReplacementApi({})
    plugin = fixture(api, tmp_path)
    plugin.exclude_node()
    with pytest.raises(RegionalFixtureError, match="replaced during cleanup"):
        plugin.restore()
    assert api.resources["job"]["metadata"]["uid"] == "foreign-job-uid"


def test_watchdog_uses_installed_executor_image_and_component_interpreter(
    tmp_path: Path,
) -> None:
    api = PluginApi({})
    plugin = fixture(api, tmp_path)
    plugin.exclude_node()
    container = api.resources["job"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "executor@sha256:" + "b" * 64
    assert container["command"][0] == "/opt/gpu-fault/executor/bin/python"
    plugin.restore()
