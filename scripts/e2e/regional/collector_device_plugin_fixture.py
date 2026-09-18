from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from scripts.e2e.regional.collector_action_guard import (
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.collector_plugin_watchdog import PluginWatchdog
from scripts.e2e.regional.probes.collector_plugin_watchdog import (
    WINDOW_KEY,
    WINDOW_PATH,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture


def excluding_node_affinity(
    original: dict[str, Any] | None, node: str
) -> dict[str, Any]:
    affinity = deepcopy(original) if original is not None else {}
    node_affinity = affinity.setdefault("nodeAffinity", {})
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution")
    if required is None:
        terms: list[dict[str, Any]] = [{}]
        node_affinity["requiredDuringSchedulingIgnoredDuringExecution"] = {
            "nodeSelectorTerms": terms
        }
    else:
        terms = required.get("nodeSelectorTerms")
        if not isinstance(terms, list) or not terms:
            raise RegionalFixtureError("existing node affinity has no selector terms")
        if any(
            not isinstance(term, dict)
            or not (term.get("matchExpressions") or term.get("matchFields"))
            for term in terms
        ):
            raise RegionalFixtureError("refusing to widen an empty node affinity term")
    for term in terms:
        fields = term.setdefault("matchFields", [])
        if not isinstance(fields, list):
            raise RegionalFixtureError("node affinity matchFields is not a list")
        fields.append({"key": "metadata.name", "operator": "NotIn", "values": [node]})
    return affinity


class DevicePluginFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        token: str,
        node: str,
        resource: str,
        case_dir: Path,
    ) -> None:
        self.regional = regional
        self.token = token
        self.node = node
        self.resource = resource
        self.case_dir = case_dir
        self.watchdog: PluginWatchdog | None = None
        self.namespace = ""
        self.name = ""
        self.uid = ""
        self.affinity: dict[str, Any] | None = None
        self.affinity_present = False
        self.excluded_affinity: dict[str, Any] | None = None
        self.update_strategy = ""

    def discover(self) -> dict[str, Any]:
        value = json.loads(
            self.regional.kubectl(
                "gpu", "get", "daemonset", "-o", "json", all_namespaces=True
            )
        )
        matches = [
            item
            for item in value.get("items", [])
            if self.token.lower() in json.dumps(item, sort_keys=True).lower()
            and int((item.get("status") or {}).get("desiredNumberScheduled") or 0) > 0
        ]
        if len(matches) != 1:
            raise RegionalFixtureError(
                f"expected one {self.token} DaemonSet, found {len(matches)}"
            )
        item = matches[0]
        self.namespace = str(item["metadata"]["namespace"])
        self.name = str(item["metadata"]["name"])
        self.uid = str(item["metadata"].get("uid") or "")
        spec = item["spec"]["template"]["spec"]
        self.affinity = deepcopy(spec.get("affinity"))
        self.affinity_present = "affinity" in spec
        self.update_strategy = str(
            (item["spec"].get("updateStrategy") or {}).get("type") or "RollingUpdate"
        )
        return {
            "namespace": self.namespace,
            "name": self.name,
            "affinity": self.affinity,
            "update_strategy": self.update_strategy,
            "supported_exclusion_policy": "OnDelete",
        }

    def _read(self) -> dict[str, Any]:
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "daemonset",
                self.name,
                "-o",
                "json",
                namespace=self.namespace,
            )
        )
        if not isinstance(value, dict):
            raise RegionalFixtureError("plugin DaemonSet read is not an object")
        return value

    def _patch_affinity(
        self,
        affinity: dict[str, Any] | None,
        *,
        present: bool,
        expected: dict[str, Any] | None,
        expected_present: bool,
    ) -> None:
        current = self._read()
        metadata = current.get("metadata") or {}
        if not self.uid or metadata.get("uid") != self.uid:
            raise RegionalFixtureError("plugin DaemonSet UID changed")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(resource_version, str) or not resource_version:
            raise RegionalFixtureError("plugin DaemonSet resourceVersion is unknown")
        spec = current["spec"]["template"]["spec"]
        current_present = "affinity" in spec
        if current_present == present and spec.get("affinity") == affinity:
            return
        if current_present != expected_present or spec.get("affinity") != expected:
            raise RegionalFixtureError(
                "plugin DaemonSet affinity changed outside this case"
            )
        operations: list[dict[str, Any]] = [
            {"op": "test", "path": "/metadata/uid", "value": self.uid},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": resource_version,
            },
        ]
        if self.excluded_affinity is not None and affinity is self.excluded_affinity:
            if self.watchdog is None:
                raise RegionalFixtureError("plugin exclusion has no durable watchdog")
            self.watchdog.require_armed()
            if (current["spec"].get("updateStrategy") or {}).get("type") != "OnDelete":
                raise RegionalFixtureError(
                    "plugin update strategy changed before exclusion"
                )
            operations.append(
                {"op": "test", "path": "/spec/updateStrategy/type", "value": "OnDelete"}
            )
            operations.append(
                {
                    "op": "test",
                    "path": WINDOW_PATH,
                    "value": self.watchdog.record["window_id"],
                }
            )
        path = "/spec/template/spec/affinity"
        operations.append(
            {
                "op": "replace" if current_present else "add",
                "path": path,
                "value": affinity,
            }
            if present
            else {"op": "remove", "path": path}
        )
        self.regional.kubectl(
            "gpu",
            "patch",
            "daemonset",
            self.name,
            "--type=json",
            "-p",
            json.dumps(operations, sort_keys=True),
            namespace=self.namespace,
        )
        after = self._read()
        after_spec = after["spec"]["template"]["spec"]
        if (
            after["metadata"].get("uid") != self.uid
            or ("affinity" in after_spec) != present
            or after_spec.get("affinity") != affinity
        ):
            raise RegionalFixtureError("plugin affinity mutation failed readback")

    def exclude_node(self) -> None:
        if not self.name:
            self.discover()
        if self.update_strategy != "OnDelete":
            raise RegionalFixtureError(
                "plugin exclusion requires the explicitly supported OnDelete deployment "
                "policy; refusing a cluster-wide rollout or strategy change"
            )
        self.excluded_affinity = excluding_node_affinity(self.affinity, self.node)
        current = self._read()
        if (
            current["metadata"].get("uid") != self.uid
            or ("affinity" in current["spec"]["template"]["spec"])
            != self.affinity_present
            or current["spec"]["template"]["spec"].get("affinity") != self.affinity
        ):
            raise RegionalFixtureError("plugin baseline changed before watchdog arming")
        executor = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "deployment",
                "gpu-fault-cluster-executor",
                "-o",
                "json",
            )
        )
        containers = [
            container
            for container in executor["spec"]["template"]["spec"]["containers"]
            if container.get("name") == "executor"
        ]
        if len(containers) != 1 or "@sha256:" not in str(
            containers[0].get("image") or ""
        ):
            raise RegionalFixtureError(
                "plugin watchdog needs the installed immutable Executor image"
            )
        self.watchdog = PluginWatchdog(
            self.regional,
            baseline=current,
            excluded_affinity=self.excluded_affinity,
            image=str(containers[0]["image"]),
            case_dir=self.case_dir,
        )
        self.watchdog.arm()
        self.watchdog.require_armed()
        self._patch_affinity(
            self.excluded_affinity,
            present=True,
            expected=self.affinity,
            expected_present=self.affinity_present,
        )
        value = json.loads(
            self.regional.kubectl(
                "gpu", "get", "pod", "-o", "json", namespace=self.namespace
            )
        )
        for pod in value.get("items", []):
            if pod.get("spec", {}).get("nodeName") != self.node:
                continue
            if not any(
                owner.get("uid") == self.uid
                and owner.get("kind") == "DaemonSet"
                and owner.get("name") == self.name
                for owner in pod["metadata"].get("ownerReferences", [])
            ):
                continue
            uid = pod["metadata"].get("uid")
            version = pod["metadata"].get("resourceVersion")
            if not uid or not version:
                raise RegionalFixtureError("target plugin Pod UID/version is unknown")
            self.watchdog.require_armed()
            require_action_time(120)
            self.regional.kubectl(
                "gpu",
                "delete",
                "--raw",
                f"/api/v1/namespaces/{self.namespace}/pods/{pod['metadata']['name']}",
                "-f",
                "-",
                input_text=json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "DeleteOptions",
                        "preconditions": {"uid": uid, "resourceVersion": version},
                        "propagationPolicy": "Foreground",
                        "gracePeriodSeconds": 0,
                    }
                ),
                namespace=self.namespace,
            )

    def wait_allocatable(
        self, expected: int, *, timeout_seconds: int = 300
    ) -> dict[str, Any]:
        deadline = time.monotonic() + finite_seconds(timeout_seconds)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.regional.node_metadata(self.node)
            node = json.loads(
                self.regional.kubectl("gpu", "get", "node", self.node, "-o", "json")
            )
            value = int(
                node.get("status", {}).get("allocatable", {}).get(self.resource, 0)
            )
            last["allocatable"] = value
            if value == expected:
                return last
            time.sleep(5)
        raise RegionalFixtureError(
            f"{self.resource} allocatable did not become {expected}: {last}"
        )

    def restore(self) -> None:
        if self.watchdog is not None:
            self.watchdog.restore()
        if self.name:
            current = self._read()
            spec = current["spec"]["template"]["spec"]
            if (
                current["metadata"].get("uid") != self.uid
                or ("affinity" in spec) != self.affinity_present
                or spec.get("affinity") != self.affinity
                or WINDOW_KEY in (current["metadata"].get("annotations") or {})
            ):
                raise RegionalFixtureError("plugin affinity restoration is unconfirmed")
