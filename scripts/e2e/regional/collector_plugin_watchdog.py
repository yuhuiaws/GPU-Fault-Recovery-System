"""Persist, arm and retire a narrowly authorized GPU-cluster watchdog Job."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.probes.collector_plugin_watchdog import (
    WINDOW_KEY,
    WINDOW_PATH,
    restoration_patch,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    component_python,
)
from scripts.e2e.regional.seeded_command_fixture import delete_owned_resource

PROBE = Path(__file__).with_name("probes") / "collector_plugin_watchdog.py"


class PluginWatchdog:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        baseline: dict[str, Any],
        excluded_affinity: dict[str, Any],
        image: str,
        case_dir: Path,
    ) -> None:
        self.regional = regional
        self.image = image
        self.namespace = str(baseline["metadata"]["namespace"])
        self.daemonset = str(baseline["metadata"]["name"])
        self.namespace_uid = json.loads(
            regional.kubectl("gpu", "get", "namespace", self.namespace, "-o", "json")
        )["metadata"]["uid"]
        self.scope = regional.evidence_identity()
        digest = hashlib.sha256(
            f"{self.namespace}/{self.daemonset}".encode()
        ).hexdigest()[:16]
        self.path = case_dir / "plugin-watchdogs" / f"{digest}.json"
        if self.path.exists():
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            if previous.get("state") != "CLOSED" or not previous.get("window_id"):
                raise RegionalFixtureError(
                    "plugin watchdog journal exists; reconcile its owned resources first"
                )
            archive = self.path.with_name(f"{digest}-{previous['window_id']}.json")
            if (
                archive.exists()
                and json.loads(archive.read_text(encoding="utf-8")) != previous
            ):
                raise RegionalFixtureError("plugin watchdog archive identity conflicts")
            write_json_atomic(archive, previous)
        window_id = uuid4().hex
        spec = baseline["spec"]["template"]["spec"]
        self.record: dict[str, Any] = {
            "schema_version": 1,
            "window_id": window_id,
            "namespace": self.namespace,
            "namespace_uid": self.namespace_uid,
            "daemonset": self.daemonset,
            "daemonset_uid": baseline["metadata"]["uid"],
            "scope": self.scope,
            "script_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
            "baseline": {"present": "affinity" in spec, "value": spec.get("affinity")},
            "excluded_affinity": excluded_affinity,
            "restore_at": time.time() + 600,
            "resources": {},
            "state": "PREPARING",
        }
        self.name = "gpu-fault-c017-" + window_id[:16]
        self.record["name"] = self.name
        self.claimed = False

    def gpu(self, *args: str, stdin: bytes | None = None, **kwargs: Any) -> str:
        return self.regional.kubectl(
            "gpu",
            *args,
            namespace=self.namespace,
            input_text=None if stdin is None else stdin.decode(),
            **kwargs,
        )

    def read(self) -> dict[str, Any]:
        return dict(
            json.loads(self.gpu("get", "daemonset", self.daemonset, "-o", "json"))
        )

    def manifests(self) -> list[dict[str, Any]]:
        metadata = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": {
                "gpu-fault.io/acceptance-run": self.record["window_id"],
                "gpu-fault.io/acceptance-case": "GF-REGIONAL-COLLECT-017",
            },
        }
        return [
            {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": metadata},
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": metadata,
                "rules": [
                    {
                        "apiGroups": ["apps"],
                        "resources": ["daemonsets"],
                        "resourceNames": [self.daemonset],
                        "verbs": ["get", "patch"],
                    }
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": metadata,
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role",
                    "name": self.name,
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": self.name,
                        "namespace": self.namespace,
                    }
                ],
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": metadata,
                "spec": {
                    "backoffLimit": 2,
                    "activeDeadlineSeconds": 900,
                    "template": {
                        "metadata": {"labels": metadata["labels"]},
                        "spec": {
                            "serviceAccountName": self.name,
                            "restartPolicy": "Never",
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 65532,
                            },
                            "containers": [
                                {
                                    "name": "watchdog",
                                    "image": self.image,
                                    "command": [
                                        component_python("gpu"),
                                        "-c",
                                        PROBE.read_text(encoding="utf-8"),
                                    ],
                                    "env": [
                                        {
                                            "name": "COLLECTOR_PLUGIN_WINDOW",
                                            "value": json.dumps(self.record),
                                        }
                                    ],
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "readOnlyRootFilesystem": True,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                    "resources": {
                                        "requests": {"cpu": "50m", "memory": "128Mi"},
                                        "limits": {"cpu": "200m", "memory": "256Mi"},
                                    },
                                }
                            ],
                        },
                    },
                },
            },
        ]

    def arm(self) -> None:
        require_action_time(780)
        if (
            not self.record["daemonset_uid"]
            or not self.namespace_uid
            or "@sha256:" not in self.image
        ):
            raise RegionalFixtureError(
                "plugin watchdog needs immutable image and resource identities"
            )
        current = self.read()
        annotations = current["metadata"].get("annotations") or {}
        if WINDOW_KEY in annotations:
            raise RegionalFixtureError("plugin DaemonSet already has an owned window")
        if current["metadata"].get("uid") != self.record["daemonset_uid"]:
            raise RegionalFixtureError("plugin DaemonSet changed before arming")
        if not current["metadata"].get("resourceVersion"):
            raise RegionalFixtureError("plugin DaemonSet resourceVersion is unknown")
        if (current["spec"].get("updateStrategy") or {}).get("type") != "OnDelete":
            raise RegionalFixtureError(
                "plugin exclusion requires the approved OnDelete policy"
            )
        patch = [
            {
                "op": "test",
                "path": "/metadata/uid",
                "value": self.record["daemonset_uid"],
            },
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": current["metadata"]["resourceVersion"],
            },
            {"op": "test", "path": "/spec/updateStrategy/type", "value": "OnDelete"},
        ]
        if "annotations" not in current["metadata"]:
            patch.append({"op": "add", "path": "/metadata/annotations", "value": {}})
        patch.append(
            {"op": "add", "path": WINDOW_PATH, "value": self.record["window_id"]}
        )
        write_json_atomic(self.path, self.record)
        self.claimed = True
        self.gpu(
            "patch", "daemonset", self.daemonset, "--type=json", "-p", json.dumps(patch)
        )
        for manifest in self.manifests():
            require_action_time(180)
            kind = manifest["kind"].lower()
            existing = self.gpu(
                "get", kind, self.name, "--ignore-not-found", "-o", "json"
            )
            if existing.strip():
                raise RegionalFixtureError(
                    "watchdog resource already exists without creation proof"
                )
            self.record["resources"][kind] = {"attempted": True, "uid": None}
            write_json_atomic(self.path, self.record)
            output = self.gpu(
                "create",
                "-f",
                "-",
                "-o",
                "json",
                stdin=json.dumps(manifest).encode(),
                timeout=120,
            )
            self.record["resources"][kind]["uid"] = json.loads(output)["metadata"][
                "uid"
            ]
            write_json_atomic(self.path, self.record)
        self.gpu(
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            f"job-name={self.name}",
            "--timeout=120s",
            timeout=150,
        )
        receipts = self.gpu("logs", f"job/{self.name}", "--tail=20", timeout=60)
        armed = False
        for line in receipts.splitlines():
            try:
                receipt = json.loads(line)
            except ValueError:
                continue
            armed |= receipt == {
                "state": "ARMED",
                "window_id": self.record["window_id"],
                "daemonset_uid": self.record["daemonset_uid"],
                "restore_at": self.record["restore_at"],
            }
        if not armed:
            raise RegionalFixtureError(
                "plugin watchdog did not acknowledge its exact window"
            )
        self.require_armed()
        self.record["state"] = "ARMED"
        write_json_atomic(self.path, self.record)

    def require_armed(self) -> None:
        require_action_time(180)
        if time.time() + 180 >= self.record["restore_at"]:
            raise RegionalFixtureError(
                "plugin watchdog window is too close to restoration"
            )
        current = self.read()
        if (current["metadata"].get("annotations") or {}).get(
            WINDOW_KEY
        ) != self.record["window_id"]:
            raise RegionalFixtureError(
                "plugin watchdog already closed the exclusion window"
            )
        restoration_patch(current, self.record)

    def restore(self) -> None:
        if not self.claimed:
            return
        if self.regional.evidence_identity() != self.scope:
            raise RegionalFixtureError("plugin watchdog release/cluster scope changed")
        namespace = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "namespace",
                self.namespace,
                "-o",
                "json",
            )
        )
        if namespace["metadata"].get("uid") != self.namespace_uid:
            raise RegionalFixtureError("plugin watchdog namespace changed")
        patch = restoration_patch(self.read(), self.record)
        if patch:
            self.gpu(
                "patch",
                "daemonset",
                self.daemonset,
                "--type=json",
                "-p",
                json.dumps(patch),
            )
        if restoration_patch(self.read(), self.record):
            raise RegionalFixtureError(
                "plugin restoration failed readback; watchdog retained"
            )
        self.record["state"] = "RESTORED"
        write_json_atomic(self.path, self.record)
        for kind, identity in reversed(list(self.record["resources"].items())):
            raw = self.gpu("get", kind, self.name, "--ignore-not-found", "-o", "json")
            if raw.strip():
                metadata = json.loads(raw)["metadata"]
                if (
                    not identity["uid"]
                    or metadata.get("uid") != identity["uid"]
                    or (metadata.get("labels") or {}).get("gpu-fault.io/acceptance-run")
                    != self.record["window_id"]
                ):
                    raise RegionalFixtureError(
                        "watchdog resource UID proof missing or changed; retained"
                    )

                def bound_resource(
                    *args: str,
                    stdin: bytes | None = None,
                    expected_uid: str = identity["uid"],
                    **kwargs: Any,
                ) -> str:
                    output = self.gpu(*args, stdin=stdin, **kwargs)
                    if args[0] == "get" and output.strip():
                        value = json.loads(output)
                        observed = value.get("metadata", value)
                        if observed.get("uid") != expected_uid:
                            raise RegionalFixtureError(
                                "watchdog resource was replaced during cleanup"
                            )
                    return output

                delete_owned_resource(
                    kind,
                    self.name,
                    self.record["window_id"],
                    client=bound_resource,
                    namespace=self.namespace,
                )
        current = self.read()
        self.gpu(
            "patch",
            "daemonset",
            self.daemonset,
            "--type=json",
            "-p",
            json.dumps(
                [
                    {
                        "op": "test",
                        "path": "/metadata/uid",
                        "value": self.record["daemonset_uid"],
                    },
                    {
                        "op": "test",
                        "path": "/metadata/resourceVersion",
                        "value": current["metadata"]["resourceVersion"],
                    },
                    {
                        "op": "test",
                        "path": WINDOW_PATH,
                        "value": "CLOSED:" + self.record["window_id"],
                    },
                    {"op": "remove", "path": WINDOW_PATH},
                ]
            ),
        )
        if WINDOW_KEY in (self.read()["metadata"].get("annotations") or {}):
            raise RegionalFixtureError(
                "plugin watchdog ownership retirement unconfirmed"
            )
        self.record["state"] = "CLOSED"
        self.claimed = False
        write_json_atomic(self.path, self.record)
