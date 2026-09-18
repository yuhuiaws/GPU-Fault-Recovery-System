"""Owned CPU namespace lifecycle for NOTIFY008; no business resources are mutated."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Callable
from urllib.parse import quote

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.notify008_bundle import source_bundle
from scripts.e2e.regional.notify008_resources import (
    CASE_LABEL,
    GATE,
    NAME,
    OWNER_LABEL,
    RUNTIME_PYTHON,
    admitted_pod_errors,
    manifests,
    metadata,
    pod_spec_errors,
)
from scripts.e2e.regional.probes.notify008_probe import (
    PROBE_COMMANDS,
    safe_error_type,
    safe_failure_fields,
)
from scripts.e2e.regional.probes.notify008_protocol import (
    CASE_ID,
    ProbeError,
    Target,
    digest,
)
from scripts.e2e.regional.regional_commands import (
    RegionalCommandFailed,
    run_fixture_command,
)

CLEANUP_SECONDS = 150
CLUSTER_SCOPED = frozenset({"namespace", "priorityclass"})
CONTAINER_REASONS = frozenset(
    {
        "Completed",
        "Error",
        "OOMKilled",
        "ContainerCannotRun",
        "StartError",
        "DeadlineExceeded",
        "ContainerCreating",
        "PodInitializing",
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "ErrImagePull",
        "CreateContainerConfigError",
        "CreateContainerError",
        "RunContainerError",
    }
)


def probe_failure(value: Any) -> dict[str, str] | None:
    if (
        not isinstance(value, dict)
        or value.get("case_id") != CASE_ID
        or value.get("verdict") != "FAIL"
    ):
        return None
    return safe_failure_fields(value)


class CpuCommandFailed(RegionalCommandFailed):
    def __init__(self, result: CompletedProcess[str], stage: str) -> None:
        super().__init__(result.returncode, result.stderr)
        stage = (
            stage
            if stage in {"get", "create", "patch", "delete", "exec"}
            else "unknown"
        )
        self.diagnostic: dict[str, Any] = {
            "command_stage": stage,
            "error_type": "RegionalCommandFailed",
            "exit_code": result.returncode,
            "reason": "command-nonzero",
        }
        if stage == "exec" and len(result.stdout) <= 65536:
            try:
                failure = probe_failure(json.loads(result.stdout))
            except (ValueError, RecursionError):
                failure = None
            if failure is not None:
                self.diagnostic["probe"] = failure


def pod_diagnostics(pod: dict[str, Any] | None) -> dict[str, Any]:
    status = pod.get("status") if isinstance(pod, dict) else None
    status = status if isinstance(status, dict) else {}
    phase = status.get("phase")
    result: dict[str, Any] = {
        "phase": phase
        if isinstance(phase, str)
        and phase in {"Pending", "Running", "Succeeded", "Failed", "Unknown"}
        else "Unknown",
        "containers": {},
    }
    statuses = status.get("containerStatuses")
    statuses = statuses if isinstance(statuses, list) else []
    for name in ("runtime", "database"):
        matches = [
            item
            for item in statuses
            if isinstance(item, dict) and item.get("name") == name
        ]
        item = matches[0] if len(matches) == 1 else {}
        state = item.get("state")
        state = state if isinstance(state, dict) else {}
        states = [
            key
            for key in ("waiting", "running", "terminated")
            if isinstance(state.get(key), dict)
        ]
        state_name = states[0] if len(states) == 1 else "unknown"
        detail = state[state_name] if len(states) == 1 else {}
        reason = detail.get("reason")
        exit_code = detail.get("exitCode")
        restarts = item.get("restartCount")
        result["containers"][name] = {
            "ready": item.get("ready") if type(item.get("ready")) is bool else None,
            "restart_count": restarts
            if type(restarts) is int and restarts >= 0
            else None,
            "state": state_name,
            "exit_code": exit_code if type(exit_code) is int else None,
            "reason": reason
            if isinstance(reason, str) and reason in CONTAINER_REASONS
            else "Unknown",
        }
    return result


class CpuAPI:
    def __init__(self, kubeconfig: Path, context: str) -> None:
        self.kubeconfig, self.context = kubeconfig, context
        self.fingerprint = hashlib.sha256(kubeconfig.read_bytes()).hexdigest()

    def call(
        self,
        *arguments: str,
        namespace: str | None = None,
        body: Any = None,
        timeout: int = 30,
    ) -> str:
        if hashlib.sha256(self.kubeconfig.read_bytes()).hexdigest() != self.fingerprint:
            raise ProbeError("CPU kubeconfig changed during the isolated case")
        command = [
            "kubectl",
            "--kubeconfig",
            str(self.kubeconfig),
            "--context",
            self.context,
        ]
        if namespace is not None:
            command.extend(["--namespace", namespace])
        command.extend(arguments)
        result = run_fixture_command(
            command,
            input_text=None if body is None else json.dumps(body, sort_keys=True),
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            stage = (
                arguments[0]
                if arguments
                and arguments[0] in {"get", "create", "patch", "delete", "exec"}
                else "unknown"
            )
            raise CpuCommandFailed(result, stage)
        return result.stdout

    def read(
        self, kind: str, name: str, *, namespace: str | None = None
    ) -> dict[str, Any] | None:
        text = self.call(
            "get", kind, name, "--ignore-not-found", "-o", "json", namespace=namespace
        )
        if not text.strip():
            return None
        value = json.loads(text)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("metadata"), dict)
            or value["metadata"].get("name") != name
        ):
            raise ProbeError("Kubernetes object read has an invalid identity")
        return value

    def pods(self, namespace: str, job_uid: str) -> list[dict[str, Any]]:
        value = json.loads(
            self.call(
                "get",
                "pods",
                "-l",
                f"batch.kubernetes.io/controller-uid={job_uid}",
                "-o",
                "json",
                namespace=namespace,
            )
        )
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise ProbeError("owned Pod inventory is malformed")
        if any(not isinstance(item, dict) for item in value["items"]):
            raise ProbeError("owned Pod inventory contains a malformed object")
        return list(value["items"])


class Sandbox:
    def __init__(
        self,
        api: CpuAPI,
        target: Target,
        case_dir: Path,
        deadline: datetime,
        check_target: Callable[[], None],
    ) -> None:
        self.api, self.target, self.deadline = api, target, deadline
        self.check_target = check_target
        self.path = case_dir / "notify008-ownership.json"
        self.record: dict[str, Any] = {
            "schema_version": 1,
            "target": asdict(target),
            "resources": {},
            "pod": None,
            "arm_started": False,
        }
        self.desired: dict[str, dict[str, Any]] = {}
        self.started = False

    def save(self) -> None:
        write_json_atomic(self.path, self.record)

    def retain_failure(
        self,
        stage: str,
        error: BaseException,
        pod: dict[str, Any] | None,
        *,
        report: Any = None,
    ) -> None:
        if "first_failure" in self.record:
            return
        failure: dict[str, Any] = {
            "stage": stage
            if stage in {*PROBE_COMMANDS, "pod-admission", "pod-readiness"}
            else "unknown",
            "error_type": safe_error_type(error),
            "exit_code": None,
            "reason": {
                "pod-admission": "pod-validation-failed",
                "pod-readiness": "pod-not-ready",
            }.get(stage, "probe-operation-failed"),
            "pod_status": pod_diagnostics(pod),
        }
        if isinstance(error, CpuCommandFailed):
            failure.update(error.diagnostic)
        remote = probe_failure(report)
        if remote is not None:
            failure["probe"] = remote
        self.record["first_failure"] = failure
        self.save()

    def remaining(self) -> int:
        value = (self.deadline - datetime.now(UTC)).total_seconds()
        if value < 1:
            raise ProbeError("approved NOTIFY008 maintenance window ended")
        return max(1, int(value))

    def before_work(self) -> None:
        self.remaining()
        self.check_target()
        if "priorityclass" in self.record["resources"]:
            self.check_priority_class()

    def read(self, kind: str) -> dict[str, Any] | None:
        return self.api.read(
            kind,
            self.target.run_id if kind in CLUSTER_SCOPED else NAME,
            namespace=None if kind in CLUSTER_SCOPED else self.target.run_id,
        )

    def owned(
        self, kind: str, value: dict[str, Any], *, create_ack: bool = False
    ) -> str:
        meta = value.get("metadata") or {}
        uid = meta.get("uid")
        expected = self.record["resources"].get(kind, {}).get("uid")
        labels = meta.get("labels") or {}
        cluster_scoped = kind in CLUSTER_SCOPED
        if (
            not isinstance(uid, str)
            or not uid
            or meta.get("name") != (self.target.run_id if cluster_scoped else NAME)
            or labels.get(OWNER_LABEL) != self.target.run_id
            or labels.get(CASE_LABEL) != CASE_ID
            or (expected is not None and uid != expected)
            or meta.get("namespace", "")
            != ("" if cluster_scoped else self.target.run_id)
        ):
            raise ProbeError("sandbox resource ownership or UID differs")
        if kind == "priorityclass" and (
            (expected is None and not create_ack)
            or value.get("apiVersion") != "scheduling.k8s.io/v1"
            or value.get("kind") != "PriorityClass"
            or meta.get("ownerReferences")
            != self.desired["priorityclass"]["metadata"]["ownerReferences"]
        ):
            raise ProbeError(
                "sandbox PriorityClass ownership or acknowledged UID differs"
            )
        return uid

    def check_priority_class(self) -> None:
        value = self.read("priorityclass")
        if value is None:
            raise ProbeError("owned PriorityClass disappeared")
        self.validate_resource("priorityclass", value)

    def validate_resource(self, kind: str, value: dict[str, Any]) -> None:
        uid = self.owned(kind, value)
        if kind == "namespace":
            required = metadata(self.target, namespaced=False)["labels"]
            if any(
                value["metadata"].get("labels", {}).get(key) != item
                for key, item in required.items()
            ):
                raise ProbeError("sandbox namespace security labels differ")
            return
        desired = self.desired[kind]
        if kind == "priorityclass":
            valid = (
                not (set(value) - set(desired))
                and type(value.get("value")) is int
                and value["value"] == 0
                and value.get("globalDefault", False) is False
                and value.get("preemptionPolicy") == "Never"
                and not value["metadata"].get("deletionTimestamp")
                and not value["metadata"].get("finalizers")
            )
        elif kind == "configmap":
            valid = (
                value.get("immutable") is True
                and value.get("data") == desired["data"]
                and not value.get("binaryData")
            )
        elif kind == "serviceaccount":
            valid = (
                value.get("automountServiceAccountToken") is False
                and not value.get("secrets")
                and not value.get("imagePullSecrets")
                and not value["metadata"].get("annotations")
            )
        elif kind == "networkpolicy":
            policy = value.get("spec") or {}
            normalized = {
                **policy,
                "ingress": policy.get("ingress", []),
                "egress": policy.get("egress", []),
            }
            valid = digest(normalized) == digest(desired["spec"])
        else:
            spec = value.get("spec") or {}
            expected = desired["spec"]
            allowed = {
                "selector",
                "manualSelector",
                "completionMode",
                "podReplacementPolicy",
            }
            valid = not (set(spec) - set(expected) - allowed)
            valid = valid and all(
                digest(spec.get(key)) == digest(item)
                for key, item in expected.items()
                if key != "template"
            )
            template = spec.get("template") or {}
            wanted_template = expected["template"]
            valid = valid and not pod_spec_errors(
                template.get("spec") or {}, wanted_template["spec"], gated=True
            )
            valid = valid and not template.get("spec", {}).get("nodeName")
            labels = (template.get("metadata") or {}).get("labels") or {}
            valid = valid and all(
                labels.get(key) == item
                for key, item in wanted_template["metadata"]["labels"].items()
            )
            valid = (
                valid
                and spec.get("manualSelector", False) is False
                and spec.get("completionMode", "NonIndexed") == "NonIndexed"
            )
            valid = valid and spec.get(
                "podReplacementPolicy", "TerminatingOrFailed"
            ) in {"TerminatingOrFailed", "Failed"}
            selector = (spec.get("selector") or {}).get("matchLabels") or {}
            valid = valid and all(
                key in {"controller-uid", "batch.kubernetes.io/controller-uid"}
                and item == uid
                for key, item in selector.items()
            )
        if not valid:
            raise ProbeError(f"admitted sandbox {kind} specification differs")

    def create_resource(self, kind: str, document: dict[str, Any]) -> None:
        self.before_work()
        if self.read(kind) is not None:
            raise ProbeError("sandbox refuses a preexisting resource")
        self.record["resources"][kind] = {"create_started": True, "uid": None}
        self.save()
        try:
            value = json.loads(
                self.api.call("create", "-f", "-", "-o", "json", body=document)
            )
        except Exception:
            if kind == "priorityclass":
                self.record["resources"][kind]["create_ack_lost"] = True
                self.save()
                raise
            value = self.read(kind)
            if value is None:
                raise
            self.record["resources"][kind]["create_ack_lost"] = True
        if kind == "priorityclass":
            # Only the create response can establish class UID custody.
            self.record["resources"][kind]["uid"] = self.owned(
                kind, value, create_ack=True
            )
            self.save()
        self.validate_resource(kind, value)
        self.record["resources"][kind]["uid"] = self.owned(kind, value)
        self.save()
        if kind == "priorityclass":
            self.check_priority_class()

    def create(self) -> None:
        self.before_work()
        if self.path.exists():
            raise ProbeError(
                "sandbox ownership journal already exists; no automatic takeover"
            )
        if self.read("priorityclass") is not None:
            raise ProbeError("sandbox refuses a preexisting PriorityClass")
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(self.record, stream)
        self.started = True
        self.create_resource(
            "namespace",
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": metadata(self.target, namespaced=False),
            },
        )
        namespace_uid = self.record["resources"]["namespace"]["uid"]
        self.desired = manifests(self.target, namespace_uid, source_bundle())
        for kind in (
            "priorityclass",
            "serviceaccount",
            "networkpolicy",
            "configmap",
            "job",
        ):
            self.create_resource(kind, self.desired[kind])

    def patch(
        self, kind: str, value: dict[str, Any], changes: list[dict[str, Any]]
    ) -> None:
        meta = value["metadata"]
        if (
            not isinstance(meta.get("resourceVersion"), str)
            or not meta["resourceVersion"]
        ):
            raise ProbeError("sandbox patch lacks a resource version")
        self.before_work()
        self.api.call(
            "patch",
            kind,
            meta["name"],
            "--type=json",
            "-p",
            json.dumps(
                [
                    {"op": "test", "path": "/metadata/uid", "value": meta["uid"]},
                    {
                        "op": "test",
                        "path": "/metadata/resourceVersion",
                        "value": meta["resourceVersion"],
                    },
                    *changes,
                ]
            ),
            namespace=self.target.run_id,
        )

    def current_pod(self) -> dict[str, Any] | None:
        job_uid = self.record["resources"].get("job", {}).get("uid")
        if not job_uid:
            return None
        pods = self.api.pods(self.target.run_id, job_uid)
        if len(pods) > 1:
            raise ProbeError("isolated Job has more than one Pod")
        if not pods:
            return None
        pod = pods[0]
        meta = pod.get("metadata") or {}
        known = self.record["pod"]
        if known is not None and (
            meta.get("uid") != known["uid"] or meta.get("name") != known["name"]
        ):
            raise ProbeError("sandbox Pod UID or name changed")
        return pod

    def check_pod(self, pod: dict[str, Any], *, gated: bool) -> None:
        errors = admitted_pod_errors(
            pod,
            self.desired["job"]["spec"]["template"]["spec"],
            self.target,
            self.record["resources"]["job"]["uid"],
            gated=gated,
        )
        if errors:
            if "pod_validation_failure" not in self.record:
                digests = {}
                for item in (pod.get("status") or {}).get("containerStatuses") or []:
                    if not isinstance(item, dict) or item.get("name") not in {
                        "runtime",
                        "database",
                    }:
                        continue
                    match = re.search(
                        r"sha256:([0-9a-f]{64})$", str(item.get("imageID") or "")
                    )
                    digests[item["name"]] = match.group(1) if match else None
                self.record["pod_validation_failure"] = {
                    "gated": gated,
                    "errors": errors,
                    "resolved_image_sha256": digests,
                }
                self.save()
            error = ProbeError("; ".join(errors))
            self.retain_failure("pod-admission", error, pod)
            raise error

    def wait_pod(self, *, gated: bool) -> dict[str, Any]:
        until = time.monotonic() + min(90, self.remaining())
        pod = None
        while time.monotonic() < until:
            self.before_work()
            pod = self.current_pod()
            if pod is not None:
                if gated or pod.get("status", {}).get("phase") == "Running":
                    self.check_pod(pod, gated=gated)
                    return pod
                if pod.get("status", {}).get("phase") in {"Failed", "Succeeded"}:
                    error = ProbeError("sandbox Pod terminated before admission")
                    self.retain_failure("pod-readiness", error, pod)
                    raise error
            time.sleep(0.2)
        error = ProbeError("sandbox Pod readiness timed out")
        self.retain_failure("pod-readiness", error, pod)
        raise error

    def admit(self) -> None:
        self.check_priority_class()
        job = self.read("job")
        if job is None:
            raise ProbeError("owned Job disappeared")
        self.validate_resource("job", job)
        self.patch(
            "job", job, [{"op": "replace", "path": "/spec/suspend", "value": False}]
        )
        self.desired["job"]["spec"]["suspend"] = False
        pod = self.wait_pod(gated=True)
        meta = pod["metadata"]
        self.record["pod"] = {"name": meta["name"], "uid": meta["uid"]}
        self.save()
        self.patch(
            "pod",
            pod,
            [
                {
                    "op": "test",
                    "path": "/spec/schedulingGates",
                    "value": [{"name": GATE}],
                },
                {"op": "remove", "path": "/spec/schedulingGates"},
            ],
        )
        pod = self.wait_pod(gated=False)
        self.record["pod"]["containers"] = {
            item["name"]: item["containerID"]
            for item in pod["status"]["containerStatuses"]
        }
        if any(not value for value in self.record["pod"]["containers"].values()):
            raise ProbeError("sandbox container identity is missing")
        self.save()

    def execute_probe(
        self, command: str, *, timeout: int = 60, cleanup: bool = False
    ) -> dict[str, Any]:
        pod = None
        value: Any = None
        try:
            if not cleanup:
                self.before_work()
            namespace = self.read("namespace")
            if namespace is None:
                raise ProbeError("sandbox namespace disappeared")
            self.validate_resource("namespace", namespace)
            pod = self.current_pod()
            if cleanup and "pre_cleanup_pod_status" not in self.record:
                self.record["pre_cleanup_pod_status"] = pod_diagnostics(pod)
                self.save()
            if pod is None:
                raise ProbeError("sandbox Pod disappeared")
            self.check_pod(pod, gated=False)
            ids = {
                item["name"]: item.get("containerID")
                for item in pod["status"]["containerStatuses"]
            }
            if ids != self.record["pod"].get("containers"):
                raise ProbeError("sandbox container was replaced")
            config = json.loads(self.desired["configmap"]["data"]["config.json"])
            self.record[f"{command}_started"] = True
            self.save()
            text = self.api.call(
                "exec",
                self.record["pod"]["name"],
                "-c",
                "runtime",
                "--",
                RUNTIME_PYTHON,
                "-m",
                "scripts.e2e.regional.probes.notify008_probe",
                command,
                "--expected-pod-uid",
                self.record["pod"]["uid"],
                "--expected-namespace-uid",
                self.record["resources"]["namespace"]["uid"],
                "--bundle-sha256",
                config["source_sha256"],
                namespace=self.target.run_id,
                timeout=timeout if cleanup else min(timeout, self.remaining()),
            )
            value = json.loads(text)
            if not isinstance(value, dict) or value.get("verdict") == "FAIL":
                raise ProbeError("isolated probe refused its operation")
            if command in {"inspect", "arm", "stop"} and (
                value.get("pod_uid") != self.record["pod"]["uid"]
                or value.get("run_id") != self.target.run_id
            ):
                raise ProbeError("isolated probe acknowledgement identity differs")
            return value
        except BaseException as exc:
            self.retain_failure(command, exc, pod, report=value)
            raise

    def arm(self) -> dict[str, Any]:
        try:
            result = self.execute_probe("arm")
        except Exception:
            result = self.execute_probe("inspect")
        if result.get("armed") is not True or result.get("stop_requested") is not False:
            raise ProbeError("sandbox ARM was not independently acknowledged")
        return result

    def termination_proof(self) -> bool:
        pod = self.current_pod()
        if pod is None:
            return not self.record.get("arm_started", False)
        if not self.record.get("arm_started", False) and pod.get("spec", {}).get(
            "schedulingGates"
        ) == [{"name": GATE}]:
            return True
        states = pod.get("status", {}).get("containerStatuses")
        expected = (self.record.get("pod") or {}).get("containers") or {}
        return (
            isinstance(states, list)
            and len(states) == 2
            and all(isinstance(item, dict) for item in states)
            and {item.get("name") for item in states} == {"runtime", "database"}
            and all(
                isinstance(item.get("containerID"), str)
                and bool(item["containerID"])
                and item["containerID"] == expected.get(item["name"])
                and isinstance(item.get("state"), dict)
                and isinstance(item["state"].get("terminated"), dict)
                and type(item["state"]["terminated"].get("exitCode")) is int
                for item in states
            )
        )

    def cleanup(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "namespace_absent": False,
            "priorityclass_absent": False,
            "process_termination_proven": False,
            "errors": [],
        }
        entry = self.record["resources"].get("namespace")
        if not self.started or not entry or not entry.get("create_started"):
            return {
                **result,
                "process_termination_proven": True,
                "errors": [
                    "no current resource custody; namespace absence not observed"
                ],
            }
        try:
            namespace = self.read("namespace")
            if namespace is None:
                result["namespace_absent"] = True
                result["process_termination_proven"] = not self.record["arm_started"]
            else:
                uid = self.owned("namespace", namespace)
                if self.record.get("pod") is not None:
                    try:
                        self.execute_probe("stop", timeout=15, cleanup=True)
                    except Exception:
                        pass
                until = time.monotonic() + 30
                while time.monotonic() < until:
                    if self.termination_proof():
                        result["process_termination_proven"] = True
                        break
                    time.sleep(0.2)
                try:
                    self.api.call(
                        "delete",
                        "--raw",
                        f"/api/v1/namespaces/{quote(self.target.run_id, safe='')}",
                        "-f",
                        "-",
                        body={
                            "apiVersion": "v1",
                            "kind": "DeleteOptions",
                            "preconditions": {"uid": uid},
                            "propagationPolicy": "Foreground",
                        },
                        timeout=30,
                    )
                except Exception:
                    self.record["namespace_delete_ack_lost"] = True
                until = time.monotonic() + 30
                while time.monotonic() < until:
                    remaining = self.read("namespace")
                    if remaining is None:
                        result["namespace_absent"] = True
                        break
                    self.owned("namespace", remaining)
                    time.sleep(0.2)
        except Exception as exc:
            result["errors"].append(safe_error_type(exc))
        try:
            result["priorityclass_absent"] = self.cleanup_priority_class()
        except Exception as exc:
            result["errors"].append(f"PriorityClass cleanup: {safe_error_type(exc)}")
        if (
            not result["namespace_absent"]
            or not result["priorityclass_absent"]
            or not result["process_termination_proven"]
        ):
            result["errors"].append(
                "owned namespace/PriorityClass cleanup or process termination is unproven"
            )
        self.record["cleanup"] = result
        self.save()
        return result

    def cleanup_priority_class(self) -> bool:
        value = self.read("priorityclass")
        if value is None:
            return True
        entry = self.record["resources"].get("priorityclass") or {}
        if entry.get("create_started") is not True:
            raise ProbeError("no current PriorityClass custody")
        if entry.get("uid") is not None:
            uid = self.owned("priorityclass", value)
            version = value["metadata"].get("resourceVersion")
            if not isinstance(version, str) or not version:
                raise ProbeError("PriorityClass deletion lacks a resource version")
            try:
                self.api.call(
                    "delete",
                    "--raw",
                    "/apis/scheduling.k8s.io/v1/priorityclasses/"
                    + quote(self.target.run_id, safe=""),
                    "-f",
                    "-",
                    body={
                        "apiVersion": "v1",
                        "kind": "DeleteOptions",
                        "preconditions": {"uid": uid, "resourceVersion": version},
                        "propagationPolicy": "Foreground",
                    },
                    timeout=30,
                )
            except Exception:
                self.record["priorityclass_delete_ack_lost"] = True
        # An unknown create UID permits only observation of Namespace-owner GC.
        until = time.monotonic() + 30
        while time.monotonic() < until:
            remaining = self.read("priorityclass")
            if remaining is None:
                return True
            if entry.get("uid") is not None:
                self.owned("priorityclass", remaining)
            time.sleep(0.2)
        return False
