"""Bounded current-log prefix receipts around a fixed token quiet window."""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.admin import rotate_token_acceptance as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.execution import (
    DeploymentDeadlineExceeded,
    current_deadline,
    remaining_timeout,
)
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT

NOW = datetime(2026, 9, 11, 10, 20, tzinfo=UTC)
QUIET_START = NOW - timedelta(seconds=60)
AUTH = "regional cluster gpu-a authenticated with the retiring token"


def _line(stamp: datetime, message: str) -> bytes:
    return f"{stamp.isoformat()} {message}\n".encode()


def _pod(name: str, namespace: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "uid": f"uid-{name}",
            "labels": {"app": CPU_INGRESS_DEPLOYMENT},
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "ReplicaSet",
                    "name": f"{CPU_INGRESS_DEPLOYMENT}-revision",
                    "uid": "uid-ingress-replicaset",
                    "controller": True,
                }
            ],
        },
        "spec": {"containers": [{"name": "api"}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "api",
                    "ready": True,
                    "restartCount": 0,
                    "containerID": f"containerd://{name}-api",
                    "state": {
                        "running": {"startedAt": (NOW - timedelta(hours=1)).isoformat()}
                    },
                }
            ],
        },
    }


class LogHost:
    def __init__(self, *names: str, namespace: str = "system") -> None:
        self.site = SimpleNamespace(
            release_config={"cpu_kubeconfig": "/cpu/config", "namespace": namespace}
        )
        self.pods = [_pod(name, namespace) for name in names or ("api-a",)]
        self.after_pods = copy.deepcopy(self.pods)
        self.deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": CPU_INGRESS_DEPLOYMENT,
                "namespace": namespace,
                "uid": "uid-ingress-deployment",
                "generation": 3,
                "creationTimestamp": (NOW - timedelta(days=1)).isoformat(),
            },
            "spec": {
                "replicas": len(self.pods),
                "selector": {"matchLabels": {"app": CPU_INGRESS_DEPLOYMENT}},
            },
            "status": {
                "observedGeneration": 3,
                "replicas": len(self.pods),
                "updatedReplicas": len(self.pods),
                "readyReplicas": len(self.pods),
                "availableReplicas": len(self.pods),
            },
        }
        self.after_deployment = copy.deepcopy(self.deployment)
        self.replicasets = [
            {
                "apiVersion": "apps/v1",
                "kind": "ReplicaSet",
                "metadata": {
                    "name": f"{CPU_INGRESS_DEPLOYMENT}-revision",
                    "namespace": namespace,
                    "uid": "uid-ingress-replicaset",
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": CPU_INGRESS_DEPLOYMENT,
                            "uid": "uid-ingress-deployment",
                            "controller": True,
                        }
                    ],
                },
            }
        ]
        self.after_replicasets = copy.deepcopy(self.replicasets)
        self.prefix = {
            item["metadata"]["name"]: _line(
                QUIET_START - timedelta(seconds=10), "anchor"
            )
            for item in self.pods
        }
        self.after_prefix = dict(self.prefix)
        self.window = {
            name: _line(NOW - timedelta(seconds=1), "healthy") for name in self.prefix
        }
        self.clock = NOW
        self.prefix_delay = timedelta()
        self.queries: list[tuple[str, str, int | None]] = []
        self.prefix_calls: dict[str, int] = {}
        self.gets = 0
        self.resource_queries: list[str] = []
        self.output_prefix: str | None = None
        self.ignore_limit = False

    def read_resource(self, resource: str) -> dict:
        self.resource_queries.append(resource)
        first = self.resource_queries.count(resource) <= (
            2 if resource == "deployment" else 1
        )
        if resource == "deployment":
            return self.deployment if first else self.after_deployment
        if resource == "replicasets":
            return {
                "apiVersion": "apps/v1",
                "kind": "ReplicaSetList",
                "items": self.replicasets if first else self.after_replicasets,
            }
        assert resource == "pods", "unexpected source-proof resource"
        self.gets += 1
        return {
            "apiVersion": "v1",
            "kind": "PodList",
            "items": self.pods if first else self.after_pods,
        }

    def run(self, arguments, *, timeout_seconds):
        if "get" in arguments:
            assert timeout_seconds == 120
            resource = arguments[arguments.index("get") + 1]
            if resource == "deployment":
                assert arguments[arguments.index("get") + 2] == CPU_INGRESS_DEPLOYMENT
            else:
                assert arguments[arguments.index("-l") + 1] == (
                    f"app={CPU_INGRESS_DEPLOYMENT}"
                )
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(self.read_resource(resource)), ""
            )
        assert timeout_seconds == 180
        for option in ("--container=api", "--timestamps", "--prefix", "--tail=-1"):
            assert option in arguments
        assert "--all-containers" not in arguments
        name = arguments[arguments.index("logs") + 1]
        limit_option = next(
            (item for item in arguments if item.startswith("--limit-bytes=")), None
        )
        if limit_option:
            limit = int(limit_option.split("=", 1)[1])
            count = self.prefix_calls.get(name, 0) + 1
            self.prefix_calls[name] = count
            body = (self.prefix if count == 1 else self.after_prefix)[name]
            self.queries.append((name, "prefix", limit))
            if not self.ignore_limit:
                body = body[:limit]
            self.clock += self.prefix_delay
        else:
            assert "--since-time=2026-09-11T10:19:00Z" in arguments
            assert not any(argument.startswith("--since=") for argument in arguments), (
                "quiet log reads must not use a relative --since window"
            )
            self.queries.append((name, "window", None))
            body = self.window[name]
        prefix = self.output_prefix or f"[pod/{name}/api] "
        output = "".join(
            prefix + line for line in body.decode().splitlines(keepends=True)
        )
        return subprocess.CompletedProcess(arguments, 0, output, "")

    def collect(self) -> list[str]:
        return module.collect_retiring_token_authentications(
            self.site, "gpu-a", since_seconds=60, run=self.run, now=lambda: self.clock
        )


def test_every_api_container_has_prefix_receipts_around_the_fixed_window() -> None:
    host = LogHost("api-a", "api-b")
    assert host.collect() == []
    assert host.queries == [
        ("api-a", "prefix", 4096),
        ("api-b", "prefix", 4096),
        ("api-a", "window", None),
        ("api-b", "window", None),
        ("api-a", "prefix", len(host.prefix["api-a"])),
        ("api-b", "prefix", len(host.prefix["api-b"])),
    ]
    assert host.gets == 2
    assert (
        host.resource_queries == ["deployment", "replicasets", "pods", "deployment"] * 2
    )


@pytest.mark.parametrize("reported_replicas", [1, 2])
def test_a_deleted_desired_replica_cannot_hide_recent_retiring_authentication(
    reported_replicas: int,
) -> None:
    host = LogHost("api-a", "api-b")
    host.window["api-b"] = _line(QUIET_START + timedelta(seconds=1), AUTH)
    host.pods.pop()
    host.after_pods.pop()
    for deployment in (host.deployment, host.after_deployment):
        for field in (
            "replicas",
            "readyReplicas",
            "updatedReplicas",
            "availableReplicas",
        ):
            deployment["status"][field] = reported_replicas

    with pytest.raises(BootstrapError, match="replica"):
        host.collect()
    assert host.queries == [], "an incomplete replica set must block log acceptance"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("kind",), "ReplicaSet"),
        (("metadata", "name"), "unmanaged-ingress"),
        (("metadata", "namespace"), "other-system"),
        (("metadata", "uid"), ""),
        (("metadata", "generation"), 0),
        (("metadata", "generation"), True),
        (("metadata", "generation"), "3"),
        (("metadata", "deletionTimestamp"), NOW.isoformat()),
        (("metadata", "creationTimestamp"), NOW.isoformat()),
        (("spec", "replicas"), 0),
        (("spec", "replicas"), -1),
        (("spec", "replicas"), True),
        (("spec", "replicas"), "1"),
        (("spec", "replicas"), None),
        (("spec", "paused"), True),
        (("spec", "selector", "matchLabels"), {"app": "other-ingress"}),
        (
            ("spec", "selector", "matchExpressions"),
            [{"key": "app", "operator": "Exists"}],
        ),
        (("status", "observedGeneration"), 2),
        (("status", "observedGeneration"), 4),
        (("status", "observedGeneration"), "3"),
        (("status", "replicas"), 2),
        (("status", "updatedReplicas"), 0),
        (("status", "readyReplicas"), True),
        (("status", "availableReplicas"), None),
        (("status", "unavailableReplicas"), 1),
        (("status", "terminatingReplicas"), 1),
    ],
)
def test_unproved_deployment_identity_or_health_blocks_log_reads(
    path: tuple[str, ...], value: object
) -> None:
    host = LogHost()
    parent = host.deployment
    for field in path[:-1]:
        parent = parent[field]
    parent[path[-1]] = value

    with pytest.raises(BootstrapError, match="replica"):
        host.collect()
    assert host.resource_queries == ["deployment"]
    assert host.queries == [], "an unproved Deployment must block all log reads"


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("resource", ["deployment", "replicasets", "pods"])
@pytest.mark.parametrize("failure", ["forbidden", "invalid-json", "missing"])
def test_failed_or_malformed_source_reads_never_prove_quiet(
    phase: str, resource: str, failure: str
) -> None:
    host = LogHost()
    run = host.run
    marker = "synthetic-unstructured-auth-value-13579"

    def failed_read(arguments, **kwargs):
        if (
            "get" in arguments
            and arguments[arguments.index("get") + 1] == resource
            and (phase == "before" or host.queries)
        ):
            return subprocess.CompletedProcess(
                arguments,
                1 if failure == "forbidden" else 0,
                marker if failure == "invalid-json" else "{}",
                f"Forbidden\n{marker}",
            )
        return run(arguments, **kwargs)

    host.run = failed_read
    with pytest.raises(BootstrapError) as error:
        host.collect()
    assert marker not in str(error.value)
    assert bool(host.queries) == (phase == "after"), (
        "source-read failure must block acceptance regardless of log availability"
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "extra",
        "duplicate-uid",
        "duplicate-name",
        "terminating",
        "not-ready",
        "api-not-ready",
        "wrong-namespace",
        "wrong-label",
        "orphan",
        "foreign-replicaset",
        "wrong-replicaset-name",
        "non-controller",
        "ambiguous-controller",
    ],
)
def test_complete_counts_cannot_substitute_for_healthy_owned_pods(invalid: str) -> None:
    host = LogHost("api-a", "api-b")
    pod = host.pods[1]
    metadata = pod["metadata"]
    if invalid == "extra":
        host.pods.append(_pod("api-c", "system"))
    elif invalid == "duplicate-uid":
        metadata["uid"] = host.pods[0]["metadata"]["uid"]
    elif invalid == "duplicate-name":
        metadata["name"] = host.pods[0]["metadata"]["name"]
    elif invalid == "terminating":
        metadata["deletionTimestamp"] = NOW.isoformat()
    elif invalid == "not-ready":
        pod["status"]["conditions"][0]["status"] = "False"
    elif invalid == "api-not-ready":
        pod["status"]["containerStatuses"][0]["ready"] = False
    elif invalid == "wrong-namespace":
        metadata["namespace"] = "other-system"
    elif invalid == "wrong-label":
        metadata["labels"]["app"] = "other-ingress"
    elif invalid == "orphan":
        metadata["ownerReferences"].clear()
    elif invalid == "foreign-replicaset":
        metadata["ownerReferences"][0]["uid"] = "foreign-replicaset"
    elif invalid == "wrong-replicaset-name":
        metadata["ownerReferences"][0]["name"] = "other-replicaset"
    elif invalid == "non-controller":
        metadata["ownerReferences"][0]["controller"] = False
    else:
        metadata["ownerReferences"].append(dict(metadata["ownerReferences"][0]))

    with pytest.raises(BootstrapError, match="stable Ready source"):
        host.collect()
    assert host.queries == [], "every desired Pod must have a healthy owned API source"


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize(
    "invalid",
    [
        "absent",
        "uid",
        "namespace",
        "terminating",
        "owner-uid",
        "owner-name",
        "owner-kind",
        "non-controller",
    ],
)
def test_pod_labels_and_replica_counts_do_not_prove_deployment_ownership(
    phase: str, invalid: str
) -> None:
    host = LogHost()
    replicasets = host.replicasets if phase == "before" else host.after_replicasets
    metadata = replicasets[0]["metadata"]
    if invalid == "absent":
        replicasets.clear()
    elif invalid == "uid":
        metadata["uid"] = "recreated-replicaset"
    elif invalid == "namespace":
        metadata["namespace"] = "other-system"
    elif invalid == "terminating":
        metadata["deletionTimestamp"] = NOW.isoformat()
    elif invalid.startswith("owner-"):
        metadata["ownerReferences"][0][invalid.removeprefix("owner-")] = "other-owner"
    else:
        metadata["ownerReferences"][0]["controller"] = False

    with pytest.raises(BootstrapError, match="replica"):
        host.collect()
    assert bool(host.queries) == (phase == "after"), (
        "ownership must be proved both before and after log collection"
    )


def test_a_pod_cannot_change_to_another_owned_replicaset_during_the_probe() -> None:
    host = LogHost()
    other = copy.deepcopy(host.replicasets[0])
    other["metadata"]["uid"] = "uid-other-replicaset"
    other["metadata"]["name"] = "other-replicaset"
    host.after_replicasets.append(other)
    host.after_pods[0]["metadata"]["ownerReferences"][0].update(
        uid=other["metadata"]["uid"], name=other["metadata"]["name"]
    )

    with pytest.raises(BootstrapError, match="sources changed"):
        host.collect()


@pytest.mark.parametrize("change", ["uid", "generation", "scale-down", "scale-up"])
def test_healthy_deployment_drift_after_log_collection_refuses(change: str) -> None:
    host = LogHost("api-a", "api-b")
    deployment = host.after_deployment
    if change == "uid":
        deployment["metadata"]["uid"] = "recreated-deployment"
        host.after_replicasets[0]["metadata"]["ownerReferences"][0]["uid"] = deployment[
            "metadata"
        ]["uid"]
    else:
        deployment["metadata"]["generation"] += 1
        deployment["status"]["observedGeneration"] += 1
        if change == "scale-down":
            host.after_pods.pop()
        elif change == "scale-up":
            host.after_pods.append(_pod("api-c", "system"))
        deployment["spec"]["replicas"] = len(host.after_pods)
        for field in (
            "replicas",
            "readyReplicas",
            "updatedReplicas",
            "availableReplicas",
        ):
            deployment["status"][field] = len(host.after_pods)

    with pytest.raises(BootstrapError, match="sources changed"):
        host.collect()
    assert host.prefix_calls == {"api-a": 2, "api-b": 2}


def test_deployment_status_updates_do_not_change_bound_identity() -> None:
    host = LogHost()
    host.deployment["metadata"]["resourceVersion"] = "10"
    host.after_deployment["metadata"]["resourceVersion"] = "11"
    host.after_deployment["status"]["unavailableReplicas"] = 0

    assert host.collect() == []


def test_deployment_drift_while_reading_the_final_pod_list_refuses() -> None:
    host = LogHost()
    run = host.run

    def changed_during_pod_read(arguments, **kwargs):
        if "get" in arguments and "pods" in arguments and host.gets:
            host.after_deployment["metadata"]["generation"] += 1
            host.after_deployment["status"]["observedGeneration"] += 1
        return run(arguments, **kwargs)

    host.run = changed_during_pod_read
    with pytest.raises(BootstrapError, match="sources changed"):
        host.collect()
    assert host.prefix_calls == {"api-a": 2}


@pytest.mark.parametrize("change", ["missing-pod", "recreated", "generation", "scale"])
def test_the_initial_quiet_wait_cannot_hide_ingress_deployment_drift(
    change: str,
) -> None:
    host = LogHost("api-a", "api-b")
    host.window["api-b"] = _line(QUIET_START + timedelta(seconds=1), AUTH)
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds
        deployment = host.after_deployment
        if change in {"missing-pod", "scale"}:
            host.after_pods.pop()
        if change == "recreated":
            deployment["metadata"]["uid"] = "recreated-deployment"
            host.after_replicasets[0]["metadata"]["ownerReferences"][0]["uid"] = (
                deployment["metadata"]["uid"]
            )
        if change in {"generation", "scale"}:
            deployment["metadata"]["generation"] += 1
            deployment["status"]["observedGeneration"] += 1
        if change == "scale":
            deployment["spec"]["replicas"] = 1
            for field in (
                "replicas",
                "readyReplicas",
                "updatedReplicas",
                "availableReplicas",
            ):
                deployment["status"][field] = 1

    with pytest.raises(BootstrapError, match="replica|sources changed"):
        module.wait_for_quiet_token_authentications(
            host.site,
            "gpu-a",
            quiet_seconds=60,
            timeout_seconds=120,
            read_logs=lambda *_args, **_kwargs: host.collect(),
            sleep=sleep,
            monotonic=lambda: clock[0],
            run=host.run,
            now=lambda: QUIET_START,
        )
    assert clock == [60.0]
    assert host.resource_queries[:3] == ["deployment", "replicasets", "pods"]


def test_the_initial_source_read_must_succeed_before_waiting() -> None:
    host = LogHost()

    def failed_read(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 1, "", "Forbidden")

    def unexpected_wait(*_args, **_kwargs):
        pytest.fail("an unproved initial source set must stop before sleeping or logs")

    with pytest.raises(BootstrapError, match="cannot verify"):
        module.wait_for_quiet_token_authentications(
            host.site,
            "gpu-a",
            quiet_seconds=60,
            timeout_seconds=120,
            read_logs=unexpected_wait,
            sleep=unexpected_wait,
            monotonic=lambda: 0.0,
            run=failed_read,
            now=lambda: QUIET_START,
        )


def test_appended_logs_do_not_change_the_bounded_prefix_receipt() -> None:
    host = LogHost()
    host.after_prefix["api-a"] += _line(NOW, "a new append")
    assert host.collect() == []
    assert host.queries[-1] == ("api-a", "prefix", len(host.prefix["api-a"]))


def test_a_cap_cut_trailing_fragment_is_not_part_of_the_complete_receipt() -> None:
    host = LogHost()
    anchor = host.prefix["api-a"]
    host.prefix["api-a"] += _line(QUIET_START, "x" * 5000)
    host.after_prefix["api-a"] = host.prefix["api-a"] + _line(NOW, "appended")
    assert host.collect() == []
    assert host.queries[-1] == ("api-a", "prefix", len(anchor))


def test_a_trailing_fragment_below_the_byte_cap_is_not_a_retention_receipt() -> None:
    host = LogHost()
    host.prefix["api-a"] += _line(QUIET_START, AUTH)[:-1]
    host.after_prefix["api-a"] = host.prefix["api-a"]
    host.window["api-a"] = b""

    with pytest.raises(BootstrapError, match="incomplete"):
        host.collect()
    assert host.queries == [("api-a", "prefix", 4096)]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"", "prefix is missing"),
        (b"x" * 4096, "no complete timestamped record"),
        (
            _line(QUIET_START - timedelta(seconds=1), "x" * 5000),
            "no complete timestamped record",
        ),
        (_line(QUIET_START + timedelta(seconds=1), "rotated"), "starts after"),
        (b"not-a-timestamp anchor\n", "invalid timestamp"),
        (b"2026-09-11T10:19:00.000000001Z anchor\n", "starts after"),
    ],
)
def test_missing_late_or_incomplete_prefix_proof_refuses_before_window_read(
    body: bytes, message: str
) -> None:
    host = LogHost()
    host.prefix["api-a"] = body
    with pytest.raises(BootstrapError, match=message):
        host.collect()
    assert all(kind == "prefix" for _name, kind, _limit in host.queries), (
        "invalid prefix proof must stop before reading the quiet window"
    )


@pytest.mark.parametrize("failure", ["rotation", "truncate", "replace", "partial"])
def test_rotation_or_truncation_during_the_window_read_refuses(failure: str) -> None:
    host = LogHost()
    if failure == "rotation":
        host.after_prefix["api-a"] = _line(NOW, "new current log")
    elif failure == "truncate":
        host.after_prefix["api-a"] = b""
    elif failure == "partial":
        host.after_prefix["api-a"] = host.prefix["api-a"][:-1]
    else:
        host.after_prefix["api-a"] = host.prefix["api-a"].replace(b"anchor", b"other!")
    with pytest.raises(BootstrapError, match="prefix"):
        host.collect()
    assert ("api-a", "window", None) in host.queries


@pytest.mark.parametrize(
    "identity", ["uid", "containerID", "restartCount", "startedAt"]
)
def test_source_reincarnation_refuses_even_when_prefix_bytes_match(
    identity: str,
) -> None:
    host = LogHost()
    if identity == "uid":
        host.after_pods[0]["metadata"]["uid"] = "replacement"
    elif identity == "startedAt":
        host.after_pods[0]["status"]["containerStatuses"][0]["state"]["running"][
            "startedAt"
        ] = (NOW - timedelta(minutes=30)).isoformat()
    else:
        host.after_pods[0]["status"]["containerStatuses"][0][identity] = (
            1 if identity == "restartCount" else "containerd://replacement"
        )
    with pytest.raises(BootstrapError, match="sources changed"):
        host.collect()


def test_a_sidecar_cannot_substitute_for_the_api_container() -> None:
    host = LogHost()
    host.pods[0]["status"]["containerStatuses"][0]["name"] = "sidecar"
    with pytest.raises(BootstrapError, match="stable Ready source"):
        host.collect()
    assert host.queries == []


def test_one_unproved_pod_prevents_acceptance_for_the_whole_source_set() -> None:
    host = LogHost("api-a", "api-b")
    host.prefix["api-b"] = b""
    with pytest.raises(BootstrapError, match="prefix is missing"):
        host.collect()
    assert all(kind == "prefix" for _name, kind, _limit in host.queries), (
        "an unproved pod must block all quiet-window reads"
    )


def test_authentication_timestamps_use_the_exact_fixed_window() -> None:
    host = LogHost()
    host.window["api-a"] = (
        f"2026-09-11T10:18:59.999999999Z {AUTH}\n"
        f"2026-09-11T10:19:00.000000000Z {AUTH}\n"
        f"2026-09-11T10:19:00.000000001Z {AUTH}\n"
        "2026-09-11T10:19:20Z regional cluster gpu-b authenticated with the retiring token\n"
    ).encode()
    result = host.collect()
    assert len(result) == 2
    assert all("10:19:00." in line for line in result), (
        "unexpected authentication timestamp escaped fixed-window filtering"
    )


def test_prefix_read_latency_does_not_move_the_window_start() -> None:
    host = LogHost()
    host.prefix_delay = timedelta(seconds=30)
    host.window["api-a"] = _line(QUIET_START + timedelta(seconds=1), AUTH)
    assert len(host.collect()) == 1
    assert host.clock == NOW + timedelta(seconds=60)


def test_subsecond_window_start_is_filtered_after_a_whole_second_query() -> None:
    host = LogHost()
    host.clock = NOW + timedelta(microseconds=123456)
    host.window["api-a"] = (
        f"2026-09-11T10:19:00.123455999Z {AUTH}\n"
        f"2026-09-11T10:19:00.123456000Z {AUTH}\n"
    ).encode()
    result = host.collect()
    assert len(result) == 1
    assert ".123456000Z" in result[0]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"truncated", "incomplete"),
        (b"missing-timestamp auth\n", "invalid timestamp"),
        (b"2026-09-11T10:19:20\n", "no timestamp"),
        (f"2026-09-11T10:20:00.000000001Z {AUTH}\n".encode(), "future timestamp"),
    ],
)
def test_incomplete_or_unbounded_authentication_timestamps_refuse(
    body: bytes, message: str
) -> None:
    host = LogHost()
    host.window["api-a"] = body
    with pytest.raises(BootstrapError, match=message):
        host.collect()


@pytest.mark.parametrize("prefix", ["[pod/api-a/sidecar] ", "[pod/other/api] ", ""])
def test_native_source_prefix_is_required(prefix: str) -> None:
    host = LogHost()
    host.output_prefix = prefix if prefix else "unprefixed "
    with pytest.raises(BootstrapError, match="unexpected source prefix"):
        host.collect()


def test_a_prefix_response_that_ignores_the_byte_limit_refuses() -> None:
    host = LogHost()
    host.prefix["api-a"] += _line(QUIET_START, "x" * 5000)
    host.ignore_limit = True
    with pytest.raises(BootstrapError, match="byte bound"):
        host.collect()


def test_empty_window_is_accepted_only_with_both_prefix_receipts() -> None:
    host = LogHost()
    host.window["api-a"] = b""
    assert host.collect() == []
    assert host.prefix_calls == {"api-a": 2}


def test_later_prefix_read_failure_remains_opaque_and_refuses() -> None:
    host = LogHost()
    run = host.run
    marker = "synthetic-unstructured-auth-value-24680"

    def failed_read(arguments, **kwargs):
        if (
            any(argument.startswith("--limit-bytes=") for argument in arguments)
            and host.prefix_calls.get("api-a") == 1
        ):
            return subprocess.CompletedProcess(arguments, 1, "", f"Forbidden\n{marker}")
        return run(arguments, **kwargs)

    host.run = failed_read
    with pytest.raises(BootstrapError) as failure:
        host.collect()
    assert marker not in str(failure.value)
    assert "Forbidden" in str(failure.value)


@pytest.mark.parametrize("command_seconds", [3, 4])
def test_all_retention_reads_share_the_acceptance_deadline(
    monkeypatch: pytest.MonkeyPatch, command_seconds: int
) -> None:
    host = LogHost()
    clock = [0.0]
    budgets = []
    expires = []
    run = host.run

    def sleep(seconds):
        clock[0] += seconds

    def timed_run(arguments, *, timeout_seconds):
        deadline = current_deadline()
        assert deadline is not None
        expires.append(deadline.expires)
        budgets.append(remaining_timeout(timeout_seconds))
        result = run(arguments, timeout_seconds=timeout_seconds)
        clock[0] += command_seconds
        return result

    def read_logs(site, cluster_id, *, since_seconds):
        assert site is host.site
        assert cluster_id == "gpu-a"
        assert since_seconds == 60
        return host.collect()

    monkeypatch.setattr("gpu_fault.admin.deadlines.time.monotonic", lambda: clock[0])
    host.run = timed_run
    arguments = {
        "quiet_seconds": 60,
        "timeout_seconds": 136,
        "read_logs": read_logs,
        "sleep": sleep,
        "monotonic": lambda: clock[0],
        "run": host.run,
        "now": lambda: QUIET_START,
    }
    if command_seconds == 4:
        with pytest.raises(DeploymentDeadlineExceeded, match="deadline"):
            module.wait_for_quiet_token_authentications(host.site, "gpu-a", **arguments)
    else:
        result = module.wait_for_quiet_token_authentications(
            host.site, "gpu-a", **arguments
        )
        assert result == {"quiet_seconds": 60, "waited_seconds": 117.0}
    assert budgets == [
        min(120, 136 - command_seconds * index - (60 if index >= 4 else 0))
        for index in range(19)
    ]
    assert expires == [136.0] * 19
