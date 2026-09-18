from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from typing import Any
from urllib import request

import boto3
import pytest

from scripts.e2e.regional.probes.notify008_protocol import Target

JOB_UID = "unit-job-uid"


def forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError(
        "resource-only tests must not perform external or host actions"
    )


@pytest.fixture(autouse=True)
def resource_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import psycopg
    except ImportError:
        pass
    else:
        monkeypatch.setattr(psycopg, "connect", forbidden)
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill", "killpg", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for name in ("client", "resource", "Session"):
        monkeypatch.setattr(boto3, name, forbidden)


@pytest.fixture(name="resource_target")
def resource_target_fixture() -> Target:
    return Target(
        run_id="notify008-0123456789abcdef",
        cluster_id="unit-cpu-cluster",
        region="unit-region",
        release_id="unit-release",
        node="unit-cpu-node",
        node_uid="unit-node-uid",
        source_pod="unit-control-plane",
        source_pod_uid="unit-source-pod-uid",
        deployment_uid="unit-deployment-uid",
        deployment_generation=3,
        runtime_image="registry.invalid/control-plane@sha256:" + "a" * 64,
        postgres_image="registry.invalid/postgres@sha256:" + "b" * 64,
        runtime_version="unit-runtime-v1",
        runtime_module_digest="c" * 64,
        seconds=450,
    )


@pytest.fixture(name="cpu_node")
def cpu_node_fixture(resource_target: Target) -> dict[str, Any]:
    return {
        "metadata": {
            "name": resource_target.node,
            "uid": resource_target.node_uid,
            "labels": {
                "kubernetes.io/os": "linux",
                "node.kubernetes.io/instance-type": "c7i.4xlarge",
            },
        },
        "status": {
            "capacity": {"cpu": "16", "memory": "32Gi"},
            "allocatable": {"cpu": "15500m", "memory": "30Gi"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def admitted_pod(
    expected: dict[str, Any], target: Target, *, gated: bool
) -> dict[str, Any]:
    spec = json.loads(json.dumps(expected))
    pod = {
        "metadata": {
            "namespace": target.run_id,
            "uid": "unit-probe-pod-uid",
            "labels": {
                "gpu-fault.io/acceptance-run": target.run_id,
                "gpu-fault.io/acceptance-case": "GF-REGIONAL-NOTIFY-008",
            },
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": "notify008",
                    "uid": JOB_UID,
                    "controller": True,
                }
            ],
        },
        "spec": spec,
        "status": {"phase": "Pending"},
    }
    if not gated:
        spec["schedulingGates"] = []
        spec["nodeName"] = target.node
        pod["status"] = {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": name,
                    "ready": True,
                    "restartCount": 0,
                    "state": {"running": {"startedAt": "2026-09-12T00:00:00Z"}},
                    "imageID": image,
                }
                for name, image in (
                    ("runtime", target.runtime_image),
                    ("database", target.postgres_image),
                )
            ],
        }
    return pod
