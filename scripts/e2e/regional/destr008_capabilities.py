"""Complete deployed-population proof for nonactivating replacement commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from scripts.e2e.regional.guardrail_audit_evidence import complete_pod_population
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    component_python,
)

# (plane, Deployment name, app label). The ingress role's manifest FILE is
# ``gpu-fault-api-ha-ingress.yaml`` but the Deployment it declares is named
# ``gpu-fault-api-ha``; the live preflight refused every DESTR-008 attempt with
# ``NotFound`` while this tuple named the file instead of the object.
TARGETS = (
    ("cpu", "gpu-fault-api-ha", "gpu-fault-api-ha"),
    ("gpu", "gpu-fault-cluster-executor", "gpu-fault-cluster-executor"),
)
PROBE_SOURCE = Path(__file__).with_name("probes") / "destr008_inhibition_probe.py"
MAX_SOURCE_BYTES = 65536
SOURCE_LOADER = (
    "import hashlib,sys\n"
    "source=sys.stdin.buffer.read(65537)\n"
    "expected=sys.argv.pop(1)\n"
    "actual=hashlib.sha256(source).hexdigest()\n"
    "if len(source)>65536 or actual!=expected:\n"
    " raise SystemExit('pinned inhibition probe source differs')\n"
    "exec(compile(source,'<inhibition-capability>','exec'),"
    "{'__name__':'__main__','_PROBE_SHA256':actual})\n"
)
CHECKS = {
    "cpu": {
        "request_field",
        "literal_true_only",
        "opt_in_only",
        "propagation_and_claim_version",
        "claim_protocol_argument",
    },
    "gpu": {
        "coordinator_version",
        "confirmation_version",
        "allocate_keyword",
        "reserve_keyword",
        "present_values_refused",
        "absence_allowed",
    },
}
COMMON_CHECKS = {"protocol", "capability_version", "inspection_complete"}


def probe_source() -> tuple[str, str]:
    fd = os.open(PROBE_SOURCE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        information = os.fstat(fd)
        if not stat.S_ISREG(information.st_mode):
            raise RegionalFixtureError("inhibition probe source is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            body = source.read(MAX_SOURCE_BYTES + 1)
        if not body or len(body) > MAX_SOURCE_BYTES or len(body) != information.st_size:
            raise RegionalFixtureError("inhibition probe source has an invalid size")
        after = os.fstat(fd)
        current = PROBE_SOURCE.stat(follow_symlinks=False)
        if (
            information.st_dev,
            information.st_ino,
            information.st_size,
            information.st_mtime_ns,
            information.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (current.st_dev, current.st_ino) != (
            information.st_dev,
            information.st_ino,
        ):
            raise RegionalFixtureError("inhibition probe source changed while reading")
    finally:
        os.close(fd)
    return body.decode("utf-8"), hashlib.sha256(body).hexdigest()


def snapshot(
    regional: RegionalLiveFixture, plane: str, name: str, label: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    deployment = json.loads(
        regional.kubectl(plane, "get", "deployment", name, "-o", "json")
    )
    inventory = json.loads(
        regional.kubectl(plane, "get", "pods", "-l", f"app={label}", "-o", "json")
    )
    ready = complete_pod_population(deployment, inventory)
    if (
        deployment.get("kind") != "Deployment"
        or deployment["metadata"].get("name") != name
        or deployment["metadata"].get("namespace") != regional.settings.namespace
    ):
        raise RegionalFixtureError("inhibition guard deployment identity differs")
    containers = deployment["spec"]["template"]["spec"]["containers"]
    if len(containers) != 1 or not containers[0].get("name"):
        raise RegionalFixtureError(
            "inhibition guard requires one bound runtime container"
        )
    container = containers[0]["name"]
    image = containers[0].get("image")
    if not isinstance(image, str) or not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        raise RegionalFixtureError("inhibition guard runtime image is mutable")
    replica_sets = json.loads(
        regional.kubectl(
            plane, "get", "replicasets", "-l", f"app={label}", "-o", "json"
        )
    )
    owned: dict[str, str] = {}
    for replica in replica_sets["items"]:
        metadata = replica.get("metadata") or {}
        owners = [
            item
            for item in metadata.get("ownerReferences", [])
            if item.get("controller") is True
        ]
        if len(owners) == 1 and (
            owners[0].get("kind") == "Deployment"
            and owners[0].get("uid") == deployment["metadata"]["uid"]
            and owners[0].get("name") == name
            and metadata.get("namespace") == regional.settings.namespace
        ):
            uid = metadata.get("uid")
            if (
                not isinstance(uid, str)
                or not uid
                or uid in owned
                or not isinstance(metadata.get("name"), str)
                or not metadata["name"]
            ):
                raise RegionalFixtureError(
                    "inhibition guard ReplicaSet identities are ambiguous"
                )
            owned[uid] = metadata["name"]
    stable = []
    for pod in inventory["items"]:
        metadata = pod["metadata"]
        owners = [
            item
            for item in metadata.get("ownerReferences", [])
            if item.get("controller") is True
        ]
        spec = pod["spec"]
        statuses = pod["status"].get("containerStatuses") or []
        if (
            metadata.get("namespace") != regional.settings.namespace
            or len(owners) != 1
            or owners[0].get("kind") != "ReplicaSet"
            or owners[0].get("uid") not in owned
            or owned.get(owners[0].get("uid")) != owners[0].get("name")
            or len(spec.get("containers") or []) != 1
            or spec["containers"][0].get("name") != container
            or spec["containers"][0].get("image") != image
            or len(statuses) != 1
            or statuses[0].get("name") != container
            or not statuses[0].get("containerID")
            or type(statuses[0].get("restartCount")) is not int
            or statuses[0]["restartCount"] < 0
            or str(statuses[0].get("imageID", "")).rsplit("sha256:", 1)[-1]
            != image.rsplit("sha256:", 1)[-1]
        ):
            raise RegionalFixtureError(
                "inhibition guard Pod ownership or image is unproven"
            )
        stable.append(
            {
                "uid": metadata["uid"],
                "name": metadata["name"],
                "container_id": statuses[0]["containerID"],
                "restarts": statuses[0].get("restartCount"),
            }
        )
    return deployment, ready, sorted(stable, key=lambda item: item["uid"]), container


def read_capabilities(regional: RegionalLiveFixture) -> dict[str, Any]:
    source, source_sha256 = probe_source()
    result: dict[str, Any] = {
        "supported": False,
        "populations": [],
        "probe_sha256": source_sha256,
    }
    for plane, name, label in TARGETS:
        before, pods, identities, container = snapshot(regional, plane, name, label)
        probes = []
        for pod in pods:
            value = json.loads(
                regional.kubectl(
                    plane,
                    "exec",
                    "-i",
                    pod["name"],
                    "-c",
                    container,
                    "--",
                    component_python(plane),
                    "-I",
                    "-B",
                    "-c",
                    SOURCE_LOADER,
                    source_sha256,
                    plane,
                    input_text=source,
                    timeout=30,
                )
            )
            if (
                not isinstance(value, dict)
                or value.get("supported") is not True
                or value.get("probe_sha256") != source_sha256
                or value.get("capability")
                != "synthetic-replacement-activation-inhibition"
                or type(value.get("capability_version")) is not int
                or value["capability_version"] != 1
                or value.get("component") != ("api" if plane == "cpu" else "executor")
                or value.get("marker") != "activation_forbidden"
                or type(value.get("minimum_executor_protocol_version")) is not int
                or value["minimum_executor_protocol_version"] != 4
                or type(value.get("executor_protocol_version")) is not int
                or value["executor_protocol_version"] < 4
                or not isinstance(value.get("checks"), dict)
                or not (CHECKS[plane] | COMMON_CHECKS) <= value["checks"].keys()
                or any(check is not True for check in value["checks"].values())
            ):
                raise RegionalFixtureError(
                    "deployed runtime cannot enforce inhibited replacement"
                )
            probes.append({"pod_uid": pod["uid"], **value})
        after, _, current, _ = snapshot(regional, plane, name, label)
        if (
            after["metadata"]["uid"] != before["metadata"]["uid"]
            or after["metadata"]["generation"] != before["metadata"]["generation"]
            or current != identities
        ):
            raise RegionalFixtureError(
                "inhibition guard population changed during verification"
            )
        result["populations"].append(
            {
                "plane": plane,
                "deployment_uid": before["metadata"]["uid"],
                "generation": before["metadata"]["generation"],
                "pods": identities,
                "probes": probes,
            }
        )
    result["supported"] = True
    return result
