"""Collect a release-bound deployed protocol subproof; never a custody verdict."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.e2e.regional.auth015_bindings import (
    BoundSnapshot,
    bind_snapshot,
    objects,
    pod_binding,
    text,
)
from scripts.e2e.regional.auth015_protocol import (
    Auth015ProofError,
    SignatureTarget,
    prove_signatures,
)
from scripts.e2e.regional.auth015_release import (
    Auth015ReleaseInputs,
    VerifiedAuth015Release,
    release_input_identity,
    verify_release_inputs,
)
from scripts.e2e.regional.identity_acceptance_common import ClusterTarget, IdentitySite
from scripts.e2e.regional.regional_live_fixture import component_python
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records

AGENT_PROBE = Path(__file__).with_name("auth015_agent_probe.py")
HTTP_WORKER = Path(__file__).with_name("auth015_http.py")


def agent_probe_script() -> str:
    # Ship the stdlib-only worker over stdin; CPU wheels need no admin package.
    return (
        "import types\n"
        f"source = {HTTP_WORKER.read_text(encoding='utf-8')!r}\n"
        "transport = types.ModuleType('auth015_private_http')\n"
        "transport.AUTH015_HTTP_SOURCE = source\n"
        f"exec(compile(source, {str(HTTP_WORKER)!r}, 'exec'), transport.__dict__)\n"
        f"exec(compile({AGENT_PROBE.read_text(encoding='utf-8')!r}, {str(AGENT_PROBE)!r}, 'exec'), "
        "{'__name__': '__main__', 'AUTH015_HTTP_REQUEST': transport.bounded_request})\n"
    )


def capture_snapshot(
    site: IdentitySite,
    target: ClusterTarget,
    nodes: tuple[str, str],
    release: VerifiedAuth015Release,
    *,
    pod_name: str | None = None,
) -> BoundSnapshot:
    pods = json.loads(
        site.cpu("get", "pod", "-l", "app=gpu-fault-api-ha", "-o", "json")
    )
    ready = sorted(str(item["name"]) for item in ready_pod_records(pods))
    if not ready or pod_name is not None and pod_name not in ready:
        raise Auth015ProofError("AUTH015 bound CPU API Pod is not Ready")
    selected = pod_name or ready[0]
    matches = [
        item
        for item in objects(pods, "Pod")
        if item.get("metadata", {}).get("name") == selected
    ]
    if len(matches) != 1:
        raise Auth015ProofError("AUTH015 CPU API Pod identity is ambiguous")
    if matches[0]["metadata"].get("namespace") != site.namespace:
        raise Auth015ProofError("AUTH015 CPU API Pod namespace differs")
    pod_binding(matches[0], release.runtime_image)
    state_document = json.loads(
        site.cpu("get", "configmap", "gpu-fault-regional-release-state", "-o", "json")
    )
    registrations = [
        item for item in site.registry() if item.get("cluster_id") == target.cluster_id
    ]
    if len(registrations) != 1:
        raise Auth015ProofError("AUTH015 cluster registration is ambiguous")
    output = site.cpu(
        "exec",
        "-i",
        selected,
        "-c",
        "api",
        "--",
        component_python("cpu"),
        "-",
        target.cluster_id,
        *nodes,
        release.release_id,
        # The Pod binds itself to the verified release through these pins; it
        # has no GPU_FAULT_RELEASE_ID to compare (see auth015_agent_probe).
        release.node_wheel_sha256,
        release.node_digest,
        input_text=agent_probe_script(),
        timeout=30,
    )
    raw = {
        "pod": matches[0],
        "namespace": site.namespace,
        "release_metadata": state_document["metadata"],
        "release_state": json.loads(state_document["data"]["state.json"]),
        "agent_snapshot": json.loads(output),
        "nodes": json.loads(site.gpu(target, "get", "nodes", *nodes, "-o", "json")),
        "registration": registrations[0],
    }
    return bind_snapshot(raw, release=release, target=target, nodes=nodes)


def confirm_key_source(site: IdentitySite, expected: dict[str, Any]) -> dict[str, str]:
    metadata = expected["metadata"]
    name = text(metadata.get("name"))
    if (
        expected.get("apiVersion") != "v1"
        or expected.get("kind") != "Secret"
        or expected.get("type") != "Opaque"
        or metadata.get("namespace") != site.namespace
        or metadata.get("deletionTimestamp")
    ):
        raise Auth015ProofError("AUTH015 CPU key source identity is invalid")
    current = json.loads(site.cpu("get", "secret", name, "-o", "json"))
    if (
        any(
            current.get(field) != expected.get(field)
            for field in ("apiVersion", "kind", "type")
        )
        or current["metadata"].get("name") != name
        or current["metadata"].get("namespace") != site.namespace
        or current["metadata"].get("deletionTimestamp")
    ):
        raise Auth015ProofError("AUTH015 CPU key source identity changed")
    for field in ("uid", "resourceVersion"):
        if text(metadata.get(field)) != current["metadata"].get(field):
            raise Auth015ProofError("AUTH015 CPU key source changed")
    if current.get("data") != expected.get("data"):
        raise Auth015ProofError("AUTH015 CPU key source changed")
    return {
        "name": name,
        "namespace": site.namespace,
        "uid": metadata["uid"],
        "resource_version": metadata["resourceVersion"],
    }


def prove_deployed_protocol(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str],
    key_document: dict[str, Any],
    release_inputs: Auth015ReleaseInputs,
    retired_key: str | None = None,
) -> dict[str, Any]:
    try:
        release = verify_release_inputs(release_inputs)
        before = capture_snapshot(site, target, nodes, release)
        key_source = confirm_key_source(site, key_document)
        keys = {
            node: base64.b64decode(key_document["data"][node], validate=True).decode(
                "utf-8"
            )
            for node in nodes
        }
        proof = prove_signatures(
            SignatureTarget(before.agents[nodes[0]], keys[nodes[0]]),
            SignatureTarget(before.agents[nodes[1]], keys[nodes[1]]),
            retired_key=retired_key,
        )
        after = capture_snapshot(
            site, target, nodes, release, pod_name=before.binding["cpu_pod"]["name"]
        )
        if before.binding != after.binding:
            raise Auth015ProofError(
                "AUTH015 release, Pod, Node or Agent identity drifted"
            )
        if release.input_identity != release_input_identity(release_inputs):
            raise Auth015ProofError(
                "AUTH015 signed-release inputs changed during the challenge"
            )
        confirm_key_source(site, key_document)
        return {
            "verdict": "PASS",
            "scope": "deployed_command_and_result_signature_rejection",
            "identity": before.binding,
            "protocol": proof,
            "key_source": key_source,
            "key_sha256": {
                node: hashlib.sha256(value.encode()).hexdigest()
                for node, value in keys.items()
            },
            "installation_custody_proved": False,
            "supplied_retired_key_denied": retired_key is not None,
            "rotated_key_activation_proved": False,
        }
    except Auth015ProofError:
        raise
    except Exception:
        raise Auth015ProofError(
            "AUTH015 deployed protocol proof could not be completed"
        ) from None
