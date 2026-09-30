"""Read the running CPU API, without constructing an ApplicationContext or Store."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any, cast
from urllib.parse import quote

from gpu_fault.fleet import AgentRecord, validate_node_identifier

MAX_RESPONSE_BYTES = 512 * 1024
# This probe is exec'd into the CPU API Pod, which exports the required agent
# pins (from gpu-fault-release-metadata) but not GPU_FAULT_RELEASE_ID: the CPU
# role environment never renders it and gpu_fault.app.factory defaults it to
# "local". The release binding is therefore the pins the caller derived from the
# verified release; the release ID is carried through as evidence only.
RELEASE_PIN_ENVIRONMENT = (
    "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256",
    "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST",
)
VERSION_FIELDS = (
    "module_digest",
    "deployment_mode",
    "required_agent_artifact_sha256",
    "required_agent_compatibility_digest",
    "required_agent_protocol_version",
    "required_agent_config_digest",
    "required_runtime_profile_version",
    "required_node_action_key_version",
)


def http_exchange(**arguments: Any) -> tuple[int, bytes]:
    exchange = globals().get("AUTH015_HTTP_REQUEST")
    if exchange is None:
        from scripts.e2e.regional.auth015_http import bounded_request

        exchange = bounded_request
    return cast(tuple[int, bytes], exchange(**arguments))


def read_snapshot(
    cluster_id: str,
    nodes: tuple[str, str],
    *,
    expected_release_id: str,
    expected_pins: tuple[str, str],
) -> dict[str, Any]:
    if not cluster_id or not expected_release_id or len(set(nodes)) != 2:
        raise ValueError("AUTH015 snapshot identity is invalid")
    for name, expected in zip(RELEASE_PIN_ENVIRONMENT, expected_pins, strict=True):
        if not expected or os.environ.get(name, "").strip() != expected:
            raise ValueError(f"AUTH015 release pin {name} does not bind this Pod")
    for node in nodes:
        validate_node_identifier(node)
    token = os.environ["GPU_FAULT_EXECUTION_TOKEN"]
    if not token:
        raise ValueError("AUTH015 snapshot authentication is unavailable")

    def query(path: str) -> dict[str, Any]:
        status, body = http_exchange(
            url="http://127.0.0.1:8080" + path,
            headers={"X-GPU-Fault-Execution-Token": token},
            timeout=5,
            max_response_bytes=MAX_RESPONSE_BYTES,
        )
        if status != 200 or len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("AUTH015 read-only API response is incomplete")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("AUTH015 read-only API response is not an object")
        return value

    version = query("/v1/version")
    records = {}
    for node in nodes:
        value = query(
            f"/v1/fleet/agents/{quote(cluster_id, safe='')}/{quote(node, safe='')}"
        )
        record = AgentRecord.model_validate_json(json.dumps(value), strict=True)
        if record.cluster_id != cluster_id or record.node_id != node:
            raise ValueError("AUTH015 API returned a foreign agent")
        records[node] = record.model_dump(mode="json")
    return {
        "release_id": expected_release_id,
        "version": {field: version.get(field) for field in VERSION_FIELDS},
        "agents": records,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    try:
        if len(values) != 6:
            raise ValueError(
                "AUTH015 snapshot needs a cluster, two nodes, release and two pins"
            )
        result = read_snapshot(
            values[0],
            (values[1], values[2]),
            expected_release_id=values[3],
            expected_pins=(values[4], values[5]),
        )
    except Exception:
        # Neither API bodies nor credential-bearing transport errors may escape.
        print(json.dumps({"error": "AUTH015 read-only agent snapshot failed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
