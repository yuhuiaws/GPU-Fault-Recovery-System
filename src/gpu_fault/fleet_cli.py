from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error as urllib_error
from urllib.parse import quote
from urllib import request as urllib_request


def _nodes(value: str) -> list[str]:
    path = Path(value)
    raw = (
        path.read_text(encoding="utf-8").splitlines()
        if path.is_file()
        else value.split(",")
    )
    nodes = [item.strip() for item in raw if item.strip()]
    if not nodes:
        raise argparse.ArgumentTypeError("at least one node is required")
    if len(nodes) != len(set(nodes)):
        raise argparse.ArgumentTypeError("nodes must be unique")
    return nodes


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Coordinate GPU fault agent fleet rollout"
    )
    result.add_argument(
        "--control-plane-url",
        required=True,
    )
    commands = result.add_subparsers(dest="command", required=True)

    agents = commands.add_parser("agents")
    agents.add_argument("--cluster-id")

    readiness = commands.add_parser("readiness")
    readiness.add_argument("--cluster-id", required=True)
    readiness.add_argument("--nodes", required=True, type=_nodes)

    create = commands.add_parser("create-deployment")
    create.add_argument("--execution-token", required=True)
    create.add_argument("--deployment-id")
    create.add_argument("--cluster-id", required=True)
    create.add_argument("--nodes", required=True, type=_nodes)
    create.add_argument("--agent-version", required=True)
    create.add_argument("--artifact-sha256", required=True)
    create.add_argument("--compatibility-digest")
    create.add_argument("--bundle-sha256")
    create.add_argument("--template-sha256")
    create.add_argument("--policy-version", required=True)
    create.add_argument("--runtime-profile-version", required=True)
    create.add_argument("--config-digest", required=True)
    create.add_argument("--max-unavailable", type=int, default=1)

    deployment = commands.add_parser("deployment")
    deployment.add_argument("--deployment-id", required=True)

    wave = commands.add_parser("next-wave")
    wave.add_argument("--execution-token", required=True)
    wave.add_argument("--deployment-id", required=True)

    run = commands.add_parser("run-deployment")
    run.add_argument("--execution-token", required=True)
    run.add_argument("--deployment-id", required=True)
    run.add_argument(
        "--transport-command",
        required=True,
        help=(
            "Command invoked once per node; supports {node_id}, "
            "{cluster_id}, {deployment_id}, {agent_version}, and "
            "{artifact_sha256}"
        ),
    )
    run.add_argument(
        "--poll-interval-seconds",
        type=_positive_int,
        default=10,
    )
    run.add_argument(
        "--wave-timeout-seconds",
        type=_positive_int,
        default=900,
    )
    return result


def _request(
    base_url: str,
    path: str,
    *,
    payload: dict | None = None,
    token: str | None = None,
):
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-GPU-Fault-Execution-Token"] = token
    request = urllib_request.Request(
        base_url.rstrip("/") + path,
        data=(
            json.dumps(payload, separators=(",", ":")).encode()
            if payload is not None
            else None
        ),
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(
            f"control plane rejected request ({exc.code}): {detail}"
        ) from exc


def _deployment_path(deployment_id: str) -> str:
    return "/v1/fleet/deployments/" + quote(deployment_id, safe="")


def _node_status(
    base_url: str,
    token: str,
    deployment_id: str,
    node_id: str,
    status: str,
    reason: str | None = None,
    *,
    requester=_request,
):
    return requester(
        base_url,
        _deployment_path(deployment_id) + "/nodes/" + quote(node_id, safe=""),
        payload={"status": status, "reason": reason},
        token=token,
    )


def _transport(
    command_template: str,
    deployment: dict,
    node_id: str,
    *,
    runner=subprocess.run,
) -> tuple[str, str | None]:
    values = {
        "node_id": node_id,
        "cluster_id": deployment["cluster_id"],
        "deployment_id": deployment["deployment_id"],
        "agent_version": deployment["desired_agent_version"],
        "artifact_sha256": deployment["desired_artifact_sha256"],
    }
    try:
        command = [token.format_map(values) for token in shlex.split(command_template)]
    except (KeyError, ValueError) as exc:
        return node_id, f"invalid transport command template: {exc}"
    if not command:
        return node_id, "transport command is empty"
    environment = {
        **os.environ,
        **{
            "GPU_FAULT_FLEET_" + key.upper(): str(value)
            for key, value in values.items()
        },
    }
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as exc:
        return node_id, f"{type(exc).__name__}: {exc}"
    if completed.returncode == 0:
        return node_id, None
    output = (completed.stderr or completed.stdout or "").strip()
    if len(output) > 1000:
        output = output[-1000:]
    suffix = f": {output}" if output else ""
    return (
        node_id,
        f"transport exited with status {completed.returncode}{suffix}",
    )


def run_deployment(
    base_url: str,
    token: str,
    deployment_id: str,
    command_template: str,
    *,
    poll_interval_seconds: int,
    wave_timeout_seconds: int,
    requester=_request,
    runner=subprocess.run,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> dict:
    path = _deployment_path(deployment_id)
    while True:
        deployment = requester(base_url, path)
        if deployment["status"] == "FAILED":
            raise SystemExit("fleet deployment is FAILED")
        if deployment["status"] == "SUCCEEDED":
            break
        lease = requester(
            base_url,
            path + "/next-wave",
            payload={},
            token=token,
        )
        node_ids = lease["node_ids"]
        with ThreadPoolExecutor(max_workers=max(1, len(node_ids))) as pool:
            results = list(
                pool.map(
                    lambda node_id: _transport(
                        command_template,
                        deployment,
                        node_id,
                        runner=runner,
                    ),
                    node_ids,
                )
            )
        failures = {node_id: error for node_id, error in results if error is not None}
        for node_id, error in failures.items():
            _node_status(
                base_url,
                token,
                deployment_id,
                node_id,
                "FAILED",
                error,
                requester=requester,
            )
        if failures:
            raise SystemExit(
                "fleet transport failed: "
                + "; ".join(
                    f"{node_id}: {error}" for node_id, error in sorted(failures.items())
                )
            )

        deadline = monotonic() + wave_timeout_seconds
        while True:
            deployment = requester(base_url, path)
            by_node = {item["node_id"]: item for item in deployment["nodes"]}
            failed = [
                node_id
                for node_id in node_ids
                if by_node[node_id]["status"] == "FAILED"
            ]
            if failed:
                raise SystemExit(
                    "fleet heartbeat reconciliation failed: " + ",".join(failed)
                )
            if all(by_node[node_id]["status"] == "READY" for node_id in node_ids):
                break
            if monotonic() >= deadline:
                waiting = [
                    node_id
                    for node_id in node_ids
                    if by_node[node_id]["status"] != "READY"
                ]
                for node_id in waiting:
                    _node_status(
                        base_url,
                        token,
                        deployment_id,
                        node_id,
                        "FAILED",
                        "target heartbeat was not observed before timeout",
                        requester=requester,
                    )
                raise SystemExit(
                    "timed out waiting for target heartbeat: " + ",".join(waiting)
                )
            sleep(poll_interval_seconds)
        if deployment["status"] == "SUCCEEDED":
            break

    readiness = requester(
        base_url,
        "/v1/fleet/readiness",
        payload={
            "cluster_id": deployment["cluster_id"],
            "node_ids": [item["node_id"] for item in deployment["nodes"]],
        },
    )
    if not readiness["ready"]:
        raise SystemExit(
            "deployment reached target identity but fleet readiness "
            "gate rejected the nodes: " + json.dumps(readiness, sort_keys=True)
        )
    return {"deployment": deployment, "readiness": readiness}


def main() -> None:
    args = parser().parse_args()
    base_url = args.control_plane_url
    if args.command == "agents":
        query = "?cluster_id=" + quote(args.cluster_id) if args.cluster_id else ""
        result = _request(base_url, "/v1/fleet/agents" + query)
    elif args.command == "readiness":
        result = _request(
            base_url,
            "/v1/fleet/readiness",
            payload={
                "cluster_id": args.cluster_id,
                "node_ids": args.nodes,
            },
        )
    elif args.command == "create-deployment":
        result = _request(
            base_url,
            "/v1/fleet/deployments",
            token=args.execution_token,
            payload={
                "deployment_id": args.deployment_id,
                "cluster_id": args.cluster_id,
                "node_ids": args.nodes,
                "desired_agent_version": args.agent_version,
                "desired_artifact_sha256": args.artifact_sha256,
                "desired_compatibility_digest": args.compatibility_digest,
                "desired_bundle_sha256": args.bundle_sha256,
                "desired_template_sha256": args.template_sha256,
                "desired_policy_version": args.policy_version,
                "desired_runtime_profile_version": (args.runtime_profile_version),
                "desired_config_digest": args.config_digest,
                "max_unavailable": args.max_unavailable,
            },
        )
    elif args.command == "deployment":
        result = _request(
            base_url,
            _deployment_path(args.deployment_id),
        )
    elif args.command == "next-wave":
        result = _request(
            base_url,
            _deployment_path(args.deployment_id) + "/next-wave",
            payload={},
            token=args.execution_token,
        )
    else:
        result = run_deployment(
            base_url,
            args.execution_token,
            args.deployment_id,
            args.transport_command,
            poll_interval_seconds=args.poll_interval_seconds,
            wave_timeout_seconds=args.wave_timeout_seconds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
