"""Single-load node rendering and bounded, read-only host preflight execution."""

from __future__ import annotations

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

import argparse
import hashlib
import json
import os
import stat
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol, Sequence, TypeVar
from uuid import uuid4

from gpu_fault.admin.deploy_limits import DEPLOY_CONCURRENCY
from gpu_fault.admin.execution import cleanup_deadline
from gpu_fault.node_installer_rendering import (
    InstallerIdentity,
    InstallerNode,
    load_installer_template,
    manifest_object,
    manifest_objects,
    preflight_job,
    render_installer_job,
)
from gpu_fault_release import repository_root

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_ADMISSION_BATCH_NODES = 4
MAX_ADMISSION_BATCH_BYTES = 1024 * 1024
_T = TypeVar("_T")


@dataclass(frozen=True)
class NodeScope:
    cluster_id: str
    context: str
    namespace: str
    node_action_keys_secret: str
    connection_secret: str


@dataclass(frozen=True)
class PreparedNode:
    node: InstallerNode
    install_manifest: Path
    preflight_manifest: Path
    job_name: str


class PreflightRunner(Protocol):
    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]: ...

    def run(
        self,
        args: list[str],
        *,
        capture: bool = False,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str: ...


def read_private_snapshot(path: Path, scope: NodeScope) -> dict[str, InstallerNode]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as handle:
        file_stat = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.geteuid()
            or stat.S_IMODE(file_stat.st_mode) not in {0o400, 0o600}
            or file_stat.st_size > MAX_SNAPSHOT_BYTES
        ):
            raise ValueError("node snapshot must be a bounded private owned file")
        document = manifest_object(json.load(handle), "node snapshot")
    expected: dict[str, object] = {
        "schema_version": 1,
        "cluster_id": scope.cluster_id,
        "context": scope.context,
        "namespace": scope.namespace,
        "node_action_keys_secret": scope.node_action_keys_secret,
        "connection_secret": scope.connection_secret,
        "references_validated": True,
    }
    if type(document.get("schema_version")) is not int or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("node snapshot scope or reference proof differs")
    if set(document) != {*expected, "nodes"}:
        raise ValueError("node snapshot has undeclared fields")
    nodes: dict[str, InstallerNode] = {}
    uids: set[str] = set()
    for name, raw in manifest_object(document.get("nodes"), "snapshot nodes").items():
        node = manifest_object(raw, "snapshot node")
        metadata = manifest_object(node.get("metadata"), "node metadata")
        if metadata.get("name", name) != name:
            raise ValueError("node snapshot identity differs from its key")
        labels = manifest_object(metadata.get("labels"), "node labels")
        status = manifest_object(node.get("status"), "node status")
        addresses = [
            item.get("address")
            for item in manifest_objects(status.get("addresses"), "node addresses")
            if item.get("type") == "InternalIP"
        ]
        values = (metadata.get("uid"), labels.get("node.kubernetes.io/instance-type"))
        if len(addresses) != 1 or not all(
            isinstance(value, str) and value for value in (*values, *addresses)
        ):
            raise ValueError("node snapshot has incomplete identity or addressing")
        nodes[name] = InstallerNode(
            name, str(values[0]), str(addresses[0]), str(values[1])
        )
        if nodes[name].uid in uids:
            raise ValueError("node snapshot contains duplicate node UIDs")
        uids.add(nodes[name].uid)
    if not nodes:
        raise ValueError("node snapshot is empty")
    return nodes


def _write_private_json(path: Path, document: object) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(document, handle, sort_keys=True)
        handle.write("\n")


def prepare_node_batch(
    inputs: Path,
    template_path: Path,
    output: Path,
    scope: NodeScope,
    identity: InstallerIdentity,
) -> list[PreparedNode]:
    metadata = output.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("node batch output must be a private owned directory")
    if (
        scope.namespace != identity.namespace
        or scope.node_action_keys_secret != identity.node_action_keys_secret
    ):
        raise ValueError("node batch identity differs from its scope")
    nodes = read_private_snapshot(inputs, scope)
    template = load_installer_template(
        template_path.read_bytes(),
        expected_sha256=identity.template_content_sha256,
        origin="node batch template",
        identity=identity,
    )
    run_id = uuid4().hex
    prepared: list[PreparedNode] = []
    for name, node in sorted(nodes.items()):
        key = hashlib.sha256(f"{name}\0{node.uid}".encode()).hexdigest()[:24]
        install_path = output / f"install-{key}.json"
        preflight_path = output / f"preflight-{key}.json"
        install = render_installer_job(
            template, node, identity, f"gpu-fault-install-{key}"
        )
        preflight = preflight_job(template, node, identity, run_id)
        _write_private_json(install_path, install)
        _write_private_json(preflight_path, preflight)
        prepared.append(
            PreparedNode(
                node,
                install_path,
                preflight_path,
                str(manifest_object(preflight["metadata"], "Job metadata")["name"]),
            )
        )
    return prepared


def _admission_payload(documents: list[dict[str, object]]) -> str:
    return json.dumps(
        {"apiVersion": "v1", "kind": "List", "items": documents},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _admission_batches(
    nodes: Sequence[PreparedNode], node_limit: int
) -> Iterator[tuple[str, str]]:
    names: list[str] = []
    documents: list[dict[str, object]] = []
    payload = ""
    for node in nodes:
        candidates = []
        for path in (node.install_manifest, node.preflight_manifest):
            with path.open("rb") as handle:
                raw = handle.read(MAX_ADMISSION_BATCH_BYTES + 1)
            if len(raw) > MAX_ADMISSION_BATCH_BYTES:
                raise ValueError("node admission manifest exceeds its size budget")
            candidates.append(manifest_object(json.loads(raw), "node admission Job"))
        combined = _admission_payload([*documents, *candidates])
        if names and (
            len(names) == node_limit or len(combined) > MAX_ADMISSION_BATCH_BYTES
        ):
            yield ", ".join(names), payload
            names, documents = [], []
            combined = _admission_payload(candidates)
        # ASCII serialization makes the character limit an exact byte limit.
        if len(combined) > MAX_ADMISSION_BATCH_BYTES:
            raise ValueError("node admission manifests exceed their batch size budget")
        names.append(node.node.name)
        documents.extend(candidates)
        payload = combined
    if names:
        yield ", ".join(names), payload


def _bounded_preflights(
    items: Iterable[tuple[str, _T]], action: Callable[[_T], None], workers: int
) -> None:
    remaining = iter(items)
    failures: list[Exception] = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="node-preflight"
    ) as pool:
        active: dict[Future[None], str] = {}
        while True:
            while not failures and len(active) < workers:
                item = next(remaining, None)
                if item is None:
                    break
                label, value = item
                active[pool.submit(copy_context().run, action, value)] = label
            if not active:
                break
            done, _pending = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                label = active.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    exc.add_note(f"node preflight failed for {label}")
                    failures.append(exc)
    if failures:
        raise failures[0]


def run_node_preflights(
    nodes: Sequence[PreparedNode],
    scope: NodeScope,
    identity: InstallerIdentity,
    runner: PreflightRunner,
    *,
    workers: int = 8,
) -> None:
    if not 1 <= workers <= DEPLOY_CONCURRENCY.read_only_checks:
        raise ValueError("node preflight concurrency is outside its safety budget")
    kubectl = ["kubectl", "--context", scope.context, "-n", scope.namespace]

    def dry_run(payload: str) -> None:
        runner.run(
            [*kubectl, "apply", "--dry-run=server", "-f", "-"],
            input_text=payload,
            capture=True,
            timeout_seconds=60,
        )

    def host(node: PreparedNode) -> None:
        failure: Exception | None = None
        try:
            runner.run(
                [*kubectl, "create", "-f", str(node.preflight_manifest)],
                capture=True,
                timeout_seconds=60,
            )
            runner.run(
                [
                    "bash",
                    str(
                        repository_root()
                        / "deploy/control-plane/tools/wait-for-kubernetes-job.sh"
                    ),
                    str(identity.deadline_seconds + 60),
                    node.job_name,
                    *kubectl,
                ],
                capture=True,
                timeout_seconds=identity.deadline_seconds + 90,
            )
        except Exception as exc:
            failure = exc
            try:
                with cleanup_deadline("node preflight diagnostics", 30):
                    logs = runner.run(
                        [
                            *kubectl,
                            "logs",
                            f"job/{node.job_name}",
                            "--all-containers",
                            "--tail=80",
                            "--limit-bytes=8192",
                        ],
                        capture=True,
                        timeout_seconds=20,
                    )
                    if logs:
                        failure.add_note("host preflight: " + diagnostic_text(logs))
            except Exception:
                pass
            raise
        finally:
            try:
                with cleanup_deadline("node preflight cleanup"):
                    document = probe_resource(
                        runner,
                        ["kubectl", "--context", scope.context],
                        ResourceRef("job", "Job", node.job_name, scope.namespace),
                    ).require_readable()
                    if document is not None:
                        metadata = manifest_object(
                            document.get("metadata"), "preflight Job metadata"
                        )
                        labels = manifest_object(
                            metadata.get("labels"), "preflight Job labels"
                        )
                        annotations = manifest_object(
                            metadata.get("annotations"), "preflight Job annotations"
                        )
                        if (
                            labels.get("gpu-fault.io/node-preflight") != "true"
                            or labels.get("gpu-fault.io/node-uid") != node.node.uid
                            or annotations.get("gpu-fault.io/installer-artifact-sha256")
                            != identity.artifact_sha256
                            or annotations.get("gpu-fault.io/installer-template-sha256")
                            != identity.template_sha256
                        ):
                            raise ValueError(
                                "refusing cleanup of a foreign preflight Job"
                            )
                        # A create ACK may have been lost. Read the unique,
                        # run-bound Job and delete only the UID we just proved.
                        uri = f"/apis/batch/v1/namespaces/{scope.namespace}/jobs/{node.job_name}"
                        runner.run(
                            [*kubectl, "delete", "--raw", uri, "-f", "-"],
                            input_text=json.dumps(
                                {
                                    "apiVersion": "v1",
                                    "kind": "DeleteOptions",
                                    "preconditions": {"uid": metadata["uid"]},
                                    "propagationPolicy": "Background",
                                }
                            ),
                            capture=True,
                            timeout_seconds=20,
                        )
                        runner.run(
                            [
                                *kubectl,
                                "wait",
                                "--for=delete",
                                f"job/{node.job_name}",
                                "--timeout=45s",
                            ],
                            capture=True,
                            timeout_seconds=50,
                        )
            except Exception as cleanup:
                if failure is None:
                    raise
                failure.add_note(
                    f"preflight Job cleanup also failed: {diagnostic_text(str(cleanup))}"
                )

    # Every manifest passes admission before any host preflight Job is created.
    # Bound CLI batches independently of the host/node window. The runner still
    # enforces the shared API ceiling and start pacing across all clusters.
    _bounded_preflights(
        _admission_batches(nodes, MAX_ADMISSION_BATCH_NODES),
        dry_run,
        DEPLOY_CONCURRENCY.read_only_checks,
    )
    _bounded_preflights(((node.node.name, node) for node in nodes), host, workers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and verify scoped node preflight Jobs."
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in (
        "cluster-id",
        "context",
        "namespace",
        "node-action-keys-secret",
        "connection-secret",
        "config-digest",
        "artifact-sha256",
        "bundle-sha256",
        "template-sha256",
        "template-content-sha256",
        "metrics-url-template",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--deadline-seconds", type=int, required=True)
    parser.add_argument("--node-dependency-image", default="")
    parser.add_argument("--node-wheelhouse-sha256", default="")
    parser.add_argument(
        "--require-rollback-slot", choices=("true", "false"), default="false"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-preflights", action="store_true")
    arguments = parser.parse_args()
    scope = NodeScope(
        arguments.cluster_id,
        arguments.context,
        arguments.namespace,
        arguments.node_action_keys_secret,
        arguments.connection_secret,
    )
    identity = InstallerIdentity(
        arguments.namespace,
        arguments.config_digest,
        arguments.artifact_sha256,
        arguments.bundle_sha256,
        arguments.template_sha256,
        arguments.node_action_keys_secret,
        arguments.deadline_seconds,
        arguments.metrics_url_template,
        template_content_sha256=arguments.template_content_sha256,
        node_dependency_image=arguments.node_dependency_image,
        node_wheelhouse_sha256=arguments.node_wheelhouse_sha256,
        require_rollback_slot=arguments.require_rollback_slot == "true",
    )
    try:
        nodes = prepare_node_batch(
            arguments.inputs, arguments.template, arguments.output, scope, identity
        )
        if arguments.run_preflights:
            from gpu_fault_release.rollout import Runner

            run_node_preflights(
                nodes, scope, identity, Runner(), workers=arguments.workers
            )
    except Exception as exc:
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        raise SystemExit(f"node preflight batch failed: {exc}") from None
    print(
        json.dumps(
            {
                "node_count": len(nodes),
                "status": "PASSED" if arguments.run_preflights else "RENDERED",
            }
        )
    )


if __name__ == "__main__":
    main()
