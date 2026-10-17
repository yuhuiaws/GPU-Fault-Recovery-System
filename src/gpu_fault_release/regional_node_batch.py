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
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar
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
#: The private file the release engine names (``regional_release_node_preflight``
#: spells the same name; a test keeps the two equal) for this batch's per-node
#: report; the reconciler script passes it through.
#: Written on success and on refusal alike: each node's DCGM observation and,
#: for every failed node, its verdict's reasons and the remedies its Job named
#: -- what the operator needs, which a sensitive command's ``ERROR:`` lines
#: (three, cut at 200 characters) cannot carry for more than one node.
NODE_PREFLIGHT_REPORT_ENV = "GPU_FAULT_NODE_PREFLIGHT_REPORT"
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
        spec = node.get("spec")
        provider_id = (
            str(manifest_object(spec, "node spec").get("providerID") or "")
            if spec is not None
            else ""
        )
        nodes[name] = InstallerNode(
            name, str(values[0]), str(addresses[0]), str(values[1]), provider_id
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
    items: Iterable[tuple[str, _T]],
    action: Callable[[_T], None],
    workers: int,
    *,
    stop_on_failure: bool = True,
) -> None:
    """Run ``action`` over ``items`` with at most ``workers`` in flight.

    ``stop_on_failure`` starts nothing new after the first failure (admission
    batches); without it every item runs and the failures are raised as one
    error carrying every item's notes, so a host preflight names every node
    the operator has to fix rather than the first one to finish.
    """

    remaining = iter(items)
    failures: list[Exception] = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="node-preflight"
    ) as pool:
        active: dict[Future[None], str] = {}
        while True:
            while (not failures or not stop_on_failure) and len(active) < workers:
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
    if len(failures) == 1:
        raise failures[0]
    if failures:
        combined = RuntimeError(
            f"{len(failures)} node preflights failed: "
            + "; ".join(str(item) for item in failures)
        )
        for failure in failures:
            for note in getattr(failure, "__notes__", ()):
                combined.add_note(note)
        raise combined


def preflight_verdict(logs: str) -> dict[str, object] | None:
    """The last JSON verdict (an object with ``status``) a preflight Job printed."""

    for line in reversed(str(logs or "").splitlines()):
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            verdict = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(verdict, dict) and "status" in verdict:
            return verdict
    return None


def preflight_dcgm_observation(logs: str) -> dict[str, object] | None:
    """The ``dcgm`` object of the last JSON verdict a preflight Job printed.

    An eks-managed node's verdict carries its DCGM state (the node preflight's
    classification); HyperPod verdicts and older scripts carry none.
    """

    dcgm = (preflight_verdict(logs) or {}).get("dcgm")
    return dict(dcgm) if isinstance(dcgm, dict) else None


def preflight_refusal(logs: str, failure: BaseException) -> dict[str, list[str]]:
    """Why one node's host preflight failed: ``{"reasons": [...], "remedies": [...]}``.

    ``reasons`` are the Job verdict's own, each whole -- the list of what to
    fix on that node; ``remedies`` what each of the Job's ``ERROR:`` lines adds
    to a reason (``<reason>; <remedy>``). A Job that printed no verdict (it
    never ran, or died before judging) gives its ``ERROR:`` lines instead, and
    one without logs the failure's first line.
    """

    verdict = preflight_verdict(logs) or {}
    raw_reasons = verdict.get("reasons")
    reasons = [
        diagnostic_text(str(item))
        for item in (raw_reasons if isinstance(raw_reasons, list) else ())
        if str(item).strip()
    ]
    errors = list(
        dict.fromkeys(
            diagnostic_text(line.strip().removeprefix("ERROR: "))
            for line in str(logs or "").splitlines()
            if line.strip().startswith("ERROR: ")
        )
    )
    remedies = [
        error[len(reason) + 2 :]
        for reason in reasons
        for error in errors
        if error.startswith(f"{reason}; ")
    ]
    if not reasons:
        first = diagnostic_text(str(failure)).strip().splitlines()
        reasons = errors or [first[0] if first else type(failure).__name__]
    return {"reasons": reasons, "remedies": list(dict.fromkeys(remedies))}


def run_node_preflights(
    nodes: Sequence[PreparedNode],
    scope: NodeScope,
    identity: InstallerIdentity,
    runner: PreflightRunner,
    *,
    workers: int = 8,
    observations: dict[str, dict[str, object]] | None = None,
    refused: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, object]]:
    """Run every node's host preflight Job; the DCGM state each one observed.

    Every node runs even after one failed (``stop_on_failure=False``), and the
    batch fails once naming every failed node with its own verdict.
    ``observations`` and ``refused`` (``preflight_refusal`` per failed node)
    are filled as the Jobs finish, so a caller still has them on failure.
    """

    if not 1 <= workers <= DEPLOY_CONCURRENCY.read_only_checks:
        raise ValueError("node preflight concurrency is outside its safety budget")
    kubectl = ["kubectl", "--context", scope.context, "-n", scope.namespace]
    observations = {} if observations is None else observations
    refused = {} if refused is None else refused

    def job_logs(node: PreparedNode) -> str:
        return runner.run(
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
            try:
                observed = preflight_dcgm_observation(job_logs(node))
            except Exception:
                observed = None  # the observation is evidence, not the verdict
            if observed is not None:
                observations[node.node.name] = observed
        except Exception as exc:
            failure = exc
            logs = ""
            try:
                with cleanup_deadline("node preflight diagnostics", 30):
                    logs = job_logs(node)
                    observed = preflight_dcgm_observation(logs)
                    if observed is not None:
                        observations[node.node.name] = observed
                    if logs:
                        # The Job's own verdict (its ERROR: line and JSON) is
                        # what the operator needs; the marker keeps it through
                        # the sensitive-command redaction upstream.
                        failure.add_note(
                            f"ERROR: host preflight {node.node.name}: "
                            + diagnostic_text(logs)
                        )
            except Exception:
                pass
            refused[node.node.name] = preflight_refusal(logs, exc)
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
    _bounded_preflights(
        ((node.node.name, node) for node in nodes),
        host,
        workers,
        stop_on_failure=False,
    )
    return dict(sorted(observations.items()))


def write_preflight_report(
    node_count: int,
    observations: Mapping[str, Mapping[str, object]],
    refused: Mapping[str, Mapping[str, list[str]]],
    *,
    failed: bool,
) -> None:
    """Write the per-node report to the engine's ``NODE_PREFLIGHT_REPORT_ENV`` path.

    Nothing without the variable (an older engine). A report that cannot be
    written is said on stderr and never replaces the batch's own verdict.
    """

    target = os.environ.get(NODE_PREFLIGHT_REPORT_ENV, "").strip()
    if not target:
        return
    document = {
        "status": "FAILED" if failed else "PASSED",
        "node_count": node_count,
        "dcgm_hostengine": {
            name: dict(value) for name, value in sorted(observations.items())
        },
        "refused": {name: dict(value) for name, value in sorted(refused.items())},
    }
    try:
        _write_private_json(Path(target), document)
    except OSError as exc:
        print(f"WARN  node preflight report not written: {exc}", file=sys.stderr)


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
    observations: dict[str, dict[str, object]] = {}
    refused: dict[str, dict[str, list[str]]] = {}
    nodes: list[PreparedNode] = []
    try:
        nodes = prepare_node_batch(
            arguments.inputs, arguments.template, arguments.output, scope, identity
        )
        if arguments.run_preflights:
            from gpu_fault_release.rollout import Runner

            observations = run_node_preflights(
                nodes,
                scope,
                identity,
                Runner(),
                workers=arguments.workers,
                observations=observations,
                refused=refused,
            )
    except Exception as exc:
        if arguments.run_preflights:
            write_preflight_report(len(nodes), observations, refused, failed=True)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        raise SystemExit(f"ERROR: node preflight batch failed: {exc}") from None
    if arguments.run_preflights:
        write_preflight_report(len(nodes), observations, refused, failed=False)
    summary: dict[str, object] = {
        "node_count": len(nodes),
        "status": "PASSED" if arguments.run_preflights else "RENDERED",
    }
    if observations:
        # Each eks node's DCGM state, as its preflight classified it; the
        # release keeps them for status and verify.
        summary["dcgm_hostengine"] = observations
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
