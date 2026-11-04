"""BOOT-029: the registry Secret and the CPU Pods after join-cluster/remove-cluster.

``join-cluster`` and ``remove-cluster`` rewrite the registry Secret
(``gpu-fault-regional-clusters``) and publish the durable registry head; by
design they never restart the CPU Deployments (membership is hot). Since batch
245 every CPU role mounts the Secret's ``clusters.json`` read-only and
``secret_drift`` compares that copy, which kubelet keeps current, with the head
(``/healthz`` ``regional_registry.secret_source=mounted-file``). Before, it
compared the copy each process had started with, so every replica that
predated a join or remove reported drift although Secret and head agreed. The
BOOT-019/024/025 checklists require both facts after every such verb -- no
drift, no restart -- and the driver asserted neither, so that surfaced only in
a manual check after a whole sequence.

Around every join-cluster and remove-cluster the driver runs, read-only on the
CPU cluster (``kubectl get``/``exec`` with the kubeconfig and namespace the
site's ``site.yaml`` records):

* ``baseline`` right before the verb: the CPU runtime Deployments
  (``regional_deployment_inventory.CPU_RUNTIME_DEPLOYMENTS``: every role that
  serves the registry ``/healthz``) and their Pods -- Deployment UID, Pod
  template digest, Pod names, UIDs and container restart counts.
* ``check`` right after it:

  - every Running CPU Pod's ``/healthz?verbose=1``, read by the perf registry
    gate's reader (``scripts/perf/regional_registry_alignment.py``) with the
    gate's bound: while a replica that compares the mounted copy still
    drifts, it gets ``MOUNTED_CONVERGENCE_SECONDS`` for kubelet to re-project
    the Secret. Then every Pod must report ``secret_source=mounted-file``,
    ``secret_drift=false`` and ``ready=true``;
  - the baseline Pods are still the Pods, with the same restart counts, and no
    Pod template changed: no restart was needed. The one exception is the
    BOOT-019 role boundary's audited roll: a membership change re-publishes
    the failure-domain map and stamps its digest on the control-worker's Pod
    template (``failure_domain_map.apply_failure_domain_map``), so the
    control-worker may roll when that annotation changed and nothing else in
    its template did.

A product that predates batch 245 reports no ``secret_source`` at all: its
replicas compare the start-up copy and, by that product's design, drift after
every join or remove until restarted. Like the GENERAL-LIVE product-generation
guards, the check records that generation as ``NOT_APPLICABLE`` with what it
saw and does not fail, so an older product is not judged by the mounted-Secret
rule. Pods of both generations at once fail the check.

Each verdict joins the stage receipt under ``registry_convergence`` (one entry
per verb, in order); the readable lines go to stderr, which the driver appends
to the stage log. Neither ever carries the Secret: digests are cut to 12 hex
digits, next to Pod names and UIDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from gpu_fault.admin.failure_domain_map import CONTROL_WORKER_DEPLOYMENT
from gpu_fault.failure_domains import FAILURE_DOMAIN_MAP_ANNOTATION
from gpu_fault_release.regional_deployment_inventory import CPU_RUNTIME_DEPLOYMENTS
from scripts.perf.regional_registry_alignment import (
    MOUNTED_CONVERGENCE_POLL_SECONDS,
    MOUNTED_CONVERGENCE_SECONDS,
    MOUNTED_SECRET_SOURCE,
    api_replica_registry_health,
    await_mounted_convergence,
)

__all__ = [
    "DriftCheckError",
    "check_operation",
    "cpu_inventory",
    "kubectl_control",
    "main",
    "pod_changes",
    "product_generation",
    "recorded_verdicts",
    "registry_findings",
    "site_cpu_target",
]

RECEIPT_KEY = "registry_convergence"
OPERATIONS = ("join-cluster", "remove-cluster")
BASELINE_SCHEMA = 1
# Overrides of the perf gate's bound and poll interval; the verdict records
# the values used. A shorter window can only fail a drifting replica sooner.
WINDOW_ENV = "BOOT029_REGISTRY_WINDOW_SECONDS"
POLL_ENV = "BOOT029_REGISTRY_POLL_SECONDS"
KUBECTL_TIMEOUT_SECONDS = 120
DIGEST_CHARS = 12
SECRET_ERROR_CHARS = 200

PASS = "PASS"
FAIL = "FAIL"
NOT_APPLICABLE = "NOT_APPLICABLE"
#: Every answering Pod names the Secret copy it compares (batch 245 onward).
MOUNTED_GENERATION = "secret-source"
#: No answering Pod does: the product compares each process's start-up copy.
STARTUP_GENERATION = "pre-secret-source"
MIXED_GENERATION = "mixed"
NO_ANSWER = "no-answer"


class DriftCheckError(RuntimeError):
    """The CPU control plane could not be read the way the check needs."""


def short(value: object) -> str | None:
    """A digest cut to 12 hex digits; anything else is None."""

    return value[:DIGEST_CHARS] if isinstance(value, str) and value else None


def site_cpu_target(state: Path) -> tuple[str, str]:
    """``(kubeconfig, namespace)`` of the CPU control plane ``site.yaml`` records.

    The two fields ``load_site`` renders into ``cpu_kubeconfig`` and
    ``namespace``; a deploy-written site records the kubeconfig as an absolute
    path, and nothing here guesses one.
    """

    path = state / "site.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DriftCheckError(
            f"the state directory's site.yaml cannot be read ({type(exc).__name__})"
        ) from None
    spec = document.get("spec") if isinstance(document, dict) else None
    spec = spec if isinstance(spec, dict) else {}
    cpu = spec.get("cpu")
    kubeconfig = cpu.get("kubeconfig") if isinstance(cpu, dict) else None
    namespace = spec.get("namespace")
    if not isinstance(kubeconfig, str) or not Path(kubeconfig).is_absolute():
        raise DriftCheckError("site.yaml records no absolute spec.cpu.kubeconfig")
    if not isinstance(namespace, str) or not namespace:
        raise DriftCheckError("site.yaml records no spec.namespace")
    return kubeconfig, namespace


def kubectl_control(
    kubeconfig: str,
    namespace: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Callable[..., str]:
    """A read-only kubectl against the CPU control plane, stdout on success."""

    def control(*arguments: str) -> str:
        shown = " ".join(arguments[:3])
        try:
            completed = run(
                ["kubectl", "--kubeconfig", kubeconfig, "-n", namespace, *arguments],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=KUBECTL_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DriftCheckError(
                f"kubectl {shown} exceeded {KUBECTL_TIMEOUT_SECONDS} s"
            ) from None
        if completed.returncode:
            detail = (completed.stderr or "").strip().splitlines()
            raise DriftCheckError(
                f"kubectl {shown} exited {completed.returncode}: "
                + (detail[-1][:300] if detail else "no output")
            )
        return str(completed.stdout)

    return control


def _items(document: str) -> list[dict[str, Any]]:
    value = json.loads(document)
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise DriftCheckError("kubectl returned no item list")
    return [item for item in items if isinstance(item, dict)]


def _template_identity(template: Mapping[str, Any]) -> dict[str, Any]:
    """The failure-domain map digest a Pod template carries, and the digest
    of everything else in it (the map annotation removed)."""

    rest = json.loads(json.dumps(template))
    metadata = rest.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    annotations = dict(metadata.get("annotations") or {})
    map_sha256 = annotations.pop(FAILURE_DOMAIN_MAP_ANNOTATION, None)
    metadata["annotations"] = annotations
    rest["metadata"] = metadata
    encoded = json.dumps(rest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "failure_domain_map_sha256": map_sha256,
        "template_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def cpu_inventory(control: Callable[..., str]) -> dict[str, dict[str, Any]]:
    """The CPU runtime Deployments and their Pods as they stand, by role.

    Per Deployment: its UID, its Pod template's identity and every Pod of the
    role that is not terminating, by UID -- name, phase and the sum of its
    containers' restart counts. A role missing from the cluster is an error.
    """

    deployments = _items(
        control("get", "deployment", *CPU_RUNTIME_DEPLOYMENTS, "-o", "json")
    )
    pods = _items(
        control(
            "get",
            "pod",
            "-l",
            f"app in ({','.join(CPU_RUNTIME_DEPLOYMENTS)})",
            "-o",
            "json",
        )
    )
    roles: dict[str, dict[str, Any]] = {}
    for item in deployments:
        metadata = item.get("metadata") or {}
        template = (item.get("spec") or {}).get("template")
        name = metadata.get("name")
        if name not in CPU_RUNTIME_DEPLOYMENTS:
            continue
        if not metadata.get("uid") or not isinstance(template, dict):
            raise DriftCheckError(f"Deployment {name} has no UID or Pod template")
        roles[name] = {
            "uid": str(metadata["uid"]),
            **_template_identity(template),
            "pods": {},
        }
    missing = [name for name in CPU_RUNTIME_DEPLOYMENTS if name not in roles]
    if missing:
        raise DriftCheckError(
            "CPU runtime Deployment(s) missing: " + ", ".join(missing)
        )
    for item in pods:
        metadata = item.get("metadata") or {}
        role = roles.get((metadata.get("labels") or {}).get("app"))
        if role is None or metadata.get("deletionTimestamp"):
            continue
        if not metadata.get("uid") or not metadata.get("name"):
            raise DriftCheckError("a CPU Pod has no name or UID")
        status = item.get("status") or {}
        role["pods"][str(metadata["uid"])] = {
            "pod": str(metadata["name"]),
            "phase": status.get("phase"),
            "restarts": sum(
                int(entry.get("restartCount") or 0)
                for entry in status.get("containerStatuses") or []
                if isinstance(entry, dict)
            ),
        }
    return roles


def _named(uids: Sequence[str], pods: Mapping[str, Mapping[str, Any]]) -> str:
    return ", ".join(f"{pods[uid]['pod']} (uid {uid})" for uid in uids) or "none"


def pod_changes(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    operation: str,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """``(findings, rolled, unchanged)`` between the baseline and the reading
    after ``operation``, by the BOOT-019 role boundary.

    No Deployment is replaced, no Pod template changes and every baseline Pod
    is still there with its restart count, with no Pod added -- except that
    the control-worker may roll when its template changed in the
    failure-domain map digest alone (``rolled``).
    """

    findings: list[str] = []
    rolled: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for name in CPU_RUNTIME_DEPLOYMENTS:
        old, new = before.get(name), after.get(name)
        if old is None or new is None:
            findings.append(
                f"{name} is missing from the {'baseline' if old is None else 'reading'}"
            )
            continue
        if old["uid"] != new["uid"]:
            findings.append(
                f"{name}: the Deployment was replaced (uid {old['uid']} -> {new['uid']})"
            )
            continue
        old_map = old["failure_domain_map_sha256"]
        new_map = new["failure_domain_map_sha256"]
        map_changed = old_map != new_map
        template_changed = old["template_sha256"] != new["template_sha256"]
        # The audited roll: the new failure-domain map digest is the only change
        # to the control-worker's Pod template.
        map_roll = (
            name == CONTROL_WORKER_DEPLOYMENT and map_changed and not template_changed
        )
        if template_changed:
            findings.append(
                f"{name}: its Pod template changed (template "
                f"{short(old['template_sha256'])} -> {short(new['template_sha256'])}"
                f"{', besides the failure-domain map' if map_changed else ''}): "
                f"{operation} started a roll of the role"
            )
        elif map_changed and not map_roll:
            findings.append(
                f"{name}: its Pod template's failure-domain map digest changed, "
                "which only the control-worker carries"
            )
        old_pods, new_pods = old["pods"], new["pods"]
        restarted = sorted(
            uid
            for uid in set(old_pods) & set(new_pods)
            if new_pods[uid]["restarts"] > old_pods[uid]["restarts"]
        )
        if restarted:
            findings.append(
                f"{name}: container restarts in " + _named(restarted, new_pods)
            )
        gone = sorted(set(old_pods) - set(new_pods))
        added = sorted(set(new_pods) - set(old_pods))
        if (gone or added) and not map_roll:
            findings.append(
                f"{name}: Pod(s) {_named(gone, old_pods)} replaced by "
                f"{_named(added, new_pods)}; {operation} must not need a restart"
            )
        if map_roll:
            rolled.append(
                {
                    "deployment": name,
                    "failure_domain_map_sha256_12": [short(old_map), short(new_map)],
                    "pods_before": len(old_pods),
                    "pods_after": len(new_pods),
                    "pods_replaced": len(gone),
                }
            )
        elif not (template_changed or map_changed or restarted or gone or added):
            unchanged.append(name)
    return findings, rolled, unchanged


def product_generation(replicas: Sequence[Mapping[str, Any]]) -> str:
    """Which product the Pods are, from the keys their ``/healthz`` reports.

    Only a Pod that answered ``regional_registry`` with a boolean
    ``secret_drift`` counts: a Pod that did not answer is a failure, never a
    sign of an older product.
    """

    answered = [item for item in replicas if isinstance(item.get("secret_drift"), bool)]
    if not answered:
        return NO_ANSWER
    reported = {bool(item.get("secret_source_reported")) for item in answered}
    if reported == {True}:
        return MOUNTED_GENERATION
    if reported == {False}:
        return STARTUP_GENERATION
    return MIXED_GENERATION


def registry_findings(
    replicas: Sequence[Mapping[str, Any]], *, window: float
) -> list[str]:
    """What keeps each Pod from ``secret_source=mounted-file``, no drift, ready."""

    findings: list[str] = []
    for item in replicas:
        pod = item["pod"]
        if not isinstance(item.get("secret_drift"), bool):
            findings.append(
                f"{pod}: /healthz answered no regional_registry "
                f"(HTTP {item.get('http_status')})"
            )
            continue
        source = item.get("secret_source")
        mounted = source == MOUNTED_SECRET_SOURCE
        if not item.get("secret_source_reported"):
            findings.append(
                f"{pod} names no secret_source: it compares the registry Secret "
                "it started with, which only a restart refreshes"
            )
        elif not mounted:
            findings.append(
                f"{pod} compares the {source or 'unnamed'} copy of the registry "
                "Secret, not the mounted file: it drifts after every join or "
                "remove until restarted"
            )
        if item.get("secret_drift") is not False or item.get("ready") is not True:
            error = item.get("secret_error")
            findings.append(
                f"{pod} reports secret_drift={json.dumps(item.get('secret_drift'))} "
                f"ready={json.dumps(item.get('ready'))}"
                + (f" after the {window:g} s convergence window" if mounted else "")
                + f" (Secret {short(item.get('secret_config_sha256')) or 'unreported'}"
                f", durable head {short(item.get('durable_config_sha256')) or 'unreported'}"
                + (f", {str(error)[:SECRET_ERROR_CHARS]}" if error else "")
                + ")"
            )
    return findings


def _pod_record(item: Mapping[str, Any]) -> dict[str, Any]:
    error = item.get("secret_error")
    return {
        "pod": item.get("pod"),
        "app": item.get("app"),
        "uid": item.get("uid"),
        "secret_source": item.get("secret_source"),
        "secret_drift": item.get("secret_drift"),
        "ready": item.get("ready"),
        "secret_config_sha256_12": short(item.get("secret_config_sha256")),
        "durable_config_sha256_12": short(item.get("durable_config_sha256")),
        "secret_error": str(error)[:SECRET_ERROR_CHARS] if error else None,
    }


def check_operation(
    control: Callable[..., str],
    *,
    baseline: Mapping[str, Mapping[str, Any]],
    operation: str,
    window_seconds: float = MOUNTED_CONVERGENCE_SECONDS,
    poll_seconds: float = MOUNTED_CONVERGENCE_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The verdict on the CPU Pods after ``operation`` against its baseline."""

    replicas, skipped = api_replica_registry_health(
        control, apps=CPU_RUNTIME_DEPLOYMENTS
    )
    replicas, skipped, waited = await_mounted_convergence(
        control,
        replicas,
        skipped,
        apps=CPU_RUNTIME_DEPLOYMENTS,
        timeout_seconds=window_seconds,
        poll_seconds=poll_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    after = cpu_inventory(control)
    generation = product_generation(replicas)
    findings: list[str] = []
    if generation == NO_ANSWER:
        findings.append("no Running CPU Pod answered /healthz with a regional_registry")
    elif generation == MIXED_GENERATION:
        findings.append(
            "CPU Pods of two product generations: "
            + ", ".join(
                f"{item['pod']} {'names' if item.get('secret_source_reported') else 'lacks'}"
                " secret_source"
                for item in replicas
                if isinstance(item.get("secret_drift"), bool)
            )
        )
    findings.extend(registry_findings(replicas, window=window_seconds))
    changes, rolled, unchanged = pod_changes(baseline, after, operation=operation)
    findings.extend(changes)
    if generation == STARTUP_GENERATION:
        verdict = NOT_APPLICABLE
        reason = (
            f"after {operation}: the product predates the mounted registry Secret "
            "(batch 245): no CPU Pod's /healthz reports secret_source, so "
            "secret_drift compares each process's start-up copy and a join or "
            "remove drifts until a restart by that product's design; recorded, "
            f"not judged ({len(findings)} finding(s) under the mounted-Secret rule)"
        )
    elif findings:
        verdict = FAIL
        reason = f"after {operation}: {findings[0]}" + (
            f" (+{len(findings) - 1} more; see the stage log)"
            if len(findings) > 1
            else ""
        )
    else:
        verdict = PASS
        reason = (
            f"after {operation}: all {len(replicas)} CPU Pod(s) compare the mounted "
            f"registry Secret and report no drift ({waited:g} s of the "
            f"{window_seconds:g} s window); unchanged: {', '.join(unchanged) or 'none'}"
            + "".join(
                f"; {item['deployment']} rolled by the failure-domain map publication"
                for item in rolled
            )
        )
    return {
        "operation": operation,
        "verdict": verdict,
        "product_generation": generation,
        "reason": reason,
        "window_seconds": window_seconds,
        "poll_seconds": poll_seconds,
        "waited_seconds": waited,
        "pods": [_pod_record(item) for item in replicas],
        "skipped_pods": [dict(item) for item in skipped],
        "unchanged_roles": unchanged,
        "rolled_by_failure_domain_map": rolled,
        "findings": findings,
    }


def recorded_verdicts(receipt: Path) -> list[dict[str, Any]]:
    """The verdicts this stage attempt recorded so far (the receipt's last list)."""

    verdicts: list[dict[str, Any]] = []
    if receipt.is_file():
        for line in receipt.read_text(encoding="utf-8").splitlines():
            value = json.loads(line) if line.strip() else None
            if isinstance(value, dict) and isinstance(value.get(RECEIPT_KEY), list):
                verdicts = list(value[RECEIPT_KEY])
    return verdicts


def _seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise DriftCheckError(f"{name} is not a number of seconds") from None
    if not math.isfinite(value) or value < 0:
        raise DriftCheckError(f"{name} must be a finite, non-negative number")
    return value


def _arn_digests(arns: Sequence[str]) -> list[str]:
    return [hashlib.sha256(arn.encode()).hexdigest()[:DIGEST_CHARS] for arn in arns]


def _without_paths(text: str, roots: Sequence[Path]) -> str:
    """``text`` with every root path replaced by its digest: the receipt
    names directories only as digests, and an error message may quote one."""

    paths = {str(path) for root in roots for path in (root, root.resolve())}
    for path in sorted(paths, key=len, reverse=True):
        digest = hashlib.sha256(path.encode()).hexdigest()[:DIGEST_CHARS]
        text = text.replace(path, f"<path sha256:{digest}>")
    return text


def report_lines(verdict: Mapping[str, Any]) -> list[str]:
    """The stage-log lines: the verdict, one line per Pod, roll and finding."""

    lines = [
        f"registry convergence #{verdict['sequence']}: {verdict['verdict']} -- "
        f"{verdict['reason']}"
    ]
    for pod in verdict["pods"]:
        lines.append(
            f"  {pod['pod']} uid={pod['uid']} source={pod['secret_source']} "
            f"drift={json.dumps(pod['secret_drift'])} ready={json.dumps(pod['ready'])} "
            f"secret={pod['secret_config_sha256_12']} "
            f"durable={pod['durable_config_sha256_12']}"
            + (f" error={pod['secret_error']}" if pod["secret_error"] else "")
        )
    for item in verdict["skipped_pods"]:
        state = "terminating" if item.get("terminating") else item.get("phase")
        lines.append(f"  {item.get('pod')} not asked ({state})")
    for item in verdict["rolled_by_failure_domain_map"]:
        old, new = item["failure_domain_map_sha256_12"]
        lines.append(
            f"  {item['deployment']} rolled by the failure-domain map publication "
            f"(map {old} -> {new}; {item['pods_replaced']} of "
            f"{item['pods_before']} Pod(s) replaced, {item['pods_after']} now)"
        )
    if verdict["unchanged_roles"]:
        lines.append(
            "  unchanged (Deployment, Pod template, Pod UIDs, restart counts): "
            + ", ".join(verdict["unchanged_roles"])
        )
    lines.extend(f"  finding: {finding}" for finding in verdict["findings"])
    return lines


def _baseline(arguments: argparse.Namespace) -> int:
    try:
        roles = cpu_inventory(kubectl_control(*site_cpu_target(arguments.state)))
        temporary = arguments.out.with_name(arguments.out.name + ".tmp")
        temporary.write_text(
            json.dumps({"schema_version": BASELINE_SCHEMA, "roles": roles}),
            encoding="utf-8",
        )
        temporary.replace(arguments.out)
    except (DriftCheckError, OSError, ValueError) as exc:
        print(
            f"the CPU Pods could not be read before {arguments.operation}, which "
            f"did not run: {exc}"
        )
        return 1
    pods = sum(len(role["pods"]) for role in roles.values())
    print(
        f"registry baseline before {arguments.operation}: {pods} CPU Pod(s) in "
        f"{len(roles)} role(s)",
        file=sys.stderr,
    )
    return 0


def _check(arguments: argparse.Namespace) -> int:
    previous = recorded_verdicts(arguments.receipt)
    window = poll = None
    try:
        window = _seconds(WINDOW_ENV, MOUNTED_CONVERGENCE_SECONDS)
        poll = _seconds(POLL_ENV, MOUNTED_CONVERGENCE_POLL_SECONDS)
        document = json.loads(arguments.baseline.read_text(encoding="utf-8"))
        if document.get("schema_version") != BASELINE_SCHEMA:
            raise DriftCheckError("the registry baseline has an unknown schema")
        result = check_operation(
            kubectl_control(*site_cpu_target(arguments.state)),
            baseline=document["roles"],
            operation=arguments.operation,
            window_seconds=window,
            poll_seconds=poll,
        )
    except Exception as exc:  # fail closed: the verdict names what went wrong
        failure = _without_paths(
            f"the check could not complete ({type(exc).__name__}: {exc})",
            (arguments.state, arguments.baseline.parent, arguments.receipt.parent),
        )
        result = {
            "operation": arguments.operation,
            "verdict": FAIL,
            "product_generation": None,
            "reason": f"after {arguments.operation}: {failure}",
            "window_seconds": window,
            "poll_seconds": poll,
            "waited_seconds": None,
            "pods": [],
            "skipped_pods": [],
            "unchanged_roles": [],
            "rolled_by_failure_domain_map": [],
            "findings": [failure],
        }
    verdict = {
        "sequence": len(previous) + 1,
        "gpu_arns_sha256_12": _arn_digests(arguments.gpu_arn),
        **result,
    }
    with arguments.receipt.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({RECEIPT_KEY: [*previous, verdict]}) + "\n")
    for line in report_lines(verdict):
        print(line, file=sys.stderr)
    print(verdict["reason"])
    return 1 if verdict["verdict"] == FAIL else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    baseline = actions.add_parser("baseline", help="the CPU Pods right before a verb")
    baseline.add_argument("--state", type=Path, required=True)
    baseline.add_argument("--operation", choices=OPERATIONS, required=True)
    baseline.add_argument("--out", type=Path, required=True)
    check = actions.add_parser("check", help="the CPU Pods right after the verb")
    check.add_argument("--state", type=Path, required=True)
    check.add_argument("--operation", choices=OPERATIONS, required=True)
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--receipt", type=Path, required=True)
    check.add_argument(
        "--gpu-arn",
        action="append",
        default=[],
        help="a GPU cluster ARN the verb named (recorded as a digest)",
    )
    arguments = parser.parse_args(argv)
    if arguments.action == "baseline":
        return _baseline(arguments)
    return _check(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
