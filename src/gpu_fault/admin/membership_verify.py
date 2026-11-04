"""The cluster-scoped verify of ``join-cluster`` and ``remove-cluster``.

Both verbs end with one ``verify`` of the candidate (or remaining) site. The
engine scopes it to the verb's target (``gpu_fault_release.regional_verify_scope``):
the shared control plane and the target cluster's own checks gate, another GPU
cluster's own findings -- a node it lost, a Deployment of its own not Ready --
are reported and never fail or roll back the verb. The deploy host:

* runs the verify with the target (``--joining-cluster-id`` per joining
  cluster, ``--removed-cluster-id`` for a removal), captures the report and
  prints it again, so the command log keeps the whole document;
* keeps the findings that did not gate in the verb's ``VERIFIED`` evidence and
  in its result (``not_gating_findings``), each redacted and capped;
* once the verb completed, writes them to one record,
  ``<state-dir>/membership-verify/findings.json``, which ``status`` prints as
  ``finding:`` lines and the ``membership_verify_findings`` field. The next
  membership verb replaces it; a passing verify of the whole site (the one
  every ``deploy`` and ``config`` runs) removes it, because that verify judged
  every cluster again.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault_release.regional_verify_scope import (
    JOIN_VERB,
    JOINING_FLAG,
    REMOVE_VERB,
    REMOVED_FLAG,
    not_gating_findings,
)

__all__ = [
    "JOIN_VERB",
    "REMOVE_VERB",
    "clear_record",
    "driver_capture",
    "echoed_report",
    "finish_join",
    "finish_removal",
    "finish_verb",
    "rollout_arguments",
    "run_verify",
    "scoped_evidence",
    "status_view",
    "verified_findings",
    "verify_arguments",
]

#: The verb's ``VERIFIED`` evidence and result key.
FINDINGS_KEY = "not_gating_findings"
#: The ``status`` document field of the record.
FIELD = "membership_verify_findings"
RECORD_FILE = Path("membership-verify") / "findings.json"
ROLLOUT = Path("deploy/control-plane/regional/rollout-regional-release.sh")
SCHEMA_VERSION = 1
FINDING_LIMIT = 1200
#: Findings one ``status`` run prints as lines; the field keeps every one.
STATUS_LINE_LIMIT = 5


def verify_arguments(verb: str, cluster_ids: Sequence[str]) -> list[str]:
    """The engine flags that scope a verify to ``verb``'s target."""

    if verb == JOIN_VERB:
        return [
            item for cluster_id in cluster_ids for item in (JOINING_FLAG, cluster_id)
        ]
    if verb == REMOVE_VERB and len(cluster_ids) == 1:
        return [REMOVED_FLAG, cluster_ids[0]]
    raise BootstrapError(f"no cluster-scoped verify for {verb} {list(cluster_ids)}")


def rollout_arguments(
    rollout: Path, mode: str, config: Path, cluster_id: str | Sequence[str] | None
) -> list[str]:
    """A join's engine command: ``verify`` names its joining cluster(s) as scope."""

    arguments = [str(rollout), mode, "--config", str(config)]
    if cluster_id is None:
        return arguments
    if mode == "verify":
        ids = [cluster_id] if isinstance(cluster_id, str) else sorted(cluster_id)
        return [*arguments, *verify_arguments(JOIN_VERB, ids)]
    if not isinstance(cluster_id, str):
        raise BootstrapError(f"{mode} takes exactly one cluster id")
    return [arguments[0], mode, "--cluster-id", cluster_id, *arguments[2:]]


def driver_capture(mode: str) -> dict[str, Any]:
    """``run_driver`` options: a verify's report is read, then printed again."""

    return {"stdout": subprocess.PIPE} if mode == "verify" else {}


def echoed_report(result: subprocess.CompletedProcess[str]) -> dict[str, Any] | None:
    """Print the captured report for the command log; return it when it is one."""

    output = result.stdout if isinstance(result.stdout, str) else ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    lines = output.splitlines()
    # The document starts at its first line; a wrapper may print lines before it.
    starts = [index for index, line in enumerate(lines) if line.strip() == "{"]
    try:
        value = json.loads("\n".join(lines[starts[0] :]) if starts else output)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_verify(
    site: Any,
    scope: Sequence[str],
    *,
    failure: str,
    drive: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]],
) -> dict[str, Any] | None:
    """The engine's ``verify`` of ``site`` with ``scope``; its report.

    ``drive`` is the caller's ``(run_driver, materialized_release_config,
    effective_environment)``, so the caller's own collaborators run it.
    """

    run_driver, materialized_release_config, effective_environment = drive
    rollout = site.repository_root / ROLLOUT
    with materialized_release_config(site) as config:
        result = run_driver(
            [str(rollout), "verify", "--config", str(config), *scope],
            cwd=site.repository_root,
            env=effective_environment(site),
            check=False,
            **driver_capture("verify"),
        )
    report = echoed_report(result)
    if result.returncode:
        raise BootstrapError(failure)
    return report


def _capped(text: object, limit: int) -> str:
    """Redacted; the head kept -- a finding says what failed first."""

    safe = diagnostic_text(str(text or ""), limit=1 << 16)
    return safe if len(safe) <= limit else safe[:limit] + " [...]"


def _redacted(findings: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "check": _capped(item.get("check"), 120),
            "cluster_id": _capped(item.get("cluster_id"), 120),
            "finding": _capped(item.get("finding"), FINDING_LIMIT),
        }
        for item in findings
    ]


def scoped_evidence(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``VERIFIED`` evidence of the findings the verify did not gate on."""

    findings = _redacted(not_gating_findings(report))
    return {FINDINGS_KEY: findings} if findings else {}


def _saved_findings(state: Mapping[str, Any]) -> list[dict[str, str]]:
    verified = ((state.get("evidence") or {}).get("VERIFIED")) or {}
    findings = verified.get(FINDINGS_KEY) if isinstance(verified, dict) else None
    return [
        {key: str(item.get(key) or "") for key in ("check", "cluster_id", "finding")}
        for item in (findings if isinstance(findings, list) else [])
        if isinstance(item, dict)
    ]


def verified_findings(state: Mapping[str, Any]) -> dict[str, Any]:
    """The verb result's ``not_gating_findings``, from its ``VERIFIED`` evidence."""

    findings = _saved_findings(state)
    return {FINDINGS_KEY: findings} if findings else {}


def record_path(state_dir: Path) -> Path:
    return state_dir / RECORD_FILE


def finish_verb(
    state_dir: Path, *, verb: str, cluster_id: str, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Record a completed verb's findings for ``status``; its result fields.

    The record is replaced by every completed membership verb: one that
    completed without findings removes it.
    """

    findings = _saved_findings(state)
    path = record_path(state_dir)
    if not findings:
        path.unlink(missing_ok=True)
        return {}
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json_atomic(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "verb": verb,
            "cluster_id": cluster_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            FINDINGS_KEY: findings,
        },
    )
    return {FINDINGS_KEY: findings}


def finish_join(site: Any, cluster_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """``finish_verb`` of a completed join, in the site's state directory."""

    return finish_verb(
        site.source.parent, verb=JOIN_VERB, cluster_id=cluster_id, state=state
    )


def finish_removal(
    site: Any, cluster_id: str, state: Mapping[str, Any]
) -> dict[str, Any]:
    """``finish_verb`` of a completed removal, in the site's state directory."""

    return finish_verb(
        site.source.parent, verb=REMOVE_VERB, cluster_id=cluster_id, state=state
    )


def clear_record(state_dir: Path) -> None:
    """A verify of the whole site passed: every cluster was judged again."""

    record_path(state_dir).unlink(missing_ok=True)


def status_view(state_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """``({FIELD: record}, finding lines)``; nothing without a record."""

    path = record_path(state_dir)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, []
    except (OSError, ValueError) as exc:
        return {}, [f"membership verify findings record {path} is unreadable: {exc}"]
    findings = record.get(FINDINGS_KEY) if isinstance(record, dict) else None
    if not isinstance(findings, list) or not findings:
        return {}, []
    head = (
        f"{record.get('verb')} {record.get('cluster_id')} completed at "
        f"{record.get('recorded_at')} past {len(findings)} finding(s) on GPU "
        "clusters it does not target (not gating; status judges them live)"
    )
    lines = [head] + [
        f"{record.get('verb')} {record.get('cluster_id')}: not gating: "
        f"{item.get('cluster_id')} {item.get('check')}: {item.get('finding')}"
        for item in findings[:STATUS_LINE_LIMIT]
        if isinstance(item, dict)
    ]
    if len(findings) > STATUS_LINE_LIMIT:
        lines.append(
            f"{len(findings) - STATUS_LINE_LIMIT} more in {FIELD}.{FINDINGS_KEY}"
        )
    return {FIELD: record}, lines
