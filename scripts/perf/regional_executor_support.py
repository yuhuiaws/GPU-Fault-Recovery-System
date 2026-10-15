"""What the perf runners share around their fake executor Jobs.

The action, integrated and correlated runners each run an Indexed Job of
fake executors (``benchmark_regional_action_executor.py``,
``benchmark_correlated_action_scenario.py``). Three things are common to them:

* the script ConfigMap ships ``benchmark_transport_retry.py`` beside the
  script, because the scripts import the product executor's transport
  retries from it (:func:`executor_scripts`);
* the Jobs keep ``backoffLimit: 0``. An executor Pod now fails only on a fatal
  condition -- a lost result, a refused or persistently failing claim, a
  protocol violation -- which already invalidates the round, and the Job
  controller then stops every sibling. A transient connection drop is retried
  and counted, so it no longer fails a Pod. Per-index backoff
  (``backoffLimitPerIndex``, Kubernetes 1.29+) would let the healthy siblings
  finish, but it needs a cluster version this harness cannot assume and a
  Job wait that ends on the Job's terminal condition instead of its first
  failed Pod; it would not rescue the round either. What the runners need is
  to stop waiting for workflows no executor is left to finish
  (:func:`executor_job_failure`) instead of waiting out their timeout;
* the executors' retried failures are reported beside the fatal ones
  (:func:`retry_fields`): ``claim_errors``/``result_errors`` stay the fatal
  counts the verdicts check.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

PERF_DIR = Path(__file__).resolve().parent
TRANSPORT_RETRY_SCRIPT = "benchmark_transport_retry.py"


def executor_scripts(
    script: str = "benchmark_regional_action_executor.py",
) -> dict[str, str]:
    """A fake executor's script ConfigMap: the script and the retries it imports."""

    return {
        name: (PERF_DIR / name).read_text() for name in (script, TRANSPORT_RETRY_SCRIPT)
    }


def executor_job_failure(job: str, client: Callable[..., str]) -> str | None:
    """Why an executor Job has failed -- its ``Failed`` condition, or a failed
    Pod that condition follows under ``backoffLimit: 0`` -- or ``None``.

    An unreadable Job is not a failure: the caller keeps waiting.
    """

    try:
        raw = client("get", "job", job, "-o", "json", check=False, timeout=60)
        document = json.loads(raw) if raw.strip() else {}
        status = document.get("status") or {}
        conditions = status.get("conditions") or []
        failed = int(status.get("failed") or 0)
    except (OSError, RuntimeError, ValueError, AttributeError, TypeError):
        return None
    for condition in conditions:
        if condition.get("type") == "Failed" and condition.get("status") == "True":
            return str(condition.get("reason") or "Failed")
    return f"{failed} executor Pod(s) failed" if failed else None


def merged_counts(documents: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in documents:
        for name, value in (item.get(key) or {}).items():
            merged[str(name)] = merged.get(str(name), 0) + int(value)
    return dict(sorted(merged.items()))


def retry_fields(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """The executors' retried transport failures (reported, never a failure of
    the round), the categories of their lost results, and each failed Pod's
    fatal conditions."""

    return {
        "claim_retries": sum(int(item.get("claim_retries", 0)) for item in documents),
        "claim_retry_counts": merged_counts(documents, "claim_retry_counts"),
        "result_retries": sum(int(item.get("result_retries", 0)) for item in documents),
        "result_retry_counts": merged_counts(documents, "result_retry_counts"),
        "result_error_counts": merged_counts(documents, "result_error_counts"),
        "fatal": {
            str(item.get("executor_id")): list(item["fatal"])
            for item in documents
            if item.get("fatal")
        },
    }
