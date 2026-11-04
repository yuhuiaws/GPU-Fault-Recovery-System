"""The remote-command half of ``control_api``: unclaimed commands and the
executor's internal errors, judged per GPU cluster where the Store says which.

Both used to be judged on fleet-wide values only, so in the verify of a
``join-cluster`` / ``remove-cluster`` (``regional_verify_scope``) a command
nobody claimed on another cluster -- the Agent of a node a spot reclaim took,
an executor rescheduled off the lost node -- or an internal error of another
cluster's executor failed the target's verify, and a join was rolled back.

* An unscoped verify (``deploy``, ``config``, ``status``) judges the fleet's
  values exactly as before, with the same messages.
* A scoped verify reads the Store's per-cluster values
  (``oldest_unclaimed_age_seconds_by_cluster``,
  ``executor_internal_error_total_by_cluster`` and
  ``executor_internal_error_last_seen_timestamp_seconds_by_cluster``): a
  target cluster's value fails the check, another cluster's is its finding
  (``details.not_gating``); a cluster the site does not name gates too.
* Fail-closed: a control plane older than the per-cluster values, or values
  that do not account for the fleet's (the newest/oldest is not one of them,
  the counts do not add up), are judged fleet-wide, which gates.

The internal-error baseline stays the one the release state recorded
(``previous``, captured at the start of the last release transaction). It is
the newest error the Store held then, in every cluster, so an error recorded
after it is newer than that timestamp in its own cluster as well: a cluster
whose last seen is past the baseline had an error since, and no per-cluster
baseline is needed. Without a recorded timestamp (``previous`` gone after a
``sync-state``, a no-op ``deploy`` or a bootstrap) the baseline is a count: a
zero count is attributed per cluster too (every error is new), a non-zero one
recorded by an engine older than the timestamp cannot be, and is judged
fleet-wide.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_verify_scope import (
    other_cluster,
    record_not_gating,
    verify_scope,
)

UNCLAIMED_BY_CLUSTER = "oldest_unclaimed_age_seconds_by_cluster"
LAST_SEEN = "executor_internal_error_last_seen_timestamp_seconds"
TOTAL = "executor_internal_error_total"
#: The per-cluster internal-error values of ``remote_command_stats()``
#: (``gpu_fault.store.shared.remote_helpers``); absent from an older Store.
INTERNAL_ERRORS_BY_CLUSTER = "executor_internal_error_total_by_cluster"
INTERNAL_ERROR_LAST_SEEN_BY_CLUSTER = (
    "executor_internal_error_last_seen_timestamp_seconds_by_cluster"
)
#: Float slack when per-cluster values are compared with the fleet's.
EPSILON = 1e-6


def judge_remote_commands(release: Any, report: dict[str, Any]) -> None:
    """Raise on a gating finding; file another cluster's under ``report``."""

    remote = report.get("remote_commands") or {}
    _judge_unclaimed(release, report, remote)
    _judge_internal_errors(release, report, remote)


def _judge_unclaimed(
    release: Any, report: dict[str, Any], remote: Mapping[str, Any]
) -> None:
    limit = release.config.health.remote_command_max_unclaimed_seconds
    oldest = float(remote.get("oldest_unclaimed_age_seconds", 0) or 0)
    if oldest <= limit:
        return
    ages = _breakdown(remote.get(UNCLAIMED_BY_CLUSTER), oldest, max)
    _judge(
        release,
        report,
        None
        if ages is None
        else {key: age for key, age in ages.items() if age > limit},
        fleet=f"oldest unclaimed remote command is {oldest:.1f}s; limit is {limit}s",
        cluster=lambda age: (
            f"oldest unclaimed remote command is {age:.1f}s; limit is {limit}s"
        ),
    )


def _judge_internal_errors(
    release: Any, report: dict[str, Any], remote: Mapping[str, Any]
) -> None:
    current_total = int(remote.get(TOTAL, 0) or 0)
    previous = release._load_state().get("previous")
    raw_baseline_timestamp = (
        previous.get(LAST_SEEN) if isinstance(previous, dict) else None
    )
    current_raw_timestamp = remote.get(LAST_SEEN)
    if raw_baseline_timestamp is not None and current_raw_timestamp is not None:
        try:
            baseline_timestamp = float(raw_baseline_timestamp)
            current_timestamp = float(current_raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise ReleaseError(
                "executor internal remote-command error timestamp is invalid"
            ) from exc
        if min(baseline_timestamp, current_timestamp) < 0:
            raise ReleaseError(
                "executor internal remote-command error timestamp is invalid"
            )
        if current_timestamp <= baseline_timestamp:
            return
        seen = _breakdown(
            remote.get(INTERNAL_ERROR_LAST_SEEN_BY_CLUSTER), current_timestamp, max
        )
        _judge(
            release,
            report,
            None
            if seen is None
            else {key: at for key, at in seen.items() if at > baseline_timestamp},
            fleet="executor internal remote-command error observed during release: "
            f"{baseline_timestamp:.6f}->{current_timestamp:.6f}",
            cluster=lambda at: (
                "executor internal remote-command error observed during release: "
                f"{baseline_timestamp:.6f}->{at:.6f}"
            ),
        )
        return
    raw_baseline = previous.get(TOTAL, 0) if isinstance(previous, dict) else 0
    try:
        baseline = int(raw_baseline or 0)
    except (TypeError, ValueError) as exc:
        raise ReleaseError(
            "executor internal remote-command error baseline is invalid"
        ) from exc
    if current_total <= baseline:
        return
    counts = (
        _breakdown(remote.get(INTERNAL_ERRORS_BY_CLUSTER), current_total, sum)
        if baseline == 0
        else None
    )
    _judge(
        release,
        report,
        None if counts is None else {key: n for key, n in counts.items() if n > 0},
        fleet="executor internal remote-command errors increased during release: "
        f"{baseline}->{current_total}",
        cluster=lambda count: (
            "executor internal remote-command errors increased during release: "
            f"0->{count:.0f}"
        ),
    )


def _breakdown(
    values: object, fleet: float, combine: Callable[[Iterable[float]], float]
) -> dict[str, float] | None:
    """The per-cluster values, when they account for the fleet's ``fleet``
    (``combine`` of them: the newest/oldest, or the sum); None otherwise."""

    if not isinstance(values, Mapping) or not values:
        return None
    try:
        parsed = {str(key): float(value) for key, value in values.items()}
    except (TypeError, ValueError):
        return None
    if any(value < 0 for value in parsed.values()):
        return None
    if abs(combine(parsed.values()) - fleet) > EPSILON:
        return None
    return parsed


def _judge(
    release: Any,
    report: dict[str, Any],
    over: dict[str, float] | None,
    *,
    fleet: str,
    cluster: Callable[[float], str],
) -> None:
    """Fail on ``fleet`` unless a scoped verify can attribute ``over`` (the
    clusters past the limit); then another site cluster's is a finding and
    any other -- the target's, an unmanaged cluster's -- fails."""

    if verify_scope(release) is None or not over:
        raise ReleaseError(fleet)
    gating = {
        key: value for key, value in over.items() if not other_cluster(release, key)
    }
    if gating:
        raise ReleaseError(
            "; ".join(
                f"{key}: {cluster(value)}" for key, value in sorted(gating.items())
            )
        )
    for key, value in sorted(over.items()):
        record_not_gating(report, key, cluster(value))
