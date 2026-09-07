"""One INFO line per retention sweep: what kind, how many, which keys.

Architecture review 2026-09-07, item D7. The six retention sweeps
(``cleanup_completed_processor_requests``, ``cleanup_expired_raw_evidence``,
``cleanup_hot_state``, ``cleanup_processor_lanes``,
``cleanup_terminal_remote_commands``, ``cleanup_terminal_fleet_deployments``)
deleted rows and returned a count. When a sweep removed a row somebody was
about to look at, or the wrong rows altogether, there was nothing to correlate
against. Every backend now routes its deletions through :func:`log_cleanup`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

LOGGER = logging.getLogger("gpu_fault.store.cleanup")

# How many deleted keys one line names. The sweeps are bounded by their
# ``limit`` (1000 by default), and a thousand keys is a log line nobody reads.
CLEANUP_LOG_KEY_LIMIT = 20


def log_cleanup(kind: str, keys: Sequence[str]) -> int:
    """Log the deletion of ``keys`` under ``kind`` and return how many there were.

    Nothing is logged for an empty sweep: the periodic runner already reports
    per-job totals, and a silent no-op every interval would only bury the lines
    that matter.
    """

    count = len(keys)
    if count == 0:
        return 0
    shown = list(keys[:CLEANUP_LOG_KEY_LIMIT])
    more = count - len(shown)
    LOGGER.info(
        "cleanup %s deleted %d rows; keys[:%d]=%s%s",
        kind,
        count,
        CLEANUP_LOG_KEY_LIMIT,
        ", ".join(str(key) for key in shown),
        f" (+{more} more)" if more else "",
    )
    return count
