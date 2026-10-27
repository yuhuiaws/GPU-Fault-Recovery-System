"""Which GPU finding states earn a ``gpu_finding_history`` row.

Shared by the three ``update_gpu_findings`` implementations (memory, SQLite,
Postgres), which otherwise each kept their own copy of the rule and would
drift. Keeping history is a pure decision over the previous and the new
finding; the stores only write the row.
"""

from __future__ import annotations

from gpu_fault.gpu_metric_models import GpuHealthFinding


def records_finding_history(
    previous: GpuHealthFinding | None,
    finding: GpuHealthFinding,
    *,
    activated: bool,
) -> bool:
    """Whether ``finding`` is written to ``gpu_finding_history``.

    Always on activation (a first mint, a severity or automatic-action change:
    the transitions that also raise ``new_findings``). Also when a newer batch
    *replaced* an active finding at the same severity and action -- a new
    ``finding_id`` carrying the batch that re-graded it. That re-grade raises
    no incident, and until now left no record either: the finding
    ``thermal_violation_total_us`` re-minted by a ``candidate-recovered`` edge
    (HyperPod, 2026-10-27 01:48:32Z, the state VALIDATE_GPU failed on at the
    grace end) was only visible as the current state; the history showed the
    confirmed read's mint and nothing after. With the replacement recorded,
    the re-grades a validation waited on or failed on can be read back. A
    finding replaced by itself (same ``finding_id``, a replay) is not a
    re-grade and is not repeated.
    """

    if activated:
        return True
    return previous is not None and previous.finding_id != finding.finding_id
