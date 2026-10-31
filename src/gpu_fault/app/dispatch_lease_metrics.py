"""The workflow-dispatch lease and tick families (b226).

Rendered with the other ``gpu_fault_workflow_dispatch_*`` families, on every
process whether or not it holds the lease, so a rate over them never starts
from a missing series. Per process: only the lease holder scans, so the tick
histogram and the counters are reset domains of their own
(``metric_aggregation``: PER_PROCESS), and the held gauge adds up to the
number of holders in the Pod (SUM; across the region it is 1 while one process
dispatches).
"""

from __future__ import annotations

from gpu_fault.execution.dispatch_lease import (
    DISPATCH_TICK_SECONDS_BUCKETS,
    LOSS_MOMENTS,
    DispatchLease,
)


def dispatch_lease_metric_lines(dispatcher: object) -> list[str]:
    """Tick durations, lease acquisitions and losses, and the held gauge."""

    lease = getattr(dispatcher, "lease", None)
    if isinstance(lease, DispatchLease):
        buckets, count, total = lease.tick_seconds.snapshot()
        acquisitions = lease.acquisitions_total
        losses = dict(lease.losses_total)
        held = lease.held()
    else:
        buckets = [(bound, 0) for bound in DISPATCH_TICK_SECONDS_BUCKETS]
        count, total, acquisitions, losses, held = 0, 0.0, 0, {}, False
    return [
        "# HELP gpu_fault_workflow_dispatch_tick_seconds Wall time of each "
        "workflow dispatch tick this process ran holding the workflow-dispatch "
        "lease (or with the lease off), from taking the lease to the end of its "
        "batch; a tick that found the lease held elsewhere is not observed. A "
        "tick longer than a third of the dispatch lease (15 s as shipped) renews "
        "the lease at its checkpoints.",
        "# TYPE gpu_fault_workflow_dispatch_tick_seconds histogram",
        *(
            f'gpu_fault_workflow_dispatch_tick_seconds_bucket{{le="{bound:g}"}} '
            f"{cumulative}"
            for bound, cumulative in buckets
        ),
        f'gpu_fault_workflow_dispatch_tick_seconds_bucket{{le="+Inf"}} {count}',
        f"gpu_fault_workflow_dispatch_tick_seconds_sum {total:.6f}",
        f"gpu_fault_workflow_dispatch_tick_seconds_count {count}",
        "# HELP gpu_fault_workflow_dispatch_lease_acquisitions_total Times this "
        "process took the workflow-dispatch lease under a new epoch: from no "
        "holder, from a holder whose lease had lapsed, or back after its own "
        "lapsed. One steady holder adds nothing; a region-wide rate above zero "
        "outside a roll is the lease changing hands.",
        "# TYPE gpu_fault_workflow_dispatch_lease_acquisitions_total counter",
        f"gpu_fault_workflow_dispatch_lease_acquisitions_total {acquisitions}",
        "# HELP gpu_fault_workflow_dispatch_lease_losses_total Times this process "
        "found the workflow-dispatch lease it held taken by another process, by "
        "when it noticed: mid_tick (a checkpoint inside its tick; it started "
        "nothing more and left its queued rows PENDING) or tick_start (it was "
        "away longer than the lease).",
        "# TYPE gpu_fault_workflow_dispatch_lease_losses_total counter",
        *(
            f'gpu_fault_workflow_dispatch_lease_losses_total{{when="{moment}"}} '
            f"{losses.get(moment, 0)}"
            for moment in LOSS_MOMENTS
        ),
        "# HELP gpu_fault_workflow_dispatch_lease_held 1 while this process holds "
        "the workflow-dispatch lease by its own clock, else 0 (0 with the lease "
        "off too). A Pod's value is its number of holders; the region's sum is 1 "
        "while one process dispatches.",
        "# TYPE gpu_fault_workflow_dispatch_lease_held gauge",
        f"gpu_fault_workflow_dispatch_lease_held {1 if held else 0}",
    ]
