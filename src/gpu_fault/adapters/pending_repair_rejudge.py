"""After a reset, a pending row remap or page retirement is re-judged on newer data (b209).

``row_remap_pending`` (Ampere and later) and ``retired_pages_pending``
(earlier parts) are the pending-repair flags of the GPU memory policy: a
repair recorded and not yet applied. ``gpu_memory_policy.memory_decision``
grades them WARNING ``RESET_GPU`` on a supported architecture, so the chain
resets the GPU and its VALIDATE_GPU decides whether the reset applied the
repair.

Evidence -- NVIDIA, docs version 615, every quote checked against the live
page:

* GPU Memory Error Management, "Error Recovery and Response Flags", item
  "Row-remapping pending flag" (page "Last updated on Sep 09, 2026"):
  https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/error-recovery-and-response-flags.html#error-recovery-and-response-flags
  -- "This flag indicates that row-remapping will happen at the next GPU
  reset." The same manual, "Row Remapping"
  (https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html#row-remapping):
  "The process of row-remapping requires a GPU reset to take effect"; its
  Table 2 "Page Retirement vs. Row-Remapping", row "Application of pending
  changes": "GPU reset is required." (legacy page retirement: "Needed a
  kernel module reload or driver re-initialization or GPU reset").
* Dynamic Page Retirement, "Visibility" > "nvidia-smi" (page "Last updated on
  Sep 09, 2026"):
  https://docs.nvidia.com/deploy/dynamic-page-retirement/visibility.html#nvidia-smi
  -- "The nvidia-smi “Pending Page Blacklist” field indicates whether a page
  has been recently retired and will be blacklisted on the next system reboot
  or driver load." Its "Blacklisting and ECC Error Recovery" > "Reattaching
  the GPU"
  (https://docs.nvidia.com/deploy/dynamic-page-retirement/blacklisting-and-ecc-error-recovery.html#reattaching-the-gpu)
  lists the ways to apply it, "Reset the GPUs" and "Reboot the machine (or
  VM)" among them.
* GPU Debug Guidelines, "GPU Node Triage", "Understanding Xid Messages", row
  63 "A100: Row-remapping recording event" (page "Last updated on Oct 04,
  2026"):
  https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages
  -- "until there is a convenient time to reset the GPU (refer to GPU reset
  capabilities/limitations section below) or reboot for row remapping to
  activate."

So once this workflow has reset the GPU or rebooted the node, a pending-flag
finding whose current state was observed at or before that step's completion
describes the GPU *before* the reset and cannot say whether the reset applied
the repair. VALIDATE_GPU failed it at once all the same: the warning grace and
b202's wait for newer GPU metrics cover only the thermal, correctable-memory
and power WARNING families. Latent -- found by code reading during the
GF-GENERAL-LIVE-024 analysis, never seen live: the freshness gates in front of
the finding check (the GPU_METRICS collector status and every targeted GPU's
metric presence must postdate the reset) and the collector's whole-scrape
deliveries keep it from biting today. It bites whenever post-reset GPU data
passes those gates without re-judging the flag: a scrape without the flag's
field, the post-reset batch's finding-state write not visible yet, a caller of
the finding check alone.

Now, while every blocking finding is a pending-repair flag whose current state
is not newer than the completion of the last GPU reset or node reboot this
workflow ran on the node (for RESET_GPU: on the finding's GPU), VALIDATE_GPU
keeps WAITING on b202's mechanism -- the same ``awaiting_newer_gpu_metrics``
evidence with ``pending_repair_reset`` (operation, step index, completion) in
place of ``grace_elapsed_at``, the vendor references, and the executor's
per-step cap (``step_bounds.bounded_waiting_outcome``) as the only bound. The
first GPU metrics batch newer than the reset decides: the flag cleared clears
the finding and the step passes; the flag still set makes the ingest re-grade
the finding at that batch's time, newer than the reset, and the existing
failure path fails it (``active_gpu_health_findings``, then the escalation
ladder). No grace: the reset, not a timer, is the barrier.

Not counted as applying the repair: RECOVER_DRAIN_P2P (it resets only the GPUs
whose driver recovery action asks for it and can succeed with
``reset_skipped``), REMEDIATE_DRIVER and UPDATE_SOFTWARE_FIRMWARE (a
site-configured command the product cannot see reset the GPU) and REPLACE_NODE
(other hardware). With no reset in the workflow the flag fails at once as
before: nothing short of a reset applies it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepStatus,
)

#: The pending-repair flags ``gpu_memory_policy.memory_decision`` grades
#: "GPU memory repair pending; reset in a service window".
PENDING_REPAIR_METRICS = frozenset({"row_remap_pending", "retired_pages_pending"})
#: The steps that apply a pending repair: a GPU reset or a node reboot.
PENDING_REPAIR_RESET_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
    }
)
NVIDIA_PENDING_REPAIR_DOCS_VERSION = "615"
NVIDIA_ROW_REMAP_PENDING_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
    "error-recovery-and-response-flags.html#error-recovery-and-response-flags"
)
NVIDIA_ROW_REMAP_PENDING_QUOTE = (
    "This flag indicates that row-remapping will happen at the next GPU reset."
)
NVIDIA_ROW_REMAP_RESET_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html"
    "#row-remapping"
)
NVIDIA_ROW_REMAP_RESET_QUOTE = (
    "The process of row-remapping requires a GPU reset to take effect"
)
NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE = (
    "https://docs.nvidia.com/deploy/dynamic-page-retirement/visibility.html#nvidia-smi"
)
NVIDIA_PAGE_RETIREMENT_PENDING_QUOTE = (
    "The nvidia-smi “Pending Page Blacklist” field indicates whether a page has "
    "been recently retired and will be blacklisted on the next system reboot or "
    "driver load."
)
NVIDIA_PAGE_RETIREMENT_REATTACH_REFERENCE = (
    "https://docs.nvidia.com/deploy/dynamic-page-retirement/"
    "blacklisting-and-ecc-error-recovery.html#reattaching-the-gpu"
)
NVIDIA_GPU_DEBUG_ROW_REMAP_REFERENCE = (
    "https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html"
    "#understanding-xid-messages"
)
NVIDIA_GPU_DEBUG_ROW_REMAP_QUOTE = (
    "until there is a convenient time to reset the GPU (refer to GPU reset "
    "capabilities/limitations section below) or reboot for row remapping to "
    "activate."
)
#: What the wait on each flag cites, and how its reason names the repair.
PENDING_REPAIR_REFERENCES = {
    "row_remap_pending": NVIDIA_ROW_REMAP_PENDING_REFERENCE,
    "retired_pages_pending": NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE,
}
PENDING_REPAIR_NAMES = {
    "row_remap_pending": "row remap",
    "retired_pages_pending": "page retirement",
}


def pending_repair_reset(
    context: Any, node_id: str, findings: Sequence[Any]
) -> WorkflowStepExecution | None:
    """The last GPU reset or node reboot this workflow completed before this
    step that reached every GPU of ``findings`` on ``node_id``; ``None`` when
    there is none or a finding is not a pending-repair flag."""

    if not findings or any(
        getattr(finding, "canonical_name", None) not in PENDING_REPAIR_METRICS
        for finding in findings
    ):
        return None
    resets = [
        execution
        for execution in context.workflow.step_executions
        if execution.step_index < context.step_index
        and execution.status is WorkflowStepStatus.SUCCEEDED
        and execution.operation in PENDING_REPAIR_RESET_OPERATIONS
        and _reaches(context.workflow, execution, node_id, findings)
    ]
    return max(resets, key=lambda execution: execution.updated_at, default=None)


def _reaches(
    workflow: Any,
    execution: WorkflowStepExecution,
    node_id: str,
    findings: Sequence[Any],
) -> bool:
    """Did the reset reach the node -- for RESET_GPU naming its GPUs, every
    finding's GPU? A record that cannot be tied to its step counts: a wrong yes
    costs a wait the step cap bounds, a wrong no a verdict on the GPU as it was
    before the reset."""

    steps = (
        workflow.safety_steps
        if execution.phase == "safety"
        else workflow.official_steps
    )
    step = steps[execution.step_index] if execution.step_index < len(steps) else None
    if step is None or step.operation is not execution.operation:
        return True
    if step.node_ids and node_id not in step.node_ids:
        return False
    if execution.operation is not WorkflowOperation.RESET_GPU or not step.gpu_uuids:
        return True
    return all(
        getattr(finding, "gpu_uuid", None) in {None, *step.gpu_uuids}
        for finding in findings
    )


def awaited_findings_text(entries: Sequence[dict[str, Any]]) -> str:
    """``finding <id> (observed <ts>[; the newest sample, <ts>, did not
    re-judge it])`` for each finding a wait on newer GPU metrics names: the
    findings' part of both waits' sentence (b202's after the grace, this one
    after the reset)."""

    named = []
    for entry in entries:
        text = f"finding {entry['finding_id']} (observed {entry['observed_at']}"
        newest = entry.get("newest_sample_observed_at")
        if newest:
            text += f"; the newest sample, {newest}, did not re-judge it"
        named.append(text + ")")
    return ", ".join(named)


def pending_repair_wait(
    entries: Sequence[dict[str, Any]],
    reset: WorkflowStepExecution,
    step_cap_at: datetime,
) -> dict[str, Any]:
    """The WAITING evidence of pending flags no batch newer than ``reset`` has
    re-judged: b202's ``awaiting_newer_gpu_metrics`` keys and sentence, the
    reset in place of the grace, and the vendor references."""

    named = awaited_findings_text(entries)
    flags = sorted({str(entry["canonical_name"]) for entry in entries})
    references = [PENDING_REPAIR_REFERENCES[flag] for flag in flags]
    repairs = " or ".join(PENDING_REPAIR_NAMES[flag] for flag in flags)
    completed = reset.updated_at.isoformat()
    return {
        "awaiting_newer_gpu_metrics": [entry["finding_id"] for entry in entries],
        "awaiting_newer_gpu_metrics_findings": list(entries),
        "pending_repair_reset": {
            "operation": reset.operation.value,
            "step_index": reset.step_index,
            "completed_at": completed,
        },
        "step_cap_at": step_cap_at.isoformat(),
        "policy_references": references,
        "reason": (
            f"awaiting a GPU metrics batch newer than {named} to re-judge it; "
            f"{reset.operation.value} completed {completed} and NVIDIA applies "
            f"a pending {repairs} at the next GPU reset ({', '.join(references)}); "
            f"step cap {step_cap_at.isoformat()}"
        ),
    }
