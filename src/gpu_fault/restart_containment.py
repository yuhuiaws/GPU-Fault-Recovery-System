"""The containment proof a passive restart carries to the data plane.

A passive recovery workflow (``no-hardware-evidence:RESTART``) has one step,
``RESTART_WORKLOAD``; the STOP that contained the failed attempt ran in the
predecessor containment workflow. The data-plane ownership guard binds every
new action to a contained STOP receipt, so the control plane signs the
predecessor's receipt into the restart authorization when it dispatches the
restart. Kept out of ``gpu_fault.models`` (at its size ceiling) and free of
project imports so both layers can import it without a cycle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

STOP_OWNERSHIP_RECEIPT_KEY = "stop_ownership_receipt_v1"


class RestartContainmentProof(BaseModel):
    """The predecessor containment's contained STOP receipt, as signed.

    ``workflow_id``/``incident_id`` name the containment workflow and incident
    the receipt belongs to; ``receipt`` is the receipt's own JSON form and is
    re-validated by the guard that reads it.
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1)
    incident_id: str = Field(min_length=1)
    receipt: dict[str, Any]
