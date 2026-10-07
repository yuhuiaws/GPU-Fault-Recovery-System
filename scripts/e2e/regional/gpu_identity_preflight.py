"""Preflight rule shared by the XID-injecting destructive cases.

The kernel collector stamps the GPU product it discovered at start-up on every
XID it posts. After a reboot it can start before the driver answers and stamp
none; the control plane then reads the injected XID as "not applicable to
product UNKNOWN" and quarantines the node instead of running the case's
workflow (``eks-managed`` DESTR-015 attempt 10, after the DESTR-025 reboot of
one of its two nodes: one event EXECUTABLE, the other NOT_APPLICABLE, two
records). The control plane now falls back to the node's fresh GPU inventory
for the product, and that inventory is what this rule reads: it must exist, be
from the node's current boot, be younger than the control plane's freshness
bound and name exactly one product. Otherwise the case refuses to inject.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

#: The control plane's default ``GPU_FAULT_GPU_INVENTORY_MAX_AGE_SECONDS``: an
#: older inventory is not used to fill a product-less XID either.
GPU_IDENTITY_MAX_AGE_SECONDS = 180


def gpu_identity_errors(
    node: str,
    identity: dict[str, Any] | None,
    *,
    boot_id: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Why ``node``'s stored GPU identity cannot carry the case, if it cannot.

    ``identity`` is the Store probe's ``gpu_identity`` reading (the node's GPU
    inventory snapshot: ``product``, ``products``, ``device_count``,
    ``source_boot_id``, ``observed_at``); ``boot_id`` the node's current boot
    from the Kubernetes node snapshot.
    """

    if not identity:
        return [
            f"{node} has no GPU inventory identity in the store (its nvidia-smi "
            "inventory was never delivered); a product-less XID from it would be "
            "quarantined as NOT_APPLICABLE instead of starting the workflow"
        ]
    errors: list[str] = []
    if not identity.get("product"):
        errors.append(
            f"{node} GPU inventory identity names no single product: "
            f"products={identity.get('products')} devices={identity.get('device_count')}"
        )
    source_boot_id = identity.get("source_boot_id")
    if boot_id and source_boot_id and source_boot_id != boot_id:
        errors.append(
            f"{node} GPU inventory identity is from boot {source_boot_id} but the "
            f"node runs boot {boot_id}: its collectors have not re-discovered the "
            "GPUs since the reboot (restart gpu-fault-metrics-collector and "
            "gpu-fault-kernel-collector, or wait one inventory interval)"
        )
    observed_at = _parse_time(identity.get("observed_at"))
    if observed_at is None:
        errors.append(f"{node} GPU inventory identity has no readable observed_at")
        return errors
    age = ((now or datetime.now(timezone.utc)) - observed_at).total_seconds()
    if age > GPU_IDENTITY_MAX_AGE_SECONDS:
        errors.append(
            f"{node} GPU inventory identity is {age:.0f} s old (limit "
            f"{GPU_IDENTITY_MAX_AGE_SECONDS} s): the metrics collector is not "
            "delivering inventories, so the control plane could not fill a "
            "product-less XID from it"
        )
    return errors


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
