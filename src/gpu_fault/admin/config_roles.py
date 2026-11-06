"""Which CPU roles read which part of the administrator config.

A role's digest (``AdminConfig.role_payload``/``role_sha256``) holds exactly
the entries that role reads, as declared in ``ROLE_INPUT_READERS``. The digest
is what selects the role in a config release
(``regional_release_diff.control_plane_role_targets``) and what its Pod
template is stamped with (``gpu-fault.io/role-config-sha256``), so a change
re-applies and rolls every role that reads it, and no other. A reader is a
role whose rendered environment carries the entry
(``deploy/control-plane/tools/render_control_plane_role_split.py``);
``tests/admin/test_admin_config_role_readers.py`` renders every administrator
field and fails when the roles whose render changes are not exactly the roles
whose digest moves.

A runtime leaf: ``config`` imports it, and it names ``AdminConfig`` for typing
only.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from gpu_fault.admin.config_parser import AdminConfigError

if TYPE_CHECKING:
    from gpu_fault.admin.config import AdminConfig

ADMIN_CONFIG_ROLES = ("ingress", "worker", "spool")
_EVERY_ROLE = frozenset(ADMIN_CONFIG_ROLES)

#: Role digest entry -> the roles that read it. ``aurora`` and
#: ``cpu_control_plane`` are in no entry: the deploy host applies them, no
#: role renders them.
ROLE_INPUT_READERS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # The tuning sections and the declared topology (with the fault
        # reserve derived from it) are rendered into every role's ConfigMaps.
        "processor": _EVERY_ROLE,
        "workflow": _EVERY_ROLE,
        "notification_delivery": _EVERY_ROLE,
        "evidence": _EVERY_ROLE,
        "largest_cluster_node_count": _EVERY_ROLE,
        "managed_node_count": _EVERY_ROLE,
        # GPU_FAULT_TELEMETRY_SPOOL, and the ingress pool sizes it selects.
        "telemetry_spool_enabled": frozenset({"ingress"}),
        # GPU_FAULT_TELEMETRY_SPOOL_MAX_CLUSTER_DEPTH: the ingress's spool
        # admission, never the spool worker. Present only off its default.
        "telemetry_spool_max_cluster_depth": frozenset({"ingress"}),
        # The control-worker replicas and the queue coordination sized by
        # them; the two remediation constants the worker renders, with any
        # retired limit a record still carries (kept so its digests verify).
        "control_worker_replicas": frozenset({"worker"}),
        "remediation": frozenset({"worker"}),
        # The spool switch and the spool-worker replicas.
        "telemetry_spool": frozenset({"spool"}),
    }
)
#: The one entry earlier releases digested into a role that does not read it:
#: the depth sat in the spool worker's ``telemetry_spool`` section.
SPOOL_DEPTH_ENTRY = "telemetry_spool_max_cluster_depth"


def role_entries(config: AdminConfig, *, node_counts: bool = True) -> dict[str, object]:
    """Every role digest entry of ``config``, keyed as ``ROLE_INPUT_READERS``.

    ``node_counts=False`` reproduces the digests a release before the node
    counts recorded, for verifying such a persisted state.
    """

    capacity = config.capacity
    spool = capacity.telemetry_spool
    entries: dict[str, object] = {
        "processor": config.processor.as_dict(),
        "workflow": config.workflow.as_dict(),
        "notification_delivery": config.notification_delivery.as_dict(),
        "evidence": config.evidence.as_dict(),
        "telemetry_spool_enabled": spool.enabled,
        "control_worker_replicas": capacity.control_worker_replicas,
        "remediation": capacity.remediation.as_dict(),
        "telemetry_spool": {"enabled": spool.enabled, "replicas": spool.replicas},
    }
    # Off its default only, as in the record, so every live digest stays.
    if "max_cluster_depth" in spool.as_dict():
        entries[SPOOL_DEPTH_ENTRY] = spool.max_cluster_depth
    if node_counts:
        entries.update(capacity.node_counts())
    return entries


def role_inputs(entries: Mapping[str, object], role: str) -> dict[str, object]:
    """The entries ``role`` reads: its digest payload."""

    if role not in _EVERY_ROLE:
        raise AdminConfigError(f"unknown control-plane role: {role}")
    undeclared = sorted(set(entries) - set(ROLE_INPUT_READERS))
    if undeclared:
        raise AdminConfigError(
            "admin config role entries have no declared reader: "
            + ", ".join(undeclared)
        )
    return {
        name: value
        for name, value in entries.items()
        if role in ROLE_INPUT_READERS[name]
    }


def earlier_role_inputs(entries: Mapping[str, object], role: str) -> dict[str, object]:
    """``role_inputs`` as the releases before ``SPOOL_DEPTH_ENTRY`` computed it.

    Only for verifying a record those releases digested; nothing renders it.
    """

    current = dict(entries)
    depth = current.pop(SPOOL_DEPTH_ENTRY, None)
    inputs = role_inputs(current, role)
    spool = inputs.get("telemetry_spool")
    if depth is not None and isinstance(spool, Mapping):
        inputs["telemetry_spool"] = {**spool, "max_cluster_depth": depth}
    return inputs
