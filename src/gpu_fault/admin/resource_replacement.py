"""``gpu-fault-admin replace-resource``: the transition for a really replaced resource.

The installation registry never lets a recorded identity change: the deploy
rebuild refuses a row whose ``resource_id``, ``resource_arn`` or
``dependencies`` differ from the recorded one and the registry API answers 409.
When the physical resource behind a key really was replaced, the sync can then
never converge and the deletion verbs keep refusing: a recreated public subnet
(the rebuild gives it a new number beside the old row, so the NLB row's
dependencies change -- refused at ``aws/nlb (dependencies)``), a re-subscribed
SNS e-mail (the same key, a new subscription ARN). This verb is the supported
path, and the only one; nothing calls it automatically.

Plan (no ``--confirm``, read-only): for the named keys it reads the registry and
rebuilds it from the bootstrap state exactly as the deploy completion step does,
but without the named rows. The row the rebuild then puts under a key is its
replacement, and may differ from the recorded row in the physical resource only.
The transition must make the registry converge: the next deploy's rebuild over
the registry as it would then stand must change no recorded identity. Live
proofs, by the cleaner's own probes: the recorded resource is gone (deleted, or
for an association detached), the replacement exists and is this site's -- its
site tag, or for a subscription or an association its exact binding.

Execute (``--confirm REPLACE_RESOURCE`` and an approved ``--reference``): under
the site operation lock, refused while an uninstall or a remove-cluster is open,
it plans again, journals the attempt (PLANNED), sends it to the registry API --
one transaction in which each recorded row moves, unchanged but for its key and
terminal status, to ``retired/<key>/<digest>`` and its replacement takes the key
-- and journals COMMITTED. A rerun finishes an open attempt by sending it again:
the API accepts a transition it already stored. The next ``deploy`` (a NOOP one
too) or ``config`` re-roll then records a successful sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from gpu_fault.admin import operator_identity
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_cleanup import ResourceProbe
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import REFERENCE_PATTERN
from gpu_fault.installation_resources import (
    STALE_RECORD,
    InstallationResource,
    InstallationResourceOwnership,
    InstallationResourceReplacement,
    InstallationResourceReplacementRequest,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
    PHYSICAL_FIELDS,
    retired_resource_key,
    retired_status,
)

CONFIRMATION = "REPLACE_RESOURCE"
STATE_ROOT = "resource-replacement"
#: Reviewed: a replacement keeps the key's role, the cleaner can prove the old
#: resource gone and the new one this site's. Cluster, Aurora, OIDC-provider,
#: hosted-zone and identity-keyed rows never change this way.
REPLACEABLE_RESOURCE_TYPES = frozenset(
    {
        "acm_certificate",
        "ec2_route_table",
        "ec2_route_table_association",
        "ec2_subnet",
        "internet_gateway",
        "nlb",
        "nlb_listener",
        "nlb_target_group",
        "security_group",
        "sns_subscription",
        "sns_topic",
    }
)


@dataclass(frozen=True)
class ReplacementPlan:
    document: dict[str, Any]
    request: InstallationResourceReplacementRequest | None


def approved_reference(reference: str | None) -> str:
    normalized = (reference or "").strip()
    if not REFERENCE_PATTERN.fullmatch(normalized):
        raise BootstrapError(
            "replace-resource requires --reference, an approved change reference"
        )
    return normalized


def _sealed(
    site_id: str, rows: Sequence[InstallationResource]
) -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(site_id=site_id, resources=list(rows))
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def _row_refusal(key: str, row: InstallationResource | None) -> str | None:
    if row is None:
        return f"{key}: the registry records no such key"
    if row.resource_type not in REPLACEABLE_RESOURCE_TYPES:
        return f"{key}: replace-resource does not replace a {row.resource_type} row"
    if row.ownership is not InstallationResourceOwnership.CREATED:
        return (
            f"{key}: the row is {row.ownership.value}; only a resource this site "
            "created carries the ownership a replacement must prove"
        )
    if row.status is not InstallationResourceStatus.ACTIVE:
        return f"{key}: the row is {row.status.value}, not ACTIVE"
    return None


def _shape_refusal(
    row: InstallationResource, replacement: InstallationResource | None
) -> str | None:
    key = row.resource_key
    if replacement is None:
        return f"{key}: the deploy rebuild records nothing under it to replace it"
    changes = row.identity_changes(replacement)
    if not changes:
        return f"{key}: the rebuild records the same resource; nothing was replaced"
    if not set(changes) <= PHYSICAL_FIELDS:
        return (
            f"{key}: the rebuild changes more than its physical resource "
            f"({', '.join(changes)}); that is a defect to fix, not a replacement"
        )
    return None


def _proof(
    site: RenderedSite, row: InstallationResource, replacement: InstallationResource
) -> tuple[dict[str, Any], list[str]]:
    """The cleaner's probes: the recorded resource gone, the new one the site's."""

    probe, ownership = ResourceProbe(site), CleanupOwnership(site)
    key, refusals = row.resource_key, []
    proof: dict[str, Any] = {}
    for name, resource in (
        ("recorded_present", row),
        ("replacement_present", replacement),
    ):
        try:
            proof[name] = probe.exists(resource)
        except BootstrapError as exc:
            proof[name] = None
            refusals.append(f"{key}: {resource.resource_id} cannot be probed: {exc}")
    if proof["recorded_present"]:
        refusals.append(
            f"{key}: the recorded {row.resource_type} {row.resource_id} still "
            "exists -- it was not replaced, or is not gone yet"
        )
    if proof["replacement_present"] is False:
        refusals.append(f"{key}: the replacement {replacement.resource_id} is absent")
    proof["replacement_owned"] = False
    if proof["replacement_present"]:
        try:
            ownership.prove_owned(replacement)
            proof["replacement_owned"] = True
        except BootstrapError as exc:
            refusals.append(f"{key}: the replacement is not proven this site's: {exc}")
    return proof, refusals


def _retired(
    row: InstallationResource,
    replacement: InstallationResource,
    *,
    reference: str,
    now: datetime,
) -> InstallationResource:
    return InstallationResource.model_validate(
        {
            **row.model_dump(),
            "resource_key": retired_resource_key(row),
            "status": retired_status(row),
            "updated_at": now,
            "error": None,
            "attributes": {
                **row.attributes,
                "replaced_by": replacement.resource_id,
                "replacement_reference": reference,
            },
        }
    )


def _dependency_hints(
    existing: InstallationResourceSnapshot, current: InstallationResourceSnapshot
) -> dict[str, list[str]]:
    """For a conflict in ``dependencies`` alone: the recorded ones no longer built.

    A recreated subnet surfaces as ``aws/nlb (dependencies)``; the key to replace
    is the dependency the rebuild stopped referencing.
    """

    recorded = {item.resource_key: item for item in existing.resources}
    return {
        item.resource_key: sorted(
            set(recorded[item.resource_key].dependencies) - set(item.dependencies)
        )
        for item in current.resources
        if item.resource_key in recorded
        and recorded[item.resource_key].identity_changes(item) == ("dependencies",)
    }


def _summary(row: InstallationResource) -> dict[str, Any]:
    return {"resource_id": row.resource_id, "resource_arn": row.resource_arn}


def plan_replacement(
    site: RenderedSite,
    keys: Sequence[str],
    *,
    reference: str,
    registry_path: Path,
    now: datetime,
) -> ReplacementPlan:
    """The read-only plan; ``request`` is ``None`` unless every proof holds."""

    site_id = site.registry_site_id
    existing = registry.fetch_installation_resource_registry(site, output=registry_path)
    bootstrap = registry.find_bootstrap_state(site)
    runtime = registry.discover_runtime_resources(site)

    def rebuild(rows: Sequence[InstallationResource]) -> InstallationResourceSnapshot:
        return registry.build_installation_snapshot(
            site, bootstrap, runtime, existing=_sealed(site_id, rows)
        )

    current = rebuild(existing.resources)
    recorded = {item.resource_key: item for item in existing.resources}
    refusals = [
        refusal for key in keys if (refusal := _row_refusal(key, recorded.get(key)))
    ]
    rows = [
        recorded[key] for key in keys if _row_refusal(key, recorded.get(key)) is None
    ]
    remaining = [item for item in existing.resources if item.resource_key not in keys]
    built = {item.resource_key: item for item in rebuild(remaining).resources}
    transitions: list[dict[str, Any]] = []
    replacements: list[InstallationResourceReplacement] = []
    for row in rows:
        replacement = built.get(row.resource_key)
        shape = _shape_refusal(row, replacement)
        if shape is not None or replacement is None:
            refusals.append(str(shape))
            continue
        proof, failures = _proof(site, row, replacement)
        refusals.extend(failures)
        try:
            retired = _retired(row, replacement, reference=reference, now=now)
            transition = InstallationResourceReplacement(
                recorded=row, retired=retired, replacement=replacement
            )
        except ValueError as exc:
            refusals.append(f"{row.resource_key}: the transition is invalid: {exc}")
            continue
        replacements.append(transition)
        transitions.append(
            {
                "resource_key": row.resource_key,
                "resource_type": row.resource_type,
                "recorded": _summary(row),
                "replacement": _summary(replacement),
                "changed": list(row.identity_changes(replacement)),
                "retired_key": retired.resource_key,
                "proof": proof,
            }
        )
    after = [
        *remaining,
        *(item.retired for item in replacements),
        *(item.replacement for item in replacements),
    ]
    left = registry.recorded_identity_conflicts(rebuild(after), _sealed(site_id, after))
    if replacements and left:
        refusals.append(
            "with this transition the next deploy rebuild would still change "
            "recorded identities: "
            + "; ".join(left)
            + " -- name every replaced key in one transition"
        )
    identity = json.dumps(
        [item | {"proof": None} for item in transitions], sort_keys=True
    )
    document = {
        "mode": "plan",
        "site_id": site_id,
        "reference": reference,
        "resource_keys": list(keys),
        "conflicts_before": registry.recorded_identity_conflicts(current, existing),
        "dependency_hints": _dependency_hints(existing, current),
        "transitions": transitions,
        "remaining_conflicts": left,
        "refusals": refusals,
        "plan_sha256": hashlib.sha256(f"{site_id}\n{identity}".encode()).hexdigest(),
    }
    request = (
        InstallationResourceReplacementRequest(
            site_id=site_id, replacements=replacements
        )
        if replacements and not refusals
        else None
    )
    return ReplacementPlan(document=document, request=request)


def refuse_during_deletion(state_dir: Path) -> None:
    """An open uninstall or removal journal binds the registry it exported."""

    journals = [state_dir / "uninstall" / "state.json"]
    journals += sorted((state_dir / "remove-cluster").glob("*/state.json"))
    for path in journals:
        if not path.is_file():
            continue
        try:
            phase = str(json.loads(path.read_text(encoding="utf-8")).get("phase"))
        except (OSError, ValueError, AttributeError):
            phase = "unreadable"
        if phase != "COMPLETED":
            raise BootstrapError(
                f"replace-resource refuses while {path.parent.relative_to(state_dir)} "
                f"is open (phase {phase}): finish it first -- its deletion list is "
                "the registry it exported"
            )


def _open_attempt(directory: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted(directory.glob("*/journal.json")):
        journal = json.loads(path.read_text(encoding="utf-8"))
        if journal.get("phase") == "PLANNED":
            return path, journal
    return None


def _commit(
    site: RenderedSite,
    path: Path,
    journal: dict[str, Any],
    request: InstallationResourceReplacementRequest,
) -> dict[str, Any]:
    try:
        registry.replace_installation_resources(site, request)
    except registry.InstallationRegistryConflict as exc:
        if exc.reason != STALE_RECORD:
            raise
        # All or nothing: a stale refusal means nothing of the attempt landed.
        journal.update(phase="ABANDONED", abandoned_reason=str(exc))
        write_json_atomic(path, journal)
        raise BootstrapError(
            f"replace-resource attempt {journal['attempt_id']} no longer applies "
            f"and nothing of it landed: {exc}; plan it again"
        ) from exc
    journal.update(
        phase="COMMITTED", committed_at=datetime.now(timezone.utc).isoformat()
    )
    write_json_atomic(path, journal)
    warnings = []
    try:
        registry.fetch_installation_resource_registry(site)
    except BootstrapError as exc:
        warnings.append(f"the local registry dump was not refreshed: {exc}")
    return {
        "mode": "execute",
        "status": "COMMITTED",
        "attempt_id": journal["attempt_id"],
        "journal": str(path),
        "reference": journal["reference"],
        "replaced": [item.recorded.resource_key for item in request.replacements],
        "retired": [item.retired.resource_key for item in request.replacements],
        "warnings": warnings,
        "next": (
            f"`gpu-fault-admin deploy --state-dir {site.source.parent}` or the next "
            "`gpu-fault-admin config` re-roll synchronizes the registry and records "
            "the success uninstall and remove-cluster require"
        ),
    }


def run_replace_resource(
    site: RenderedSite,
    keys: Sequence[str],
    *,
    reference: str,
    now: datetime | None = None,
    identity: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Execute the transition and journal it (the caller holds the lock)."""

    state_dir = site.source.parent
    refuse_during_deletion(state_dir)
    directory = state_dir / STATE_ROOT
    opened = _open_attempt(directory)
    if opened is not None:
        open_path, open_journal = opened
        request = InstallationResourceReplacementRequest.model_validate(
            open_journal["request"]
        )
        planned = sorted(item.recorded.resource_key for item in request.replacements)
        if request.site_id != site.registry_site_id or planned != sorted(keys):
            raise BootstrapError(
                f"replace-resource attempt {open_journal['attempt_id']} for "
                f"{', '.join(planned)} is open: rerun with exactly those keys"
            )
        return _commit(site, open_path, open_journal, request) | {"resumed": True}
    recorded_at = now or datetime.now(timezone.utc)
    attempt = f"{recorded_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
    path = directory / attempt / "journal.json"
    path.parent.mkdir(mode=0o700, parents=True)
    plan = plan_replacement(
        site,
        keys,
        reference=reference,
        registry_path=path.parent / "installation-resources-before.json",
        now=recorded_at,
    )
    resolve = identity or (
        lambda: operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
    )
    journal: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": attempt,
        "phase": "PLANNED" if plan.request is not None else "REFUSED",
        "site_id": site.registry_site_id,
        "reference": reference,
        "operator": resolve(),
        "planned_at": recorded_at.isoformat(),
        "plan": plan.document,
    }
    if plan.request is None:
        write_json_atomic(path, journal)
        raise BootstrapError(
            "replace-resource refused: " + "; ".join(plan.document["refusals"])
        )
    journal["request"] = plan.request.model_dump(mode="json")
    write_json_atomic(path, journal)
    return _commit(site, path, journal, plan.request)


def add_replace_resource_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin replace-resource``; the CLI passes site options."""

    command = commands.add_parser(
        "replace-resource",
        usage=(
            "gpu-fault-admin replace-resource --state-dir STATE_DIR --resource-key "
            "KEY [--resource-key KEY ...] --reference CHANGE_ID "
            f"[--confirm {CONFIRMATION}]"
        ),
        help=(
            "retire the installation registry row of a resource that was really "
            "replaced and record its replacement: without --confirm prints the "
            "plan and its live proofs (old resource gone, new one this site's) and "
            f"changes nothing; --confirm {CONFIRMATION} executes it, journaled"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--resource-key",
        action="append",
        required=True,
        metavar="KEY",
        help="registry key whose physical resource was replaced (repeat per key)",
    )
    command.add_argument(
        "--reference",
        required=True,
        metavar="CHANGE_ID",
        help="approved change reference recorded on the retired rows and journal",
    )
    command.add_argument(
        "--confirm",
        default=None,
        metavar="TOKEN",
        help=f"execute requires exactly {CONFIRMATION}; without it, plan only",
    )


def run_replace_resource_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    reference = approved_reference(arguments.reference)
    keys = sorted(set(arguments.resource_key))
    confirmation = getattr(arguments, "confirm", None)
    if confirmation is None:
        with tempfile.TemporaryDirectory() as scratch:
            plan = plan_replacement(
                site,
                keys,
                reference=reference,
                registry_path=Path(scratch) / "installation-resources.json",
                now=datetime.now(timezone.utc),
            )
        print(json.dumps(plan.document, indent=2, sort_keys=True))
        return 1 if plan.document["refusals"] else 0
    if confirmation != CONFIRMATION:
        raise BootstrapError(f"replace-resource requires --confirm {CONFIRMATION}")
    with administrator_operation_lock(site.source.parent):
        site = reload_site_for_mutation(site)
        result = run_replace_resource(site, keys, reference=reference)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
