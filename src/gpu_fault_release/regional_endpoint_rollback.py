"""Capture and restore the control-plane endpoint so rollback can compensate it.

The endpoint is two mutations, not one: ``apply_control_plane_nlb`` applies the
NLB Service from the candidate checkout, then ``ensure_control_plane_dns``
UPSERTs the Route53 CNAME that every Agent resolves. Automatic rollback used to
refuse any release whose diff touched either, because neither previous form was
recoverable -- the Service only existed as a file in the previous checkout, and
the record had already been overwritten by the time anything wanted it back.

Both are captured here before the endpoint component runs. The Service is
captured the same way the ADOT collector is (``regional_manifest_snapshot``);
the record is captured whole, so an alias target is restored as an alias target.

Restore runs in the reverse order of apply, and the ordering is the part that
matters:

* when the previous release had the Service, the Service is restored first and
  the record second, so the record is never pointed at a load balancer whose
  configuration has not been put back yet;
* when the candidate *created* the Service, rollback deletes it -- and then the
  record has to go first, because deleting the Service destroys the load
  balancer it published and would leave the record dangling.

The load balancer itself is not recreated: its name is pinned by the
``aws-load-balancer-name`` annotation, so restoring the Service reconciles the
same load balancer back in place rather than building a new one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from gpu_fault_release.regional_dns import (
    SERVICE_HOSTNAME_JSONPATH,
    SERVICE_HOSTNAME_POLL_SECONDS,
    normalized_record_name,
    read_dns_record,
    service_hostname_wait_args,
    submit_dns_change,
)
from gpu_fault_release.regional_manifest_snapshot import (
    apply_snapshot_objects,
    capture_declared_objects,
    declared_manifest_objects,
    delete_absent_objects,
    snapshot_parts,
)
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_rollout_wait import bounded_kubectl_wait

ROOT = Path(__file__).resolve().parents[2]
NLB_MANIFEST = ROOT / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
NLB_SERVICE = "gpu-fault-api-nlb"
ENDPOINT_RECORD_TYPE = "CNAME"
PUBLISHED_HOSTNAME_TIMEOUT_SECONDS = 300


def declared_endpoint_objects(manifest_text: str) -> tuple[dict[str, str], ...]:
    return declared_manifest_objects(manifest_text, label="NLB manifest")


def capture_endpoint_snapshot(release: Any) -> dict[str, Any]:
    """Read the endpoint state the endpoint component is about to mutate.

    A site without ``nlb`` configured has no endpoint component work to
    compensate -- ``apply_control_plane_nlb`` returns before touching anything --
    so the snapshot records that and stays empty. Capturing objects anyway would
    invite a rollback into deleting a Service this release never applied.
    """

    if not release.config.nlb:
        return {"configured": False}
    captured = capture_declared_objects(
        release,
        declared_endpoint_objects(NLB_MANIFEST.read_text(encoding="utf-8")),
        label="live NLB object",
    )
    hosted_zone_id = str(release.config.dns.hosted_zone_id or "")
    hostname = str(release.config.dns.hostname or "")
    if not hosted_zone_id or not hostname:
        # The forward path fails on exactly this, in
        # ``verify_control_plane_dns_prerequisites``. Failing here too keeps the
        # capture from recording an endpoint whose DNS half it cannot describe.
        raise ReleaseError(
            "NLB deployment requires dns.hosted_zone_id and dns.hostname"
        )
    return {
        "configured": True,
        **captured,
        "dns": {
            "hosted_zone_id": hosted_zone_id,
            "hostname": hostname,
            "record": read_dns_record(
                release,
                hosted_zone_id=hosted_zone_id,
                hostname=hostname,
                record_type=ENDPOINT_RECORD_TYPE,
            ),
        },
    }


def _dns_parts(snapshot: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    dns = snapshot.get("dns")
    if not isinstance(dns, dict):
        raise ReleaseError("previous endpoint snapshot is invalid")
    hosted_zone_id = str(dns.get("hosted_zone_id") or "")
    hostname = str(dns.get("hostname") or "")
    record = dns.get("record")
    if not hosted_zone_id or not hostname:
        raise ReleaseError("previous endpoint snapshot is invalid")
    if record is not None and not isinstance(record, dict):
        raise ReleaseError("previous endpoint snapshot is invalid")
    return hosted_zone_id, hostname, record


def _restore_dns_record(
    release: Any,
    *,
    hosted_zone_id: str,
    hostname: str,
    record: dict[str, Any] | None,
) -> None:
    if record is not None:
        submit_dns_change(
            release,
            [{"Action": "UPSERT", "ResourceRecordSet": record}],
            hosted_zone_id=hosted_zone_id,
        )
        return
    # Nothing was there before this release, so the compensation is a deletion.
    # Route53 only deletes a record set that is described exactly, and the
    # candidate's own values are the ones that got written -- so the record is
    # re-read rather than reconstructed from the release configuration, which
    # may itself have been what changed.
    current = read_dns_record(
        release,
        hosted_zone_id=hosted_zone_id,
        hostname=hostname,
        record_type=ENDPOINT_RECORD_TYPE,
    )
    if current is None:
        return
    submit_dns_change(
        release,
        [{"Action": "DELETE", "ResourceRecordSet": current}],
        hosted_zone_id=hosted_zone_id,
    )


def _published_hostname(release: Any, namespace: str) -> str:
    """Read the load balancer hostname the Service publishes.

    Restoring an existing Service keeps its status, so this normally answers on
    the first read; the wait covers the case where the Service had to be created
    again and its controller has not published an address yet.
    """

    deadline = time.monotonic() + PUBLISHED_HOSTNAME_TIMEOUT_SECONDS
    while True:
        hostname = str(
            release.runner.run(
                release._cpu(
                    "-n",
                    namespace,
                    "get",
                    "service",
                    NLB_SERVICE,
                    "-o",
                    f"jsonpath={SERVICE_HOSTNAME_JSONPATH}",
                ),
                capture=True,
            )
        ).strip()
        if hostname or time.monotonic() >= deadline:
            return hostname
        # Wake when the controller publishes the address, not up to 5s later.
        bounded_kubectl_wait(
            release,
            service_hostname_wait_args(release, namespace, NLB_SERVICE),
            seconds=min(SERVICE_HOSTNAME_POLL_SECONDS, deadline - time.monotonic()),
        )


def _require_record_target(
    release: Any,
    namespace: str,
    record: dict[str, Any] | None,
) -> None:
    """Prove the restored record points at the restored Service's load balancer.

    Without this the rollback could report success while the record named a load
    balancer the restored Service no longer publishes -- Agents would resolve the
    endpoint and reach nothing, which is worse than the failed upgrade the
    rollback was compensating.
    """

    if record is None or getattr(release.runner, "dry_run", False):
        return
    values = [
        normalized_record_name(str(item.get("Value") or ""))
        for item in (record.get("ResourceRecords") or [])
        if isinstance(item, dict)
    ]
    if not values:
        # An alias record names its target by hosted zone rather than by value;
        # there is nothing to compare it against here.
        return
    published = normalized_record_name(_published_hostname(release, namespace))
    if not published:
        raise ReleaseError(
            f"restored {NLB_SERVICE} publishes no load balancer hostname"
        )
    if published not in values:
        raise ReleaseError(
            "restored endpoint record does not point at the restored "
            f"{NLB_SERVICE} load balancer"
        )


def restore_endpoint_snapshot(release: Any, snapshot: object) -> None:
    """Put the captured endpoint back, Service and record together."""

    if not isinstance(snapshot, dict) or "configured" not in snapshot:
        raise ReleaseError("previous endpoint snapshot is unavailable")
    if not snapshot["configured"]:
        return
    namespace, objects, absent = snapshot_parts(
        snapshot,
        label="previous endpoint snapshot",
    )
    hosted_zone_id, hostname, record = _dns_parts(snapshot)
    if objects:
        apply_snapshot_objects(release, objects)
        _restore_dns_record(
            release,
            hosted_zone_id=hosted_zone_id,
            hostname=hostname,
            record=record,
        )
        _require_record_target(release, namespace, record)
        return
    _restore_dns_record(
        release,
        hosted_zone_id=hosted_zone_id,
        hostname=hostname,
        record=record,
    )
    delete_absent_objects(release, namespace, absent)
