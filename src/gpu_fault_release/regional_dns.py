from __future__ import annotations

import fnmatch
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError, render_nlb_manifest

ROOT = Path(__file__).resolve().parents[2]


def _aws_json(
    release: Any,
    arguments: list[str],
    *,
    regional: bool = True,
    sensitive: bool = False,
) -> dict[str, Any]:
    """One `aws` call, deliberately outside the release's read snapshot.

    Unlike the administrator checks, nothing here asks the same question twice
    for the same answer. Most of these reads are the bodies of wait loops -- the
    NLB becoming active, its targets becoming healthy, a Route53 change reaching
    INSYNC -- and a cache that served the previous reading would turn "not
    converged yet" into a loop that can never end, or worse, into a pass. The
    two that are not loops, the hosted-zone and certificate checks in
    `verify_control_plane_dns_prerequisites`, run once each per release, so
    routing them through the snapshot would add a key and save nothing.
    """

    command = ["aws", *arguments]
    if regional:
        command.extend(["--region", release.config.aws_region])
    command.extend(["--output", "json"])
    return json.loads(
        release.runner.run(
            command,
            capture=True,
            sensitive=sensitive,
        )
    )


def _hostname_matches(pattern: str, hostname: str) -> bool:
    normalized_pattern = pattern.rstrip(".").lower()
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_pattern.startswith("*."):
        return fnmatch.fnmatchcase(
            normalized_hostname,
            normalized_pattern,
        ) and normalized_hostname.count(".") == normalized_pattern.count(".")
    return normalized_pattern == normalized_hostname


def verify_control_plane_dns_prerequisites(release: Any) -> None:
    if release.runner.dry_run:
        return
    hosted_zone_id = release.config.dns.hosted_zone_id
    hostname = release.config.dns.hostname
    certificate_arn = release.config.nlb.get("certificate_arn")
    if not all((hosted_zone_id, hostname, certificate_arn)):
        raise ReleaseError(
            "NLB deployment requires dns.hosted_zone_id, dns.hostname, "
            "and nlb.certificate_arn"
        )

    zone_document = _aws_json(
        release,
        [
            "route53",
            "get-hosted-zone",
            "--id",
            str(hosted_zone_id),
        ],
        regional=False,
    )
    zone = zone_document.get("HostedZone") or {}
    actual_zone_id = str(zone.get("Id") or "").rsplit("/", 1)[-1]
    expected_zone_id = str(hosted_zone_id).rsplit("/", 1)[-1]
    if actual_zone_id != expected_zone_id:
        raise ReleaseError(
            f"Route53 hosted zone {hosted_zone_id} was not returned exactly once"
        )
    if not (zone.get("Config") or {}).get("PrivateZone"):
        raise ReleaseError(
            f"Route53 hosted zone {hosted_zone_id} must be a private hosted zone"
        )
    zone_name = str(zone.get("Name") or "").rstrip(".").lower()
    normalized_hostname = str(hostname).rstrip(".").lower()
    if not zone_name or (
        normalized_hostname != zone_name
        and not normalized_hostname.endswith(f".{zone_name}")
    ):
        raise ReleaseError(
            f"DNS hostname {hostname} is not inside hosted zone {zone_name or '<empty>'}"
        )

    certificate = (
        _aws_json(
            release,
            [
                "acm",
                "describe-certificate",
                "--certificate-arn",
                str(certificate_arn),
            ],
        ).get("Certificate")
        or {}
    )
    if certificate.get("Status") != "ISSUED":
        raise ReleaseError(
            f"ACM certificate is {certificate.get('Status')}, expected ISSUED"
        )
    names = {
        str(value)
        for value in (
            certificate.get("SubjectAlternativeNames")
            or [certificate.get("DomainName")]
        )
        if value
    }
    if not any(_hostname_matches(pattern, normalized_hostname) for pattern in names):
        raise ReleaseError(f"ACM certificate does not cover DNS hostname {hostname}")
    raw_expiry = certificate.get("NotAfter")
    if not raw_expiry:
        raise ReleaseError("ACM certificate has no NotAfter")
    expiry = datetime.fromisoformat(str(raw_expiry).replace("Z", "+00:00"))
    if expiry <= datetime.now(timezone.utc):
        raise ReleaseError("ACM certificate is expired")


def _wait_service_hostname(release: Any) -> str:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        hostname = release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "service",
                "gpu-fault-api-nlb",
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].hostname}",
            ),
            capture=True,
        )
        if hostname:
            return hostname
        if release.runner.dry_run:
            return ""
        time.sleep(5)
    raise ReleaseError("NLB Service did not publish a hostname within 600s")


def _wait_nlb_active(release: Any, hostname: str) -> str:
    deadline = time.monotonic() + 900
    load_balancer_arn = ""
    last_state = "not-found"
    last_listener = "not-found"
    name = release.config.nlb.get("name") or (
        f"gpu-fault-regional-{release.config.aws_region}"
    )
    while time.monotonic() < deadline:
        document = _aws_json(
            release,
            [
                "elbv2",
                "describe-load-balancers",
                "--names",
                name,
            ],
        )
        load_balancers = document.get("LoadBalancers", [])
        if load_balancers:
            load_balancer = load_balancers[0]
            load_balancer_arn = str(load_balancer["LoadBalancerArn"])
            last_state = str(
                (load_balancer.get("State") or {}).get("Code") or "unknown"
            )
            if last_state == "active":
                if str(load_balancer.get("DNSName") or "").rstrip(
                    "."
                ) != hostname.rstrip("."):
                    raise ReleaseError(
                        "NLB Service hostname does not match the AWS NLB DNS name"
                    )
                listeners = _aws_json(
                    release,
                    [
                        "elbv2",
                        "describe-listeners",
                        "--load-balancer-arn",
                        load_balancer_arn,
                    ],
                ).get("Listeners", [])
                tls_listeners = [
                    item
                    for item in listeners
                    if item.get("Protocol") == "TLS" and item.get("Port") == 443
                ]
                if len(tls_listeners) == 1:
                    certificates = {
                        item.get("CertificateArn")
                        for item in tls_listeners[0].get("Certificates", [])
                        if item.get("CertificateArn")
                    }
                    if release.config.nlb["certificate_arn"] in certificates:
                        return load_balancer_arn
                    last_listener = "TLS/443 certificates=" + ",".join(
                        sorted(certificates)
                    )
                else:
                    last_listener = json.dumps(
                        [
                            {
                                "protocol": item.get("Protocol"),
                                "port": item.get("Port"),
                            }
                            for item in listeners
                        ],
                        sort_keys=True,
                    )
        if release.runner.dry_run:
            return load_balancer_arn
        time.sleep(5)
    raise ReleaseError(
        "NLB did not become active with the configured TLS listener within "
        f"900s; last state={last_state}; last listener={last_listener}"
    )


def _wait_raw_nlb_dns(release: Any, hostname: str) -> None:
    deadline = time.monotonic() + 300
    last_error = "no address returned"
    while time.monotonic() < deadline:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(hostname, 443)}
            if addresses:
                return
            last_error = "no address returned"
        except socket.gaierror as exc:
            last_error = str(exc)
        if release.runner.dry_run:
            return
        time.sleep(5)
    raise ReleaseError(
        f"raw NLB DNS did not resolve within 300s: {hostname}: {last_error}"
    )


def _wait_targets_healthy(release: Any, load_balancer_arn: str) -> None:
    deadline = time.monotonic() + 600
    last_states: dict[str, list[str]] = {}
    raw_expected = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            "gpu-fault-api-ha",
            "-o",
            "jsonpath={.spec.replicas}",
        ),
        capture=True,
    )
    try:
        expected_targets = max(1, int(raw_expected))
    except (TypeError, ValueError) as exc:
        raise ReleaseError(
            f"cannot determine expected NLB target count: {raw_expected!r}"
        ) from exc
    while time.monotonic() < deadline:
        groups = _aws_json(
            release,
            [
                "elbv2",
                "describe-target-groups",
                "--load-balancer-arn",
                load_balancer_arn,
            ],
        ).get("TargetGroups", [])
        all_groups_healthy = bool(groups)
        current_states: dict[str, list[str]] = {}
        for group in groups:
            target_group_arn = str(group["TargetGroupArn"])
            target_health = _aws_json(
                release,
                [
                    "elbv2",
                    "describe-target-health",
                    "--target-group-arn",
                    target_group_arn,
                ],
            )
            descriptions = target_health.get("TargetHealthDescriptions", [])
            states = [
                str((item.get("TargetHealth") or {}).get("State") or "unknown")
                for item in descriptions
            ]
            current_states[target_group_arn] = states
            if len(states) < expected_targets or any(
                state != "healthy" for state in states
            ):
                all_groups_healthy = False
        last_states = current_states
        if all_groups_healthy:
            return
        if release.runner.dry_run:
            return
        time.sleep(5)
    raise ReleaseError(
        f"NLB did not reach {expected_targets} healthy targets per target "
        "group within 600s; last states=" + json.dumps(last_states, sort_keys=True)
    )


def normalized_record_name(value: str) -> str:
    """Route53 stores names fully qualified; compare them without the root dot."""

    return str(value).rstrip(".").lower()


def read_dns_record(
    release: Any,
    *,
    hosted_zone_id: str,
    hostname: str,
    record_type: str = "CNAME",
) -> dict[str, Any] | None:
    """Return the record set a later change would overwrite, or ``None``.

    Rollback needs the record exactly as Route53 holds it -- alias targets and
    all -- so this returns the whole ``ResourceRecordSet`` rather than the value
    the release happens to care about.
    """

    document = _aws_json(
        release,
        [
            "route53",
            "list-resource-record-sets",
            "--hosted-zone-id",
            str(hosted_zone_id),
            "--start-record-name",
            str(hostname),
            "--start-record-type",
            record_type,
            "--max-items",
            "1",
        ],
        regional=False,
        sensitive=True,
    )
    for record in document.get("ResourceRecordSets", []):
        if not isinstance(record, dict):
            continue
        if normalized_record_name(
            str(record.get("Name") or "")
        ) == normalized_record_name(hostname) and str(record.get("Type") or "") == (
            record_type
        ):
            return dict(record)
    return None


def submit_dns_change(
    release: Any,
    changes: list[dict[str, Any]],
    *,
    hosted_zone_id: str,
) -> None:
    """Submit one change batch and wait until Route53 reports it INSYNC.

    Both the release and its compensation have to know the change landed before
    they report success, so the wait belongs to the submission rather than to
    either caller. The zone is explicit because a rollback has to change the
    zone the candidate wrote into, which is not necessarily the one the current
    configuration names.
    """

    if not changes:
        return
    change_batch = json.dumps({"Changes": changes}, separators=(",", ":"))
    response = _aws_json(
        release,
        [
            "route53",
            "change-resource-record-sets",
            "--hosted-zone-id",
            str(hosted_zone_id),
            "--change-batch",
            change_batch,
        ],
        regional=False,
        sensitive=True,
    )
    change_id = str((response.get("ChangeInfo") or {}).get("Id") or "")
    if not change_id:
        raise ReleaseError("Route53 change did not return a change ID")
    release.runner.run(
        [
            "aws",
            "route53",
            "wait",
            "resource-record-sets-changed",
            "--id",
            change_id,
        ],
        capture=False,
    )
    change = (
        _aws_json(
            release,
            ["route53", "get-change", "--id", change_id],
            regional=False,
        ).get("ChangeInfo")
        or {}
    )
    if change.get("Status") != "INSYNC":
        raise ReleaseError(
            f"Route53 change {change_id} is {change.get('Status')}, expected INSYNC"
        )


def cname_already_points_at(record: dict[str, Any] | None, hostname: str) -> bool:
    """True when the live CNAME already carries exactly what the release wants.

    Route53 hands the name and the value back fully qualified, so both sides are
    compared without the trailing dot; the TTL has to match too, because the
    release owns it.
    """

    if not record or str(record.get("Type") or "") != "CNAME":
        return False
    if str(record.get("TTL") or "") != "60":
        return False
    values = [
        normalized_record_name(str(item.get("Value") or ""))
        for item in record.get("ResourceRecords") or []
        if isinstance(item, dict)
    ]
    return values == [normalized_record_name(hostname)]


def ensure_cname_points_at(release: Any, hostname: str) -> None:
    # A release whose NLB kept its hostname would otherwise submit a no-op
    # UPSERT and then sit through Route53's 30s change-propagation wait for a
    # record that never changed. Read first; only a real difference is worth
    # the change batch. Rollback is unaffected: it restores a different value,
    # so it always submits.
    current = read_dns_record(
        release,
        hosted_zone_id=str(release.config.dns.hosted_zone_id),
        hostname=str(release.config.dns.hostname),
    )
    if cname_already_points_at(current, hostname):
        return
    submit_dns_change(
        release,
        [
            {
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": release.config.dns.hostname,
                    "Type": "CNAME",
                    "TTL": 60,
                    "ResourceRecords": [{"Value": hostname}],
                },
            }
        ],
        hosted_zone_id=str(release.config.dns.hosted_zone_id),
    )


def ensure_control_plane_dns(release: Any) -> None:
    hostname = _wait_service_hostname(release)
    if release.runner.dry_run and not hostname:
        return
    load_balancer_arn = _wait_nlb_active(release, hostname)
    _wait_raw_nlb_dns(release, hostname)
    _wait_targets_healthy(release, load_balancer_arn)
    ensure_cname_points_at(release, hostname)


def apply_control_plane_nlb(release: Any) -> None:
    if not release.config.nlb:
        return
    verify_control_plane_dns_prerequisites(release)
    text = (
        ROOT / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
    ).read_text(encoding="utf-8")
    text = render_nlb_manifest(release.config, text)
    release.runner.run(release._cpu("apply", "-f", "-"), input_text=text)
    ensure_control_plane_dns(release)
