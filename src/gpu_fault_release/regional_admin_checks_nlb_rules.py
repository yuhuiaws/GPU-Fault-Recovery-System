"""Which NLB Security Group ingress rules the preflight accepts for TCP 443.

Pure rule judgements of ``_check_nlb_inputs`` (``regional_admin_checks``, which
re-exports every name), kept apart from the checks that read AWS.
"""

from __future__ import annotations

from typing import Any


def _is_exact_https_tcp(permission: dict[str, Any]) -> bool:
    """True only for a rule scoped to exactly tcp/443-443."""

    return (
        str(permission.get("IpProtocol")) == "tcp"
        and permission.get("FromPort") == 443
        and permission.get("ToPort") == 443
    )


def permission_reaches_https(permission: dict[str, Any]) -> bool:
    """True if the rule admits TCP traffic to port 443.

    A `FromPort`-only check misses a wide range (443-65535), the all-traffic
    protocol ("-1"), or a range that starts below 443 and spans past it. Any of
    those admits the HTTPS port and must be inspected, not skipped.
    """

    protocol = str(permission.get("IpProtocol"))
    if protocol == "-1":
        return True
    if protocol != "tcp":
        return False
    from_port = permission.get("FromPort")
    to_port = permission.get("ToPort")
    if from_port is None or to_port is None:
        # A tcp rule with no port range covers every tcp port, 443 included.
        return True
    try:
        return int(from_port) <= 443 <= int(to_port)
    except (TypeError, ValueError):
        return True


def nlb_https_rule_violation(permission: dict[str, Any]) -> str | None:
    """Return why an ingress rule that reaches TCP 443 is unacceptable, or None.

    Only a rule scoped to exactly tcp/443-443 counts as the intended HTTPS
    ingress. A wider port range (e.g. 443-65535), the all-traffic protocol
    ("-1"), or any other shape that still admits 443 opens more than the NLB
    listener and is rejected outright -- otherwise it would bypass the
    0.0.0.0/0 exposure check below by not matching an exact-port comparison.
    """

    if not permission_reaches_https(permission):
        return None
    exposed_v4 = any(
        item.get("CidrIp") == "0.0.0.0/0" for item in permission.get("IpRanges", [])
    )
    exposed_v6 = any(
        item.get("CidrIpv6") == "::/0" for item in permission.get("Ipv6Ranges", [])
    )
    if not _is_exact_https_tcp(permission):
        if exposed_v4 or exposed_v6:
            return (
                "NLB Security Group admits TCP 443 through an overly broad rule "
                "exposed to the internet"
            )
        return (
            "NLB Security Group admits TCP 443 through an overly broad rule "
            "(expected exactly tcp/443-443)"
        )
    if exposed_v4:
        return "NLB Security Group exposes TCP 443 to 0.0.0.0/0"
    if exposed_v6:
        return "NLB Security Group exposes TCP 443 to ::/0"
    return None
