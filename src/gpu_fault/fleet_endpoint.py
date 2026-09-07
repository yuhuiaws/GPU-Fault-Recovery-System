from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit


DEFAULT_AGENT_ENDPOINT_PORTS = frozenset({9099})
EndpointNetworks = tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
NODE_ADDRESS_LABEL_PATTERN = re.compile(r"^ip-(\d{1,3})-(\d{1,3})-(\d{1,3})-(\d{1,3})$")


def node_name_address(node_id: str) -> str | None:
    """Return the IPv4 address encoded by an EC2 private DNS name."""
    match = NODE_ADDRESS_LABEL_PATTERN.fullmatch(node_id.split(".", 1)[0].lower())
    if match is None:
        return None
    octets = [int(item) for item in match.groups()]
    if any(octet > 255 for octet in octets):
        return None
    return ".".join(str(octet) for octet in octets)


def parse_endpoint_networks(
    value: str,
) -> tuple[
    ipaddress.IPv4Network | ipaddress.IPv6Network,
    ...,
]:
    """Parse a comma-separated CIDR allow-list."""
    return tuple(
        ipaddress.ip_network(item.strip(), strict=False)
        for item in value.split(",")
        if item.strip()
    )


class ClusterEndpointNetworks(dict[str, EndpointNetworks]):
    """Per-cluster allow-list that resolves unseen clusters on first sight.

    Built once at start-up from the store, the plain dict refused every cluster
    joined online until the Pods restarted (review H4). The resolver reads the
    registry runtime's current snapshot, so a committed or PENDING revision is
    honoured immediately; a cluster the snapshot does not contain resolves to
    ``None`` and the caller fails closed exactly as before. Nothing is cached:
    a later revision that changes a cluster's CIDRs must win on the next
    heartbeat, and a snapshot lookup is one lock and one dict read.
    """

    def __init__(
        self,
        initial: Mapping[str, EndpointNetworks] | None = None,
        *,
        resolver: Callable[[str], EndpointNetworks | None],
    ) -> None:
        super().__init__(initial or {})
        self.resolver = resolver

    def __missing__(self, cluster_id: str) -> EndpointNetworks:
        networks = self.resolver(cluster_id)
        if networks is None:
            raise KeyError(cluster_id)
        return networks


def endpoint_networks_for_cluster(
    cluster_id: str,
    configured: Mapping[str, EndpointNetworks],
    fallback: EndpointNetworks,
) -> EndpointNetworks:
    # An empty registry-backed map is still a per-cluster policy: it means "no
    # cluster has been seen yet", never "use the global list for everyone".
    if not configured and not isinstance(configured, ClusterEndpointNetworks):
        return fallback
    try:
        return configured[cluster_id]
    except KeyError as exc:
        raise ValueError(
            f"agent endpoint CIDRs are unavailable for cluster {cluster_id}"
        ) from exc


def _routable_private_address(
    host: str,
    allowed_networks: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network,
        ...,
    ] = (),
) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if not (
        address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
    ):
        return False
    if not allowed_networks:
        return True
    return any(address in network for network in allowed_networks)


def validate_agent_endpoint(
    node_id: str,
    endpoint: str,
    *,
    allowed_ports: frozenset[int] = DEFAULT_AGENT_ENDPOINT_PORTS,
    allowed_host_suffixes: tuple[str, ...] = (),
    allowed_networks: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network,
        ...,
    ] = (),
) -> None:
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"}:
        raise ValueError("agent endpoint must use http or https")
    if parts.username or parts.password:
        raise ValueError("agent endpoint must not carry credentials")
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise ValueError("agent endpoint must be a bare scheme://host:port")
    try:
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ValueError("agent endpoint port is not a number") from exc
    if not host:
        raise ValueError("agent endpoint must name a host")
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    if port not in allowed_ports:
        raise ValueError(
            f"agent endpoint port {port} is not one of "
            + ",".join(str(item) for item in sorted(allowed_ports))
        )
    host = host.strip("[]").lower()
    identities = {
        node_id.lower(),
        node_id.split(".", 1)[0].lower(),
    }
    if host in identities:
        return
    if _routable_private_address(
        host,
        allowed_networks,
    ):
        return
    encoded = node_name_address(host)
    if encoded and _routable_private_address(
        encoded,
        allowed_networks,
    ):
        return
    if any(host.endswith(suffix.lower()) for suffix in allowed_host_suffixes if suffix):
        return
    if allowed_networks and (
        _routable_private_address(host) or _routable_private_address(encoded or "")
    ):
        raise ValueError(
            f"agent endpoint host {host} is outside "
            "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_CIDRS ("
            + ",".join(str(item) for item in allowed_networks)
            + f"); node {node_id} must advertise an address "
            "in its own node subnet"
        )
    raise ValueError(
        f"agent endpoint host {host} does not address "
        f"node {node_id}; advertise the node's own name "
        "or private address, or add its domain to "
        "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_HOST_SUFFIXES"
    )
