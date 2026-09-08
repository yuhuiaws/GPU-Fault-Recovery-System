"""Security-review fixes for fleet identity and endpoint validation.

Covers the 2026-09-07 findings:

* M-10 / M-11 -- a client that names itself a bare IP cannot use the
  ``host == node_id`` identity shortcut to steer the control plane at the
  instance metadata service (or any loopback/link-local address).
* node_id audit -- client-supplied node identifiers are constrained by a
  strict anchored regex (no whitespace, control characters, path traversal,
  or URL userinfo punctuation).
* M-29 -- the TLS-certificate check parses every block in a PEM chain, not
  only the first, so a malformed trailing certificate is rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpu_fault.fleet import AgentHeartbeat
from tests._builders import build_store, copy_model
from tests.fleet._support import NOW, heartbeat, registry, signed

VALID_CERT_BLOCK = (
    "-----BEGIN CERTIFICATE-----\ndGVzdC1jZXJ0aWZpY2F0ZQ==\n-----END CERTIFICATE-----"
)


def _with_cert(pem: str) -> AgentHeartbeat:
    """Rebuild a heartbeat through validation with an HTTPS endpoint + cert."""
    data = heartbeat("node-a").model_dump()
    data["endpoint"] = "https://node-a:9099"
    data["tls_certificate_pem"] = pem
    return AgentHeartbeat.model_validate(data)


def test_ip_shaped_node_id_cannot_bypass_the_address_gate() -> None:
    """A node that calls itself an IP does not get the identity shortcut.

    ``host == node_id`` used to short-circuit the address safety net, so a
    holder of the cluster secret could register node_id ``169.254.169.254``
    with a matching endpoint and have the control plane POST signed commands
    to the instance metadata service. A bare IP must now always clear the
    routable-private gate, which link-local/loopback addresses never do.
    """
    fleet = registry()
    for metadata_ip in (
        "169.254.169.254",  # AWS/GCP instance metadata service
        "127.0.0.1",  # loopback
    ):
        value = heartbeat(metadata_ip)  # endpoint defaults to http://<id>:9099
        assert value.node_id == metadata_ip
        with pytest.raises(ValueError, match="does not address node"):
            fleet.register(signed(value))


def test_private_ip_node_still_registers_through_the_routable_gate() -> None:
    """The IP restriction only removes the bypass; legitimate private IPs work.

    A node whose id is its own RFC1918 address still registers, because the
    routable-private path accepts it -- only the unsafe identity shortcut was
    removed.
    """
    assert (
        registry().register(signed(heartbeat("10.0.5.5"))).endpoint
        == "http://10.0.5.5:9099"
    )


@pytest.mark.parametrize(
    "bad_node_id",
    [
        "node a",  # whitespace
        "node\ta",  # control character
        "node-a\nnode-b",  # newline injection
        "../../etc/passwd",  # path traversal
        "node-a/../evil",  # path separator
        "attacker@node-a",  # URL userinfo punctuation
        "http://node-a:9099",  # a whole URL
        "-node-a",  # leading punctuation
        "",  # empty
    ],
)
def test_node_id_rejects_dangerous_identifiers(bad_node_id: str) -> None:
    with pytest.raises(ValidationError, match="node_id must be a bare host label"):
        heartbeat(bad_node_id)


@pytest.mark.parametrize(
    "good_node_id",
    [
        "node-a",
        "hyperpod-i-00000000000000001",
        "ip-10-1-2-3.us-west-2.compute.internal",
        "worker_1",
    ],
)
def test_node_id_accepts_real_fleet_identifiers(good_node_id: str) -> None:
    assert heartbeat(good_node_id).node_id == good_node_id


def test_single_certificate_still_validates() -> None:
    """Preserve existing single-cert behaviour."""
    hb = _with_cert(VALID_CERT_BLOCK)
    assert hb.tls_certificate_pem == VALID_CERT_BLOCK + "\n"


def test_full_pem_chain_is_validated() -> None:
    """A well-formed multi-cert bundle is accepted whole."""
    bundle = VALID_CERT_BLOCK + "\n" + VALID_CERT_BLOCK
    hb = _with_cert(bundle)
    assert hb.tls_certificate_pem == bundle + "\n"


def test_malformed_trailing_certificate_is_rejected() -> None:
    """A garbage second block no longer hides behind a valid leaf.

    The old check only inspected the overall BEGIN/END markers, so any bytes
    between a valid leaf and the final END marker slipped through.
    """
    bundle = (
        VALID_CERT_BLOCK + "\n-----BEGIN CERTIFICATE-----\n@@@ not base64 @@@\n"
        "-----END CERTIFICATE-----"
    )
    with pytest.raises(ValidationError, match="not valid base64"):
        _with_cert(bundle)


def test_content_smuggled_between_pem_blocks_is_rejected() -> None:
    bundle = VALID_CERT_BLOCK + "\nid-of-a-real-host\n" + VALID_CERT_BLOCK
    with pytest.raises(ValidationError, match="outside a PEM block"):
        _with_cert(bundle)


def test_endpoint_is_still_rechecked_on_every_heartbeat() -> None:
    """Regression guard: a registered node cannot be re-pointed at metadata."""
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    moved = copy_model(
        heartbeat("node-a"),
        heartbeat_id="heartbeat-moved",
        endpoint="http://169.254.169.254:9099",
    )
    with pytest.raises(ValueError, match="does not address node"):
        fleet.register(signed(moved))


def test_registry_construction_is_unaffected() -> None:
    """Sanity: a plain registry still builds with the shared store."""
    assert registry(build_store(), now=lambda: NOW) is not None
