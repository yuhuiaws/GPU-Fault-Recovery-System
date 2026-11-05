"""The RDS CA bundle reader on inputs AWS and the cluster do not normally send.

``test_rds_ca_bundle_drift`` checks the subject parser against cryptography on
every vendored certificate and the findings on real bundle rotations. The
parser is hand-written (no crypto library on the status path), so it must also
refuse what is not a certificate -- truncated or oversized DER lengths, the
wrong element where a SEQUENCE belongs, a TBS without a subject, a subject
attribute without a type or with an empty one -- and read what is unusual but
valid: a version-1 certificate (no ``[0]`` version), an attribute type whose
arcs take more than one byte (``emailAddress``), a subject without a CN. The
remaining reads must turn a ConfigMap or an AWS answer of the wrong shape, and
an apply script that pins nothing or twice, into a refusal ``status`` can
print, not a crash.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import rds_ca_bundle_drift as drift
from gpu_fault.admin.rds_ca_bundle import (
    APPLY_SCRIPT,
    BUNDLE_KEY,
    CONFIGMAP_NAME,
    VENDORED_BUNDLE,
)

ROOT = Path(__file__).resolve().parents[2]
EMAIL_ADDRESS = "1.2.840.113549.1.9.1"
UTF8, PRINTABLE, IA5 = 0x0C, 0x13, 0x16


def tlv(tag: int, content: bytes) -> bytes:
    """One DER element; long-form length past 127 bytes."""

    if len(content) < 0x80:
        return bytes([tag, len(content)]) + content
    size = (len(content).bit_length() + 7) // 8
    return bytes([tag, 0x80 | size]) + len(content).to_bytes(size, "big") + content


def oid(dotted: str) -> bytes:
    first, second, *rest = (int(arc) for arc in dotted.split("."))
    body = bytearray([40 * first + second])
    for arc in rest:
        septets = [arc & 0x7F]
        arc >>= 7
        while arc:
            septets.append(0x80 | (arc & 0x7F))
            arc >>= 7
        body.extend(reversed(septets))
    return tlv(0x06, bytes(body))


def attribute(kind: bytes, value: str, tag: int = UTF8) -> bytes:
    """One RDN: SET { SEQUENCE { type, value } }."""

    return tlv(0x31, tlv(0x30, kind + tlv(tag, value.encode())))


def certificate(*rdns: bytes, version: bool = True, fields: int = 5) -> bytes:
    """A certificate as far as the subject parser reads it (no valid signature)."""

    subject = tlv(0x30, b"".join(rdns))
    algorithm = tlv(0x30, oid("1.2.840.10045.4.3.2"))
    validity = tlv(0x30, tlv(0x17, b"260101000000Z") + tlv(0x17, b"360101000000Z"))
    body = [tlv(0x02, b"\x2a"), algorithm, subject, validity, subject][:fields]
    if version:
        body.insert(0, tlv(0xA0, tlv(0x02, b"\x02")))
    tbs = tlv(0x30, b"".join(body))
    return tlv(0x30, tbs + algorithm + tlv(0x03, b"\x00" + b"\x01" * 16))


def pem(*certificates: bytes) -> bytes:
    return b"".join(
        b"-----BEGIN CERTIFICATE-----\n"
        + base64.encodebytes(der)
        + b"-----END CERTIFICATE-----\n"
        for der in certificates
    )


def test_a_version_one_certificate_with_a_multi_byte_attribute_type_is_read() -> None:
    der = certificate(
        attribute(oid("2.5.4.3"), "Unit Root CA"),
        attribute(oid(EMAIL_ADDRESS), "pki@example.com", IA5),
        version=False,
    )

    assert drift.certificate_subject(der) == (
        ("CN", "Unit Root CA"),
        (EMAIL_ADDRESS, "pki@example.com"),
    ), "an unknown attribute type is named by its dotted OID"
    bundle = drift.parse_bundle(pem(der))
    [entry] = bundle.certificates
    assert entry.name == "Unit Root CA", entry
    assert entry.sha256 == hashlib.sha256(der).hexdigest(), entry
    assert entry.sha1 == hashlib.sha1(der, usedforsecurity=False).hexdigest(), entry
    assert bundle.describe() == f"sha256 {bundle.sha256}, 1 CAs", bundle.describe()


def test_a_ca_without_a_common_name_is_named_by_its_whole_subject() -> None:
    der = certificate(
        attribute(oid("2.5.4.10"), "Example Corp", PRINTABLE),
        attribute(oid("2.5.4.11"), "Databases"),
    )
    [entry] = drift.parse_bundle(pem(der)).certificates

    assert entry.name == "O=Example Corp, OU=Databases", entry.name
    assert entry.label() == f"O=Example Corp, OU=Databases sha256:{entry.sha256}"
    nameless = drift.BundleCertificate((), "0" * 64, "1" * 40)
    assert nameless.name == "?", "an empty subject still reads as something"


@pytest.mark.parametrize(
    ("der", "problem"),
    [
        (b"\x30", "truncated DER element"),
        (b"\x30\x85\x00\x00\x00\x00\x01", "unsupported DER length"),
        (b"\x30\x80", "unsupported DER length"),
        (b"\x30\x82\x01", "unsupported DER length"),
        (b"\x30\x05\x00", "truncated DER element"),
        (b"\x31\x00", "unexpected DER tag 0x31"),
        (
            certificate(attribute(oid("2.5.4.3"), "x"), fields=3),
            "certificate has no subject",
        ),
        (
            certificate(attribute(tlv(UTF8, b"CN"), "Unit Root CA")),
            "certificate subject attribute has no type",
        ),
        (
            certificate(attribute(tlv(0x06, b""), "Unit Root CA")),
            "empty DER object identifier",
        ),
    ],
    ids=[
        "no-length",
        "five-byte-length",
        "indefinite-length",
        "length-past-end",
        "content-past-end",
        "set-for-sequence",
        "three-field-tbs",
        "untyped-attribute",
        "empty-oid",
    ],
)
def test_what_is_not_a_certificate_is_refused_with_its_cause(
    der: bytes, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        drift.certificate_subject(der)
    with pytest.raises(ValueError, match=problem):
        drift.parse_bundle(pem(der))


def test_no_difference_reads_as_the_same_cas() -> None:
    assert drift.BundleDrift((), (), ()).summary() == "the same CAs"


@pytest.mark.parametrize(
    ("pins", "count"), [([], 0), (["a" * 64, "b" * 64], 2)], ids=["none", "two"]
)
def test_an_apply_script_must_pin_exactly_one_digest(
    tmp_path: Path, pins: list[str], count: int
) -> None:
    script = tmp_path / APPLY_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/bin/bash\n" + "".join(f'RDS_CA_BUNDLE_SHA256="{pin}"\n' for pin in pins)
    )
    (tmp_path / VENDORED_BUNDLE).write_bytes((ROOT / VENDORED_BUNDLE).read_bytes())

    with pytest.raises(ValueError, match=f"pins {count} RDS_CA_BUNDLE_SHA256"):
        drift.pinned_bundle(tmp_path)


class Reads:
    """The status reads, answered from fixtures; every argv is recorded."""

    def __init__(self, *, configmap: Any, instances: Any) -> None:
        self.configmap = configmap
        self.instances = instances
        self.calls: list[list[str]] = []

    def run(self, arguments: Any, **kwargs: Any) -> str:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        assert kwargs.get("timeout_seconds") == drift.READ_TIMEOUT_SECONDS, kwargs
        if argv[0] == "kubectl":
            return json.dumps(self.configmap)
        if argv[0] == "curl":
            target = Path(argv[argv.index("-o") + 1])
            target.write_bytes((ROOT / VENDORED_BUNDLE).read_bytes())
            return ""
        if argv[:3] == ["aws", "rds", "describe-db-instances"]:
            return json.dumps({"DBInstances": self.instances})
        if argv[:3] == ["aws", "rds", "describe-certificates"]:
            return json.dumps({"Certificates": []})
        raise AssertionError(f"unexpected command: {argv}")


@pytest.mark.parametrize(
    ("configmap", "problem"),
    [
        ([{"kind": "ConfigMap"}], f"ConfigMap {CONFIGMAP_NAME} is not a JSON object"),
        (
            {"kind": "ConfigMap", "data": {BUNDLE_KEY: None}},
            f"ConfigMap {CONFIGMAP_NAME} has no {BUNDLE_KEY}",
        ),
        ({"kind": "ConfigMap"}, f"ConfigMap {CONFIGMAP_NAME} has no {BUNDLE_KEY}"),
    ],
    ids=["list", "null-bundle", "no-data"],
)
def test_a_configmap_of_the_wrong_shape_is_refused(
    configmap: Any, problem: str
) -> None:
    reads = Reads(configmap=configmap, instances=[])

    with pytest.raises(ValueError, match=problem):
        drift.installed_bundle(reads, kubeconfig="/state/cpu", namespace="ns")  # type: ignore[arg-type]
    assert reads.calls == [
        [
            *("kubectl", "--kubeconfig", "/state/cpu", "-n", "ns"),
            *("get", "configmap", CONFIGMAP_NAME, "-o", "json"),
        ]
    ], reads.calls


def test_a_cluster_without_instances_is_refused() -> None:
    reads = Reads(configmap={}, instances=[])

    with pytest.raises(ValueError, match="Aurora cluster aurora-a has no instances"):
        drift.aurora_certificate_authorities(
            reads,  # type: ignore[arg-type]
            aws_region="us-west-2",
            cluster_id="aurora-a",
        )
    [call] = reads.calls
    assert call[:3] == ["aws", "rds", "describe-db-instances"], call
    assert "Name=db-cluster-id,Values=aurora-a" in call, call


def test_status_names_each_unreadable_input_and_still_compares_the_rest() -> None:
    reads = Reads(configmap=["not", "an", "object"], instances=[])
    site = SimpleNamespace(
        repository_root=ROOT,
        release_config={
            "aws_region": "us-west-2",
            "cpu_kubeconfig": "/state/cpu.kubeconfig",
            "namespace": "gpu-fault-system",
            "health": {"aurora_cluster_id": "aurora-a"},
        },
    )

    lines = drift.rds_ca_bundle_findings(site, runner=reads)  # type: ignore[arg-type]

    assert lines == [
        f"RDS CA bundle: ConfigMap {CONFIGMAP_NAME} could not be read: "
        f"ConfigMap {CONFIGMAP_NAME} is not a JSON object",
        "RDS CA bundle: the Aurora certificate CAs could not be read: "
        "Aurora cluster aurora-a has no instances",
    ], lines
    assert [call[0] for call in reads.calls] == ["kubectl", "curl", "aws"], (
        "the published bundle is still compared after the ConfigMap failed"
    )
