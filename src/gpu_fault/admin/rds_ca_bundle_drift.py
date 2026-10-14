"""``status``'s RDS CA bundle lines: what the Pods trust, what AWS publishes.

The deploy installs the RDS CA bundle vendored in the release
(``rds_ca_bundle.VENDORED_BUNDLE``) once its SHA-256 matches the pin in
``apply-rds-ca-bundle.sh``, and never a download. AWS republishes the global
bundle whenever an RDS region launches (three new roots each time); while the
script installed the download, the rotations of 2026-09-30 and 2026-10-14
failed every deploy and every ``config`` of the site until a release shipped a
new pin.

``status`` therefore compares, read-only and never for trust:

* the ``gpu-fault-rds-ca-bundle`` ConfigMap (what the Pods trust) with the
  pinned copy of the release the site runs; they differ from a release that
  rolls the copy until the release engine next applies it;
* the bundle AWS publishes with that pinned copy: added, removed and changed
  CAs by subject and SHA-256 fingerprint, and the steps to roll the copy;
* the CA each Aurora instance's server certificate is issued under
  (``CACertificateIdentifier``; ``describe-certificates`` gives the SHA-1
  thumbprint of that root as it appears in the bundle) with the installed and
  the pinned bundle: one that lacks it breaks ``sslmode=verify-full``.

Each difference is a finding, never a failing check (``healthy`` and the exit
code stay the engine's), and so is an input that cannot be read. Certificates
are parsed here without a crypto library: only the subject is read, for the
report; trust stays the pin.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import CommandRunner, ReadOnlyProbeRunner
from gpu_fault.admin.rds_ca_bundle import (
    APPLY_SCRIPT,
    BUNDLE_KEY,
    CONFIGMAP_NAME,
    PUBLISHED_BUNDLE_URL,
    VENDORED_BUNDLE,
)
from gpu_fault.admin.site import RenderedSite

READ_TIMEOUT_SECONDS = 30.0
# curl's own bound on the download; the runner's timeout is the outer one.
FETCH_MAX_SECONDS = 20
# What a fail-soft read survives (as ``aurora_placement``): a failed or timed
# out command, output that is not what it should be, a missing field. A test
# binary blocked by the suite's guard is an AssertionError and still surfaces.
READ_ERRORS = (RuntimeError, OSError, ValueError, LookupError, TypeError)
_PIN = re.compile(r'^RDS_CA_BUNDLE_SHA256="(?P<sha256>[0-9a-f]{64})"$', re.MULTILINE)
_PEM = re.compile(
    r"-----BEGIN CERTIFICATE-----(?P<body>[A-Za-z0-9+/=\s]*?)-----END CERTIFICATE-----"
)
_SEQUENCE, _SET, _OID, _VERSION = 0x30, 0x31, 0x06, 0xA0
_ATTRIBUTES = {
    "2.5.4.3": "CN",
    "2.5.4.6": "C",
    "2.5.4.7": "L",
    "2.5.4.8": "ST",
    "2.5.4.10": "O",
    "2.5.4.11": "OU",
}
# DER string types -> codec: UTF8String, PrintableString, IA5String,
# TeletexString, BMPString, UniversalString.
_STRINGS = {
    0x0C: "utf-8",
    0x13: "ascii",
    0x16: "ascii",
    0x14: "latin-1",
    0x1E: "utf-16-be",
    0x1C: "utf-32-be",
}


class PinMismatch(ValueError):
    """The vendored copy is not the bundle its script pins."""

    def __init__(self, pinned: str, actual: str) -> None:
        super().__init__(f"{VENDORED_BUNDLE} has sha256 {actual}, pinned {pinned}")
        self.pinned = pinned
        self.actual = actual


def _element(der: bytes, offset: int, end: int) -> tuple[int, int, int]:
    """``(tag, content start, content end)`` of the DER element at ``offset``."""

    if offset + 2 > end:
        raise ValueError("truncated DER element")
    tag, length = der[offset], der[offset + 1]
    start = offset + 2
    if length & 0x80:
        size = length & 0x7F
        if not 1 <= size <= 4 or start + size > end:
            raise ValueError("unsupported DER length")
        length = int.from_bytes(der[start : start + size], "big")
        start += size
    if start + length > end:
        raise ValueError("truncated DER element")
    return tag, start, start + length


def _children(
    der: bytes, element: tuple[int, int, int], tag: int
) -> list[tuple[int, int, int]]:
    if element[0] != tag:
        raise ValueError(f"unexpected DER tag 0x{element[0]:02x}")
    children: list[tuple[int, int, int]] = []
    offset = element[1]
    while offset < element[2]:
        children.append(_element(der, offset, element[2]))
        offset = children[-1][2]
    return children


def _oid(value: bytes) -> str:
    if not value:
        raise ValueError("empty DER object identifier")
    head = min(value[0] // 40, 2)
    arcs = [head, value[0] - 40 * head]
    current = 0
    for byte in value[1:]:
        current = (current << 7) | (byte & 0x7F)
        if not byte & 0x80:
            arcs.append(current)
            current = 0
    return ".".join(str(arc) for arc in arcs)


def certificate_subject(der: bytes) -> tuple[tuple[str, str], ...]:
    """The subject attributes of one DER certificate, in certificate order."""

    tbs = _children(der, _element(der, 0, len(der)), _SEQUENCE)[0]
    fields = _children(der, tbs, _SEQUENCE)
    if fields and fields[0][0] == _VERSION:
        fields = fields[1:]
    # serialNumber, signature, issuer, validity, subject
    if len(fields) < 5:
        raise ValueError("certificate has no subject")
    attributes = []
    for name in _children(der, fields[4], _SEQUENCE):
        for pair in _children(der, name, _SET):
            kind, value = _children(der, pair, _SEQUENCE)[:2]
            if kind[0] != _OID:
                raise ValueError("certificate subject attribute has no type")
            oid = _oid(der[kind[1] : kind[2]])
            raw = der[value[1] : value[2]]
            codec = _STRINGS.get(value[0])
            attributes.append(
                (_ATTRIBUTES.get(oid, oid), raw.decode(codec) if codec else raw.hex())
            )
    return tuple(attributes)


@dataclass(frozen=True)
class BundleCertificate:
    subject: tuple[tuple[str, str], ...]
    sha256: str
    sha1: str

    @property
    def name(self) -> str:
        """``CN (O=..., OU=...)``: how an RDS CA reads in a report."""

        common = [value for key, value in self.subject if key == "CN"]
        owner = [f"{key}={value}" for key, value in self.subject if key in ("O", "OU")]
        if not common:
            return ", ".join(f"{key}={value}" for key, value in self.subject) or "?"
        return ", ".join(common) + (f" ({', '.join(owner)})" if owner else "")

    def label(self) -> str:
        return f"{self.name} sha256:{self.sha256}"


@dataclass(frozen=True)
class Bundle:
    sha256: str
    certificates: tuple[BundleCertificate, ...]

    @property
    def thumbprints(self) -> frozenset[str]:
        return frozenset(certificate.sha1 for certificate in self.certificates)

    def describe(self) -> str:
        return f"sha256 {self.sha256}, {len(self.certificates)} CAs"


def parse_bundle(data: bytes) -> Bundle:
    certificates = []
    for block in _PEM.finditer(data.decode("ascii")):
        der = base64.b64decode("".join(block.group("body").split()), validate=True)
        certificates.append(
            BundleCertificate(
                certificate_subject(der),
                hashlib.sha256(der).hexdigest(),
                hashlib.sha1(der, usedforsecurity=False).hexdigest(),
            )
        )
    if not certificates:
        raise ValueError("holds no PEM certificate")
    return Bundle(hashlib.sha256(data).hexdigest(), tuple(certificates))


@dataclass(frozen=True)
class ChangedCA:
    """The certificates of one subject that another certificate replaced."""

    before: tuple[BundleCertificate, ...]
    after: tuple[BundleCertificate, ...]

    def label(self) -> str:
        return (
            f"{self.after[0].name} sha256:"
            + "/".join(item.sha256 for item in self.before)
            + " -> sha256:"
            + "/".join(item.sha256 for item in self.after)
        )


@dataclass(frozen=True)
class BundleDrift:
    added: tuple[BundleCertificate, ...]
    removed: tuple[BundleCertificate, ...]
    changed: tuple[ChangedCA, ...]

    def summary(self) -> str:
        if not (self.added or self.removed or self.changed):
            return "the same CAs"
        parts = [
            f"{len(items)} {kind}: " + "; ".join(item.label() for item in items)
            for kind, items in (
                ("added", self.added),
                ("REMOVED", self.removed),
                ("CHANGED", self.changed),
            )
            if items
        ]
        quiet = not (self.removed or self.changed)
        return ", ".join(parts) + (", nothing removed" if quiet else "")


def bundle_drift(reference: Bundle, candidate: Bundle) -> BundleDrift:
    """What ``candidate`` adds to, removes from and changes in ``reference``.

    A CA is its SHA-256 fingerprint; a subject whose certificates were removed
    while another certificate of that subject was added is a changed CA.
    """

    before = {item.sha256: item for item in reference.certificates}
    after = {item.sha256: item for item in candidate.certificates}
    gone = sorted(
        (item for key, item in before.items() if key not in after), key=_order
    )
    new = sorted((item for key, item in after.items() if key not in before), key=_order)
    replaced = {item.subject for item in gone} & {item.subject for item in new}
    return BundleDrift(
        added=tuple(item for item in new if item.subject not in replaced),
        removed=tuple(item for item in gone if item.subject not in replaced),
        changed=tuple(
            ChangedCA(
                tuple(item for item in gone if item.subject == subject),
                tuple(item for item in new if item.subject == subject),
            )
            for subject in sorted(replaced)
        ),
    )


def _order(item: BundleCertificate) -> tuple[str, str]:
    return item.name, item.sha256


def pinned_bundle(repository_root: Path) -> Bundle:
    """The release's vendored copy, refused unless it is the bundle it pins."""

    script = (repository_root / APPLY_SCRIPT).read_text(encoding="utf-8")
    pins = _PIN.findall(script)
    if len(pins) != 1:
        raise ValueError(f"{APPLY_SCRIPT} pins {len(pins)} RDS_CA_BUNDLE_SHA256")
    data = (repository_root / VENDORED_BUNDLE).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != pins[0]:
        raise PinMismatch(pins[0], actual)
    return parse_bundle(data)


def _json_object(raw: str, what: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    return value


def installed_bundle(
    runner: CommandRunner, *, kubeconfig: str, namespace: str
) -> Bundle:
    """The bundle in the ConfigMap the Pods mount."""

    document = _json_object(
        runner.run(
            [
                *("kubectl", "--kubeconfig", kubeconfig, "-n", namespace),
                *("get", "configmap", CONFIGMAP_NAME, "-o", "json"),
            ],
            timeout_seconds=READ_TIMEOUT_SECONDS,
        ),
        f"ConfigMap {CONFIGMAP_NAME}",
    )
    value = (document.get("data") or {}).get(BUNDLE_KEY)
    if not isinstance(value, str):
        raise ValueError(f"ConfigMap {CONFIGMAP_NAME} has no {BUNDLE_KEY}")
    return parse_bundle(value.encode("utf-8"))


def published_bundle(
    runner: CommandRunner, *, url: str = PUBLISHED_BUNDLE_URL
) -> Bundle:
    """The bundle AWS publishes now, downloaded to compare and then discarded."""

    with tempfile.TemporaryDirectory(prefix="gpu-fault-rds-ca-") as directory:
        target = Path(directory) / "published.pem"
        runner.run(
            [
                *("curl", "--fail", "--silent", "--show-error", "--location"),
                *("--proto", "=https", "--connect-timeout", "5"),
                *("--max-time", str(FETCH_MAX_SECONDS), url, "-o", str(target)),
            ],
            timeout_seconds=READ_TIMEOUT_SECONDS,
        )
        return parse_bundle(target.read_bytes())


@dataclass(frozen=True)
class AuroraCertificateAuthority:
    instance_id: str
    identifier: str
    # The SHA-1 thumbprint ``describe-certificates`` gives for the CA's root.
    thumbprint: str | None


def aurora_certificate_authorities(
    runner: CommandRunner, *, aws_region: str, cluster_id: str
) -> tuple[AuroraCertificateAuthority, ...]:
    """Each instance's server certificate CA and that CA's root thumbprint."""

    def aws(*arguments: str) -> dict[str, Any]:
        return _json_object(
            runner.run(
                ["aws", "rds", *arguments, "--region", aws_region, "--output", "json"],
                timeout_seconds=READ_TIMEOUT_SECONDS,
            ),
            f"aws rds {arguments[0]}",
        )

    instances = aws(
        "describe-db-instances", "--filters", f"Name=db-cluster-id,Values={cluster_id}"
    ).get("DBInstances")
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"Aurora cluster {cluster_id} has no instances")
    thumbprints = {
        str(item.get("CertificateIdentifier") or ""): str(item.get("Thumbprint") or "")
        for item in aws("describe-certificates").get("Certificates") or []
    }
    return tuple(
        sorted(
            (
                AuroraCertificateAuthority(
                    str(item.get("DBInstanceIdentifier") or ""),
                    identifier := str(item.get("CACertificateIdentifier") or ""),
                    thumbprints.get(identifier, "").lower() or None,
                )
                for item in instances
            ),
            key=lambda authority: authority.instance_id,
        )
    )


def roll_steps(published: Bundle) -> str:
    return (
        f"to roll it, in the source checkout run curl -fsS --proto =https "
        f"{PUBLISHED_BUNDLE_URL} -o {VENDORED_BUNDLE}, check that sha256sum prints "
        f'{published.sha256}, set RDS_CA_BUNDLE_SHA256="{published.sha256}" with a '
        f"provenance line in {APPLY_SCRIPT} and deploy; the next deploy or config "
        "that runs the release engine installs it"
    )


def _authority_groups(
    authorities: Iterable[AuroraCertificateAuthority],
) -> dict[tuple[str, str | None], list[str]]:
    groups: dict[tuple[str, str | None], list[str]] = {}
    for authority in authorities:
        key = (authority.identifier, authority.thumbprint)
        groups.setdefault(key, []).append(authority.instance_id)
    return groups


def _chaining(
    certificates: Sequence[BundleCertificate],
    authorities: Sequence[AuroraCertificateAuthority] | None,
) -> str:
    """Which Aurora instances chain to ``certificates``, as a clause."""

    them = "it" if len(certificates) == 1 else "them"
    if authorities is None:
        return f"whether an Aurora instance chains to {them} could not be read"
    thumbprints = {item.sha1 for item in certificates}
    chained = [item for item in authorities if item.thumbprint in thumbprints]
    if not chained:
        return f"no Aurora instance of the site chains to {them}"
    names = ", ".join(f"{item.instance_id} ({item.identifier})" for item in chained)
    return (
        f"Aurora instance(s) {names} chain to {them}: keep the pinned copy until "
        "the cluster uses a CA the published bundle keeps"
    )


def drift_findings(
    pinned: Bundle,
    published: Bundle,
    authorities: Sequence[AuroraCertificateAuthority] | None,
) -> list[str]:
    """The published bundle against the copy this release pins and installs."""

    if published.sha256 == pinned.sha256:
        return []
    drift = bundle_drift(pinned, published)
    if not (drift.added or drift.removed or drift.changed):
        return [
            f"RDS CA bundle drift: AWS publishes {published.describe()} at "
            f"{PUBLISHED_BUNDLE_URL}, the same CAs as the copy this release pins "
            f"({pinned.describe()}) in other bytes; nothing to review; "
            + roll_steps(published)
        ]
    lines = [
        f"RDS CA bundle drift: AWS {kind} {len(items)} CA(s) of the bundle this "
        f"release pins: {'; '.join(item.label() for item in items)}; "
        f"{_chaining(certificates, authorities)}; review this before rolling the "
        "vendored copy, the Pods stop trusting what it no longer holds"
        for kind, items, certificates in (
            ("REMOVED", drift.removed, drift.removed),
            (
                "CHANGED",
                drift.changed,
                [item for change in drift.changed for item in change.before],
            ),
        )
        if items
    ]
    lines.append(
        f"RDS CA bundle drift: AWS publishes {published.describe()} at "
        f"{PUBLISHED_BUNDLE_URL}, this release pins and installs "
        f"{pinned.describe()}: {drift.summary()}; the download is never "
        "installed, so trust is unchanged until the vendored copy is rolled; "
        + roll_steps(published)
    )
    return lines


def aurora_findings(
    bundles: Mapping[str, Bundle],
    authorities: Sequence[AuroraCertificateAuthority],
) -> list[str]:
    """Aurora CAs the installed or the pinned bundle does not hold."""

    lines = []
    for (identifier, thumbprint), instances in _authority_groups(authorities).items():
        names = ", ".join(instances)
        if thumbprint is None:
            lines.append(
                f"RDS CA bundle: describe-certificates lists no {identifier or '<none>'}"
                f", the CA of Aurora instance(s) {names}: whether the bundles "
                "hold it could not be checked"
            )
            continue
        what = f"{identifier} (root sha1:{thumbprint}) of Aurora instance(s) {names}"
        installed = bundles.get("installed")
        pinned = bundles.get("pinned")
        if installed is not None and thumbprint not in installed.thumbprints:
            lines.append(
                f"RDS CA bundle: CRITICAL: ConfigMap {CONFIGMAP_NAME} does not hold "
                f"the CA {what}: new sslmode=verify-full connections fail; apply a "
                "release whose pinned copy holds it, or move the instances to a CA "
                "the installed bundle holds"
            )
        if pinned is not None and thumbprint not in pinned.thumbprints:
            lines.append(
                f"RDS CA bundle: ACTION BEFORE THE NEXT APPLY: the copy this release "
                f"pins does not hold the CA {what}: installing it (the next deploy "
                "or config that runs the release engine) breaks sslmode=verify-full; "
                "roll the vendored copy to a bundle that holds it first"
            )
    return lines


def installed_findings(installed: Bundle, pinned: Bundle) -> list[str]:
    if installed.sha256 == pinned.sha256:
        return []
    return [
        f"RDS CA bundle: ConfigMap {CONFIGMAP_NAME} holds {installed.describe()}, "
        f"not the copy this release pins ({pinned.describe()}; "
        f"{bundle_drift(installed, pinned).summary()}); the next deploy or config "
        "that runs the release engine installs the pinned copy"
    ]


def rds_ca_bundle_findings(
    site: RenderedSite, *, runner: CommandRunner | None = None
) -> list[str]:
    """``status``'s RDS CA bundle lines; an unreadable input is a line itself.

    Read-only by construction: the runner refuses any mutating command.
    """

    probe = ReadOnlyProbeRunner(runner or CommandRunner())
    config = site.release_config
    problems: list[str] = []
    bundles: dict[str, Bundle] = {}
    authorities: tuple[AuroraCertificateAuthority, ...] | None = None
    reads: list[tuple[str, Callable[[], Bundle]]] = [
        ("pinned", lambda: pinned_bundle(site.repository_root)),
        (
            "installed",
            lambda: installed_bundle(
                probe,
                kubeconfig=str(config["cpu_kubeconfig"]),
                namespace=str(config["namespace"]),
            ),
        ),
        ("published", lambda: published_bundle(probe)),
    ]
    for name, read in reads:
        try:
            bundles[name] = read()
        except PinMismatch as exc:
            problems.append(
                f"RDS CA bundle: {exc} in {APPLY_SCRIPT}: every deploy and config "
                "of this release refuses to install it"
            )
        except READ_ERRORS as exc:
            problems.append(_unreadable(name, exc))
    try:
        authorities = aurora_certificate_authorities(
            probe,
            aws_region=str(config["aws_region"]),
            cluster_id=str(config["health"]["aurora_cluster_id"]),
        )
    except READ_ERRORS as exc:
        problems.append(
            f"RDS CA bundle: the Aurora certificate CAs could not be read: {exc}"
        )
    lines = [*aurora_findings(bundles, authorities or ()), *problems]
    if "installed" in bundles and "pinned" in bundles:
        lines.extend(installed_findings(bundles["installed"], bundles["pinned"]))
    if "pinned" in bundles and "published" in bundles:
        lines.extend(
            drift_findings(bundles["pinned"], bundles["published"], authorities)
        )
    return lines


def _unreadable(name: str, exc: BaseException) -> str:
    if name == "published":
        return (
            f"RDS CA bundle drift could not be checked: {PUBLISHED_BUNDLE_URL} could "
            f"not be read ({exc}); the installed bundle does not depend on it"
        )
    what = {
        "pinned": f"the vendored copy {VENDORED_BUNDLE} of the release this site runs",
        "installed": f"ConfigMap {CONFIGMAP_NAME}",
    }[name]
    return f"RDS CA bundle: {what} could not be read: {exc}"
