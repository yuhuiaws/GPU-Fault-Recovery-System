"""``status``'s RDS CA bundle lines.

Live 2026-10-14 (and 2026-09-30 before it): AWS republished the RDS global CA
bundle for a region launch (the three sa-west-1 G1 roots, nothing removed) and
every deploy and every ``config`` of the site failed in
``apply-rds-ca-bundle.sh`` on the pinned digest, until a release shipped a new
pin. The release now installs its vendored copy, and ``status`` reports, as
findings and read-only, what the Pods trust against that copy, what AWS
publishes against it (added, removed, changed CAs) and whether the Aurora
instances' CA is in both.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import rds_ca_bundle_drift as drift
from gpu_fault.admin import status_report
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.rds_ca_bundle import (
    APPLY_SCRIPT,
    CONFIGMAP_NAME,
    PUBLISHED_BUNDLE_URL,
    VENDORED_BUNDLE,
)
from tests.admin.test_admin_site import site_file

ROOT = Path(__file__).resolve().parents[2]
VENDORED = (ROOT / VENDORED_BUNDLE).read_bytes()
CLUSTER_ID = "aurora-a"
INSTANCES = ("aurora-a-reader", "aurora-a-writer")
RSA2048_US_WEST_2 = "Amazon RDS us-west-2 Root CA RSA2048 G1"
SA_WEST_1 = "Amazon RDS sa-west-1 Root CA"
ATTRIBUTE_NAMES = {
    NameOID.COMMON_NAME: "CN",
    NameOID.COUNTRY_NAME: "C",
    NameOID.LOCALITY_NAME: "L",
    NameOID.STATE_OR_PROVINCE_NAME: "ST",
    NameOID.ORGANIZATION_NAME: "O",
    NameOID.ORGANIZATIONAL_UNIT_NAME: "OU",
}


def _blocks(data: bytes = VENDORED) -> list[bytes]:
    return re.findall(
        rb"-----BEGIN CERTIFICATE-----\n.*?-----END CERTIFICATE-----\n", data, re.S
    )


def _common_name(block: bytes) -> str:
    certificate = x509.load_pem_x509_certificate(block)
    (attribute,) = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return str(attribute.value)


def _without(*names: str) -> bytes:
    """The vendored bundle without the CAs whose common name contains a name."""

    return b"".join(
        block
        for block in _blocks()
        if not any(name in _common_name(block) for name in names)
    )


def _block(name: str) -> bytes:
    (block,) = [item for item in _blocks() if _common_name(item) == name]
    return block


def _sha1(block: bytes) -> str:
    return x509.load_pem_x509_certificate(block).fingerprint(hashes.SHA1()).hex()


def _sha256(block: bytes) -> str:
    return x509.load_pem_x509_certificate(block).fingerprint(hashes.SHA256()).hex()


def _reissued(block: bytes) -> bytes:
    """A self-signed CA with ``block``'s subject and a new key: same name, other CA."""

    subject = x509.load_pem_x509_certificate(block).subject
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM)


def _authorities(*thumbprints: str | None) -> tuple[Any, ...]:
    return tuple(
        drift.AuroraCertificateAuthority(name, "rds-ca-rsa2048-g1", thumbprint)
        for name, thumbprint in zip(INSTANCES, thumbprints, strict=True)
    )


def test_the_subject_parser_reads_every_vendored_certificate_like_cryptography() -> (
    None
):
    bundle = drift.parse_bundle(VENDORED)
    reference = x509.load_pem_x509_certificates(VENDORED)

    assert bundle.sha256 == hashlib.sha256(VENDORED).hexdigest()
    assert len(bundle.certificates) == len(reference) == len(_blocks())
    for mine, theirs in zip(bundle.certificates, reference, strict=True):
        expected = tuple(
            (
                ATTRIBUTE_NAMES.get(attribute.oid, attribute.oid.dotted_string),
                str(attribute.value),
            )
            for rdn in theirs.subject.rdns
            for attribute in rdn
        )
        assert mine.subject == expected
        assert mine.sha256 == theirs.fingerprint(hashes.SHA256()).hex()
        assert mine.sha1 == theirs.fingerprint(hashes.SHA1()).hex()
    names = {certificate.name for certificate in bundle.certificates}
    assert f"{RSA2048_US_WEST_2} (O=Amazon Web Services, Inc., OU=Amazon RDS)" in names


@pytest.mark.parametrize(
    "data",
    [b"", b"<html>503 Service Unavailable</html>\n", b"-----BEGIN CERTIFICATE-----\n"],
    ids=["empty", "error-page", "truncated"],
)
def test_a_download_that_is_no_bundle_is_refused(data: bytes) -> None:
    with pytest.raises(ValueError, match="holds no PEM certificate"):
        drift.parse_bundle(data)


def test_drift_classifies_added_removed_and_changed_cas() -> None:
    added = [_block(f"{SA_WEST_1} ECC384 G1"), _block(f"{SA_WEST_1} RSA2048 G1")]
    retired = _block(f"{SA_WEST_1} RSA4096 G1")
    original = _block(RSA2048_US_WEST_2)
    replaced = _reissued(original)
    # The reference lacks two CAs the candidate holds, holds one whose subject
    # the candidate lacks, and holds another certificate under one subject.
    reference = b"".join(
        replaced if block == original else block
        for block in _blocks()
        if block not in added
    )
    candidate = _without(f"{SA_WEST_1} RSA4096 G1")

    result = drift.bundle_drift(
        drift.parse_bundle(reference), drift.parse_bundle(candidate)
    )

    assert [item.sha256 for item in result.added] == [
        _sha256(block) for block in added
    ], "added CAs, in name order"
    assert [item.sha256 for item in result.removed] == [_sha256(retired)]
    (change,) = result.changed
    assert [item.sha256 for item in change.before] == [_sha256(replaced)]
    assert [item.sha256 for item in change.after] == [_sha256(original)]
    summary = result.summary()
    assert summary.startswith("2 added: ") and "1 REMOVED: " in summary, summary
    assert "1 CHANGED: " in summary and "nothing removed" not in summary, summary
    assert f"sha256:{_sha256(replaced)} -> sha256:{_sha256(original)}" in summary
    assert drift.bundle_drift(
        drift.parse_bundle(candidate), drift.parse_bundle(candidate)
    ) == drift.BundleDrift((), (), ())


def test_the_same_cas_in_another_order_are_no_drift() -> None:
    reordered = b"".join(reversed(_blocks()))
    pinned, published = drift.parse_bundle(VENDORED), drift.parse_bundle(reordered)

    result = drift.bundle_drift(pinned, published)

    assert (result.added, result.removed, result.changed) == ((), (), ())
    (line,) = drift.drift_findings(pinned, published, _authorities(None, None))
    assert "the same CAs as the copy this release pins" in line
    assert "nothing to review" in line and published.sha256 in line


def test_the_sa_west_1_rotation_is_one_added_only_line_with_the_roll_steps() -> None:
    """The 2026-10-14 rotation as ``status`` reports it on the old pin."""

    pinned = drift.parse_bundle(_without(SA_WEST_1))
    published = drift.parse_bundle(VENDORED)
    thumbprint = _sha1(_block(RSA2048_US_WEST_2))

    (line,) = drift.drift_findings(
        pinned, published, _authorities(thumbprint, thumbprint)
    )

    assert line.startswith(
        f"RDS CA bundle drift: AWS publishes {published.describe()} at "
        f"{PUBLISHED_BUNDLE_URL}, this release pins and installs "
        f"{pinned.describe()}: 3 added: "
    ), line
    for kind in ("ECC384", "RSA2048", "RSA4096"):
        block = _block(f"{SA_WEST_1} {kind} G1")
        assert (
            f"{SA_WEST_1} {kind} G1 (O=Amazon Web Services, Inc., OU=Amazon RDS) "
            f"sha256:{_sha256(block)}"
        ) in line
    assert ", nothing removed; the download is never installed" in line
    assert (
        f"run curl -fsS --proto =https {PUBLISHED_BUNDLE_URL} -o {VENDORED_BUNDLE}, "
        f"check that sha256sum prints {published.sha256}, set "
        f'RDS_CA_BUNDLE_SHA256="{published.sha256}" with a provenance line in '
        f"{APPLY_SCRIPT} and deploy"
    ) in line
    assert "REMOVED" not in line and "CRITICAL" not in line


def test_a_removed_ca_the_aurora_cluster_chains_to_is_reported_first() -> None:
    pinned = drift.parse_bundle(VENDORED)
    published = drift.parse_bundle(_without(RSA2048_US_WEST_2))
    removed = _block(RSA2048_US_WEST_2)
    thumbprint = _sha1(removed)

    first, summary = drift.drift_findings(
        pinned, published, _authorities(thumbprint, thumbprint)
    )

    assert first.startswith(
        "RDS CA bundle drift: AWS REMOVED 1 CA(s) of the bundle this release pins: "
        f"{RSA2048_US_WEST_2} (O=Amazon Web Services, Inc., OU=Amazon RDS) "
        f"sha256:{_sha256(removed)}; Aurora instance(s) aurora-a-reader "
        "(rds-ca-rsa2048-g1), aurora-a-writer (rds-ca-rsa2048-g1) chain to it: "
        "keep the pinned copy until the cluster uses a CA the published bundle keeps"
    ), first
    assert "1 REMOVED: " in summary and "nothing removed" not in summary
    other = drift.drift_findings(pinned, published, _authorities("0" * 40, "0" * 40))
    assert "no Aurora instance of the site chains to it" in other[0]
    unknown = drift.drift_findings(pinned, published, None)
    assert "whether an Aurora instance chains to it could not be read" in unknown[0]


def test_a_changed_ca_is_reported_with_both_fingerprints() -> None:
    original = _block(RSA2048_US_WEST_2)
    replaced = _reissued(original)
    pinned = drift.parse_bundle(VENDORED)
    published = drift.parse_bundle(VENDORED.replace(original, replaced))

    first, _summary = drift.drift_findings(
        pinned, published, _authorities(_sha1(original), None)
    )

    assert first.startswith("RDS CA bundle drift: AWS CHANGED 1 CA(s) "), first
    assert f"sha256:{_sha256(original)} -> sha256:{_sha256(replaced)}" in first
    assert "Aurora instance(s) aurora-a-reader (rds-ca-rsa2048-g1) chain to it" in first


def test_an_aurora_ca_missing_from_a_bundle_is_critical_or_blocks_the_next_apply() -> (
    None
):
    thumbprint = _sha1(_block(RSA2048_US_WEST_2))
    holding = drift.parse_bundle(VENDORED)
    lacking = drift.parse_bundle(_without(RSA2048_US_WEST_2))
    authorities = _authorities(thumbprint, thumbprint)

    assert (
        drift.aurora_findings({"installed": holding, "pinned": holding}, authorities)
        == []
    )
    critical, blocking = drift.aurora_findings(
        {"installed": lacking, "pinned": lacking}, authorities
    )
    what = (
        f"rds-ca-rsa2048-g1 (root sha1:{thumbprint}) of Aurora instance(s) "
        "aurora-a-reader, aurora-a-writer"
    )
    assert critical.startswith(
        f"RDS CA bundle: CRITICAL: ConfigMap {CONFIGMAP_NAME} does not hold the CA "
        f"{what}: new sslmode=verify-full connections fail"
    ), critical
    assert blocking.startswith(
        "RDS CA bundle: ACTION BEFORE THE NEXT APPLY: the copy this release pins "
        f"does not hold the CA {what}"
    ), blocking
    (unknown,) = drift.aurora_findings({"installed": holding}, _authorities(None, None))
    assert "describe-certificates lists no rds-ca-rsa2048-g1" in unknown


class World:
    """The reads ``rds_ca_bundle_findings`` makes, answered from fixtures."""

    def __init__(
        self,
        *,
        installed: bytes = VENDORED,
        published: bytes | None = VENDORED,
        cluster_id: str = CLUSTER_ID,
    ) -> None:
        self.installed = installed
        self.published = published
        self.cluster_id = cluster_id
        self.thumbprint = _sha1(_block(RSA2048_US_WEST_2))
        self.calls: list[list[str]] = []
        self.timeouts: list[float | None] = []

    def run(self, arguments: Any, **kwargs: Any) -> str:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        self.timeouts.append(kwargs.get("timeout_seconds"))
        if kwargs.get("mutate"):
            raise AssertionError(f"an RDS CA bundle read mutated: {argv}")
        if argv[0] == "kubectl":
            assert argv[argv.index("get") + 1 : argv.index("-o")] == [
                "configmap",
                CONFIGMAP_NAME,
            ], argv
            data = {"ca-bundle.pem": self.installed.decode("ascii")}
            return json.dumps({"kind": "ConfigMap", "data": data})
        if argv[0] == "curl":
            if self.published is None:
                raise BootstrapError("command failed (6): curl: unit outage")
            Path(argv[argv.index("-o") + 1]).write_bytes(self.published)
            return ""
        if argv[:3] == ["aws", "rds", "describe-db-instances"]:
            assert f"Name=db-cluster-id,Values={self.cluster_id}" in argv, argv
            return json.dumps(
                {
                    "DBInstances": [
                        {
                            "DBInstanceIdentifier": name,
                            "CACertificateIdentifier": "rds-ca-rsa2048-g1",
                        }
                        for name in INSTANCES
                    ]
                }
            )
        if argv[:3] == ["aws", "rds", "describe-certificates"]:
            return json.dumps(
                {
                    "Certificates": [
                        {
                            "CertificateIdentifier": "rds-ca-rsa2048-g1",
                            "Thumbprint": self.thumbprint,
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected command: {argv}")


def _site(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        repository_root=root,
        release_config={
            "aws_region": "us-west-2",
            "cpu_kubeconfig": "/state/cpu.kubeconfig",
            "namespace": "gpu-fault-system",
            "health": {"aurora_cluster_id": CLUSTER_ID},
        },
    )


def _findings(world: World, root: Path = ROOT) -> list[str]:
    return drift.rds_ca_bundle_findings(
        _site(root),  # type: ignore[arg-type]
        runner=world,  # type: ignore[arg-type]
    )


def test_a_site_that_trusts_what_aws_publishes_has_no_finding() -> None:
    world = World()

    assert _findings(world) == []
    assert [call[0] for call in world.calls] == ["kubectl", "curl", "aws", "aws"]
    assert all(timeout == drift.READ_TIMEOUT_SECONDS for timeout in world.timeouts), (
        f"every read is bounded: {world.timeouts}"
    )
    curl = world.calls[1]
    assert curl[curl.index("--proto") + 1] == "=https" and PUBLISHED_BUNDLE_URL in curl
    assert world.calls[0][1:5] == [
        "--kubeconfig",
        "/state/cpu.kubeconfig",
        "-n",
        "gpu-fault-system",
    ]


def test_the_configmap_left_by_the_previous_release_is_named_until_it_is_applied() -> (
    None
):
    """After a deploy that ships the rolled copy, until the engine applies it."""

    world = World(installed=_without(SA_WEST_1))

    (line,) = _findings(world)

    installed = drift.parse_bundle(world.installed)
    assert line.startswith(
        f"RDS CA bundle: ConfigMap {CONFIGMAP_NAME} holds {installed.describe()}, "
        f"not the copy this release pins ({drift.parse_bundle(VENDORED).describe()}; "
        "3 added: "
    ), line
    assert line.endswith(
        "nothing removed); the next deploy or config that runs the release engine "
        "installs the pinned copy"
    ), line


def test_an_unreadable_input_is_a_line_and_the_other_checks_still_run() -> None:
    world = World(installed=_without(RSA2048_US_WEST_2), published=None)

    critical, unreadable, installed = _findings(world)

    assert critical.startswith("RDS CA bundle: CRITICAL: ConfigMap "), critical
    assert unreadable.startswith(
        f"RDS CA bundle drift could not be checked: {PUBLISHED_BUNDLE_URL} could not "
        "be read (command failed (6): curl: unit outage); the installed bundle does "
        "not depend on it"
    ), unreadable
    assert installed.startswith(f"RDS CA bundle: ConfigMap {CONFIGMAP_NAME} holds "), (
        installed
    )
    assert f"1 added: {RSA2048_US_WEST_2} " in installed, installed


def test_a_vendored_copy_that_is_not_its_pin_is_a_finding(tmp_path: Path) -> None:
    tools = tmp_path / Path(APPLY_SCRIPT).parent
    tools.mkdir(parents=True)
    shutil.copy2(ROOT / APPLY_SCRIPT, tmp_path / APPLY_SCRIPT)
    tampered = b"".join(_blocks()[1:])
    (tmp_path / VENDORED_BUNDLE).write_bytes(tampered)

    lines = _findings(World(), root=tmp_path)

    pin = drift.parse_bundle(VENDORED).sha256
    assert lines == [
        f"RDS CA bundle: {VENDORED_BUNDLE} has sha256 "
        f"{hashlib.sha256(tampered).hexdigest()}, pinned {pin} in {APPLY_SCRIPT}: "
        "every deploy and config of this release refuses to install it"
    ], lines


def test_the_status_reader_runs_behind_a_read_only_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()

    def mutating_reader(runner: Any, **_kwargs: Any) -> Any:
        runner.run(["kubectl", "apply", "-f", "-"], mutate=True)
        raise AssertionError("the read-only runner must refuse first")

    monkeypatch.setattr(drift, "installed_bundle", mutating_reader)

    (line,) = _findings(world)

    assert line.startswith(
        f"RDS CA bundle: ConfigMap {CONFIGMAP_NAME} could not be read: "
    ), line
    assert not any("apply" in call for call in world.calls), "nothing is ever applied"


def _status_report() -> str:
    document = {
        "mode": "status",
        "healthy": True,
        "health_scope": "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [{"name": "cpu_workloads", "status": "PASS"}],
        },
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def test_status_prints_the_drift_finding_and_keeps_the_engine_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    repository = tmp_path / "repo"
    for relative in (APPLY_SCRIPT, VENDORED_BUNDLE):
        (repository / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, repository / relative)
    published = VENDORED + _reissued(_block(RSA2048_US_WEST_2))
    world = World(published=published, cluster_id="gpu-fault-aurora")
    monkeypatch.setattr(drift, "CommandRunner", lambda: world)
    monkeypatch.setattr(status_report, "status_findings", lambda _site: [])
    monkeypatch.setattr(
        admin_cli,
        "run_driver",
        lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=_status_report()
        ),
    )
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=False,
    )

    assert admin_cli.run(arguments) == 0, "a finding is not a failing check"

    captured = capsys.readouterr()
    assert json.loads(captured.out)["healthy"] is True
    findings = [
        line.removeprefix("gpu-fault-admin status: finding: ")
        for line in captured.err.splitlines()
        if line.startswith("gpu-fault-admin status: finding: RDS CA bundle")
    ]
    assert len(findings) == 1, captured.err
    assert findings[0].startswith("RDS CA bundle drift: AWS publishes "), findings
    assert f"1 added: {RSA2048_US_WEST_2} (O=Amazon Web Services, Inc." in findings[0]
    assert not any("apply" in call for call in world.calls), "status never applies"
