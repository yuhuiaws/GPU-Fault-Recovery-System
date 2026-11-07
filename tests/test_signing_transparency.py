"""``scripts/signing_transparency.py`` and the signing wrapper's two exits.

Live 2026-11-07 the R24 deploy stopped at ``cosign sign-blob`` because the
public Sigstore transparency log answered ``status 500``. These tests pin what
the wrapper does about that -- bounded retries, then a refusal that names
``--allow-unlogged-signature --reason`` and writes nothing -- what the unlogged
signing requires (a declaration inside the signed payload, a key), and the
site-pinned Sigstore trust root (only from cosign's TUF cache, never a lower
TUF version, used only when the live TUF fetch fails). A fake ``cosign`` on a
temporary path stands in for the real one; nothing signs, uploads or reaches
the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_release_signing as admin_signing
from scripts import cosign_sign_blob, run_static_gates
from scripts import signing_transparency as transparency

# The live refusal, as the R24 deploy log recorded it (cosign v3.1.2).
REKOR_500 = (
    "Error: signing dist/current-attestation.json: signing bundle: error signing "
    'bundle: Post "https://rekor.sigstore.dev/api/v1/log/entries": giving up '
    'after 2 attempt(s): status 500: {"code":500,"message":"unexpected result '
    'from transparency log"}\n'
)
TUF_VERIFY_FAILURE = (
    "Error: setting trusted material: getting trusted root from TUF for new bundle "
    "verification: error getting live trusted root: failed to create TUF client "
    'failed to load metadata: tuf refresh failed: Get "https://tuf-repo-cdn.'
    'sigstore.dev/15.root.json": dial tcp: network is unreachable\n'
)
TUF_REFRESH_FAILURE = (
    'Error: failed to load metadata: tuf refresh failed: Get "https://tuf-repo-cdn.'
    'sigstore.dev/15.root.json": dial tcp: i/o timeout\n'
)
TUF_SIGNING_CONFIG_FAILURE = (
    "Error: error getting signing config from TUF: error getting signing config "
    "from TUF: failed to load metadata\n"
)
BAD_PASSWORD = "Error: reading key: decrypt: encrypted: decryption failed\n"
UNLOGGED_BUNDLE = {
    "mediaType": cosign_sign_blob.BUNDLE_MEDIA_TYPE,
    "verificationMaterial": {"publicKey": {"hint": "aGludA=="}},
    "messageSignature": {
        "messageDigest": {"algorithm": "SHA2_256", "digest": "ZGlnZXN0"},
        "signature": "c2lnbmF0dXJl",
    },
}
LOGGED_BUNDLE = {
    **UNLOGGED_BUNDLE,
    "verificationMaterial": {
        "publicKey": {"hint": "aGludA=="},
        "tlogEntries": [{"logIndex": "1"}],
    },
}

FAKE_COSIGN = """\
import json, os, sys
from pathlib import Path

plan_path = Path(os.environ["FAKE_COSIGN_PLAN"])
plan = json.loads(plan_path.read_text())
argv = sys.argv[1:]
with Path(os.environ["FAKE_COSIGN_LOG"]).open("a") as log:
    log.write(json.dumps({
        "argv": argv,
        "reason": os.environ.get("GPU_FAULT_RELEASE_UNLOGGED_SIGNATURE_REASON"),
    }) + "\\n")
steps = plan.setdefault(argv[0], [])
step = steps.pop(0) if steps else {"rc": 0}
plan_path.write_text(json.dumps(plan))
if step.get("bundle") is not None and "--bundle" in argv:
    Path(argv[argv.index("--bundle") + 1]).write_text(json.dumps(step["bundle"]))
sys.stderr.write(step.get("stderr", ""))
raise SystemExit(step.get("rc", 0))
"""


@dataclass
class FakeCosign:
    path: Path
    plan: Path
    log: Path

    def configure(self, **steps: list[dict[str, Any]]) -> None:
        self.plan.write_text(json.dumps(steps), encoding="utf-8")

    def calls(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def environment(self, home: Path, **extra: str) -> dict[str, str]:
        return {
            "PATH": f"{self.path.parent}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(home),
            "FAKE_COSIGN_PLAN": str(self.plan),
            "FAKE_COSIGN_LOG": str(self.log),
            **extra,
        }


@pytest.fixture
def cosign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCosign:
    directory = tmp_path / "bin"
    directory.mkdir()
    path = directory / "cosign"
    path.write_text(f"#!{sys.executable}\n{FAKE_COSIGN}", encoding="utf-8")
    path.chmod(0o755)
    fake = FakeCosign(path=path, plan=tmp_path / "plan.json", log=tmp_path / "log")
    fake.configure()
    monkeypatch.setenv("FAKE_COSIGN_PLAN", str(fake.plan))
    monkeypatch.setenv("FAKE_COSIGN_LOG", str(fake.log))
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cosign_sign_blob, "SIGN_RETRY_SECONDS", (0.0, 0.0))
    return fake


def _declared(path: Path, reason: str = "rekor returns 500 since 14:21Z") -> Path:
    path.write_text(
        json.dumps(
            {
                "subject": {"release_id": "abc"},
                "signing": transparency.unlogged_declaration(reason),
            }
        ),
        encoding="utf-8",
    )
    return path


def _archive(path: Path, manifest: dict[str, Any]) -> Path:
    data = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo("gpu-fault-deploy-host/manifest.json")
        member.size = len(data)
        bundle.addfile(member, io.BytesIO(data))
    return path


def _site_key_directory(root: Path) -> Path:
    directory = root / "release-signing"
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)
    (directory / "cosign.key").write_text("encrypted private key", encoding="utf-8")
    (directory / "cosign.pub").write_text("public key", encoding="utf-8")
    return directory


def _tuf_cache(
    home: Path,
    *,
    timestamp_version: int = 7,
    targets_version: int = 3,
    trusted_root: bytes = b'{"mediaType":"trusted-root","tlogs":["rekor"]}',
    signing_config: bytes = b'{"mediaType":"signing-config"}',
    corrupt: bool = False,
) -> Path:
    cache = home / ".sigstore" / "root" / transparency.TUF_MIRROR
    (cache / "targets").mkdir(parents=True, exist_ok=True)
    (cache / "targets" / "trusted_root.json").write_bytes(trusted_root)
    (cache / "targets" / "signing_config.v0.2.json").write_bytes(signing_config)
    listed_root = b"another root" if corrupt else trusted_root
    (cache / "targets.json").write_text(
        json.dumps(
            {
                "signed": {
                    "version": targets_version,
                    "targets": {
                        "trusted_root.json": {
                            "length": len(listed_root),
                            "hashes": {
                                "sha256": hashlib.sha256(listed_root).hexdigest()
                            },
                        },
                        "signing_config.v0.2.json": {
                            "length": len(signing_config),
                            "hashes": {
                                "sha256": hashlib.sha256(signing_config).hexdigest()
                            },
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (cache / "timestamp.json").write_text(
        json.dumps({"signed": {"version": timestamp_version}}), encoding="utf-8"
    )
    (cache.parent / f"{transparency.TUF_MIRROR}.json").write_text(
        json.dumps({"lastTimestamp": "2026-11-07T14:21:08.770496336Z"}),
        encoding="utf-8",
    )
    return cache


# -- the declaration and the reason --------------------------------------------


def test_the_reason_is_one_bounded_line_of_text() -> None:
    assert transparency.validate_reason("  rekor 500  ") == "rekor 500"
    for value in ("", "   ", None, "two\nlines", "x" * 301, "tab\there"):
        with pytest.raises(transparency.SigningTransparencyError):
            transparency.validate_reason(value)
    assert transparency.validate_reason("x" * 300) == "x" * 300


def test_the_admin_cli_mirror_spells_everything_the_same() -> None:
    """``src/gpu_fault`` may not hold the literal (env-inventory.json, which
    every component wheel ships); the mirrors must agree with the source."""

    assert (
        admin_signing.UNLOGGED_SIGNATURE_REASON_ENV
        == transparency.UNLOGGED_SIGNATURE_REASON_ENV
        == run_static_gates.UNLOGGED_SIGNATURE_REASON_ENV
    )
    assert admin_signing.ALLOW_UNLOGGED_FLAG == transparency.ALLOW_UNLOGGED_FLAG
    assert admin_signing.REASON_FLAG == transparency.REASON_FLAG
    assert admin_signing.REASON_MAX_CHARACTERS == transparency.REASON_MAX_CHARACTERS
    assert admin_signing.REFUSAL_MARKER_SUFFIX == transparency.REFUSAL_MARKER_SUFFIX
    assert admin_signing.PIN_RECORD == transparency.PIN_RECORD
    assert admin_signing.PIN_USE_RECORD == transparency.PIN_USE_RECORD
    assert admin_signing.PIN_AGE_WARNING_DAYS == transparency.PIN_AGE_WARNING_DAYS
    for value in ("ok", "  padded ", "", "two\nlines", "x" * 301):
        try:
            expected: object = transparency.validate_reason(value)
        except transparency.SigningTransparencyError:
            expected = ValueError
        try:
            actual: object = admin_signing.validate_reason(value)
        except admin_signing.UnloggedSignatureError:
            actual = ValueError
        assert actual == expected, value


def test_the_variable_never_enters_the_env_inventory_every_wheel_ships() -> None:
    inventory = (
        Path(transparency.__file__).resolve().parents[1]
        / "src/gpu_fault/data/env-inventory.json"
    ).read_text(encoding="utf-8")
    assert transparency.UNLOGGED_SIGNATURE_REASON_ENV not in inventory


def test_the_gate_environment_never_carries_the_consent() -> None:
    environment = run_static_gates.isolated_gate_environment(
        {transparency.UNLOGGED_SIGNATURE_REASON_ENV: "consent", "KEEP": "1"}
    )
    assert environment == {"KEEP": "1"}


def test_a_declaration_is_strict_and_a_logged_one_declares_nothing() -> None:
    declaration = transparency.unlogged_declaration(
        "reason", now=datetime(2026, 11, 7, 14, 30, tzinfo=UTC)
    )
    assert declaration == {
        "transparency_log": "unlogged",
        "reason": "reason",
        "declared_at": "2026-11-07T14:30:00Z",
    }
    assert transparency.payload_declaration({"signing": declaration}) == declaration
    assert transparency.payload_declaration({}) is None
    assert (
        transparency.payload_declaration({"signing": {"transparency_log": "logged"}})
        is None
    )
    for tampered in (
        {**declaration, "extra": "x"},
        {**declaration, "transparency_log": "skipped"},
        {**declaration, "declared_at": "yesterday"},
        {**declaration, "reason": ""},
        {**declaration, "reason": " padded"},
        "unlogged",
    ):
        with pytest.raises(transparency.SigningTransparencyError):
            transparency.payload_declaration({"signing": tampered})


def test_the_archive_declaration_lives_in_its_signed_manifest(tmp_path: Path) -> None:
    declaration = transparency.unlogged_declaration("reason")
    declared = _archive(tmp_path / "a.tar.gz", {"signing": declaration})
    plain = _archive(tmp_path / "b.tar.gz", {"schema_version": 2})
    assert transparency.artifact_declaration(declared) == declaration
    assert transparency.artifact_declaration(plain) is None
    (tmp_path / "c.tar.gz").write_bytes(b"not an archive")
    with pytest.raises(transparency.SigningTransparencyError):
        transparency.artifact_declaration(tmp_path / "c.tar.gz")


def test_only_a_parseable_bundle_without_entries_reads_as_unlogged(
    tmp_path: Path,
) -> None:
    cases = {
        "unlogged": UNLOGGED_BUNDLE,
        "logged": LOGGED_BUNDLE,
        "empty-object": {},
        "old-format": {"base64Signature": "x", "rekorBundle": {}},
    }
    expected = {
        "unlogged": "unlogged",
        "logged": "logged",
        "empty-object": "unknown",
        "old-format": "unknown",
    }
    for name, value in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        assert transparency.bundle_transparency(path) == expected[name], name
    (tmp_path / "garbage.json").write_text("{", encoding="utf-8")
    assert transparency.bundle_transparency(tmp_path / "garbage.json") == "unknown"
    assert transparency.bundle_transparency(tmp_path / "missing.json") == "unknown"


@pytest.mark.parametrize(
    ("stderr", "transient"),
    [
        (REKOR_500, True),
        (TUF_VERIFY_FAILURE, True),
        ('Post "https://timestamp.sigstore.dev/api/v1/timestamp": i/o timeout', True),
        ("[POST /api/v1/log/entries][503] createLogEntryDefault", True),
        (BAD_PASSWORD, False),
        ("[POST /api/v1/log/entries][409] createLogEntryConflict", False),
        ("Error: open dist/current-attestation.json: no such file", False),
        ("Error: status 500 from an unrelated registry", False),
    ],
)
def test_only_sigstore_service_failures_are_transient(
    stderr: str, transient: bool
) -> None:
    assert (transparency.transient_service_failure(stderr) is not None) is transient


# -- the signing wrapper: normal path, bounded retry, refusal -----------------


def _sign(
    cosign: FakeCosign, root: Path, *, key: str | None = "cosign.key", **options: Any
) -> cosign_sign_blob.SignResult:
    return cosign_sign_blob.sign_blob(
        Path("payload.json"),
        bundle=Path("payload.bundle.json"),
        key=key,
        cosign=str(cosign.path),
        environment=cosign.environment(root / "home"),
        cwd=root,
        echo=False,
        **options,
    )


def test_the_default_signing_command_is_unchanged(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(**{"sign-blob": [{"rc": 0, "bundle": LOGGED_BUNDLE}]})

    result = _sign(cosign, tmp_path)

    assert result.returncode == 0
    assert [call["argv"] for call in cosign.calls()] == [
        [
            "sign-blob",
            "--yes",
            "--key",
            "cosign.key",
            "--bundle",
            "payload.bundle.json",
            "payload.json",
        ]
    ]


def test_rekor_500_is_retried_then_refused_naming_the_switch(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(**{"sign-blob": [{"rc": 1, "stderr": REKOR_500}] * 3})

    result = _sign(cosign, tmp_path)

    assert len(cosign.calls()) == cosign_sign_blob.SIGN_ATTEMPTS == 3
    assert result.returncode == transparency.TRANSPARENCY_LOG_UNAVAILABLE_EXIT
    assert '--allow-unlogged-signature --reason "<why>"' in result.stderr
    assert "no bundle was written and nothing was signed" in result.stderr
    assert not (tmp_path / "payload.bundle.json").exists(), "a refused signing wrote"
    marker = transparency.read_refusal_marker(tmp_path / "payload.bundle.json")
    assert marker is not None and "status 500" in marker["cause"]
    assert marker["attempts"] == 3


def test_a_transient_failure_that_clears_signs_normally(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(
        **{
            "sign-blob": [
                {"rc": 1, "stderr": REKOR_500},
                {"rc": 0, "bundle": LOGGED_BUNDLE},
            ]
        }
    )

    result = _sign(cosign, tmp_path)

    assert result.returncode == 0 and len(cosign.calls()) == 2
    assert transparency.read_refusal_marker(tmp_path / "payload.bundle.json") is None


def test_any_other_failure_is_neither_retried_nor_rewritten(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(**{"sign-blob": [{"rc": 1, "stderr": BAD_PASSWORD}]})

    result = _sign(cosign, tmp_path)

    assert result.returncode == 1 and len(cosign.calls()) == 1
    assert "allow-unlogged-signature" not in result.stderr


def test_keyless_signing_is_retried_but_never_offered_the_unlogged_exit(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(**{"sign-blob": [{"rc": 1, "stderr": REKOR_500}] * 3})

    result = _sign(cosign, tmp_path, key=None)

    assert result.returncode == transparency.TRANSPARENCY_LOG_UNAVAILABLE_EXIT
    assert "allow-unlogged-signature" not in result.stderr
    assert "keyless signature always needs the transparency log" in result.stderr


def test_the_command_line_exits_75_for_make_and_leaves_the_marker(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(**{"sign-blob": [{"rc": 1, "stderr": REKOR_500}] * 3})

    returncode = cosign_sign_blob.main(
        [
            "--cosign",
            str(cosign.path),
            "--key",
            str(tmp_path / "cosign.key"),
            "--bundle",
            str(tmp_path / "payload.bundle.json"),
            str(tmp_path / "payload.json"),
        ]
    )

    assert returncode == transparency.TRANSPARENCY_LOG_UNAVAILABLE_EXIT
    assert transparency.read_refusal_marker(tmp_path / "payload.bundle.json"), (
        "make would report only the exit status"
    )


# -- the unlogged signing -----------------------------------------------------


def test_unlogged_signing_uses_the_key_alone_and_needs_the_declaration(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    _declared(tmp_path / "payload.json")
    cosign.configure(**{"sign-blob": [{"rc": 0, "bundle": UNLOGGED_BUNDLE}]})

    result = _sign(cosign, tmp_path, unlogged=True)

    assert result.returncode == 0 and result.unlogged
    ((call,),) = [[call["argv"] for call in cosign.calls()]]
    assert call == [
        "sign-blob",
        "--yes",
        "--key",
        "cosign.key",
        "--use-signing-config=false",
        "--tlog-upload=false",
        "--bundle",
        "payload.bundle.json",
        "payload.json",
    ]
    assert "WITHOUT a transparency-log entry" in result.stderr


@pytest.mark.parametrize("payload", ["undeclared", "logged", "tampered"])
def test_unlogged_signing_refuses_a_payload_that_does_not_declare_it(
    cosign: FakeCosign, tmp_path: Path, payload: str
) -> None:
    value: dict[str, Any] = {"subject": {}}
    if payload == "logged":
        value["signing"] = {"transparency_log": "logged"}
    if payload == "tampered":
        value["signing"] = {**transparency.unlogged_declaration("x"), "more": 1}
    (tmp_path / "payload.json").write_text(json.dumps(value), encoding="utf-8")

    result = _sign(cosign, tmp_path, unlogged=True)

    assert result.returncode == 2
    assert cosign.calls() == [], "cosign signed a payload without the declaration"


def test_unlogged_signing_is_never_keyless(cosign: FakeCosign, tmp_path: Path) -> None:
    _declared(tmp_path / "payload.json")
    result = _sign(cosign, tmp_path, key=None, unlogged=True)
    assert result.returncode == 2 and cosign.calls() == []


def test_an_unlogged_signing_that_still_produced_a_log_entry_is_refused(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    _declared(tmp_path / "payload.json")
    cosign.configure(**{"sign-blob": [{"rc": 0, "bundle": LOGGED_BUNDLE}]})
    assert _sign(cosign, tmp_path, unlogged=True).returncode == 2


def test_the_deploy_host_archive_declares_in_its_manifest(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    _archive(
        tmp_path / "bundle.tar.gz",
        {"signing": transparency.unlogged_declaration("rekor down")},
    )
    cosign.configure(**{"sign-blob": [{"rc": 0, "bundle": UNLOGGED_BUNDLE}]})
    result = cosign_sign_blob.sign_blob(
        Path("bundle.tar.gz"),
        bundle=Path("bundle.sigstore.json"),
        key="cosign.key",
        cosign=str(cosign.path),
        environment=cosign.environment(tmp_path / "home"),
        cwd=tmp_path,
        echo=False,
        unlogged=True,
    )
    assert result.returncode == 0


def test_every_makefile_signing_passes_the_unlogged_switch_explicitly() -> None:
    makefile = (Path(cosign_sign_blob.__file__).parents[1] / "Makefile").read_text(
        encoding="utf-8"
    )
    assert "$(if $(filter 1,$(UNLOGGED_SIGNATURE)),--unlogged,)" in makefile
    # The reason never travels on make's command line (make and the shell would
    # expand it); the builders read it from the environment the deploy sets.
    assert "UNLOGGED_SIGNATURE_REASON" not in makefile.replace(
        "UNLOGGED_SIGNATURE_DECLARE", ""
    )
    for target in ("release-build", "release-build-promoted", "release-build-staging"):
        recipe = makefile.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
        assert "$(UNLOGGED_SIGNATURE_DECLARE)" in recipe, target
    recipe = makefile.split("\ndeploy-host-bundle:\n", 1)[1].split("\n\n", 1)[0]
    assert "$(UNLOGGED_SIGNATURE_DECLARE)" in recipe


# -- the pinned Sigstore trust root -------------------------------------------


def test_the_fallback_strings_are_cosign_v312s_tuf_forms() -> None:
    assert transparency.TUF_FETCH_FAILURES == (
        "getting trusted root from TUF",
        "tuf refresh failed",
        "error getting signing config from TUF",
    )


@pytest.mark.parametrize(
    "stderr", [TUF_VERIFY_FAILURE, TUF_REFRESH_FAILURE, TUF_SIGNING_CONFIG_FAILURE]
)
def test_each_tuf_form_is_a_tuf_fetch_failure(stderr: str) -> None:
    assert transparency.tuf_fetch_failure(stderr) is not None


@pytest.mark.parametrize(
    "stderr",
    [
        REKOR_500,
        "Error: failed to verify signature: invalid signature",
        "Error: something unexpected",
        'Post "https://timestamp.sigstore.dev/api/v1/timestamp": status 503',
    ],
)
def test_no_other_failure_is_a_tuf_fetch_failure(stderr: str) -> None:
    assert transparency.tuf_fetch_failure(stderr) is None


def test_the_pin_is_only_beside_the_site_key(tmp_path: Path) -> None:
    directory = _site_key_directory(tmp_path / "state")
    assert transparency.pin_directory_for_key(directory / "cosign.pub") == directory
    assert transparency.pin_directory_for_key(directory / "cosign.key") == directory
    assert transparency.pin_directory_for_key("awskms:///alias/site") is None
    assert transparency.pin_directory_for_key(None) is None
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "cosign.pub").write_text("public", encoding="utf-8")
    assert transparency.pin_directory_for_key(loose / "cosign.pub") is None
    directory.chmod(0o770)
    try:
        assert transparency.pin_directory_for_key(directory / "cosign.pub") is None
    finally:
        directory.chmod(0o700)


def test_refresh_pins_only_what_matches_the_tuf_targets(tmp_path: Path) -> None:
    directory = _site_key_directory(tmp_path / "state")
    home = tmp_path / "home"
    lines: list[str] = []
    _tuf_cache(home, corrupt=True)
    assert (
        transparency.refresh_trust_pin(
            directory, environment={"HOME": str(home)}, log=lines.append
        )
        is None
    )
    assert transparency.load_trust_pin(directory) is None
    assert "does not match the TUF targets metadata" in lines[-1]

    _tuf_cache(home)
    record = transparency.refresh_trust_pin(
        directory, environment={"HOME": str(home)}, log=lines.append
    )
    assert record is not None
    assert record["tuf_versions"] == {"timestamp": 7, "targets": 3}
    assert record["tuf_last_timestamp"] == "2026-11-07T14:21:08.770496336Z"
    pin = transparency.load_trust_pin(directory)
    assert pin is not None
    assert (
        pin.trusted_root.read_bytes()
        == b'{"mediaType":"trusted-root","tlogs":["rekor"]}'
    )
    assert {path.stat().st_mode & 0o777 for path in directory.glob("sigstore-*")} == {
        0o600
    }


def test_refresh_never_takes_a_lower_tuf_version_and_keeps_the_previous_pin(
    tmp_path: Path,
) -> None:
    directory = _site_key_directory(tmp_path / "state")
    home = tmp_path / "home"
    lines: list[str] = []
    _tuf_cache(home, timestamp_version=9, targets_version=4)
    first = transparency.refresh_trust_pin(directory, environment={"HOME": str(home)})
    assert first is not None

    _tuf_cache(home, timestamp_version=8, targets_version=4, trusted_root=b'{"old":1}')
    assert (
        transparency.refresh_trust_pin(
            directory, environment={"HOME": str(home)}, log=lines.append
        )
        is None
    )
    assert "timestamp version 8 is lower than the pinned 9" in lines[-1]
    pin = transparency.load_trust_pin(directory)
    assert pin is not None and pin.record["files"] == first["files"]

    _tuf_cache(home, timestamp_version=10, targets_version=5, trusted_root=b'{"new":1}')
    assert transparency.refresh_trust_pin(directory, environment={"HOME": str(home)}), (
        "a higher TUF version was not pinned"
    )
    assert (directory / "sigstore-trusted-root.json").read_bytes() == b'{"new":1}'
    assert (directory / "sigstore-trusted-root.json.prev").read_bytes() == (
        b'{"mediaType":"trusted-root","tlogs":["rekor"]}'
    )
    assert (
        json.loads((directory / "sigstore-trust-pin.json.prev").read_bytes()) == first
    )


def test_a_failed_re_pin_never_raises(tmp_path: Path) -> None:
    directory = _site_key_directory(tmp_path / "state")
    lines: list[str] = []
    assert (
        transparency.refresh_trust_pin(
            directory, environment={"HOME": str(tmp_path / "nothing")}, log=lines.append
        )
        is None
    )
    assert lines and "not re-pinned" in lines[-1]
    (directory / transparency.PIN_RECORD).write_text('{"files": []}', encoding="utf-8")
    _tuf_cache(tmp_path / "home")
    transparency.refresh_trust_pin(
        directory, environment={"HOME": str(tmp_path / "home")}, log=lines.append
    )


def test_a_pin_whose_file_changed_is_no_pin(tmp_path: Path) -> None:
    directory = _site_key_directory(tmp_path / "state")
    _tuf_cache(tmp_path / "home")
    assert transparency.refresh_trust_pin(
        directory, environment={"HOME": str(tmp_path / "home")}
    ), "the TUF cache fixture was not pinned"
    (directory / "sigstore-trusted-root.json").write_bytes(b'{"attacker":1}')
    assert transparency.load_trust_pin(directory) is None


def _pinned(tmp_path: Path) -> Path:
    directory = _site_key_directory(tmp_path / "state")
    _tuf_cache(tmp_path / "home")
    assert transparency.refresh_trust_pin(
        directory, environment={"HOME": str(tmp_path / "home")}
    ), "the TUF cache fixture was not pinned"
    return directory


def _verify(cosign: FakeCosign, tmp_path: Path, directory: Path | None) -> Any:
    return transparency.run_verify_blob(
        [
            str(cosign.path),
            "verify-blob",
            "--key",
            "k.pub",
            "--bundle",
            "b",
            "artifact",
        ],
        pin_directory=directory,
        environment=cosign.environment(tmp_path / "home"),
        log=lambda _line: None,
    )


@pytest.mark.parametrize(
    "stderr", [TUF_VERIFY_FAILURE, TUF_REFRESH_FAILURE, TUF_SIGNING_CONFIG_FAILURE]
)
def test_a_tuf_failure_verifies_once_more_against_the_pin(
    cosign: FakeCosign, tmp_path: Path, stderr: str
) -> None:
    directory = _pinned(tmp_path)
    cosign.configure(**{"verify-blob": [{"rc": 1, "stderr": stderr}, {"rc": 0}]})

    outcome = _verify(cosign, tmp_path, directory)

    assert outcome.returncode == 0 and outcome.pinned is not None
    first, second = (call["argv"] for call in cosign.calls())
    assert "--trusted-root" not in first
    assert second == [
        "verify-blob",
        "--key",
        "k.pub",
        "--bundle",
        "b",
        "--trusted-root",
        str(directory / "sigstore-trusted-root.json"),
        "artifact",
    ]
    usage = outcome.pinned.usage()
    assert usage["trusted_root"] == "pinned"
    assert set(usage["pin_sha256"]) == {
        "sigstore-trusted-root.json",
        "sigstore-signing-config.json",
    }
    assert usage["tuf_last_timestamp"] == "2026-11-07T14:21:08.770496336Z"
    assert usage["pin_older_than_warning"] is False


@pytest.mark.parametrize(
    "stderr", [REKOR_500, "Error: failed to verify signature", "Error: other"]
)
def test_no_other_verification_failure_touches_the_pin(
    cosign: FakeCosign, tmp_path: Path, stderr: str
) -> None:
    directory = _pinned(tmp_path)
    cosign.configure(**{"verify-blob": [{"rc": 1, "stderr": stderr}, {"rc": 0}]})
    outcome = _verify(cosign, tmp_path, directory)
    assert outcome.returncode == 1 and outcome.pinned is None
    assert len(cosign.calls()) == 1


@pytest.mark.parametrize("pin", ["absent", "mismatched"])
def test_without_a_usable_pin_a_tuf_failure_fails_as_today(
    cosign: FakeCosign, tmp_path: Path, pin: str
) -> None:
    directory = _pinned(tmp_path)
    if pin == "absent":
        for path in directory.glob("sigstore-*"):
            path.unlink()
    else:
        (directory / "sigstore-signing-config.json").write_bytes(b"{}")
    cosign.configure(**{"verify-blob": [{"rc": 1, "stderr": TUF_VERIFY_FAILURE}]})
    outcome = _verify(cosign, tmp_path, directory)
    assert outcome.returncode == 1 and outcome.pinned is None
    assert outcome.stderr == TUF_VERIFY_FAILURE
    assert len(cosign.calls()) == 1


def test_a_successful_online_verification_re_pins(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    directory = _site_key_directory(tmp_path / "state")
    _tuf_cache(tmp_path / "home")
    cosign.configure(**{"verify-blob": [{"rc": 0}]})
    assert _verify(cosign, tmp_path, directory).returncode == 0
    assert transparency.load_trust_pin(directory) is not None


def test_a_tuf_failure_signs_with_the_pinned_service_list(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    directory = _pinned(tmp_path)
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(
        **{
            "sign-blob": [
                {"rc": 1, "stderr": TUF_SIGNING_CONFIG_FAILURE},
                {"rc": 0, "bundle": LOGGED_BUNDLE},
            ]
        }
    )

    result = cosign_sign_blob.sign_blob(
        Path("payload.json"),
        bundle=Path("payload.bundle.json"),
        key=str(directory / "cosign.key"),
        cosign=str(cosign.path),
        environment=cosign.environment(tmp_path / "home"),
        cwd=tmp_path,
        echo=False,
    )

    assert result.returncode == 0 and result.trust_pin is not None
    second = cosign.calls()[1]["argv"]
    assert second[second.index("--signing-config") + 1] == str(
        directory / "sigstore-signing-config.json"
    )
    assert second[second.index("--trusted-root") + 1] == str(
        directory / "sigstore-trusted-root.json"
    )
    use = json.loads((directory / transparency.PIN_USE_RECORD).read_bytes())
    assert use["trusted_root"] == "pinned"


def test_a_rekor_failure_never_signs_with_the_pin(
    cosign: FakeCosign, tmp_path: Path
) -> None:
    directory = _pinned(tmp_path)
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    cosign.configure(**{"sign-blob": [{"rc": 1, "stderr": REKOR_500}] * 3})
    result = cosign_sign_blob.sign_blob(
        Path("payload.json"),
        bundle=Path("payload.bundle.json"),
        key=str(directory / "cosign.key"),
        cosign=str(cosign.path),
        environment=cosign.environment(tmp_path / "home"),
        cwd=tmp_path,
        echo=False,
    )
    assert result.returncode == 75
    assert all("--trusted-root" not in call["argv"] for call in cosign.calls()), (
        "a transparency-log failure used the pinned trust root"
    )


def test_a_pin_older_than_thirty_days_is_flagged_at_use(tmp_path: Path) -> None:
    directory = _pinned(tmp_path)
    pin = transparency.load_trust_pin(directory)
    assert pin is not None
    later = datetime.now(UTC) + timedelta(days=31)
    assert pin.usage(later)["pin_older_than_warning"] is True
    assert pin.usage()["pin_older_than_warning"] is False
