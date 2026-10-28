"""``scripts/cosign_sign_blob.py``: Rekor's 409 for this signing's own entry.

GF-REGIONAL-BOOT-020 failed in round 13 when ``make release-build-staging``
signed candidate D: Rekor integrated the upload, cosign's client re-sent it,
the re-send got ``409 createLogEntryConflict`` naming that same entry and
cosign v3.1.2 exited 1 without a bundle. The tests run the wrapper against a
fake ``cosign`` and a fake Rekor on localhost; nothing signs or uploads.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from scripts import cosign_sign_blob, staging_deploy
from scripts.staging_state_hygiene import SigningMaterial, SourceCheckout

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/cosign_sign_blob.py"
PASSWORD = "test-only-signing-password"
# Throwaway P-256 public keys; their private halves were discarded.
SIGNER_PEM = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEESFXE3kS1dflKaBQx+IWJbgs3xXM
jME6AXcluBD9Ej5GBq7VS0tjJDXOoMNq3W71kVapUjOkWTbxblbx0+ZHtQ==
-----END PUBLIC KEY-----
"""
OTHER_PEM = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEICA36zEfcu43Kq2tydHqbsf7eITF
Tl+zsBk2pz4wst4uzKfYADTKgKWrD1h0CWC/39nX237Zd5JKmdl2gRyu0A==
-----END PUBLIC KEY-----
"""
TREE_ID = "108e9186e8c5677a"
LOG_ID = "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d"
CHECKPOINT = "rekor.sigstore.dev - 1193050959916656506\n3416259420\nroot\n"
FAKE_COSIGN = """\
import json, os, sys
from pathlib import Path

plan = json.loads(Path(os.environ["FAKE_COSIGN_PLAN"]).read_text())
argv = sys.argv[1:]
record = {"argv": argv, "password": os.environ.get("COSIGN_PASSWORD")}
if argv[0] == "verify-blob":
    staged = Path(argv[argv.index("--bundle") + 1])
    record["bundle"] = staged.read_text()
    record["bundle_mode"] = staged.stat().st_mode & 0o777
with Path(os.environ["FAKE_COSIGN_LOG"]).open("a") as log:
    log.write(json.dumps(record) + "\\n")
outcome = plan[argv[0]]
if argv[0] == "sign-blob" and outcome == "ok":
    Path(argv[argv.index("--bundle") + 1]).write_text("{}")
if argv[0] == "public-key" and outcome == "ok":
    sys.stdout.write(plan["public_key_pem"])
sys.stderr.write(plan.get(argv[0] + "_stderr", ""))
raise SystemExit(0 if outcome == "ok" else 1)
"""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def conflict_error(uuid: str) -> str:
    """cosign v3.1.2's output, as the BOOT-020 log recorded it."""

    return (
        "Using payload from: dist/current-attestation.json\n"
        "Signing artifact...\n"
        "Error: signing dist/current-attestation.json: signing bundle: error "
        "signing bundle: [POST /api/v1/log/entries][409] createLogEntryConflict "
        '{"code":409,"message":"an equivalent entry already exists in the '
        f'transparency log with UUID {uuid}"}}\n'
        "error during command execution: signing dist/current-attestation.json: "
        "signing bundle: ...\n"
    )


def log_entry(
    artifact: bytes,
    *,
    key_pem: str = SIGNER_PEM,
    kind: str = "hashedrekord",
    digest: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """A Rekor v1 GET answer for ``artifact``: (UUID, LogEntryAnon)."""

    body = json.dumps(
        {
            "apiVersion": "0.0.1",
            "kind": kind,
            "spec": {
                "data": {
                    "hash": {
                        "algorithm": "sha256",
                        "value": digest or hashlib.sha256(artifact).hexdigest(),
                    }
                },
                "signature": {
                    "content": _b64(b"signature over the artifact"),
                    "publicKey": {"content": _b64(key_pem.encode())},
                },
            },
        },
        separators=(",", ":"),
    ).encode()
    uuid = TREE_ID + hashlib.sha256(b"\x00" + body).hexdigest()
    return uuid, {
        "body": _b64(body),
        "integratedTime": 1793189892,
        "logID": LOG_ID,
        "logIndex": 3538163678,
        "verification": {
            "inclusionProof": {
                "checkpoint": CHECKPOINT,
                "hashes": ["ab" * 32, "cd" * 32],
                "logIndex": 3416259416,
                "rootHash": "ef" * 32,
                "treeSize": 3416259420,
            },
            "signedEntryTimestamp": "MEUCIQDsignedEntryTimestamp==",
        },
    }


@dataclass
class FakeRekor:
    url: str
    answers: list[tuple[int, Any]] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)


@pytest.fixture
def rekor() -> Iterator[FakeRekor]:
    state = FakeRekor(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.requests.append(self.path)
            status, payload = (
                state.answers.pop(0) if state.answers else (500, {"code": 500})
            )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_arguments: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@dataclass
class Signing:
    root: Path
    cosign: Path
    plan: Path
    log: Path
    artifact: Path
    bundle: Path
    key: Path

    def configure(self, **plan: str) -> None:
        values = {
            "sign-blob": "ok",
            "public-key": "ok",
            "verify-blob": "ok",
            "public_key_pem": SIGNER_PEM,
            **plan,
        }
        self.plan.write_text(json.dumps(values), encoding="utf-8")

    def environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "COSIGN_PASSWORD": PASSWORD,
            "FAKE_COSIGN_PLAN": str(self.plan),
            "FAKE_COSIGN_LOG": str(self.log),
        }

    def calls(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def sign(self, rekor: FakeRekor, *, key: str | None = "cosign.key") -> Any:
        return cosign_sign_blob.sign_blob(
            Path("dist/current-attestation.json"),
            bundle=Path("dist/current-attestation.bundle.json"),
            key=key,
            cosign=str(self.cosign),
            rekor_url=rekor.url,
            environment=self.environment(),
            cwd=self.root,
            echo=False,
        )


@pytest.fixture
def signing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Signing:
    monkeypatch.setattr(cosign_sign_blob, "FETCH_RETRY_SECONDS", 0.0)
    root = tmp_path / "repository"
    (root / "dist").mkdir(parents=True)
    artifact = root / "dist/current-attestation.json"
    artifact.write_text('{"subject":{"release_id":"214673cbd03e"}}\n', "utf-8")
    cosign = tmp_path / "cosign"
    cosign.write_text(f"#!{sys.executable}\n{FAKE_COSIGN}", encoding="utf-8")
    cosign.chmod(0o755)
    value = Signing(
        root=root,
        cosign=cosign,
        plan=tmp_path / "plan.json",
        log=tmp_path / "cosign-calls.jsonl",
        artifact=artifact,
        bundle=root / "dist/current-attestation.bundle.json",
        key=root / "cosign.key",
    )
    value.configure()
    return value


def test_the_boot020_conflict_line_names_its_entry() -> None:
    uuid = TREE_ID + "392b2659ffef2fc3bec7bd6672fda01cb89047d3c12f9d10f5a573f388d37de2"

    assert cosign_sign_blob.conflict_entry_uuid(conflict_error(uuid)) == uuid
    assert cosign_sign_blob.conflict_entry_uuid(conflict_error(uuid[16:])) == uuid[16:]
    assert cosign_sign_blob.conflict_entry_uuid("Error: [409] conflict") is None
    assert (
        cosign_sign_blob.conflict_entry_uuid(
            f"[POST /api/v1/log/entries][400] createLogEntryBadRequest with UUID {uuid}"
        )
        is None
    )


def test_a_successful_signing_passes_through_untouched(
    signing: Signing, rekor: FakeRekor
) -> None:
    result = signing.sign(rekor)

    assert result.returncode == 0
    assert result.recovered_entry is None
    assert [call["argv"] for call in signing.calls()] == [
        [
            "sign-blob",
            "--yes",
            "--key",
            "cosign.key",
            "--bundle",
            "dist/current-attestation.bundle.json",
            "dist/current-attestation.json",
        ]
    ]
    assert signing.calls()[0]["password"] == PASSWORD
    assert rekor.requests == []


def test_any_other_failure_keeps_cosign_status_and_output(
    signing: Signing, rekor: FakeRekor
) -> None:
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": "Error: no key\n"})

    result = signing.sign(rekor)

    assert result.returncode == 1
    assert result.stderr == "Error: no key\n"
    assert not signing.bundle.exists(), "a failed signing left a bundle"
    assert rekor.requests == []


def test_rekor_409_for_this_signing_becomes_the_verified_bundle(
    signing: Signing, rekor: FakeRekor
) -> None:
    uuid, entry = log_entry(signing.artifact.read_bytes())
    rekor.answers.append((200, {uuid: entry}))
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": conflict_error(uuid)})

    result = signing.sign(rekor)

    assert result.returncode == 0, result.stderr
    assert result.recovered_entry == uuid
    assert f"entry {uuid} (integrated 2026-10-28T12:18:12Z)" in result.stderr
    assert rekor.requests == [f"/api/v1/log/entries/{uuid}"]
    sign, public_key, verify = signing.calls()
    assert sign["argv"][0] == "sign-blob"
    assert public_key["argv"] == ["public-key", "--key", "cosign.key"]
    assert public_key["password"] == PASSWORD
    assert verify["argv"][0:2] == ["verify-blob", "--key"]
    assert verify["argv"][-1] == str(signing.artifact)
    assert verify["password"] is None, "verify-blob must not see COSIGN_PASSWORD"
    assert verify["bundle_mode"] == 0o600
    assert signing.bundle.read_text(encoding="utf-8") == verify["bundle"], (
        "the installed bundle is not the one cosign verify-blob accepted"
    )
    assert sorted(path.name for path in signing.bundle.parent.iterdir()) == [
        "current-attestation.bundle.json",
        "current-attestation.json",
    ]
    assert signing.bundle.stat().st_mode & 0o777 == 0o600
    bundle = json.loads(verify["bundle"])
    der = base64.b64decode("".join(SIGNER_PEM.splitlines()[1:-1]))
    proof = entry["verification"]["inclusionProof"]
    assert bundle == {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "publicKey": {"hint": _b64(hashlib.sha256(der).digest())},
            "tlogEntries": [
                {
                    "logIndex": "3538163678",
                    "logId": {"keyId": _b64(bytes.fromhex(LOG_ID))},
                    "kindVersion": {"kind": "hashedrekord", "version": "0.0.1"},
                    "integratedTime": "1793189892",
                    "inclusionPromise": {
                        "signedEntryTimestamp": "MEUCIQDsignedEntryTimestamp=="
                    },
                    "inclusionProof": {
                        "logIndex": "3416259416",
                        "rootHash": _b64(bytes.fromhex(proof["rootHash"])),
                        "treeSize": "3416259420",
                        "hashes": [
                            _b64(bytes.fromhex(item)) for item in proof["hashes"]
                        ],
                        "checkpoint": {"envelope": CHECKPOINT},
                    },
                    "canonicalizedBody": entry["body"],
                }
            ],
        },
        "messageSignature": {
            "messageDigest": {
                "algorithm": "SHA2_256",
                "digest": _b64(hashlib.sha256(signing.artifact.read_bytes()).digest()),
            },
            "signature": _b64(b"signature over the artifact"),
        },
    }


def _foreign_digest(artifact: bytes) -> tuple[str, dict[str, Any]]:
    return log_entry(artifact, digest=hashlib.sha256(b"other").hexdigest())


def _foreign_key(artifact: bytes) -> tuple[str, dict[str, Any]]:
    return log_entry(artifact, key_pem=OTHER_PEM)


def _other_kind(artifact: bytes) -> tuple[str, dict[str, Any]]:
    return log_entry(artifact, kind="dsse")


def _body_not_named_by_uuid(artifact: bytes) -> tuple[str, dict[str, Any]]:
    uuid, entry = log_entry(artifact)
    _other_uuid, other = log_entry(artifact, key_pem=OTHER_PEM)
    return uuid, {**entry, "body": other["body"]}


def _no_inclusion_proof(artifact: bytes) -> tuple[str, dict[str, Any]]:
    uuid, entry = log_entry(artifact)
    verification = {"signedEntryTimestamp": "MEUCIQ=="}
    return uuid, {**entry, "verification": verification}


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        (_foreign_digest, "is not for this artifact"),
        (_foreign_key, "signed by another key"),
        (_other_kind, "is not a hashedrekord 0.0.1"),
        (_body_not_named_by_uuid, "does not hash to the UUID"),
        (_no_inclusion_proof, "no inclusion proof"),
    ],
)
def test_an_entry_that_is_not_this_signature_fails_closed(
    signing: Signing, rekor: FakeRekor, variant: Any, reason: str
) -> None:
    uuid, entry = variant(signing.artifact.read_bytes())
    rekor.answers.append((200, {uuid: entry}))
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": conflict_error(uuid)})

    result = signing.sign(rekor)

    assert result.returncode == 1
    assert f"cannot use Rekor entry {uuid}: " in result.stderr
    assert reason in result.stderr
    assert [call["argv"][0] for call in signing.calls()] == ["sign-blob", "public-key"]
    assert not signing.bundle.exists(), "an unbound entry became a bundle"


def test_a_rebuilt_bundle_cosign_rejects_is_never_installed(
    signing: Signing, rekor: FakeRekor
) -> None:
    uuid, entry = log_entry(signing.artifact.read_bytes())
    rekor.answers.append((200, {uuid: entry}))
    signing.configure(
        **{
            "sign-blob": "fail",
            "sign-blob_stderr": conflict_error(uuid),
            "verify-blob": "fail",
            "verify-blob_stderr": "Error: invalid signature\n",
        }
    )

    result = signing.sign(rekor)

    assert result.returncode == 1
    assert "cosign verify-blob rejected the rebuilt bundle: Error: invalid" in (
        result.stderr
    )
    assert sorted(path.name for path in signing.bundle.parent.iterdir()) == [
        "current-attestation.json"
    ]


def test_keyless_signing_is_not_recovered(signing: Signing, rekor: FakeRekor) -> None:
    uuid, _entry = log_entry(signing.artifact.read_bytes())
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": conflict_error(uuid)})

    result = signing.sign(rekor, key=None)

    assert result.returncode == 1
    assert signing.calls()[0]["argv"] == [
        "sign-blob",
        "--yes",
        "--bundle",
        "dist/current-attestation.bundle.json",
        "dist/current-attestation.json",
    ]
    assert len(signing.calls()) == 1
    assert rekor.requests == []


def test_the_rekor_read_retries_a_lagging_or_failing_read(
    signing: Signing, rekor: FakeRekor
) -> None:
    uuid, entry = log_entry(signing.artifact.read_bytes())
    rekor.answers.extend([(404, {"code": 404}), (503, {}), (200, {uuid: entry})])
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": conflict_error(uuid)})

    assert signing.sign(rekor).returncode == 0
    assert len(rekor.requests) == 3


def test_an_unreadable_entry_fails_closed_after_bounded_retries(
    signing: Signing, rekor: FakeRekor
) -> None:
    uuid, _entry = log_entry(signing.artifact.read_bytes())
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": conflict_error(uuid)})

    result = signing.sign(rekor)

    assert result.returncode == 1
    assert "cannot read the Rekor entry (HTTP 500)" in result.stderr
    assert len(rekor.requests) == cosign_sign_blob.FETCH_ATTEMPTS
    assert not signing.bundle.exists(), "an unread entry became a bundle"


def test_the_command_line_recovers_as_make_runs_it(
    signing: Signing, rekor: FakeRekor
) -> None:
    uuid, entry = log_entry(signing.artifact.read_bytes())
    rekor.answers.append((200, {uuid: entry}))
    signing.configure(**{"sign-blob": "fail", "sign-blob_stderr": conflict_error(uuid)})

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cosign",
            str(signing.cosign),
            "--key",
            "cosign.key",
            "--bundle",
            "dist/current-attestation.bundle.json",
            "--rekor-url",
            rekor.url,
            "dist/current-attestation.json",
        ],
        cwd=signing.root,
        env=signing.environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "createLogEntryConflict" in completed.stderr, "cosign output was hidden"
    assert f"Rekor answered 409, entry {uuid}" in completed.stderr
    assert json.loads(signing.bundle.read_text(encoding="utf-8"))["mediaType"] == (
        cosign_sign_blob.BUNDLE_MEDIA_TYPE
    )


def test_every_makefile_sign_blob_runs_through_the_wrapper() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert (
        'COSIGN_SIGN_BLOB = $(PYTHON) scripts/cosign_sign_blob.py --cosign "$(COSIGN)"'
        in makefile
    )
    assert "$(COSIGN) sign-blob" not in makefile
    assert "\tcosign sign-blob" not in makefile
    for target in (
        "release-build",
        "release-build-promoted",
        "release-build-staging",
        "deploy-host-bundle",
        "deploy-host-sign",
    ):
        recipe = makefile.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
        assert "\t$(COSIGN_SIGN_BLOB) \\\n\t\t--bundle " in recipe, target


def test_the_source_deploy_record_is_signed_through_the_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(arguments: list[str], **options: Any) -> str:
        calls.append((list(arguments), dict(options["env"])))
        bundle = Path(arguments[arguments.index("--bundle") + 1])
        bundle.write_text("{}", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    recorded = staging_deploy.record_successful_source_deploy(
        state,
        source_repository_root=tmp_path,
        source=SourceCheckout(
            repository_root=tmp_path,
            git_commit="a" * 40,
            fingerprint="f" * 64,
            snapshot=False,
        ),
        identities={
            "application": {"sha256": "1" * 64},
            "deploy_host": {"sha256": "2" * 64, "bundle": {"sha256": "3" * 64}},
        },
        signing=SigningMaterial(
            private_key=tmp_path / "cosign.key",
            public_key=tmp_path / "cosign.pub",
            password_file=tmp_path / "cosign.password",
            password=PASSWORD,
        ),
        mode="deploy",
        live_evidence={},
    )

    ((arguments, environment),) = calls
    assert arguments[:2] == [sys.executable, str(SCRIPT)]
    assert arguments[2:4] == ["--key", str(tmp_path / "cosign.key")]
    assert environment["COSIGN_PASSWORD"] == PASSWORD
    assert recorded.is_file(), "the signed source deploy record was not written"
