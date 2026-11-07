"""Release signatures without a transparency-log entry, and the pinned trust root.

The site signs four things with its own cosign key: the release attestation,
the deploy-host archive, the CI candidate receipt and the successful source
deploy record (``scripts/cosign_sign_blob.py``). By default each signature is
uploaded to the public Sigstore transparency log (Rekor) and every verifier
requires that entry. Two public services can therefore stop a deploy that has
nothing wrong with it: Rekor (live 2026-11-07, ``status 500 ... unexpected
result from transparency log``) and the Sigstore TUF repository that every
default ``cosign verify-blob`` and ``cosign sign-blob`` reads first.

The log stays the default. This module is what both exits share:

* **Unlogged signing.** ``gpu-fault-admin deploy --allow-unlogged-signature
  --reason TEXT`` signs with the site key only. Each signed payload then
  carries the declaration ``"signing": {"transparency_log": "unlogged",
  "reason": ..., "declared_at": ...}`` -- inside the signed bytes, so it
  cannot be added, removed or edited without breaking the signature. A
  verifier takes a bundle without a log entry only with that declaration
  *and* an approval a key holder alone cannot produce. The approval is either
  the operator's switch on the running command, or the cluster's own record
  that this exact release is the one deployed (see ``release_attestation``).
  ``cosign verify-blob --insecure-ignore-tlog=true --key`` then checks the
  signature offline.
* **The pinned trust root.** After every online cosign run that read TUF, the
  TUF-verified ``trusted_root.json`` and ``signing_config.v0.2.json`` are
  copied from cosign's own TUF cache into ``<state-dir>/release-signing/``
  (sha256 checked against the cached TUF targets metadata, never a lower
  metadata version, previous pin kept as ``.prev``). Only when cosign's live
  TUF fetch fails does a verifier rerun with ``--trusted-root <pin>``, and a
  signer with ``--signing-config <pin> --trusted-root <pin>``. A logged bundle
  carries its Rekor inclusion proof and signed entry timestamp, so it verifies
  offline against the pin. Without a usable pin the command fails as before.

Standard library only: ``scripts/setup_deploy_host.py`` imports it with the
system Python before any venv exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

#: The reason the deploy's hop hands its own children (make, the signing
#: wrapper, the attestation builder). An exported value alone does nothing:
#: every signer and verifier also needs its explicit flag, and the deploy
#: command scrubs an inherited value before it sets its own.
#: ``gpu_fault_release.regional_release_signing`` spells it for the admin CLI
#: (a literal under ``src/gpu_fault`` would enter ``env-inventory.json``, which
#: every component wheel ships); a test pins the two equal.
UNLOGGED_SIGNATURE_REASON_ENV = "GPU_FAULT_RELEASE_UNLOGGED_SIGNATURE_REASON"
ALLOW_UNLOGGED_FLAG = "--allow-unlogged-signature"
REASON_FLAG = "--reason"
DECLARATION_FIELD = "signing"
UNLOGGED = "unlogged"
LOGGED = "logged"
UNKNOWN = "unknown"
REASON_MAX_CHARACTERS = 300
#: What makes ``cosign verify-blob`` accept a bundle without a log entry; added
#: only after the declaration and the approval were checked.
IGNORE_TLOG_FLAG = "--insecure-ignore-tlog=true"
#: ``cosign sign-blob`` with the key only. cosign v3.1.2 refuses
#: ``--tlog-upload=false`` while ``--use-signing-config`` is on (its default),
#: and a signing config without a log still reads TUF; these two flags together
#: contact no Sigstore service at all (checked in a network namespace without
#: interfaces).
UNLOGGED_SIGN_FLAGS = ("--use-signing-config=false", "--tlog-upload=false")
#: The exit status of a signing the transparency log refused after retries
#: (``EX_TEMPFAIL``); make reports it as ``Error 75``.
TRANSPARENCY_LOG_UNAVAILABLE_EXIT = 75
DEPLOY_HOST_ARCHIVE_SUFFIX = ".tar.gz"
DEPLOY_HOST_MANIFEST = "manifest.json"
_MANIFEST_LIMIT = 64 * 1024 * 1024
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_DECLARATION_KEYS = frozenset({"transparency_log", "reason", "declared_at"})


class SigningTransparencyError(ValueError):
    """A declaration, a consent or a bundle that cannot be accepted."""


def utc_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def validate_reason(value: object) -> str:
    """The operator's reason: one line of printable text, 1-300 characters."""

    if not isinstance(value, str) or not value.strip():
        raise SigningTransparencyError(
            f"{ALLOW_UNLOGGED_FLAG} requires {REASON_FLAG} saying why the release "
            "is signed without a transparency-log entry"
        )
    text = value.strip()
    if len(text) > REASON_MAX_CHARACTERS:
        raise SigningTransparencyError(
            f"{REASON_FLAG} is longer than {REASON_MAX_CHARACTERS} characters"
        )
    if not text.isprintable():
        raise SigningTransparencyError(
            f"{REASON_FLAG} must be one line of printable text"
        )
    return text


def unlogged_declaration(reason: str, *, now: datetime | None = None) -> dict[str, str]:
    """The block a payload signed without a log entry carries."""

    return {
        "transparency_log": UNLOGGED,
        "reason": validate_reason(reason),
        "declared_at": utc_timestamp(now),
    }


def parse_declaration(value: object) -> dict[str, str]:
    """``value`` as a well-formed unlogged declaration, or an error."""

    if not isinstance(value, Mapping) or set(value) != _DECLARATION_KEYS:
        raise SigningTransparencyError(
            "signing declaration must have exactly transparency_log, reason "
            "and declared_at"
        )
    if value["transparency_log"] != UNLOGGED:
        raise SigningTransparencyError(
            "signing declaration transparency_log must be unlogged"
        )
    reason = validate_reason(value["reason"])
    if reason != value["reason"]:
        raise SigningTransparencyError("signing declaration reason is not normalized")
    if parse_timestamp(value["declared_at"]) is None:
        raise SigningTransparencyError(
            "signing declaration declared_at is not a UTC timestamp"
        )
    return {
        "transparency_log": UNLOGGED,
        "reason": reason,
        "declared_at": str(value["declared_at"]),
    }


#: What a payload signed with a log entry may say instead (the successful
#: source deploy record always names its mode).
LOGGED_DECLARATION = {"transparency_log": LOGGED}


def logged_declaration() -> dict[str, str]:
    return dict(LOGGED_DECLARATION)


def payload_declaration(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """The payload's unlogged declaration; None without one; error if malformed.

    No ``signing`` field, or exactly ``{"transparency_log": "logged"}``, is no
    unlogged declaration. Anything else must be a well-formed one.
    """

    if DECLARATION_FIELD not in payload:
        return None
    if payload[DECLARATION_FIELD] == LOGGED_DECLARATION:
        return None
    return parse_declaration(payload[DECLARATION_FIELD])


def _archive_manifest(archive: Path) -> Mapping[str, Any]:
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = [
                member
                for member in bundle.getmembers()
                if member.isfile()
                and len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).name == DEPLOY_HOST_MANIFEST
            ]
            if len(members) != 1 or members[0].size > _MANIFEST_LIMIT:
                raise SigningTransparencyError(
                    f"{archive.name} has no single bundle manifest"
                )
            source = bundle.extractfile(members[0])
            if source is None:
                raise SigningTransparencyError(f"{archive.name} manifest is unreadable")
            data = source.read()
    except (OSError, tarfile.TarError) as exc:
        raise SigningTransparencyError(f"{archive.name} is not readable") from exc
    try:
        value = json.loads(data)
    except ValueError as exc:
        raise SigningTransparencyError(f"{archive.name} manifest is not JSON") from exc
    if not isinstance(value, Mapping):
        raise SigningTransparencyError(f"{archive.name} manifest is not an object")
    return value


def artifact_declaration(artifact: Path) -> dict[str, str] | None:
    """The declaration a signed artifact carries: a JSON payload's own
    ``signing`` field, or the deploy-host archive's ``manifest.json`` one."""

    if artifact.name.endswith(DEPLOY_HOST_ARCHIVE_SUFFIX):
        return payload_declaration(_archive_manifest(artifact))
    try:
        value = json.loads(artifact.read_bytes())
    except (OSError, ValueError) as exc:
        raise SigningTransparencyError(
            f"{artifact.name} is not a JSON payload"
        ) from exc
    if not isinstance(value, Mapping):
        raise SigningTransparencyError(f"{artifact.name} is not a JSON object")
    return payload_declaration(value)


def bundle_transparency(bundle: Path) -> str:
    """``logged`` or ``unlogged`` for a Sigstore bundle; ``unknown`` otherwise.

    Only a parseable bundle with a message signature and no log entry is
    ``unlogged``. Anything else -- a logged bundle, another format, a file that
    does not parse -- keeps today's verification command unchanged and lets
    cosign decide, so the default path cannot get weaker by misreading.
    """

    try:
        value = json.loads(bundle.read_bytes())
    except (OSError, ValueError):
        return UNKNOWN
    if not isinstance(value, Mapping) or not isinstance(
        value.get("messageSignature"), Mapping
    ):
        return UNKNOWN
    material = value.get("verificationMaterial")
    if not isinstance(material, Mapping):
        return UNKNOWN
    entries = material.get("tlogEntries")
    if entries is None or entries == []:
        return UNLOGGED
    if isinstance(entries, list):
        return LOGGED
    return UNKNOWN


def require_unlogged_acceptance(
    *,
    artifact: Path,
    declaration: Mapping[str, str] | None,
    keyed: bool,
    approved: bool,
    approval_hint: str,
) -> None:
    """Refuse a bundle without a log entry unless every condition holds."""

    if not keyed:
        raise SigningTransparencyError(
            f"{artifact.name} has no transparency-log entry; a keyless signature "
            "is never accepted without one"
        )
    if declaration is None:
        raise SigningTransparencyError(
            f"{artifact.name} has no transparency-log entry and its signed "
            "payload declares no unlogged signing; refused"
        )
    if not approved:
        raise SigningTransparencyError(
            f"{artifact.name} is signed without a transparency-log entry "
            f"(declared {declaration['declared_at']}: {declaration['reason']}); "
            f"{approval_hint}"
        )


#: The exits a refusal of an unlogged signature names.
UNLOGGED_REFUSAL_HINT = (
    "accepted only by the deploy that signed it with "
    f"{ALLOW_UNLOGGED_FLAG} {REASON_FLAG} ..., or as the release the site "
    "runs; run `gpu-fault-admin backfill-transparency-log --state-dir STATE_DIR` "
    "once the transparency log answers"
)


def consented_reason(environment: Mapping[str, str]) -> str:
    """The reason the deploy hop handed down, required by an explicit flag."""

    return validate_reason(environment.get(UNLOGGED_SIGNATURE_REASON_ENV, ""))


# -- transient Sigstore failures of a signing ---------------------------------

_SIGSTORE_SERVICE = re.compile(
    r"rekor|/api/v1/log/entries|transparency log|tuf|trusted.root|signing config|"
    r"/api/v1/timestamp|timestamp authority|sigstore",
    re.IGNORECASE,
)
_TRANSIENT_SYMPTOM = re.compile(
    r"status 5\d\d\b|\]\[5\d\d\]|\b50[0234]\b|\b429\b|too many requests|"
    r"time(?:d)? ?out|deadline exceeded|connection refused|connection reset|"
    r"no such host|network is unreachable|server misbehaving|temporary failure|"
    r"unexpected eof|tls handshake",
    re.IGNORECASE,
)


def transient_service_failure(stderr: str) -> str | None:
    """The last line of a failure on a Sigstore service that may come back.

    A 5xx, a 429, a timeout or a connection error on Rekor, the TSA or TUF:
    the failures the bounded retry is for and that ``--allow-unlogged-
    signature`` avoids (the unlogged signing contacts none of them). A bad
    password, a 4xx or a signature problem is not transient and passes through.
    """

    for line in reversed([item.strip() for item in stderr.splitlines()]):
        if line and _SIGSTORE_SERVICE.search(line) and _TRANSIENT_SYMPTOM.search(line):
            return line
    return None


REFUSAL_MARKER_SUFFIX = ".transparency-log-refused.json"


def refusal_marker(bundle: Path) -> Path:
    """Where the wrapper notes a refusal for its caller (make hides the status)."""

    return bundle.with_name(bundle.name + REFUSAL_MARKER_SUFFIX)


def write_refusal_marker(
    bundle: Path, *, artifact: Path, attempts: int, cause: str
) -> None:
    marker = refusal_marker(bundle)
    _atomic_write(
        marker,
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact": str(artifact),
                    "attempts": attempts,
                    "cause": cause,
                    "refused_at": utc_timestamp(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        mode=0o600,
    )


def read_refusal_marker(bundle: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(refusal_marker(bundle).read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# -- the pinned Sigstore trust root -------------------------------------------

#: cosign v3.1.2's stderr when its live TUF fetch fails -- the only failures
#: that may use the pin. Verification: "getting trusted root from TUF" (wrapped
#: around "error getting live trusted root" / "tuf refresh failed"); signing:
#: "error getting signing config from TUF". Any other failure (a signature,
#: Rekor, the TSA) never touches the pin.
TUF_FETCH_FAILURES = (
    "getting trusted root from TUF",
    "tuf refresh failed",
    "error getting signing config from TUF",
)
TUF_MIRROR = "tuf-repo-cdn.sigstore.dev"
TRUSTED_ROOT_TARGET = "trusted_root.json"
SIGNING_CONFIG_TARGET = "signing_config.v0.2.json"
TRUSTED_ROOT_PIN = "sigstore-trusted-root.json"
SIGNING_CONFIG_PIN = "sigstore-signing-config.json"
PIN_RECORD = "sigstore-trust-pin.json"
PIN_USE_RECORD = "sigstore-trust-pin-use.json"
PIN_FILES = {
    TRUSTED_ROOT_TARGET: TRUSTED_ROOT_PIN,
    SIGNING_CONFIG_TARGET: SIGNING_CONFIG_PIN,
}
#: A pin older than this at use is flagged in the deploy's record; status warns.
PIN_AGE_WARNING_DAYS = 30
SITE_SIGNING_KEY = "cosign.key"


def tuf_fetch_failure(stderr: str) -> str | None:
    """The cosign line naming a failed live TUF fetch, or None."""

    for line in reversed([item.strip() for item in stderr.splitlines()]):
        if any(form in line for form in TUF_FETCH_FAILURES):
            return line
    return None


def pin_directory_for_key(key: str | os.PathLike[str] | None) -> Path | None:
    """The site's ``release-signing`` directory, beside its key pair.

    The pin lives where the key lives: only a directory that holds the site's
    signing key (``cosign.key``) and is not group/other writable. A key URI
    (``awskms://`` ...) or a public key copied elsewhere has no pin.
    """

    if key is None or "://" in str(key):
        return None
    directory = Path(key).expanduser().resolve().parent
    try:
        if not (directory / SITE_SIGNING_KEY).is_file():
            return None
        if directory.stat().st_mode & 0o022:
            return None
    except OSError:
        return None
    return directory


def tuf_cache(environment: Mapping[str, str] | None = None) -> Path:
    """cosign v3's TUF cache (sigstore-go ``tuf.DefaultOptions``)."""

    values = os.environ if environment is None else environment
    home = values.get("HOME") or os.path.expanduser("~")
    return Path(home) / ".sigstore" / "root"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _metadata_version(path: Path) -> int:
    value = json.loads(path.read_bytes())
    version = value["signed"]["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{path.name} has no TUF version")
    return version


@dataclass(frozen=True)
class TrustPin:
    directory: Path
    trusted_root: Path
    signing_config: Path
    record: Mapping[str, Any]

    def age_days(self, now: datetime | None = None) -> float | None:
        pinned = parse_timestamp(self.record.get("pinned_at"))
        if pinned is None:
            return None
        return ((now or datetime.now(UTC)) - pinned).total_seconds() / 86400

    def usage(self, now: datetime | None = None) -> dict[str, Any]:
        """What a record says about a signature or verification that used it."""

        age = self.age_days(now)
        files = self.record.get("files") or {}
        return {
            "trusted_root": "pinned",
            "pin_sha256": {
                name: (files.get(name) or {}).get("sha256")
                for name in (TRUSTED_ROOT_PIN, SIGNING_CONFIG_PIN)
            },
            "tuf_last_timestamp": self.record.get("tuf_last_timestamp"),
            "pinned_at": self.record.get("pinned_at"),
            "pin_age_days": round(age, 2) if age is not None else None,
            "pin_older_than_warning": age is None or age > PIN_AGE_WARNING_DAYS,
            "used_at": utc_timestamp(now),
        }

    def describe(self, now: datetime | None = None) -> str:
        age = self.age_days(now)
        return (
            f"the site-pinned Sigstore trust root (pinned {self.record.get('pinned_at')}, "
            + (f"{age:.1f} days old" if age is not None else "age unknown")
            + f", TUF timestamp {self.record.get('tuf_last_timestamp')})"
        )


def load_trust_pin(directory: Path | None) -> TrustPin | None:
    """The pin, only if every file still has the recorded SHA-256."""

    if directory is None:
        return None
    try:
        record = json.loads((directory / PIN_RECORD).read_bytes())
        files = record["files"]
        for name in (TRUSTED_ROOT_PIN, SIGNING_CONFIG_PIN):
            if _sha256((directory / name).read_bytes()) != files[name]["sha256"]:
                return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if not isinstance(record, dict):
        return None
    return TrustPin(
        directory=directory,
        trusted_root=directory / TRUSTED_ROOT_PIN,
        signing_config=directory / SIGNING_CONFIG_PIN,
        record=record,
    )


def refresh_trust_pin(
    directory: Path | None,
    *,
    environment: Mapping[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Re-pin from cosign's TUF cache right after an online run; never raises.

    The copy is taken only when each target's length and SHA-256 equal the
    cached TUF ``targets.json`` entry, and never when the cached timestamp or
    targets metadata version is lower than the pinned one. The previous pin is
    kept as ``<file>.prev``. A failed re-pin is reported and changes nothing:
    the signing or verification that just succeeded stands.
    """

    report = log or (lambda line: print(line, file=sys.stderr, flush=True))
    if directory is None:
        return None
    try:
        return _refresh_trust_pin(directory, environment=environment, report=report)
    except Exception as exc:  # noqa: BLE001 - a re-pin never fails its caller
        report(f"signing_transparency: not re-pinned: {type(exc).__name__}: {exc}")
        return None


def _refresh_trust_pin(
    directory: Path,
    *,
    environment: Mapping[str, str] | None,
    report: Callable[[str], None],
) -> dict[str, Any] | None:
    try:
        cache = tuf_cache(environment) / TUF_MIRROR
        targets_metadata = cache / "targets.json"
        targets = json.loads(targets_metadata.read_bytes())["signed"]["targets"]
        versions = {
            "timestamp": _metadata_version(cache / "timestamp.json"),
            "targets": _metadata_version(targets_metadata),
        }
        last = json.loads((cache.parent / f"{TUF_MIRROR}.json").read_bytes()).get(
            "lastTimestamp"
        )
        payloads: dict[str, bytes] = {}
        for target, pinned_name in PIN_FILES.items():
            data = (cache / "targets" / target).read_bytes()
            entry = targets[target]
            if entry["length"] != len(data) or entry["hashes"]["sha256"] != _sha256(
                data
            ):
                report(
                    f"signing_transparency: not re-pinned: the cached {target} does "
                    "not match the TUF targets metadata"
                )
                return None
            payloads[pinned_name] = data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report(
            "signing_transparency: not re-pinned: cosign's TUF cache is not "
            f"usable ({type(exc).__name__})"
        )
        return None
    current = load_trust_pin(directory)
    if current is not None:
        pinned = current.record.get("tuf_versions") or {}
        lower = [
            name
            for name, version in versions.items()
            if isinstance(pinned.get(name), int) and version < pinned[name]
        ]
        if lower:
            report(
                "signing_transparency: not re-pinned: the TUF "
                + " and ".join(
                    f"{name} version {versions[name]} is lower than the pinned "
                    f"{pinned[name]}"
                    for name in lower
                )
            )
            return None
        unchanged = (
            all(
                (current.record.get("files") or {}).get(name, {}).get("sha256")
                == _sha256(data)
                for name, data in payloads.items()
            )
            and pinned == versions
        )
        if unchanged:
            record = dict(current.record)
            record["tuf_last_timestamp"] = last
            record["pinned_at"] = utc_timestamp()
            try:
                _atomic_write(
                    directory / PIN_RECORD,
                    (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
                    mode=0o600,
                )
            except OSError as exc:
                report(f"signing_transparency: pin not refreshed: {exc}")
                return None
            return record
    record = {
        "schema_version": 1,
        "source": f"cosign TUF cache ({TUF_MIRROR})",
        "tuf_last_timestamp": last,
        "tuf_versions": versions,
        "pinned_at": utc_timestamp(),
        "files": {
            name: {"sha256": _sha256(data), "size": len(data)}
            for name, data in payloads.items()
        },
    }
    try:
        for name in (*payloads, PIN_RECORD):
            path = directory / name
            if path.is_file():
                os.replace(path, path.with_name(path.name + ".prev"))
        for name, data in payloads.items():
            _atomic_write(directory / name, data, mode=0o600)
        _atomic_write(
            directory / PIN_RECORD,
            (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
            mode=0o600,
        )
    except OSError as exc:
        report(f"signing_transparency: not re-pinned: {exc}")
        return None
    return record


@dataclass(frozen=True)
class CosignOutcome:
    returncode: int
    stdout: str
    stderr: str
    pinned: TrustPin | None = None
    tuf_failure: str | None = None


def run_verify_blob(
    command: Sequence[str],
    *,
    pin_directory: Path | None,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> CosignOutcome:
    """``cosign verify-blob`` as given; on a failed live TUF fetch only, once
    more against the site's pinned trust root.

    The first run is exactly the caller's command. A success re-pins (it read
    TUF). Any failure other than one of ``TUF_FETCH_FAILURES`` is returned as
    is, and so is a TUF failure when there is no usable pin.
    """

    report = log or (lambda line: print(line, file=sys.stderr, flush=True))
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        if IGNORE_TLOG_FLAG not in command:
            refresh_trust_pin(pin_directory, environment=environment, log=report)
        return CosignOutcome(0, completed.stdout, completed.stderr)
    cause = tuf_fetch_failure(completed.stderr)
    pin = load_trust_pin(pin_directory) if cause is not None else None
    if cause is None or pin is None or "--trusted-root" in command:
        return CosignOutcome(completed.returncode, completed.stdout, completed.stderr)
    retried = subprocess.run(
        [*command[:-1], "--trusted-root", str(pin.trusted_root), command[-1]],
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    if retried.returncode == 0:
        report(
            f"signing_transparency: Sigstore TUF did not answer ({cause}); "
            f"verified against {pin.describe()}"
        )
    return CosignOutcome(
        retried.returncode,
        retried.stdout,
        completed.stderr + retried.stderr,
        pinned=pin,
        tuf_failure=cause,
    )


def record_pin_use(directory: Path, usage: Mapping[str, Any]) -> None:
    """Remember the last use of the pin for ``gpu-fault-admin status``; never raises."""

    try:
        _atomic_write(
            directory / PIN_USE_RECORD,
            (json.dumps(dict(usage), indent=2, sort_keys=True) + "\n").encode(),
            mode=0o600,
        )
    except Exception:  # noqa: BLE001 - bookkeeping never fails its caller
        pass
