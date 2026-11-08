"""The source deploy's half of ``--allow-unlogged-signature`` and the trust pin.

``scripts/staging_deploy.py`` signs the deploy-host archive, the CI candidate
receipt and the successful source deploy record, and verifies the previous
deploy's record and the deploy-host archive. These helpers are what it adds for
a deploy that signs without a transparency-log entry
(``signing_transparency``): the consent is the operator's two arguments on this
command, never an inherited variable; a cached artefact signed without a log
entry is reused only by a deploy that allows it; a refused signing says that
nothing on the site changed; every ``cosign verify-blob`` falls back to the
site-pinned Sigstore trust root only when the live TUF fetch fails. A deploy
without the switch first backfills the site key's signatures that have no log
entry (``backfill_before_deploy``) and records what it backfilled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic

if __package__:
    from scripts import backfill_transparency_log as backfill
    from scripts import cosign_sign_blob
    from scripts import signing_transparency as transparency
    from scripts.staging_live_evidence import LiveEvidenceError, latest_release_record
    from scripts.staging_state_hygiene import (
        DeployHostArtifacts,
        SigningMaterial,
        StagingDeployError,
    )
else:
    import backfill_transparency_log as backfill
    import cosign_sign_blob
    import signing_transparency as transparency
    from staging_live_evidence import LiveEvidenceError, latest_release_record
    from staging_state_hygiene import (
        DeployHostArtifacts,
        SigningMaterial,
        StagingDeployError,
    )

#: Where ``source-deploy-success.json`` lists the signatures its deploy
#: backfilled: beside ``signing``, the signed declaration, which takes no other
#: key. The release record lists them as ``signing.backfilled``.
BACKFILL_RECORD_FIELD = "backfilled_signatures"


def consent_environment(
    base: Mapping[str, str], unlogged_reason: str | None
) -> dict[str, str]:
    """``base`` carrying this deploy's unlogged-signature reason, or none.

    An inherited value is always dropped: only the operator's
    ``--allow-unlogged-signature --reason`` on this command puts one back, and
    only for the children that sign or verify for this deploy.
    """

    environment = {
        name: value
        for name, value in base.items()
        if name != transparency.UNLOGGED_SIGNATURE_REASON_ENV
    }
    if unlogged_reason is not None:
        environment[transparency.UNLOGGED_SIGNATURE_REASON_ENV] = unlogged_reason
    return environment


def resolve_unlogged_consent(arguments: argparse.Namespace) -> str | None:
    """This command's ``--allow-unlogged-signature --reason``: both or neither."""

    allowed = bool(getattr(arguments, "allow_unlogged_signature", False))
    reason = getattr(arguments, "reason", None)
    if not allowed:
        if reason is not None:
            raise StagingDeployError(
                f"{transparency.REASON_FLAG} is only used with "
                f"{transparency.ALLOW_UNLOGGED_FLAG}"
            )
        return None
    try:
        return transparency.validate_reason(reason)
    except transparency.SigningTransparencyError as exc:
        raise StagingDeployError(str(exc)) from None


def verify_with_trust_pin(
    command: Sequence[str],
    *,
    cwd: Path,
    public_key: Path,
    run: Callable[..., str],
) -> None:
    """``command`` (a ``cosign verify-blob``) as always; when only cosign's
    live TUF fetch failed, once more against the site's pinned trust root."""

    pin_directory = transparency.pin_directory_for_key(public_key)
    try:
        run(command, cwd=cwd, capture=True)
    except StagingDeployError as exc:
        cause = transparency.tuf_fetch_failure(str(exc))
        pin = transparency.load_trust_pin(pin_directory) if cause else None
        if pin is None:
            raise
        run(
            [*command[:-1], "--trusted-root", str(pin.trusted_root), command[-1]],
            cwd=cwd,
            capture=True,
        )
        transparency.record_pin_use(pin.directory, pin.usage())
        print(
            f"staging-deploy: Sigstore TUF did not answer ({cause}); verified "
            f"against {pin.describe()}",
            file=sys.stderr,
        )
        return
    if transparency.IGNORE_TLOG_FLAG not in command:
        transparency.refresh_trust_pin(pin_directory)


def success_record_command(
    state_path: Path,
    signature_path: Path,
    *,
    public_key: Path,
    allow_unlogged: bool,
) -> list[str]:
    """The verification of the previous deploy's success record.

    Without a log entry it needs the record's own signed declaration and this
    deploy's switch; the refusal names the backfill.
    """

    command = [
        "cosign",
        "verify-blob",
        "--bundle",
        str(signature_path),
        "--key",
        str(public_key),
        str(state_path),
    ]
    if transparency.bundle_transparency(signature_path) != transparency.UNLOGGED:
        return command
    try:
        transparency.require_unlogged_acceptance(
            artifact=state_path,
            declaration=transparency.artifact_declaration(state_path),
            keyed=True,
            approved=allow_unlogged,
            approval_hint=transparency.UNLOGGED_REFUSAL_HINT,
        )
    except transparency.SigningTransparencyError as exc:
        raise StagingDeployError(
            f"successful source deploy authorization: {exc}"
        ) from None
    return [*command[:-1], transparency.IGNORE_TLOG_FLAG, command[-1]]


def success_record_signing(
    unlogged_reason: str | None,
    release_signing: Mapping[str, object] | None,
    trust_pin: Mapping[str, object] | None,
    backfill: Sequence[Mapping[str, str]] = (),
) -> dict[str, object]:
    """The success record's signing fields.

    ``signing`` is how this deploy signed -- its unlogged declaration, under
    which the record's own signature is then made, or ``logged``;
    ``release_signing`` is the release attestation's verification from the
    release record; ``trust_pin`` the pinned trust root's use, if any;
    ``backfilled_signatures`` what this deploy backfilled before it started
    (``backfill_before_deploy``), if anything.
    """

    return {
        transparency.DECLARATION_FIELD: (
            transparency.unlogged_declaration(unlogged_reason)
            if unlogged_reason is not None
            else transparency.logged_declaration()
        ),
        **(
            {"release_signing": dict(release_signing)}
            if release_signing is not None
            else {}
        ),
        **({"trust_pin": dict(trust_pin)} if trust_pin is not None else {}),
        **(
            {BACKFILL_RECORD_FIELD: [dict(item) for item in backfill]}
            if backfill
            else {}
        ),
    }


def cached_unlogged_bundle(
    artifacts: DeployHostArtifacts, unlogged_reason: str | None
) -> bool:
    """Whether the cached deploy-host bundle must be signed again: it has no
    log entry and this deploy does not allow one."""

    if unlogged_reason is not None:
        return False
    if (
        transparency.bundle_transparency(artifacts.signature_bundle)
        != transparency.UNLOGGED
    ):
        return False
    print(
        "staging-deploy: the cached deploy-host bundle is signed without a "
        "transparency-log entry; rebuilding it with the log",
        file=sys.stderr,
    )
    return True


def deploy_host_signing_refusal(
    artifacts: DeployHostArtifacts,
) -> StagingDeployError | None:
    """The refusal of a deploy-host bundle signing the Sigstore services refused."""

    refused = transparency.read_refusal_marker(artifacts.signature_bundle)
    if refused is None:
        return None
    return StagingDeployError(
        "the deploy-host bundle could not be signed: the Sigstore services "
        f"refused it after {refused.get('attempts')} attempts "
        f"({refused.get('cause')}); nothing on the site was changed. Retry "
        "once they answer, or rerun the deploy with "
        f'{transparency.ALLOW_UNLOGGED_FLAG} {transparency.REASON_FLAG} "<why>" '
        "to sign with the site key only"
    )


def release_record_signing(
    state_dir: Path,
    mode: str = "APPLICATION_RELEASE",
    backfill: Sequence[Mapping[str, str]] = (),
) -> dict[str, object] | None:
    """How the release just deployed was signed and verified (its record's
    ``signing`` block); ``None`` without one, or when the deploy ran none.

    What the deploy backfilled before it started is added to that block first,
    as ``signing.backfilled``: the release record is the deploy's own record
    of the release it ran. A record that cannot be rewritten keeps its block
    and is reported; the success record names the backfill either way.
    """

    if mode != "APPLICATION_RELEASE":
        return None
    try:
        path = latest_release_record(state_dir)
        record = json.loads(path.read_text(encoding="utf-8"))
    except (LiveEvidenceError, OSError, ValueError):
        return None
    value = record.get("signing") if isinstance(record, dict) else None
    if not isinstance(value, dict):
        return None
    if not backfill:
        return dict(value)
    signing = {**value, "backfilled": [dict(item) for item in backfill]}
    try:
        write_json_atomic(path, {**record, "signing": signing})
    except OSError as exc:
        print(
            f"staging-deploy: {path} does not name the backfill: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return dict(value)
    return signing


def trust_pin_used_since(
    public_key: Path, started_at: str | None
) -> dict[str, object] | None:
    """The pinned trust root's last use, when this deploy made it."""

    directory = transparency.pin_directory_for_key(public_key)
    if directory is None or started_at is None:
        return None
    try:
        value = json.loads((directory / transparency.PIN_USE_RECORD).read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or str(value.get("used_at") or "") < started_at:
        return None
    return value


def report_unlogged_deploy(state_dir: Path, unlogged_reason: str | None) -> None:
    """The deploy summary's line for a deploy that signed without the log."""

    if unlogged_reason is None:
        return
    print(
        "staging-deploy: this deploy signed WITHOUT a transparency-log entry "
        f"(reason: {unlogged_reason}); "
        + transparency.backfill_exit(state_dir, what="its signatures"),
        file=sys.stderr,
        flush=True,
    )


def signing_result(
    unlogged_reason: str | None, backfill: Sequence[Mapping[str, str]] = ()
) -> dict[str, object]:
    """The ``signing`` field of the source deploy's result."""

    if unlogged_reason is None:
        return {
            "transparency_log": transparency.LOGGED,
            **({"backfilled": [dict(item) for item in backfill]} if backfill else {}),
        }
    return {"transparency_log": transparency.UNLOGGED, "reason": unlogged_reason}


def recorded_repository_root(state_dir: Path) -> Path | None:
    """The site's ``spec.repositoryRoot`` as ``load_site`` resolves it: the
    tree whose ``dist/`` the backfill verb reads; ``None`` without a site."""

    try:
        document = yaml.safe_load((state_dir / "site.yaml").read_text("utf-8"))
        configured = document["spec"]["repositoryRoot"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None
    if not isinstance(configured, str) or not configured.strip():
        return None
    root = Path(configured).expanduser()
    return (root if root.is_absolute() else state_dir / root).resolve()


def _named(bundles: Sequence[Path], state_dir: Path) -> str:
    return ", ".join(
        str(path.relative_to(state_dir))
        if path.is_relative_to(state_dir)
        else str(path)
        for path in bundles
    )


def backfill_before_deploy(
    state_dir: Path,
    signing: SigningMaterial,
    unlogged_reason: str | None,
    *,
    rekor_url: str | None = None,
) -> list[dict[str, str]]:
    """A deploy without the switch first puts the site key's signatures that
    have no transparency-log entry into the log.

    Called under the site lock the deploy holds, before the previous success
    record is verified. The bundles are the backfill verb's
    (``backfill_transparency_log.candidates``) without a log entry; with none,
    nothing is asked. Otherwise the signing retries' probe
    (``cosign_sign_blob.transparency_log_unavailable``) says whether the log
    answers, and the verb's own code path logs each signature in this process
    -- its line per bundle -- then one summary line. The entries ``{bundle,
    log_entry, at}`` go into the deploy's records.

    A log that does not answer, or any bundle that fails, refuses the deploy
    (exit 2) naming the switch and the manual command; each bundle is replaced
    by its own atomic rename, so the ones backfilled keep their entry and a
    failed one is untouched. A deploy with the switch backfills nothing.
    """

    if unlogged_reason is not None:
        return []
    pending = [
        (artifact, bundle)
        for artifact, bundle in backfill.candidates(
            state_dir, recorded_repository_root(state_dir)
        )
        if transparency.bundle_transparency(bundle) == transparency.UNLOGGED
    ]
    if not pending:
        return []
    url = rekor_url or cosign_sign_blob.DEFAULT_REKOR_URL
    exits = (
        "rerun this deploy once the transparency log answers (it backfills them "
        "first) or run `gpu-fault-admin backfill-transparency-log --state-dir "
        f"{state_dir}`; to deploy before that, rerun it with "
        f'{transparency.ALLOW_UNLOGGED_FLAG} {transparency.REASON_FLAG} "<why>"'
    )
    cause = cosign_sign_blob.transparency_log_unavailable(url, label="staging-deploy")
    if cause is not None:
        raise StagingDeployError(
            f"{len(pending)} signature(s) of the site key have no transparency-log "
            f"entry ({_named([bundle for _a, bundle in pending], state_dir)}) and "
            f"the transparency log does not answer ({cause}); nothing was "
            f"backfilled and nothing was deployed; {exits}"
        )
    outcomes = backfill.backfill_bundles(
        pending, public_key=signing.public_key, rekor_url=url
    )
    done = [item for item in outcomes if item.outcome == "backfilled"]
    failed = [item for item in outcomes if item.outcome == "failed"]
    for item in failed:
        print(
            f"staging-deploy: {item.bundle} was not backfilled: {item.error}",
            file=sys.stderr,
            flush=True,
        )
    if failed:
        raise StagingDeployError(
            f"{len(failed)} of {len(pending)} signature(s) of the site key without "
            "a transparency-log entry could not be backfilled "
            f"({_named([item.bundle for item in failed], state_dir)}); nothing was "
            "deployed"
            + (f", and the {len(done)} backfilled keep their entry" if done else "")
            + f"; {exits}"
        )
    if done:
        print(
            f"deploy: backfilled {len(done)} unlogged signature(s) into the "
            "transparency log before deploying",
            file=sys.stderr,
            flush=True,
        )
    return [
        {
            "bundle": str(item.bundle),
            "log_entry": str(item.log_entry),
            "at": str(item.at),
        }
        for item in done
    ]
