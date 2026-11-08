"""A deploy without ``--allow-unlogged-signature`` backfills the site key's
unlogged signatures itself before it deploys.

The previous deploy carried the switch, so its success record, the release
record's attestation copy, the deploy-host archive and the ``dist/`` attestation
have no transparency-log entry. The next normal deploy asks the transparency
log the signing retries' way (``cosign_sign_blob.transparency_log_unavailable``)
and, when it answers, runs the backfill verb's code path under the lock it holds,
prints the verb's line per bundle and one summary line, records the entries in
``source-deploy-success.json`` and the release record, and goes on: the previous
success record then verifies with the log. A log that does not answer, or a
bundle that fails, refuses (exit 2) naming the switch and the manual command;
what was backfilled keeps its entry, a failed bundle is untouched, and a rerun
picks up where it stopped. ``deploy --rollback`` neither takes the switch nor
backfills. A fake transparency log on localhost and a fake ``cosign`` stand in
for Sigstore; nothing here reaches a network.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cli, deploy_command
from gpu_fault.admin.site import SiteConfigError
from gpu_fault_release import regional_release_signing as admin_signing
from scripts import (
    backfill_transparency_log,
    cosign_sign_blob,
    release_deploy,
    staging_deploy,
    staging_signing,
)
from scripts import signing_transparency as transparency
from tests._staging_deploy_support import (
    DEPLOY_EVENT,
    deploy_arguments,
    install_admin_stub,
    managed_state,
    signing_material,
    stub_deploy_orchestration,
)
from tests._unlogged_signing_support import (
    LOGGED_BUNDLE,
    PUBLIC_PEM,
    REASON,
    UNLOGGED_BUNDLE,
    fake_cosign,
    fake_rekor,
)
from tests._unlogged_signing_support import write_json as _write

ARCHIVE = "gpu-fault-deploy-host-p"
SWITCH = '--allow-unlogged-signature --reason "<why>"'


def _closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``cosign`` first on ``PATH``, no TUF cache in ``HOME``, no retry
    waits, and a transparency log that is a closed local port unless a test
    starts the fake one."""

    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cosign_sign_blob, "SIGN_RETRY_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(cosign_sign_blob, "DEFAULT_REKOR_URL", _closed_port_url())
    return fake_cosign(directory)


@pytest.fixture
def rekor(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    with fake_rekor() as state:
        monkeypatch.setattr(cosign_sign_blob, "DEFAULT_REKOR_URL", state["url"])
        yield state


def _unlogged_site(
    tmp_path: Path,
) -> tuple[Path, Path, staging_deploy.SigningMaterial, list[Path]]:
    """A managed site whose last deploy carried the switch: four signatures of
    the site key without a log entry, in the order the backfill reads them
    (``dist/``, the release record's copy, the success record, the archive)."""

    repository, state = managed_state(tmp_path)
    (state / "site.yaml").write_text(
        yaml.safe_dump(
            {"kind": "RegionalSite", "spec": {"repositoryRoot": str(repository)}}
        ),
        encoding="utf-8",
    )
    signing = signing_material(state)
    signing.public_key.write_text(PUBLIC_PEM, encoding="utf-8")
    install_admin_stub(state)
    declaration = transparency.unlogged_declaration(REASON)
    record = state / "release-deploy" / "release-a"
    _write(
        record / "state.json",
        {
            "phase": "COMPLETED",
            "release_id": "release-a",
            "signing": {
                "transparency_log": "unlogged",
                "reason": REASON,
                "declared_at": declaration["declared_at"],
                "bundle": str(record / "attestation.sigstore.json"),
            },
        },
    )
    _write(record / "attestation.json", {"release_id": "release-a", "copy": True})
    _write(repository / "dist/current-attestation.json", {"release_id": "release-a"})
    _write(
        state / "source-deploy-success.json",
        {
            "schema_version": 1,
            "status": "PASSED",
            "mode": "DEPLOY_HOST_ONLY",
            "identities": {
                "application": {"sha256": "0" * 64},
                "deploy_host": {"sha256": "1" * 64},
            },
            "source": {"fingerprint": "e" * 64},
            "live": {},
            "signing": declaration,
        },
    )
    archive = state / "deploy-host/by-content" / ("c" * 64) / f"{ARCHIVE}.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"deploy-host archive")
    bundles = [
        repository / "dist/current-attestation.bundle.json",
        record / "attestation.sigstore.json",
        state / "source-deploy-success.sigstore.json",
        archive.with_name(f"{ARCHIVE}.sigstore.json"),
    ]
    for bundle in bundles:
        _write(bundle, UNLOGGED_BUNDLE)
    return repository, state, signing, bundles


def _digests(state: Path, repository: Path) -> dict[Path, str]:
    """The SHA-256 the backfill uploads for each bundle: its artifact's."""

    return {
        bundle: hashlib.sha256(artifact.read_bytes()).hexdigest()
        for artifact, bundle in backfill_transparency_log.candidates(state, repository)
    }


def _stub_deploy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository: Path,
    state: Path,
    signing: staging_deploy.SigningMaterial,
    events: list[str],
) -> dict[str, Any]:
    """Everything ``deploy`` shells out to, except the backfill and the real
    verification of the previous success record (whose cosign is recorded)."""

    real_backfill = staging_deploy.backfill_before_deploy
    real_load = staging_deploy.load_successful_source_deploy
    stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=staging_deploy.SourceCheckout(
            repository_root=repository,
            git_commit="a" * 40,
            fingerprint="f" * 64,
            snapshot=False,
        ),
        signing=signing,
        previous=None,
        events=events,
    )
    seen: dict[str, Any] = {"cosign": []}

    def backfill(*arguments: Any, **options: Any) -> list[dict[str, str]]:
        events.append("backfill")
        return real_backfill(*arguments, **options)

    def load(*arguments: Any, **options: Any) -> dict[str, object] | None:
        events.append("verify-previous-success")
        return real_load(*arguments, **options)

    def record(*_arguments: Any, **options: Any) -> Path:
        events.append("success")
        seen["success"] = options
        return state / "source-deploy-success.json"

    monkeypatch.setattr(staging_deploy, "backfill_before_deploy", backfill)
    monkeypatch.setattr(staging_deploy, "load_successful_source_deploy", load)
    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda command, **_options: seen["cosign"].append(list(command)) or "",
    )
    monkeypatch.setattr(staging_deploy, "record_successful_source_deploy", record)
    return seen


def _main(repository: Path, state: Path) -> int:
    arguments = deploy_arguments(repository, state)
    return staging_deploy.main(
        [
            "--state-dir",
            str(state),
            "--repo-root",
            str(repository),
            "--cpu-cluster-arn",
            arguments.cpu_cluster_arn,
            "--gpu-cluster-arn",
            arguments.gpu_cluster_arn[0],
            "--admin-email",
            arguments.admin_email,
            "--quiet",
        ]
    )


# -- the deploy ------------------------------------------------------------------


def test_a_normal_deploy_backfills_first_then_deploys_and_records_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rekor: dict[str, Any],
) -> None:
    repository, state, signing, bundles = _unlogged_site(tmp_path)
    events: list[str] = []
    seen = _stub_deploy(
        monkeypatch, repository=repository, state=state, signing=signing, events=events
    )

    result = staging_deploy.deploy(deploy_arguments(repository, state))

    assert events.index("lock-enter") < events.index("backfill"), events
    assert events.index("backfill") < events.index("verify-previous-success"), events
    assert DEPLOY_EVENT in events and "success" in events, "the deploy did not go on"
    assert rekor["log_info"] == 1, "the log was not asked once before the uploads"
    digests = _digests(state, repository)
    assert [post["spec"]["data"]["hash"]["value"] for post in rekor["posts"]] == [
        digests[bundle] for bundle in bundles
    ]
    for bundle in bundles:
        assert transparency.bundle_transparency(bundle) == transparency.LOGGED, bundle
    success_check = seen["cosign"][0]
    assert success_check[-1] == str(state / "source-deploy-success.json")
    assert transparency.IGNORE_TLOG_FLAG not in success_check, (
        "the previous success record was not verified with the log"
    )
    entries = [
        {"bundle": str(bundle), "log_entry": uuid, "at": entry["at"]}
        for bundle, uuid, entry in zip(
            bundles, rekor["entries"], seen["success"]["backfill"], strict=True
        )
    ]
    assert seen["success"]["backfill"] == entries
    assert all(transparency.parse_timestamp(entry["at"]) for entry in entries), entries
    release = json.loads(
        (state / "release-deploy/release-a/state.json").read_text(encoding="utf-8")
    )
    assert release["signing"]["backfilled"] == entries
    assert release["signing"]["transparency_log"] == "unlogged", "the block was lost"
    assert seen["success"]["release_signing"] == release["signing"]
    assert result["signing"] == {"transparency_log": "logged", "backfilled": entries}
    err = capsys.readouterr().err
    for bundle, uuid in zip(bundles, rekor["entries"], strict=True):
        assert (
            f"backfill-transparency-log: {bundle} now carries log entry {uuid}" in err
        )
    assert (
        "deploy: backfilled 4 unlogged signature(s) into the transparency log "
        "before deploying"
    ) in err


def test_a_log_that_does_not_answer_refuses_as_before_and_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rekor: dict[str, Any],
) -> None:
    repository, state, signing, bundles = _unlogged_site(tmp_path)
    before = {bundle: bundle.read_bytes() for bundle in bundles}
    record = (state / "source-deploy-success.json").read_bytes()
    events: list[str] = []
    _stub_deploy(
        monkeypatch, repository=repository, state=state, signing=signing, events=events
    )
    rekor["down"] = True

    assert _main(repository, state) == 2

    assert rekor["log_info"] == cosign_sign_blob.SIGN_ATTEMPTS == 3
    assert rekor["posts"] == [], "something was uploaded to a log that did not answer"
    assert {bundle: bundle.read_bytes() for bundle in bundles} == before
    assert (state / "source-deploy-success.json").read_bytes() == record
    assert "verify-previous-success" not in events and DEPLOY_EVENT not in events
    assert events[-1] == "lock-exit", events
    err = capsys.readouterr().err
    assert err.count("staging-deploy: transient Sigstore failure (") == 2
    assert (
        "staging-deploy: 4 signature(s) of the site key have no transparency-log "
        "entry (" in err
    )
    assert (
        f"and the transparency log does not answer (transparency log {rekor['url']}"
        "/api/v1/log answered status 500); nothing was backfilled and nothing was "
        "deployed; rerun this deploy once the transparency log answers (it "
        "backfills them first) or run `gpu-fault-admin backfill-transparency-log "
        f"--state-dir {state}`; to deploy before that, rerun it with {SWITCH}"
    ) in err


@pytest.mark.parametrize("refused", [2, 3], ids=["success-record", "archive"])
def test_a_failed_bundle_refuses_and_keeps_what_was_logged_then_a_rerun_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rekor: dict[str, Any],
    refused: int,
) -> None:
    """A failed success record and any other failed bundle refuse alike."""

    repository, state, signing, bundles = _unlogged_site(tmp_path)
    failed = bundles[refused]
    untouched = failed.read_bytes()
    events: list[str] = []
    _stub_deploy(
        monkeypatch, repository=repository, state=state, signing=signing, events=events
    )
    rekor["refuse"] = {_digests(state, repository)[failed]}

    assert _main(repository, state) == 2

    assert failed.read_bytes() == untouched, "the failed bundle was rewritten"
    for bundle in bundles:
        if bundle != failed:
            assert transparency.bundle_transparency(bundle) == transparency.LOGGED
    assert "verify-previous-success" not in events and DEPLOY_EVENT not in events
    err = capsys.readouterr().err
    assert (
        f"staging-deploy: {failed} was not backfilled: the transparency log did not "
        "accept the upload (HTTP 400)"
    ) in err
    assert (
        "1 of 4 signature(s) of the site key without a transparency-log entry could "
        f"not be backfilled ({failed.relative_to(state)}); nothing was deployed, and "
        "the 3 backfilled keep their entry; rerun this deploy once the transparency "
        "log answers (it backfills them first) or run `gpu-fault-admin "
        f"backfill-transparency-log --state-dir {state}`; to deploy before that, "
        f"rerun it with {SWITCH}"
    ) in err
    assert "deploy: backfilled" not in err, "a refused backfill printed its summary"

    # The rerun once the log takes it: only the failed bundle is uploaded.
    rekor["refuse"] = set()
    posts = len(rekor["posts"])
    (entry,) = staging_signing.backfill_before_deploy(state, signing, None)
    assert entry["bundle"] == str(failed)
    assert len(rekor["posts"]) == posts + 1
    assert transparency.bundle_transparency(failed) == transparency.LOGGED
    assert (
        "deploy: backfilled 1 unlogged signature(s) into the transparency log "
        "before deploying"
    ) in capsys.readouterr().err


def test_a_rerun_is_idempotent_and_reuses_an_entry_the_log_already_holds(
    tmp_path: Path, rekor: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    repository, state, signing, bundles = _unlogged_site(tmp_path)
    first = staging_signing.backfill_before_deploy(state, signing, None)
    assert [entry["bundle"] for entry in first] == [str(bundle) for bundle in bundles]
    asked, uploaded = rekor["log_info"], len(rekor["posts"])

    assert staging_signing.backfill_before_deploy(state, signing, None) == []
    assert (rekor["log_info"], len(rekor["posts"])) == (asked, uploaded), (
        "a site without unlogged signatures asked the log"
    )

    # A rewrite lost after the upload: the log answers 409 with the same entry.
    _write(bundles[2], UNLOGGED_BUNDLE)
    rekor["conflict"] = True
    (again,) = staging_signing.backfill_before_deploy(state, signing, None)
    assert again["bundle"] == str(bundles[2])
    assert again["log_entry"] == first[2]["log_entry"]
    assert transparency.bundle_transparency(bundles[2]) == transparency.LOGGED

    # The verb out of band finds nothing left to do.
    capsys.readouterr()
    assert (
        backfill_transparency_log.main(
            [
                "--state-dir",
                str(state),
                "--repository-root",
                str(repository),
                "--public-key",
                str(signing.public_key),
                "--rekor-url",
                rekor["url"],
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["already_logged"] == [str(bundle) for bundle in bundles]
    assert summary["backfilled"] == summary["failed"] == []


def test_a_deploy_with_the_switch_or_nothing_unlogged_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, state, signing, bundles = _unlogged_site(tmp_path)

    def never(*_arguments: object, **_options: object) -> str:
        raise AssertionError("the transparency log was asked")

    monkeypatch.setattr(cosign_sign_blob, "transparency_log_unavailable", never)
    assert staging_signing.backfill_before_deploy(state, signing, REASON) == []
    assert transparency.bundle_transparency(bundles[0]) == transparency.UNLOGGED
    for bundle in bundles:
        _write(bundle, LOGGED_BUNDLE)
    assert staging_signing.backfill_before_deploy(state, signing, None) == []


# -- the rollback ------------------------------------------------------------------


def test_a_rollback_neither_takes_the_switch_nor_backfills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, state, _signing, bundles = _unlogged_site(tmp_path)
    before = {bundle: bundle.read_bytes() for bundle in bundles}
    rolled: list[Path] = []

    def never(*_arguments: object, **_options: object) -> Any:
        raise AssertionError("a rollback ran the source deploy or asked the log")

    monkeypatch.setattr(cosign_sign_blob, "transparency_log_unavailable", never)
    hooks = deploy_command.DeployHooks(
        run_source_deploy=never,
        bootstrap_from_arns=never,
        run_automatic_release=never,
        join_clusters=never,
        run_rollback=lambda _site, **options: rolled.append(options["state_dir"]) or 0,
        approve_profile_plan_inline=never,
        discover_cluster=never,
        load_site=lambda *_a, **_k: object(),
        release_consent_environment=lambda _arguments: {},
        grafana_environment=lambda _arguments: {},
        grafana_request_fields=lambda _arguments: {},
    )

    def deploy(*flags: str) -> int:
        arguments = cli.parser().parse_args(
            ["deploy", "--state-dir", str(state), "--rollback", *flags]
        )
        return deploy_command.run_deploy(arguments, hooks=hooks)

    with pytest.raises(SiteConfigError) as refused:
        deploy("--allow-unlogged-signature", "--reason", REASON)
    assert str(refused.value) == (
        "deploy --rollback takes no other deploy option; remove "
        "--allow-unlogged-signature, --reason"
    )
    assert rolled == [], "the refused rollback ran"
    assert deploy() == 0
    assert rolled == [state]
    assert {bundle: bundle.read_bytes() for bundle in bundles} == before


# -- the probe -------------------------------------------------------------------


def test_the_probe_asks_once_when_the_log_answers(rekor: dict[str, Any]) -> None:
    assert cosign_sign_blob.transparency_log_unavailable(rekor["url"]) is None
    assert rekor["gets"] == [cosign_sign_blob.LOG_INFO_PATH]
    assert rekor["posts"] == [], "the probe uploaded something"


def test_the_probe_retries_a_transient_failure_as_a_signing_would(
    rekor: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    rekor["down"] = True
    cause = cosign_sign_blob.transparency_log_unavailable(rekor["url"], label="probe")
    assert cause == f"transparency log {rekor['url']}/api/v1/log answered status 500"
    assert transparency.transient_service_failure(cause) == cause
    assert rekor["log_info"] == cosign_sign_blob.SIGN_ATTEMPTS
    err = capsys.readouterr().err
    assert f"probe: transient Sigstore failure ({cause}); attempt 2 of 3 in 0 s" in err
    assert f"probe: transient Sigstore failure ({cause}); attempt 3 of 3 in 0 s" in err

    rekor["down"], rekor["fail_next"] = False, 1
    assert cosign_sign_blob.transparency_log_unavailable(rekor["url"]) is None
    assert rekor["log_info"] == cosign_sign_blob.SIGN_ATTEMPTS + 2


def test_a_refused_connection_is_transient_and_any_other_answer_final(
    rekor: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    closed = _closed_port_url()
    cause = cosign_sign_blob.transparency_log_unavailable(closed)
    assert cause is not None and cause.startswith(
        f"transparency log {closed}/api/v1/log did not answer ("
    ), cause
    assert "refused" in cause.lower(), cause
    assert capsys.readouterr().err.count("transient Sigstore failure") == 2

    elsewhere = cosign_sign_blob.transparency_log_unavailable(f"{rekor['url']}/v0")
    assert elsewhere == (
        f"transparency log {rekor['url']}/v0/api/v1/log answered status 404"
    )
    rekor["log_info_body"] = {"status": "ok"}
    no_log = cosign_sign_blob.transparency_log_unavailable(rekor["url"])
    assert no_log == (
        f"transparency log {rekor['url']}/api/v1/log answered without its tree size"
    )
    assert rekor["gets"] == ["/v0/api/v1/log", "/api/v1/log"], "a final one was retried"
    assert capsys.readouterr().err == ""


# -- the records and the lines ---------------------------------------------------


def test_the_records_name_the_backfill_and_the_declaration_stays_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [
        {
            "bundle": str(tmp_path / "state/source-deploy-success.sigstore.json"),
            "log_entry": "108e9186e8c5677a" + "a" * 64,
            "at": transparency.utc_timestamp(),
        }
    ]

    def run(arguments: list[str], **_options: Any) -> str:
        Path(arguments[arguments.index("--bundle") + 1]).write_text("{}", "utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    path = staging_deploy.record_successful_source_deploy(
        state,
        source_repository_root=tmp_path,
        source=staging_deploy.SourceCheckout(
            repository_root=tmp_path,
            git_commit="a" * 40,
            fingerprint="f" * 64,
            snapshot=False,
        ),
        identities={
            "application": {"sha256": "1" * 64},
            "deploy_host": {"sha256": "2" * 64, "bundle": {"sha256": "3" * 64}},
        },
        signing=signing_material(tmp_path / "keys"),
        mode="UNCHANGED",
        live_evidence={},
        backfill=entries,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record[staging_signing.BACKFILL_RECORD_FIELD] == entries
    assert record["signing"] == {"transparency_log": "logged"}
    assert transparency.payload_declaration(record) is None, "declaration widened"

    release = _write(
        state / "release-deploy/release-a/state.json",
        {"phase": "COMPLETED", "signing": {"transparency_log": "logged"}},
    )
    original = release.read_bytes()
    assert staging_signing.release_record_signing(state, "UNCHANGED", entries) is None
    assert staging_signing.release_record_signing(state) == {
        "transparency_log": "logged"
    }
    assert release.read_bytes() == original, "a release record was rewritten"
    signing = staging_signing.release_record_signing(
        state, "APPLICATION_RELEASE", entries
    )
    assert signing == {"transparency_log": "logged", "backfilled": entries}
    assert json.loads(release.read_text(encoding="utf-8")) == {
        "phase": "COMPLETED",
        "signing": signing,
    }
    assert staging_signing.signing_result(None, entries) == {
        "transparency_log": "logged",
        "backfilled": entries,
    }
    assert staging_signing.signing_result(REASON) == {
        "transparency_log": "unlogged",
        "reason": REASON,
    }


def test_the_verb_keeps_its_summary_through_the_shared_code_path(
    tmp_path: Path, rekor: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    repository, state, signing, bundles = _unlogged_site(tmp_path)
    _write(bundles[1], LOGGED_BUNDLE)
    bundles[3].with_name(f"{ARCHIVE}.tar.gz").unlink()

    code = backfill_transparency_log.main(
        [
            "--state-dir",
            str(state),
            "--repository-root",
            str(repository),
            "--public-key",
            str(signing.public_key),
            "--rekor-url",
            rekor["url"],
        ]
    )

    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "already_logged": [str(bundles[1])],
        "backfilled": [str(bundles[0]), str(bundles[2])],
        "failed": [
            {"bundle": str(bundles[3]), "error": f"{ARCHIVE}.tar.gz is missing"}
        ],
        "skipped": [],
    }


def test_every_unlogged_line_names_the_next_deploy_and_the_manual_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    signing = {
        "transparency_log": "unlogged",
        "reason": REASON,
        "declared_at": transparency.utc_timestamp(),
        "bundle": str(tmp_path / "release-deploy/release-a/attestation.sigstore.json"),
    }
    _write(tmp_path / "release-deploy/release-a/state.json", {"signing": signing})
    _write(Path(signing["bundle"]), UNLOGGED_BUNDLE)
    _write(tmp_path / "source-deploy-success.sigstore.json", UNLOGGED_BUNDLE)

    _field, (release_line,), _check = admin_signing.release_signing_view(
        tmp_path, release_id="release-a"
    )
    _names, (other_line,) = admin_signing.unlogged_signatures_view(
        tmp_path, release_id="release-a"
    )
    (summary_line,) = release_deploy.signing_summary_lines(
        signing, release_id="release-a", state_dir=tmp_path
    )
    staging_signing.report_unlogged_deploy(tmp_path, REASON)
    deploy_line = capsys.readouterr().err.strip()

    for line, what in (
        (release_line, "it"),
        (summary_line, "it"),
        (other_line, "them"),
        (deploy_line, "its signatures"),
    ):
        assert line.endswith(transparency.backfill_exit(tmp_path, what=what)), line
        assert "the next deploy without --allow-unlogged-signature backfills" in line, (
            line
        )
        assert f"gpu-fault-admin backfill-transparency-log --state-dir {tmp_path}" in (
            line
        ), line
