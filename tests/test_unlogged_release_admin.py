"""``--allow-unlogged-signature`` in the admin CLI, the records and ``status``.

The deploy's hops hand the operator's consent on as explicit arguments and set
the variable only for their own children; ``deploy --rollback`` takes neither
flag; the release build turns a refused signing into "nothing was rolled out";
the release record keeps the mode and a copy of the signature; ``status`` names
an unlogged release and fails ``release_signature`` 72 hours after the
declaration -- a check the next deploy reads as healthy; the backfill logs the
same signature later; and no in-cluster component verifies a cosign signature,
so the first release that carries the switch can use it. Fake ``cosign``
binaries and a fake transparency log on localhost stand in for Sigstore. The
signers and verifiers are ``tests/test_unlogged_release_signing.py``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import (
    cli,
    deploy_command,
    release_artifacts,
    release_child,
    source_deploy,
    status_report,
)
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import SiteConfigError
from gpu_fault_release import regional_release_signing as admin_signing
from scripts import (
    backfill_transparency_log,
    release_attestation,
    release_deploy,
    staging_deploy,
    staging_live_evidence,
)
from scripts import signing_transparency as transparency
from scripts.release_deploy_evidence import PreparedRelease
from tests._unlogged_signing_support import (
    ENV,
    LOGGED_BUNDLE,
    REASON,
    ROOT,
    UNLOGGED_BUNDLE,
)
from tests._unlogged_signing_support import write_json as _write


def _signing(root: Path) -> staging_deploy.SigningMaterial:
    from tests._staging_deploy_support import signing_material

    return signing_material(root)


# -- the admin CLI hops ----------------------------------------------------------


def _deploy(*arguments: str) -> argparse.Namespace:
    return cli.parser().parse_args(["deploy", "--state-dir", "/state", *arguments])


def test_the_deploy_command_takes_both_flags_or_neither() -> None:
    assert deploy_command.unlogged_signature_reason(_deploy()) is None
    assert (
        deploy_command.unlogged_signature_reason(
            _deploy("--allow-unlogged-signature", "--reason", REASON)
        )
        == REASON
    )
    for arguments in (
        _deploy("--reason", REASON),
        _deploy("--allow-unlogged-signature"),
        _deploy("--allow-unlogged-signature", "--reason", "x" * 301),
    ):
        with pytest.raises(SiteConfigError):
            deploy_command.unlogged_signature_reason(arguments)


def test_a_rollback_takes_neither_flag() -> None:
    with pytest.raises(SiteConfigError, match="--allow-unlogged-signature"):
        deploy_command.refuse_rollback_options(
            _deploy("--rollback", "--allow-unlogged-signature", "--reason", REASON)
        )


def test_the_rerun_line_repeats_the_consent_and_its_reason() -> None:
    line = deploy_command.deploy_rerun_command(
        _deploy("--allow-unlogged-signature", "--reason", REASON), Path("/state")
    )
    assert "--allow-unlogged-signature" in line
    assert f"--reason '{REASON}'" in line
    assert "--reason" not in deploy_command.deploy_rerun_command(
        _deploy(), Path("/state")
    )


def test_the_public_hop_hands_the_preparer_both_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def driver(command: list[str], **options: Any) -> Any:
        commands.append(list(command))
        environments.append(dict(options["env"]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(source_deploy, "run_driver", driver)
    monkeypatch.setattr(
        source_deploy, "resolve_source_repository", lambda *_a, **_k: tmp_path
    )
    for reason in (REASON, None):
        source_deploy.run_source_deploy(
            cpu_cluster_arn="cpu",
            gpu_cluster_arns=("gpu",),
            state_dir=tmp_path / "state",
            admin_email="operations@example.com",
            impact_base="origin/main",
            current_directory=tmp_path,
            unlogged_signature_reason=reason,
        )
    with_reason, without = commands
    index = with_reason.index("--allow-unlogged-signature")
    assert with_reason[index + 1 : index + 3] == ["--reason", REASON]
    assert "--allow-unlogged-signature" not in without and "--reason" not in without


def test_the_inner_hop_sets_the_reason_only_for_its_own_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setenv(ENV, "stale export")

    class Result:
        site_file = tmp_path / "site.yaml"
        pending_gpu_cluster_arns: tuple[str, ...] = ()

    def bootstrap(_request: Any) -> Result:
        seen["bootstrap"] = os.environ.get(ENV)
        return Result()

    def release(**options: Any) -> int:
        seen["release"] = options.get("allow_unlogged_signature", False)
        seen["release_env"] = os.environ.get(ENV)
        return 0

    from contextlib import nullcontext

    monkeypatch.setattr(
        deploy_command, "administrator_operation_lock", lambda _path: nullcontext()
    )
    hooks = deploy_command.DeployHooks(
        run_source_deploy=lambda **_k: 0,
        bootstrap_from_arns=bootstrap,
        run_automatic_release=release,
        join_clusters=lambda *_a: None,
        run_rollback=lambda *_a, **_k: 0,
        approve_profile_plan_inline=lambda *_a, **_k: {},
        discover_cluster=lambda *_a, **_k: None,
        load_site=lambda *_a, **_k: None,
        release_consent_environment=lambda _arguments: {},
        grafana_environment=lambda _arguments: {},
        grafana_request_fields=lambda _arguments: {},
    )
    for flags, expected in (
        (("--allow-unlogged-signature", "--reason", REASON), REASON),
        ((), None),
    ):
        arguments = cli.parser().parse_args(
            [
                "deploy",
                "--state-dir",
                str(tmp_path),
                "--prepared-source-release",
                "--repo-root",
                str(tmp_path),
                *flags,
            ]
        )
        assert (
            deploy_command.run_prepared_deploy(
                arguments,
                cpu_cluster_arn="cpu",
                gpu_cluster_arns=("gpu",),
                admin_email="operations@example.com",
                hooks=hooks,
            )
            == 0
        )
        assert seen["bootstrap"] == seen["release_env"] == expected
        assert seen["release"] is (expected is not None)
        assert os.environ.get(ENV) == "stale export", "the hop leaked its consent"


def test_the_release_child_passes_the_switch_to_the_release_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        release_child,
        "run_driver",
        lambda command, **_k: commands.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    for allowed in (True, False):
        release_child.run_automatic_release(
            repository_root=tmp_path,
            site_file=tmp_path / "site.yaml",
            state_dir=tmp_path,
            staging_only_release=False,
            allow_unlogged_signature=allowed,
        )
    assert "--allow-unlogged-signature" in commands[0]
    assert "--allow-unlogged-signature" not in commands[1]
    parsed = release_deploy.build_parser().parse_args(
        ["--prebuilt-attestation", "a.json", "--allow-unlogged-signature"]
    )
    assert parsed.allow_unlogged_signature is True


# -- the release build ----------------------------------------------------------


@pytest.mark.parametrize(
    ("bundle", "allowed", "expected"),
    [
        (LOGGED_BUNDLE, False, []),
        ({}, False, []),
        (UNLOGGED_BUNDLE, True, ["--allow-unlogged-signature"]),
        (UNLOGGED_BUNDLE, False, ["--live-release-site", "SITE"]),
    ],
    ids=["logged", "unknown", "unlogged-consent", "unlogged-live-release"],
)
def test_the_prebuilt_release_verifier_gets_exactly_one_approval_path(
    tmp_path: Path, bundle: dict[str, Any], allowed: bool, expected: list[str]
) -> None:
    root, state = tmp_path / "repo", tmp_path / "state"
    _write(root / "dist/current-attestation.json", {})
    _write(root / "dist/current-attestation.bundle.json", bundle)
    _write(state / "release-signing/cosign.pub", "public")
    (state / "site.yaml").write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    class Runner:
        def run(self, arguments: list[str], **_options: Any) -> str:
            commands.append(list(arguments))
            return ""

    release_artifacts.verify_prebuilt_release(
        Runner(),  # type: ignore[arg-type]
        repository_root=root,
        state_dir=state,
        allow_unlogged=allowed,
    )
    tail = commands[0][commands[0].index("--cosign-key") + 2 :]
    assert tail == [item.replace("SITE", str(state / "site.yaml")) for item in expected]


def test_an_unlogged_release_is_not_reused_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    for name in ("current-release.json", "current-attestation.json"):
        _write(root / "dist" / name, {})
    _write(root / "dist/current-attestation.bundle.json", UNLOGGED_BUNDLE)

    class Runner:
        def run(self, *_arguments: Any, **_options: Any) -> str:
            raise AssertionError("an unlogged release was verified for reuse")

    assert (
        release_artifacts.load_reusable_signed_release(
            Runner(),  # type: ignore[arg-type]
            repository_root=root,
            state_dir=tmp_path / "state",
            region="us-west-2",
            runtime_repository="repo",
            runtime_profile="hyperpod-v1",
            staging_only=False,
            impact_base="origin/main",
        )
        is None
    )


def test_the_build_reads_the_reason_its_hop_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert release_artifacts.consented_unlogged_reason() is None
    monkeypatch.setenv(ENV, f" {REASON} ")
    assert release_artifacts.consented_unlogged_reason() == REASON
    monkeypatch.setenv(ENV, "")
    with pytest.raises(BootstrapError):
        release_artifacts.consented_unlogged_reason()


class MakeRunner:
    """A ``CommandRunner`` whose make fails after the wrapper's refusal note."""

    def __init__(self, repository: Path, *, refused: bool) -> None:
        self.repository = repository
        self.refused = refused
        self.commands: list[list[str]] = []

    def run(self, arguments: list[str], **_options: Any) -> str:
        self.commands.append(list(arguments))
        bundle = self.repository / "dist/current-attestation.bundle.json"
        if self.refused:
            transparency.write_refusal_marker(
                bundle, artifact=bundle, attempts=3, cause="status 500"
            )
        raise BootstrapError("command failed (2): make release-build")


@pytest.mark.parametrize("refused", [True, False])
def test_a_refused_release_signing_says_nothing_was_rolled_out(
    tmp_path: Path, refused: bool
) -> None:
    (tmp_path / "dist").mkdir()
    stale = tmp_path / "dist/current-attestation.bundle.json"
    transparency.write_refusal_marker(
        stale, artifact=stale, attempts=3, cause="an older build"
    )
    runner = MakeRunner(tmp_path, refused=refused)

    with pytest.raises(BootstrapError) as failure:
        release_artifacts.run_release_make(
            runner,  # type: ignore[arg-type]
            ["make", "release-build"],
            repository_root=tmp_path,
            environment={},
        )

    if refused:
        message = str(failure.value)
        assert "nothing was rolled out and the live release is unchanged" in message
        assert '--allow-unlogged-signature --reason "<why>"' in message
        assert "status 500" in message
    else:
        assert str(failure.value) == "command failed (2): make release-build", (
            "a stale refusal note of an earlier build was reported"
        )
    assert runner.commands == [["make", "release-build"]]


# -- the release record, the summary line and status ------------------------------


def _prepared(state: Path) -> PreparedRelease:
    record = state / "release-deploy" / "release-a"
    _write(record / "state.json", {"phase": "PREPARED", "release_id": "release-a"})
    return PreparedRelease(
        site_file=state / "site.yaml",
        release_id="release-a",
        runtime_profile_version="hyperpod-v1",
        agent_config_digest="a" * 64,
        profile_change_kind="NONE",
        profile_approval=None,
        state_dir=record,
        site_changed=False,
    )


def _record_unlogged(state: Path, *, declared_at: str) -> Path:
    prepared = _prepared(state)
    attestation = _write(state / "dist/current-attestation.json", {"a": 1})
    bundle = _write(state / "dist/current-attestation.bundle.json", UNLOGGED_BUNDLE)
    release_deploy.record_release_signing(
        prepared,
        {
            "transparency_log": "unlogged",
            "reason": REASON,
            "declared_at": declared_at,
            "approval": "consent",
            "attestation_sha256": release_attestation.sha256(attestation),
            "trusted_root": "tuf",
        },
        (attestation, bundle),
    )
    return prepared.state_dir


def test_the_release_record_keeps_the_mode_and_a_copy_of_the_signature(
    tmp_path: Path,
) -> None:
    record = _record_unlogged(tmp_path, declared_at=transparency.utc_timestamp())
    state = json.loads((record / "state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "PREPARED"
    assert state["signing"]["transparency_log"] == "unlogged"
    assert state["signing"]["bundle"] == str(record / "attestation.sigstore.json")
    assert (record / "attestation.json").read_bytes() == (
        tmp_path / "dist/current-attestation.json"
    ).read_bytes()
    lines = release_deploy.signing_summary_lines(
        state["signing"], release_id="release-a", state_dir=tmp_path
    )
    assert lines and "is signed without a transparency-log entry" in lines[0]
    assert "backfill-transparency-log" in lines[0]


def _status(state: Path, capsys: pytest.CaptureFixture[str]) -> tuple[bool, dict, str]:
    (state / "site.yaml").write_text("{}", encoding="utf-8")
    failing = status_report.print_status_report(
        json.dumps(
            {
                "mode": "status",
                "healthy": True,
                "health": {"summary": {"PASS": 1}, "checks": []},
                "live_release": {"release_id": "release-a"},
            }
        ),
        site_file=state / "site.yaml",
    )
    captured = capsys.readouterr()
    return failing, json.loads(captured.out), captured.err


def test_status_names_an_unlogged_release_then_fails_after_72_hours(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = transparency.utc_timestamp(datetime.now(UTC) - timedelta(hours=2))
    _record_unlogged(tmp_path, declared_at=fresh)
    failing, document, header = _status(tmp_path, capsys)
    assert failing is False and document["healthy"] is True
    assert document["release_signing"]["status"] == "ADVISORY"
    assert "finding: release release-a is signed without a transparency-log" in header

    overdue = transparency.utc_timestamp(datetime.now(UTC) - timedelta(hours=73))
    _record_unlogged(tmp_path, declared_at=overdue)
    failing, document, header = _status(tmp_path, capsys)
    assert failing is True and document["healthy"] is False
    assert document["release_signing"]["status"] == "FAIL"
    assert "failing checks: release_signature" in header
    assert staging_live_evidence.deploy_healthy(document), (
        "the overdue signing check gated a deploy"
    )


def test_a_backfilled_release_clears_the_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record_unlogged(
        tmp_path,
        declared_at=transparency.utc_timestamp(datetime.now(UTC) - timedelta(hours=80)),
    )
    _write(record / "attestation.sigstore.json", LOGGED_BUNDLE)
    failing, document, header = _status(tmp_path, capsys)
    assert failing is False and document["release_signing"]["status"] == "BACKFILLED"
    assert "transparency-log entry" not in header


def test_status_warns_only_about_an_old_trust_pin(tmp_path: Path) -> None:
    directory = tmp_path / "release-signing"
    for age, findings in ((5, 0), (31, 1)):
        _write(
            directory / admin_signing.PIN_RECORD,
            {
                "pinned_at": transparency.utc_timestamp(
                    datetime.now(UTC) - timedelta(days=age)
                ),
                "tuf_last_timestamp": "2026-11-07T14:21:08Z",
            },
        )
        field, lines = admin_signing.trust_pin_view(tmp_path)
        assert field is not None and len(lines) == findings
        assert all(line.startswith("WARN") for line in lines), lines


# -- the backfill -------------------------------------------------------------------


@pytest.fixture
def rekor() -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"url": "", "posts": [], "entries": {}, "conflict": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["posts"].append(json.loads(body))
            canonical = json.dumps(json.loads(body), separators=(",", ":")).encode()
            uuid = "108e9186e8c5677a" + hashlib.sha256(b"\x00" + canonical).hexdigest()
            entry = {
                "body": base64.b64encode(canonical).decode(),
                "integratedTime": 1793189892,
                "logID": "c0" * 32,
                "logIndex": 7,
                "verification": {
                    "inclusionProof": {
                        "checkpoint": "rekor - 1\n2\nroot\n",
                        "hashes": ["ab" * 32],
                        "logIndex": 6,
                        "rootHash": "ef" * 32,
                        "treeSize": 8,
                    },
                    "signedEntryTimestamp": "MEUCIQ==",
                },
            }
            state["entries"][uuid] = entry
            if state["conflict"]:
                payload = {"code": 409, "message": f"exists with UUID {uuid}"}
                self.send_response(409)
                self.send_header("Location", f"/api/v1/log/entries/{uuid}")
            else:
                payload = {uuid: entry}
                self.send_response(201)
            data = json.dumps(payload).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            uuid = self.path.rsplit("/", 1)[-1]
            data = json.dumps({uuid: state["entries"][uuid]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_arguments: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


PUBLIC_PEM = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEESFXE3kS1dflKaBQx+IWJbgs3xXM
jME6AXcluBD9Ej5GBq7VS0tjJDXOoMNq3W71kVapUjOkWTbxblbx0+ZHtQ==
-----END PUBLIC KEY-----
"""


def _fake_cosign(tmp_path: Path, *, offline_ok: bool = True) -> Path:
    path = tmp_path / "cosign"
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "argv = sys.argv[1:]\n"
        f"offline_ok = {offline_ok!r}\n"
        "if '--insecure-ignore-tlog=true' in argv and not offline_ok:\n"
        "    sys.stderr.write('invalid signature'); raise SystemExit(1)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("conflict", [False, True], ids=["fresh", "already-logged"])
def test_backfill_logs_the_same_signature_and_rewrites_the_bundle(
    tmp_path: Path, rekor: dict[str, Any], conflict: bool
) -> None:
    rekor["conflict"] = conflict
    artifact = _write(tmp_path / "attestation.json", {"a": 1})
    bundle = _write(tmp_path / "attestation.sigstore.json", UNLOGGED_BUNDLE)
    public_key = tmp_path / "cosign.pub"
    public_key.write_text(PUBLIC_PEM, encoding="utf-8")

    outcome = backfill_transparency_log.backfill(
        artifact,
        bundle,
        public_key=public_key,
        cosign=str(_fake_cosign(tmp_path)),
        rekor_url=rekor["url"],
    )

    assert outcome == "backfilled"
    (posted,) = rekor["posts"]
    assert posted["spec"]["signature"]["content"] == "c2lnbmF0dXJl"
    assert (
        posted["spec"]["data"]["hash"]["value"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    rewritten = json.loads(bundle.read_text(encoding="utf-8"))
    assert rewritten["messageSignature"]["signature"] == "c2lnbmF0dXJl"
    assert transparency.bundle_transparency(bundle) == transparency.LOGGED
    assert (
        backfill_transparency_log.backfill(
            artifact,
            bundle,
            public_key=public_key,
            cosign=str(_fake_cosign(tmp_path)),
            rekor_url=rekor["url"],
        )
        == "already_logged"
    )


def test_backfill_uploads_nothing_the_site_key_did_not_sign(
    tmp_path: Path, rekor: dict[str, Any]
) -> None:
    artifact = _write(tmp_path / "attestation.json", {"a": 1})
    bundle = _write(tmp_path / "attestation.sigstore.json", UNLOGGED_BUNDLE)
    public_key = tmp_path / "cosign.pub"
    public_key.write_text(PUBLIC_PEM, encoding="utf-8")
    with pytest.raises(
        backfill_transparency_log.BackfillError, match="nothing uploaded"
    ):
        backfill_transparency_log.backfill(
            artifact,
            bundle,
            public_key=public_key,
            cosign=str(_fake_cosign(tmp_path, offline_ok=False)),
            rekor_url=rekor["url"],
        )
    assert rekor["posts"] == []
    assert transparency.bundle_transparency(bundle) == transparency.UNLOGGED


def test_backfill_finds_every_signature_the_site_key_makes(tmp_path: Path) -> None:
    state, root = tmp_path / "state", tmp_path / "repo"
    expected = [
        (
            root / "dist/current-attestation.json",
            root / "dist/current-attestation.bundle.json",
        ),
        (
            state / "release-deploy/release-a/attestation.json",
            state / "release-deploy/release-a/attestation.sigstore.json",
        ),
        (
            state / "source-deploy-success.json",
            state / "source-deploy-success.sigstore.json",
        ),
        (
            state / "deploy-host/by-content/x/gpu-fault-deploy-host-p.tar.gz",
            state / "deploy-host/by-content/x/gpu-fault-deploy-host-p.sigstore.json",
        ),
        (
            state / "ci-candidates/c/verification.json",
            state / "ci-candidates/c/verification.sigstore.json",
        ),
    ]
    for artifact, bundle in expected:
        _write(artifact, {})
        _write(bundle, {})
    assert backfill_transparency_log.candidates(state, root) == expected


def test_the_backfill_verb_runs_the_script_under_the_site_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from gpu_fault.admin import transparency_backfill

    events: list[str] = []

    @contextmanager
    def lock(path: Path) -> Iterator[int]:
        events.append(f"lock {path}")
        yield 3

    commands: list[list[str]] = []
    monkeypatch.setattr(transparency_backfill, "administrator_operation_lock", lock)
    monkeypatch.setattr(
        transparency_backfill,
        "run_driver",
        lambda command, **_k: commands.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    class Site:
        source = tmp_path / "site.yaml"
        repository_root = tmp_path / "repo"

    assert transparency_backfill.run_command(argparse.Namespace(), site=Site()) == 0  # type: ignore[arg-type]
    assert events == [f"lock {tmp_path}"]
    (command,) = commands
    assert command[1] == str(tmp_path / "repo/scripts/backfill_transparency_log.py")
    assert command[command.index("--public-key") + 1] == str(
        tmp_path / "release-signing/cosign.pub"
    )
    parsed = cli.parser().parse_args(
        ["backfill-transparency-log", "--state-dir", str(tmp_path)]
    )
    assert parsed.command == transparency_backfill.VERB


# -- the rollout order ---------------------------------------------------------------


def test_no_in_cluster_component_ever_verifies_a_cosign_signature() -> None:
    """Nodes, the installer, the reconciler and the control plane check SHA-256
    pins the release engine hands them; only the deploy host runs cosign. So a
    bundle without a log entry never meets an older in-cluster verifier, and the
    first release that carries the switch can use it."""

    offenders = [
        str(path.relative_to(ROOT))
        for directory in ("deploy", "src/gpu_fault_release")
        for path in (ROOT / directory).rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh", ".yaml", ".json"}
        and "cosign" in path.read_text(encoding="utf-8", errors="replace").lower()
        and path.name != "regional_release_signing.py"
    ]
    assert offenders == []
    for module in ("api", "app", "node_agent", "cluster_executor", "orchestration"):
        for path in (ROOT / "src/gpu_fault" / module).rglob("*.py"):
            assert "verify-blob" not in path.read_text(encoding="utf-8"), path


def test_the_deploy_host_venv_is_rebuilt_before_the_inner_hop_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every verifier of a deploy runs the candidate's code: the source preparer
    from the candidate checkout, and the inner deploy from the venv just built
    from the candidate's own signed archive."""

    from tests._staging_deploy_support import (
        deploy_arguments,
        managed_state,
        stub_deploy_orchestration,
    )

    repository, state = managed_state(tmp_path)
    events: list[str] = []
    stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=staging_deploy.SourceCheckout(
            repository_root=repository,
            git_commit="a" * 40,
            fingerprint="f" * 64,
            snapshot=False,
        ),
        signing=_signing(state),
        previous=None,
        events=events,
    )
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        staging_deploy,
        "ensure_deploy_host_venv",
        lambda *_a, **kwargs: events.append("venv")
        or seen.setdefault("venv", kwargs)
        and state / "deployer-venv",
    )
    monkeypatch.setattr(
        staging_deploy,
        "run_admin_deploy",
        lambda **kwargs: events.append("deploy") or seen.setdefault("deploy", kwargs),
    )
    monkeypatch.setattr(
        staging_deploy, "release_record_evidence", lambda _state: {"release_id": "r"}
    )
    arguments = deploy_arguments(repository, state)
    arguments.allow_unlogged_signature = True
    arguments.reason = REASON

    staging_deploy.deploy(arguments)

    assert events.index("venv") < events.index("deploy")
    assert seen["venv"]["allow_unlogged"] is True
    assert seen["deploy"]["unlogged_reason"] == REASON


@pytest.mark.parametrize("allowed", [True, False])
def test_the_release_driver_carries_the_switch_and_records_the_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allowed: bool,
) -> None:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("{}", encoding="utf-8")
    attestation = _write(tmp_path / "attestation.json", {"a": 1})
    bundle = _write(tmp_path / "attestation.bundle.json", UNLOGGED_BUNDLE)
    seen: dict[str, Any] = {}

    def verify(*_arguments: Any, **options: Any) -> tuple[dict[str, Any], Any]:
        seen["verify"] = options
        return (
            {"subject": {"release_id": "release-a", "manifest_sha256": "m" * 64}},
            release_attestation.SignatureVerification(
                transparency_log="unlogged",
                declaration=transparency.unlogged_declaration(REASON),
                approval="consent" if allowed else "live_release",
            ),
        )

    def execute(_site: Path, **options: Any) -> PreparedRelease:
        seen["execute"] = options
        prepared = _prepared(tmp_path)
        release_deploy.record_release_signing(
            prepared, options["signing"], options["signing_artifacts"]
        )
        return prepared

    monkeypatch.setattr(release_deploy, "resolve_site_file", lambda *_a: site_file)
    monkeypatch.setattr(release_deploy, "verify_attestation_signature", verify)
    monkeypatch.setattr(
        release_deploy, "release_manifest", lambda _root: ({}, "release-a")
    )
    monkeypatch.setattr(release_deploy, "sha256_file", lambda _path: "m" * 64)
    monkeypatch.setattr(release_deploy, "execute_release", execute)

    assert (
        release_deploy.main(
            [
                "--site",
                str(site_file),
                "--prebuilt-attestation",
                str(attestation),
                "--prebuilt-bundle",
                str(bundle),
                "--cosign-key",
                str(tmp_path / "cosign.pub"),
                *(["--allow-unlogged-signature"] if allowed else []),
            ]
        )
        == 0
    )

    assert seen["verify"]["allow_unlogged"] is allowed
    assert callable(seen["verify"]["live_release"]), "config cannot approve"
    assert seen["execute"]["signing"]["transparency_log"] == "unlogged"
    assert seen["execute"]["signing_artifacts"] == (attestation, bundle)
    captured = capsys.readouterr()
    assert json.loads(captured.out)["signing"]["reason"] == REASON
    assert "release release-a is signed without a transparency-log entry" in (
        captured.err
    )
