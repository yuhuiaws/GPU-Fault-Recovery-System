from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import ci_candidate_receipt, restore_ci_candidate


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_candidate_lookup_requires_local_origin_main(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "value").write_text("one\n", encoding="utf-8")
    _git(repository, "add", "value")
    _git(repository, "commit", "-m", "one")
    _git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")

    assert restore_ci_candidate.matches_local_main_ref(repository), (
        "matching local origin/main was not eligible for candidate lookup"
    )

    (repository / "value").write_text("two\n", encoding="utf-8")
    _git(repository, "commit", "-am", "two")

    assert not restore_ci_candidate.matches_local_main_ref(repository), (
        "a local commit ahead of origin/main was eligible for candidate lookup"
    )


def test_non_main_candidate_restore_does_not_request_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        restore_ci_candidate, "repository_slug", lambda _root: "owner/repository"
    )
    monkeypatch.setattr(
        restore_ci_candidate, "_existing_candidate", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        restore_ci_candidate, "matches_local_main_ref", lambda _root: False
    )
    monkeypatch.setattr(
        restore_ci_candidate,
        "github_token",
        lambda: pytest.fail("non-main restore requested GitHub credentials"),
    )

    with pytest.raises(
        restore_ci_candidate.CandidateUnavailable, match="does not equal local"
    ):
        restore_ci_candidate.restore_candidate(tmp_path, tmp_path / "dist")


def test_candidate_verification_covers_gate_unit_and_all_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    unit = candidate / "ci-domains/unit"
    shards = unit / "shards"
    unit.mkdir(parents=True)
    gate = {
        "repository": "owner/repository",
        "run_id": "123",
        "source": {"git_commit": "a" * 40, "git_tree": "b" * 40},
        "domains": {"unit": {"path": "ci-domains/unit/unit-gate.json"}},
    }
    (candidate / "ci-gate.json").write_text(json.dumps(gate), encoding="utf-8")
    (candidate / "ci-gate.bundle.json").write_text("{}", encoding="utf-8")
    (unit / "unit-gate.json").write_text("{}", encoding="utf-8")
    (unit / "unit-gate.bundle.json").write_text("{}", encoding="utf-8")
    for name in restore_ci_candidate.EXPECTED_SHARDS:
        root = shards / name
        root.mkdir(parents=True)
        (root / "coverage-shard-gate.json").write_text("{}", encoding="utf-8")
        (root / "coverage-shard-gate.bundle.json").write_text("{}", encoding="utf-8")

    verified: list[Path] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(
        restore_ci_candidate,
        "_git",
        lambda _root, *arguments: {
            ("rev-parse", "HEAD"): "a" * 40,
            ("rev-parse", "HEAD^{tree}"): "b" * 40,
        }[arguments],
    )
    monkeypatch.setattr(
        restore_ci_candidate,
        "_verify_blob",
        lambda path, **_kwargs: verified.append(path),
    )
    monkeypatch.setattr(
        restore_ci_candidate,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )

    assert (
        restore_ci_candidate.verify_candidate(
            tmp_path, candidate, repository="owner/repository"
        )
        == gate
    )
    assert len(verified) == 8
    assert {path.parent.name for path in verified if "coverage" in path.name} == (
        restore_ci_candidate.EXPECTED_SHARDS
    )
    assert any("ci_gate.py" in item for item in commands[0]), (
        "candidate artifact inventory was not verified by ci_gate.py"
    )


def test_candidate_receipt_binds_gate_and_current_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    state = tmp_path / "state"
    gate = root / "dist/ci-gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "repository": "owner/repository",
                "run_id": "123",
                "source": {"git_commit": "a" * 40, "git_tree": "b" * 40},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ci_candidate_receipt,
        "_git",
        lambda _root, *arguments: {
            ("rev-parse", "HEAD"): "a" * 40,
            ("rev-parse", "HEAD^{tree}"): "b" * 40,
        }[arguments],
    )

    def cosign(command, **_kwargs):
        if "sign-blob" in command:
            bundle = Path(command[command.index("--bundle") + 1])
            bundle.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ci_candidate_receipt.subprocess, "run", cosign)
    value = ci_candidate_receipt.write_receipt(
        root,
        state,
        gate_path=gate,
        repository="owner/repository",
        run_id=123,
        signing_key=tmp_path / "cosign.key",
    )

    verified = ci_candidate_receipt.verify_receipt(
        root, state, public_key=tmp_path / "cosign.pub"
    )

    assert verified is not None
    assert verified["ci_gate"] == str(gate)
    assert value["source"]["git_tree"] == "b" * 40
    gate.write_text("{}", encoding="utf-8")
    with pytest.raises(
        ci_candidate_receipt.CandidateReceiptError, match="gate has changed"
    ):
        ci_candidate_receipt.verify_receipt(
            root, state, public_key=tmp_path / "cosign.pub"
        )
