from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_NAME = "verification.json"
SIGNATURE_NAME = "verification.sigstore.json"


class CandidateReceiptError(RuntimeError):
    pass


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CandidateReceiptError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateReceiptError(f"{description} is invalid") from exc
    if not isinstance(value, dict):
        raise CandidateReceiptError(f"{description} must be an object")
    return value


def _receipt_paths(root: Path, state_dir: Path) -> tuple[Path, Path]:
    commit = _git(root, "rev-parse", "HEAD")
    directory = state_dir / "ci-candidates" / commit
    return directory / RECEIPT_NAME, directory / SIGNATURE_NAME


def _validated_gate(
    root: Path,
    gate_path: Path,
    *,
    repository: str,
    run_id: int,
) -> dict[str, Any]:
    resolved = gate_path.expanduser().resolve()
    try:
        resolved.relative_to((root / "dist").resolve())
    except ValueError as exc:
        raise CandidateReceiptError("CI gate leaves repository dist") from exc
    gate = _load_object(resolved, "CI gate")
    source = gate.get("source")
    if (
        gate.get("repository") != repository
        or int(gate.get("run_id") or 0) != run_id
        or not isinstance(source, dict)
        or source.get("git_commit") != _git(root, "rev-parse", "HEAD")
        or source.get("git_tree") != _git(root, "rev-parse", "HEAD^{tree}")
    ):
        raise CandidateReceiptError("CI gate identity does not match checkout")
    return gate


def write_receipt(
    root: Path,
    state_dir: Path,
    *,
    gate_path: Path,
    repository: str,
    run_id: int,
    signing_key: Path,
) -> dict[str, Any]:
    gate = _validated_gate(
        root,
        gate_path,
        repository=repository,
        run_id=run_id,
    )
    receipt_path, signature_path = _receipt_paths(root, state_dir)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path.parent.chmod(0o700)
    value = {
        "schema_version": 1,
        "repository": repository,
        "run_id": run_id,
        "source": gate["source"],
        "ci_gate": {
            "path": str(gate_path.expanduser().resolve()),
            "sha256": _sha256(gate_path.expanduser().resolve()),
        },
    }
    temporary_receipt = receipt_path.with_suffix(".tmp")
    temporary_signature = signature_path.with_suffix(".tmp")
    temporary_receipt.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_receipt.chmod(0o600)
    temporary_signature.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            "cosign",
            "sign-blob",
            "--yes",
            "--key",
            str(signing_key),
            "--bundle",
            str(temporary_signature),
            str(temporary_receipt),
        ],
        cwd=root,
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        temporary_receipt.unlink(missing_ok=True)
        temporary_signature.unlink(missing_ok=True)
        raise CandidateReceiptError(
            "cannot sign CI candidate verification receipt: "
            + (completed.stderr or "").strip()
        )
    temporary_signature.chmod(0o600)
    os.replace(temporary_receipt, receipt_path)
    os.replace(temporary_signature, signature_path)
    return value


def verify_receipt(
    root: Path,
    state_dir: Path,
    *,
    public_key: Path,
) -> dict[str, Any] | None:
    receipt_path, signature_path = _receipt_paths(root, state_dir)
    present = receipt_path.is_file(), signature_path.is_file()
    if not any(present):
        return None
    if not all(present):
        raise CandidateReceiptError("CI candidate verification receipt is incomplete")
    completed = subprocess.run(
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(signature_path),
            "--key",
            str(public_key),
            str(receipt_path),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CandidateReceiptError(
            "CI candidate verification receipt signature is invalid"
        )
    value = _load_object(receipt_path, "CI candidate verification receipt")
    if value.get("schema_version") != 1:
        raise CandidateReceiptError(
            "CI candidate verification receipt schema is invalid"
        )
    source = value.get("source")
    gate = value.get("ci_gate")
    if (
        not isinstance(source, dict)
        or source.get("git_commit") != _git(root, "rev-parse", "HEAD")
        or source.get("git_tree") != _git(root, "rev-parse", "HEAD^{tree}")
        or not isinstance(gate, dict)
    ):
        raise CandidateReceiptError(
            "CI candidate verification receipt source is invalid"
        )
    gate_path = Path(str(gate.get("path") or "")).expanduser().resolve()
    if not gate_path.is_file() or _sha256(gate_path) != gate.get("sha256"):
        raise CandidateReceiptError(
            "CI candidate verification receipt gate has changed"
        )
    _validated_gate(
        root,
        gate_path,
        repository=str(value.get("repository") or ""),
        run_id=int(value.get("run_id") or 0),
    )
    return {**value, "ci_gate": str(gate_path)}


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write")
    write.add_argument("--root", type=Path, default=ROOT)
    write.add_argument("--state-dir", type=Path, required=True)
    write.add_argument("--gate", type=Path, required=True)
    write.add_argument("--repository", required=True)
    write.add_argument("--run-id", type=int, required=True)
    write.add_argument("--signing-key", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, default=ROOT)
    verify.add_argument("--state-dir", type=Path, required=True)
    verify.add_argument("--public-key", type=Path, required=True)
    options = parser.parse_args(arguments)
    root = options.root.expanduser().resolve()
    state_dir = options.state_dir.expanduser().resolve()
    try:
        if options.command == "write":
            value = write_receipt(
                root,
                state_dir,
                gate_path=options.gate,
                repository=options.repository,
                run_id=options.run_id,
                signing_key=options.signing_key,
            )
            result = {"available": True, **value}
        else:
            value = verify_receipt(
                root,
                state_dir,
                public_key=options.public_key,
            )
            result = (
                {"available": True, **value}
                if value is not None
                else {"available": False}
            )
    except (
        CandidateReceiptError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"ci-candidate-receipt: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
