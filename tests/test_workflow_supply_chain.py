"""Workflow supply-chain contract: least-privilege tokens and SHA-pinned actions.

M-20: the CI workflow default must not hand every job an OIDC token-mint
capability; only the jobs that keyless-sign may carry ``id-token: write``.
M-21: every external action must be pinned to a full 40-char commit SHA, not a
mutable tag, so a compromised tag cannot inject code into the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github/workflows/ci.yml"
RELEASE_PATH = ROOT / ".github/workflows/release.yml"
CI = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))

# Only these jobs cosign-sign a gate (or restore/read a prior signed shard) and
# therefore legitimately need the OIDC token.
JOBS_ALLOWED_ID_TOKEN = {"coverage", "postgres", "unit", "test"}
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
SHA_PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def test_ci_workflow_default_permissions_are_least_privilege() -> None:
    assert CI["permissions"] == {"contents": "read"}, CI["permissions"]


def test_id_token_write_is_scoped_to_signing_jobs_only() -> None:
    for name, job in CI["jobs"].items():
        token = (job.get("permissions") or {}).get("id-token")
        if name in JOBS_ALLOWED_ID_TOKEN:
            assert token == "write", f"{name} must mint an OIDC token to sign"
        else:
            assert token != "write", (
                f"{name} is granted id-token: write but does not sign"
            )


def _external_uses(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [value for value in USES.findall(text) if not value.startswith("./")]


def test_every_external_action_is_pinned_to_a_commit_sha() -> None:
    for path in (CI_PATH, RELEASE_PATH):
        for value in _external_uses(path):
            assert SHA_PINNED.match(value), (
                f"{path.name}: action is not pinned to a 40-char SHA: {value}"
            )


def test_no_external_action_uses_a_mutable_tag() -> None:
    for path in (CI_PATH, RELEASE_PATH):
        for value in _external_uses(path):
            assert not re.search(r"@v\d", value), (
                f"{path.name}: action still uses a mutable tag: {value}"
            )
