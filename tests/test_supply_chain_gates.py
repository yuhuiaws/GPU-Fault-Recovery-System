"""The supply-chain and string-table gates are wired the same way in make and CI.

``make ci-tooling-check`` already proves that ruff, yamllint and shellcheck
walk the same roots in the Makefile and in ``ci.yml``. These tests extend that
parity to the gates added for review items S8 and S10: the lazy-export table
check, CloudFormation linting, the documentation-facts check, ``pip-audit``
over every shipped lock, and the CycloneDX SBOM bound into the release
attestation. Tool versions live in one place (the Makefile) so the workflow
cannot drift from it; strictness is decided by ``CI`` so the same targets are
advisory on a developer machine and load-bearing on the runner.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
CI = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
RELEASE = yaml.safe_load(
    (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
)
SHIPPED_LOCKS = (
    "requirements/build.lock",
    "requirements/runtime.lock",
    "requirements/deploy-host.lock",
    "requirements/node-runtime.lock",
)
NEW_TARGETS = (
    "lazy-export-check",
    "cfn-lint-check",
    "doc-facts-check",
    "pip-audit-check",
    "sbom",
    "ci-supply-chain-tools",
)


def _target(name: str) -> str:
    body = MAKEFILE.split(f"\n{name}:\n", 1)[1]
    return body.split("\n\n", 1)[0]


def _static_runs() -> list[str]:
    return [
        step["run"]
        for step in CI["jobs"]["static"]["steps"]
        if isinstance(step.get("run"), str)
    ]


def test_new_targets_are_phony_and_defined() -> None:
    phony = MAKEFILE.split(".PHONY:", 1)[1].split("\n", 1)[0].split()

    for name in NEW_TARGETS:
        assert name in phony, f"{name} missing from .PHONY"
        assert f"\n{name}:\n" in MAKEFILE, f"{name} target is not defined"


def test_lazy_export_gate_runs_in_make_static_and_ci() -> None:
    assert "scripts/check-lazy-exports.py" in _target("lazy-export-check")
    sequential = _target("check-static-sequential")
    assert "$(MAKE) lazy-export-check" in sequential

    runs = _static_runs()
    architecture = runs.index("python scripts/check-python-architecture.py")
    assert runs[architecture + 1] == "make lazy-export-check PYTHON=python"


def test_doc_facts_gate_is_part_of_docs_check() -> None:
    assert "scripts/check-doc-facts.py" in _target("doc-facts-check")
    assert "$(MAKE) doc-facts-check" in _target("docs-static-check")


def test_tool_versions_are_pinned_once_in_the_makefile() -> None:
    for variable in ("PIP_AUDIT_VERSION", "CFN_LINT_VERSION", "CYCLONEDX_BOM_VERSION"):
        match = re.search(rf"^{variable} \?= (\d+\.\d+\.\d+)$", MAKEFILE, re.MULTILINE)
        assert match is not None, f"{variable} is not pinned in the Makefile"
    install = _target("ci-supply-chain-tools")
    assert "pip-audit==$(PIP_AUDIT_VERSION)" in install
    assert "cfn-lint==$(CFN_LINT_VERSION)" in install
    assert "cyclonedx-bom==$(CYCLONEDX_BOM_VERSION)" in install
    assert "$(SUPPLY_CHAIN_TOOLS_VENV)" in install, (
        "tools must install into their own venv, not the hash-locked one"
    )

    workflows = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (".github/workflows/ci.yml", ".github/workflows/release.yml")
    )
    for tool in ("pip-audit==", "cfn-lint==", "cyclonedx-bom=="):
        assert tool not in workflows, f"{tool} version must come from the Makefile"


def test_ci_installs_tools_then_lints_templates_and_audits_locks() -> None:
    runs = _static_runs()
    tools = runs.index("make ci-supply-chain-tools PYTHON=python")
    cfn = runs.index("make cfn-lint-check PYTHON=python")
    audit = runs.index("make pip-audit-check PYTHON=python")

    assert tools < cfn
    assert tools < audit


def test_cfn_lint_covers_the_lambda_templates() -> None:
    target = _target("cfn-lint-check")

    assert "cfn-lint" in target
    assert "deploy/aws/lambda/*.yaml" in MAKEFILE
    assert list((ROOT / "deploy/aws/lambda").glob("*.yaml")), (
        "the template set the gate covers is empty"
    )


def test_pip_audit_covers_every_shipped_lock_with_hashes() -> None:
    target = _target("pip-audit-check")

    assert "--require-hashes" in target
    assert "$(SHIPPED_LOCKS)" in target
    locks = re.search(r"^SHIPPED_LOCKS = (.+)$", MAKEFILE, re.MULTILINE)
    assert locks is not None
    assert tuple(locks.group(1).split()) == SHIPPED_LOCKS
    for lock in SHIPPED_LOCKS:
        assert (ROOT / lock).is_file(), lock
    assert "$(PIP_AUDIT_IGNORE_FILE)" in target


def test_pip_audit_ignore_entries_carry_a_fix_version() -> None:
    entries = [
        line
        for line in (ROOT / "requirements/pip-audit-ignore.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    for entry in entries:
        identifier, _, comment = entry.partition("#")
        assert re.fullmatch(r"(PYSEC|GHSA|CVE)-[A-Za-z0-9-]+", identifier.strip()), (
            entry
        )
        assert "fix:" in comment, f"ignore entry has no fix version: {entry}"
        assert re.search(r"\b[a-z0-9_.-]+ \d+\.\d+", comment), (
            f"ignore entry does not name package and pinned version: {entry}"
        )


def test_sbom_is_generated_from_shipped_locks_and_bound_into_attestations() -> None:
    sbom = _target("sbom")

    assert "cyclonedx-py" in sbom
    assert "requirements" in sbom
    assert "$(SHIPPED_LOCKS)" in sbom
    assert "--output-reproducible" in sbom
    for name in ("release-build", "release-build-promoted", "release-build-staging"):
        target = _target(name)
        sbom_step = target.index("$(MAKE) sbom")
        attestation = target.index("scripts/build-release-attestation.py")
        assert sbom_step < attestation, f"{name} must build the SBOM first"
        assert '--sbom-dir "$(SBOM_DIR)"' in target, name
    # Observable contract, not source text: the builder must accept the flag
    # the release targets pass it.
    usage = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build-release-attestation.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--sbom-dir" in usage

    release_runs = [
        step["run"]
        for step in RELEASE["jobs"]["build"]["steps"]
        if isinstance(step.get("run"), str)
    ]
    tools = next(
        index
        for index, run in enumerate(release_runs)
        if "make ci-supply-chain-tools PYTHON=python" in run
    )
    promoted = next(
        index
        for index, run in enumerate(release_runs)
        if "make release-build-promoted" in run
    )
    assert tools < promoted


def _make(target: str, *assignments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "make",
            target,
            "PYTHON=/nonexistent/bin/python",
            "SUPPLY_CHAIN_PYTHON=/nonexistent/bin/python",
            "DEPLOY_HOST_PLATFORM=test",
            *assignments,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_tools_are_advisory_locally_and_fatal_in_ci() -> None:
    for target in ("cfn-lint-check", "promtool-check", "pip-audit-check", "sbom"):
        local = _make(target, "CI=")
        assert local.returncode == 0, (target, local.stdout, local.stderr)
        assert "not installed" in local.stdout + local.stderr, target

        strict = _make(target, "CI=true")
        assert strict.returncode == 2, (target, strict.stdout, strict.stderr)
        assert "ci-supply-chain-tools" in strict.stderr, target
