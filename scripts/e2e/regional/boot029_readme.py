"""README-verbatim first deploy for GF-REGIONAL-BOOT-029 stage 5.

Stage 5 deploys the way ``README.md`` tells an administrator to: the build
block under ``## 构建和验证`` (its verification lines skipped -- the deploy runs
the release gates itself) followed by the four-argument ``gpu-fault-admin
deploy`` under ``## 部署：先区分角色``, executed from a pristine copy of the
checkout in a clean environment. The README text is the source of truth:
nothing here spells out the commands, so a README that no longer deploys fails
this stage, and so does a README procedure that only works with something the
README does not mention.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.e2e.regional.boot029_receipts import digest
from scripts.staging_source_snapshot import prepare_source_checkout
from scripts.staging_state_hygiene import SourceCheckout, StagingDeployError

BUILD_HEADING = "## 构建和验证"
DEPLOY_HEADING = "## 部署：先区分角色"
DEPLOY_COMMAND = "gpu-fault-admin deploy"
SETUP_LINE = "make deploy-host-setup-online"
ACTIVATE_LINES = (". .venv/bin/activate", "source .venv/bin/activate")
CPU_PLACEHOLDER = "<cpu-eks-or-hyperpod-arn>"
GPU_PLACEHOLDER = "<gpu-eks-or-hyperpod-arn>"
STATE_DIR_PLACEHOLDER = "/secure/gpu-fault"
EMAIL_PLACEHOLDER = "<operations-email>"
PLACEHOLDERS = (
    CPU_PLACEHOLDER,
    GPU_PLACEHOLDER,
    STATE_DIR_PLACEHOLDER,
    EMAIL_PLACEHOLDER,
)
VENV_COMPONENTS = frozenset({".venv", "deployer-venv", ".deployer-venv.versions"})
KEPT_NAMES = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "TMPDIR",
        "TERM",
        "SSH_AUTH_SOCK",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
    }
)
KEPT_PREFIXES = ("LC_", "DOCKER_", "AWS_CONTAINER_", "AWS_WEB_IDENTITY_")
SCRIPT_NAME = "readme-procedure.sh"
HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s.*|)$")
FENCE = re.compile(r"^ {0,3}```")


class ReadmeProcedureError(ValueError):
    """The README does not describe the procedure the stage exists to test."""


@dataclass(frozen=True)
class ReadmeProcedure:
    readme_sha256: str
    build_block: str
    deploy_block: str
    executed_lines: tuple[str, ...]
    skipped_verification_lines: tuple[str, ...]


def fenced_bash_block(text: str, heading: str) -> str:
    """The first ```` ```bash ```` block between ``heading`` and the next heading."""

    lines = text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == heading
        )
    except StopIteration:
        raise ReadmeProcedureError(f"README lacks the heading {heading!r}") from None
    block: list[str] = []
    inside = False
    for line in lines[start + 1 :]:
        if inside:
            if FENCE.match(line) and line.strip() == "```":
                return "\n".join(block)
            block.append(line)
        elif HEADING.match(line):
            break
        elif line.strip() == "```bash":
            inside = True
    raise ReadmeProcedureError(f"README has no ```bash block under {heading!r}")


def logical_commands(block: str) -> list[str]:
    """The block's commands, verbatim: a backslash continuation keeps its lines."""

    result: list[str] = []
    pending: list[str] = []
    for raw in block.splitlines():
        if raw.rstrip().endswith("\\"):
            pending.append(raw)
            continue
        if pending or raw.strip():
            result.append("\n".join([*pending, raw]))
        pending = []
    if pending:
        raise ReadmeProcedureError("README block ends in a line continuation")
    return result


def shell_words(command: str) -> list[str]:
    try:
        return [word for word in shlex.split(command) if word != "\n"]
    except ValueError as exc:
        raise ReadmeProcedureError(f"README command is not valid shell: {exc}") from exc


def is_verification_line(command: str) -> bool:
    """``make ... check`` and ``make ...-check`` verify; they do not deploy."""

    words = shell_words(command)
    return (
        bool(words)
        and words[0] == "make"
        and (any(word == "check" or word.endswith("-check") for word in words[1:]))
    )


def parse_readme(text: str) -> ReadmeProcedure:
    build_block = fenced_bash_block(text, BUILD_HEADING)
    deploy_block = fenced_bash_block(text, DEPLOY_HEADING)
    deploy_commands = logical_commands(deploy_block)
    if not deploy_commands or shell_words(deploy_commands[0])[:2] != (
        DEPLOY_COMMAND.split()
    ):
        raise ReadmeProcedureError(
            f"the block under {DEPLOY_HEADING!r} must start with {DEPLOY_COMMAND!r}"
        )
    for placeholder in PLACEHOLDERS:
        if deploy_block.count(placeholder) != 1:
            raise ReadmeProcedureError(
                f"the block under {DEPLOY_HEADING!r} must contain "
                f"{placeholder!r} exactly once"
            )
    executed: list[str] = []
    skipped: list[str] = []
    for command in logical_commands(build_block):
        (skipped if is_verification_line(command) else executed).append(command)
    words = [" ".join(shell_words(command)) for command in executed]
    if SETUP_LINE not in words or not any(
        activate in words[words.index(SETUP_LINE) + 1 :] for activate in ACTIVATE_LINES
    ):
        raise ReadmeProcedureError(
            f"the block under {BUILD_HEADING!r} must run {SETUP_LINE!r} and then "
            f"{ACTIVATE_LINES[0]!r} before the deploy command"
        )
    return ReadmeProcedure(
        readme_sha256=digest(text.encode()),
        build_block=build_block,
        deploy_block=deploy_block,
        executed_lines=tuple(executed),
        skipped_verification_lines=tuple(skipped),
    )


def substitute(block: str, values: Mapping[str, str]) -> str:
    for placeholder, value in values.items():
        block = block.replace(placeholder, value)
    return block


def render_deploy_command(
    block: str, *, cpu_arn: str, gpu_arn: str, state_dir: str, admin_email: str
) -> str:
    """The README command with the run's values; nothing added."""

    return substitute(
        block,
        {
            CPU_PLACEHOLDER: shlex.quote(cpu_arn),
            GPU_PLACEHOLDER: shlex.quote(gpu_arn),
            STATE_DIR_PLACEHOLDER: shlex.quote(state_dir),
            EMAIL_PLACEHOLDER: shlex.quote(admin_email),
        },
    )


def redacted_deploy_command(
    block: str, *, cpu_arn: str, gpu_arn: str, state_dir: str, admin_email: str
) -> str:
    """The same command for the receipt: every substituted value as a digest."""

    return substitute(
        block,
        {
            placeholder: "sha256:" + digest(value.encode())
            for placeholder, value in (
                (CPU_PLACEHOLDER, cpu_arn),
                (GPU_PLACEHOLDER, gpu_arn),
                (STATE_DIR_PLACEHOLDER, state_dir),
                (EMAIL_PLACEHOLDER, admin_email),
            )
        },
    )


def is_venv_entry(entry: str) -> bool:
    path = Path(entry)
    return (
        bool(VENV_COMPONENTS.intersection(path.parts))
        or (path.parent / "pyvenv.cfg").is_file()
    )


def supplies_admin(entry: str) -> bool:
    admin = Path(entry) / "gpu-fault-admin"
    return admin.is_file() and os.access(admin, os.X_OK)


def shadow_tool_directory(entry: str, shadow_root: Path, position: int) -> str:
    """A private stand-in for a tool directory that also offers gpu-fault-admin.

    Operators commonly keep kubectl, helm and an earlier deploy-host shim side
    by side in one directory (``/usr/local/bin``). Dropping the directory would
    take the README's prerequisite tools with it; keeping it would let the old
    shim pre-empt the README's venv. The stand-in links every other executable
    of the directory and omits ``gpu-fault-admin``.
    """

    source = Path(entry)
    shadow = shadow_root / f"{position:02d}-{source.name or 'root'}"
    shadow.mkdir(mode=0o700, parents=True, exist_ok=True)
    for tool in sorted(source.iterdir()):
        if tool.name == "gpu-fault-admin" or not tool.is_file():
            continue
        if not os.access(tool, os.X_OK):
            continue
        link = shadow / tool.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(tool)
    return str(shadow)


def operator_path(parent_path: str, shadow_root: Path | None = None) -> str:
    """The operator's PATH minus every entry that could pre-empt the README.

    The README's prerequisite is that the deploy host *has* the tools (aws,
    cosign, docker, kubectl, helm, ...); where the operator installed them is
    their environment, so their PATH stays, in order and de-duplicated. What
    must not stay is anything that already offers a ``gpu-fault-admin`` or a
    Python virtual environment: the README's ``. .venv/bin/activate`` has to be
    the only thing that puts a ``gpu-fault-admin`` (and its ``python``) on PATH,
    otherwise the stage would deploy with the driver's CLI and prove nothing.

    A plain tool directory that happens to hold a ``gpu-fault-admin`` too is
    replaced by a stand-in under ``shadow_root`` (see
    :func:`shadow_tool_directory`) so its other tools stay reachable; without a
    ``shadow_root`` it is dropped. Virtual environments are always dropped.
    """

    kept: list[str] = []
    seen: set[str] = set()
    for entry in parent_path.split(os.pathsep):
        if not entry or entry in seen or is_venv_entry(entry):
            continue
        seen.add(entry)
        if supplies_admin(entry):
            if shadow_root is None:
                continue
            entry = shadow_tool_directory(entry, shadow_root, len(kept))
        kept.append(entry)
    if not kept:
        raise ReadmeProcedureError("operator PATH has no entry left for the README")
    return os.pathsep.join(kept)


def clean_environment(
    parent: Mapping[str, str], shadow_root: Path | None = None
) -> dict[str, str]:
    """The allowlist the README procedure runs under: no driver state, operator PATH."""

    kept = {
        name: value
        for name, value in parent.items()
        if name in KEPT_NAMES or name.startswith(KEPT_PREFIXES)
    }
    kept["PATH"] = operator_path(parent.get("PATH", ""), shadow_root)
    return kept


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def pristine_source(repository: Path, work_dir: Path) -> SourceCheckout:
    """A worktree copy of the checkout as the deploy snapshots it: HEAD plus the
    uncommitted diff and untracked files, ignored files left behind, dirty
    trees committed as a snapshot. Refuses a copy that is not clean or already
    holds a venv."""

    checkout = prepare_source_checkout(repository, state_dir=work_dir)
    root = checkout.repository_root
    if (root / ".venv").exists() or (root / ".venv").is_symlink():
        raise ReadmeProcedureError("pristine source copy already holds .venv")
    status = _git(root, "status", "--porcelain")
    if status:
        raise ReadmeProcedureError("pristine source copy is not clean")
    if _git(root, "rev-parse", "HEAD") != checkout.git_commit:
        raise ReadmeProcedureError("pristine source copy commit changed")
    return checkout


def stage_script(
    source_dir: Path, executed_lines: tuple[str, ...], deploy_command: str
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# GF-REGIONAL-BOOT-029 stage 5: README.md's deployment procedure,",
            f"# verbatim. Build block {BUILD_HEADING} (verification lines skipped)",
            f"# then the deploy block {DEPLOY_HEADING} with its placeholders",
            "# substituted. Nothing else is added.",
            "set -euo pipefail",
            f"cd -- {shlex.quote(str(source_dir))}",
            *executed_lines,
            deploy_command,
            "",
        ]
    )


def readme_receipt(
    procedure: ReadmeProcedure,
    *,
    redacted_command: str,
    script: str,
    checkout: SourceCheckout,
    base_commit: str,
    environment: Mapping[str, str],
    parent_path: str,
    attempt: int,
    shadow_root: Path | None = None,
) -> dict[str, Any]:
    shadowed = (
        sorted(child.name for child in shadow_root.iterdir() if child.is_dir())
        if shadow_root is not None and shadow_root.is_dir()
        else []
    )
    return {
        "readme": {
            "readme_sha256": procedure.readme_sha256,
            "build_heading": BUILD_HEADING,
            "deploy_heading": DEPLOY_HEADING,
            "build_block_sha256": digest(procedure.build_block.encode()),
            "deploy_block_sha256": digest(procedure.deploy_block.encode()),
            "executed_lines": [*procedure.executed_lines, redacted_command],
            "skipped_verification_lines": list(procedure.skipped_verification_lines),
            "script_sha256": digest(script.encode()),
            "source_commit": checkout.git_commit,
            "source_base_commit": base_commit,
            "source_snapshot": checkout.snapshot,
            "source_dir_sha256": digest(str(checkout.repository_root).encode()),
            "venv_preexisted": False,
            "environment_names": sorted(environment),
            "path_sha256": digest(environment["PATH"].encode()),
            "path_entries_kept": len(environment["PATH"].split(os.pathsep)),
            "path_entries_dropped": len(
                {entry for entry in parent_path.split(os.pathsep) if entry}
            )
            - len(environment["PATH"].split(os.pathsep)),
            "path_entries_shadowed": len(shadowed),
            "attempt": attempt,
        }
    }


def run(
    *,
    readme: Path,
    repository: Path,
    work_dir: Path,
    attempt: int,
    cpu_arn: str,
    gpu_arn: str,
    state_dir: Path,
    admin_email: str,
    receipt: Path,
    parent_environment: Mapping[str, str],
) -> int:
    procedure = parse_readme(readme.read_text(encoding="utf-8"))
    attempt_dir = work_dir / f"attempt-{attempt}"
    attempt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    base_commit = _git(repository, "rev-parse", "HEAD")
    checkout = pristine_source(repository, attempt_dir)
    values = {
        "cpu_arn": cpu_arn,
        "gpu_arn": gpu_arn,
        "state_dir": str(state_dir),
        "admin_email": admin_email,
    }
    script = stage_script(
        checkout.repository_root,
        procedure.executed_lines,
        render_deploy_command(procedure.deploy_block, **values),
    )
    script_path = attempt_dir / SCRIPT_NAME
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o700)
    shadow_root = attempt_dir / "operator-path"
    environment = clean_environment(parent_environment, shadow_root)
    entry = readme_receipt(
        procedure,
        redacted_command=redacted_deploy_command(procedure.deploy_block, **values),
        script=script,
        checkout=checkout,
        base_commit=base_commit,
        environment=environment,
        parent_path=parent_environment.get("PATH", ""),
        attempt=attempt,
        shadow_root=shadow_root,
    )
    with receipt.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    print(
        f"+ README procedure attempt {attempt}: source {checkout.git_commit[:12]}, "
        f"{len(procedure.executed_lines)} build line(s), "
        f"{len(procedure.skipped_verification_lines)} verification line(s) "
        "skipped, then the deploy command",
        file=sys.stderr,
        flush=True,
    )
    completed = subprocess.run(["bash", str(script_path)], env=environment, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--cpu-arn", required=True)
    parser.add_argument("--gpu-arn", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--admin-email", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("attempt must be positive")
    try:
        code = run(
            readme=args.readme,
            repository=args.repo.resolve(),
            work_dir=args.work_dir,
            attempt=args.attempt,
            cpu_arn=args.cpu_arn,
            gpu_arn=args.gpu_arn,
            state_dir=args.state_dir.resolve(),
            admin_email=args.admin_email,
            receipt=args.receipt,
            parent_environment=os.environ,
        )
    except (
        ReadmeProcedureError,
        StagingDeployError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"FAIL: README procedure refused: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
