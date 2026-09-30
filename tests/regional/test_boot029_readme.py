"""Contract of the README-verbatim stage of GF-REGIONAL-BOOT-029.

The parser is exercised against the repository's real README.md: the stage
exists so that the README's own deployment steps are what gets verified, so a
README edit that breaks the procedure must show up here before it shows up on
a build host.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.e2e.regional import boot029_readme as readme
from scripts.staging_state_hygiene import SourceCheckout

ROOT = Path(__file__).resolve().parents[2]
README_TEXT = (ROOT / "README.md").read_text(encoding="utf-8")
CPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/cpu-fixture"
GPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-fixture"
ADMIN_EMAIL = "admin@example.com"
VALUES = {
    "cpu_arn": CPU_ARN,
    "gpu_arn": GPU_ARN,
    "state_dir": "/private/state dir/second",
    "admin_email": ADMIN_EMAIL,
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(tmp_path: Path, name: str = "repo") -> Path:
    repository = tmp_path / name
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / ".gitignore").write_text("/.venv\ndist/\n", encoding="utf-8")
    (repository / "README.md").write_text(README_TEXT, encoding="utf-8")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "fixture")
    return repository


def test_real_readme_yields_the_three_step_procedure() -> None:
    procedure = readme.parse_readme(README_TEXT)

    assert procedure.readme_sha256 == _sha256(README_TEXT)
    assert procedure.executed_lines == (
        "make deploy-host-setup-online",
        ". .venv/bin/activate",
    ), "the README's build block must set up and activate the venv, nothing else"
    assert procedure.skipped_verification_lines == (
        "make PYTHON=.venv/bin/python check",
    ), "the README's check line is verification, not deployment"
    assert procedure.build_block.strip().splitlines() == [
        *procedure.executed_lines,
        *procedure.skipped_verification_lines,
    ], "every block line is either executed or recorded as skipped"
    words = readme.shell_words(procedure.deploy_block)
    assert words[:2] == ["gpu-fault-admin", "deploy"], words
    assert words[2:] == [
        "--cpu-cluster-arn",
        readme.CPU_PLACEHOLDER,
        "--gpu-cluster-arn",
        readme.GPU_PLACEHOLDER,
        "--state-dir",
        readme.STATE_DIR_PLACEHOLDER,
        "--admin-email",
        readme.EMAIL_PLACEHOLDER,
    ], "the README first deploy is the four-argument command and nothing more"


def test_rendering_substitutes_only_the_four_placeholders() -> None:
    procedure = readme.parse_readme(README_TEXT)

    rendered = readme.render_deploy_command(procedure.deploy_block, **VALUES)

    assert readme.shell_words(rendered) == [
        "gpu-fault-admin",
        "deploy",
        "--cpu-cluster-arn",
        CPU_ARN,
        "--gpu-cluster-arn",
        GPU_ARN,
        "--state-dir",
        VALUES["state_dir"],
        "--admin-email",
        ADMIN_EMAIL,
    ], "the run's values replace the placeholders; no flag is added"
    assert not any(placeholder in rendered for placeholder in readme.PLACEHOLDERS), (
        "every placeholder is substituted"
    )
    assert "--wait-for-email-confirmation" not in rendered, (
        "the README command has no wait; the stage must not invent one"
    )
    assert rendered.count("\n") == procedure.deploy_block.count("\n"), (
        "the command keeps the README's line shape"
    )


def _drop_heading(heading: str) -> Callable[[str], str]:
    return lambda text: text.replace(heading + "\n", "## 其他\n")


def _replace(old: str, new: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert old in text, old
        return text.replace(old, new, 1)

    return mutate


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_drop_heading(readme.BUILD_HEADING), "lacks the heading '## 构建和验证'"),
        (
            _drop_heading(readme.DEPLOY_HEADING),
            "lacks the heading '## 部署：先区分角色'",
        ),
        (
            _replace("  --admin-email <operations-email>\n", "  --admin-email ops\n"),
            "'<operations-email>' exactly once",
        ),
        (
            _replace(
                "  --state-dir /secure/gpu-fault \\\n",
                "  --state-dir /secure/gpu-fault --backup /secure/gpu-fault \\\n",
            ),
            "'/secure/gpu-fault' exactly once",
        ),
        (
            _replace(
                "```bash\ngpu-fault-admin deploy \\\n  --cpu-cluster-arn <cpu-eks",
                "```bash\ngpu-fault-admin status \\\n  --cpu-cluster-arn <cpu-eks",
            ),
            "must start with 'gpu-fault-admin deploy'",
        ),
        (
            _replace(". .venv/bin/activate\n", ""),
            "must run 'make deploy-host-setup-online' and then",
        ),
        (
            _replace(
                "make deploy-host-setup-online\n. .venv/bin/activate\n",
                ". .venv/bin/activate\nmake deploy-host-setup-online\n",
            ),
            "must run 'make deploy-host-setup-online' and then",
        ),
        (
            _replace(
                "## 构建和验证\n\n要求 Python 3.12。\n\n```bash\n",
                "## 构建和验证\n\n要求 Python 3.12。\n\n## 构建细节\n\n```bash\n",
            ),
            "no ```bash block under '## 构建和验证'",
        ),
    ],
)
def test_readme_that_no_longer_states_the_procedure_is_refused(
    mutate: Callable[[str], str], message: str
) -> None:
    with pytest.raises(readme.ReadmeProcedureError, match=message):
        readme.parse_readme(mutate(README_TEXT))


def test_source_activate_spelling_is_the_same_procedure() -> None:
    text = README_TEXT.replace(". .venv/bin/activate\n", "source .venv/bin/activate\n")

    procedure = readme.parse_readme(text)

    assert procedure.executed_lines[1] == "source .venv/bin/activate"


@pytest.mark.parametrize(
    "line,verification",
    [
        ("make PYTHON=.venv/bin/python check", True),
        ("make check", True),
        ("make mypy-check", True),
        ("make architecture-check", True),
        ("make deploy-host-setup-online", False),
        (". .venv/bin/activate", False),
        ("make test-parallel", False),
        ("echo check", False),
    ],
)
def test_only_make_check_lines_are_verification(line: str, verification: bool) -> None:
    assert readme.is_verification_line(line) is verification, line


def test_clean_environment_is_an_allowlist_with_the_operator_path() -> None:
    parent = {
        "HOME": "/home/operator",
        "USER": "operator",
        "LOGNAME": "operator",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
        "TERM": "xterm",
        "SSH_AUTH_SOCK": "/run/agent",
        "DOCKER_HOST": "unix:///run/docker.sock",
        "AWS_PROFILE": "site",
        "AWS_SHARED_CREDENTIALS_FILE": "/home/operator/.aws/credentials",
        "AWS_CONFIG_FILE": "/home/operator/.aws/config",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/x",
        "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token",
        "PATH": "/home/operator/.venv/bin:/home/operator/.local/bin:/usr/bin",
        "PYTHONPATH": "/home/operator/checkout/src",
        "VIRTUAL_ENV": "/home/operator/.venv",
        "AWS_REGION": "us-west-2",
        "AWS_DEFAULT_REGION": "us-west-2",
        "AWS_BEARER_TOKEN_BEDROCK": "secret",
        "GPU_FAULT_REPOSITORY_ROOT": "/home/operator/checkout",
        "GPU_FAULT_DEPLOY_API_BUDGET_DIR": "/tmp/budget",
        "BUILDKIT_PROGRESS": "plain",
        "COSIGN_PASSWORD": "secret",
        "KUBECONFIG": "/home/operator/.kube/other",
    }

    environment = readme.clean_environment(parent)

    assert environment == {
        "HOME": "/home/operator",
        "USER": "operator",
        "LOGNAME": "operator",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
        "TERM": "xterm",
        "SSH_AUTH_SOCK": "/run/agent",
        "DOCKER_HOST": "unix:///run/docker.sock",
        "AWS_PROFILE": "site",
        "AWS_SHARED_CREDENTIALS_FILE": "/home/operator/.aws/credentials",
        "AWS_CONFIG_FILE": "/home/operator/.aws/config",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/x",
        "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token",
        "PATH": "/home/operator/.local/bin:/usr/bin",
    }, "only the allowlisted names survive; PATH loses its venv entry only"
    with pytest.raises(readme.ReadmeProcedureError, match="no entry left"):
        readme.clean_environment({})


def test_operator_path_keeps_tool_directories_and_drops_venvs_and_admins(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snap/bin"
    local = tmp_path / ".local/bin"
    venv = tmp_path / "checkout/.venv/bin"
    by_config = tmp_path / "envs/py312/bin"
    deployer = tmp_path / "state/deployer-venv/bin"
    versions = tmp_path / "state/.deployer-venv.versions/online-1/bin"
    admin_dir = tmp_path / "tools/bin"
    plain = tmp_path / "opt/bin"
    for directory in (
        snap,
        local,
        venv,
        by_config,
        deployer,
        versions,
        admin_dir,
        plain,
    ):
        directory.mkdir(parents=True)
    (by_config.parent / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    admin = admin_dir / "gpu-fault-admin"
    admin.write_text("#!/bin/sh\n", encoding="utf-8")
    admin.chmod(0o755)
    (plain / "gpu-fault-admin.txt").write_text("not a tool\n", encoding="utf-8")
    parent = os.pathsep.join(
        [
            str(venv),
            str(snap),
            str(local),
            "",
            str(by_config),
            str(admin_dir),
            str(snap),
            str(deployer),
            str(versions),
            str(plain),
            "/usr/bin",
        ]
    )

    kept = readme.operator_path(parent)

    assert kept.split(os.pathsep) == [str(snap), str(local), str(plain), "/usr/bin"], (
        "snap and user tool directories survive in order, once; venvs and any"
        " directory offering a gpu-fault-admin are dropped"
    )
    with pytest.raises(readme.ReadmeProcedureError, match="no entry left"):
        readme.operator_path(f"{venv}{os.pathsep}{admin_dir}")


def test_pristine_copy_carries_the_dirty_checkout_but_not_its_venv(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    head = _git(repository, "rev-parse", "HEAD")
    (repository / ".venv/bin").mkdir(parents=True)
    (repository / ".venv/bin/activate").write_text("stale\n", encoding="utf-8")
    (repository / "README.md").write_text(README_TEXT + "\nlocal edit\n")
    (repository / "untracked.txt").write_text("carried\n", encoding="utf-8")
    (repository / "dist").mkdir()
    (repository / "dist/current-release.json").write_text("{}", encoding="utf-8")

    checkout = readme.pristine_source(repository, tmp_path / "work")

    copy = checkout.repository_root
    assert copy != repository and copy.is_relative_to(tmp_path / "work")
    assert not (copy / ".venv").exists(), "ignored files are left behind"
    assert not (copy / "dist").exists(), "ignored files are left behind"
    assert (copy / "README.md").read_text(encoding="utf-8").endswith("local edit\n"), (
        "the uncommitted edit is carried"
    )
    assert (copy / "untracked.txt").read_text(encoding="utf-8") == "carried\n"
    assert _git(copy, "status", "--porcelain") == "", "the copy is committed clean"
    assert checkout.snapshot is True and checkout.git_commit != head, (
        "a dirty checkout is snapshotted as its own commit"
    )
    assert _git(copy, "rev-parse", "HEAD") == checkout.git_commit
    assert _git(repository, "status", "--porcelain") != "", (
        "the checkout itself is not touched"
    )


def test_clean_checkout_copy_is_its_head_commit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    checkout = readme.pristine_source(repository, tmp_path / "work")

    assert checkout.snapshot is False
    assert checkout.git_commit == _git(repository, "rev-parse", "HEAD")


def test_copy_that_already_holds_a_venv_is_refused(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / ".venv/bin").mkdir(parents=True)
    (repository / ".venv/bin/activate").write_text("tracked\n", encoding="utf-8")
    _git(repository, "add", "-f", ".venv/bin/activate")
    _git(repository, "commit", "-q", "-m", "venv committed by mistake")

    with pytest.raises(readme.ReadmeProcedureError, match="already holds .venv"):
        readme.pristine_source(repository, tmp_path / "work")


def test_stage_script_runs_the_readme_lines_in_order_from_the_copy() -> None:
    procedure = readme.parse_readme(README_TEXT)
    deploy = readme.render_deploy_command(procedure.deploy_block, **VALUES)

    script = readme.stage_script(
        Path("/work/attempt-1/repository x"), procedure.executed_lines, deploy
    )

    lines = [line for line in script.splitlines() if not line.startswith("#")]
    assert lines == [
        "set -euo pipefail",
        "cd -- '/work/attempt-1/repository x'",
        "make deploy-host-setup-online",
        ". .venv/bin/activate",
        *deploy.splitlines(),
    ], "the script is the README procedure and nothing else"
    assert "check" not in script.replace("set -euo pipefail", ""), (
        "verification lines are not executed"
    )


def test_receipt_holds_digests_only(tmp_path: Path) -> None:
    procedure = readme.parse_readme(README_TEXT)
    redacted = readme.redacted_deploy_command(procedure.deploy_block, **VALUES)
    checkout = SourceCheckout(
        repository_root=tmp_path / "work/attempt-1/repository-1",
        git_commit="c" * 40,
        fingerprint="f" * 64,
        snapshot=True,
        isolated=True,
    )

    receipt = readme.readme_receipt(
        procedure,
        redacted_command=redacted,
        script="#!/usr/bin/env bash\n",
        checkout=checkout,
        base_commit="b" * 40,
        environment=readme.clean_environment(
            {"HOME": "/home/operator", "PATH": "/usr/bin:/home/operator/.venv/bin"}
        ),
        parent_path="/usr/bin:/home/operator/.venv/bin",
        attempt=1,
    )

    entry = receipt["readme"]
    assert entry["readme_sha256"] == _sha256(README_TEXT)
    assert entry["build_block_sha256"] == _sha256(procedure.build_block)
    assert entry["deploy_block_sha256"] == _sha256(procedure.deploy_block)
    assert entry["executed_lines"] == [
        "make deploy-host-setup-online",
        ". .venv/bin/activate",
        redacted,
    ]
    assert entry["skipped_verification_lines"] == ["make PYTHON=.venv/bin/python check"]
    assert entry["venv_preexisted"] is False
    assert entry["source_commit"] == "c" * 40
    assert entry["source_base_commit"] == "b" * 40
    assert entry["source_snapshot"] is True
    assert entry["environment_names"] == ["HOME", "PATH"]
    assert entry["path_sha256"] == _sha256("/usr/bin")
    assert entry["path_entries_kept"] == 1 and entry["path_entries_dropped"] == 1
    assert "/usr/bin" not in json.dumps(entry), "PATH entries appear as a digest only"
    assert f"sha256:{_sha256(CPU_ARN)}" in redacted
    assert f"sha256:{_sha256(VALUES['state_dir'])}" in redacted
    serialized = json.dumps(receipt)
    for secret in (CPU_ARN, GPU_ARN, ADMIN_EMAIL, VALUES["state_dir"], str(tmp_path)):
        assert secret not in serialized, f"{secret} leaked into the receipt"
    assert "/home/operator" not in serialized, "environment values stay private"


FIXTURE_SETUP = """#!/usr/bin/env bash
set -euo pipefail
mkdir -p .venv/bin
printf 'export VIRTUAL_ENV=%q\\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\\n' "$PWD/.venv" \\
  > .venv/bin/activate
cat > .venv/bin/gpu-fault-admin <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
import json, os, sys
json.dump({"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()},
          open({dump}, "w"))
PY
exit {exit_code}
WRAPPER
chmod 755 .venv/bin/gpu-fault-admin
"""


def _procedure_repository(tmp_path: Path, *, exit_code: int) -> Path:
    repository = _repository(tmp_path)
    (repository / "Makefile").write_text(
        ".PHONY: deploy-host-setup-online\n"
        "deploy-host-setup-online:\n\tbash fixture-setup.sh\n",
        encoding="utf-8",
    )
    (repository / "fixture-setup.sh").write_text(
        FIXTURE_SETUP.replace("{dump}", repr(str(tmp_path / "admin.json"))).replace(
            "{exit_code}", str(exit_code)
        ),
        encoding="utf-8",
    )
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "fixture make target")
    return repository


@pytest.mark.parametrize("exit_code", [0, 7])
def test_run_executes_the_readme_procedure_in_a_clean_environment(
    tmp_path: Path, exit_code: int
) -> None:
    repository = _procedure_repository(tmp_path, exit_code=exit_code)
    receipt = tmp_path / "stage.extra.json"
    parent = {
        **os.environ,
        "GPU_FAULT_SENTINEL": "driver",
        "PYTHONPATH": "/driver/src",
        "AWS_REGION": "us-west-2",
        "HOME": str(tmp_path),
    }

    code = readme.run(
        readme=repository / "README.md",
        repository=repository,
        work_dir=tmp_path / "work",
        attempt=1,
        cpu_arn=CPU_ARN,
        gpu_arn=GPU_ARN,
        state_dir=tmp_path / "second",
        admin_email=ADMIN_EMAIL,
        receipt=receipt,
        parent_environment=parent,
    )

    assert code == exit_code, "the deploy's exit status is the stage's"
    dump = json.loads((tmp_path / "admin.json").read_text(encoding="utf-8"))
    assert dump["argv"] == [
        "deploy",
        "--cpu-cluster-arn",
        CPU_ARN,
        "--gpu-cluster-arn",
        GPU_ARN,
        "--state-dir",
        str(tmp_path / "second"),
        "--admin-email",
        ADMIN_EMAIL,
    ], "the venv's gpu-fault-admin receives the rendered README command"
    copy = Path(dump["cwd"])
    assert copy.is_relative_to(tmp_path / "work/attempt-1") and copy != repository
    environment = dump["env"]
    operator = readme.operator_path(parent["PATH"])
    assert environment["PATH"] == f"{copy}/.venv/bin:{operator}", (
        "only the README's activation extends the operator's filtered PATH"
    )
    assert environment["VIRTUAL_ENV"] == f"{copy}/.venv"
    incidental = {"PATH", "VIRTUAL_ENV", "PWD", "OLDPWD", "SHLVL", "_"}
    for name in set(environment) - incidental:
        assert name in readme.KEPT_NAMES or name.startswith(readme.KEPT_PREFIXES), (
            f"{name} leaked from the driver into the README deploy"
        )
    assert "GPU_FAULT_SENTINEL" not in environment and "PYTHONPATH" not in environment
    assert "AWS_REGION" not in environment, "the README derives the Region from ARNs"
    lines = receipt.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "one receipt line per attempt, written before the run"
    entry = json.loads(lines[0])["readme"]
    assert entry["attempt"] == 1 and entry["venv_preexisted"] is False
    assert entry["source_commit"] == _git(repository, "rev-parse", "HEAD")
    script = tmp_path / "work/attempt-1" / readme.SCRIPT_NAME
    assert script.stat().st_mode & 0o777 == 0o700
    assert entry["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert shlex.quote(str(copy)) in script.read_text(encoding="utf-8")


def test_run_refuses_before_touching_anything_when_the_readme_is_defective(
    tmp_path: Path,
) -> None:
    repository = _procedure_repository(tmp_path, exit_code=0)
    (repository / "README.md").write_text(
        README_TEXT.replace(readme.DEPLOY_HEADING + "\n", "## 部署\n"), encoding="utf-8"
    )
    receipt = tmp_path / "stage.extra.json"

    with pytest.raises(readme.ReadmeProcedureError, match="lacks the heading"):
        readme.run(
            readme=repository / "README.md",
            repository=repository,
            work_dir=tmp_path / "work",
            attempt=1,
            cpu_arn=CPU_ARN,
            gpu_arn=GPU_ARN,
            state_dir=tmp_path / "second",
            admin_email=ADMIN_EMAIL,
            receipt=receipt,
            parent_environment={},
        )

    assert not (tmp_path / "work").exists(), "no pristine copy is made"
    assert not receipt.exists() and not (tmp_path / "admin.json").exists()
