"""The public deploy's hop into the source preparer, and the command line it
hands down: ``run_source_deploy`` starts ``scripts/staging_deploy.py`` with the
resolved inputs, and the operator's own ``gpu-fault-admin deploy`` line goes
with them for the first-minute email stop to print."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import source_deploy as admin_source_deploy
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin._first_deploy_discovery import stub_first_deploy_discovery
from tests.admin.test_admin_site import site_file


def repository(path: Path) -> Path:
    path.mkdir()
    (path / "deploy").mkdir()
    scripts = path / "scripts"
    scripts.mkdir()
    (scripts / "staging_deploy.py").write_text("", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    return path


def test_first_deploy_uses_current_repository(tmp_path: Path) -> None:
    source = repository(tmp_path / "repo")

    resolved = admin_source_deploy.resolve_source_repository(
        tmp_path / "state", current_directory=source
    )

    assert resolved == source


def test_recorded_repository_is_reused_outside_checkout(tmp_path: Path) -> None:
    source = repository(tmp_path / "repo")
    state = tmp_path / "state"
    state.mkdir()
    (state / admin_source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps({"schema_version": 1, "source_repository_root": str(source)}),
        encoding="utf-8",
    )

    resolved = admin_source_deploy.resolve_source_repository(
        state, current_directory=tmp_path
    )

    assert resolved == source


def test_current_checkout_wins_over_a_recorded_one_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A first deploy resumed from a corrected checkout, or an upgrade run from a
    fresh clone, must build the checkout the operator is standing in, not the
    directory an earlier attempt recorded."""

    earlier = repository(tmp_path / "attempt-1")
    current = repository(tmp_path / "attempt-2")
    state = tmp_path / "state"
    state.mkdir()
    (state / admin_source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps({"schema_version": 1, "source_repository_root": str(earlier)}),
        encoding="utf-8",
    )

    resolved = admin_source_deploy.resolve_source_repository(
        state, current_directory=current
    )

    assert resolved == current, "the operator's checkout is the source"
    notice = capsys.readouterr().err
    assert str(current) in notice and str(earlier) in notice, (
        "the switch away from the recorded source is announced"
    )
    assert (
        admin_source_deploy.resolve_source_repository(state, current_directory=earlier)
        == earlier
        and capsys.readouterr().err == ""
    ), "no notice when the current directory is the recorded one"


def test_invalid_recorded_repository_fails_closed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / admin_source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps(
            {"schema_version": 1, "source_repository_root": str(tmp_path / "missing")}
        ),
        encoding="utf-8",
    )

    with pytest.raises(BootstrapError, match="recorded source repository"):
        admin_source_deploy.resolve_source_repository(state, current_directory=tmp_path)


def test_source_deploy_hides_release_and_artifact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository(tmp_path / "repo")
    commands: list[tuple[list[str], dict[str, object]]] = []

    def run(command, **kwargs):
        commands.append(([str(item) for item in command], kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(admin_source_deploy, "run_driver", run)

    result = admin_source_deploy.run_source_deploy(
        cpu_cluster_arn="cpu",
        gpu_cluster_arns=("gpu-a", "gpu-b"),
        state_dir=tmp_path / "state",
        admin_email="operations@example.com",
        impact_base="origin/main",
        current_directory=source,
    )

    assert result == 0
    command, options = commands[0]
    assert command[1].endswith("scripts/staging_deploy.py"), (
        "public deploy did not invoke the private source preparer"
    )
    assert command.count("--gpu-cluster-arn") == 2
    assert "--quiet" in command
    assert "--release-ref" not in command
    assert "--site" not in command
    assert "--prebuilt-attestation" not in command
    assert "--profile-approval" not in command
    assert options["cwd"] == source
    assert options["env"]["PYTHONPATH"] == str(source / "src")


def test_source_deploy_hands_the_rerun_line_to_the_source_preparer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The email stop happens in the source preparer, which only sees the
    resolved inputs; the operator's own command line travels with them."""

    source = repository(tmp_path / "repo")
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(admin_source_deploy, "run_driver", run)
    line = "gpu-fault-admin deploy --state-dir /secure/x --accept-schema-change"

    for rerun_command in (line, None):
        admin_source_deploy.run_source_deploy(
            cpu_cluster_arn="cpu",
            gpu_cluster_arns=("gpu-a",),
            state_dir=tmp_path / "state",
            admin_email="operations@example.com",
            impact_base="origin/main",
            current_directory=source,
            rerun_command=rerun_command,
        )

    with_line, without_line = commands
    assert with_line[with_line.index("--rerun-command") + 1] == line, (
        "the line must reach the preparer as one argument, unchanged"
    )
    assert "--rerun-command" not in without_line, (
        "without a line the preparer builds its own, as a direct run does"
    )


# --- the operator's command line, for the email stop to print ----------------


def _managed_site(tmp_path: Path) -> Path:
    """A state directory holding a loadable managed ``site.yaml``."""

    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    first = document["spec"]["clusters"][0]
    document["spec"]["clusters"] = [
        {
            **first,
            "clusterId": "gpu-a",
            "context": "gpu-a",
            "hyperpodClusterName": "hp-gpu-a",
            "eksClusterArn": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        }
    ]
    document["spec"]["notifications"] = {"adminEmail": "operations@example.com"}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return tmp_path


def _rerun_namespace(line: str) -> argparse.Namespace:
    words = shlex.split(line)
    assert words[:2] == ["gpu-fault-admin", "deploy"], line
    return admin_cli.parser().parse_args(words[1:])


def test_the_rerun_line_repeats_the_whole_deploy_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop in the first minute (the email confirmation) prints the command to
    rerun. It was ``deploy --state-dir X`` whatever was typed, so the rerun
    silently dropped a joining cluster, a new administrator address, a consent
    and the Grafana options. The line handed down now parses to the same
    command."""

    state_dir = _managed_site(tmp_path)
    calls: list[dict] = []
    # The source deploy stopped at the email check: no join follows.
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 2
    )
    monkeypatch.setattr(
        admin_cli,
        "approve_profile_plan_inline",
        lambda *_a, **_k: {
            "plan_sha256": "a" * 64,
            "approved_at": "now",
            "approver_identity": "operator",
            "reference": "CHG-12345",
        },
    )
    typed = [
        "deploy",
        "--state-dir",
        str(state_dir),
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        "--admin-email",
        "new-admin@example.com",
        "--approve-profile-plan",
        "a" * 64,
        "--reference",
        "CHG-12345",
        "--grafana-workspace-id",
        "g-1234567890",
        "--grafana-viewer",
        "sso-user-1",
        "--accept-schema-change",
        "--supersede-failed-transaction",
        "--allow-inflight-installs",
        "--wait-for-email-confirmation",
        "5",
    ]
    original = admin_cli.parser().parse_args(typed)

    assert admin_cli.run(original) == 2
    line = str(calls[0]["rerun_command"])
    assert vars(_rerun_namespace(line)) == {
        **vars(original),
        "state_dir": state_dir.resolve(),
    }, f"the rerun line is not the command the operator gave: {line}"


def test_the_rerun_line_of_a_first_deploy_keeps_its_admin_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 2
    )
    monkeypatch.chdir(tmp_path)
    stub_first_deploy_discovery(monkeypatch)
    config = tmp_path / "admin config.yaml"
    config.write_text(
        yaml.safe_dump({"apiVersion": "gpu-fault.aws/v1alpha1", "kind": "AdminConfig"}),
        encoding="utf-8",
    )
    config.chmod(0o600)
    original = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
            "--state-dir",
            "state",
            "--admin-email",
            "operations@example.com",
            "--config",
            config.name,
        ]
    )

    assert admin_cli.run(original) == 2
    rerun = _rerun_namespace(str(calls[0]["rerun_command"]))
    assert rerun.admin_config_file == config.resolve(), (
        "a quoted, absolute --config so the rerun reads the same file anywhere"
    )
    assert rerun.state_dir == (tmp_path / "state").resolve()
    assert (rerun.cpu_cluster_arn, rerun.gpu_cluster_arn, rerun.alert_email) == (
        original.cpu_cluster_arn,
        original.gpu_cluster_arn,
        original.alert_email,
    )


def test_the_rerun_line_leaves_out_what_already_happened_and_what_was_not_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--registration-cluster`` moved the anchor, journaled, before the email
    check: a second move is refused, so its ``--reference`` goes too unless it
    belongs to ``--approve-profile-plan``. A consent inherited from the shell is
    not something the operator typed and is not added."""

    from gpu_fault.admin.deploy_command import deploy_rerun_command

    monkeypatch.setenv("GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE", "snapshot")
    moved = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(tmp_path),
            "--registration-cluster",
            "gpu-b",
            "--reference",
            "CHG-1",
        ]
    )
    both = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(tmp_path),
            "--registration-cluster",
            "gpu-b",
            "--approve-profile-plan",
            "b" * 64,
            "--reference",
            "CHG-2",
        ]
    )

    assert deploy_rerun_command(moved, tmp_path) == (
        f"gpu-fault-admin deploy --state-dir {tmp_path}"
    )
    rerun = _rerun_namespace(deploy_rerun_command(both, tmp_path))
    assert rerun.registration_cluster is None
    assert (rerun.approve_profile_plan, rerun.reference) == ("b" * 64, "CHG-2")


def test_the_real_source_preparer_command_carries_the_rerun_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real ``run_source_deploy``: only the process it starts is
    replaced, so the CLI and the hop that builds the preparer's argv agree."""

    from gpu_fault.admin import source_deploy

    state_dir = _managed_site(tmp_path / "site")
    checkout = tmp_path / "checkout"
    (checkout / "deploy").mkdir(parents=True)
    (checkout / "scripts").mkdir()
    (checkout / "scripts/staging_deploy.py").write_text("", encoding="utf-8")
    (checkout / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    monkeypatch.chdir(checkout)
    started: list[list[str]] = []

    def driver(command, **_kwargs):
        started.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(source_deploy, "run_driver", driver)
    original = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), "--accept-schema-change"]
    )

    assert admin_cli.run(original) == 2
    (command,) = started
    line = command[command.index("--rerun-command") + 1]
    assert vars(_rerun_namespace(line)) == {
        **vars(original),
        "state_dir": state_dir.resolve(),
    }


def test_every_public_deploy_option_is_repeated_or_deliberately_left_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A new option of ``gpu-fault-admin deploy`` must be decided for the rerun
    line; read from the public help, where hidden options do not appear."""

    from gpu_fault.admin import deploy_command

    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(["deploy", "--help"])
    help_text = capsys.readouterr().out
    public = set(re.findall(r"^\s{2}(?:-\w, )?(--[a-z0-9-]+)", help_text, re.M))

    assert public - {"--help"} == (
        deploy_command.RERUN_REPEATED_OPTIONS | deploy_command.RERUN_OMITTED_OPTIONS
    )
    assert (
        not deploy_command.RERUN_REPEATED_OPTIONS & deploy_command.RERUN_OMITTED_OPTIONS
    )
