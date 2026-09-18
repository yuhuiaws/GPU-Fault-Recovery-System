from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault import training_submit_cli as submit
from gpu_fault import workload_annotate_cli as annotate
from tests.hyperpod._cov95_submit_support import manifest_file


@pytest.mark.parametrize("error", [OSError("unreadable"), ValueError("bad Profile")])
def test_site_read_failure_is_reported_without_submission(
    tmp_path, monkeypatch, error
) -> None:
    calls = []

    def load_site(path):
        calls.append(path)
        raise error

    monkeypatch.setattr(submit, "load_site", load_site)
    monkeypatch.setenv(submit.SITE_ENV, str(tmp_path / "site.yaml"))
    args = argparse.Namespace(site=None, runtime_profile_version=None)
    with pytest.raises(submit.TrainingSubmitError, match="cannot load site Profile"):
        submit.resolve_runtime_profile_version(args)
    assert calls == [tmp_path / "site.yaml"]


def test_site_and_explicit_profile_agree_and_override_legacy_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_RUNTIME_PROFILE", "legacy")
    monkeypatch.setattr(
        submit,
        "load_site",
        lambda path: SimpleNamespace(
            release_config={"runtime_profile": {"version": "profile-current"}}
        ),
    )
    args = argparse.Namespace(
        site=tmp_path / "site.yaml", runtime_profile_version=" profile-current "
    )
    assert submit.resolve_runtime_profile_version(args) == "profile-current"
    args.site = None
    args.runtime_profile_version = None
    assert submit.resolve_runtime_profile_version(args) == "legacy"
    monkeypatch.delenv("GPU_FAULT_RUNTIME_PROFILE")
    assert submit.resolve_runtime_profile_version(args) == "hyperpod-v1"


@pytest.mark.parametrize(
    "options", [[], ["--kubeconfig", "fake", "--context", "local"]]
)
def test_submit_passes_exact_manifest_to_fake_transport(tmp_path, capsys, options):
    path = manifest_file(tmp_path)
    args = submit.parser().parse_args(
        [str(path), "--job-id", "training", "--attempt-id", "attempt", *options]
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 23)

    assert submit.run(args, runner=runner) == 23
    command, kwargs = calls[0]
    assert command == ["kubectl", *options, "apply", "-f", "-"]
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    rendered = yaml.safe_load(kwargs["input"])
    assert rendered["metadata"]["labels"][submit.JOB_LABEL] == "training"
    assert rendered["spec"]["template"]["metadata"]["labels"][submit.ATTEMPT_LABEL] == (
        "attempt"
    )
    assert capsys.readouterr().out == ""


def test_transport_start_failure_is_a_cli_error(tmp_path) -> None:
    args = submit.parser().parse_args([str(manifest_file(tmp_path))])

    def runner(command, **kwargs):
        raise OSError("fake binary unavailable")

    with pytest.raises(submit.TrainingSubmitError, match="cannot execute kubectl"):
        submit.run(args, runner=runner)


@pytest.mark.parametrize("module", [submit, annotate])
@pytest.mark.parametrize("valid", [True, False])
def test_main_exit_status_and_user_diagnostic(
    tmp_path, monkeypatch, capsys, module, valid
):
    path = manifest_file(tmp_path)
    options = ["--dry-run"] if module is submit else []
    if not valid:
        options.extend(["--attempt-number", "0"])
    monkeypatch.setattr(
        sys, "argv", [module.__name__, str(path), "--job-id", "training", *options]
    )
    with pytest.raises(SystemExit) as exited:
        module.main()
    if valid:
        assert exited.value.code == 0
        assert yaml.safe_load(capsys.readouterr().out)["kind"] == "Job"
    else:
        assert str(exited.value.code).endswith("--attempt-number must be positive"), (
            "invalid attempts need the actionable CLI validation diagnostic"
        )


def test_annotation_output_failure_is_reported_without_partial_success(
    tmp_path, capsys
) -> None:
    args = annotate.parser().parse_args(
        [str(manifest_file(tmp_path)), "--output", str(tmp_path)]
    )
    with pytest.raises(submit.TrainingSubmitError, match="cannot write manifest"):
        annotate.run(args)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "module_name", ["gpu_fault.training_submit_cli", "gpu_fault.workload_annotate_cli"]
)
def test_module_entrypoint_help_never_contacts_a_cluster(
    monkeypatch, capsys, module_name
):
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])
    monkeypatch.delitem(sys.modules, module_name)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(module_name, run_name="__main__")
    assert exited.value.code == 0
    assert "--runtime-profile-version" in capsys.readouterr().out
