from __future__ import annotations

import json
import subprocess
import sys
import traceback
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin.execution import deadline_scope
from tests.regional.test_installed_resource_registry import MODULE


class CredentialRunner:
    def __init__(self) -> None:
        self.clock = SimpleNamespace(wall=1_800_000_000.0, monotonic=10_000.0)
        self.expires_in: float | None = 900
        self.calls: list[tuple[list[str], dict]] = []
        self.tokens = 0
        self.requests = 0
        self.session_paths: list[Path] = []
        self.fail_at = ""
        self.failure: Exception | None = None
        self.output_override: str | None = None
        self.marker = "opaque-credential-fixture-without-a-keyword"
        self.config = {
            "apiVersion": "v1",
            "kind": "Config",
            "current-context": "fixture",
            "clusters": [
                {
                    "name": "fixture",
                    "cluster": {
                        "server": "https://example.invalid",
                        "certificate-authority-data": "Zml4dHVyZQ==",
                    },
                }
            ],
            "contexts": [
                {
                    "name": "fixture",
                    "context": {"cluster": "fixture", "user": "fixture"},
                }
            ],
            "users": [
                {
                    "name": "fixture",
                    "user": {
                        "exec": {
                            "apiVersion": "client.authentication.k8s.io/v1beta1",
                            "command": "aws",
                            "args": ["--region", "fixture-region", "eks", "get-token"],
                            "env": [
                                {"name": "FIXTURE_EXEC_ENV", "value": "only-child"}
                            ],
                        }
                    },
                }
            ],
        }

    def install_clock(self, monkeypatch) -> None:
        monkeypatch.setattr(
            MODULE,
            "time",
            SimpleNamespace(
                time=lambda: self.clock.wall, monotonic=lambda: self.clock.monotonic
            ),
        )

    def __call__(self, arguments: list[str], **kwargs):
        self.calls.append((arguments, kwargs))
        stage = (
            "config"
            if "view" in arguments
            else "credential"
            if arguments[0] == "aws"
            else "request"
        )
        if stage == self.fail_at:
            if self.failure is not None:
                raise self.failure
            return subprocess.CompletedProcess(arguments, 1, self.marker, self.marker)
        if stage == "config":
            assert "--raw" in arguments and "--minify" in arguments
            assert "--flatten" in arguments
            output = json.dumps(self.config)
        elif stage == "credential":
            self.tokens += 1
            assert kwargs["environment"]["FIXTURE_EXEC_ENV"] == "only-child"
            status = {"token": f"opaque-token-fixture-{self.tokens}"}
            if self.expires_in is not None:
                status["expirationTimestamp"] = datetime.fromtimestamp(
                    self.clock.wall + self.expires_in, timezone.utc
                ).isoformat()
            output = (
                self.output_override
                if self.output_override is not None
                else json.dumps({"status": status})
            )
        else:
            self.requests += 1
            path = Path(arguments[arguments.index("--kubeconfig") + 1])
            self.session_paths.append(path)
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
            document = json.loads(path.read_text())
            assert document["users"][0]["user"] == {
                "token": f"opaque-token-fixture-{self.tokens}"
            }
            assert document["clusters"] == self.config["clusters"]
            assert document["contexts"] == self.config["contexts"]
            output = json.dumps({"items": []})
        return subprocess.CompletedProcess(arguments, 0, output, "")


def client(runner: CredentialRunner):
    return MODULE.Kubectl(
        kubeconfig=None, context="fixture", reuse_exec_credential=True, runner=runner
    )


def test_cached_credentials_refresh_before_expiry_and_survive_clock_rollback(
    monkeypatch,
) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    with closing(client(runner)) as kubectl:
        kubectl.run(["get", "deployments"])
        runner.clock.monotonic += 839
        kubectl.run(["get", "cronjobs"])
        assert runner.tokens == 1
        runner.clock.monotonic += 2
        runner.clock.wall -= 3600
        kubectl.run(["get", "roles"])
        assert runner.tokens == 2
        assert runner.requests == 3
        assert len(set(runner.session_paths)) == 1
        assert all(
            "opaque-token" not in " ".join(arguments)
            for arguments, _kwargs in runner.calls
        ), "cached credentials must not appear in command arguments"
    assert all(not path.exists() for path in runner.session_paths), (
        "closing kubectl must remove credential session files"
    )


def test_exec_credentials_without_an_expiry_are_not_cached(monkeypatch) -> None:
    runner = CredentialRunner()
    runner.expires_in = None
    runner.install_clock(monkeypatch)
    with closing(client(runner)) as kubectl:
        kubectl.run(["get", "deployments"])
        kubectl.run(["get", "roles"])
    assert runner.tokens == 2
    assert runner.requests == 2


def test_cached_credentials_refresh_after_a_forward_wall_clock_jump(
    monkeypatch,
) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    with closing(client(runner)) as kubectl:
        kubectl.run(["get", "deployments"])
        runner.clock.wall += 1000
        kubectl.run(["get", "roles"])
    assert runner.tokens == 2
    assert runner.requests == 2


def test_failed_refresh_never_uses_the_previous_token(monkeypatch) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    with closing(client(runner)) as kubectl:
        kubectl.run(["get", "deployments"])
        runner.clock.monotonic += 850
        runner.fail_at = "credential"
        with pytest.raises(MODULE.RegistryError, match="credential failed"):
            kubectl.run(["apply", "-f", "-"], input_text="{}")
        assert runner.requests == 1
    assert all(not path.exists() for path in runner.session_paths), (
        "failed refresh must not leave credential session files"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"provideClusterInfo": True},
        {"interactiveMode": "Always"},
        {"apiVersion": "client.authentication.k8s.io/unsupported"},
    ],
)
def test_exec_protocol_options_are_not_bypassed_by_the_eks_cache(options) -> None:
    config = CredentialRunner().config
    config["users"][0]["user"]["exec"].update(options)
    calls = []

    def execute(arguments, **_kwargs):
        calls.append(arguments)
        assert arguments[:3] == ["kubectl", "--context", "fixture"]
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps(config) if "view" in arguments else "{}", ""
        )

    with closing(
        MODULE.Kubectl(
            kubeconfig=None,
            context="fixture",
            reuse_exec_credential=True,
            runner=execute,
        )
    ) as kubectl:
        kubectl.run(["get", "deployment"])
        assert kubectl.session_directory is None
    assert len(calls) == 2
    assert calls[-1][-2:] == ["get", "deployment"]


@pytest.mark.parametrize("expires_in", [-1, 0, 15, 30])
def test_expired_or_too_short_credentials_stop_before_a_request(
    monkeypatch, expires_in
) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    runner.expires_in = expires_in
    with closing(client(runner)) as kubectl:
        with pytest.raises(MODULE.RegistryError, match="expires too soon"):
            kubectl.run(["apply", "-f", "-"], input_text="{}")
    assert runner.requests == 0


@pytest.mark.parametrize(
    "output",
    [
        "opaque-credential-fixture",
        "[]",
        json.dumps({"status": "opaque-credential-fixture"}),
        json.dumps({"status": {"token": ["opaque-credential-fixture"]}}),
        json.dumps({"status": {"token": "fixture", "expirationTimestamp": "invalid"}}),
        json.dumps({"status": {"token": "fixture", "expirationTimestamp": 123}}),
        json.dumps(
            {
                "status": {
                    "token": "fixture",
                    "expirationTimestamp": "2027-01-01T12:00:00",
                }
            }
        ),
    ],
)
def test_invalid_credential_responses_fail_closed_without_echoing_output(
    monkeypatch, output
) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    runner.output_override = output
    with closing(client(runner)) as kubectl:
        with pytest.raises(MODULE.RegistryError) as error:
            kubectl.run(["get", "deployments"])
    assert runner.requests == 0
    assert "opaque-credential-fixture" not in "".join(
        traceback.format_exception(error.value)
    )


@pytest.mark.parametrize("stage", ["config", "credential", "request"])
@pytest.mark.parametrize(
    "failure_kind", ["exit", "timeout", "oserror", "called-process"]
)
def test_command_errors_never_echo_sensitive_output_or_arguments(
    monkeypatch, capsys, stage: str, failure_kind: str
) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    runner.fail_at = stage
    marker = runner.marker
    if failure_kind == "timeout":
        runner.failure = subprocess.TimeoutExpired(
            [marker], 1, output=marker, stderr=marker
        )
    elif failure_kind == "oserror":
        runner.failure = OSError(marker)
    elif failure_kind == "called-process":
        runner.failure = subprocess.CalledProcessError(
            1, [marker], output=marker, stderr=marker
        )
    with pytest.raises(MODULE.RegistryError) as error:
        with closing(client(runner)) as kubectl:
            kubectl.run(["get", "deployments"])
    assert marker not in "".join(traceback.format_exception(error.value))
    captured = capsys.readouterr()
    assert marker not in captured.out + captured.err
    assert all(not path.exists() for path in runner.session_paths), (
        "command errors must not leave credential session files"
    )


def test_config_refresh_and_probe_share_the_parent_deadline(monkeypatch) -> None:
    runner = CredentialRunner()
    runner.install_clock(monkeypatch)
    with deadline_scope("registry test parent", 5), closing(client(runner)) as kubectl:
        kubectl.probe_output(["get", "configmap", "fixture"], timeout_seconds=3)
    assert len(runner.calls) == 3
    assert 0 < runner.calls[0][1]["timeout_seconds"] <= 5
    assert all(
        0 < kwargs["timeout_seconds"] <= 3 for _args, kwargs in runner.calls[1:]
    ), "credential refresh and probe must honor the requested timeout"
    assert all(kwargs["capture"] is True for _args, kwargs in runner.calls), (
        "registry commands must capture output privately"
    )


def test_default_kubectl_runner_uses_supervised_execution(monkeypatch) -> None:
    calls = []

    def execute(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "{}", "")

    monkeypatch.setattr(MODULE, "run_command", execute)
    kubectl = MODULE.Kubectl(kubeconfig=None, context="fixture")
    kubectl.run(["get", "deployments"])
    assert calls[0][0] == ["kubectl", "--context", "fixture", "get", "deployments"]
    assert 0 < calls[0][1]["timeout_seconds"] <= MODULE.COMMAND_TIMEOUT_SECONDS


def test_real_local_hung_command_is_bounded_and_its_output_is_private(capsys) -> None:
    kubectl = MODULE.Kubectl(kubeconfig=None, context="fixture")
    kubectl.prefix = [
        sys.executable,
        "-c",
        (
            "import sys,time; "
            "print('opaque-command-fixture',flush=True); "
            "print('opaque-command-fixture',file=sys.stderr,flush=True); "
            "time.sleep(30)"
        ),
    ]
    with pytest.raises(MODULE.RegistryError, match="deadline") as error:
        kubectl.run([], timeout_seconds=0.2)
    assert "opaque-command-fixture" not in "".join(
        traceback.format_exception(error.value)
    )
    captured = capsys.readouterr()
    assert "opaque-command-fixture" not in captured.out + captured.err
