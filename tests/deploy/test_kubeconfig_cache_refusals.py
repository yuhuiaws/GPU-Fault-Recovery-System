"""Fail-closed paths of the kubeconfig token cache.

Each case feeds the cache a kubeconfig or an exec-plugin answer that is almost
right and pins the refusal (or the fallback to the original file) so a bad
credential can never be published into the private copy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release import regional_kubeconfig_cache as CACHE
from gpu_fault_release.regional_release_config import ReleaseError

API_VERSION = "client.authentication.k8s.io/v1beta1"
EXPIRY = "2099-01-01T00:00:00+00:00"


def exec_user(**extra: Any) -> dict[str, Any]:
    return {
        "exec": {
            "apiVersion": API_VERSION,
            "command": "aws",
            "args": ["eks", "get-token", "--cluster-name", "cpu"],
            **extra,
        }
    }


def write_kubeconfig(path: Path, document: Any) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def kubeconfig(
    path: Path,
    *,
    user: dict[str, Any] | None = None,
    cluster: dict[str, Any] | None = None,
    users: list[Any] | None = None,
    clusters: list[Any] | None = None,
) -> Path:
    document = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": clusters
        if clusters is not None
        else [{"name": "cpu", "cluster": cluster or {"server": "https://cpu"}}],
        "contexts": [{"name": "cpu", "context": {"cluster": "cpu", "user": "cpu"}}],
        "current-context": "cpu",
        "users": users if users is not None else [{"name": "cpu", "user": user}],
    }
    return write_kubeconfig(path, document)


def credential(status: Any, *, kind: str = "ExecCredential") -> str:
    return json.dumps({"apiVersion": API_VERSION, "kind": kind, "status": status})


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake exec plugin: ``answer["stdout"]`` is what ``aws`` prints."""

    answer: dict[str, Any] = {
        "stdout": credential(
            {"token": "k8s-aws-v1.fixture", "expirationTimestamp": EXPIRY}
        ),
        "calls": [],
    }

    def run_command(arguments: list[str], **options: Any) -> CompletedProcess[str]:
        answer["calls"].append((list(arguments), options["environment"]))
        return CompletedProcess(arguments, 0, stdout=answer["stdout"], stderr="")

    monkeypatch.setattr(CACHE, "run_command", run_command)
    monkeypatch.delenv(CACHE.TOKEN_CACHE_ENV, raising=False)
    return answer


def test_unreadable_or_non_mapping_kubeconfig_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "broken.kubeconfig"
    broken.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="is not a mapping"):
        CACHE.KubeconfigTokenCache(broken, directory=tmp_path)


def test_non_mapping_user_and_cluster_entries_are_ignored(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    source = kubeconfig(
        tmp_path / "cpu.kubeconfig", users=["dangling"], clusters=["dangling"]
    )
    cache = CACHE.KubeconfigTokenCache(source, directory=tmp_path)
    assert cache.has_exec_users is False
    assert cache.refresh_if_needed() is False, "nothing to refresh without a plugin"
    assert plugin["calls"] == []


def test_user_with_more_than_an_exec_entry_keeps_native_authentication(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    user = {**exec_user(), "token": "static-fixture-token"}
    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=user)
    cache = CACHE.KubeconfigTokenCache(source, directory=tmp_path)
    assert cache.has_exec_users is False
    assert plugin["calls"] == []


def test_relative_file_references_are_anchored_at_the_source_directory(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    source = kubeconfig(
        tmp_path / "cpu.kubeconfig",
        user=exec_user(),
        cluster={"server": "https://cpu", "certificate-authority": "certs/ca.pem"},
    )
    cache = CACHE.KubeconfigTokenCache(source, directory=tmp_path)
    copied = yaml.safe_load(cache.path.read_text(encoding="utf-8"))
    assert copied["clusters"][0]["cluster"]["certificate-authority"] == str(
        (tmp_path / "certs/ca.pem").resolve()
    )
    assert copied["users"][0]["user"] == {"token": "k8s-aws-v1.fixture"}
    assert os.environ.get("KUBERNETES_EXEC_INFO") is None, "process env untouched"
    assert json.loads(plugin["calls"][0][1]["KUBERNETES_EXEC_INFO"])["spec"] == {
        "interactive": False
    }


@pytest.mark.parametrize(
    "env,message",
    [
        ("not-a-list", "environment is invalid"),
        ([{"name": "AWS_PROFILE"}], "environment is invalid"),
        ([{"name": "", "value": "x"}], "environment is invalid"),
    ],
    ids=["scalar", "no-value", "empty-name"],
)
def test_invalid_exec_environment_entries_are_refused(
    tmp_path: Path, plugin: dict[str, Any], env: Any, message: str
) -> None:
    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=exec_user(env=env))
    with pytest.raises(ReleaseError, match=message):
        CACHE.KubeconfigTokenCache(source, directory=tmp_path)
    assert plugin["calls"] == [], "an invalid env never reaches the plugin"


def test_valid_exec_environment_is_passed_to_the_plugin(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    env = [{"name": "AWS_PROFILE", "value": "fixture"}]
    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=exec_user(env=env))
    CACHE.KubeconfigTokenCache(source, directory=tmp_path)
    assert plugin["calls"][0][1]["AWS_PROFILE"] == "fixture"


@pytest.mark.parametrize(
    "stdout,message",
    [
        (
            credential({"token": "t", "expirationTimestamp": EXPIRY}, kind="Secret"),
            "identity is invalid",
        ),
        (
            json.dumps({"apiVersion": API_VERSION, "kind": "ExecCredential"}),
            "no status",
        ),
        (
            credential({"token": "t", "expirationTimestamp": EXPIRY, "extra": 1}),
            "unsupported status fields",
        ),
        (credential({"token": "", "expirationTimestamp": EXPIRY}), "no status.token"),
        (credential({"token": "t", "expirationTimestamp": 1234}), "expiry is invalid"),
        (
            credential({"token": "t", "expirationTimestamp": "2099-01-01T00:00:00"}),
            "expiry is invalid",
        ),
        ("x" * (CACHE.MAX_CREDENTIAL_OUTPUT_BYTES + 1), "exceeds its size limit"),
    ],
    ids=[
        "kind",
        "no-status",
        "extra-field",
        "empty-token",
        "numeric-expiry",
        "naive-expiry",
        "oversized",
    ],
)
def test_malformed_plugin_answers_fail_closed(
    tmp_path: Path, plugin: dict[str, Any], stdout: str, message: str
) -> None:
    plugin["stdout"] = stdout
    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=exec_user())
    with pytest.raises(ReleaseError, match=message):
        CACHE.KubeconfigTokenCache(source, directory=tmp_path)
    assert not list(tmp_path.glob("*-cpu.kubeconfig")), "no copy is published"


# --- the release-wide cache ----------------------------------------------------


def test_empty_cpu_kubeconfig_and_shared_gpu_file_reuse_one_copy(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    source = kubeconfig(tmp_path / "shared.kubeconfig", user=exec_user())
    environ = {"KUBECONFIG": os.pathsep.join([str(source), str(source)])}
    with CACHE.ReleaseKubeconfigCache("", environ=environ) as cache:
        assert cache.cpu_kubeconfig == "", "an empty path has nothing to cache"
        _prepared, values = cache.command_inputs(
            ["kubectl", "get", "pods"], None, timeout_seconds=1.0
        )
        assert values is not None
        copies = values["KUBECONFIG"].split(os.pathsep)
        assert len(copies) == 2 and copies[0] == copies[1], (
            "the same source file maps to one cached copy"
        )
    assert len(plugin["calls"]) == 1, "the plugin runs once per source file"


def test_inactive_cache_returns_the_config_unchanged(tmp_path: Path) -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Config:
        cpu_kubeconfig: str

    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=exec_user())
    config = Config(cpu_kubeconfig=str(source))
    with CACHE.ReleaseKubeconfigCache(str(source), dry_run=True) as cache:
        assert cache.rewrite_config(config) is config


def test_commands_with_a_foreign_kubeconfig_env_keep_their_inputs(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=exec_user())
    with CACHE.ReleaseKubeconfigCache(str(source), environ={}) as cache:
        arguments = ["kubectl", "--kubeconfig", str(source), "get", "pods"]
        environment = {"KUBECONFIG": "/elsewhere"}
        assert cache.command_inputs(arguments, environment, timeout_seconds=1.0) == (
            arguments,
            environment,
        )


def test_only_the_cpu_kubeconfig_argument_forms_are_rewritten(
    tmp_path: Path, plugin: dict[str, Any]
) -> None:
    source = kubeconfig(tmp_path / "cpu.kubeconfig", user=exec_user())
    other = str(tmp_path / "other.kubeconfig")
    with CACHE.ReleaseKubeconfigCache(str(source), environ={}) as cache:
        prepared, _values = cache.command_inputs(
            [
                "kubectl",
                "--kubeconfig",
                other,
                f"--kubeconfig={source}",
                "--kubeconfig",
                str(source),
            ],
            None,
            timeout_seconds=1.0,
        )
        assert prepared == [
            "kubectl",
            "--kubeconfig",
            other,
            f"--kubeconfig={cache.cpu_kubeconfig}",
            "--kubeconfig",
            cache.cpu_kubeconfig,
        ]
        assert cache.cpu_kubeconfig != str(source)
