"""The release engine runs each kubeconfig exec plugin once, not once per kubectl.

Measured on the deploy host: every ``kubectl`` call spends ~0.8 s of its ~1.2 s in
``aws eks get-token``, and a CONTROL_PLANE_ONLY ``config`` release issues ~250
calls. The cache writes a private copy of each kubeconfig with the exec entry
replaced by the token it returned, refreshes it before the token expires, and
fails closed when the plugin fails.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release import regional_kubeconfig_cache as CACHE
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_release_config import ReleaseError

PLUGIN = """\
import json, os, sys, time
from pathlib import Path
Path(os.environ["PLUGIN_LOG"]).open("a").write(" ".join(sys.argv[1:]) + "\\n")
if os.environ.get("PLUGIN_FAIL"):
    print("get-token: expired credentials", file=sys.stderr)
    sys.exit(3)
lifetime = float(os.environ.get("PLUGIN_LIFETIME", "900"))
expiry = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + lifetime))
print(json.dumps({
    "apiVersion": "client.authentication.k8s.io/v1beta1",
    "kind": "ExecCredential",
    "status": {"token": "k8s-aws-v1." + os.environ.get("PLUGIN_TOKEN", "one"),
               "expirationTimestamp": expiry},
}))
"""


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "aws"
    script.write_text(f"#!{sys.executable}\n{PLUGIN}", encoding="utf-8")
    script.chmod(0o700)
    monkeypatch.setenv("PLUGIN_LOG", str(tmp_path / "plugin.log"))
    monkeypatch.delenv("PLUGIN_FAIL", raising=False)
    monkeypatch.delenv("PLUGIN_LIFETIME", raising=False)
    monkeypatch.setenv("PLUGIN_TOKEN", "one")
    monkeypatch.delenv(CACHE.TOKEN_CACHE_ENV, raising=False)
    run = CACHE.run_command

    def fixture_command(arguments, **kwargs):
        assert arguments[0] == "aws"
        source = Path(shutil.which("aws", path=kwargs["environment"]["PATH"]) or "")
        assert source.name == "aws" and source.is_relative_to(tmp_path), (
            "credential tests may execute only their own local fixture"
        )
        executable = source.with_name("exec-credential-fixture.py")
        shutil.copyfile(source, executable)
        return run([sys.executable, str(executable), *arguments[1:]], **kwargs)

    monkeypatch.setattr(CACHE, "run_command", fixture_command)
    return script


def _plugin_calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "plugin.log"
    return log.read_text(encoding="utf-8").splitlines() if log.is_file() else []


def _kubeconfig(path: Path, plugin: Path, *, name: str = "cpu") -> Path:
    document = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": name,
                "cluster": {
                    "server": f"https://{name}.example",
                    "certificate-authority-data": "Q0E=",
                },
            }
        ],
        "contexts": [
            {"name": name, "context": {"cluster": name, "user": f"{name}-exec"}},
            {
                "name": f"{name}-static",
                "context": {"cluster": name, "user": f"{name}-static"},
            },
        ],
        "current-context": name,
        "preferences": {"colors": True},
        "users": [
            {
                "name": f"{name}-exec",
                "user": {
                    "exec": {
                        "apiVersion": "client.authentication.k8s.io/v1beta1",
                        "command": "aws",
                        "args": [
                            "--region",
                            "us-west-2",
                            "eks",
                            "get-token",
                            "--cluster-name",
                            name,
                        ],
                        "env": [
                            {"name": "AWS_PROFILE", "value": "deploy"},
                            {
                                "name": "PATH",
                                "value": f"{plugin.parent}:{os.environ.get('PATH', '')}",
                            },
                        ],
                    }
                },
            },
            {"name": f"{name}-static", "user": {"token": "static-token"}},
        ],
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return path


def _load(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _users(document: dict) -> dict[str, dict]:
    return {entry["name"]: entry["user"] for entry in document["users"]}


def test_exec_users_become_static_tokens_and_the_rest_is_untouched(
    tmp_path: Path, plugin: Path
) -> None:
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    original = _load(source)

    with CACHE.ReleaseKubeconfigCache(str(source)) as cache:
        cached = Path(cache.cpu_kubeconfig)
        assert cached != source, "the engine must be pointed at a copy"
        assert cached.is_file(), "the copy must exist while the cache is open"
        mode = stat.S_IMODE(cached.stat().st_mode)
        assert mode == 0o600, f"the copy carries a bearer token; mode was {mode:o}"
        parent_mode = stat.S_IMODE(cached.parent.stat().st_mode)
        assert parent_mode == 0o700, f"private temp dir expected, got {parent_mode:o}"
        document = _load(cached)
        users = _users(document)
        assert "exec" not in users["cpu-exec"], "the exec entry must be replaced"
        assert users["cpu-exec"]["token"] == "k8s-aws-v1.one", (
            "the token is the plugin's status.token"
        )
        assert users["cpu-static"] == {"token": "static-token"}, (
            "a user without exec is copied as-is"
        )
        for key in ("clusters", "contexts", "current-context", "preferences"):
            assert document[key] == original[key], f"{key} must survive unchanged"
        assert _plugin_calls(tmp_path) == [
            "--region us-west-2 eks get-token --cluster-name cpu"
        ], "the plugin runs exactly once with the kubeconfig's own args"
        assert _load(source) == original, "the site's kubeconfig is never rewritten"
    assert not cached.parent.exists(), "cleanup removes the private directory"


def test_the_plugin_env_entries_reach_the_plugin(
    tmp_path: Path, plugin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    script = tmp_path / "env-plugin" / "aws"
    script.parent.mkdir()
    script.write_text(
        f"#!{sys.executable}\n"
        + PLUGIN.replace(
            '" ".join(sys.argv[1:])', 'os.environ.get("AWS_PROFILE", "<unset>")'
        ),
        encoding="utf-8",
    )
    script.chmod(0o700)
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", script)
    with CACHE.ReleaseKubeconfigCache(str(source)):
        assert _plugin_calls(tmp_path) == ["deploy"], (
            "users[].user.exec.env must be applied to the plugin's environment"
        )


def test_refresh_happens_only_inside_the_lead_window_and_replaces_in_place(
    tmp_path: Path, plugin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 1_000_000.0}
    monkeypatch.setenv("PLUGIN_LIFETIME", "900")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    # The plugin stamps expiry from the real clock; the cache reads the fake
    # one, so the fake clock starts at the real time and is then advanced.
    clock["now"] = time.time()
    with CACHE.ReleaseKubeconfigCache(str(source), now=lambda: clock["now"]) as cache:
        cached = Path(cache.cpu_kubeconfig)
        first_inode = cached.stat().st_ino
        monkeypatch.setenv("PLUGIN_TOKEN", "two")

        clock["now"] += 600  # 300 s left: outside the 180 s lead
        cache.refresh_if_needed()
        assert len(_plugin_calls(tmp_path)) == 1, (
            "a token with more than the lead time left is not re-fetched"
        )
        assert _users(_load(cached))["cpu-exec"]["token"] == "k8s-aws-v1.one", (
            "the copy is untouched outside the lead window"
        )

        clock["now"] += 150  # 150 s left: inside the lead
        cache.refresh_if_needed()
        assert len(_plugin_calls(tmp_path)) == 2, (
            "a token inside the lead window is fetched again"
        )
        assert _users(_load(cached))["cpu-exec"]["token"] == "k8s-aws-v1.two", (
            "the copy carries the new token"
        )
        assert cache.cpu_kubeconfig == str(cached), "the path handed out is stable"
        assert cached.stat().st_ino != first_inode, (
            "the file is replaced atomically (new inode), not rewritten in place"
        )
        assert stat.S_IMODE(cached.stat().st_mode) == 0o600, (
            "the replacement keeps the private mode"
        )
        leftovers = [p for p in cached.parent.iterdir() if p.name.startswith(".")]
        assert leftovers == [], f"no temp files may linger: {leftovers}"


def test_a_failing_plugin_is_a_release_error_not_a_credential_less_kubeconfig(
    tmp_path: Path, plugin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PLUGIN_FAIL", "1")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with pytest.raises(ReleaseError, match="exited 3") as caught:
        CACHE.ReleaseKubeconfigCache(str(source))
    assert "expired credentials" not in str(caught.value)
    assert _plugin_calls(tmp_path) == [
        "--region us-west-2 eks get-token --cluster-name cpu"
    ], "the failure comes from the plugin the kubeconfig names"


def test_a_failed_construction_leaves_no_temp_directory(
    tmp_path: Path, plugin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("PLUGIN_FAIL", "1")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with pytest.raises(ReleaseError):
        CACHE.ReleaseKubeconfigCache(str(source))
    assert list((tmp_path / "tmp").iterdir()) == [], (
        "a cache that failed to build must remove its private directory"
    )


def test_dry_run_neither_runs_the_plugin_nor_moves_the_path(
    tmp_path: Path, plugin: Path
) -> None:
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    environ = {"KUBECONFIG": str(source)}
    with CACHE.ReleaseKubeconfigCache(
        str(source), dry_run=True, environ=environ
    ) as cache:
        assert cache.cpu_kubeconfig == str(source), "dry-run keeps the original path"
        assert environ["KUBECONFIG"] == str(source), "dry-run leaves KUBECONFIG alone"
        assert _plugin_calls(tmp_path) == [], "dry-run must not run the plugin"
        assert cache.active is False, "dry-run reports the cache as inactive"


def test_opt_out_environment_restores_the_per_call_plugin(
    tmp_path: Path, plugin: Path
) -> None:
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    environ = {CACHE.TOKEN_CACHE_ENV: "false", "KUBECONFIG": str(source)}
    with CACHE.ReleaseKubeconfigCache(str(source), environ=environ) as cache:
        assert cache.cpu_kubeconfig == str(source), "opt-out keeps the original path"
        assert environ["KUBECONFIG"] == str(source), "opt-out leaves KUBECONFIG alone"
        assert _plugin_calls(tmp_path) == [], "opt-out must not run the plugin"
    assert CACHE.token_cache_enabled({CACHE.TOKEN_CACHE_ENV: "0"}) is False, (
        "0 disables the cache"
    )
    assert CACHE.token_cache_enabled({}) is True, "the cache is on by default"


def test_kubeconfig_env_is_cached_only_for_bounded_direct_gpu_calls(
    tmp_path: Path, plugin: Path
) -> None:
    cpu = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin, name="cpu")
    gpu = _kubeconfig(tmp_path / "gpu.kubeconfig", plugin, name="gpu")
    environ = {"KUBECONFIG": str(gpu)}
    with CACHE.ReleaseKubeconfigCache(str(cpu), environ=environ) as cache:
        command, environment = cache.command_inputs(
            ["kubectl", "--context", "gpu", "get", "nodes"], None, timeout_seconds=60
        )
        assert command == ["kubectl", "--context", "gpu", "get", "nodes"]
        assert environment is not None
        assert environ["KUBECONFIG"] == str(gpu)
        assert environment["KUBECONFIG"] != str(gpu)
        gpu_users = _users(_load(environment["KUBECONFIG"]))
        assert gpu_users["gpu-exec"]["token"] == "k8s-aws-v1.one", (
            "the GPU copy carries a static token"
        )
        assert (
            Path(environment["KUBECONFIG"]).parent == Path(cache.cpu_kubeconfig).parent
        ), "both copies share the one private directory"
        assert len(_plugin_calls(tmp_path)) == 2, "one plugin run per kubeconfig"
    assert environ["KUBECONFIG"] == str(gpu), "source authentication never changes"


def test_a_kubeconfig_without_exec_users_keeps_its_original_path(
    tmp_path: Path, plugin: Path
) -> None:
    source = tmp_path / "static.kubeconfig"
    source.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "clusters": [{"name": "c", "cluster": {"server": "https://c"}}],
                "contexts": [{"name": "c", "context": {"cluster": "c", "user": "u"}}],
                "current-context": "c",
                "users": [{"name": "u", "user": {"token": "static"}}],
            }
        ),
        encoding="utf-8",
    )
    missing = tmp_path / "does-not-exist.kubeconfig"
    environ = {"KUBECONFIG": str(missing)}
    with CACHE.ReleaseKubeconfigCache(str(source), environ=environ) as cache:
        assert cache.cpu_kubeconfig == str(source), (
            "nothing to cache: kubectl keeps reading the site file"
        )
        assert environ["KUBECONFIG"] == str(missing), (
            "a missing file is left for kubectl to report exactly as before"
        )


def test_rewrite_config_points_the_release_at_the_cached_copy(
    tmp_path: Path, plugin: Path
) -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Config:
        cpu_kubeconfig: str
        namespace: str = "gpu-fault-system"

    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with CACHE.ReleaseKubeconfigCache(str(source)) as cache:
        rewritten = cache.rewrite_config(Config(cpu_kubeconfig=str(source)))
        assert rewritten.cpu_kubeconfig == cache.cpu_kubeconfig, (
            "every module reading release.config.cpu_kubeconfig must see the copy"
        )
        assert rewritten.namespace == "gpu-fault-system", "other fields are kept"


def test_runner_calls_the_refresh_hook_before_each_real_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    runner = ROLLOUT.Runner(before_command=lambda: calls.append("refresh"))
    runner.run([sys.executable, "-c", "pass"])
    assert runner.condition([sys.executable, "-c", "pass", "wait"]) is True
    runner.probe_output([sys.executable, "-c", "pass"])
    assert calls == ["refresh"] * 3, "run, condition and probe_output prepare first"

    dry = ROLLOUT.Runner(dry_run=True, before_command=lambda: calls.append("dry"))
    dry.run(["kubectl", "apply"])
    assert "dry" not in calls, "a dry-run command spawns nothing, so no refresh"
    capsys.readouterr()


@pytest.mark.parametrize(
    "overrides",
    [
        {"command": "custom-auth"},
        {"command": "/usr/local/bin/aws"},
        {"provideClusterInfo": True},
        {"interactiveMode": "Always"},
        {"apiVersion": "client.authentication.k8s.io/v1alpha1"},
        {"unknownOption": True},
    ],
)
def test_nonstandard_plugins_retain_native_exec_protocol(
    tmp_path: Path, plugin: Path, overrides: dict
) -> None:
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    document = _load(source)
    document["users"][0]["user"]["exec"].update(overrides)
    source.write_text(yaml.safe_dump(document))
    with CACHE.ReleaseKubeconfigCache(str(source)) as cache:
        assert cache.cpu_kubeconfig == str(source)
        assert _plugin_calls(tmp_path) == []


def test_missing_expiry_is_not_an_invented_token_lifetime(
    tmp_path: Path, plugin: Path
) -> None:
    plugin.write_text(
        f"#!{sys.executable}\n"
        + PLUGIN.replace('"expirationTimestamp": expiry', '"unused": expiry').replace(
            ',\n               "unused": expiry', ""
        )
    )
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with CACHE.ReleaseKubeconfigCache(str(source)) as cache:
        assert cache.cpu_kubeconfig == str(source)
    assert "exec" in _users(_load(source))["cpu-exec"]


@pytest.mark.parametrize("lifetime", ["-1", "30"])
def test_expired_or_short_lived_token_is_rejected(
    tmp_path: Path, plugin: Path, monkeypatch: pytest.MonkeyPatch, lifetime: str
) -> None:
    monkeypatch.setenv("PLUGIN_LIFETIME", lifetime)
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with pytest.raises(ReleaseError, match="expiry"):
        CACHE.ReleaseKubeconfigCache(str(source))


def test_scripts_and_long_calls_keep_native_authentication(
    tmp_path: Path, plugin: Path
) -> None:
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with CACHE.ReleaseKubeconfigCache(str(source)) as cache:
        for arguments, timeout in (
            (["bash", "child.sh"], 600),
            (["kubectl", "--kubeconfig", str(source), "exec", "pod"], 900),
        ):
            command, environment = cache.command_inputs(
                arguments, None, timeout_seconds=timeout
            )
            assert command == arguments
            assert environment is None
        command, _ = cache.command_inputs(
            ["kubectl", "--kubeconfig", str(source), "get", "nodes"],
            None,
            timeout_seconds=60,
        )
        assert command[2] == cache.cpu_kubeconfig


def test_cached_authentication_rejects_source_identity_drift(
    tmp_path: Path, plugin: Path
) -> None:
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    with CACHE.ReleaseKubeconfigCache(str(source)) as cache:
        source.write_text(source.read_text() + "\n# changed\n")
        with pytest.raises(ReleaseError, match="kubeconfig changed"):
            cache.refresh_if_needed()
