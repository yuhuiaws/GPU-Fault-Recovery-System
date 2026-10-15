"""The release engine runs each kubeconfig exec plugin once, not once per kubectl.

Measured on the deploy host: every ``kubectl`` call spends ~0.8 s of its ~1.2 s in
``aws eks get-token``, and a CONTROL_PLANE_ONLY ``config`` release issues ~250
calls. The cache writes a private copy of each kubeconfig with the exec entry
replaced by the token it returned, refreshes it before the token expires, and
fails closed when the plugin fails.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import api_budget as API_BUDGET
from gpu_fault.admin.deadlines import deadline_scope
from gpu_fault_release import regional_kubeconfig_cache as CACHE
from gpu_fault_release import regional_release_observation as OBSERVATION
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


def _run_local_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the cache's `aws` to the test's own fixture, never a real CLI."""

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
    _run_local_plugin(tmp_path, monkeypatch)
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
    pauses: list[float] = []
    with pytest.raises(ReleaseError, match="exited 3") as caught:
        CACHE.ReleaseKubeconfigCache(str(source), sleep=pauses.append)
    assert "expired credentials" not in str(caught.value)
    assert (
        _plugin_calls(tmp_path)
        == ["--region us-west-2 eks get-token --cluster-name cpu"] * CACHE.EXEC_ATTEMPTS
    ), "every attempt runs the plugin the kubeconfig names"
    assert pauses == list(CACHE.EXEC_RETRY_DELAYS), "attempts back off in order"


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


# A plugin that follows a script, one outcome per attempt: ``ok``, an exit
# status, ``hang`` or ``garbage``. Every failing attempt prints the token on
# both streams, so a leak of plugin output into any message shows up as a leak
# of the token.
SCRIPTED_PLUGIN = """\
import json, os, sys, time
from pathlib import Path
state = Path(os.environ["PLUGIN_STATE"])
attempt = int(state.read_text()) if state.is_file() else 0
state.write_text(str(attempt + 1))
Path(os.environ["PLUGIN_STATE"] + ".env").write_text(json.dumps({
    name: os.environ.get(name)
    for name in ("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "AWS_METADATA_SERVICE_TIMEOUT")
}))
outcomes = os.environ["PLUGIN_SCRIPT"].split(",")
outcome = outcomes[min(attempt, len(outcomes) - 1)]
token = "k8s-aws-v1." + os.environ["PLUGIN_TOKEN"]
if outcome == "hang":
    time.sleep(30)
if outcome == "garbage":
    print("not a credential " + token)
    sys.exit(0)
if outcome != "ok":
    print(os.environ.get("PLUGIN_STDERR", "") + " " + token, file=sys.stderr)
    print(token)
    sys.exit(int(outcome))
expiry = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 900))
print(json.dumps({
    "apiVersion": "client.authentication.k8s.io/v1beta1",
    "kind": "ExecCredential",
    "status": {"token": token, "expirationTimestamp": expiry},
}))
"""
SECRET = "do-not-print-this-token"
NO_CREDENTIALS = (
    "Unable to locate credentials. You can configure credentials by running "
    '"aws configure".'
)


@pytest.fixture
def scripted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "scripted" / "aws"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n{SCRIPTED_PLUGIN}", encoding="utf-8")
    script.chmod(0o700)
    monkeypatch.setenv("PLUGIN_STATE", str(tmp_path / "attempts"))
    monkeypatch.setenv("PLUGIN_TOKEN", SECRET)
    monkeypatch.setenv("PLUGIN_SCRIPT", "ok")
    monkeypatch.delenv("PLUGIN_STDERR", raising=False)
    monkeypatch.delenv(CACHE.TOKEN_CACHE_ENV, raising=False)
    return script


def _attempts(tmp_path: Path) -> int:
    state = tmp_path / "attempts"
    return int(state.read_text()) if state.is_file() else 0


def _retry_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("exec-credential-")]


def test_a_plugin_that_fails_once_is_retried_and_its_token_cached(
    tmp_path: Path,
    scripted: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The live failure: one `aws eks get-token` exited 253 and took verify with it."""

    _run_local_plugin(tmp_path, monkeypatch)
    monkeypatch.setenv("PLUGIN_SCRIPT", "253,ok")
    monkeypatch.setenv("PLUGIN_STDERR", NO_CREDENTIALS)
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)
    pauses: list[float] = []

    with CACHE.ReleaseKubeconfigCache(str(source), sleep=pauses.append) as cache:
        token = _users(_load(cache.cpu_kubeconfig))["cpu-exec"]["token"]

    assert token == f"k8s-aws-v1.{SECRET}", "the second attempt's token is cached"
    assert _attempts(tmp_path) == 2, "one failure costs exactly one more attempt"
    assert pauses == [CACHE.EXEC_RETRY_DELAYS[0]], "the retry backs off first"
    output = capfd.readouterr()
    [line] = _retry_lines(output.err)
    for field in (
        "attempt=1/3",
        "failure=exited-253",
        "reason=no-credentials",
        "next=2s",
    ):
        assert field in line, f"the retry line must name {field}: {line}"
    assert SECRET not in output.out + output.err, "the token reached the log"
    assert "Unable to locate" not in output.err, "plugin stderr reached the log"


@pytest.mark.parametrize(
    "stderr,reason",
    [
        (NO_CREDENTIALS, "no-credentials"),
        (
            "Error when retrieving credentials from container-role: Error retrieving "
            "metadata",
            "no-credentials",
        ),
        (
            "An error occurred (ExpiredToken) when calling the AssumeRole operation: "
            "The security token included in the request is expired",
            "expired",
        ),
        (
            "An error occurred (AccessDenied) when calling the AssumeRole operation: "
            "User is not authorized to perform: sts:AssumeRole",
            "access-denied",
        ),
        (
            "An error occurred (Throttling) when calling the AssumeRole operation: "
            "Rate exceeded",
            "throttled",
        ),
        (
            'Could not connect to the endpoint URL: "https://sts.us-west-2.amazonaws.com/"',
            "network",
        ),
        ("a message nobody has seen before", "unknown"),
    ],
)
def test_a_persistent_failure_fails_closed_with_the_classified_reason(
    tmp_path: Path,
    scripted: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    stderr: str,
    reason: str,
) -> None:
    _run_local_plugin(tmp_path, monkeypatch)
    monkeypatch.setenv("PLUGIN_SCRIPT", "253")
    monkeypatch.setenv("PLUGIN_STDERR", stderr)
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)
    pauses: list[float] = []

    with pytest.raises(OBSERVATION.ApiUnavailable) as caught:
        CACHE.ReleaseKubeconfigCache(str(source), sleep=pauses.append)

    message = str(caught.value)
    assert caught.value.reason == reason, message
    assert message.startswith(
        f"exec credential plugin exited 253 ({reason}) after 3 attempts in "
    ), message
    assert _attempts(tmp_path) == CACHE.EXEC_ATTEMPTS, "the retries were not bounded"
    assert pauses == list(CACHE.EXEC_RETRY_DELAYS)
    assert SECRET not in message and stderr not in message, (
        "the error carries plugin output"
    )
    assert caught.value.__context__ is None, "a chained error could carry output"
    output = capfd.readouterr()
    assert SECRET not in output.out + output.err, "the token reached the log"
    assert len(_retry_lines(output.err)) == CACHE.EXEC_ATTEMPTS - 1


def test_output_that_is_not_json_is_retried_and_never_echoed(
    tmp_path: Path, scripted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_local_plugin(tmp_path, monkeypatch)
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)
    monkeypatch.setenv("PLUGIN_SCRIPT", "garbage,ok")
    with CACHE.ReleaseKubeconfigCache(str(source), sleep=lambda _seconds: None):
        assert _attempts(tmp_path) == 2, "a truncated response is worth one more try"

    (tmp_path / "attempts").unlink()
    monkeypatch.setenv("PLUGIN_SCRIPT", "garbage")
    with pytest.raises(OBSERVATION.ApiUnavailable) as caught:
        CACHE.ReleaseKubeconfigCache(str(source), sleep=lambda _seconds: None)
    assert str(caught.value).startswith(
        "exec credential plugin did not return JSON (unknown) after 3 attempts"
    ), str(caught.value)
    assert SECRET not in str(caught.value)


def test_a_contract_violation_is_not_retried(
    tmp_path: Path, plugin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same plugin returns the same wrong document: retrying only costs time."""

    plugin.write_text(
        f"#!{sys.executable}\n"
        + PLUGIN.replace('"kind": "ExecCredential"', '"kind": "Something"')
    )
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", plugin)
    pauses: list[float] = []
    with pytest.raises(ReleaseError, match="identity is invalid") as caught:
        CACHE.ReleaseKubeconfigCache(str(source), sleep=pauses.append)
    assert not isinstance(caught.value, OBSERVATION.ApiUnavailable), (
        "a contract violation is a finding, not an unavailable API"
    )
    assert len(_plugin_calls(tmp_path)) == 1, "the plugin ran more than once"
    assert pauses == [], "a contract violation paused for a retry"


def test_a_hung_plugin_is_stopped_at_its_timeout_and_retried(
    tmp_path: Path,
    scripted: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    _run_local_plugin(tmp_path, monkeypatch)
    monkeypatch.setattr(CACHE, "EXEC_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setenv("PLUGIN_SCRIPT", "hang,ok")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)

    started = time.monotonic()
    with CACHE.ReleaseKubeconfigCache(str(source), sleep=lambda _seconds: None):
        elapsed = time.monotonic() - started

    assert _attempts(tmp_path) == 2
    assert elapsed < 15, f"the hung attempt was not stopped at its timeout: {elapsed}"
    [line] = _retry_lines(capfd.readouterr().err)
    assert "failure=timed-out" in line, line


def test_the_fetch_budget_bounds_every_attempt_and_pause(
    tmp_path: Path, scripted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No time left for a pause and an attempt: the fetch stops instead of starting one."""

    _run_local_plugin(tmp_path, monkeypatch)
    monkeypatch.setattr(CACHE, "EXEC_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(CACHE, "EXEC_FETCH_SECONDS", 2.5)
    monkeypatch.setenv("PLUGIN_SCRIPT", "hang")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)
    pauses: list[float] = []

    started = time.monotonic()
    with pytest.raises(OBSERVATION.ApiUnavailable) as caught:
        CACHE.ReleaseKubeconfigCache(str(source), sleep=pauses.append)
    elapsed = time.monotonic() - started

    assert str(caught.value).startswith("exec credential plugin timed out"), str(
        caught.value
    )
    assert "after 1 attempt in" in str(caught.value), str(caught.value)
    assert pauses == [], "a pause that cannot be followed by an attempt is not taken"
    assert elapsed < 12, f"the fetch outran its own budget: {elapsed}"


def test_the_fetch_never_outlives_the_callers_deadline(
    tmp_path: Path, scripted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh before a kubectl call runs inside that call's deadline."""

    _run_local_plugin(tmp_path, monkeypatch)
    monkeypatch.setenv("PLUGIN_SCRIPT", "hang")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)

    started = time.monotonic()
    with deadline_scope("caller", 2.0):
        with pytest.raises(OBSERVATION.ApiUnavailable, match="timed out"):
            CACHE.ReleaseKubeconfigCache(str(source), sleep=lambda _seconds: None)
    elapsed = time.monotonic() - started

    assert _attempts(tmp_path) == 1, "a retry started after the caller's deadline"
    assert elapsed < 12, f"the fetch outlived the caller's deadline: {elapsed}"


@pytest.mark.allows_cluster_binaries("aws")
def test_every_attempt_is_admitted_through_the_deployment_api_budget(
    tmp_path: Path, scripted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No private path: each attempt is an `aws` call the shared budget admits.

    It also reaches the plugin with the scope's IMDS defaults
    (`api_budget.metadata_service_defaults`), the cheapest retry there is.
    """

    monkeypatch.delenv(API_BUDGET.ROOT_ENV, raising=False)
    for name in API_BUDGET.METADATA_SERVICE_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "aws-config"
    config.write_text("[default]\nregion = us-west-2\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("PLUGIN_SCRIPT", "253,ok")
    source = _kubeconfig(tmp_path / "cpu.kubeconfig", scripted)

    with API_BUDGET.deployment_api_budget():
        with CACHE.ReleaseKubeconfigCache(str(source), sleep=lambda _seconds: None):
            commands = API_BUDGET.statistics()["backends"]["aws"]["commands"]

    assert commands == 2, "an attempt bypassed the deployment API budget"
    seen = json.loads((tmp_path / "attempts.env").read_text())
    assert seen == {
        "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "3",
        "AWS_METADATA_SERVICE_TIMEOUT": "2",
    }, "the plugin ran without the deployment scope's IMDS defaults"
