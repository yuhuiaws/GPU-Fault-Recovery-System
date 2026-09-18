"""Local Make/release gates refuse missing native tools before expensive work."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin.postgres_grant import ALLOCATION_ENV
from scripts import run_release_gates
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
ALERTS = lazy_script_module(ROOT / "scripts/check-alert-rules.py")
FULL_TEST_TARGETS = ("test", "test-parallel", "test-parallel-release", "test-shuffled")


@pytest.fixture
def version() -> str:
    match = re.search(
        r"^PROMTOOL_VERSION \?= (\d+\.\d+\.\d+)$",
        (ROOT / "Makefile").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, "the version pin must remain authoritative in Make"
    return match.group(1)


def write_tool(path: Path, version: str, *, status: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        '[ "$#" = 1 ] && [ "$1" = "--version" ] || exit 91\n'
        f"printf '%s\\n' 'probe' >> {shlex.quote(str(path.parent / 'probes'))}\n"
        f"printf '%s\\n' {shlex.quote(f'promtool, version {version}')}\n"
        f"exit {status}\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "Makefile").symlink_to(ROOT / "Makefile")
    (root / "scripts/check-alert-rules.py").symlink_to(
        ROOT / "scripts/check-alert-rules.py"
    )
    return root


def environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root),
        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPYCACHEPREFIX": str(root / "pycache"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "KUBECONFIG": "/dev/null",
    }


def make(
    root: Path, target: str, *assignments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            target,
            f"PYTHON={sys.executable}",
            f"SUPPLY_CHAIN_TOOLS_VENV={root / 'supply-chain'}",
            "PYTEST_XDIST_WORKERS=2",
            "GPU_FAULT_TEST_SHUFFLE_SEED=17",
            *assignments,
        ],
        cwd=root,
        env=env if env is not None else environment(root),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_tool_only_checks_the_executable_without_rules_or_pytest(
    tmp_path: Path,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = write_tool(tmp_path / "promtool", version)
    monkeypatch.setattr(ALERTS, "RULE_FILES", (tmp_path / "absent-rules.yaml",))
    status = ALERTS.main(["--tool-only", "--promtool", str(tool), "--version", version])
    assert status == 0, "tool-only preflight must not read alert rules or run pytest"
    assert json.loads(capsys.readouterr().out) == {
        "status": "ready",
        "promtool": str(tool),
        "version": version,
    }, "a dependency check must not claim rule or behavioral evidence"
    assert (tool.parent / "probes").read_text().splitlines() == ["probe"], (
        "tool-only mode must execute exactly one version probe"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--tool-only"],
        ["--tool-only", "--version", "3.14.0", "--pytest-files", "tests/example.py"],
    ],
)
def test_tool_only_rejects_unpinned_or_ambiguous_requests(
    arguments: list[str], tmp_path: Path
) -> None:
    assert ALERTS.main([*arguments, "--promtool", str(tmp_path / "absent")]) == 2, (
        "tool-only cannot remove requested behavior tests or accept an unpinned tool"
    )


@pytest.mark.parametrize(
    "defect", ["failed-probe", "wrong-program", "encoding", "exec"]
)
def test_unusable_version_probes_fail_closed(
    tmp_path: Path, version: str, defect: str, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = write_tool(
        tmp_path / "promtool", version, status=1 if defect == "failed-probe" else 0
    )
    if defect == "wrong-program":
        tool.write_text(
            f"#!/bin/sh\nprintf '%s\\n' 'another-tool, version {version}'\n",
            encoding="utf-8",
        )
    elif defect == "encoding":
        tool.write_text("#!/bin/sh\nprintf '\\377'\n", encoding="utf-8")
    elif defect == "exec":
        tool.write_text("not an executable format\n", encoding="utf-8")
    assert (
        ALERTS.main(["--tool-only", "--promtool", str(tool), "--version", version]) == 2
    ), "an executable flag or a matching substring does not prove a usable tool"
    assert "ci-supply-chain-tools" in capsys.readouterr().err, (
        "failed probes must give actionable guidance without dumping tool output"
    )


def test_version_probe_is_bounded_and_does_not_echo_failed_output(
    tmp_path: Path,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = write_tool(tmp_path / "promtool", version)

    def timeout(command, **options):
        assert options["timeout"] == 10, "tool preflight must not hang a deployment"
        raise subprocess.TimeoutExpired(
            command, options["timeout"], output="untrusted-probe-output"
        )

    monkeypatch.setattr(
        ALERTS,
        "subprocess",
        SimpleNamespace(run=timeout, SubprocessError=subprocess.SubprocessError),
    )
    assert (
        ALERTS.main(["--tool-only", "--promtool", str(tool), "--version", version]) == 2
    ), "a timed out version probe must refuse all subsequent gates"
    assert "untrusted-probe-output" not in capsys.readouterr().err, (
        "a failed native tool must not leak its arbitrary output"
    )


@pytest.mark.parametrize("binding", ["environment", "make"])
def test_explicit_empty_tool_binding_does_not_fall_back(
    checkout: Path, version: str, binding: str
) -> None:
    tools = checkout / "supply-chain/bin"
    tools.mkdir(parents=True)
    (tools / "python").symlink_to(sys.executable)
    tool = write_tool(tools / "promtool", version)
    env = environment(checkout)
    assignments: list[str] = []
    if binding == "environment":
        env["PROMTOOL"] = ""
    else:
        assignments.append("PROMTOOL=")
    completed = make(checkout, "promtool-preflight", *assignments, env=env)
    assert completed.returncode == 2, "an explicit empty binding must fail closed"
    assert "ci-supply-chain-tools" in completed.stderr, "missing tool needs guidance"
    assert not (tool.parent / "probes").exists(), (
        "preflight must not silently repair an invalid explicit tool selection"
    )


@pytest.mark.parametrize("ci", ["", "true"])
@pytest.mark.parametrize("target", [*FULL_TEST_TARGETS, "promtool-check"])
@pytest.mark.parametrize("defect", ["missing", "nonexecutable", "wrong-version"])
def test_full_test_targets_refuse_bad_tools_before_pytest(
    checkout: Path, version: str, target: str, defect: str, ci: str
) -> None:
    tool = checkout / "promtool"
    if defect != "missing":
        write_tool(tool, version + "1" if defect == "wrong-version" else version)
        if defect == "nonexecutable":
            tool.chmod(0o600)
    completed = make(checkout, target, f"PROMTOOL={tool}", f"CI={ci}")
    output = completed.stdout + completed.stderr
    assert completed.returncode == 2, output
    assert "ci-supply-chain-tools" in output, "failure must name the pinned installer"
    assert "-m pytest" not in completed.stdout, (
        "missing or wrong native tools must fail before test launch"
    )
    assert '"status": "skipped"' not in output, "missing dependencies are not skips"


@pytest.mark.parametrize("target", FULL_TEST_TARGETS)
@pytest.mark.parametrize("override", ["unset", "environment", "make", "path"])
def test_make_exports_the_selected_tool_to_pytest_workers_and_children(
    checkout: Path, version: str, target: str, override: str
) -> None:
    tools = checkout / "supply-chain/bin"
    tools.mkdir(parents=True)
    (tools / "python").symlink_to(sys.executable)
    default = write_tool(tools / "promtool", version)
    selected = default
    env = environment(checkout)
    assignments: list[str] = []
    if override != "unset":
        selected = write_tool(checkout / "explicit tools/promtool", version)
        if override == "make":
            env["PROMTOOL"] = str(checkout / "absent-ambient-tool")
            assignments.append(f"PROMTOOL={selected}")
        elif override == "environment":
            env["PROMTOOL"] = str(selected)
        else:
            env["PATH"] = str(selected.parent) + os.pathsep + env["PATH"]
            env["PROMTOOL"] = "promtool"
    env["EXPECTED_PROMTOOL"] = env.get("PROMTOOL", str(selected))
    if override == "make":
        env["EXPECTED_PROMTOOL"] = str(selected)
    env["EXPECTED_VERSION"] = version
    (checkout / "test_inheritance.py").write_text(
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "@pytest.mark.parametrize('case', range(4))\n"
        "def test_inheritance(case):\n"
        "    tool = os.environ['PROMTOOL']\n"
        "    assert tool == os.environ['EXPECTED_PROMTOOL']\n"
        "    child = subprocess.run([tool, '--version'], check=True,\n"
        "                           capture_output=True, text=True)\n"
        "    assert child.stdout.strip() == (\n"
        "        'promtool, version ' + os.environ['EXPECTED_VERSION'])\n"
        "    worker = os.environ.get('PYTEST_XDIST_WORKER', 'master')\n"
        "    Path(f'worker-{case}.json').write_text(json.dumps(worker))\n",
        encoding="utf-8",
    )
    completed = make(checkout, target, *assignments, env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    workers = {json.loads(path.read_text()) for path in checkout.glob("worker-*.json")}
    assert workers == ({"master"} if target == "test" else {"gw0", "gw1"}), (
        "the real Make target must reach pytest and both xdist workers"
    )
    assert len((selected.parent / "probes").read_text().splitlines()) == 5, (
        "one preflight and four child probes must use the same selected executable"
    )
    if override != "unset":
        assert not (default.parent / "probes").exists(), (
            "an explicit override must not probe or execute the default tool"
        )


@pytest.mark.parametrize("mode", ["check", "release"])
@pytest.mark.parametrize("defect", ["missing", "nonexecutable", "wrong-version"])
def test_release_refuses_bad_tools_before_static_native_or_artifact_gates(
    checkout: Path,
    version: str,
    mode: str,
    defect: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = checkout / "promtool"
    if defect != "missing":
        write_tool(tool, version + "1" if defect == "wrong-version" else version)
        if defect == "nonexecutable":
            tool.chmod(0o600)
    monkeypatch.setattr(run_release_gates, "ROOT", checkout)
    monkeypatch.setenv("PROMTOOL", str(tool))
    monkeypatch.setenv("GPU_FAULT_TEST_POSTGRES_URL", "explicit-test-reference")
    calls: list[list[str]] = []

    def run(command, **options):
        calls.append(command)
        assert command[1] == "promtool-preflight", (
            "release started an expensive gate before resolving its native tool"
        )
        return subprocess.run(command, **options)

    def popen(*_args, **_kwargs):
        pytest.fail("static/native/artifact gates must not start without promtool")

    monkeypatch.setattr(
        run_release_gates,
        "subprocess",
        SimpleNamespace(**{**vars(subprocess), "run": run, "Popen": popen}),
    )
    assert run_release_gates.main(["--mode", mode, "--python", sys.executable]) == 2, (
        "both actual local release entrypoints must refuse unavailable prerequisites"
    )
    assert calls == [["make", "promtool-preflight", f"PYTHON={sys.executable}"]], (
        "an invalid tool must be the only child launched"
    )


def test_release_preflight_preserves_scope_and_does_not_consume_pg_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMTOOL", "/controlled/promtool")
    monkeypatch.setenv("COSIGN_PASSWORD", "example-test-only-signing-password")
    monkeypatch.setenv("HOME", "/original/build-home")
    monkeypatch.setenv("PGPASSFILE", "/original/pgpass")
    monkeypatch.setenv(ALLOCATION_ENV, str(tmp_path / "absent"))
    monkeypatch.setenv("GPU_FAULT_DEPLOY_API_BUDGET_DIR", "/original/api-budget")
    monkeypatch.setenv("GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC", "123")
    monkeypatch.setenv("PATH", "/original/api-budget/bin:/test/tools:/usr/bin")
    before = dict(os.environ)
    calls: list[dict[str, str]] = []

    def run(command, *, env, **_options):
        calls.append(env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_release_gates, "subprocess", SimpleNamespace(run=run))
    run_release_gates.preflight_promtool(sys.executable, cache_root=tmp_path)
    assert len(calls) == 1, "the preflight must use a single Make-owned probe"
    child = calls[0]
    assert child["PROMTOOL"] == "/controlled/promtool", "explicit tool binding was lost"
    assert child["HOME"] == "/original/build-home", "only native PG may change HOME"
    assert child["PGPASSFILE"] == "/original/pgpass", (
        "the native-tool preflight must not load the private PG allocation"
    )
    assert child["PATH"] == "/test/tools:/usr/bin", "deploy shims must stay isolated"
    assert "COSIGN_PASSWORD" not in child, "a tool probe must not inherit signing data"
    assert "GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC" not in child, (
        "the tool gate must use the existing deployment-scope isolation"
    )
    assert dict(os.environ) == before, "preflight changed the deployment environment"


def prepare_impact_checkout(root: Path, target: str, test_body: str) -> None:
    shutil.copyfile(
        ROOT / "scripts/select-affected-tests.py",
        root / "scripts/select-affected-tests.py",
    )
    for directory in ("src", "tools"):
        (root / directory).symlink_to(ROOT / directory, target_is_directory=True)
    (root / "scripts/run_static_gates.py").symlink_to(
        ROOT / "scripts/run_static_gates.py"
    )
    (root / "testcases").mkdir()
    for name in ("fault-scenarios.yaml", "regional-execution-order.yaml"):
        (root / "testcases" / name).symlink_to(ROOT / "testcases" / name)
    (root / "testcases/change-impact.yaml").write_text(
        json.dumps(
            {
                "version": 1,
                "max_domains_before_full": 3,
                "safe_regional_risks": ["non-destructive", "read-only-signal-replay"],
                "full": {
                    "checks": ["check"],
                    "postgres_command": ["make", "test-postgres-stress"],
                },
                "rules": [
                    {
                        "id": "selected-rules",
                        "description": "Isolated direct selector regression.",
                        "paths": ["inputs/changed.txt"],
                        "pytest": [f"{target}::test_selected"],
                        "regional_cases": [],
                        "checks": ["config-check"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "Makefile").unlink()
    (root / "Makefile").write_text(
        f"include {ROOT / 'Makefile'}\n"
        "config-check:\n\t@printf 'static\\n' > static-reached\n",
        encoding="utf-8",
    )
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        test_body
        + "\ndef test_not_selected():\n"
        + "    raise AssertionError('the selector expanded the exact nodeid')\n",
        encoding="utf-8",
    )


def run_impact(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/select-affected-tests.py",
            "--changed-file",
            "inputs/changed.txt",
            "--execute",
        ],
        cwd=root,
        env={**env, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("override", ["unset", "absolute", "relative", "path"])
def test_direct_selective_python_binds_the_validated_tool_to_the_exact_pytest_child(
    checkout: Path, version: str, override: str
) -> None:
    listing = make(checkout, "promql-test-files")
    assert listing.returncode == 0, listing.stdout + listing.stderr
    target = json.loads(listing.stdout)[0]
    prepare_impact_checkout(
        checkout,
        target,
        "import json, os, subprocess\n"
        "from pathlib import Path\n"
        "def test_selected():\n"
        "    assert Path('static-reached').is_file()\n"
        "    tool = os.environ['PROMTOOL']\n"
        "    assert Path(tool).is_absolute()\n"
        "    assert tool == os.environ['EXPECTED_PROMTOOL']\n"
        "    assert 'COSIGN_PASSWORD' not in os.environ\n"
        "    subprocess.run([tool, '--version'], check=True)\n"
        "    Path('selected-tool.json').write_text(json.dumps(tool))\n",
    )
    tools = checkout / "supply-chain/bin"
    tools.mkdir(parents=True)
    (tools / "python").symlink_to(sys.executable)
    default = write_tool(tools / "promtool", version)
    env = environment(checkout)
    env["SUPPLY_CHAIN_TOOLS_VENV"] = str(checkout / "supply-chain")
    env["COSIGN_PASSWORD"] = "example-test-only-signing-password"
    selected = default
    if override != "unset":
        selected = write_tool(checkout / "explicit tools/promtool", version)
        if override == "path":
            env["PROMTOOL"] = "promtool"
            env["PATH"] = str(selected.parent) + os.pathsep + env["PATH"]
        else:
            env["PROMTOOL"] = (
                str(selected)
                if override == "absolute"
                else str(selected.relative_to(checkout))
            )
    env["EXPECTED_PROMTOOL"] = str(selected)
    before = dict(env)
    completed = run_impact(checkout, env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((checkout / "selected-tool.json").read_text()) == str(selected), (
        "direct Python must bind the Make-validated executable into pytest"
    )
    assert len((selected.parent / "probes").read_text().splitlines()) == 2, (
        "one tool-only preflight and the actual pytest child must use the same tool"
    )
    if override != "unset":
        assert not (default.parent / "probes").exists(), (
            "an explicit override must not use the default installation"
        )
    assert env == before, "the direct selector must not mutate its caller's environment"


@pytest.mark.parametrize("defect", ["missing", "nonexecutable", "wrong-version"])
def test_direct_selective_python_refuses_bad_tools_before_static_or_pytest(
    checkout: Path, version: str, defect: str
) -> None:
    listing = make(checkout, "promql-test-files")
    assert listing.returncode == 0, listing.stdout + listing.stderr
    target = json.loads(listing.stdout)[0]
    prepare_impact_checkout(checkout, target, "def test_selected():\n    pass\n")
    tool = checkout / "promtool"
    if defect != "missing":
        write_tool(tool, version + "1" if defect == "wrong-version" else version)
        if defect == "nonexecutable":
            tool.chmod(0o600)
    env = environment(checkout)
    env["PROMTOOL"] = str(tool)
    completed = run_impact(checkout, env)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "ci-supply-chain-tools" in completed.stderr, (
        "direct invocation must expose the shared pinned installer guidance"
    )
    assert not (checkout / "static-reached").exists(), (
        "missing native dependencies must refuse before even selected static gates"
    )
    assert "test session starts" not in completed.stdout, "pytest started after refusal"


def test_direct_selective_python_without_promql_does_not_require_the_tool(
    checkout: Path,
) -> None:
    prepare_impact_checkout(
        checkout,
        "tests/test_plain.py",
        "from pathlib import Path\n"
        "def test_selected():\n"
        "    assert Path('static-reached').is_file()\n"
        "    Path('plain-reached').write_text('passed')\n",
    )
    env = environment(checkout)
    env["PROMTOOL"] = str(checkout / "missing-promtool")
    completed = run_impact(checkout, env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (checkout / "plain-reached").read_text() == "passed", (
        "a non-PromQL selection must remain executable without a native tool"
    )
