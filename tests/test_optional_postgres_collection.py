from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDS = (
    ("tests.hyperpod._cov95_provider_extra_safety", "provider-extra"),
    ("tests.orchestration._cov95_closure_extra_safety", "closure-extra"),
    ("tests.orchestration._cov95_orch_extra_safety", "orchestration extra"),
    ("tests.regional._cov95_boot_extra_safety", "BOOT extra"),
    ("tests.regional._cov95_residual_support", "residual runner"),
    ("tests.regional._cov95_notify008_resource_extra_support", "resource-only"),
    ("tests.regional._cov95_scenario_peer_safety", "scenario-peer"),
    ("tests.regional._cov95_ha011_support", "HA011 unit"),
)


def run_probe(tmp_path, *, blocked, targets, guard=None, run_tests=False):
    output = tmp_path / ("blocked.json.gz" if blocked else "available.json.gz")
    command = [
        sys.executable,
        "-m",
        "tests._optional_postgres_probe",
        "--output",
        str(output),
    ]
    if blocked:
        command.append("--block-drivers")
    if guard:
        command.extend(["--guard", guard])
    if run_tests:
        command.append("--run-tests")
    command.extend(targets)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            "HOME": str(tmp_path),
            "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.defpath)),
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "GPU_FAULT_TEST_POSTGRES_URL": "",
            "AWS_CONFIG_FILE": os.devnull,
            "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": os.devnull,
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert output.is_file(), completed.stdout + completed.stderr
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    assert completed.returncode == value["exitstatus"] == 0, (
        value["errors"][:10],
        completed.stdout[-3000:],
        completed.stderr[-3000:],
    )
    assert value["errors"] == [], "collection and guard failures cannot be ignored"
    return value


def test_entire_suite_collects_without_optional_postgres_drivers(tmp_path):
    available = run_probe(tmp_path, blocked=False, targets=["tests"])
    blocked = run_probe(tmp_path, blocked=True, targets=["tests"])
    assert available["nodeids"] and blocked["nodeids"], (
        "both probes must collect the real test tree"
    )
    assert available["phases"] == blocked["phases"] == {}, (
        "the whole-tree probe must never execute tests or native fixtures"
    )
    # This existing module parameterizes real psycopg exception classes at import.
    optional_skips = {"tests/store/test_store_error_classification.py"}
    assert set(blocked["skips"]) - set(available["skips"]) <= optional_skips, (
        "a new module-level skip must not hide an optional import regression"
    )
    skipped = set(blocked["skips"])
    expected = {
        nodeid
        for nodeid in available["nodeids"]
        if nodeid.partition("::")[0] not in skipped
    }
    assert set(blocked["nodeids"]) == expected, (
        "driver absence must not drop ordinary tests or invent another collection",
        sorted(expected - set(blocked["nodeids"]))[:10],
        sorted(set(blocked["nodeids"]) - expected)[:10],
    )


@pytest.mark.parametrize(("module", "message"), GUARDS)
@pytest.mark.parametrize(
    "blocked", [False, True], ids=["normal-imports", "blocked-imports"]
)
def test_external_io_guards_stay_active_when_drivers_are_unavailable(
    tmp_path, module, message, blocked
):
    path = tmp_path / "test_guard.py"
    path.write_text(
        "import os\n"
        "import socket\n"
        "import subprocess\n"
        "import pytest\n"
        "from scripts.e2e.regional import ha011_resources\n"
        "\n"
        "def test_guard_is_active():\n"
        f"    message = {message!r}\n"
        "    try:\n"
        "        import psycopg\n"
        "    except ImportError:\n"
        "        pass\n"
        "    else:\n"
        f"        assert not {blocked!r}, 'the driver import blocker was bypassed'\n"
        "        with pytest.raises(AssertionError, match=message):\n"
        "            psycopg.connect('forbidden-connection')\n"
        f"    if {module.endswith('_cov95_ha011_support')!r}:\n"
        "        with pytest.raises(AssertionError, match=message):\n"
        "            ha011_resources.run_fixture_command(['forbidden-command'])\n"
        "        return\n"
        "    with pytest.raises(AssertionError, match=message):\n"
        "        subprocess.run(['forbidden-command'])\n"
        "    with pytest.raises(AssertionError, match=message):\n"
        "        os.system('forbidden-command')\n"
        "    with socket.socket() as connection:\n"
        "        with pytest.raises(AssertionError, match=message):\n"
        "            connection.connect(('192.0.2.1', 1))\n"
        f"    if {not module.endswith('_cov95_notify008_resource_extra_support')!r}:\n"
        "        with pytest.raises(AssertionError, match=message):\n"
        "            open('/etc/gpu-fault/optional-driver-probe', 'rb')\n",
        encoding="utf-8",
    )
    value = run_probe(tmp_path, blocked=blocked, targets=[str(path)], guard=module)
    assert value["skips"] == {}, "missing drivers must not skip the safety fixture"
    assert len(value["nodeids"]) == len(value["phases"]) == 1
    assert next(iter(value["phases"].values())) == {
        "setup": "passed",
        "call": "passed",
        "teardown": "passed",
    }, "the actual autouse guard must run and finish in either dependency environment"


def test_actual_postgres_entrypoints_still_require_the_real_driver(tmp_path):
    path = tmp_path / "test_required.py"
    path.write_text(
        "import pytest\n"
        "from scripts import ci_postgres_grant as grant\n"
        "from scripts.perf import regional_capacity_data as data\n"
        "from scripts.e2e.regional import run_cap005_postgres_suite as cap005\n"
        "from scripts.e2e.regional.probes import state_table_snapshot as snapshot\n"
        "from scripts.e2e.regional.probes import notify008_postgres as notify\n"
        "\n"
        "def test_native_dependency_is_required():\n"
        "    calls = [\n"
        "        lambda: cap005.validate_server('postgresql://127.0.0.1:55432/postgres'),\n"
        "        lambda: grant.connection_target('', container='a' * 64, owner='', inspected={}),\n"
        "        lambda: data.run_request({}),\n"
        "        lambda: data.scope_records(None, ['perf-cap-000']),\n"
        "        lambda: snapshot.ReadOnlySchemaCursor,\n"
        "        lambda: snapshot.snapshot('workflow', verify=True),\n"
        "        lambda: notify.connect(),\n"
        "    ]\n"
        "    for call in calls:\n"
        "        with pytest.raises(ModuleNotFoundError, match='intentionally unavailable'):\n"
        "            call()\n",
        encoding="utf-8",
    )
    value = run_probe(tmp_path, blocked=True, targets=[str(path)], run_tests=True)
    assert value["skips"] == {}
    assert len(value["phases"]) == 1
    assert next(iter(value["phases"].values())) == {
        "setup": "passed",
        "call": "passed",
        "teardown": "passed",
    }, "missing dependencies must cause real errors, not synthetic native success"
