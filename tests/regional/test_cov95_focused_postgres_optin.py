from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import focused_pytest, regional_commands
from tests.regional import test_cov95_focused_pytest_contracts as contracts

COMMAND = contracts.COMMAND
NODEID = contracts.NODEID
transport = contracts.transport

DATABASE = "gpu_fault_cap005_0123456789ab"
URL = f"postgresql://127.0.0.1:55432/{DATABASE}"


@pytest.mark.parametrize(
    "url",
    [
        URL,
        f"postgres://localhost:55432/{DATABASE}",
        f"postgresql://[::1]:55432/{DATABASE}?sslmode=disable&connect_timeout=3",
        f"postgresql://127.1.2.3:65535/{DATABASE}?sslrootcert=%2Fprivate%2Fca.pem",
    ],
)
def test_explicit_generated_loopback_database_reaches_only_child_environment(
    url, transport, tmp_path
):
    supplied = {
        "PATH": "/unit/bin",
        "GPU_FAULT_STORE_URL": "postgresql://production.invalid/business",
        "GPU_FAULT_STORE_URL_FILE": "/private/production-dsn",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://foreign.invalid/business",
        "GPU_FAULT_EXECUTION_TOKEN": "unit-not-a-real-token",
        "AWS_ACCESS_KEY_ID": "unit-not-a-real-key",
        "AWS_PROFILE": "production",
        "AWS_WEB_IDENTITY_TOKEN_FILE": "/private/production-token",
        "PGHOST": "production.invalid",
        "PGSERVICE": "production",
        "PGPASSFILE": "/private/production-pgpass",
        "KUBECONFIG": "/private/production-kubeconfig",
        "PYTEST_ADDOPTS": "-k unrepresentative",
        "PYTEST_GPU_FAULT_PARTITION_COUNT": "16",
        "PYTEST_GPU_FAULT_PARTITION_INDEX": "7",
        "PYTEST_XDIST_WORKER": "gw7",
        "PYTEST_XDIST_WORKER_COUNT": "16",
        "PYTEST_CURRENT_TEST": "foreign-suite",
        "PYTEST_GPU_FAULT_CASE_REPORT": "/private/foreign-report.json",
    }
    result = regional_commands.run_fixture_command(
        COMMAND, cwd=tmp_path, env=supplied, isolated_postgres_url=url, timeout=11
    )
    assert result.returncode == 0, (
        "the explicit database must not bypass receipt verification"
    )
    command, options = transport["calls"][0]
    sent = options["environment"]
    assert sent["GPU_FAULT_TEST_POSTGRES_URL"] == url, (
        "only the explicitly supplied generated URL may survive sanitization"
    )
    assert sent["GPU_FAULT_STORE_URL"] == "", (
        "production Store selection must stay disabled"
    )
    assert (
        sent["AWS_CONFIG_FILE"] == sent["AWS_SHARED_CREDENTIALS_FILE"] == "/dev/null"
    ), "the isolated database opt-in must not restore any AWS credential file"
    assert (
        sent["KUBECONFIG"] == "/dev/null"
        and sent["AWS_EC2_METADATA_DISABLED"] == "true"
    ), "the local test must not inherit cluster or metadata-service authority"
    retained = {
        "PATH",
        "GPU_FAULT_STORE_URL",
        "GPU_FAULT_TEST_POSTGRES_URL",
        "KUBECONFIG",
        "PYTEST_GPU_FAULT_CASE_REPORT",
    }
    assert not (set(supplied) - retained) & set(sent), (
        "production credentials, PostgreSQL overrides and pytest selection state must be absent"
    )
    assert (
        sent["PYTEST_GPU_FAULT_CASE_REPORT"] != supplied["PYTEST_GPU_FAULT_CASE_REPORT"]
    ), "the verifier must allocate its own fresh receipt"
    assert url not in repr(command) and options["timeout_seconds"] == 11, (
        "the opt-in URL must stay out of argv without changing the supervised deadline"
    )


def test_inherited_generated_database_without_explicit_optin_is_still_cleared(
    transport, tmp_path
):
    result = regional_commands.run_fixture_command(
        COMMAND, cwd=tmp_path, env={"GPU_FAULT_TEST_POSTGRES_URL": URL}
    )
    assert result.returncode == 0, (
        "ordinary prerequisites still need a complete receipt"
    )
    assert (
        transport["calls"][0][1]["environment"]["GPU_FAULT_TEST_POSTGRES_URL"] == ""
    ), "a valid-looking inherited URL must not imply authorization"


@pytest.mark.parametrize(
    "url",
    [
        "",
        1,
        True,
        [],
        f"postgresql://localhost/{DATABASE}",
        f"postgresql://localhost:0/{DATABASE}",
        f"postgresql://localhost:65536/{DATABASE}",
        f"postgresql://localhost:not-a-port/{DATABASE}",
        f"postgresql://192.0.2.1:55432/{DATABASE}",
        f"postgresql://cluster.example.rds.amazonaws.com:55432/{DATABASE}",
        "postgresql://localhost:55432/postgres",
        "postgresql://localhost:55432/business",
        "postgresql://localhost:55432/gpu_fault_cap005_short",
        "postgresql://localhost:55432/gpu_fault_cap005_0123456789AB",
        f"postgresql://localhost:55432/{DATABASE}/extra",
        f"postgresql://localhost:55432/{DATABASE}%2Fextra",
        f"postgresql://localhost:55432/{DATABASE}?host=production.invalid",
        f"postgresql://localhost:55432/{DATABASE}?host=",
        f"postgresql://localhost:55432/{DATABASE}?hostaddr=192.0.2.1",
        f"postgresql://localhost:55432/{DATABASE}?%68ostaddr=",
        f"postgresql://localhost:55432/{DATABASE}?service=production",
        f"postgresql://localhost:55432/{DATABASE}?options=-c%20role=admin",
        f"postgresql://localhost:55432/{DATABASE}?sslmode=require&sslmode=disable",
        f"postgresql://localhost:55432/{DATABASE}?sslmode",
        f"postgresql://localhost:55432/{DATABASE}#fragment",
        f"postgresql://localhost:55432,production.invalid:5432/{DATABASE}",
        f"postgresql://[::1:55432/{DATABASE}",
        "\n" + URL,
        URL + "\t",
        f"mysql://localhost:55432/{DATABASE}",
    ],
)
def test_invalid_optin_is_rejected_before_identity_or_supervisor(
    url, transport, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        focused_pytest.evidence,
        "source_identity",
        lambda root: pytest.fail("invalid opt-in reached source preparation"),
    )
    with pytest.raises(ValueError, match="isolated PostgreSQL opt-in"):
        regional_commands.run_fixture_command(
            COMMAND, cwd=tmp_path, isolated_postgres_url=url
        )
    assert transport["calls"] == [], "invalid database scope must not start a child"


@pytest.mark.parametrize(
    "command",
    [
        ["make", "test-postgres-stress"],
        ["kubectl", "exec", "pod", "--", "python", "-m", "pytest"],
        ["sh", "-c", "python -m pytest"],
    ],
)
def test_nonpytest_command_cannot_request_the_local_pytest_database_optin(
    command, transport, tmp_path
):
    with pytest.raises(ValueError, match="only valid for direct local pytest"):
        regional_commands.run_fixture_command(
            command, cwd=tmp_path, isolated_postgres_url=URL
        )
    assert transport["calls"] == [], (
        "the opt-in must not become a generic environment escape"
    )


@pytest.mark.parametrize(
    "defect", ["missing", "failed", "phase", "discovery", "source"]
)
def test_database_optin_does_not_relax_any_receipt_proof(defect, transport, tmp_path):
    value = transport["value"]
    if defect == "missing":
        transport["value"] = None
    elif defect == "failed":
        value["records"][NODEID]["status"] = "FAIL"
    elif defect == "phase":
        del value["records"][NODEID]["phases"]["teardown"]
    elif defect == "discovery":
        value["session"]["discovered_nodeids"].append("test_unit.py::test_unexecuted")
    else:
        transport["source_changed"] = True
    result = regional_commands.run_fixture_command(
        COMMAND, cwd=tmp_path, isolated_postgres_url=URL, check=False
    )
    assert (
        result.returncode == 1 and "focused pytest evidence rejected" in result.stderr
    ), "the isolated database cannot excuse incomplete or drifted test evidence"
    assert (
        transport["calls"][0][1]["environment"]["GPU_FAULT_TEST_POSTGRES_URL"] == URL
    ), "the negative case must exercise the real explicit opt-in"


def test_optin_url_never_enters_prepared_repr_or_validation_errors(transport, tmp_path):
    marker = "unit-private-url-marker"
    url = f"postgresql://unit:{marker}@localhost:55432/{DATABASE}"
    with focused_pytest.prepare_focused_pytest(
        COMMAND, cwd=tmp_path, environment={}, isolated_postgres_url=url
    ) as prepared:
        assert (
            prepared is not None
            and prepared.environment["GPU_FAULT_TEST_POSTGRES_URL"] == url
        ), "the protected URL belongs only in the child environment"
        assert marker not in repr(prepared) and url not in repr(prepared), (
            "dataclass diagnostics must not reveal the authorized URL or its credentials"
        )
        assert url not in repr(prepared.command), (
            "the URL must never become a command argument"
        )
    invalid = f"postgresql://unit:{marker}@production.invalid:5432/{DATABASE}"
    with pytest.raises(ValueError) as raised:
        focused_pytest.validate_isolated_postgres_url(invalid)
    assert marker not in str(raised.value) + repr(raised.value), (
        "URL validation errors must contain only the safe scope diagnostic"
    )


@pytest.mark.parametrize("failure", ["timeout", "deadline", "supervision", "spawn"])
def test_optin_keeps_supervision_timeout_and_receipt_cleanup(
    failure, transport, tmp_path, monkeypatch
):
    reports = []
    markers = []
    error = {
        "timeout": subprocess.TimeoutExpired(COMMAND, 3),
        "deadline": DeploymentDeadlineExceeded("unit deadline"),
        "supervision": ProcessSupervisionLost("unit supervision loss"),
        "spawn": OSError(5, "unit-private-startup-detail"),
    }[failure]

    def execute(command, **options):
        assert options["environment"]["GPU_FAULT_TEST_POSTGRES_URL"] == URL, (
            "the failing child must still use only the explicitly scoped database"
        )
        reports.append(Path(options["environment"]["PYTEST_GPU_FAULT_CASE_REPORT"]))
        raise error

    monkeypatch.setattr(regional_commands, "run_command", execute)
    monkeypatch.setattr(
        regional_commands, "record_supervision_loss", lambda: markers.append("lost")
    )
    expected = (
        ProcessSupervisionLost
        if failure == "supervision"
        else regional_commands.RegionalCommandTimeout
        if failure in {"timeout", "deadline"}
        else regional_commands.RegionalFixtureError
    )
    with pytest.raises(expected) as raised:
        regional_commands.run_fixture_command(
            COMMAND, cwd=tmp_path, isolated_postgres_url=URL, timeout=3
        )
    assert markers == (["lost"] if failure == "supervision" else []), (
        "a database opt-in must preserve the supervisor ownership-loss marker"
    )
    assert len(reports) == 1 and not reports[0].parent.exists(), (
        "transport failure must close the owned temporary receipt directory"
    )
    assert "unit-private" not in str(raised.value), (
        "startup diagnostics must retain the public boundary's redaction"
    )
