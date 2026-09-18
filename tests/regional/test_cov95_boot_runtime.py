from __future__ import annotations

import copy
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs

import pytest

from scripts.e2e.regional import boot_acceptance_runtime as runtime
from tests.regional._cov95_boot_owner import OwnerFixture
from tests.regional._cov95_cap_cases import Clock
from tests.regional.test_boot_acceptance_behavior import RuntimeFixture


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_amp_request_signs_the_scoped_request_and_uses_a_bounded_transport(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    signed = []
    requests = []
    credentials = object()
    monkeypatch.setattr(
        runtime,
        "boto3",
        SimpleNamespace(
            Session=lambda: SimpleNamespace(get_credentials=lambda: credentials)
        ),
    )

    class Signer:
        def __init__(self, value: Any, service: str, region: str) -> None:
            assert value is credentials
            signed.append((service, region))

        def add_auth(self, request: Any) -> None:
            request.headers["Authorization"] = "unit-signed"

    def transport(request: Any, *, timeout: int) -> io.StringIO:
        requests.append((request, timeout))
        return io.StringIO('{"status":"success","data":{"result":[]}}')

    monkeypatch.setattr(runtime, "SigV4Auth", Signer)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", transport)
    result = runtime.amp_request(
        region="us-east-1",
        workspace_id="unit-workspace",
        method=method,
        path="/api/v1/query",
        parameters={"query": "sum(metric) > 1"},
    )
    assert result == {"status": "success", "data": {"result": []}}
    assert signed == [("aps", "us-east-1")]
    request, timeout = requests[0]
    assert (
        request.full_url
        == "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/unit-workspace/api/v1/query"
    )
    assert timeout == 30
    assert request.get_header("Authorization") == "unit-signed"
    if method == "POST":
        assert parse_qs(request.data.decode()) == {"query": ["sum(metric) > 1"]}
    else:
        assert request.data is None


def test_amp_missing_credentials_stops_before_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "boto3",
        SimpleNamespace(Session=lambda: SimpleNamespace(get_credentials=lambda: None)),
    )
    monkeypatch.setattr(
        runtime.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail(
            "missing credentials reached a transport"
        ),
    )
    with pytest.raises(
        runtime.BootAcceptanceError, match="credentials are unavailable"
    ):
        runtime.amp_request(
            region="us-east-1", workspace_id="unit", method="GET", path="/api/v1/alerts"
        )


@pytest.mark.parametrize(
    "fault", ["missing-ca", "missing-volume", "writable", "missing-secret"]
)
def test_ca_contract_refuses_missing_or_untrusted_mounts(fault: str) -> None:
    manifest = runtime.manifest_deployment()
    live = copy.deepcopy(manifest)
    spec = live["spec"]["template"]["spec"]
    container = spec["containers"][0]
    if fault == "missing-ca":
        container["env"] = [
            item for item in container["env"] if item["name"] != runtime.CA_FILE_ENV
        ]
    elif fault == "missing-volume":
        spec["volumes"] = []
    elif fault == "writable":
        for mount in container["volumeMounts"]:
            mount["readOnly"] = False
    else:
        for volume in spec["volumes"]:
            if "secret" in volume:
                volume["secret"]["secretName"] = ""
    assert runtime.ca_file_contract(manifest, live)["passed"] is False


def test_secret_contract_rejects_undocumented_keys_and_indented_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = runtime.manifest_deployment()
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    container["command"] = ["python3", "-c"]
    container["args"] = [" import example"]
    manual = tmp_path / "manual.txt"
    manual.write_text("no required key declarations")
    monkeypatch.setattr(runtime, "manifest_deployment", lambda: manifest)
    monkeypatch.setattr(runtime, "OPERATIONS_MANUAL", manual)
    result = runtime.secret_key_contract()
    assert result["passed"] is False
    assert result["missing"], "undocumented required keys were not reported"
    assert result["python_c_leading_whitespace"] == [container["name"]]


@pytest.mark.parametrize("fault", ["readiness", "matrix", "empty-ca"])
def test_tls_runner_records_the_actual_failed_probe_and_diagnostic(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    fixture = RuntimeFixture()
    original = fixture.exec

    def execute(
        plane: str, pod: str, *args: str, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        match = (
            fault == "readiness"
            and args[0] == "gpu-fault-cluster-executor-readiness"
            or fault == "matrix"
            and kwargs.get("input_text") == runtime.READINESS_PROBE
            or fault == "empty-ca"
            and args[0] == "sh"
        )
        if match:
            return subprocess.CompletedProcess(
                args, 2, "", "synthetic unexpected failure"
            )
        return original(plane, pod, *args, **kwargs)

    monkeypatch.setattr(fixture, "exec", execute)
    result = runtime.run_boot012(fixture)
    assert result["verdict"] == "FAIL"
    key = {
        "readiness": "readiness_stderr_tail",
        "matrix": "stale_probe_stderr_tail",
        "empty-ca": "empty_ca_stderr_tail",
    }[fault]
    assert all(
        item[key] == "synthetic unexpected failure" for item in result["replicas"]
    ), "failed replica probe lost its diagnostic"


@pytest.mark.parametrize(
    "fault", ["", "metrics", "leased", "readiness", "alert", "delete"]
)
def test_owner_probe_pass_requires_metric_lease_alert_and_cleanup_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    clock = Clock()
    fixture = OwnerFixture(clock, failure=fault)
    monkeypatch.setattr(runtime, "time", clock)
    monkeypatch.setattr(runtime, "amp_request", fixture.amp_request)
    monkeypatch.setattr(
        runtime,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    result = runtime.run_boot015(fixture, case_dir=tmp_path, attempt=2)
    assert result["verdict"] == ("FAIL" if fault else "PASS"), (
        f"owner probe verdict must reflect the observed failure: {fault!r}"
    )
    saved = json.loads((tmp_path / "boot015-details.json").read_text())
    assert saved["verdict"] == result["verdict"], (
        "persisted owner evidence must match the returned verdict"
    )
    assert fixture.probes.index("delete") > fixture.probes.index("insert"), (
        "cleanup must follow the owned synthetic insertion"
    )
    assert fixture.probes.count("metrics") == 2, (
        "the owner probe requires independent before/after metric samples"
    )
    assert fixture.metric_targets == [
        ("cpu", "gpu-fault-control-worker", fixture.cluster_id),
        ("cpu", "gpu-fault-control-worker", fixture.cluster_id),
    ], "both metric samples must use the worker role and the bound cluster"
    assert saved["cleanup"]["synthetic_command_removed"] is (fault != "delete"), (
        "failed deletion cannot be reported as confirmed removal"
    )
    if not fault:
        assert saved["cleanup"]["passed"] is True, (
            "successful cleanup must verify removal, alert resolution and readiness"
        )
        assert fixture.rows == {"existing-command"}, (
            "cleanup must preserve the original command inventory"
        )
    if fault == "leased":
        assert result["checks"]["never_leased"] is False, (
            "an observed lease must invalidate the unclaimed-command proof"
        )


@pytest.mark.parametrize("fault", ["owners", "profile", "baseline"])
def test_owner_probe_rejects_invalid_prerequisites_before_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    fixture = OwnerFixture(Clock(), failure=fault)
    monkeypatch.setattr(
        runtime,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    with pytest.raises(runtime.BootAcceptanceError):
        runtime.run_boot015(fixture, case_dir=tmp_path, attempt=1)
    assert "insert" not in fixture.probes
    assert fixture.rows == {"existing-command"}
