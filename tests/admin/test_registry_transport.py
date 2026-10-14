from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin import resource_registry_scripts as scripts
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file


def snapshot(site_id: str = "test-site") -> InstallationResourceSnapshot:
    value = InstallationResourceSnapshot(
        site_id=site_id,
        resources=[
            InstallationResource(
                site_id=site_id,
                resource_key="aws/test/resource",
                resource_type="test_resource",
                resource_id="test-resource",
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
            )
        ],
    )
    return value.model_copy(update={"source_sha256": value.digest()})


@pytest.mark.parametrize(
    "operation",
    [
        registry.sync_installation_resource_snapshot,
        registry.sync_installation_resource_snapshot_direct,
    ],
)
def test_registry_snapshot_is_transferred_on_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: Any
) -> None:
    site = load_site(site_file(tmp_path))
    value = snapshot()
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(registry, "_cpu_pod", lambda _site: "cpu-test")

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        response = (
            "1"
            if arguments[-1] == scripts.DIRECT_SYNC_SCRIPT
            else json.dumps([item.model_dump(mode="json") for item in value.resources])
        )
        return subprocess.CompletedProcess(arguments, 0, response, "")

    monkeypatch.setattr(registry, "run_command", run)
    operation(site, value)
    arguments, kwargs = calls[0]
    assert arguments[arguments.index("exec") + 1 :][:2] == ["-i", "cpu-test"]
    assert not any("GPU_FAULT_INSTALLATION_SNAPSHOT" in item for item in arguments), (
        "registry snapshot must not be passed in command arguments"
    )
    assert json.loads(kwargs["input_text"]) == value.model_dump(mode="json")
    assert kwargs["timeout_seconds"] > 0


@pytest.mark.parametrize(
    "operation",
    [
        registry.sync_installation_resource_snapshot,
        registry.sync_installation_resource_snapshot_direct,
    ],
)
def test_registry_sync_rejects_foreign_site_before_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: Any
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(
        registry,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("foreign snapshot reached a command"),
    )
    with pytest.raises(BootstrapError, match="another site"):
        operation(site, snapshot("different-site"))


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (1, "", "Forbidden: backend 404 Not Found"),
        (1, "", "credential helper failed: HTTP Error 503"),
        (44, "", scripts.LEGACY_REGISTRY_MARKER),
        (75, scripts.UNAVAILABLE_REGISTRY_MARKER + " extra", ""),
    ],
)
def test_registry_sync_cannot_infer_a_fallback_from_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(registry, "_cpu_pod", lambda _site: "cpu-test")
    monkeypatch.setattr(
        registry,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, returncode, stdout, stderr
        ),
    )
    monkeypatch.setattr(
        registry,
        "sync_installation_resource_snapshot_direct",
        lambda *_args: pytest.fail("an unproven outage bypassed the registry API"),
    )
    with pytest.raises(BootstrapError, match="synchronization failed"):
        registry.sync_installation_resource_snapshot(site, snapshot())


@pytest.mark.parametrize("script", [scripts.FETCH_SCRIPT, scripts.SYNC_SCRIPT])
def test_registry_script_reports_confirmed_legacy_api_on_machine_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], script: str
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b'{"paths":{}}')
    )
    with pytest.raises(SystemExit) as stopped:
        exec(script, {})
    assert stopped.value.code == 44
    assert capsys.readouterr().out.strip() == scripts.LEGACY_REGISTRY_MARKER


@pytest.mark.parametrize("status", [403, 404, 503])
def test_registry_script_does_not_expose_http_errors_or_reclassify_missing_endpoints(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(
            "https://example.invalid/private",
            status,
            "private-body-sentinel",
            Message(),
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(SystemExit) as stopped:
        exec(scripts.FETCH_SCRIPT, {})
    captured = capsys.readouterr()
    assert "private-body-sentinel" not in captured.out + captured.err
    assert "example.invalid" not in captured.out + captured.err
    if status == 503:
        assert stopped.value.code == 75
        assert captured.out.strip() == scripts.UNAVAILABLE_REGISTRY_MARKER
    else:
        assert stopped.value.code == 1
        assert captured.out == ""


def test_sync_script_consumes_stdin_only_after_proving_the_route(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value = snapshot().model_dump_json()
    requests: list[object] = []

    def respond(request: object, **_kwargs: object) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO(
            b'{"paths":{"/v1/installation-resources/sync":{}}}'
            if isinstance(request, str)
            else b'{"saved":1}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "test-execution-credential")
    monkeypatch.setattr(
        sys, "stdin", io.TextIOWrapper(io.BytesIO(value.encode()), encoding="utf-8")
    )
    exec(scripts.SYNC_SCRIPT, {})
    assert len(requests) == 2
    request = requests[1]
    assert isinstance(request, urllib.request.Request), (
        "registry sync must build an HTTP Request after route proof"
    )
    assert request.data == value.encode()
    assert json.loads(capsys.readouterr().out) == {"saved": 1}


def test_bootstrap_state_is_bound_to_current_cpu_when_identity_exists(
    tmp_path: Path,
) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    state: dict[str, object] = {
        "site_id": "test-site",
        "resources": {
            "initial_deploy_target": {
                "cpu": {
                    "eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/other",
                    "hyperpod_name": "control",
                }
            }
        },
    }
    (tmp_path / "bootstrap-state.json").write_text(json.dumps(state))
    with pytest.raises(BootstrapError, match="CPU identity"):
        registry.find_bootstrap_state(site)


def test_foreign_local_bootstrap_state_cannot_fall_back_to_legacy_discovery(
    tmp_path: Path,
) -> None:
    site = load_site(site_file(tmp_path))
    (tmp_path / "bootstrap-state.json").write_text(
        json.dumps({"site_id": "other-site", "resources": {}})
    )
    with pytest.raises(BootstrapError, match="another site"):
        registry.find_bootstrap_state(site)


def test_invalid_registry_rows_do_not_echo_sensitive_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    value = snapshot().resources[0].model_dump(mode="json")
    value["attributes"] = {"password": "opaque-private-fixture"}
    monkeypatch.setattr(registry, "_cpu_pod", lambda _site: "cpu-test")
    monkeypatch.setattr(
        registry,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, json.dumps([value]), ""
        ),
    )
    with pytest.raises(BootstrapError, match="invalid resources") as rejected:
        registry.fetch_installation_resource_registry(site)
    assert "opaque-private-fixture" not in str(rejected.value)
    assert not (tmp_path / "installation-resources.json").exists(), (
        "invalid registry rows must not be persisted locally"
    )


@pytest.mark.parametrize("response", ["[]", "{}", "", "null"])
def test_registry_sync_requires_complete_success_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(registry, "_cpu_pod", lambda _site: "cpu-test")
    monkeypatch.setattr(
        registry,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, response, ""
        ),
    )
    with pytest.raises(BootstrapError, match="registry"):
        registry.sync_installation_resource_snapshot(site, snapshot())


def _replaced(value: InstallationResourceSnapshot) -> InstallationResourceSnapshot:
    changed = value.model_copy(
        update={
            "resources": [
                item.model_copy(update={"resource_id": "replaced-resource"})
                for item in value.resources
            ]
        }
    )
    return changed.model_copy(update={"source_sha256": changed.digest()})


def test_a_registry_api_conflict_names_key_and_field_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch 91: the sync API answers an identity conflict with 409; the admin
    side names the key and field from its body, writes nothing and does not
    retry or fall back to the direct path (the refusal is deterministic)."""

    from tests.admin._registry_pod_support import PodRegistry

    site = load_site(site_file(tmp_path))
    pod = PodRegistry(monkeypatch)
    original = snapshot()
    registry.sync_installation_resource_snapshot(site, original)
    monkeypatch.setattr(
        registry,
        "sync_installation_resource_snapshot_direct",
        lambda *_args: pytest.fail("a deterministic conflict reached the fallback"),
    )

    with pytest.raises(registry.InstallationRegistryConflict) as refused:
        registry.sync_installation_resource_snapshot(site, _replaced(original))

    assert refused.value.conflicts == [("aws/test/resource", ["resource_id"])]
    assert isinstance(refused.value, BootstrapError), "every caller fails closed"
    assert (
        "installation resource identity cannot change: aws/test/resource "
        "(resource_id)" in str(refused.value)
    )
    assert pod.scripts == [scripts.SYNC_SCRIPT, scripts.SYNC_SCRIPT], (
        "the conflicting sync was retried"
    )
    assert pod.store.list_installation_resources("test-site") == original.resources


def _conflict_error(body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/private", 409, "Conflict", Message(), io.BytesIO(body)
    )


def test_sync_script_reports_a_registry_conflict_on_machine_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    body = {
        "detail": "private-body-sentinel",
        "reason": "installation resource identity cannot change",
        "conflicts": [{"resource_key": "aws/nlb", "fields": ["dependencies"]}],
    }

    def respond(request: object, **_kwargs: object) -> io.BytesIO:
        if isinstance(request, str):
            return io.BytesIO(b'{"paths":{"/v1/installation-resources/sync":{}}}')
        raise _conflict_error(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "test-execution-credential")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"{}")))
    with pytest.raises(SystemExit) as stopped:
        exec(scripts.SYNC_SCRIPT, {})
    captured = capsys.readouterr()
    assert stopped.value.code == scripts.CONFLICT_EXIT
    assert "private-body-sentinel" not in captured.out + captured.err
    assert scripts.conflict_report(stopped.value.code, captured.out) == (
        "installation resource identity cannot change",
        [("aws/nlb", ["dependencies"])],
    )


@pytest.mark.parametrize(
    "body",
    [
        b'{"detail": "private-body-sentinel"}',
        b"private-body-sentinel",
        b'{"reason": "private-body-sentinel", "conflicts": '
        b'[{"resource_key": "aws/nlb", "fields": ["resource_id"]}]}',
        b'{"reason": "installation resource identity cannot change", "conflicts": '
        b'[{"resource_key": "private body sentinel", "fields": ["resource_id"]}]}',
        b'{"reason": "installation resource identity cannot change", "conflicts": '
        b'[{"resource_key": "aws/nlb", "fields": ["private-body-sentinel"]}]}',
        b'{"reason": "installation resource identity cannot change", "conflicts": []}',
    ],
)
def test_sync_script_keeps_an_unrecognized_409_body_private(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], body: bytes
) -> None:
    def respond(request: object, **_kwargs: object) -> io.BytesIO:
        if isinstance(request, str):
            return io.BytesIO(b'{"paths":{"/v1/installation-resources/sync":{}}}')
        raise _conflict_error(body)

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "test-execution-credential")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"{}")))
    with pytest.raises(SystemExit) as stopped:
        exec(scripts.SYNC_SCRIPT, {})
    captured = capsys.readouterr()
    assert stopped.value.code == 1
    assert captured.out == ""
    assert captured.err.strip() == "registry HTTP request failed: 409"


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, scripts.CONFLICT_REGISTRY_MARKER + ' {"reason": "x", "conflicts": []}'),
        (scripts.CONFLICT_EXIT, "GPU_FAULT_OTHER_MARKER {}"),
        (scripts.CONFLICT_EXIT, scripts.CONFLICT_REGISTRY_MARKER + " not-json"),
        (
            scripts.CONFLICT_EXIT,
            scripts.CONFLICT_REGISTRY_MARKER
            + ' {"reason": "installation resource identity cannot change",'
            + ' "conflicts": [{"resource_key": "aws/a", "fields": []}]}',
        ),
        (
            scripts.CONFLICT_EXIT,
            "extra line\n"
            + scripts.CONFLICT_REGISTRY_MARKER
            + ' {"reason": "installation resource identity cannot change",'
            + ' "conflicts": [{"resource_key": "aws/a", "fields": ["region"]}]}',
        ),
    ],
)
def test_only_the_exact_conflict_protocol_is_a_conflict(
    returncode: int, stdout: str
) -> None:
    assert scripts.conflict_report(returncode, stdout) is None
