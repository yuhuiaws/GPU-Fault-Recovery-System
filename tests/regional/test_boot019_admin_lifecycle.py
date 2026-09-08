"""BOOT-019 runner contracts from the 2026-09-07 review.

The captured cluster token used to be copied to ``run_dir/secure`` and deleted
only on success; the uninstall checks compared literals the uninstall code
writes unconditionally; and every check was a bare ``assert`` that ``python -O``
would strip.
"""

from __future__ import annotations

import ast
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_boot019_admin_lifecycle as boot019
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"


class FakeBackend:
    def __init__(
        self, *, revoked_status: int = 403, uninstall: dict[str, Any] | None = None
    ):
        self.cluster_ids = {"cluster-a"}
        self.calls: list[str] = []
        self.revoked_status = revoked_status
        self.uninstall_result = uninstall or {
            "cpu_cluster": "keep",
            "registry_entries_preserved": 2,
            "final_registry_statuses": {
                "cluster/cluster-a/eks": "PRESERVED",
                boot019.AURORA_CLUSTER_RESOURCE_KEY: "PRESERVED",
            },
        }

    def snapshot(self) -> dict[str, Any]:
        values = sorted(self.cluster_ids)
        return {
            "site_cluster_ids": values,
            "registry_secret_cluster_ids": values,
            "release_state_cluster_ids": values,
            "installation_registry_cluster_ids": values,
            "cpu_control_plane_ready": True,
        }

    def join(self, fault: str | None = None) -> dict[str, Any]:
        self.calls.append(f"join:{fault}")
        if fault == "before-site-commit":
            return {"phase": "ROLLED_BACK", "cluster_id": "cluster-b"}
        self.cluster_ids.add("cluster-b")
        if fault == "after-site-commit":
            return {"phase": "FAILED_AFTER_COMMIT", "cluster_id": "cluster-b"}
        return {"phase": "COMPLETED", "cluster_id": "cluster-b"}

    def capture_joined_token(self, cluster_id: str) -> dict[str, Any]:
        self.calls.append("capture")
        return {"cluster_id": cluster_id, "token_storage": "memory"}

    def remove(self, cluster_id: str) -> dict[str, Any]:
        self.calls.append(f"remove:{cluster_id}")
        self.cluster_ids.remove(cluster_id)
        return {
            "phase": "COMPLETED",
            "cluster_id": cluster_id,
            "remaining_cluster_ids": sorted(self.cluster_ids),
        }

    def probe_revoked_token(self, capture: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("probe")
        return {
            "status": self.revoked_status,
            "detail": "regional cluster authentication failed",
        }

    def uninstall(self) -> dict[str, Any]:
        self.calls.append("uninstall")
        return dict(self.uninstall_result)

    def cleanup_sensitive_files(self) -> None:
        self.calls.append("cleanup")


def _recorder(tmp_path: Path) -> EvidenceRecorder:
    return EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-019.json",
        case_id="GF-REGIONAL-BOOT-019",
        inputs={"site": "test"},
    )


def test_lifecycle_completes_and_cleans_up_last(tmp_path: Path) -> None:
    backend = FakeBackend()

    result = boot019.run_admin_lifecycle(backend, _recorder(tmp_path))

    assert result["status"] == "COMPLETED"
    assert backend.calls[-2:] == ["uninstall", "cleanup"]
    assert backend.calls.count("cleanup") == 1


def test_sensitive_cleanup_runs_when_a_check_fails(tmp_path: Path) -> None:
    backend = FakeBackend(revoked_status=200)

    with pytest.raises(boot019.AcceptanceCheckError, match="revoked token status"):
        boot019.run_admin_lifecycle(backend, _recorder(tmp_path))

    assert backend.calls[-1] == "cleanup"
    document = json.loads((tmp_path / "GF-REGIONAL-BOOT-019.json").read_text())
    assert document["status"] == "FAILED"


def test_uninstall_assertions_read_live_values_not_literals() -> None:
    boot019.assert_uninstall_result(
        {
            "cpu_cluster": "keep",
            "registry_entries_preserved": 3,
            "final_registry_statuses": {
                "a": "PRESERVED",
                "b": "DELETED",
                boot019.AURORA_CLUSTER_RESOURCE_KEY: "PRESERVED",
            },
        }
    )
    with pytest.raises(boot019.AcceptanceCheckError, match="preserve the Aurora"):
        boot019.assert_uninstall_result(
            {
                "registry_entries_preserved": 2,
                "final_registry_statuses": {
                    "a": "PRESERVED",
                    boot019.AURORA_CLUSTER_RESOURCE_KEY: "DELETED",
                },
            }
        )
    with pytest.raises(boot019.AcceptanceCheckError, match="fewer than 2"):
        boot019.assert_uninstall_result(
            {"registry_entries_preserved": 1, "final_registry_statuses": {"a": "X"}}
        )
    with pytest.raises(boot019.AcceptanceCheckError, match="DELETE_PENDING"):
        boot019.assert_uninstall_result(
            {
                "registry_entries_preserved": 2,
                "final_registry_statuses": {
                    "a": "PRESERVED",
                    "b": "DELETE_PENDING",
                    boot019.AURORA_CLUSTER_RESOURCE_KEY: "PRESERVED",
                },
            }
        )
    with pytest.raises(
        boot019.AcceptanceCheckError, match="no final registry statuses"
    ):
        boot019.assert_uninstall_result(
            {
                "cpu_cluster": "keep",
                "gpu_clusters": "preserved",
                "delete_policy_residuals": 0,
                "registry_entries_preserved": 2,
            }
        )


def test_final_registry_statuses_come_from_the_snapshot_file(tmp_path: Path) -> None:
    registry = tmp_path / "installation-resources.json"
    registry.write_text(
        json.dumps(
            {
                "resources": [
                    {"resource_key": "cluster/a/eks", "status": "PRESERVED"},
                    {"resource_key": "aurora/x", "status": "DELETED"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert boot019.final_registry_statuses({"final_registry": str(registry)}) == {
        "cluster/a/eks": "PRESERVED",
        "aurora/x": "DELETED",
    }
    with pytest.raises(boot019.AcceptanceCheckError, match="names no final registry"):
        boot019.final_registry_statuses({})


def _live_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    token_file = tmp_path / "cluster-b.token"
    token_file.write_bytes(b"t" * 40 + b"\n")
    backend = boot019.LiveAdminLifecycleBackend(
        site_path=tmp_path / "site.yaml",
        gpu_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/gpu",
        cluster_id=None,
        allowed_namespaces=("default",),
        join_state_dir=tmp_path / "join",
        run_dir=tmp_path / "run",
    )
    site = SimpleNamespace(
        release_config={
            "clusters": [
                {
                    "cluster_id": "cluster-b",
                    "token_file": str(token_file),
                    "control_plane_url": "https://control.example.invalid",
                    "ca_file": str(tmp_path / "ca.pem"),
                }
            ]
        }
    )
    monkeypatch.setattr(backend, "_site", lambda: site)
    return backend


def test_live_backend_keeps_the_captured_token_in_memory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _live_backend(tmp_path, monkeypatch)

    capture = backend.capture_joined_token("cluster-b")

    assert capture["token_storage"] == "memory"
    assert "token_path" not in capture
    assert not list((tmp_path / "run").rglob("*revoked-token*")), (
        "the revoked token never reaches the run directory on disk"
    )

    seen: dict[str, str] = {}

    def fake_urlopen(request: Any, **_kwargs: Any) -> Any:
        seen["authorization"] = request.get_header("Authorization")
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"detail": "regional cluster authentication failed"}'),
        )

    monkeypatch.setattr(boot019.ssl, "create_default_context", lambda **k: None)
    monkeypatch.setattr(boot019.urllib.request, "urlopen", fake_urlopen)

    revoked = backend.probe_revoked_token(capture)

    assert revoked == {
        "status": 403,
        "detail": "regional cluster authentication failed",
    }
    assert seen["authorization"] == "Bearer " + "t" * 40

    backend.cleanup_sensitive_files()
    with pytest.raises(boot019.AcceptanceCheckError, match="no captured token"):
        backend.probe_revoked_token(capture)


@pytest.mark.parametrize(
    "name",
    [
        "run_boot019_admin_lifecycle.py",
        "run_boot020_release_rolling.py",
        "audit_executor_readiness.py",
    ],
)
def test_runner_checks_survive_python_optimisation(name: str) -> None:
    """``python -O`` drops ``assert``; a check that vanishes is a constant PASS."""

    tree = ast.parse((REGIONAL / name).read_text(encoding="utf-8"))
    asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == [], f"{name} still uses bare assert at lines {asserts}"
