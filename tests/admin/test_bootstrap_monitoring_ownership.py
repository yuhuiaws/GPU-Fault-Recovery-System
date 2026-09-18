from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import bootstrap_services
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_platform_probes import release_owns_monitoring
from gpu_fault_release.regional_dataplane_observability import amp_installer_environment
from gpu_fault_release.regional_release_config import amp_writer_role_name


class StateRunner(CommandRunner):
    def __init__(self, response: str | Exception) -> None:
        super().__init__()
        self.response = response

    def run(self, arguments, **kwargs):
        assert "--ignore-not-found" in arguments
        assert not kwargs.get("mutate"), "monitoring ownership must be read-only"
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (None, False),
        ("bootstrap-failed", False),
        ("bootstrap-unknown", True),
        ("complete", True),
        ("preflight", True),
        ("rolled-back", True),
        ("partial-convergence", True),
    ],
)
def test_only_initial_bootstrap_owns_versioned_monitoring(
    tmp_path: Path, phase: str | None, expected: bool
) -> None:
    raw = (
        json.dumps(
            {"data": {"state.json": json.dumps({"phase": phase, "release_id": "a"})}}
        )
        if phase
        else ""
    )

    assert (
        release_owns_monitoring(
            StateRunner(raw), cpu_kubeconfig=tmp_path / "cpu", namespace="system"
        )
        is expected
    )


@pytest.mark.parametrize(
    "raw",
    [
        "broken",
        "{}",
        "[]",
        '{"data":{}}',
        '{"data":{"state.json":""}}',
        '{"data":{"state.json":"broken"}}',
        '{"data":{"state.json":"{}"}}',
        '{"data":{"state.json":"[]"}}',
        json.dumps({"data": {"state.json": json.dumps({"phase": "complete"})}}),
        json.dumps(
            {"data": {"state.json": json.dumps({"phase": 1, "release_id": "a"})}}
        ),
    ],
)
def test_unknown_release_state_never_authorizes_monitoring_mutation(
    tmp_path: Path, raw: str
) -> None:
    with pytest.raises(BootstrapError, match="cannot determine monitoring owner"):
        release_owns_monitoring(
            StateRunner(raw), cpu_kubeconfig=tmp_path / "cpu", namespace="system"
        )


def test_unreadable_state_is_not_treated_as_first_bootstrap(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="AccessDenied"):
        release_owns_monitoring(
            StateRunner(BootstrapError("AccessDenied")),
            cpu_kubeconfig=tmp_path / "cpu",
            namespace="system",
        )


@pytest.mark.parametrize("site_name", ["site-test", "long-site-" * 8])
def test_bootstrap_prepares_one_registered_role_without_installing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site_name: str
) -> None:
    roles = []
    associations = []

    def ensure_role(_runner, **kwargs):
        roles.append(kwargs["role_name"])
        return {
            "role_arn": f"arn:aws:iam::123456789012:role/{kwargs['role_name']}",
            "role_name": kwargs["role_name"],
            "ownership": "CREATED",
        }

    def ensure_association(_runner, **kwargs):
        associations.append(kwargs)
        return {
            "association_id": "association",
            "ownership": "CREATED",
            "cluster_name": "cpu",
            "namespace": "system",
            "service_account": "gpu-fault-adot",
        }

    monkeypatch.setattr(bootstrap_services, "_ensure_role", ensure_role)
    monkeypatch.setattr(bootstrap_services, "_pod_identity_trust", lambda _cpu: {})
    monkeypatch.setattr(
        bootstrap_services, "_ensure_pod_identity_association", ensure_association
    )
    monkeypatch.setenv("IAM_ROLE_NAME", "unrelated-region-role")
    monkeypatch.setenv("SERVICE_ACCOUNT", "unrelated-account")
    cpu = SimpleNamespace(region="us-east-1", account_id="123456789012", eks_name="cpu")
    monitoring = {
        "workspace_id": "ws-test",
        "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:alerts",
    }
    result = bootstrap_services.install_monitoring(
        StateRunner(AssertionError("bootstrap must not run the runtime installer")),
        repository_root=tmp_path,
        cpu=cpu,
        cpu_kubeconfig=tmp_path / "cpu",
        namespace="system",
        site_id=site_name,
        monitoring=monitoring,
        adot_image="",
        alert_email=None,
        runtime_managed_by_release=True,
    )
    environment = amp_installer_environment(
        SimpleNamespace(
            adot_image="registry/adot@sha256:" + "1" * 64,
            config=SimpleNamespace(
                site_name=site_name,
                aws_region=cpu.region,
                namespace="system",
                cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
                cpu_kubeconfig=str(tmp_path / "cpu"),
                notifications=SimpleNamespace(admin_email=None),
                health=SimpleNamespace(
                    amp_workspace_id="ws-test",
                    sns_topic_arn=monitoring["sns_topic_arn"],
                    amp_rule_namespace="rules",
                    require_confirmed_sns_subscription=True,
                ),
            ),
        )
    )
    assert roles == [amp_writer_role_name(site_name)], (
        "bootstrap prepared more than the single registered site role"
    )
    assert environment["IAM_ROLE_NAME"] == result["role_name"], (
        "release runtime installation would switch the bootstrap role"
    )
    assert environment["SERVICE_ACCOUNT"] == result["service_account"], (
        "the runtime installer inherited a foreign service account"
    )
    assert (
        len(associations) == 1 and associations[0]["role_arn"] == result["role_arn"]
    ), "the recorded Pod Identity association does not match the prepared role"
    assert result["runtime_managed_by_release"] is True, (
        "bootstrap claimed ownership of versioned monitoring"
    )
