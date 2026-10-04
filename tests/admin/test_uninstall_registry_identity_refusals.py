"""Uninstall refusals around the registry rows and the CPU cluster deletion.

The uninstall reads the installation resource registry before mutating
anything. These tests pin the refusal when the legacy LBC attachment rows
carry impossible resource types, and the fail-closed stop when a CPU cluster
the uninstall just asked AWS to delete is still reported as existing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.uninstall import uninstall
from gpu_fault.installation_resources import InstallationResource
from tests.admin._aws_cleanup_support import resource
from tests.admin.test_cov95_uninstall_lifecycle import replace_snapshot
from tests.admin.test_uninstall_lifecycle import Harness

LBC_ROLE = "aws/iam/lbc/role"
LBC_POLICY = "aws/iam/lbc/policy"


def test_legacy_lbc_attachment_with_wrong_resource_types_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    role = resource("iam_policy", "lbc-role-as-policy").model_copy(
        update={"resource_key": LBC_ROLE}
    )
    policy = resource("iam_policy", "lbc-policy").model_copy(
        update={"resource_key": LBC_POLICY, "dependencies": [LBC_ROLE]}
    )
    replace_snapshot(harness, [*harness.snapshot.resources, role, policy])
    with pytest.raises(BootstrapError, match="invalid resource types"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []
    assert harness.syncs == 0


class CpuSurvives(Harness):
    """AWS acknowledges the CPU deletion but keeps reporting the clusters."""

    def delete_cpu_cluster(
        self, hyperpod: InstallationResource, eks: InstallationResource
    ) -> None:
        self.events.append("delete:cpu")

    def wait_absent(self, resource: InstallationResource, **_kwargs: object) -> None:
        self.events.append("wait:" + resource.resource_key)


def test_cpu_cluster_still_existing_after_deletion_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = CpuSurvives(tmp_path, monkeypatch)
    with pytest.raises(BootstrapError, match="CPU cluster resource still exists"):
        uninstall(harness.request(delete=True), runner=harness)
    assert "delete:cpu" in harness.events
    assert "delete:aurora" not in harness.events
    assert "cluster/cpu-eks" in harness.existing
    assert harness.state()["phase"] == "CPU_DELETE_IN_PROGRESS"
