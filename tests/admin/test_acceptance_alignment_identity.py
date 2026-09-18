from __future__ import annotations

import json

import pytest

from gpu_fault.admin.deploy_identity import INITIAL_REQUEST_FILE, initial_deploy_request
from gpu_fault.admin.site import SiteConfigError

CPU = "arn:aws:eks:us-west-2:123456789012:cluster/cpu"
GPU = "arn:aws:eks:us-west-2:123456789012:cluster/gpu"


def test_early_deploy_resume_recovers_original_request_before_site_exists(tmp_path):
    expected = (CPU, (GPU,), "operations@example.invalid")
    assert (
        initial_deploy_request(
            tmp_path, cpu_arn=CPU, gpu_arns=(GPU,), admin_email=expected[2]
        )
        == expected
    )
    assert not (tmp_path / "site.yaml").exists(), (
        "the initial request must be recoverable before a site is created"
    )
    assert (tmp_path / INITIAL_REQUEST_FILE).stat().st_mode & 0o777 == 0o600
    assert (
        initial_deploy_request(tmp_path, cpu_arn="", gpu_arns=(), admin_email="")
        == expected
    )


@pytest.mark.parametrize("field", ["cpu_arn", "gpu_arns", "admin_email"])
def test_early_resume_cannot_replace_the_original_identity(tmp_path, field):
    arguments = dict(
        cpu_arn=CPU, gpu_arns=(GPU,), admin_email="operations@example.invalid"
    )
    initial_deploy_request(tmp_path, **arguments)
    before = (tmp_path / INITIAL_REQUEST_FILE).read_bytes()
    arguments[field] = ("other",) if field == "gpu_arns" else "other"
    with pytest.raises(SiteConfigError, match="conflicts"):
        initial_deploy_request(tmp_path, **arguments)
    assert (tmp_path / INITIAL_REQUEST_FILE).read_bytes() == before


@pytest.mark.parametrize(
    "damage", ["schema", "scope", "type", "permissions", "symlink"]
)
def test_early_resume_refuses_unbound_or_unsafe_request(tmp_path, damage):
    initial_deploy_request(
        tmp_path, cpu_arn=CPU, gpu_arns=(GPU,), admin_email="ops@example.invalid"
    )
    path = tmp_path / INITIAL_REQUEST_FILE
    if damage == "permissions":
        path.chmod(0o644)
    elif damage == "symlink":
        target = tmp_path / "other.json"
        path.rename(target)
        path.symlink_to(target)
    else:
        value = json.loads(path.read_text())
        value[
            {
                "schema": "schema_version",
                "scope": "state_dir",
                "type": "gpu_cluster_arns",
            }[damage]
        ] = 0
        path.write_text(json.dumps(value))
    with pytest.raises(SiteConfigError, match="invalid"):
        initial_deploy_request(tmp_path, cpu_arn="", gpu_arns=(), admin_email="")
