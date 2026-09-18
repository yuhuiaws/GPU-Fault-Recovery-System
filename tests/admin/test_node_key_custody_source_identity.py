from __future__ import annotations

from pathlib import Path

import yaml

from scripts.deploy_source_identity import deploy_host_identity
from scripts.release_identity import file_set_identity

ROOT = Path(__file__).resolve().parents[2]


def test_provisioning_closure_is_bound_to_release_and_deploy_host_identities():
    config = yaml.safe_load((ROOT / "config/release-identity.yaml").read_text())
    node_template = file_set_identity(ROOT, tuple(config["node_template_inputs"]))
    deploy_host = deploy_host_identity(ROOT)["orchestration"]
    expected = {
        "deploy/node/provision-node-action-keys.sh",
        "deploy/node/provision_node_action_keys.py",
        "src/gpu_fault/admin/node_key_proof.py",
        *(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "src/gpu_fault/admin").glob("node_key_custody*.py")
        ),
    }
    assert expected <= node_template["files"].keys()
    assert expected <= deploy_host["files"].keys()
    for path in expected:
        assert node_template["files"][path] == deploy_host["files"][path]
