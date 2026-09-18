from __future__ import annotations

import io
import json
import subprocess
from contextlib import redirect_stdout

import pytest

from gpu_fault.admin import cli, operator_identity
from gpu_fault.admin.config import load_desired_admin_config
from gpu_fault.admin.config_file import write_admin_config_file
from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from scripts.e2e.regional import boot020_admin_config as runner
from tests.admin import test_admin_config_recovery as config_support

scenario = config_support.scenario
release_document = config_support.release_document


def test_boot020_uses_actual_public_config_plan_apply_noop_and_restore(
    scenario, monkeypatch
):
    before = scenario.before
    desired = before.patched(
        {
            "capacity": {
                "controlWorkerReplicas": before.capacity.control_worker_replicas - 1
            }
        }
    )
    write_admin_config_file(scenario.root / "admin-config.yaml", before, overwrite=True)
    generation = dict.fromkeys(("worker", "ingress", "spool"), 1)
    names = {
        "worker": "gpu-fault-control-worker",
        "ingress": "gpu-fault-api-ha",
        "spool": "gpu-fault-telemetry-spool-worker",
    }
    calls = []

    def apply_driver():
        current = load_desired_admin_config(scenario.root)
        for role, digest in current.role_sha256().items():
            if digest != scenario.live["admin_config_role_sha256"][role]:
                generation[role] += 1
        scenario.live = release_document(current)

    def observation(_site):
        current = load_desired_admin_config(scenario.root)
        return {
            names[role]: {
                "uid": role,
                "generation": generation[role],
                "replicas": 1,
                "template_sha256": current.role_sha256()[role],
            }
            for role in generation
        }

    def public_driver(arguments, **options):
        descriptor = int(options["env"][SITE_OPERATION_LOCK_FD_ENV])
        assert options["pass_fds"] == (descriptor,)
        calls.append(arguments[3:])
        output = io.StringIO()
        with monkeypatch.context() as environment, redirect_stdout(output):
            environment.setenv(SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
            code = cli.run(cli.parser().parse_args(arguments[3:]))
        return subprocess.CompletedProcess(arguments, code, output.getvalue(), "")

    scenario.driver_code = 0
    scenario.driver_action = apply_driver
    monkeypatch.setattr(runner, "run_driver", public_driver)
    monkeypatch.setattr(
        operator_identity,
        "caller_identity_arn",
        lambda **_kw: "arn:aws:sts::123456789012:assumed-role/unit/operator",
    )
    monkeypatch.setattr(runner, "cpu_observation", observation)
    monkeypatch.setattr(
        runner,
        "deployment_generations",
        lambda _path: {"gpu": {"gpu-a": {"executor": 1}}},
    )

    result = runner.public_config_roundtrip(
        scenario.root, desired=desired, reference="CHG-ALIGNMENT"
    )

    assert [item[0] for item in calls] == ["config"] * 4
    assert "--dry-run" in calls[0]
    assert all("--reference" in item for item in calls), (
        "every public config invocation must carry the approval reference"
    )
    assert result["repeat_noop"] and result["audit_present"]
    assert result["affected_deployments"] == ["gpu-fault-control-worker"]
    assert generation == {"worker": 3, "ingress": 1, "spool": 1}
    assert scenario.driver_calls == 2
    assert load_desired_admin_config(scenario.root) == before
    assert not (scenario.root / "admin-config/pending.json").exists(), (
        "the completed config roundtrip must not leave a pending administrator edit"
    )


def test_public_config_witness_cannot_replace_pending_or_uncommitted_admin_edits(
    scenario,
):
    desired = scenario.before.patched(
        {
            "capacity": {
                "controlWorkerReplicas": scenario.before.capacity.control_worker_replicas
                + 1
            }
        }
    )
    with pytest.raises(ValueError, match="administrator edit"):
        runner.public_config_roundtrip(
            scenario.root, desired=desired, reference="CHG-ALIGNMENT"
        )
    assert scenario.driver_calls == 0


def test_public_config_scope_refuses_a_different_control_plane(scenario, tmp_path):
    value = dict(scenario.site.release_config)
    value["cpu_eks_arn"] = "arn:aws:eks:us-east-1:123456789012:cluster/foreign"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value, default=str))
    with pytest.raises(ValueError, match="differs"):
        runner.validate_admin_target(scenario.root, path)
