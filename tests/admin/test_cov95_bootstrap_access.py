from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault.admin import bootstrap, bootstrap_common
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    CommandRunner,
)
from tests.admin._cov95_join_support import target


@pytest.mark.parametrize(
    "status,error",
    [
        (
            1,
            "Error from server (Forbidden): NotFound is not permission to replace a Secret",
        ),
        (1, "Unable to connect to the server: proxy returned NotFound"),
        (1, "untrusted authentication helper returned NotFound"),
        (
            126,
            'Error from server (NotFound): secrets "gpu-fault-control-plane-active" not found',
        ),
        (1, 'Error from server (NotFound): secrets "foreign-secret" not found'),
        (1, 'Error from server (NotFound): namespaces "gpu-fault-system" not found'),
    ],
)
def test_base_secret_read_error_cannot_authorize_creation(
    tmp_path, monkeypatch, status, error
):
    """COV95-ADMIN-001: a diagnostic keyword cannot authorize replacement of fleet keys."""
    applied_kinds = []

    def command(arguments, **options):
        output = ""
        code = 0
        errors = ""
        if "update-kubeconfig" in arguments:
            Path(arguments[arguments.index("--kubeconfig") + 1]).write_text(
                "example-kubeconfig"
            )
        elif "create" in arguments and "namespace" in arguments:
            output = '{"kind":"Namespace","metadata":{"name":"gpu-fault-system"}}'
        elif "get" in arguments and "secret" in arguments:
            code, errors = status, error
        elif "apply" in arguments:
            applied_kinds.append(json.loads(options["input_text"])["kind"])
        else:
            raise AssertionError("unexpected bootstrap access command")
        return subprocess.CompletedProcess(arguments, code, output, errors)

    monkeypatch.setattr(bootstrap_common, "run_command", command)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    plan = bootstrap.plan_cluster_access(
        CommandRunner(),
        state=BootstrapState(state_dir / "bootstrap.json", site_id="test-site"),
        cpu=replace(target("a"), role="cpu"),
        gpu_clusters=(),
        state_dir=state_dir,
        namespace="gpu-fault-system",
        secure_dir=state_dir / "secure",
        site_id="test-site",
        ensure_pod_identity_agent=lambda *_args: {},
    )
    with pytest.raises(BootstrapError):
        plan.graph.tasks["cpu_access"]()
    assert applied_kinds == ["Namespace"], (
        "an unproven absence replaced CPU credentials"
    )
    assert not plan.fleet_master_file.exists(), (
        "an unproven read minted a new fleet master"
    )
