"""Fixtures shared by the bootstrap test modules.

``test_admin_bootstrap.py`` covers the release side of bootstrap and
``test_admin_bootstrap_site.py`` the site and GPU-scope side; both drive the same
control-plane cluster identity, so it is defined once here. Keeping it out of
either module also keeps the two files independent -- neither has to be imported
to run the other.
"""

from __future__ import annotations

import subprocess

from gpu_fault.admin import bootstrap_common
from gpu_fault.admin.bootstrap_common import ClusterIdentity


def patch_bootstrap_commands(monkeypatch, subprocess_fake):
    """Keep raw legacy probes and the bounded command executor on one fake CLI."""
    monkeypatch.setattr(subprocess, "run", subprocess_fake)

    def run(arguments, **kwargs):
        return subprocess_fake(
            arguments,
            input=kwargs.get("input_text"),
            env=kwargs.get("environment"),
            capture_output=kwargs.get("capture", True),
            text=True,
            check=False,
            cwd=kwargs.get("cwd"),
        )

    monkeypatch.setattr(bootstrap_common, "run_command", run)


def _cluster() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        role="cpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/control"),
        hyperpod_name="control",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        eks_name="control",
        vpc_id="vpc-control",
        subnet_ids=("subnet-private-a", "subnet-private-b"),
        node_recovery="None",
        context="control",
    )
