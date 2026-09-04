"""Fixtures shared by the bootstrap test modules.

``test_admin_bootstrap.py`` covers the release side of bootstrap and
``test_admin_bootstrap_site.py`` the site and GPU-scope side; both drive the same
control-plane cluster identity, so it is defined once here. Keeping it out of
either module also keeps the two files independent -- neither has to be imported
to run the other.
"""

from __future__ import annotations

from gpu_fault.admin.bootstrap_common import ClusterIdentity


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
