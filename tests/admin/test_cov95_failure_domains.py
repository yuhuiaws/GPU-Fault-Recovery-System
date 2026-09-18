from __future__ import annotations

import argparse
import copy
import json
import subprocess

import pytest

from gpu_fault.admin import failure_domain_map as domains
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_failure_domain_map import FakeKubectl
from tests.admin.test_admin_site import site_file


@pytest.fixture
def site(tmp_path):
    return load_site(site_file(tmp_path))


@pytest.mark.parametrize(
    "output,code",
    [("invalid", 0), ("[]", 0), ("{}", 0), ('{"items":null}', 0), ("", 1)],
)
def test_failure_domain_inventory_errors_do_not_become_empty_maps(site, output, code):
    calls = []

    def read(arguments, **_options):
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments, code, output, "example unavailable"
        )

    with pytest.raises(BootstrapError, match="cannot read|invalid node"):
        domains.cluster_nodes(site, "gpu-a", run=read)
    assert len(calls) == 1


@pytest.mark.parametrize("builder", [False, True])
def test_failure_domain_unknown_cluster_is_rejected_before_inventory(site, builder):
    calls = []
    with pytest.raises(BootstrapError, match="not in the managed site"):
        if builder:
            domains.build_failure_domain_map(
                site,
                cluster_ids=["foreign"],
                list_nodes=lambda *_args: calls.append("read"),
            )
        else:
            domains.cluster_nodes(
                site, "foreign", run=lambda *_args, **_options: calls.append("read")
            )
    assert calls == []


def test_failure_domain_subset_does_not_query_unselected_clusters(site):
    sibling = copy.deepcopy(site.release_config["clusters"][0])
    sibling["cluster_id"] = "gpu-b"
    site.release_config["clusters"].append(sibling)
    calls = []

    def nodes(_site, cluster_id):
        calls.append(cluster_id)
        return {}

    result = domains.build_failure_domain_map(
        site, cluster_ids=["gpu-b"], list_nodes=nodes
    )
    assert calls == ["gpu-b"]
    assert result.mapping == {"gpu-b": {}}


@pytest.mark.parametrize(
    "change",
    ["missing-version", "terminating", "list-annotations", "nontext-annotation"],
)
def test_failure_domain_worker_identity_must_be_stable_before_map_apply(site, change):
    fake = FakeKubectl()

    def read(arguments, **options):
        result = fake(arguments, **options)
        if "get" in arguments and "deployment" in arguments:
            worker = json.loads(result.stdout)
            if change == "missing-version":
                worker["metadata"].pop("resourceVersion")
            elif change == "terminating":
                worker["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
            else:
                worker["spec"]["template"]["metadata"]["annotations"] = (
                    [] if change == "list-annotations" else {"example": None}
                )
            return subprocess.CompletedProcess(arguments, 0, json.dumps(worker), "")
        return result

    with pytest.raises(BootstrapError, match="worker identity|worker annotations"):
        domains.apply_failure_domain_map(site, run=read)
    assert "apply" not in fake.verbs()
    assert "patch" not in fake.verbs()


@pytest.mark.parametrize("error", [False, True])
def test_failure_domain_map_requires_worker_readback_after_rollout(site, error):
    fake = FakeKubectl()

    def read(arguments, **options):
        after_rollout = "rollout status" in fake.verbs()
        result = fake(arguments, **options)
        if after_rollout and "get" in arguments and "deployment" in arguments:
            return subprocess.CompletedProcess(
                arguments, int(error), "", "example refused" if error else ""
            )
        return result

    with pytest.raises(BootstrapError, match="worker is unavailable after rollout"):
        domains.apply_failure_domain_map(site, run=read)
    assert fake.verbs().count("rollout status") == 1


def test_failure_domain_debug_command_without_output_path_is_read_only(
    site, monkeypatch, capsys
):
    fake = FakeKubectl()
    monkeypatch.setattr(domains, "run_command", fake)
    assert (
        domains.run_failure_domain_map_command(
            argparse.Namespace(output=None), site=site
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert "output" not in report
    assert "apply" not in fake.verbs()
    assert "patch" not in fake.verbs()
