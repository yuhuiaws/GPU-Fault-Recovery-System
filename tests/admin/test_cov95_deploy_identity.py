from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import deploy_command as deploy
from gpu_fault.admin.site import SiteConfigError, load_site
from tests.admin._cov95_join_support import target
from tests.admin.test_admin_site import site_file


@pytest.fixture
def context(tmp_path):
    site = load_site(site_file(tmp_path))
    events = []

    def discover(_runner, **options):
        events.append(("discover", options["cluster_arn"]))
        return target()

    hooks = deploy.DeployHooks(
        run_source_deploy=lambda **_options: 0,
        bootstrap_from_arns=lambda _request: SimpleNamespace(
            site_file=site.source, pending_gpu_cluster_arns=(target().eks_arn,)
        ),
        run_automatic_release=lambda **_options: 7,
        join_clusters=lambda requests: events.append(("join", requests))
        or {"phase": "COMPLETED"},
        run_rollback=lambda *_args, **_options: 0,
        approve_profile_plan_inline=lambda *_args, **_options: {},
        discover_cluster=discover,
        load_site=load_site,
        release_consent_environment=lambda _arguments: {},
        grafana_environment=lambda _arguments: {},
        grafana_request_fields=lambda _arguments: {},
    )
    return SimpleNamespace(site=site, root=tmp_path, events=events, hooks=hooks)


@pytest.mark.parametrize(
    "document", [{}, {"spec": []}, {"spec": {"cpu": {}, "clusters": None}}]
)
def test_managed_deploy_inputs_require_structured_cluster_identity(tmp_path, document):
    with pytest.raises(SiteConfigError, match="no valid cluster identity"):
        deploy.managed_site_deploy_inputs(document, tmp_path)


def test_managed_deploy_aliases_use_only_recorded_hyperpod_bindings(context):
    document = yaml.safe_load(context.site.source.read_text())
    cpu = document["spec"]["cpu"]["eksArn"]
    gpu = document["spec"]["clusters"][0]["eksClusterArn"]
    document["spec"]["clusters"].extend([None, {}])
    (context.root / "bootstrap-state.json").write_text(
        json.dumps(
            {
                "resources": {
                    "hyperpod_by_eks": {
                        cpu: target("a").hyperpod_arn,
                        gpu: target().hyperpod_arn,
                    }
                }
            }
        )
    )
    _cpu, aliases, managed, email = deploy.managed_site_deploy_inputs(
        document, context.root
    )
    assert aliases == {cpu, target("a").hyperpod_arn}
    assert managed == {gpu: gpu, target().hyperpod_arn: gpu}
    assert email is None


def test_first_deploy_requires_both_cluster_scopes(tmp_path, context):
    with pytest.raises(SiteConfigError, match="requires --cpu-cluster-arn"):
        deploy.resolve_deploy_identity(
            argparse.Namespace(), tmp_path / "new", hooks=context.hooks
        )
    assert context.events == []


def test_managed_deploy_requires_an_administrator_address(context):
    with pytest.raises(SiteConfigError, match="records no administrator email"):
        deploy.resolve_deploy_identity(
            argparse.Namespace(), context.root, hooks=context.hooks
        )
    assert context.events == []


def test_unknown_hyperpod_alias_is_discovered_once_and_bound_to_the_delta(context):
    managed = context.site.release_config["clusters"][0]["eks_cluster_arn"]
    arguments = argparse.Namespace(
        gpu_cluster_arn=[managed, target().hyperpod_arn],
        alert_email="example@example.invalid",
    )
    _cpu, requested, email, delta = deploy.resolve_deploy_identity(
        arguments, context.root, hooks=context.hooks
    )
    assert requested == (managed, target().hyperpod_arn)
    assert delta == (target().hyperpod_arn,)
    assert email == "example@example.invalid"
    assert context.events == [("discover", target().hyperpod_arn)]


def test_deploy_reference_without_profile_plan_cannot_authorize_approval(context):
    with pytest.raises(SiteConfigError, match="only used with --approve-profile-plan"):
        deploy.approve_pending_profile_plan(
            argparse.Namespace(reference="CHG-EXAMPLE"),
            context.root,
            hooks=context.hooks,
        )
    assert context.events == []


@pytest.mark.parametrize(
    "contents",
    [None, "invalid", "[]", '{"mode":"NOOP"}', '{"mode":"APPLICATION_RELEASE"}'],
)
def test_source_deploy_success_marker_requires_explicit_application_release(
    tmp_path, contents
):
    if contents is not None:
        (tmp_path / deploy.SOURCE_DEPLOY_SUCCESS_STATE).write_text(contents)
    assert deploy.source_deploy_applied_release(tmp_path) is (
        contents == '{"mode":"APPLICATION_RELEASE"}'
    )


def test_join_delta_reports_its_public_result_without_running_a_driver(context, capsys):
    deploy.join_gpu_clusters(
        context.site.source,
        [target().eks_arn],
        repository_root=None,
        hooks=context.hooks,
    )
    assert json.loads(capsys.readouterr().out) == {"phase": "COMPLETED"}
    assert context.events[0][0] == "join"
    assert context.events[0][1][0].gpu_cluster_arn == target().eks_arn


def test_failed_prepared_release_never_starts_pending_cluster_joins(context):
    arguments = argparse.Namespace(
        repo_root=context.site.repository_root, state_dir=context.root
    )
    assert (
        deploy.run_prepared_deploy(
            arguments,
            cpu_cluster_arn=context.site.release_config["cpu_eks_arn"],
            gpu_cluster_arns=(target().eks_arn,),
            admin_email="example@example.invalid",
            hooks=context.hooks,
        )
        == 7
    )
    assert context.events == []
