from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import (
    bootstrap,
    bootstrap_services,
    bootstrap_tasks,
    cli,
    notification_bootstrap,
)
from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import BootstrapState
from gpu_fault.admin.bootstrap_site import site_identifier
from gpu_fault.admin.bootstrap_task_inputs import task_input_spec
from gpu_fault.admin.deploy_command import DeployHooks, run_deploy
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from gpu_fault.admin.notifications import NotificationRouting
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    world = AdminWorld(tmp_path, monkeypatch)
    world.site_id = site_identifier(world.cpu, (world.gpu,))
    world.configure()
    monkeypatch.setattr(bootstrap, "validate_bootstrap_dependencies", lambda: None)
    monkeypatch.setattr(
        bootstrap,
        "discover_bootstrap_scope",
        lambda **kwargs: (
            yaml.safe_load((world.state_dir / "site.yaml").read_text())
            if (world.state_dir / "site.yaml").exists()
            else None,
            world.cpu,
            [world.gpu],
        ),
    )
    monkeypatch.setattr(
        notification_bootstrap,
        "notification_routing",
        lambda *a, **k: (
            "admin@example.com",
            NotificationRouting(
                sender="admin@example.com",
                recipients=("admin@example.com",),
                subject_prefix="",
            ),
        ),
    )

    def prepared_release(**kwargs):
        world.release_ready = True
        state = kwargs["state"]
        bind_bootstrap_inputs(
            state,
            request=kwargs["request"],
            cpu=world.cpu,
            gpu_clusters=(world.gpu,),
            release=world.release,
        )
        state.record("release", world.release)
        state.complete("release")
        return world.release

    monkeypatch.setattr(bootstrap, "prepare_signed_release", prepared_release)

    def access(**kwargs):
        graph = bootstrap_tasks.TaskGraph(
            {
                name: bootstrap_tasks.TaskSpec(
                    ensure=lambda: world.access() or {},
                    input_policy=task_input_spec(name),
                    revalidate=True,
                )
                for name in ("cpu_access", "gpu_access", "pod_identity_agent")
            }
        )
        return SimpleNamespace(
            graph=graph,
            cpu_kubeconfig=world.cpu_kubeconfig,
            gpu_kubeconfig=world.gpu_kubeconfig,
            fleet_master_file=world.master_file,
        )

    monkeypatch.setattr(
        bootstrap, "plan_cluster_access", lambda *args, **kwargs: access(**kwargs)
    )
    results = {
        "control_plane_role": {},
        "email_notifications": {},
        "load_balancer_controller": {},
        "aurora": {"cluster_id": "db"},
        "monitoring_resources": {
            "workspace_id": "ws",
            "sns_topic_arn": "arn:aws:sns:test-1:111122223333:topic",
        },
        "pki": {
            "hostname": "api.invalid",
            "ca_file": str(world.root / "public.pem"),
            "certificate_arn": "arn:aws:acm:test-1:111122223333:certificate/test",
            "hosted_zone_id": "zone",
        },
        "nlb_network": {
            "name": "nlb",
            "public_subnets": ["subnet-public"],
            "security_group": "sg",
        },
        "executor_role:hyperpod-a": {
            "role_arn": "arn:aws:iam::111122223333:role/executor"
        },
    }

    def foundation(**kwargs):
        return bootstrap_tasks.TaskGraph(
            {
                name: bootstrap_tasks.TaskSpec(
                    ensure=lambda value=value: value, input_policy=task_input_spec(name)
                )
                for name, value in results.items()
            }
        )

    monkeypatch.setattr(bootstrap, "foundation_task_graph", foundation)
    original_platform = bootstrap.platform_task_graph

    def platform(**kwargs):
        kwargs["ensure_aurora_ready"] = lambda *a, **k: {}
        return original_platform(**kwargs)

    monkeypatch.setattr(bootstrap, "platform_task_graph", platform)
    monkeypatch.setattr(bootstrap_services, "install_monitoring", lambda *a, **k: {})
    monkeypatch.setattr(
        bootstrap_services, "install_aurora_refresh", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        bootstrap_services,
        "ensure_grafana_dashboards",
        lambda *a, **k: {"status": "SKIPPED"},
    )
    rollout = []

    def automatic_release(**kwargs):
        assert world.helper_calls == 1, (
            "application rollout preceded custody provisioning"
        )
        assert (
            world.api.state["secrets"]["cpu"]["data"]
            == world.api.state["secrets"]["gpu"]["data"]
        )
        rollout.append(kwargs)
        return 0

    hooks = DeployHooks(
        run_source_deploy=lambda **k: pytest.fail(
            "unexpected source bootstrap in prepared test"
        ),
        bootstrap_from_arns=lambda request: bootstrap.bootstrap_from_arns(
            request, runner=world
        ),
        run_automatic_release=automatic_release,
        join_clusters=lambda *a: pytest.fail("unexpected additional cluster"),
        run_rollback=lambda **k: pytest.fail("custody pause must not roll back"),
        approve_profile_plan_inline=lambda *a, **k: {},
        discover_cluster=lambda *a, **k: world.gpu,
        load_site=lambda *a, **k: world.site(),
        release_consent_environment=lambda _: {},
        grafana_environment=lambda _: {},
        grafana_request_fields=lambda _: {},
    )
    arguments = cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(world.state_dir),
            "--cpu-cluster-arn",
            world.cpu.eks_arn,
            "--gpu-cluster-arn",
            world.gpu.eks_arn,
            "--admin-email",
            "admin@example.com",
            "--repo-root",
            str(world.repo),
            "--prepared-source-release",
        ]
    )
    return world, hooks, arguments, rollout


def test_admin_deploy_prepares_real_identities_then_resumes_signed_provisioning(
    deployment,
):
    world, hooks, arguments, rollout = deployment
    with pytest.raises(CustodyPreparationRequired):
        run_deploy(arguments, hooks=hooks)
    assert world.release_ready, "custody preparation must use a completed release"
    assert world.helper_calls == 0 and not rollout
    namespace_uid = world.namespaces["gpu", world.context().namespace]
    state = BootstrapState(
        world.state_dir / "bootstrap-state.json", site_id=world.site_id
    )
    assert not state.is_complete("node_keys:hyperpod-a"), (
        "node keys must remain incomplete while custody awaits authorization"
    )
    assert state.is_complete("release"), (
        "custody preparation pause must preserve the release checkpoint"
    )
    chain = world.authorize()
    assert run_deploy(arguments, hooks=hooks) == 0
    assert chain.exists() and len(rollout) == 1
    assert world.namespaces["gpu", world.context().namespace] == namespace_uid
    before = world.helper_calls
    assert run_deploy(arguments, hooks=hooks) == 0
    assert world.helper_calls == before
    state = BootstrapState(
        world.state_dir / "bootstrap-state.json", site_id=world.site_id
    )
    assert state.value["resources"]["node_keys:hyperpod-a"]["custody_chain"] == str(
        chain
    )


def test_old_node_key_checkpoint_does_not_bypass_configured_custody(deployment):
    world, hooks, arguments, rollout = deployment
    with pytest.raises(CustodyPreparationRequired):
        run_deploy(arguments, hooks=hooks)
    state = BootstrapState(
        world.state_dir / "bootstrap-state.json", site_id=world.site_id
    )
    state.record("node_keys:hyperpod-a", {"cluster_id": "hyperpod-a"})
    state.complete("node_keys:hyperpod-a")
    with pytest.raises(CustodyPreparationRequired):
        run_deploy(arguments, hooks=hooks)
    assert world.helper_calls == 0 and not rollout
    assert not state.result("node_keys:hyperpod-a").get("custody_chain"), (
        "legacy key checkpoint must not acquire unsigned custody proof"
    )


def test_completed_key_shape_with_no_signed_receipt_cannot_be_promoted_on_resume(
    deployment,
):
    world, hooks, arguments, rollout = deployment
    with pytest.raises(CustodyPreparationRequired):
        run_deploy(arguments, hooks=hooks)
    world.authorize()
    from tests.deploy._node_action_key_api import encoded, secret

    world.api.state["secrets"] = {
        plane: secret(plane, {"node-a": encoded("a"), "node-b": encoded("b")})
        for plane in ("gpu", "cpu")
    }
    with pytest.raises(CustodyReconciliationRequired, match="installation provenance"):
        run_deploy(arguments, hooks=hooks)
    assert world.helper_calls == 0 and not rollout
