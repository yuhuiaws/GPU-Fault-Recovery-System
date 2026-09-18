from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    BootstrapMutationRequired,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.bootstrap_services import (
    ensure_monitoring_resources,
    site_sns_topic_arn,
)
from gpu_fault.admin.monitoring_policy import (
    AMP_SNS_POLICY_ASSET,
    SNS_TOPIC_GENERATION_TAG,
    amp_sns_publish_statement,
    ensure_amp_sns_publish_policy,
)
from gpu_fault_release import repository_root
from scripts.component_wheels import component_modules
from scripts.release_identity import build_release_identity

SITE = "policy-test"
WORKSPACE = "ws-1234"
GENERATION = "a" * 32


@pytest.fixture
def cpu() -> ClusterIdentity:
    arn = "arn:aws:eks:us-east-1:123456789012:cluster/control"
    return ClusterIdentity(
        input_arn=arn,
        role="cpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/control",
        hyperpod_name="control",
        eks_arn=arn,
        eks_name="control",
        vpc_id="vpc-control",
        subnet_ids=("subnet-private",),
        node_recovery="None",
        context="control",
    )


def grant(cpu: ClusterIdentity) -> dict[str, Any]:
    return amp_sns_publish_statement(
        cpu=cpu,
        site_id=SITE,
        workspace_id=WORKSPACE,
        topic_arn=site_sns_topic_arn(cpu, SITE),
    )


class MonitoringAWS(CommandRunner):
    def __init__(
        self,
        cpu: ClusterIdentity,
        *,
        initial: bool = False,
        site_id: str = SITE,
        workspace_id: str = WORKSPACE,
    ) -> None:
        super().__init__()
        self.cpu = cpu
        self.site_id = site_id
        self.topic_arn = site_sns_topic_arn(cpu, site_id)
        self.workspace = {
            "workspaceId": workspace_id,
            "arn": f"arn:aws:aps:{cpu.region}:{cpu.account_id}:workspace/{workspace_id}",
            "alias": f"gpu-fault-{site_id}",
            "status": {"statusCode": "ACTIVE"},
        }
        self.workspace_exists = not initial
        self.topic_exists = not initial
        self.amp_tags: object = {SITE_TAG_KEY: site_id}
        self.sns_tags: object = [
            {"Key": SITE_TAG_KEY, "Value": site_id},
            {"Key": SNS_TOPIC_GENERATION_TAG, "Value": GENERATION},
        ]
        self.policy: object = json.dumps(
            {
                "Version": "2008-10-17",
                "Id": "existing-policy-id",
                "Statement": [
                    {
                        "Sid": "KeepOperatorGrant",
                        "Effect": "Allow",
                        "Principal": {"AWS": f"arn:aws:iam::{cpu.account_id}:root"},
                        "Action": ["sns:GetTopicAttributes"],
                        "Resource": self.topic_arn,
                    }
                ],
            }
        )
        self.calls: list[tuple[str, str, bool]] = []
        self.overrides: dict[tuple[str, str], object] = {}
        self.failures: dict[tuple[str, str], str] = {}
        self.keep_old_policy = False

    def document(self) -> dict[str, Any]:
        assert isinstance(self.policy, str), "test fixture policy is not encoded JSON"
        value: dict[str, Any] = json.loads(self.policy)
        return value

    def mutations(self) -> list[tuple[str, str]]:
        return [(service, action) for service, action, mutate in self.calls if mutate]

    def run(self, arguments: Sequence[str], **options: Any) -> str:
        assert arguments[0] == "aws", "monitoring fixture reached a non-AWS command"
        service, action = arguments[1:3]
        self.calls.append((service, action, bool(options.get("mutate"))))
        key = (service, action)
        if key in self.failures:
            raise BootstrapError(self.failures[key])
        if key in self.overrides:
            return json.dumps(self.overrides[key])
        if key == ("amp", "list-workspaces"):
            return json.dumps(
                {"workspaces": [self.workspace] if self.workspace_exists else []}
            )
        if key == ("amp", "create-workspace"):
            assert options.get("mutate"), "AMP creation was not declared a mutation"
            self.workspace_exists = True
            return json.dumps(
                {
                    "workspaceId": self.workspace["workspaceId"],
                    "arn": self.workspace["arn"],
                }
            )
        if key == ("amp", "describe-workspace"):
            return json.dumps({"workspace": self.workspace})
        if key == ("amp", "list-tags-for-resource"):
            return json.dumps({"tags": self.amp_tags})
        if key == ("sns", "get-topic-attributes"):
            if not self.topic_exists:
                raise BootstrapError("NotFound")
            return json.dumps(
                {
                    "Attributes": {
                        "TopicArn": self.topic_arn,
                        "Owner": self.cpu.account_id,
                        "Policy": self.policy,
                    }
                }
            )
        if key == ("sns", "list-tags-for-resource"):
            return json.dumps({"Tags": self.sns_tags})
        if key == ("sns", "create-topic"):
            assert options.get("mutate"), "SNS creation was not declared a mutation"
            self.topic_exists = True
            tag = next(
                value
                for value in arguments
                if value.startswith(f"Key={SNS_TOPIC_GENERATION_TAG},Value=")
            )
            self.sns_tags = [
                {"Key": SITE_TAG_KEY, "Value": self.site_id},
                {"Key": SNS_TOPIC_GENERATION_TAG, "Value": tag.split("Value=", 1)[1]},
            ]
            return self.topic_arn
        if key == ("sns", "set-topic-attributes"):
            assert options.get("mutate"), "SNS policy write was not declared a mutation"
            assert arguments[arguments.index("--attribute-name") + 1] == "Policy", (
                "policy convergence changed a different topic attribute"
            )
            if not self.keep_old_policy:
                self.policy = arguments[arguments.index("--attribute-value") + 1]
            return ""
        if key == ("sns", "list-subscriptions-by-topic"):
            return json.dumps({"Subscriptions": []})
        if key == ("sns", "subscribe"):
            assert options.get("mutate"), "subscription was not declared a mutation"
            return self.topic_arn + ":subscription"
        raise AssertionError(f"unexpected monitoring operation: {service}/{action}")


def ensure(runner: CommandRunner, cpu: ClusterIdentity) -> dict[str, Any]:
    return ensure_monitoring_resources(runner, cpu=cpu, site_id=SITE, alert_email=None)


def test_first_monitoring_bootstrap_establishes_publish_policy_before_subscriptions(
    cpu: ClusterIdentity,
) -> None:
    runner = MonitoringAWS(cpu, initial=True)
    original = runner.document()
    result = ensure_monitoring_resources(
        runner, cpu=cpu, site_id=SITE, alert_email="ops@example.test"
    )
    assert runner.document() == {
        **original,
        "Statement": [*original["Statement"], grant(cpu)],
    }, "initial monitoring bootstrap did not preserve and extend the topic policy"
    assert runner.mutations() == [
        ("amp", "create-workspace"),
        ("sns", "create-topic"),
        ("sns", "set-topic-attributes"),
        ("sns", "subscribe"),
    ], "monitoring was published or subscribed before its SNS permission was ready"
    assert result["workspace_id"] == WORKSPACE, "bootstrap lost its AMP workspace"
    assert result["sns_topic_arn"] == runner.topic_arn, (
        "bootstrap changed topic identity"
    )


def test_current_publish_policy_is_read_only_regardless_of_statement_order(
    cpu: ClusterIdentity,
) -> None:
    runner = MonitoringAWS(cpu)
    original = runner.document()
    runner.policy = json.dumps(
        {**original, "Statement": [grant(cpu), *original["Statement"]]}, indent=4
    )
    before = runner.policy
    ensure(ReadOnlyProbeRunner(runner), cpu)
    assert runner.mutations() == [], "current monitoring prerequisites caused writes"
    assert runner.policy == before, "current policy was normalized by rewriting it"


@pytest.mark.parametrize("drift", ["missing", "broad", "wrong-workspace", "duplicate"])
def test_policy_drift_blocks_probe_then_converges_only_the_owned_statement(
    cpu: ClusterIdentity, drift: str
) -> None:
    runner = MonitoringAWS(cpu)
    original = runner.document()
    stale = grant(cpu)
    if drift == "broad":
        stale["Principal"] = "*"
        stale["Resource"] = "*"
        stale["Condition"] = {"ArnLike": {"AWS:SourceArn": "*"}}
    elif drift == "wrong-workspace":
        stale["Condition"]["ArnEquals"]["AWS:SourceArn"] += "-different"
    statements = (
        []
        if drift == "missing"
        else [stale, copy.deepcopy(stale)]
        if drift == "duplicate"
        else [stale]
    )
    runner.policy = json.dumps(
        {**original, "Statement": [*original["Statement"], *statements]}
    )
    with pytest.raises(BootstrapMutationRequired):
        ensure(ReadOnlyProbeRunner(runner), cpu)
    assert runner.mutations() == [], "read-only policy probe reached an AWS mutation"
    ensure(runner, cpu)
    assert runner.document() == {
        **original,
        "Statement": [*original["Statement"], grant(cpu)],
    }, "SNS convergence dropped unrelated statements or kept an unsafe owned grant"
    assert runner.mutations() == [("sns", "set-topic-attributes")], (
        "policy drift created or changed foundation resources other than its policy"
    )
    ensure(ReadOnlyProbeRunner(runner), cpu)
    ensure(runner, cpu)
    assert runner.mutations() == [("sns", "set-topic-attributes")], (
        "a current SNS policy was rewritten on retry"
    )


@pytest.mark.parametrize(
    "policy",
    [
        None,
        "",
        "not json",
        "[]",
        "{}",
        '{"Statement":{}}',
        '{"Statement":[false]}',
        '{"Statement":[{}]}',
        '{"Statement":[{"Effect":null,"Action":"sns:Publish"}]}',
        '{"Statement":[{"Effect":"Allow","Action":[]}]}',
        '{"Version":{},"Statement":[]}',
        '{"Version":"future","Statement":[]}',
        '{"Statement":[],"Statement":[]}',
        '{"Statement":[],"Id":NaN}',
    ],
)
def test_unknown_policy_never_authorizes_a_blind_replacement(
    cpu: ClusterIdentity, policy: object
) -> None:
    runner = MonitoringAWS(cpu)
    runner.policy = policy
    with pytest.raises(BootstrapError, match="policy"):
        ensure(runner, cpu)
    assert runner.mutations() == [], "unknown policy was treated as an empty policy"


@pytest.mark.parametrize(
    "operation",
    [
        ("amp", "list-workspaces"),
        ("amp", "describe-workspace"),
        ("amp", "list-tags-for-resource"),
        ("sns", "get-topic-attributes"),
        ("sns", "list-tags-for-resource"),
    ],
)
def test_failed_prerequisite_reads_never_trigger_policy_or_resource_writes(
    cpu: ClusterIdentity, operation: tuple[str, str]
) -> None:
    runner = MonitoringAWS(cpu)
    runner.failures[operation] = "AccessDenied: monitoring read was refused"
    with pytest.raises(BootstrapError, match="AccessDenied"):
        ensure(runner, cpu)
    assert runner.mutations() == [], "a failed read was interpreted as absence or drift"


@pytest.mark.parametrize(
    "invalid",
    [
        "workspace-arn",
        "workspace-alias",
        "workspace-list",
        "workspace-owner",
        "workspace-tags",
        "topic-owner",
        "topic-arn",
        "topic-tags",
        "topic-duplicate-tags",
        "topic-generation",
    ],
)
def test_monitoring_identity_and_tag_errors_fail_closed_before_mutation(
    cpu: ClusterIdentity, invalid: str
) -> None:
    runner = MonitoringAWS(cpu)
    if invalid == "workspace-arn":
        runner.workspace["arn"] = str(runner.workspace["arn"]).replace(
            cpu.account_id, "111122223333"
        )
    elif invalid == "workspace-alias":
        runner.workspace["alias"] = "foreign"
    elif invalid == "workspace-list":
        runner.overrides[("amp", "list-workspaces")] = {
            "workspaces": [runner.workspace, runner.workspace]
        }
    elif invalid == "workspace-owner":
        runner.amp_tags = {SITE_TAG_KEY: "other-site"}
    elif invalid == "workspace-tags":
        runner.amp_tags = None
    elif invalid in {"topic-owner", "topic-arn"}:
        runner.overrides[("sns", "get-topic-attributes")] = {
            "Attributes": {
                "Owner": "111122223333" if invalid == "topic-owner" else cpu.account_id,
                "TopicArn": "arn:foreign"
                if invalid == "topic-arn"
                else runner.topic_arn,
                "Policy": runner.policy,
            }
        }
    elif invalid == "topic-tags":
        runner.sns_tags = None
    elif invalid == "topic-duplicate-tags":
        runner.sns_tags = [
            {"Key": SITE_TAG_KEY, "Value": SITE},
            {"Key": SITE_TAG_KEY, "Value": SITE},
        ]
    else:
        runner.sns_tags = [
            {"Key": SITE_TAG_KEY, "Value": SITE},
            {"Key": SNS_TOPIC_GENERATION_TAG, "Value": "invalid"},
        ]
    with pytest.raises(BootstrapError):
        ensure(runner, cpu)
    assert runner.mutations() == [], "invalid identity or ownership allowed mutation"


def test_policy_write_requires_verified_readback_before_monitoring_completes(
    cpu: ClusterIdentity,
) -> None:
    runner = MonitoringAWS(cpu)
    runner.keep_old_policy = True
    with pytest.raises(BootstrapError, match="did not converge"):
        ensure(runner, cpu)
    assert runner.mutations() == [("sns", "set-topic-attributes")], (
        "readback failure caused repeated writes or started dependent resources"
    )


def test_concurrent_policy_change_is_not_overwritten(cpu: ClusterIdentity) -> None:
    class ConcurrentWriter(MonitoringAWS):
        reads = 0

        def run(self, arguments: Sequence[str], **options: Any) -> str:
            if arguments[1:3] == ["sns", "get-topic-attributes"]:
                self.reads += 1
                if self.reads == 2:
                    current = self.document()
                    current["Id"] = "another-writer-updated-this-policy"
                    self.policy = json.dumps(current)
            return super().run(arguments, **options)

    runner = ConcurrentWriter(cpu)
    with pytest.raises(BootstrapError, match="changed during convergence"):
        ensure_amp_sns_publish_policy(
            runner,
            cpu=cpu,
            site_id=SITE,
            workspace_id=WORKSPACE,
            topic_arn=runner.topic_arn,
            topic_generation=GENERATION,
        )
    assert runner.mutations() == [], "concurrent policy change was overwritten"
    assert runner.document()["Id"] == "another-writer-updated-this-policy", (
        "policy convergence discarded the other writer's update"
    )


def test_direct_policy_entry_refuses_stale_topic_generation(
    cpu: ClusterIdentity,
) -> None:
    runner = MonitoringAWS(cpu)
    with pytest.raises(BootstrapError, match="generation changed"):
        ensure_amp_sns_publish_policy(
            runner,
            cpu=cpu,
            site_id=SITE,
            workspace_id=WORKSPACE,
            topic_arn=runner.topic_arn,
            topic_generation="b" * 32,
        )
    assert runner.mutations() == [], "stale topic generation authorized publication"


def test_shared_statement_is_bound_to_exact_account_workspace_and_topic(
    cpu: ClusterIdentity,
) -> None:
    statement = grant(cpu)
    assert statement == {
        "Sid": "AllowAmpAlertmanagerPublish",
        "Effect": "Allow",
        "Principal": {"Service": "aps.amazonaws.com"},
        "Action": "sns:Publish",
        "Resource": site_sns_topic_arn(cpu, SITE),
        "Condition": {
            "StringEquals": {"AWS:SourceAccount": cpu.account_id},
            "ArnEquals": {
                "AWS:SourceArn": f"arn:aws:aps:{cpu.region}:{cpu.account_id}:workspace/{WORKSPACE}"
            },
        },
    }, "shared AMP publish statement weakened its exact-source restrictions"
    with pytest.raises(BootstrapError, match="ARN differs"):
        amp_sns_publish_statement(
            cpu=cpu,
            site_id=SITE,
            workspace_id=WORKSPACE,
            topic_arn=site_sns_topic_arn(cpu, "foreign"),
        )
    with pytest.raises(BootstrapError, match="binding"):
        grant(replace(cpu, account_id="*"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("Sid", "OtherPolicyOwner"),
        ("Effect", "Deny"),
        ("Principal", "*"),
        ("Action", "*"),
    ],
)
def test_invalid_shared_statement_is_not_used_as_a_permission_template(
    tmp_path: Path, cpu: ClusterIdentity, field: str, value: object
) -> None:
    source = json.loads((repository_root() / AMP_SNS_POLICY_ASSET).read_text("utf-8"))
    source[field] = value
    asset = tmp_path / AMP_SNS_POLICY_ASSET
    asset.parent.mkdir(parents=True)
    asset.write_text(json.dumps(source))
    with pytest.raises(BootstrapError, match="invalid shape"):
        amp_sns_publish_statement(
            cpu=cpu,
            site_id=SITE,
            workspace_id=WORKSPACE,
            topic_arn=site_sns_topic_arn(cpu, SITE),
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "changed", [AMP_SNS_POLICY_ASSET, "src/gpu_fault/admin/monitoring_policy.py"]
)
def test_policy_asset_and_helper_invalidate_only_the_monitoring_foundation_task(
    tmp_path: Path, cpu: ClusterIdentity, changed: str
) -> None:
    root = tmp_path / "repository"
    path = root / changed
    path.parent.mkdir(parents=True)
    path.write_text("initial fixture input")
    request = BootstrapRequest(
        cpu_cluster_arn=cpu.input_arn,
        gpu_cluster_arns=(),
        state_dir=tmp_path / "state",
        repository_root=root,
        alert_email="ops@example.test",
    )
    state = BootstrapState(tmp_path / "state.json", site_id=SITE)
    bind_bootstrap_inputs(state, request=request, cpu=cpu, gpu_clusters=())
    previous = dict(state.value["task_input_sha256"])
    path.write_text("updated fixture input")
    bind_bootstrap_inputs(state, request=request, cpu=cpu, gpu_clusters=())
    current = state.value["task_input_sha256"]
    assert {name for name in previous if previous[name] != current[name]} == {
        "monitoring_resources"
    }, (
        "policy identity did not invalidate its owning task or invalidated unrelated tasks"
    )


def test_deploy_host_closure_and_release_identity_include_monitoring_policy() -> None:
    modules = component_modules("deploy_host")
    assert "gpu_fault.admin.monitoring_policy" in modules, (
        "deploy-host wheel omitted SNS policy convergence"
    )
    assert "gpu_fault.admin.monitoring_policy" not in component_modules(
        "control_plane"
    ), "bootstrap SNS policy convergence entered the business runtime wheel"
    root = repository_root()
    identity = build_release_identity(root)
    expected = hashlib.sha256((root / AMP_SNS_POLICY_ASSET).read_bytes()).hexdigest()
    for value in (identity["manifest_inputs"], identity["components"]["observability"]):
        assert value["files"][AMP_SNS_POLICY_ASSET]["sha256"] == expected, (
            "release identity omitted or mismatched the shared SNS policy asset"
        )
