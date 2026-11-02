"""The vendor evidence behind batch 231's advice, pinned in code and in both manuals.

Project rule: advice that rests on vendor documentation carries the exact URL,
the section, the version where the vendor versions its documentation, and a
verbatim quote -- in the deciding code, in the CN and EN operations manuals,
and in a test that keeps the two in step. The quotes are checked against the
manuals with whitespace collapsed (the manuals wrap lines); the typographic
apostrophes are the vendors' own.

- HyperPod: what a platform software update does to the nodes it updates
  (``automatic_node_updates.HYPERPOD_UPDATE_EVIDENCE``);
- Kueue v0.18.0: why its Pod webhook covers the product namespace
  (``regional_release_replica_failure.KUEUE_POD_WEBHOOK_EVIDENCE``);
- Kubernetes v1.33.13: the ReplicaSet controller's backoff and its sync on
  every update (``REPLICASET_BACKOFF_EVIDENCE``);
- EKS: the control-plane upgrade at the end of a version's support
  (``regional_release_eks_support.EKS_SUPPORT_EVIDENCE``);
- batch 239, a rolling HyperPod schedule and its rollback alarms
  (``hyperpod_rollback.ROLLBACK_EVIDENCE``): the SageMaker API reference
  (API version and page date given: the pages print no date), the HyperPod AMI
  update page, the NLB metrics and health-check pages and CloudWatch's
  DescribeAlarms -- and the service's own ValidationException texts of
  2026-11-02, which contradict the reference's "Required: No";
- batch 240, the deploy keeping the alarms and the schedule
  (``cpu_rolling_update.EVIDENCE``): PutMetricAlarm (a new alarm starts
  INSUFFICIENT_DATA, an update keeps the state and ignores tags), the
  instance group's update statuses, the HyperPod cron rules, DescribeAlarms'
  permission scope and DeleteAlarms' read-back;
- batch 242, the EKS support-dates request
  (``regional_release_eks_support.DESCRIBE_CLUSTER_VERSIONS_EVIDENCE``): the
  EKS API reference's texts of ``clusterVersions`` and ``includeAll`` -- the
  page states no rule against sending both -- and the service's refusal of
  both in one request (``VERSIONS_SERVICE_REFUSAL``), fetched and probed
  2026-11-02.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cpu_rolling_update
from gpu_fault.admin.automatic_node_updates import HYPERPOD_UPDATE_EVIDENCE
from gpu_fault.admin.hyperpod_rollback import (
    ALARM_DETAILS_EVIDENCE,
    AMI_UPDATE_EVIDENCE,
    DEPLOYMENT_CONFIGURATION_EVIDENCE,
    DESCRIBE_ALARMS_EVIDENCE,
    NLB_HEALTH_CHECKS_EVIDENCE,
    NLB_HEALTHY_HOSTS_EVIDENCE,
    ROLLBACK_EVIDENCE,
    ROLLING_POLICY_EVIDENCE,
    SERVICE_REFUSALS,
)
from gpu_fault.admin.hyperpod_schedule_window import SCHEDULE_EVIDENCE
from gpu_fault_release.regional_release_eks_support import (
    DESCRIBE_CLUSTER_VERSIONS_EVIDENCE,
    EKS_SUPPORT_EVIDENCE,
    VERSIONS_SERVICE_REFUSAL,
)
from gpu_fault_release.regional_release_replica_failure import (
    KUEUE_POD_WEBHOOK_EVIDENCE,
    REPLICASET_BACKOFF_EVIDENCE,
    RETRY_ANNOTATION,
)

ROOT = Path(__file__).resolve().parents[2]
MANUALS = ("docs/管理员日常运维.md", "docs/en/administrator-operations.md")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _needles(evidence: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("url", "page", "section", "sections", "sources", "quotes"):
        value = evidence.get(key)
        values.extend(value if isinstance(value, tuple) else [value] if value else [])
    if evidence.get("version") and evidence.get("vendor") == "Kueue":
        values.append(f"Kueue {evidence['version']}")
    if evidence.get("vendor") == "AWS":
        values.extend(
            str(evidence[key]) for key in ("version", "docs_date") if key in evidence
        )
    return values


def test_the_hyperpod_update_evidence_is_the_aws_page_verbatim() -> None:
    assert HYPERPOD_UPDATE_EVIDENCE == {
        "vendor": "AWS",
        "url": (
            "https://docs.aws.amazon.com/sagemaker/latest/dg/"
            "sagemaker-hyperpod-eks-operate-cli-command-update-cluster-software.html"
        ),
        "page": "Updating the SageMaker HyperPod platform software",
        "section": "Important",
        "quotes": (
            "When this API is called, SageMaker HyperPod doesn’t drain or "
            "redistribute the jobs (Pods) running on the nodes.",
            "The patching process replaces the root volume with the updated AMI, "
            "which means that your previous data stored in the instance root volume "
            "will be lost.",
            "All cluster nodes experience downtime (nodes appear as `<NotReady>` in "
            "the output of `kubectl get node`) while the patching is in progress.",
        ),
    }


@pytest.mark.parametrize("manual", MANUALS)
@pytest.mark.parametrize(
    "evidence",
    [
        HYPERPOD_UPDATE_EVIDENCE,
        KUEUE_POD_WEBHOOK_EVIDENCE,
        REPLICASET_BACKOFF_EVIDENCE,
        EKS_SUPPORT_EVIDENCE,
        DESCRIBE_CLUSTER_VERSIONS_EVIDENCE,
        *ROLLBACK_EVIDENCE,
        *cpu_rolling_update.EVIDENCE,
    ],
    ids=[
        "hyperpod-update",
        "kueue-pod-webhook",
        "replicaset-backoff",
        "eks-support",
        "eks-cluster-versions",
        "rolling-policy",
        "deployment-configuration",
        "alarm-details",
        "ami-update",
        "nlb-healthy-hosts",
        "nlb-health-checks",
        "describe-alarms",
        "put-metric-alarm",
        "software-update-status",
        "cron-expressions",
        "describe-alarms-scope",
        "delete-alarms",
    ],
)
def test_each_manual_carries_each_citation_verbatim(
    manual: str, evidence: dict[str, Any]
) -> None:
    text = _flat((ROOT / manual).read_text(encoding="utf-8"))

    for needle in _needles(evidence):
        assert _flat(needle) in text, f"{manual} lacks {needle!r}"


@pytest.mark.parametrize("manual", MANUALS)
def test_each_manual_names_the_remedies_the_product_prints(manual: str) -> None:
    text = _flat((ROOT / manual).read_text(encoding="utf-8"))

    for needle in (
        RETRY_ANNOTATION,
        "hyperpod-scheduled-update --state-dir",
        "--rolling",
        "eks:DescribeClusterVersions",
        "it is not an authorization failure: read the dates with aws eks "
        "describe-cluster-versions --region <region> --cluster-versions",
        "additionalProperties: false",
        "--rollback-alarm gpu-fault-cpu-roll-api-capacity",
        "RollbackMaximumBatchSize",
        "cloudwatch:DescribeAlarms",
        "elasticloadbalancing:DescribeTargetGroups",
        "--statistic Maximum",
        "gpu-fault-admin cpu-rolling-update --state-dir",
        "spec.cpuControlPlane",
        "cpu_rolling_update",
        "cpu_update_window",
        "cpu_rolling_update_restore",
        "cloudwatch:PutMetricAlarm",
        "cloudwatch:TagResource",
        "cloudwatch:DeleteAlarms",
        "elasticloadbalancing:DescribeTargetHealth",
        "sagemaker:UpdateCluster",
    ):
        assert needle in text, f"{manual} lacks {needle!r}"


def test_the_rolling_schedule_evidence_is_the_aws_pages_verbatim() -> None:
    """The quotes that decide batch 239's request, read 2026-11-02."""

    assert ROLLBACK_EVIDENCE == (
        ROLLING_POLICY_EVIDENCE,
        DEPLOYMENT_CONFIGURATION_EVIDENCE,
        ALARM_DETAILS_EVIDENCE,
        AMI_UPDATE_EVIDENCE,
        NLB_HEALTHY_HOSTS_EVIDENCE,
        NLB_HEALTH_CHECKS_EVIDENCE,
        DESCRIBE_ALARMS_EVIDENCE,
    )
    api = "https://docs.aws.amazon.com/sagemaker/latest/APIReference/"
    for evidence, page in (
        (ROLLING_POLICY_EVIDENCE, "RollingDeploymentPolicy"),
        (DEPLOYMENT_CONFIGURATION_EVIDENCE, "DeploymentConfiguration"),
        (ALARM_DETAILS_EVIDENCE, "AlarmDetails"),
    ):
        assert (evidence["url"], evidence["page"], evidence["section"]) == (
            f"{api}API_{page}.html",
            page,
            "Contents",
        )
        assert (evidence["version"], evidence["docs_date"]) == (
            "API version 2017-07-24",
            "Last-Modified 2026-10-30",
        )
    assert ROLLING_POLICY_EVIDENCE["quotes"][1] == (
        "The maximum amount of instances in the cluster that SageMaker can roll back "
        "at a time. Type: CapacitySizeConfig object Required: No"
    )
    assert DEPLOYMENT_CONFIGURATION_EVIDENCE["quotes"][0] == (
        "An array that contains the alarms that SageMaker monitors to know whether to "
        "roll back the AMI update. Type: Array of AlarmDetails objects Array Members: "
        "Minimum number of 1 item. Maximum number of 10 items. Required: No"
    )
    assert AMI_UPDATE_EVIDENCE["quotes"][1] == (
        "To utilize automatic rollbacks, your IAM execution role must have the "
        "permission `cloudwatch:DescribeAlarms`."
    )
    assert NLB_HEALTHY_HOSTS_EVIDENCE["quotes"][-1].endswith(
        "It's recommended to monitor maximum `HealthyHostCount`, invoking the alarm "
        "when the maximum `HealthyHostCount` falls below your required minimum, or "
        "being `0`."
    ), "the reason the recommended pair reads the Maximum"


@pytest.mark.parametrize("manual", MANUALS)
def test_each_manual_records_the_live_refusals_with_their_times(manual: str) -> None:
    text = _flat((ROOT / manual).read_text(encoding="utf-8"))

    assert [item["member"] for item in SERVICE_REFUSALS] == [
        "RollbackMaximumBatchSize",
        "AutoRollbackConfiguration",
        "ScheduledUpdateConfig",
    ]
    for refusal in SERVICE_REFUSALS:
        assert f'{refusal["at"]} "{_flat(refusal["text"])}"' in text, (
            f"{manual} lacks the {refusal['at']} refusal verbatim"
        )


@pytest.mark.parametrize("manual", MANUALS)
def test_each_manual_records_the_versions_refusal_and_the_request(manual: str) -> None:
    text = _flat((ROOT / manual).read_text(encoding="utf-8"))
    refusal = VERSIONS_SERVICE_REFUSAL

    assert f'{refusal["at"]} "{_flat(refusal["text"])}"' in text, (
        f"{manual} lacks the {refusal['at']} refusal verbatim"
    )
    assert f"`{refusal['request']}`" in text, f"{manual} lacks the refused request"


def test_the_deploy_step_evidence_is_the_aws_pages_verbatim() -> None:
    """The quotes batch 240's step is built on, read 2026-11-02."""

    assert cpu_rolling_update.EVIDENCE == (
        cpu_rolling_update.PUT_METRIC_ALARM_EVIDENCE,
        cpu_rolling_update.SOFTWARE_UPDATE_EVIDENCE,
        SCHEDULE_EVIDENCE,
        cpu_rolling_update.DESCRIBE_ALARMS_SCOPE_EVIDENCE,
        cpu_rolling_update.DELETE_ALARMS_EVIDENCE,
    )
    put = cpu_rolling_update.PUT_METRIC_ALARM_EVIDENCE
    assert put["quotes"][0].startswith(
        "When this operation creates an alarm, the alarm state is immediately set to "
        "`INSUFFICIENT_DATA`."
    ), "why the alarms are put only once the control plane is full"
    assert "any tags you specify in this parameter are ignored" in put["quotes"][2], (
        "why an adopted alarm is tagged with TagResource"
    )
    assert SCHEDULE_EVIDENCE["quotes"][1].endswith("`cron(00 12 ? * 1 *)`"), (
        "the guide's Monday example"
    )
    assert "3 refers to Tuesday" in SCHEDULE_EVIDENCE["quotes"][2], (
        "the guide's two numberings of the day of the week"
    )
    assert (
        cpu_rolling_update.SOFTWARE_UPDATE_EVIDENCE["version"],
        cpu_rolling_update.SOFTWARE_UPDATE_EVIDENCE["docs_date"],
    ) == ("API version 2017-07-24", "Last-Modified 2026-10-30")
    assert cpu_rolling_update.DELETE_ALARMS_EVIDENCE["quotes"][0].endswith(
        "you can use the DescribeAlarms operation after using DeleteAlarms."
    ), "why uninstall reads a deleted alarm back"
