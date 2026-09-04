"""What the CloudWatch log-shipping installer is not allowed to get wrong.

Logs leaving the cluster is a new data flow with a bill attached, so the
properties pinned here are the ones whose absence is invisible until either the
invoice or an audit arrives: retention set before the first byte, the addon kept
off GPU clusters unless somebody decided otherwise, an uninstall that removes
what it created and keeps the evidence, and a success criterion stronger than
"the pods are Ready".

These are assertions about the script text rather than a live install, which is
how the rest of ``deploy/observability`` is gated -- the live proof is the
runbook's own status action.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin.bootstrap_common import safe_name

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "deploy/observability/install-cloudwatch-observability.sh"


def installer() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def code(text: str) -> str:
    """The script with its comment lines dropped.

    Every negative assertion below is about what the script *does*, and the
    comments name the exact command they explain not using -- so matching the
    raw text would make a documented decision look like the mistake it
    documents.
    """

    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def shell_safe_name(value: str, maximum: int) -> str:
    """Run the installer's own ``safe_name`` against a value.

    The function is lifted out of the script rather than reimplemented here, so
    this compares the shipped code with the Python original instead of comparing
    two copies of the test author's understanding.
    """

    text = installer()
    start = text.index("safe_name() {")
    end = text.index("\n}\n", start) + len("\n}\n")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'{text[start:end]}\nsafe_name "$1" "$2"',
            "--",
            value,
            str(maximum),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize(
    "site_id",
    [
        # A real site id's shape: 47 characters, which puts the derived role
        # name at 74. Synthetic on purpose -- no live site identifier belongs in
        # the repository.
        "ap-example-1-control-plane-gpu-fault-so-1234abcd",
        "short-site",
        "ap-example-1-control-plane-gpu-fault-so-1234abce",
    ],
)
def test_the_role_name_is_derived_exactly_as_the_bootstrap_derives_names(
    site_id: str,
) -> None:
    """A site id is long enough that IAM's 64-character limit decides the name.

    If this shell truncation ever diverged from ``safe_name``, the second run
    would not find the role the first one created: it would build a differently
    truncated name, create a second role, and leave the association pointing at
    whichever one it wrote last. The two sibling ids differing in the final
    character are here because a prefix truncation alone would map them to the
    same role.
    """

    value = f"gpu-fault-{site_id}-cloudwatch-agent"

    derived = shell_safe_name(value, 64)

    assert derived == safe_name(value, maximum=64), (
        "the installer's truncation has to be the site's truncation"
    )
    assert len(derived) <= 64, derived


def test_the_default_action_is_read_only() -> None:
    """A mistyped cluster name must cost a report, not a log-ingestion bill."""

    text = installer()

    assert 'ACTION="${GPU_FAULT_CLOUDWATCH_ACTION:-status}"' in text, (
        "the default action has to be the one that changes nothing"
    )
    assert 'if [[ "${ACTION}" == "status" ]]; then' in text, (
        "the read-only action must return before any mutating call"
    )
    read_only_exit = text.index('if [[ "${ACTION}" == "status" ]]; then')
    for mutating in ("create-addon", "put-retention-policy", "create-role"):
        assert text.index(mutating) > read_only_exit, (
            f"{mutating} appears above the status early exit, so a status run mutates"
        )


def test_retention_is_set_before_the_addon_can_ship_anything() -> None:
    """A log group Fluent Bit creates itself never expires.

    Setting retention afterwards does not shorten what already landed, so the
    order of these two calls is the difference between 30 days of application
    logs and a group that keeps every line for the life of the account.
    """

    text = installer()

    assert text.index("put-retention-policy") < text.index("aws eks create-addon"), (
        "retention must be in place before the addon starts shipping"
    )
    assert "GPU_FAULT_CLOUDWATCH_RETENTION_DAYS:-30" in text, (
        "an explicit default retention is what bounds the cost"
    )
    assert "/performance" in text, (
        "the agent's own embedded-metric group is a log group too, and the one "
        "most often left unbounded"
    )


def test_an_invalid_retention_is_rejected_before_the_group_exists() -> None:
    """CloudWatch validates retention only when it is applied.

    A typo would then create the log groups, fail on the retention call, and
    leave never-expire groups behind -- which is exactly the state this script
    exists to prevent, reached by running it.
    """

    text = installer()
    validation = text.index("RETENTION_VALUES=")

    assert validation < text.index("create-log-group"), (
        "the retention value has to be checked before anything is created"
    )
    assert " 30 " in text[validation : validation + 200], (
        "the accepted set has to be the CloudWatch one, spelled out"
    )


def test_a_gpu_cluster_needs_an_explicit_decision() -> None:
    """Two different failures share one guard.

    The application log group would carry every training job's stdout, and the
    agent's own dcgm-exporter wants host port 9400 -- which
    ``gpu-fault-dcgm-exporter.service`` already holds on every GPU node.
    """

    text = installer()

    assert 'GPU_FAULT_CLOUDWATCH_ALLOW_GPU_CLUSTER:-false}"' in text, (
        "shipping a GPU cluster's logs must be opt-in"
    )
    assert "9400" in text, (
        "the port collision is the half of the reason nobody expects, so it has "
        "to be in the message"
    )
    assert (
        'ACCELERATED_METRICS="${GPU_FAULT_CLOUDWATCH_ACCELERATED_METRICS:-false}"'
        in text
    ), "the tier that deploys the second dcgm-exporter defaults off"
    assert (
        'ENHANCED_INSIGHTS="${GPU_FAULT_CLOUDWATCH_ENHANCED_INSIGHTS:-false}"' in text
    ), "the per-pod metric tier is priced per metric and this site has AMP"


def test_the_installer_refuses_a_kubeconfig_for_another_cluster() -> None:
    """Everything else here names the cluster by string.

    With a kubeconfig pointing elsewhere the addon lands in one cluster while the
    readiness and log-arrival checks pass against another, and both look green.
    """

    text = installer()

    assert "aws eks describe-cluster" in text
    assert "config view --minify" in text
    assert '"${KUBECONFIG_EKS_ARN}" != "${EKS_ARN}"' in text, (
        "the two ARNs have to be compared, not just fetched"
    )


def test_pod_identity_comes_before_the_addon() -> None:
    """The association is by name, so it can exist before the service account.

    Creating it afterwards leaves a first-start window where the agent has no
    credentials, comes up Ready, and silently fails every PutLogEvents.
    """

    text = installer()

    assert text.index("create-pod-identity-association") < text.index(
        "aws eks create-addon"
    ), "the credential has to exist before the pod that needs it"
    assert "eks-pod-identity-agent" in text, (
        "a cluster without the agent addon cannot assume the role at all, and "
        "that is worth saying instead of timing out"
    )


def test_success_is_a_log_stream_not_a_ready_pod() -> None:
    text = installer()

    assert "wait_for_log_arrival" in text, (
        "Pod Ready with a broken association ships nothing and looks healthy"
    )
    assert "describe-log-streams" in text
    assert "/application" in text, (
        "the group that proves the whole chain is the one carrying our own logs"
    )


def test_no_query_reads_a_pagination_token_as_data() -> None:
    """``--max-items`` is client-side paging, so the CLI adds a line of its own.

    A scalar ``--query`` then returns two lines, which the first live status run
    printed as a stray ``None`` under every log group and which the log-arrival
    wait was comparing against as ``"1\\nNone"``. ``--limit`` is the server-side
    parameter and returns only what was asked for.
    """

    text = code(installer())

    assert "--max-items" not in text, (
        "a client-side page token in a scalar query is data the script did not ask for"
    )
    assert text.count("--limit 1") == 2, (
        "both describe-log-streams calls have to bound the result server-side"
    )


def test_the_daemonset_wait_names_no_daemonset() -> None:
    """The addon's DaemonSet set depends on its version and configuration.

    ``rollout status daemonset/fluent-bit`` turns an upstream rename or a
    configuration change into a failed install of something that is actually fine.
    """

    text = installer()

    assert "rollout status daemonset" not in code(text), (
        "waiting on a hardcoded DaemonSet name is a version-bump landmine"
    )
    assert "desiredNumberScheduled" in text, (
        "whatever is in the namespace is what has to become ready"
    )


def test_uninstall_removes_what_it_created_and_keeps_the_evidence() -> None:
    """The rollback path, and the one boundary it does not cross.

    The log groups are the only remaining copy of what the cluster said before it
    stopped shipping, and retention already bounds them; deleting them is a
    separate decision. The IAM role is only deleted when it carries this site's
    tag, because an untagged role was not created here.
    """

    text = installer()
    uninstall = text.index('if [[ "${ACTION}" == "uninstall" ]]; then')
    install = text.index("# --- install ---")
    block = code(text[uninstall:install])

    assert "delete-addon" in block
    assert "--preserve" not in block, (
        "preserving the addon's resources leaves pods shipping after the addon "
        "is reported gone"
    )
    assert "delete-pod-identity-association" in block
    deletions = [
        line
        for line in block.splitlines()
        if "delete-log-group" in line and "printf" not in line
    ]
    assert deletions == [], (
        f"the uninstall may print that command but must not run it: {deletions}"
    )
    assert "list-role-tags" in block, (
        "a role without this site's tag was not created here"
    )
    assert "aws eks wait addon-deleted" in block, (
        "returning before the addon is gone makes a reinstall race it"
    )


def test_the_scope_of_what_is_shipped_is_written_down() -> None:
    """The scope is wider than the log group names suggest, in one direction.

    The live ``host-log.conf`` reads journald with a ``PRIORITY=0-6`` filter and
    excludes only by syslog facility, so it is unit-agnostic: wherever
    ``gpu-fault-*`` units run, their journal ships. That is the opposite of the
    "only a fixed file list" reading, and it is the half an operator sizing a
    CloudWatch bill or an audit scope has to be told.

    In the other direction the training log files really are outside it, and
    either way the control plane never reads CloudWatch -- so
    ``NodeLogBatch.collection_errors`` stays the only evidence channel.
    """

    text = installer()

    assert "PRIORITY 0-6" in text, (
        "the host group's real filter is by priority, not by unit, and the report "
        "has to say so"
    )
    assert "training log files" in text, (
        "the final report has to say what it did not ship"
    )
    assert "NodeLogBatch.collection_errors" in text, (
        "an operator who thinks this replaces node collection stops watching the "
        "only channel that reports collection gaps"
    )
    assert "logging_setup" in text, (
        "central redaction is the precondition for logs leaving the node"
    )


def test_the_report_says_it_starts_at_the_tail() -> None:
    """``READ_FROM_HEAD=Off`` on every input, so there is no backfill.

    Without this an operator reads "logs are centralized now" as "logs are
    centralized", goes looking in CloudWatch for the incident that prompted the
    install, finds nothing, and concludes the install failed.
    """

    text = installer()

    assert "starts at the tail" in text
    assert "nothing written before this install is sent" in text
