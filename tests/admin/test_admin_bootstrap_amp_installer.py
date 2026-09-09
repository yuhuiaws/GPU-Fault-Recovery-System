"""What ``install-amp-monitoring.sh`` writes when nothing has changed.

The installer runs on every deploy, and every write it makes unconditionally is
paid for on every deploy: an audited ``iam put-role-policy``, an SNS policy
rewrite, and an ADOT rollout whose Pod has to terminate and come back before the
release may continue. These tests drive the real script with fake ``aws``,
``kubectl`` and ``python3`` executables on ``PATH``, which log their argv and
answer from canned files, so the question "did a converged installer write
anything" is answered by the script itself rather than by reading it.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/observability/install-amp-monitoring.sh"
ACCOUNT = "123456789012"
REGION = "us-east-1"
CLUSTER = "control"
CLUSTER_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{CLUSTER}"
NAMESPACE = "gpu-fault-system"
WORKSPACE_ID = "ws-a"
WORKSPACE_ARN = f"arn:aws:aps:{REGION}:{ACCOUNT}:workspace/{WORKSPACE_ID}"
TOPIC_NAME = "gpu-fault-site-a-alerts"
TOPIC_ARN = f"arn:aws:sns:{REGION}:{ACCOUNT}:{TOPIC_NAME}"
ROLE_NAME = "gpu-fault-site-a-amp-writer"
ADOT_IMAGE = "adot@sha256:bbb"
CONVERGED_APPLY = (
    "configmap/gpu-fault-adot unchanged\ndeployment.apps/gpu-fault-adot unchanged"
)

# The fakes answer from ``${FAKE_RESPONSES}/<service>_<operation>``, so a test
# expresses drift by writing a different file rather than by patching a script.
# A missing file with the key listed in ``FAKE_ABSENT`` is the AWS "not found"
# error; a missing file otherwise is a write, which succeeds silently.
_FAKE_AWS = """#!/usr/bin/env bash
set -u
printf '%s\\n' "aws $*" >>"${FAKE_CALL_LOG}"
key="${1}_${2}"
# A resource that is created starts answering the read that reported it absent,
# so a script that reads before writing sees what AWS would show it.
if [[ "${key}" == "sns_create-topic" &&
    -f "${FAKE_RESPONSES}/sns_created-topic-policy" ]]; then
    cp "${FAKE_RESPONSES}/sns_created-topic-policy" \
        "${FAKE_RESPONSES}/sns_get-topic-attributes.text"
fi
file="${FAKE_RESPONSES}/${key}"
if [[ "$*" == *--query* && -f "${file}.text" ]]; then
    cat "${file}.text"
    exit 0
fi
if [[ -f "${file}" ]]; then
    cat "${file}"
    exit 0
fi
for absent in ${FAKE_ABSENT:-}; do
    if [[ "${absent}" == "${key}" ]]; then
        printf 'An error occurred (NoSuchEntity) when calling %s\\n' "${key}" >&2
        exit 254
    fi
done
exit 0
"""

_FAKE_KUBECTL = """#!/usr/bin/env bash
set -u
printf '%s\\n' "kubectl $*" >>"${FAKE_CALL_LOG}"
case "$*" in
*"config view"*)
    printf '%s\\n' "${FAKE_CLUSTER_ARN}"
    ;;
*"get deployment gpu-fault-adot"*)
    printf '%s\\n' "${FAKE_ADOT_REPLICAS}"
    ;;
*apply*)
    printf '%s\\n' "${FAKE_APPLY_OUTPUT}"
    ;;
*)
    :
    ;;
esac
exit 0
"""

_FAKE_PYTHON = """#!/usr/bin/env bash
set -u
printf '%s\\n' "python3 $*" >>"${FAKE_CALL_LOG}"
exit 0
"""


def _sns_policy(*, with_amp_statement: bool, amp_statement_first: bool = False) -> str:
    statements: list[dict[str, object]] = [
        {
            "Sid": "__default_statement_ID",
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["SNS:Publish"],
            "Resource": TOPIC_ARN,
        }
    ]
    if with_amp_statement:
        statements.append(
            {
                "Sid": "AllowAmpAlertmanagerPublish",
                "Effect": "Allow",
                "Principal": {"Service": "aps.amazonaws.com"},
                "Action": "sns:Publish",
                "Resource": TOPIC_ARN,
                "Condition": {
                    "StringEquals": {"AWS:SourceAccount": ACCOUNT},
                    "ArnEquals": {"AWS:SourceArn": WORKSPACE_ARN},
                },
            }
        )
    if amp_statement_first:
        statements.reverse()
    return json.dumps(
        {"Version": "2008-10-17", "Id": "__default_policy_ID", "Statement": statements}
    )


def _amp_write_policy() -> str:
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aps:RemoteWrite"],
                    "Resource": WORKSPACE_ARN,
                }
            ],
        }
    )


def _rendered_alertmanager() -> str:
    source = (ROOT / "deploy/observability/amp-alertmanager.yaml").read_text(
        encoding="utf-8"
    )
    return source.replace("REPLACE_WITH_SNS_TOPIC_ARN", TOPIC_ARN).replace(
        "REPLACE_WITH_AWS_REGION", REGION
    )


def _amp_definition(root: str, payload: bytes) -> str:
    return json.dumps(
        {
            root: {
                "status": {"statusCode": "ACTIVE"},
                "data": base64.b64encode(payload).decode(),
            }
        }
    )


class Installer:
    """The real installer, with fake CLIs and a fully converged account."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.bin = tmp_path / "bin"
        self.responses = tmp_path / "responses"
        self.log = tmp_path / "calls.log"
        self.bin.mkdir()
        self.responses.mkdir()
        self.log.write_text("", encoding="utf-8")
        for name, body in (
            ("aws", _FAKE_AWS),
            ("kubectl", _FAKE_KUBECTL),
            ("python3", _FAKE_PYTHON),
        ):
            executable = self.bin / name
            executable.write_text(body, encoding="utf-8")
            executable.chmod(0o755)
        self.apply_output = CONVERGED_APPLY
        self.adot_replicas = "1"
        self.absent: tuple[str, ...] = ()
        self.environment: dict[str, str] = {}
        # Positional arguments for the script (the per-cluster expected-rules
        # hand-off); the release engine is the only caller that passes any.
        self.arguments: list[str] = []
        self.answer("sts_get-caller-identity", ACCOUNT)
        self.answer("eks_describe-cluster", CLUSTER_ARN)
        self.answer("iam_get-role", json.dumps({"Role": {"RoleName": ROLE_NAME}}))
        self.answer("iam_get-role-policy", _amp_write_policy())
        self.answer(
            "sns_get-topic-attributes.text", _sns_policy(with_amp_statement=True)
        )
        self.answer("sns_list-subscriptions-by-topic.text", "1")
        self.answer("eks_list-pod-identity-associations.text", "assoc-a")
        self.answer(
            "eks_describe-pod-identity-association.text",
            f"arn:aws:iam::{ACCOUNT}:role/{ROLE_NAME}",
        )
        self.answer(
            "amp_describe-rule-groups-namespace",
            _amp_definition(
                "ruleGroupsNamespace",
                (ROOT / "deploy/observability/amp-rules.yaml").read_bytes(),
            ),
        )
        self.answer("amp_describe-rule-groups-namespace.text", "ACTIVE")
        self.answer(
            "amp_describe-alert-manager-definition",
            _amp_definition(
                "alertManagerDefinition", _rendered_alertmanager().encode("utf-8")
            ),
        )
        self.answer("amp_describe-alert-manager-definition.text", "ACTIVE")

    def answer(self, key: str, body: str) -> None:
        (self.responses / key).write_text(body, encoding="utf-8")

    def missing(self, *keys: str) -> None:
        """Make these reads fail the way AWS fails on a resource that is absent."""

        for key in keys:
            (self.responses / key).unlink(missing_ok=True)
            (self.responses / f"{key}.text").unlink(missing_ok=True)
        self.absent = tuple(dict.fromkeys((*self.absent, *keys)))

    def run(self) -> subprocess.CompletedProcess[str]:
        completed = self.attempt()
        assert completed.returncode == 0, (
            f"the installer failed:\n{completed.stdout}\n{completed.stderr}"
        )
        return completed

    def attempt(self) -> subprocess.CompletedProcess[str]:
        environment = {
            # The fakes shadow the real CLIs; everything else the script needs
            # (bash, jq, sed, base64, cmp) comes from the inherited PATH.
            "PATH": os.pathsep.join(
                (str(self.bin), os.environ.get("PATH", "/usr/bin:/bin"))
            ),
            "HOME": str(self.tmp_path),
            "AWS_REGION": REGION,
            "CPU_EKS_CLUSTER": CLUSTER,
            "CPU_KUBECONFIG": str(self.tmp_path / "cpu.kubeconfig"),
            "AMP_WORKSPACE_ID": WORKSPACE_ID,
            "SNS_TOPIC_NAME": TOPIC_NAME,
            "IAM_ROLE_NAME": ROLE_NAME,
            "NAMESPACE": NAMESPACE,
            "GPU_FAULT_ADOT_IMAGE": ADOT_IMAGE,
            "GPU_FAULT_ENABLE_ADOT": "true",
            "GPU_FAULT_ENABLE_AMP": "true",
            "GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION": "true",
            "FAKE_CALL_LOG": str(self.log),
            "FAKE_RESPONSES": str(self.responses),
            "FAKE_CLUSTER_ARN": CLUSTER_ARN,
            "FAKE_APPLY_OUTPUT": self.apply_output,
            "FAKE_ADOT_REPLICAS": self.adot_replicas,
            "FAKE_ABSENT": " ".join(self.absent),
            **self.environment,
        }
        return subprocess.run(
            ["bash", str(SCRIPT), *self.arguments],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls(self) -> list[str]:
        return [
            line
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def matching(self, fragment: str) -> list[str]:
        return [line for line in self.calls() if fragment in line]


def test_a_converged_installer_writes_nothing(tmp_path: Path) -> None:
    """A rerun against an account that already matches must make no writes.

    ``put-role-policy`` and ``set-topic-attributes`` are audited IAM/SNS
    mutations, and issuing them on every deploy makes the audit trail useless for
    spotting a real change. The ADOT restart costs the collector's termination
    grace plus a 300s ``rollout status`` on a Pod whose ConfigMap and Deployment
    are byte-identical to what is already running.
    """

    installer = Installer(tmp_path)

    result = installer.run()

    for fragment in (
        "iam put-role-policy",
        "iam create-role",
        "sns set-topic-attributes",
        "sns create-topic",
        "rollout restart",
        "amp put-rule-groups-namespace",
        "amp create-rule-groups-namespace",
        "amp put-alert-manager-definition",
        "amp create-alert-manager-definition",
    ):
        assert not installer.matching(fragment), (
            f"a converged installer still ran: {fragment}"
        )
    assert "amp-step-elapsed" in result.stdout, (
        "the per-step timings never reach the deploy log, which captures stdout"
    )


def test_a_drifted_inline_policy_is_rewritten(tmp_path: Path) -> None:
    """Skipping the write must depend on the document, not on the role existing."""

    installer = Installer(tmp_path)
    installer.answer(
        "iam_get-role-policy",
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": ["aps:*"], "Resource": "*"}
                ],
            }
        ),
    )

    installer.run()

    assert installer.matching("iam put-role-policy"), (
        "a widened inline policy was left in place"
    )


def test_a_missing_role_is_created_with_its_policy(tmp_path: Path) -> None:
    """A first run still creates the role and attaches the inline policy."""

    installer = Installer(tmp_path)
    installer.missing("iam_get-role", "iam_get-role-policy")

    installer.run()

    assert installer.matching("iam create-role"), "the AMP writer role was not created"
    assert installer.matching("iam put-role-policy"), (
        "the created role was left without its inline policy"
    )


def test_a_drifted_topic_policy_is_rewritten(tmp_path: Path) -> None:
    """The AMP publish statement must be restored when it is missing."""

    installer = Installer(tmp_path)
    installer.answer(
        "sns_get-topic-attributes.text", _sns_policy(with_amp_statement=False)
    )

    installer.run()

    assert installer.matching("sns set-topic-attributes"), (
        "Alertmanager was left unable to publish to the topic"
    )


def test_a_missing_topic_is_created(tmp_path: Path) -> None:
    """A topic that does not answer get-topic-attributes is created once."""

    installer = Installer(tmp_path)
    installer.missing("sns_get-topic-attributes")
    installer.answer("sns_create-topic.text", TOPIC_ARN)
    installer.answer("sns_created-topic-policy", _sns_policy(with_amp_statement=False))

    installer.run()

    assert installer.matching("sns create-topic"), "the alert topic was never created"
    assert installer.matching("sns set-topic-attributes"), (
        "a new topic was left without the AMP publish statement"
    )


def test_a_topic_that_cannot_be_read_stops_the_installer(tmp_path: Path) -> None:
    """Skipping a write on a failed read would be the fail-open version of this.

    ``get-topic-attributes`` failing means "absent" only if creating the topic
    then makes it readable. A read that keeps failing -- a denied SNS permission,
    a throttle -- must stop the release rather than let the deploy continue with a
    topic whose policy nobody has seen.
    """

    installer = Installer(tmp_path)
    installer.missing("sns_get-topic-attributes")
    installer.answer("sns_create-topic.text", TOPIC_ARN)

    completed = installer.attempt()

    assert completed.returncode != 0, (
        "an unreadable topic policy was treated as converged"
    )
    assert not installer.matching("kubectl apply"), (
        "the installer went on to apply the collector"
    )


def test_a_reordered_topic_policy_is_left_alone(tmp_path: Path) -> None:
    """Statement order carries no meaning, and SNS may reorder what it stores.

    Comparing the raw order would make every deploy rewrite the policy of a topic
    that already says exactly what it should -- the same audited write this change
    exists to stop.
    """

    installer = Installer(tmp_path)
    installer.answer(
        "sns_get-topic-attributes.text",
        _sns_policy(with_amp_statement=True, amp_statement_first=True),
    )

    installer.run()

    assert not installer.matching("sns set-topic-attributes"), (
        "a reordered but equivalent topic policy was rewritten"
    )


@pytest.mark.parametrize(
    ("apply_output", "detail"),
    [
        (
            "configmap/gpu-fault-adot configured\n"
            "deployment.apps/gpu-fault-adot unchanged",
            "a changed collector config",
        ),
        (
            "configmap/gpu-fault-adot unchanged\ndeployment.apps/gpu-fault-adot created",
            "a recreated collector",
        ),
    ],
)
def test_a_changed_collector_is_restarted(
    tmp_path: Path, apply_output: str, detail: str
) -> None:
    """The collector reads its ConfigMap once, at start."""

    installer = Installer(tmp_path)
    installer.apply_output = apply_output

    installer.run()

    assert installer.matching("rollout restart"), (
        f"{detail} was applied without restarting the collector"
    )


def test_an_operator_can_force_the_collector_restart(tmp_path: Path) -> None:
    """An escape hatch for the case the apply output cannot describe.

    A Pod that is running the right manifest but has stale credentials or a wedged
    exporter is not visible in ``kubectl apply`` output, so the restart the
    installer now skips has to remain reachable without editing the manifest.
    """

    installer = Installer(tmp_path)
    installer.environment = {"GPU_FAULT_FORCE_ADOT_RESTART": "true"}

    installer.run()

    assert installer.matching("rollout restart"), (
        "GPU_FAULT_FORCE_ADOT_RESTART did not restart the collector"
    )


def test_the_installer_parses(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


# --- the per-cluster expected-collector rules (F10 fix 1, F3) --------------------

EXPECTED_NAMESPACE = "gpu-fault-dataplane-expected"
STATIC_NAMESPACE = "gpu-fault-control-plane-capacity"


def test_the_expected_rules_namespace_is_left_alone_without_an_argument(
    tmp_path: Path,
) -> None:
    """The bootstrap runs this script with no arguments and must not delete the
    rules a deploy rendered: only the release engine knows the expected set."""
    installer = Installer(tmp_path)

    completed = installer.run()

    assert installer.matching(EXPECTED_NAMESPACE) == []
    assert "left as-is" in completed.stdout


def test_rendered_expected_rules_are_put_under_their_own_namespace(
    tmp_path: Path,
) -> None:
    """The static namespace stays byte-stable; the rendered rules get their own,
    written with the same put/create + wait-for-ACTIVE path as the static file."""
    installer = Installer(tmp_path)
    rules = tmp_path / "expected.yaml"
    rules.write_text(
        "groups:\n- name: gpu-fault-dataplane-expected\n  rules: []\n", encoding="utf-8"
    )
    installer.arguments = ["--dataplane-expected-rules", str(rules)]

    installer.run()

    puts = installer.matching("amp put-rule-groups-namespace")
    assert len(puts) == 1, puts
    assert f"--name {EXPECTED_NAMESPACE}" in puts[0]
    assert f"fileb://{rules}" in puts[0]
    assert not any(f"--name {STATIC_NAMESPACE}" in line for line in puts), (
        "the converged static namespace was rewritten"
    )
    assert installer.matching("amp delete-rule-groups-namespace") == []


def test_matching_expected_rules_are_not_rewritten(tmp_path: Path) -> None:
    """The fake answers every describe with the checked-in static rules, so
    handing it that very file is the converged case: no put, no wait."""
    installer = Installer(tmp_path)
    installer.arguments = [
        "--dataplane-expected-rules",
        str(ROOT / "deploy/observability/amp-rules.yaml"),
    ]

    completed = installer.run()

    assert installer.matching("put-rule-groups-namespace") == []
    assert "already matches the rendered per-cluster rules" in completed.stdout


def test_no_expected_rules_deletes_the_namespace_when_present(tmp_path: Path) -> None:
    """Deleting on an empty expected set is the honest choice: an empty group
    would keep a namespace nobody reads, and a stale per-cluster rule would fire
    for a cluster whose role was removed."""
    installer = Installer(tmp_path)
    installer.arguments = ["--no-dataplane-expected-rules"]

    installer.run()

    deletes = installer.matching("amp delete-rule-groups-namespace")
    assert len(deletes) == 1, deletes
    assert f"--name {EXPECTED_NAMESPACE}" in deletes[0]
    assert installer.matching("put-rule-groups-namespace") == []


def test_no_expected_rules_is_idempotent_when_the_namespace_is_absent(
    tmp_path: Path,
) -> None:
    installer = Installer(tmp_path)
    installer.missing("amp_describe-rule-groups-namespace")
    # The static namespace is then created and waited for; the wait polls the
    # status query, which has to answer.
    installer.answer("amp_describe-rule-groups-namespace.text", "ACTIVE")
    installer.arguments = ["--no-dataplane-expected-rules"]

    completed = installer.run()

    assert installer.matching("amp delete-rule-groups-namespace") == []
    assert "nothing to delete" in completed.stdout


def test_an_unknown_argument_stops_the_installer(tmp_path: Path) -> None:
    installer = Installer(tmp_path)
    installer.arguments = ["--bogus"]

    completed = installer.attempt()

    assert completed.returncode == 2, completed.stderr
    assert "unknown argument" in completed.stderr
    assert installer.calls() == [], "an argument error still ran the installer"
