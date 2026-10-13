"""The subject line of the administrator's alert mail.

Without a ``subject`` the AMP Alertmanager SNS notifier uses its default
``[STATUS:N] <every group label value, sorted by label name>``, and the
subject is cut at 100 characters: ``alertname`` (up to 48 characters),
``cluster_id`` and the site constant ``control_plane_cluster`` filled it, so
the administrator never saw which node an alert was about. The template puts
the cluster and the node first; the message body still lists every label and
annotation.

AMP renders the template with Go ``text/template`` and Alertmanager's
functions. No Go toolchain is available to the tests, so ``render`` below
interprets exactly the actions the template uses and refuses any other: a
template change that needs more must extend it deliberately. The expected
subjects below are what ``amtool template render`` of Alertmanager v0.27
prints for the same data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ALERTMANAGER = ROOT / "deploy/observability/amp-alertmanager.yaml"
RULES = ROOT / "deploy/observability/amp-rules.yaml"
SNS_SUBJECT_LIMIT = 100
# More alerts than one cluster/node group ever holds.
FIRING_BOUND = 99_999
# Stand-ins exactly as long as the longest identities a regional site runs
# with: a HyperPod cluster id, an EKS cluster id and both node-name shapes.
CLUSTER_IDS = ("hyperpod-cluster-example-site", "gpu-eks-cluster")
GPU_NODES = (
    "ip-10-0-38-180.us-west-2.compute.internal",
    "hyperpod-i-00000000000000001",
)
ACTION = re.compile(r"{{(.*?)}}")


def sns_config() -> dict[str, Any]:
    document = yaml.safe_load(ALERTMANAGER.read_text(encoding="utf-8"))
    config = yaml.safe_load(document["alertmanager_config"])
    receivers = {item["name"]: item for item in config["receivers"]}
    [sns] = receivers["gpu-fault-sns"]["sns_configs"]
    return dict(sns)


def group_by() -> list[str]:
    document = yaml.safe_load(ALERTMANAGER.read_text(encoding="utf-8"))
    return list(yaml.safe_load(document["alertmanager_config"])["route"]["group_by"])


def subject_template() -> str:
    subject = sns_config().get("subject")
    assert isinstance(subject, str), "gpu-fault-sns has no subject template"
    return subject


def longest_alertname() -> str:
    document = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    names = [
        str(rule["alert"])
        for group in document["groups"]
        for rule in group["rules"]
        if "alert" in rule
    ]
    assert names, "amp-rules.yaml defines no alert"
    return max(names, key=len)


def _value(action: str, *, status: str, firing: int, dot: str) -> str:
    if action == ".Status | toUpper":
        return status.upper()
    if action == ".Alerts.Firing | len":
        return str(firing)
    if action == ".":
        return dot
    replace = re.fullmatch(r'reReplaceAll "([^"$\\]*)" "([^"$\\]*)" \.', action)
    if replace is not None:
        return re.sub(replace.group(1), replace.group(2), dot)
    raise AssertionError(f"render() does not interpret {{{{ {action} }}}}")


def render(template: str, *, status: str, firing: int, labels: dict[str, str]) -> str:
    """The subject Alertmanager sends for one alert group.

    ``labels`` is ``.GroupLabels``: the ``group_by`` labels the group's alerts
    carry. A label the group lacks renders as the empty string, which ``with``
    treats as false -- Alertmanager's templates run with ``missingkey=zero``.
    """

    out: list[str] = []
    # (emitting, dot) per open block; the bottom entry is the template itself.
    stack: list[tuple[bool, str]] = [(True, "")]
    position = 0
    for match in ACTION.finditer(template):
        emitting, dot = stack[-1]
        if emitting:
            out.append(template[position : match.start()])
        position = match.end()
        action = match.group(1).strip()
        if action == "end":
            assert len(stack) > 1, "unbalanced {{ end }}"
            stack.pop()
            continue
        if action == 'if eq .Status "firing"':
            stack.append((emitting and status == "firing", dot))
            continue
        label = re.fullmatch(r"with \.GroupLabels\.(\w+)", action)
        if label is not None:
            value = labels.get(label.group(1), "")
            stack.append((emitting and bool(value), value))
            continue
        value = _value(action, status=status, firing=firing, dot=dot)
        if emitting:
            out.append(value)
    assert len(stack) == 1, "unclosed block in the subject template"
    out.append(template[position:])
    return "".join(out)


def test_subject_is_one_ascii_line_and_the_body_stays_the_default() -> None:
    sns = sns_config()
    subject = subject_template()

    assert subject.isascii(), "SNS subjects must be ASCII"
    assert "\n" not in subject and "\r" not in subject, "the subject is one line"
    assert sns.get("send_resolved") is True, "resolved notifications still mail"
    # The default body lists every label and annotation of every alert; the
    # site constants dropped from the subject stay readable there.
    assert "message" not in sns, "the default message body must stay"


def test_subject_names_cluster_node_alert_and_severity_in_that_order() -> None:
    subject = subject_template()
    references = re.findall(r"\.(Status|GroupLabels\.\w+)", subject)

    assert [item for item in references if item != "Status"] == [
        "GroupLabels.cluster_id",
        "GroupLabels.gpu_node",
        "GroupLabels.alertname",
        "GroupLabels.severity",
    ], references
    assert subject.index(".Status") < subject.index(".GroupLabels."), subject
    # .GroupLabels holds nothing but the route's group_by labels.
    assert {
        item.removeprefix("GroupLabels.")
        for item in references
        if item.startswith("GroupLabels.")
    } <= set(group_by()), group_by()
    for constant in ("control_plane_cluster", "region"):
        assert constant not in subject, f"site constant {constant} wastes the subject"


@pytest.mark.parametrize("cluster_id", CLUSTER_IDS)
@pytest.mark.parametrize("gpu_node", GPU_NODES)
def test_cluster_and_node_survive_the_100_character_sns_cut(
    cluster_id: str, gpu_node: str
) -> None:
    alertname = longest_alertname()
    labels = {
        "alertname": alertname,
        "severity": "critical",
        "control_plane_cluster": "gpu-fault-regional-control-plane",
        "region": "us-west-2",
        "cluster_id": cluster_id,
        "gpu_node": gpu_node,
    }

    subject = render(
        subject_template(), status="firing", firing=FIRING_BOUND, labels=labels
    )
    delivered = subject[:SNS_SUBJECT_LIMIT]

    assert delivered.startswith(f"[FIRING:{FIRING_BOUND}] {cluster_id} {gpu_node}"), (
        f"cluster and node must lead the subject: {delivered!r}"
    )
    assert f"{cluster_id} {gpu_node}" in delivered, (
        f"the {SNS_SUBJECT_LIMIT}-character cut reached the node: {subject!r}"
    )


def test_subject_drops_the_product_prefix_and_ends_with_the_severity() -> None:
    labels = {
        "alertname": "GpuFaultExclusiveNodeOwnershipInvariantViolation",
        "severity": "critical",
        "cluster_id": "gpu-eks-cluster",
        "gpu_node": "ip-10-0-42-179.us-west-2.compute.internal",
    }

    firing = render(subject_template(), status="firing", firing=2, labels=labels)
    resolved = render(subject_template(), status="resolved", firing=0, labels=labels)

    assert firing == (
        "[FIRING:2] gpu-eks-cluster ip-10-0-42-179.us-west-2.compute.internal "
        "ExclusiveNodeOwnershipInvariantViolation critical"
    ), firing
    assert resolved == (
        "[RESOLVED] gpu-eks-cluster ip-10-0-42-179.us-west-2.compute.internal "
        "ExclusiveNodeOwnershipInvariantViolation critical"
    ), resolved


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        # A cluster-wide alert has no gpu_node.
        (
            {
                "alertname": "GpuFaultCollectorSilent",
                "severity": "critical",
                "cluster_id": "gpu-eks-cluster",
            },
            "[FIRING:1] gpu-eks-cluster CollectorSilent critical",
        ),
        # A control-plane alert names no cluster either.
        (
            {"alertname": "GpuFaultMetricsContributorFailing", "severity": "warning"},
            "[FIRING:1] MetricsContributorFailing warning",
        ),
    ],
)
def test_a_label_the_group_lacks_leaves_no_empty_token(
    labels: dict[str, str], expected: str
) -> None:
    subject = render(subject_template(), status="firing", firing=1, labels=labels)

    assert subject == expected, subject
    assert "  " not in subject and subject == subject.strip(), repr(subject)


def test_render_refuses_actions_it_does_not_interpret() -> None:
    with pytest.raises(AssertionError, match="does not interpret"):
        render("{{ .CommonLabels.alertname }}", status="firing", firing=1, labels={})
    with pytest.raises(AssertionError, match="unclosed block"):
        render(
            "{{ with .GroupLabels.gpu_node }}{{ . }}",
            status="firing",
            firing=1,
            labels={},
        )
