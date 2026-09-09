"""Per-cluster expected-collector rules are rendered at release time (F10 fix 1, F3).

The static ``absent(up{job="gpu-fault-dataplane"} == 1)`` half of
``GpuFaultDataplaneCollectorMissing`` fired forever on every site with an AMP
workspace and no collector yet -- an SNS page every four hours for a component
the site had deliberately not enabled. The absence half now comes from rules the
release renders from the set of clusters that carry an IRSA role (the same set
the observability digest already folds), one ``absent(...)`` per expected
cluster, put under their own AMP rule-groups namespace so the static file stays
byte-stable and deleted when no cluster is expected to have a collector.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_dataplane_observability as DATAPLANE
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import ReleaseError
from tests._script_loader import lazy_script_module
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = lazy_script_module(ROOT / "scripts/verify-regional-alerting.py")
ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-adot-writer"


def _target(cluster_id: str, *, role: str | None = ROLE_ARN) -> SimpleNamespace:
    return SimpleNamespace(
        cluster_id=cluster_id,
        context=f"{cluster_id}-context",
        region="us-east-1",
        adot_irsa_role_arn=role,
    )


def _release(*clusters: SimpleNamespace, workspace: str | None = "ws-a") -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(
            clusters=clusters,
            aws_region="us-east-1",
            health=SimpleNamespace(amp_workspace_id=workspace),
        )
    )


def _rules(text: str) -> list[dict[str, Any]]:
    payload = yaml.safe_load(text)
    assert list(payload) == ["groups"], payload
    assert [group["name"] for group in payload["groups"]] == [
        DATAPLANE.DATAPLANE_EXPECTED_RULE_GROUP
    ]
    return list(payload["groups"][0]["rules"])


def test_each_cluster_with_a_role_gets_its_own_absent_rule() -> None:
    """One rule per expected cluster; the cluster without a role gets none, so a
    site that has not enabled the collector anywhere renders nothing."""
    text = DATAPLANE.render_dataplane_expected_rules(
        _release(_target("gpu-a"), _target("gpu-b", role=None), _target("gpu-c"))
    )

    assert text is not None
    rules = _rules(text)
    assert [rule["alert"] for rule in rules] == [
        DATAPLANE.DATAPLANE_COLLECTOR_ALERT,
        DATAPLANE.DATAPLANE_COLLECTOR_ALERT,
    ]
    assert [rule["labels"]["gpu_cluster"] for rule in rules] == ["gpu-a", "gpu-c"]
    rule = rules[0]
    assert " ".join(str(rule["expr"]).split()) == (
        'absent(up{job="gpu-fault-dataplane", gpu_cluster="gpu-a"} == 1)'
    )
    assert rule["for"] == "15m"
    assert rule["labels"] == {
        "severity": "warning",
        "gpu_cluster": "gpu-a",
        "control_plane_cluster": DATAPLANE.DATAPLANE_CONTROL_PLANE_CLUSTER,
        "region": "us-east-1",
    }
    annotations = rule["annotations"]
    assert annotations["runbook_url"] == DATAPLANE.DATAPLANE_COLLECTOR_RUNBOOK_URL
    assert "gpu-a" in annotations["summary"]
    for needle in ("gpu-a", "gpu-fault-admin deploy", "adot_irsa_role_arn"):
        assert needle in annotations["description"], needle
    assert "kubectl apply -f deploy/" not in annotations["description"]


def test_the_rendered_rules_satisfy_the_static_alert_contracts() -> None:
    """The same checks the checked-in rules pass: runbook card exists and is
    two-way, summary and description present and distinct, no aggregation that
    drops the Alertmanager grouping labels."""
    text = DATAPLANE.render_dataplane_expected_rules(_release(_target("gpu-a")))
    assert text is not None
    rules = _rules(text)
    static = yaml.safe_load(
        (ROOT / "deploy/observability/amp-rules.yaml").read_text(encoding="utf-8")
    )
    static_rules = [rule for group in static["groups"] for rule in group["rules"]]

    # Two-way: the runbook check also demands an alert for every card, so the
    # rendered rules are judged together with the static file AMP evaluates.
    assert VERIFIER.runbook_defects(ROOT, static_rules + rules) == []
    assert VERIFIER.annotation_defects(rules) == []
    assert VERIFIER.aggregation_defects(ROOT, rules) == []


def test_the_rule_labels_match_the_collectors_static_relabels() -> None:
    """``control_plane_cluster`` on the rendered rule must be the value the
    collector stamps on every series, or the rendered alert lands in its own
    Alertmanager group."""
    documents = yaml.safe_load_all(
        (ROOT / "deploy/dataplane/adot-dataplane.yaml").read_text(encoding="utf-8")
    )
    config_map = next(d for d in documents if d and d.get("kind") == "ConfigMap")
    collector = yaml.safe_load(config_map["data"]["collector.yaml"])
    job = next(
        item
        for item in collector["receivers"]["prometheus"]["config"]["scrape_configs"]
        if item["job_name"] == "gpu-fault-dataplane"
    )
    replacement = next(
        relabel["replacement"]
        for relabel in job["relabel_configs"]
        if relabel.get("target_label") == "control_plane_cluster"
    )

    assert replacement == DATAPLANE.DATAPLANE_CONTROL_PLANE_CLUSTER


def test_no_rules_without_a_workspace_or_without_any_role() -> None:
    assert (
        DATAPLANE.render_dataplane_expected_rules(_release(_target("gpu-a", role=None)))
        is None
    )
    assert (
        DATAPLANE.render_dataplane_expected_rules(
            _release(_target("gpu-a"), workspace=None)
        )
        is None
    )


def test_a_cluster_id_that_cannot_be_quoted_into_promql_fails_closed() -> None:
    with pytest.raises(ReleaseError, match="cluster_id"):
        DATAPLANE.render_dataplane_expected_rules(_release(_target('gpu"a')))


# --- wiring into the observability install --------------------------------------


class _Runner:
    dry_run = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any], str | None]] = []

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        handed = None
        if "--dataplane-expected-rules" in arguments:
            path = Path(arguments[arguments.index("--dataplane-expected-rules") + 1])
            # The file lives in a temporary directory the release removes once
            # the installer returns, so it is read while the call is happening.
            handed = path.read_text(encoding="utf-8")
        self.calls.append((list(arguments), kwargs, handed))
        return ""


def _config(tmp_path: Path, *, adot_irsa_role_arn: str | None, name: str) -> Path:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    if adot_irsa_role_arn:
        value["clusters"][0]["adot_irsa_role_arn"] = adot_irsa_role_arn
    value["health"] = {
        "amp_workspace_id": "ws-a",
        "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts",
    }
    target = tmp_path / name
    target.write_text(json.dumps(value))
    return target


def _real_release(path: Path) -> Any:
    return MODULE.RegionalRelease(MODULE.ReleaseConfig.load(path), _Runner())


def test_the_observability_install_hands_the_rendered_rules_to_the_installer(
    tmp_path: Path,
) -> None:
    """Every OBSERVABILITY node re-runs the installer, so re-putting the rendered
    rules here is what keeps them current whenever the digest moves. The
    release's observability step builds the installer environment and delegates
    to this runner (the orchestration tests stub that step by name)."""
    release = _real_release(_config(tmp_path, adot_irsa_role_arn=ROLE_ARN, name="a"))

    DATAPLANE.run_amp_monitoring_installer(release, {"AMP_WORKSPACE_ID": "ws-a"})

    (arguments, kwargs, handed), *rest = release.runner.calls
    assert rest == []
    assert arguments[:2] == ["bash", str(DATAPLANE.AMP_MONITORING_INSTALLER)]
    assert "--dataplane-expected-rules" in arguments
    assert handed == DATAPLANE.render_dataplane_expected_rules(release)
    assert kwargs["env"] == {"AMP_WORKSPACE_ID": "ws-a"}


def test_the_observability_install_asks_for_deletion_when_nothing_is_expected(
    tmp_path: Path,
) -> None:
    release = _real_release(_config(tmp_path, adot_irsa_role_arn=None, name="b"))

    DATAPLANE.run_amp_monitoring_installer(release, {})

    (arguments, _kwargs, handed), *rest = release.runner.calls
    assert rest == []
    assert arguments[2:] == ["--no-dataplane-expected-rules"], arguments
    assert handed is None


def test_the_rendered_rules_are_an_input_of_the_observability_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change to the rule template alone has to move the digest, or a site
    keeps the old rendered rules until something else about observability moves."""
    config = _config(tmp_path, adot_irsa_role_arn=ROLE_ARN, name="c")
    baseline = _real_release(config).observability_adot_digest
    monkeypatch.setattr(
        MODULE, "dataplane_expected_rules_sha256", lambda _release: "f" * 64
    )

    assert _real_release(config).observability_adot_digest != baseline


# --- the previous definition is captured and put back on rollback ------------------


def _amp_release(*, present: bool) -> tuple[Any, list[list[str]], list[bytes]]:
    """A fake AMP whose namespace starts ``present`` and follows the writes: a
    delete makes it gone and a create makes it exist, so the restore's wait after
    each write reads what AMP would report (ACTIVE at once, or absent)."""
    calls: list[list[str]] = []
    written: list[bytes] = []
    state = {"present": present}

    def run(arguments: list[str], **_kwargs: Any) -> str:
        calls.append(list(arguments))
        if arguments[2] == "delete-rule-groups-namespace":
            state["present"] = False
        elif arguments[2] == "create-rule-groups-namespace":
            state["present"] = True
        if "--data" in arguments:
            path = arguments[arguments.index("--data") + 1].removeprefix("fileb://")
            written.append(Path(path).read_bytes())
        return ""

    def probe_output(arguments: list[str], **_kwargs: Any) -> tuple[int, str, str]:
        # Existence is read from describe's exit and stderr (LOW-5), never from a
        # boolean probe that reads a throttle as "absent".
        assert arguments[:3] == ["aws", "amp", "describe-rule-groups-namespace"], (
            arguments
        )
        if state["present"]:
            payload = {"ruleGroupsNamespace": {"status": {"statusCode": "ACTIVE"}}}
            return 0, json.dumps(payload), ""
        return 254, "", "An error occurred (ResourceNotFoundException) when calling"

    release = SimpleNamespace(
        config=SimpleNamespace(
            aws_region="us-east-1", health=SimpleNamespace(amp_workspace_id="ws-a")
        ),
        runner=SimpleNamespace(run=run, probe_output=probe_output),
    )
    return release, calls, written


def test_restoring_a_present_definition_puts_or_creates_it() -> None:
    data = base64.b64encode(b"groups: []\n").decode()

    release, calls, written = _amp_release(present=True)
    assert (
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": True, "data_base64": data}
        )
        == "restored"
    )
    assert [c[2] for c in calls] == ["put-rule-groups-namespace"]
    assert written == [b"groups: []\n"]
    assert DATAPLANE.DATAPLANE_EXPECTED_RULE_NAMESPACE in calls[0]

    release, calls, _written = _amp_release(present=False)
    DATAPLANE.restore_dataplane_expected_rules(
        release, {"present": True, "data_base64": data}
    )
    assert [c[2] for c in calls] == ["create-rule-groups-namespace"]


def test_restoring_an_absent_definition_deletes_only_what_exists() -> None:
    release, calls, _written = _amp_release(present=True)
    assert (
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": False, "data_base64": None}
        )
        == "deleted"
    )
    assert [c[2] for c in calls] == ["delete-rule-groups-namespace"]

    release, calls, _written = _amp_release(present=False)
    assert (
        DATAPLANE.restore_dataplane_expected_rules(
            release, {"present": False, "data_base64": None}
        )
        == "absent"
    )
    assert calls == []


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "snapshot", [None, {"present": True}, {"present": True, "data_base64": "%%%"}]
)
def test_an_invalid_expected_rules_snapshot_refuses_before_mutation(
    snapshot: object,
) -> None:
    release, calls, _written = _amp_release(present=True)

    with pytest.raises(ReleaseError, match="expected-collector rules snapshot"):
        DATAPLANE.restore_dataplane_expected_rules(release, snapshot)

    assert calls == []
