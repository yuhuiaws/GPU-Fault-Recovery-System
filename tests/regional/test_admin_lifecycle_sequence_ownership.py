"""The BOOT-029 driver and the ownership label: proven gone after stage 2 and
stage 4, put back with the ``site-id`` commands before stage 3 and stage 5.

The fakes come from the single-provider contract
(``test_admin_lifecycle_sequence_script``): the fake ``gpu-fault-admin``
releases the label when it removes or uninstalls (a marker the fake ``aws``
reads), journals the step, and the fake ``aws`` puts the label back on the
operator's ``update-cluster`` / ``update-nodegroup-config`` call.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.regional.test_admin_lifecycle_sequence_script import (
    GPU_ARN,
    _admin_verbs,
    _install_fakes,
    _record,
    _run,
    _sha256,
    _tool_log,
)


def _aws_lines(lines: list[str], operation: str) -> list[int]:
    return [index for index, line in enumerate(lines) if f" {operation} " in line]


def test_the_label_is_proven_gone_after_de_registration_and_put_back_before_re_join(
    tmp_path: Path,
) -> None:
    """Stage 2 and stage 4 release gpu-fault.io/control-plane=<site id> from the
    node groups the site selected; the driver proves it gone with the product's
    describe reads, then puts it back -- with the operator's credentials and the
    site-id commands -- before stage 3 re-joins and before stage 5 re-deploys."""

    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    verbs = {
        verb: next(
            index
            for index, line in enumerate(lines)
            if line.startswith(f"gpu-fault-admin {verb} ")
        )
        for verb in ("remove-cluster", "join-cluster", "uninstall")
    }
    redeploy = [index for index, line in enumerate(lines) if "admin deploy " in line][1]
    relabels = _aws_lines(lines, "update-cluster")
    assert len(relabels) == 2, (
        "the HyperPod instance group is re-labelled exactly twice: before the "
        f"re-join and before the re-deploy ({relabels})"
    )
    assert verbs["remove-cluster"] < relabels[0] < verbs["join-cluster"], (
        "stage 3 puts the label back after stage 2 released it and before joining"
    )
    assert verbs["uninstall"] < relabels[1] < redeploy, (
        "stage 5 puts the label back after stage 4 released it and before deploying"
    )
    assert not _aws_lines(lines, "update-nodegroup-config"), (
        "the fixture's cluster is HyperPod: no EKS node group is written"
    )
    for index in relabels:
        assert "--cluster-name fixture-hyperpod --instance-groups " in lines[index], (
            lines[index]
        )
    describes_after_removal = [
        index
        for index in _aws_lines(lines, "describe-cluster")
        if verbs["remove-cluster"] < index < relabels[0]
    ]
    assert describes_after_removal, (
        "the driver reads the instance groups after the removal to prove the label gone"
    )
    assert not (tmp_path / "ownership-released").exists(), (
        "the label is back on the node groups when the sequence ends"
    )

    record = _record(tmp_path)
    for number in ("2", "4"):
        released = record["stages"][number]["ownership_label_released"]
        assert released["ownership_label"] == "gpu-fault.io/control-plane"
        assert [item["status"] for item in released["clusters"]] == ["released"], (
            number,
            released,
        )
        assert released["clusters"][0]["labelled_groups"] == 0
    for number in ("3", "5"):
        reapplied = record["stages"][number]["ownership_label_reapplied"]
        assert reapplied["stage"] == int(number)
        (cluster,) = reapplied["clusters"]
        assert cluster["provider"] == "hyperpod" and cluster["applied"] == 1, cluster
        assert cluster["groups_sha256"] == [f"sha256:{_sha256('spot-p5en')}"]
    assert (
        record["stages"]["3"]["ownership_label_reapplied"]["clusters"][0][
            "journal_path_sha256"
        ]
        == f"sha256:{_sha256('remove-cluster/cluster-a/state.json')}"
    ), "the journal the re-apply read enters the record as a digest"
    assert (
        record["stages"]["5"]["ownership_label_reapplied"]["clusters"][0][
            "journal_path_sha256"
        ]
        == f"sha256:{_sha256('uninstall/state.json')}"
    )
    (journal,) = record["stages"]["2"]["journals"]
    assert journal["ownership_release"] == {
        "outcome": "released",
        "released_groups": 1,
        "retained_groups": 0,
        "manual": 0,
    }, "the stage-2 receipt carries what the removal journal recorded"
    assert record["stages"]["4"]["ownership_release"]["outcome"] == "released"
    assert (
        record["stages"]["4"]["ownership_release"]["clusters"][0]["released_groups"]
        == 1
    )
    serialized = json.dumps(record)
    for secret in (GPU_ARN, "spot-p5en", "fixture-hyperpod"):
        assert secret not in serialized, f"{secret} leaked into the record"


def test_a_de_registration_that_keeps_the_label_fails_the_stage(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, {**env, "FAKE_ADMIN_KEEP_LABEL": "1"})

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 2 (remove-cluster)" in result.stderr, result.stderr
    assert "did not release the ownership label" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == ["deploy", "remove-cluster"], (
        "stage 3 never starts on a stale claim"
    )
    stage = _record(tmp_path)["stages"]["2"]
    assert stage["status"] == "FAIL", stage
    log = (tmp_path / "state/acceptance" / stage["log_name"]).read_text()
    assert "still carries the label on instance group(s) spot-p5en" in log, (
        "the stage log names the group that still claims the site"
    )
    assert "journals" in stage and stage["journals"][0]["ownership_release"] == {
        "outcome": "retained",
        "released_groups": 0,
        "retained_groups": 1,
        "manual": 0,
    }, "the journal's own outcome is recorded next to the live refusal"
