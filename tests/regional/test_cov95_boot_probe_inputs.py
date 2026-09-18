from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.boot_guard import mutate, registry


@pytest.mark.parametrize(
    "arguments,length,disabled,removed,added",
    [
        (["31"], 31, False, None, None),
        (["32", "disabled"], 32, True, None, None),
        (
            ["64", "", "eks_cluster_arn", "alowed_namespaces"],
            64,
            False,
            "eks_cluster_arn",
            "alowed_namespaces",
        ),
    ],
)
def test_registry_probe_builds_only_the_selected_negative_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    length: int,
    disabled: bool,
    removed: str | None,
    added: str | None,
) -> None:
    monkeypatch.setattr(registry, "sys", SimpleNamespace(argv=["registry", *arguments]))
    registry.main()
    entries = json.loads(capsys.readouterr().out)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["cluster_id"] == "guardprobe-fake-cluster"
    assert entry["hyperpod_cluster_name"] == "guardprobe-fake-cluster"
    assert len(entry["token"]) == length
    assert entry["agent_endpoint_allowed_cidrs"] == ["192.0.2.0/24"]
    assert entry.get("enabled", True) is (not disabled)
    if removed is not None:
        assert removed not in entry
    else:
        assert ":000000000000:" in entry["eks_cluster_arn"]
    if added is not None:
        assert entry[added] == ["default"]


@pytest.mark.parametrize("arguments,error", [([], SystemExit), (["bad"], ValueError)])
def test_registry_probe_rejects_missing_or_noninteger_length(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], error: type[BaseException]
) -> None:
    monkeypatch.setattr(registry, "sys", SimpleNamespace(argv=["registry", *arguments]))
    with pytest.raises(error):
        registry.main()


def baseline() -> dict[str, Any]:
    return {
        "kind": "Deployment",
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "guard",
                            "env": [
                                {"name": "KEPT", "value": "unchanged"},
                                {
                                    "name": "TARGET",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": "old-config",
                                            "key": "value",
                                        }
                                    },
                                },
                            ],
                        }
                    ]
                }
            }
        },
    }


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["del", "TARGET"], [{"name": "KEPT", "value": "unchanged"}]),
        (
            ["set", "TARGET", "candidate"],
            [
                {"name": "KEPT", "value": "unchanged"},
                {"name": "TARGET", "value": "candidate"},
            ],
        ),
        (["set", "EXTRA", "candidate"], None),
        (
            ["sref", "TARGET", "example-secret", "example-key"],
            [
                {"name": "KEPT", "value": "unchanged"},
                {
                    "name": "TARGET",
                    "valueFrom": {
                        "secretKeyRef": {"name": "example-secret", "key": "example-key"}
                    },
                },
            ],
        ),
    ],
)
def test_probe_manifest_changes_one_environment_entry_without_mutating_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected: list[dict[str, Any]] | None,
) -> None:
    original = baseline()
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(mutate, "BASE", str(path))
    monkeypatch.setattr(
        mutate, "sys", SimpleNamespace(argv=["mutate", *arguments], stdout=sys.stdout)
    )
    mutate.main()
    result = json.loads(capsys.readouterr().out)
    target = copy.deepcopy(original)
    if expected is None:
        expected = original["spec"]["template"]["spec"]["containers"][0]["env"] + [
            {"name": "EXTRA", "value": "candidate"}
        ]
    target["spec"]["template"]["spec"]["containers"][0]["env"] = expected
    assert result == target
    assert json.loads(path.read_text()) == original


@pytest.mark.parametrize(
    "arguments,error",
    [
        ([], SystemExit),
        (["unknown", "TARGET"], SystemExit),
        (["del", "MISSING"], AssertionError),
        (["sref", "MISSING", "example", "key"], AssertionError),
    ],
)
def test_probe_manifest_refuses_unsupported_or_unmet_case_preconditions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    error: type[BaseException],
) -> None:
    original = baseline()
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(mutate, "BASE", str(path))
    monkeypatch.setattr(
        mutate, "sys", SimpleNamespace(argv=["mutate", *arguments], stdout=sys.stdout)
    )
    with pytest.raises(error):
        mutate.main()
    assert capsys.readouterr().out == ""
    assert json.loads(path.read_text()) == original
