from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import failure_domain_map as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.execution.remediation_budget import (
    FAILURE_DOMAIN_MAP_ENV,
    load_failure_domain_map,
)


def _site() -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "clusters": [
                {"cluster_id": "gpu-a", "context": "gpu-a-context"},
                {"cluster_id": "gpu-b", "context": "gpu-b-context"},
            ]
        },
        environment={},
    )


def _inventory(cluster_id: str) -> dict[str, dict]:
    group = f"{cluster_id}-group"
    return {
        f"{cluster_id}-node-1": {
            "metadata": {
                "name": f"{cluster_id}-node-1",
                "labels": {"sagemaker.amazonaws.com/instance-group-name": group},
            }
        },
        f"{cluster_id}-node-2": {
            "metadata": {"name": f"{cluster_id}-node-2", "labels": {}}
        },
    }


def test_map_covers_every_managed_cluster_and_reports_unmapped_nodes() -> None:
    listed: list[str] = []

    def list_nodes(site, cluster_id):
        listed.append(cluster_id)
        return _inventory(cluster_id)

    result = module.build_failure_domain_map(_site(), list_nodes=list_nodes)

    assert listed == ["gpu-a", "gpu-b"]
    assert result.mapping == {
        "gpu-a": {"gpu-a-node-1": "gpu-a-group"},
        "gpu-b": {"gpu-b-node-1": "gpu-b-group"},
    }
    assert result.unmapped == {"gpu-a": ["gpu-a-node-2"], "gpu-b": ["gpu-b-node-2"]}


def test_map_can_be_limited_to_named_clusters_and_rejects_unknown_ones() -> None:
    result = module.build_failure_domain_map(
        _site(),
        cluster_ids=("gpu-b",),
        list_nodes=lambda site, cluster_id: _inventory(cluster_id),
    )
    assert list(result.mapping) == ["gpu-b"]

    with pytest.raises(BootstrapError, match="gpu-z"):
        module.build_failure_domain_map(
            _site(),
            cluster_ids=("gpu-z",),
            list_nodes=lambda site, cluster_id: _inventory(cluster_id),
        )


def test_configmap_carries_the_document_the_executor_loads(tmp_path: Path) -> None:
    mapping = {"gpu-a": {"node-1": "group-a"}}

    manifest = module.failure_domain_configmap(mapping)

    assert manifest["kind"] == "ConfigMap"
    assert manifest["metadata"] == {
        "name": module.FAILURE_DOMAIN_CONFIGMAP,
        "namespace": "gpu-fault-system",
    }
    document = manifest["data"][module.FAILURE_DOMAIN_FILE]
    assert manifest["data"]["map-path"] == module.FAILURE_DOMAIN_MAP_PATH
    # Round trip through the executor-side loader: what the admin renders is
    # exactly what the worker will refuse or accept at start-up.
    path = tmp_path / module.FAILURE_DOMAIN_FILE
    path.write_text(document, encoding="utf-8")
    assert load_failure_domain_map(path) == mapping


def test_cli_renders_the_configmap_for_the_managed_site(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: _site())
    monkeypatch.setattr(
        module, "cluster_nodes", lambda site, cluster_id: _inventory(cluster_id)
    )
    output = tmp_path / "failure-domain-map.json"

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "failure-domain-map",
                "--state-dir",
                str(state_dir),
                "--cluster-id",
                "gpu-a",
                "--output",
                str(output),
            ]
        )
    )

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(manifest["data"][module.FAILURE_DOMAIN_FILE]) == {
        "gpu-a": {"gpu-a-node-1": "gpu-a-group"}
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["unmapped"] == {"gpu-a": ["gpu-a-node-2"]}
    assert summary["environment"] == {
        FAILURE_DOMAIN_MAP_ENV: module.FAILURE_DOMAIN_MAP_PATH
    }
