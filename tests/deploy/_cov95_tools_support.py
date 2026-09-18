from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "deploy/control-plane/tools"
RENDERER = lazy_script_module(TOOLS / "render_control_plane_role_split.py")


def base_deployment() -> dict[str, Any]:
    return next(
        document
        for document in yaml.safe_load_all(
            (
                ROOT / "deploy/control-plane/base/control-plane-deployment.yaml"
            ).read_text()
        )
        if document["kind"] == "Deployment"
        and document["metadata"]["name"] == "gpu-fault-api-ha"
    )


def render_documents(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: dict[str, Any] | None = None,
    as_json: bool = True,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    source = base_deployment() if source is None else source
    raw = (
        json.dumps(source)
        if as_json
        else yaml.safe_dump_all([None, {"kind": "ConfigMap"}, source])
    )
    arguments = ["render_control_plane_role_split"]
    if as_json:
        arguments.append("--json")
    if out_dir is not None:
        arguments.extend(["--out-dir", str(out_dir)])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(sys, "stdin", io.StringIO(raw))
    output = io.StringIO()
    with redirect_stdout(output):
        RENDERER.main()
    if out_dir is not None:
        assert output.getvalue() == "", "file rendering must not emit another manifest"
        return [
            yaml.safe_load((out_dir / name).read_text())
            for name in (out_dir / "manifest-list.txt").read_text().splitlines()
        ]
    document = json.loads(output.getvalue())
    assert (document["apiVersion"], document["kind"]) == ("v1", "List")
    return document["items"]


def by_name(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {document["metadata"]["name"]: document for document in documents}


def effective_env(
    documents: list[dict[str, Any]], deployment_name: str
) -> dict[str, str]:
    resources = by_name(documents)
    container = resources[deployment_name]["spec"]["template"]["spec"]["containers"][0]
    result = {}
    for source in container.get("envFrom", []):
        if "configMapRef" in source:
            result.update(resources[source["configMapRef"]["name"]]["data"])
    result.update(
        {item["name"]: item["value"] for item in container["env"] if "value" in item}
    )
    return result
