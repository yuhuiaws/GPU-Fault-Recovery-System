from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/release-artifact-path.py"
MODULE = lazy_script_module("release_artifact_path", MODULE_PATH)


def test_release_artifact_path_verifies_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "release.whl"
    wheel.write_bytes(b"wheel")
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "wheel": str(wheel),
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            }
        )
    )

    assert MODULE.resolve_artifact(manifest, "wheel") == wheel

    wheel.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash"):
        MODULE.resolve_artifact(manifest, "wheel")


def test_release_artifact_path_verifies_the_release_matches_this_checkout(
    tmp_path: Path,
) -> None:
    # path + sha256 只证明「manifest 点的那个文件没被改过」，不证明它是用当前
    # 这棵树编出来的。manifest 里本来就带 module_digest，但没人比过，于是
    # 手册里每一处 WHEEL="$(release-artifact-path.py wheel)" 都可能装上旧树的
    # wheel，而 /v1/version 的 module_digest 才是唯一能识别进程身份的字段。
    live = MODULE.checkout_module_digest()
    current = json.loads(
        (ROOT / "dist/current-release.json").read_text(encoding="utf-8")
    )

    assert current["module_digest"] == live
    assert MODULE.verify_module_digest(ROOT / "dist/current-release.json") == live

    stale = tmp_path / "release.json"
    stale_document = json.loads(json.dumps(current))
    stale_document["module_digest"] = "0" * 64
    stale_document["components"]["control_plane"]["module_digest"] = "0" * 64
    stale.write_text(json.dumps(stale_document))
    with pytest.raises(RuntimeError, match="built from different code"):
        MODULE.verify_module_digest(stale)

    without = tmp_path / "no-digest.json"
    without_document = {
        key: value for key, value in current.items() if key != "module_digest"
    }
    without_document["components"]["control_plane"].pop("module_digest", None)
    without.write_text(json.dumps(without_document))
    with pytest.raises(RuntimeError, match="no module_digest"):
        MODULE.verify_module_digest(without)


def test_release_artifact_path_checks_the_digest_before_printing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    # MODULE 是 LazyScriptModule 包装器，setattr 只会落在包装器上，main()
    # 查的是脚本自己的 globals。
    monkeypatch.setitem(
        MODULE.main.__globals__,
        "verify_module_digest",
        lambda path, _key="wheel": calls.append(path),
    )
    monkeypatch.setattr("sys.argv", ["release-artifact-path.py", "wheel"])

    assert MODULE.main() == 0
    assert calls == [ROOT / "dist/current-release.json"]

    calls.clear()
    monkeypatch.setattr(
        "sys.argv", ["release-artifact-path.py", "wheel", "--skip-module-digest"]
    )

    assert MODULE.main() == 0
    assert calls == []
