"""COLLECT-022 must collide with an emitted offset, not skipped baseline history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import run_collect022_fm_cursor_recovery as verdict
from scripts.e2e.regional.probes import collect022_fm_cursor_probe as probe

NONCE = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def private_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    base = tmp_path / "private"
    base.mkdir(mode=0o700)
    boot = tmp_path / "boot"
    boot.write_text("private-boot")
    external = Mock(side_effect=AssertionError("external commands are prohibited"))
    monkeypatch.setattr(probe, "PRIVATE_BASE", base)
    monkeypatch.setattr(probe, "BOOT_FILE", boot)
    monkeypatch.setattr(probe, "refuse_external_command", external)
    monkeypatch.setattr(
        probe,
        "deployed_identity",
        lambda: {
            "python_prefix": "/private/runtime",
            "package_version": "test",
            "collector_module_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        probe,
        "service_identity",
        lambda: {
            "LoadState": "loaded",
            "ActiveState": "active",
            "MainPID": "123",
            "InvocationID": "private-service",
        },
    )
    processes = iter(f"123:{index}" for index in range(30))
    monkeypatch.setattr(probe, "process_identity", lambda: next(processes))
    args = {
        "nonce": NONCE,
        "cluster_id": "private-cluster",
        "node_id": "private-node",
        "expires_at": datetime.now(timezone.utc).timestamp() + 900,
    }
    initial = probe.run_action("init", **args)
    receipts = []
    for step in probe.STEPS:
        if step == "truncate":
            break
        receipt = probe.run_action("step", step=step, **args)
        assert (
            verdict.phase_errors(receipt, step=step, initial=initial, previous=receipts)
            == []
        )
        receipts.append(receipt)
    return SimpleNamespace(
        args=args,
        root=base / f"c022-{NONCE}",
        initial=initial,
        receipts=receipts,
        external=external,
    )


def test_private_copytruncate_reuses_an_emitted_device_inode_and_offset(
    private_probe: SimpleNamespace,
) -> None:
    env = private_probe
    preserved = env.root / "rotated.log"
    untouched = preserved.read_bytes()
    receipt = probe.run_action("step", step="truncate", **env.args)
    assert (
        verdict.phase_errors(
            receipt, step="truncate", initial=env.initial, previous=env.receipts
        )
        == []
    )
    old = receipt["truncation"]["reused_event"]
    current = receipt["events"][0]
    emitted = [event for previous in env.receipts for event in previous["events"]]
    assert old in emitted, "truncation must reuse a previously emitted record offset"
    for field in ("path", "device", "inode", "offset"):
        assert current["fields"][field] == old["fields"][field], (
            f"copytruncate did not reuse the original {field}"
        )
    assert old["fields"]["offset"] == "0"
    assert int(current["fields"]["generation"]) == int(old["fields"]["generation"]) + 1
    assert current["record_id"] != old["record_id"]
    assert current["evidence_ref"] != old["evidence_ref"]
    assert len({event["record_id"] for event in [*emitted, current]}) == 6
    assert preserved.read_bytes() == untouched, "the rotated sibling must stay intact"
    retry = probe.run_action("step", step="truncate", **env.args)
    assert retry == receipt, "lost probe ACK must not repeat the truncation"
    restarted = probe.run_action("step", step="truncated-restart", **env.args)
    assert restarted["events"] == []
    assert restarted["after"]["cursor"] == receipt["after"]["cursor"]
    assert probe.run_action("cleanup", **env.args)["private_root"] is False
    assert not env.root.exists(), "collector probe cleanup must remove its private root"
    env.external.assert_not_called()


@pytest.mark.parametrize(
    "damage", ["missing-event", "wrong-offset", "wrong-generation"]
)
def test_private_truncation_refuses_to_substitute_an_unemitted_offset(
    private_probe: SimpleNamespace, damage: str
) -> None:
    env = private_probe
    progress_path = env.root / "progress.json"
    progress = json.loads(progress_path.read_text())
    rotated = progress["receipts"]["rotate"]
    event = next(
        item
        for item in rotated["events"]
        if item["fields"]["path"] == str(env.root / probe.TRUNCATED_LOG)
    )
    if damage == "missing-event":
        rotated["events"].remove(event)
    elif damage == "wrong-offset":
        event["fields"]["offset"] = "1"
    else:
        event["fields"]["generation"] = "missing"
    progress_path.write_text(json.dumps(progress))
    log = env.root / probe.TRUNCATED_LOG
    previous = log.read_bytes()
    with pytest.raises(probe.ProbeError, match="truncation"):
        probe.run_action("step", step="truncate", **env.args)
    assert log.read_bytes() == previous, "unproven reuse must fail before truncation"
    assert probe.run_action("cleanup", **env.args)["private_root"] is False
    env.external.assert_not_called()
