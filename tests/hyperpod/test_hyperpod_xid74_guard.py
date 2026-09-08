"""Live-safety guard around the HyperPod XID 74 injection case (H-15).

The case applies a privileged Pod to a node and execs into the control plane,
so it must refuse to run without explicit opt-in and a context pinned to the
intended cluster, and it must delete the Pod it created even when the run
fails partway through.
"""

from __future__ import annotations

import argparse
import sys

import pytest

import scripts.e2e.hyperpod.run_hyperpod_xid74_case as live_tool


def _args(**overrides: object) -> argparse.Namespace:
    base = {
        "live_run": True,
        "context": "target-ctx",
        "confirm_context": "target-ctx",
        "kubeconfig": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_refuses_without_live_run_opt_in() -> None:
    with pytest.raises(SystemExit, match="--live-run"):
        live_tool.assert_live_target(_args(live_run=False))


def test_refuses_when_confirm_context_does_not_repeat_context() -> None:
    with pytest.raises(SystemExit, match="confirm-context"):
        live_tool.assert_live_target(_args(context="prod", confirm_context="staging"))


def test_accepts_explicit_opt_in_with_matching_context() -> None:
    live_tool.assert_live_target(_args())


def test_context_flag_is_required() -> None:
    parser = live_tool.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--case",
                "safe-only",
                "--cluster-id",
                "c",
                "--job-id",
                "j",
                "--node",
                "n",
                "--pci-bdf",
                "0000:5A:00",
                "--live-run",
            ]
        )


def test_configure_kube_target_pins_kubeconfig_and_context(monkeypatch) -> None:
    monkeypatch.setattr(live_tool, "KUBE_PREFIX", ["kubectl"])
    monkeypatch.setattr(live_tool, "KUBE_CONFIG_PREFIX", ["kubectl"])

    live_tool.configure_kube_target("/tmp/kubeconfig", "target-ctx")

    assert live_tool.KUBE_PREFIX == [
        "kubectl",
        "--kubeconfig",
        "/tmp/kubeconfig",
        "--context",
        "target-ctx",
    ]
    assert live_tool.KUBE_CONFIG_PREFIX == [
        "kubectl",
        "--kubeconfig",
        "/tmp/kubeconfig",
    ]


def test_verify_context_available_rejects_unknown_context(monkeypatch) -> None:
    monkeypatch.setattr(live_tool, "run", lambda *_a, **_k: "ctx-a\nctx-b\n")

    with pytest.raises(SystemExit, match="not present in the kubeconfig"):
        live_tool.verify_context_available("ctx-missing")

    # A present context passes silently.
    live_tool.verify_context_available("ctx-a")


def test_main_deletes_injection_pod_when_run_fails(monkeypatch, tmp_path) -> None:
    # main() rebinds these module globals via configure_kube_target; restore
    # them so the mutation cannot leak into other tests in the same process.
    monkeypatch.setattr(live_tool, "KUBE_PREFIX", ["kubectl"])
    monkeypatch.setattr(live_tool, "KUBE_CONFIG_PREFIX", ["kubectl"])
    calls: list[tuple[str, list[str]]] = []

    def fake_kubectl(namespace, arguments, *, input_text=None):
        calls.append((namespace, list(arguments)))
        if arguments and arguments[0] == "wait":
            raise RuntimeError("pod never reached Succeeded")
        return ""

    monkeypatch.setattr(live_tool, "kubectl", fake_kubectl)
    monkeypatch.setattr(live_tool, "verify_context_available", lambda *_a, **_k: None)
    monkeypatch.setattr(
        live_tool, "snapshot", lambda *_a, **_k: {"node": {}, "managed_pods": []}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hyperpod_xid74_case.py",
            "--case",
            "safe-only",
            "--cluster-id",
            "c",
            "--job-id",
            "j",
            "--node",
            "node-a",
            "--pci-bdf",
            "0000:5A:00",
            "--context",
            "target-ctx",
            "--confirm-context",
            "target-ctx",
            "--live-run",
            "--report",
            str(tmp_path / "report.json"),
        ],
    )

    with pytest.raises(RuntimeError, match="pod never reached Succeeded"):
        live_tool.main()

    delete_calls = [
        arguments
        for _namespace, arguments in calls
        if arguments[:2] == ["delete", "pod"]
    ]
    assert delete_calls, "the injection Pod must be deleted on failure"
    assert "--ignore-not-found" in delete_calls[0]
