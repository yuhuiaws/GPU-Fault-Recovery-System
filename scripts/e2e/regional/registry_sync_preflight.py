"""The registry sync gate of the deletion verbs, judged before a harness runs one.

``gpu-fault-admin uninstall`` and ``remove-cluster`` refuse to start a new
attempt while the site's last installation registry sync on the deploy host
(``<state-dir>/release-deploy/installation-registry-sync.json``) is not a
recorded success -- it failed, is not recorded (the site's last deploy predates
recording it, or a first deploy stopped before its release step) or is
unreadable (``resource_registry_outcome.deletion_gate``). An attempt that
already started resumes on the verdict it started with. The product's refusal
names the remedy, but an acceptance harness used to report only the verb's
exit status.

So a harness judges the same record first, read-only, through the reader
``status`` uses (``resource_registry_outcome.status_view``), and reports the
refusal with the product's remedy before it runs the verb. No harness passes
``--accept-unsynchronized-registry``: deleting only what the registry records
is an operator's decision. Used by BOOT-032 (native uninstall), the BOOT
isolated-site cleanup (BOOT-016/018) and the BOOT-029 driver (stage 2
remove-cluster, stage 4 uninstall), which calls this module's command line.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError
from gpu_fault.admin.resource_registry_outcome import ACCEPT_FLAG, status_view

#: What the BOOT-029 driver prints after a refusal: its own failure line then
#: names the stage to resume.
DRIVER_REMEDY = "resynchronize the site, then resume the stage"


def deletion_refusal(
    state_dir: Path, *, verb: str, harness: str, then: str
) -> str | None:
    """Why ``verb`` would refuse to start a new attempt on ``state_dir``, else ``None``.

    The sentence carries the recorded outcome (status, time, reason) and the
    product's remedy, then what ``harness`` expects the operator to do.
    """

    view = status_view(state_dir, release_id=None)
    if not view["deletion_verbs_refused"]:
        return None
    return (
        f"{verb} would refuse to start (installation registry sync "
        f"{view['status']}): {view['finding']}; {harness} never passes "
        f"{ACCEPT_FLAG}: {then}"
    )


def uninstall_started(state_dir: Path) -> bool:
    """An ``uninstall`` journal exists: the next uninstall resumes it."""

    return (state_dir / "uninstall" / "state.json").is_file()


def uninstall_refusal(
    state_dir: Path, *, harness: str, then: str, verb: str = "uninstall"
) -> str | None:
    """:func:`deletion_refusal` for a new uninstall attempt; ``None`` for a resume."""

    if uninstall_started(state_dir):
        return None
    return deletion_refusal(state_dir, verb=verb, harness=harness, then=then)


def _journal(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _names_cluster(journal: dict[str, Any], gpu_arn: str) -> bool | None:
    """Whether a removal journal is ``gpu_arn``'s: ``None`` when it cannot tell.

    The journal names its cluster by the site entry's EKS ARN (``target``) and,
    once discovered, by the provider identity (EKS and HyperPod ARN). An EKS ARN
    always compares with the target; a SageMaker ARN only with a discovered
    identity, so an attempt that stopped before discovery cannot tell.
    """

    target = journal.get("target")
    target = target if isinstance(target, dict) else {}
    evidence = journal.get("evidence")
    discovered = evidence.get("DISCOVERED") if isinstance(evidence, dict) else None
    identity = (
        discovered.get("provider_identity") if isinstance(discovered, dict) else None
    )
    identity = identity if isinstance(identity, dict) else {}
    names = {
        str(value)
        for value in (
            target.get("eks_cluster_arn"),
            identity.get("eks_arn"),
            identity.get("hyperpod_arn"),
        )
        if value
    }
    if gpu_arn in names:
        return True
    try:
        service = Arn.parse(gpu_arn).service
    except BootstrapError:
        return None  # remove-cluster refuses the ARN itself
    if service == "sagemaker":
        return False if identity.get("hyperpod_arn") else None
    return False if target.get("eks_cluster_arn") else None


def removal_started(state_dir: Path, gpu_arn: str) -> bool:
    """Whether ``remove-cluster --gpu-cluster-arn gpu_arn`` may continue a journal.

    The product resumes ``remove-cluster/<cluster>/state.json`` while it is
    unfinished and only reports a COMPLETED one once the cluster left the site
    (``cluster_removal_state.load_removal_state``); neither judges the registry
    again. Any journal that is this cluster's, may be (it cannot tell) or is
    unreadable therefore leaves the verdict to the product -- whose refusal
    names the same remedy -- rather than refuse a resume the product would
    allow; only journals of other clusters are passed over.
    """

    for path in sorted((state_dir / "remove-cluster").glob("*/state.json")):
        journal = _journal(path)
        if journal is None or _names_cluster(journal, gpu_arn) is not False:
            return True
    return False


def removal_refusal(
    state_dir: Path, gpu_arn: str, *, harness: str, then: str
) -> str | None:
    """:func:`deletion_refusal` for a new removal of ``gpu_arn``; ``None`` otherwise."""

    if removal_started(state_dir, gpu_arn):
        return None
    return deletion_refusal(
        state_dir, verb="remove-cluster", harness=harness, then=then
    )


def preflight(
    state_dir: Path, verb: str, *, gpu_arn: str | None, harness: str
) -> tuple[str | None, str]:
    """``(refusal, verdict line)`` of the command line: one sentence either way."""

    if verb == "uninstall":
        started = uninstall_started(state_dir)
    else:
        assert gpu_arn is not None
        started = removal_started(state_dir, gpu_arn)
    if started:
        return None, (
            f"{verb} may continue an attempt it already started (its journal is "
            "in the state directory): such an attempt keeps the registry sync "
            f"verdict it started on, so {verb} itself decides"
        )
    refusal = deletion_refusal(
        state_dir, verb=verb, harness=harness, then=DRIVER_REMEDY
    )
    if refusal is not None:
        return refusal, refusal
    view = status_view(state_dir, release_id=None)
    return None, (
        f"installation registry sync {view['status']} (release "
        f"{view['release_id']}, synced at {view['synced_at']}): {verb} may start "
        "a new attempt"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("uninstall", "remove-cluster"))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-arn", help="remove-cluster: the GPU cluster ARN the command removes"
    )
    parser.add_argument("--harness", default="BOOT-029")
    arguments = parser.parse_args(argv)
    if (arguments.verb == "remove-cluster") != bool(arguments.gpu_arn):
        parser.error("--gpu-arn is required for remove-cluster and only for it")
    refusal, line = preflight(
        arguments.state_dir,
        arguments.verb,
        gpu_arn=arguments.gpu_arn,
        harness=arguments.harness,
    )
    print(line)
    return 1 if refusal is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
