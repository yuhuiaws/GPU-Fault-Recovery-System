#!/usr/bin/env python3
"""Read-only census of the warm-spare marks on a GPU cluster's nodes.

GF-GENERAL-LIVE-030 borrows another lane's node as its warm spare and must
hand it back exactly as found; this is the proof the variant records after
its release and the read the lane runner's heal can call between cases:

    warm_spare_residue.py --site-profile <profile> [--expect <node> ...]

prints one JSON document -- every node carrying ``gpu-fault.io/spare=true``
(``declared_spares``), those of them not listed with ``--expect``
(``unexpected``), the nodes with a spare reservation and the nodes whose pool
state is anything but absent/AVAILABLE -- and exits 0 when nothing unexpected
is left, 3 otherwise. It mutates nothing: a residue is released through the
admin CLI (``gpu-fault-admin config spare`` / ``declare_warm_spare.py
--release``), never by deleting labels by hand. A HyperPod site that keeps a
spare deliberately declared names it with ``--expect``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    install_abort_signals,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.site_profile import (  # noqa: E402
    bind_site_profile,
    install_site_profile,
)

SPARE_LABEL = "gpu-fault.io/spare"
RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
#: Exit status when a spare mark is left that ``--expect`` did not name.
RESIDUE_EXIT = 3


def spare_residue_report(
    nodes_document: Mapping[str, Any], *, expected: Iterable[str] = ()
) -> dict[str, Any]:
    """The spare marks of ``kubectl get node -o json``; pure, no cluster read."""

    allowed = set(expected)
    declared: list[str] = []
    cordoned: list[str] = []
    reserved: dict[str, str] = {}
    pool_state: dict[str, str] = {}
    for item in nodes_document.get("items") or []:
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name"))
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        if labels.get(SPARE_LABEL) == "true":
            declared.append(name)
            if (item.get("spec") or {}).get("unschedulable"):
                cordoned.append(name)
        if annotations.get(RESERVATION_ANNOTATION):
            reserved[name] = str(annotations[RESERVATION_ANNOTATION])
        state = annotations.get(POOL_STATE_ANNOTATION)
        if state not in (None, "AVAILABLE"):
            pool_state[name] = str(state)
    unexpected = sorted(name for name in declared if name not in allowed)
    return {
        "declared_spares": sorted(declared),
        "cordoned_spares": sorted(cordoned),
        "unexpected": unexpected,
        "reserved": dict(sorted(reserved.items())),
        "pool_state": dict(sorted(pool_state.items())),
        "expected": sorted(allowed),
        "clean": not unexpected,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "List the nodes of the GPU cluster that still carry warm-spare marks "
            "(read-only); exit 3 when a spare label is left that --expect did not name."
        )
    )
    value.add_argument("--site-profile", default="")
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument(
        "--expect",
        action="append",
        default=[],
        help="a node deliberately kept declared as a spare (repeatable)",
    )
    value.add_argument("--report", type=Path, default=None)
    return bind_site_profile(value)


def main() -> int:
    install_site_profile()
    install_abort_signals()
    arguments = parser().parse_args()
    regional = RegionalLiveFixture(settings_from_arguments(arguments))
    nodes = json.loads(regional.kubectl("gpu", "get", "node", "-o", "json"))
    report = spare_residue_report(nodes, expected=arguments.expect)
    if arguments.report is not None:
        from scripts.e2e.regional.acceptance_runner_common import write_json_atomic

        write_json_atomic(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["clean"] else RESIDUE_EXIT


__all__ = [
    "POOL_STATE_ANNOTATION",
    "RESERVATION_ANNOTATION",
    "RESIDUE_EXIT",
    "SPARE_LABEL",
    "spare_residue_report",
]


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
