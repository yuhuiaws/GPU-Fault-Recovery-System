"""CPU-only acceptance against the installed, real PyTorch 2.10 FR engine."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import tempfile
from pathlib import Path

from gpu_fault.pytorch_fr_matcher import COLLECTIVES, match_collective
from tools.analyze_nccl_flights import analyze


def entry(kind: str) -> dict:
    inputs, outputs = [[4]], [[4]]
    if kind in {"all_gather", "_all_gather_base", "all_gather_into_tensor_coalesced"}:
        outputs = [[8]]
    if kind in {
        "reduce_scatter",
        "_reduce_scatter_base",
        "reduce_scatter_tensor_coalesced",
    }:
        inputs = [[8]]
    return {
        "record_id": 0,
        "pg_id": 0,
        "process_group": ["0", "default_pg"],
        "collective_seq_id": 1,
        "p2p_seq_id": 0,
        "op_id": 1,
        "is_p2p": False,
        "profiling_name": f"nccl:{kind}",
        "input_sizes": inputs,
        "output_sizes": outputs,
        "input_dtypes": ["Float"],
        "output_dtypes": ["Float"],
        "state": "completed",
        "frames": [],
        "time_created_ns": 123,
    }


def verify(root: Path) -> dict:
    import torch
    from torch.distributed.flight_recorder.components.types import Op

    if not torch.__version__.startswith("2.10."):
        raise ValueError("acceptance requires the real PyTorch 2.10 engine")
    cases = 0
    for kind in sorted(COLLECTIVES | {"coalesced"}):
        for mutation in ("none", "input", "output", "dtype", "state", "type"):
            left = entry(kind)
            right = copy.deepcopy(left)
            if mutation == "input":
                right["input_sizes"] = [[7]]
            elif mutation == "output":
                right["output_sizes"] = [[7]]
            elif mutation == "dtype":
                right["input_dtypes"] = right["output_dtypes"] = ["Half"]
            elif mutation == "state":
                right["state"] = "started"
            elif mutation == "type":
                right["profiling_name"] = (
                    "nccl:broadcast" if kind != "broadcast" else "nccl:all_reduce"
                )
            native = (
                Op(left, {"group": {0, 1}}, "group")
                .match(Op(right, {"group": {0, 1}}, "group"))
                .state.name
            )
            adapted = match_collective(left, right, 2)
            if native != adapted:
                raise AssertionError(
                    f"matcher differs for {kind}/{mutation}: {native} != {adapted}"
                )
            cases += 1
    builder_cases = 0
    for kind in (
        "all_reduce",
        "broadcast",
        "all_gather",
        "_reduce_scatter_base",
        "allreduce_coalesced",
    ):
        for mismatch in (False, True):
            records = []
            for rank in range(2):
                operation = entry(kind)
                if mismatch and rank == 1:
                    operation["profiling_name"] = "nccl:reduce"
                payload = {
                    "version": "2.10",
                    "pg_config": {
                        "0": {"name": "0", "desc": "default_pg", "ranks": "[0,1]"}
                    },
                    "entries": [operation],
                }
                raw = pickle.dumps(payload, protocol=2)
                path = root / f"{rank}.dump"
                path.write_bytes(raw)
                records.append(
                    {
                        "rank": rank,
                        "pid": 100 + rank,
                        "process_start_ticks": 1000,
                        "node_id": f"node-{rank}",
                        "path": path.name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            result = analyze({"records": records}, root)
            if result["status"] != "ANALYZED" or not result["collectives"]:
                raise AssertionError(f"official builder did not analyze {kind}")
            if all(row["pass_check"] for row in result["collectives"]) == mismatch:
                raise AssertionError(
                    f"wrong official builder verdict for {kind}/{mismatch}"
                )
            builder_cases += 1
    return {
        "status": "PASS",
        "torch_version": torch.__version__,
        "native_matcher_cases": cases,
        "official_builder_cases": builder_cases,
        "real_nccl_execution": False,
        "physical_acceptance": "NOT_RUN",
        "pp_in_scope": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gfrs-native-fr-") as temporary:
        result = verify(Path(temporary))
    text = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
