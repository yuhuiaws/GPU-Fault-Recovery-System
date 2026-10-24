"""Collective branch adapted from PyTorch v2.10.0 FR Op.match.

Source: torch/distributed/flight_recorder/components/types.py.
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

PyTorch license:
Copyright (c) 2016- Facebook, Inc (Adam Paszke)
Copyright (c) 2014- Facebook, Inc (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America
    (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006 Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute
    (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2016-present Facebook Inc. All rights reserved.
Copyright (c) 2016 Facebook Inc.
Copyright (c) 2015 Google Inc. All rights reserved.
Copyright (c) 2015 Yangqing Jia. All rights reserved.
Copyright 2019-2020 Kakao Brain.
Copyright (c) 2022 Cruise LLC. All rights reserved.
Copyright (c) 2024 Tri Dao. All rights reserved.
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates.
Copyright (c) 2013, 2014, 2015 the respective Caffe contributors.
Copyright (c) 2015, 2016 the respective other contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
   America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

GFRS adaptations: dictionaries instead of Op objects, result codes instead of
MatchInfo objects, bounded input validation at the caller, P2P excluded.
The non-P2P predicates retain upstream ordering and are checked against the real
PyTorch matcher by tools.verify_fr_analyzer. No torch import is needed on nodes.
"""

from __future__ import annotations

import math
from typing import Any

COLLECTIVES = {
    "broadcast",
    "_broadcast_oop",
    "reduce",
    "_reduce_oop",
    "all_gather",
    "all_reduce",
    "_all_gather_base",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter",
    "reduce_scatter_tensor_coalesced",
    "_reduce_scatter_base",
    "gather",
    "scatter",
    "all_to_all",
    "all_reduce_barrier",
    "allreduce_coalesced",
    "ALLGATHER_coalesced",
    "REDUCE_SCATTER_coalesced",
}


def valid_entry(entry: dict[str, Any]) -> bool:
    for field in ("input_sizes", "output_sizes"):
        values = entry.get(field)
        if not isinstance(values, list) or not values or len(values) > 128:
            return False
        if any(
            not isinstance(shape, list)
            or len(shape) > 32
            or any(type(size) is not int or not 0 <= size < 2**63 for size in shape)
            for shape in values
        ):
            return False
    return all(
        isinstance(entry.get(field), list)
        and len(entry[field]) <= 128
        and all(isinstance(item, str) and len(item) <= 64 for item in entry[field])
        for field in ("input_dtypes", "output_dtypes")
    )


def match_collective(left: dict[str, Any], right: dict[str, Any], size: int) -> str:
    names = [str(entry.get("profiling_name", "")).split(":") for entry in (left, right)]
    if any(len(parts) != 2 or parts[0] not in {"nccl", "xccl"} for parts in names):
        return "UNDECIDED"
    kind, other = names[0][1], names[1][1]
    if kind not in COLLECTIVES | {"coalesced"} or other not in COLLECTIVES | {
        "coalesced"
    }:
        return "UNDECIDED"
    if kind not in COLLECTIVES:
        return "FULLY_MATCHED" if kind == other else "SIZE_OR_SYNTAX_MISMATCH"
    if kind != other:
        return "COLLECTIVE_TYPE_MISMATCH"
    if not valid_entry(left) or not valid_entry(right) or not 0 < size <= 65536:
        return "UNDECIDED"
    inputs, outputs = left["input_sizes"], left["output_sizes"]
    other_inputs, other_outputs = right["input_sizes"], right["output_sizes"]
    if kind not in {"all_to_all", "scatter"} and inputs != other_inputs:
        return "SIZE_OR_SYNTAX_MISMATCH"
    if kind not in {"all_to_all", "gather"} and outputs != other_outputs:
        return "SIZE_OR_SYNTAX_MISMATCH"
    if kind in {"all_reduce", "allreduce_coalesced"} and inputs != other_outputs:
        return "SIZE_OR_SYNTAX_MISMATCH"
    if (
        kind in {"all_gather", "all_gather_base", "all_gather_into_tensor_coalesced"}
        and math.prod(other_outputs[0]) != math.prod(inputs[0]) * size
    ):
        return "SIZE_OR_SYNTAX_MISMATCH"
    if (
        kind
        in {"reduce_scatter", "_reduce_scatter_base", "reduce_scatter_tensor_coalesced"}
        and math.prod(other_inputs[0]) != math.prod(outputs[0]) * size
    ):
        return "SIZE_OR_SYNTAX_MISMATCH"
    if (
        kind not in {"scatter", "gather", "broadcast"}
        and set(left["input_dtypes"]) != set(left["output_dtypes"])
        and inputs[0]
        and outputs[0]
        or kind not in {"scatter", "broadcast"}
        and set(left["input_dtypes"]) != set(right["input_dtypes"])
        and inputs[0]
        and other_inputs[0]
        or kind not in {"gather"}
        and set(left["output_dtypes"]) != set(right["output_dtypes"])
        and outputs[0]
        and other_outputs[0]
    ):
        return "COLLECTIVE_DTYPE_MISMATCH"
    if left.get("state") != right.get("state"):
        return "COLLECTIVE_STATE_MISMATCH"
    return "UNDECIDED" if kind == "all_to_all" else "FULLY_MATCHED"
