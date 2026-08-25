#!/usr/bin/env python3
"""按用例要求改一处配置，输出可 apply 的探针清单（手册 §4.0 步骤 P5）。

每个 BOOT 用例只允许改**一个**环境变量，这样失败原因唯一。

用法:
  mutate.py del  ENV_NAME              删除某个环境变量
  mutate.py set  ENV_NAME VALUE        设为字面值（覆盖 valueFrom）
  mutate.py sref ENV_NAME SECRET KEY   改为引用另一个 Secret 的 key

基线清单默认读 /tmp/guard-probe-base.json（由 derive.sh 生成），
可用 GUARD_PROBE_BASE 覆盖。
"""

from __future__ import annotations

import json
import os
import sys

BASE = os.getenv("GUARD_PROBE_BASE", "/tmp/guard-probe-base.json")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    with open(BASE) as handle:
        doc = json.load(handle)
    env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
    op, name = sys.argv[1], sys.argv[2]
    idx = next((i for i, e in enumerate(env) if e["name"] == name), None)
    if op == "del":
        assert idx is not None, f"{name} 本来就不存在，用例前提不成立"
        env.pop(idx)
    elif op == "set":
        entry = {"name": name, "value": sys.argv[3]}
        if idx is not None:
            env[idx] = entry
        else:
            env.append(entry)
    elif op == "sref":
        assert idx is not None, name
        env[idx] = {
            "name": name,
            "valueFrom": {"secretKeyRef": {"name": sys.argv[3], "key": sys.argv[4]}},
        }
    else:
        raise SystemExit(f"unknown op {op}")
    json.dump(doc, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
