#!/usr/bin/env python3
"""生成探针用的 regional cluster registry JSON（BOOT-007 / BOOT-008）。

字段形状与生产 registry 一致，但 cluster_id 换成 guardprobe-fake-cluster，
避免与生产注册表撞 key。token 是可预测的 "g" * n，因为 BOOT-008 分支 3
需要用它自打一次数据面请求验证 403。

用法: registry.py <token长度> [enabled] [要删掉的字段] [要加的拼错字段]
  registry.py 31                              token 31 字符（BOOT-007 负向）
  registry.py 32                              token 32 字符（BOOT-007 正向）
  registry.py 64 disabled                     enabled=false（BOOT-008 分支 3）
  registry.py 64 "" eks_cluster_arn           缺字段（BOOT-008 分支 1）
  registry.py 64 "" "" alowed_namespaces      拼错字段（BOOT-008 分支 2）
"""

from __future__ import annotations

import json
import sys

ACCOUNT = "000000000000"
TEST_AGENT_ENDPOINT_CIDR = "192.0.2.0/24"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = int(sys.argv[1])
    entry = {
        "cluster_id": "guardprobe-fake-cluster",
        "region": "us-west-2",
        "hyperpod_cluster_name": "guardprobe-fake-cluster",
        "eks_cluster_arn": f"arn:aws:eks:us-west-2:{ACCOUNT}:cluster/guardprobe-fake",
        "token": "g" * n,
        "allowed_namespaces": ["default"],
        # RFC 5737 TEST-NET-1 cannot overlap a real VPC or node subnet.
        "agent_endpoint_allowed_cidrs": [TEST_AGENT_ENDPOINT_CIDR],
    }
    if len(sys.argv) > 2 and sys.argv[2] == "disabled":
        entry["enabled"] = False
    if len(sys.argv) > 3 and sys.argv[3]:
        entry.pop(sys.argv[3])
    if len(sys.argv) > 4 and sys.argv[4]:
        entry[sys.argv[4]] = ["default"]
    print(json.dumps([entry]))


if __name__ == "__main__":
    main()
