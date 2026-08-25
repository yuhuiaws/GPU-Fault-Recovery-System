# NVIDIA 官方策略实现审计

## 固定上游

- Catalog：NVIDIA Xid Catalog 610
- 官方地址：
  `https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`
- XLSX SHA-256：
  `7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5`
- 生成物 canonical SHA-256：
  `ffb82509abb574577db3c5d759cec94df40ea25edaf4691e7bd4ba0a12ba0462`
- 获取和生成日期：2026-07-20

生成物
`src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml`
包含 172 条 XID、95 条 XID 144-150 decode 和 32 个 Resolution Bucket。

重新生成：

```bash
python tools/generate_nvidia_xid_policy.py \
  Xid-Catalog.xlsx \
  src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml \
  --catalog-version 610 \
  --expected-sha256 \
  7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5
```

摘要不匹配时生成过程直接失败，禁止静默接受上游变化。
`--expected-sha256` 即使省略也默认使用上述固定 XLSX 摘要。

无需 XLSX 的常驻门禁：

```bash
PYTHONPATH=src python tools/generate_nvidia_xid_policy.py --check
```

该门禁检查 canonical 格式、上游摘要、172/95/32 条数和
`metadata.generatedSha256`。运行时 loader 会重新计算生成物摘要；
`mapping_version` 使用生成物摘要前 16 位，因此手改规则会直接加载失败，
或在显式更新摘要后触发部署 pin 不匹配。

上游 XLSX 不进入源码树。重新生成前应把已校验文件存入
`artifacts/upstream/nvidia/xid-catalog-610/<source-sha256>/`，并将该目录归档到
受控对象存储；`artifacts/` 只作为本地工作区。

## 决策语义

策略结果分别保存：

- `official_action`：NVIDIA 原始 Immediate Action/workflow 名称。
- `investigatory_action`：NVIDIA 原始 Investigatory Action。
- `action`：当前控制面经过批准且能够表达的执行动作。
- `safety_action`：官方 workflow 因证据或 executor 不完整而阻塞时的站点
  fail-closed 动作，当前仅允许 `QUARANTINE`。
- `source`：`NVIDIA_CATALOG`、`NVIDIA_XID_154`、
  `NVIDIA_FABRIC_MANAGER` 或 `SITE_SAFETY`。
- `disposition`：可执行、只监控、缺证据阻塞、缺 workflow 阻塞或不适用。

只有下列直接映射自动进入恢复计划：

| NVIDIA Immediate Action | 控制面动作 |
|---|---|
| `IGNORE` | `NO_ACTION` |
| `RESTART_APP` | `RESTART_WORKLOAD` |
| `RESET_GPU` | `RESET_GPU` |
| `RESTART_BM` | `REBOOT_NODE` |

在 `hyperpod-eks` 和 `hyperpod-slurm` runtime profile 中，
`RESTART_VM` 表示 reboot 当前 HyperPod 节点，runtime compiler 保留
`official_action=RESTART_VM`，设置 `effective_action=REBOOT_NODE`，并由
`BatchRebootClusterNodes` 执行。该映射不表示 EC2 stop/start；stop/start 和
节点 replacement 的宿主机放置、本地存储及身份语义仍是独立动作。

`CONTACT_SUPPORT`、`CHECK_MECHANICALS`、`UPDATE_SWFW` 和尚未实现的专用
workflow 不会被粗略替换成另一个“官方动作”，而是保留原始动作、返回
`BLOCKED_WORKFLOW`，并通过独立的 `safety_action=QUARANTINE` 阻止继续调度。

未知 XID 使用 `SITE_SAFETY` 来源和 `safety_action=QUARANTINE`，明确表示它是本系统的 fail-closed
安全策略，不是 NVIDIA Catalog 建议。

Catalog 的 `A100/H100/B100/GB200` 列按产品系列解释。具体 SKU 会规范化为
`A* -> A100`、`H*/GH* -> H100`、`B* -> B100`、`GB* -> GB200`，再按每条 XID
允许的系列执行适用性门禁。例如 H200/H800 属于 H100 列，B200 属于 B100 列，
GB300 属于 GB200 列。

## 已实现的官方 workflow

- XID 45：先持久化为 `PENDING_CORRELATION`，等待 30 秒窗口闭合；伴随其他 XID
  时复用主 XID 的 incident/workflow，solo 时保留 `RESTART_FM` 并阻塞，直到
  Fabric Manager executor 可用。共享 Store lease 支持 HA、乱序到达和 Pod 重启。
- XID 48：solo 执行 `RESET_GPU`；伴随 XID 63/64 时执行
  `DRAIN_AND_RESET`。
- XID 94/95：分别保存 application/all-applications containment；XID 95 在 reset
  前要求停止受影响工作负载。
- XID 154：驱动报告动作优先并原样映射。
- XID 159 `CHECK_UVM`：显式确认使用 UVM/vGPU 时 reset，否则 ignore；缺少证据阻塞。
- XID 144-150：按驱动 R575 边界选择 V1/V2 `IntrInfo` pattern，同时匹配
  Error Status 和 Action 2；无官方表项匹配时阻塞，不猜测动作。

## SXID

SXID 恢复依据 Fabric Manager User Guide：

- non-fatal：信息性，保持监控。
- fatal access link：必须已解析 affected GPU 和 workload participating GPUs，
  然后停止任务并 reset 整组 GPU。
- fatal trunk 或 always-fatal：要求完整 GPU/NVSwitch/FM 协调流程；当前没有该
  executor，因此返回 `BLOCKED_WORKFLOW`，不会错误执行单 GPU reset。
- B200/B300：传统 fatal/non-fatal SXID 不适用，要求 DCGM/NVSDM telemetry。

SXID 的 `classification_source` 必须是 `NVIDIA_FABRIC_MANAGER`，否则因证据来源
不可信而阻塞。
