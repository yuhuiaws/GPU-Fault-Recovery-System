# Xid linkage 与 Drain P2P 端到端测试

English: [Xid recovery E2E](../en/components/xid154-recovery-e2e.md).

本规格针对高风险 1（把 Xid 154 linkage 误作原始 Xid 的版本门禁）和高风险 3
（Drain P2P 缺少释放、读回和条件恢复链）。本地软件端到端与真机验收分别记录，
前者通过不能代表后者通过。

## 测试边界

本地入口为真实 `/v1/gpu-events/xid` API，使用真实策略、关联协调器、工作流编译器、
ApplicationContext 装配的生产执行器和升级逻辑、NodeActionWorkflowAdapter、
签名命令校验、NodeActionExecutor、quiesce 管理器及 SQLite 控制面/Agent 账本。

边界替身只有调度/任务/provider/验证 adapter、Node Action 网络传输和主机命令 runner。
主机命令不启动子进程，`/proc` 使用临时空目录；所有服务、GPU、训练状态均为夹具。
因此不覆盖真实 collector/kmsg 读取、区域队列/TLS、Kubernetes 控制器、DCGM 验证、
daemon RPC、实际 GPU reset、实际 reboot 或训练 checkpoint 正确性。

Xid 154 测试提交原始日志，不信任客户端传入的 `xid_154_action`。必须先观察
`PENDING_CORRELATION` 且没有工作流/节点命令，再推进协调器的测试时钟并执行
`run_once()`，最后经 API 重读最终决策。不能直接调用 policy 跳过这些步骤。

## 运行

在当前 checkout 根目录的锁定开发环境中运行：

```bash
.venv/bin/python -m pytest tests/execution/test_xid_recovery_e2e.py -n 0 -q -o addopts=''
```

用例注册在 `testcases/fault-scenarios.yaml`，独立需求及变体数量注册在
`testcases/scenario-requirements.yaml`。六组自动用例包含 16 个变体；UVM 缺口是
单独的手工 BLOCKED 用例，不用 skip/xfail 冒充覆盖。

## 高风险 1

### GF-E2E-XID-LINKAGE-001

前置：H100，目标 GPU-a、非目标 GPU-b，关联训练任务 ACTIVE；调度和硬件均为夹具。

| 变体 | 输入 | 必须观察到 |
|---|---|---|
| old-version | Xid 95，R550、CUDA 12.4 | EXECUTABLE / RESET_GPU |
| missing-version | Xid 95，不提供 driver/CUDA | 与有版本信息的原始故障同样执行 RESET_GPU |
| linkage-boundary | Xid 95，R575、CUDA 12.9 | 不把 linkage 边界误写成排除条件 |
| xid48-old-version | Xid 48，R550、CUDA 12.4 | 原始 Xid 48 仍进入 RESET_GPU 工作流 |

步骤：提交 API，读取决策和工作流，执行到终态，检查 Agent 调用记录和 incident。
通过标准：每个变体只 reset GPU-a 一次；禁调度和停止任务先于 reset；
persistence 释放、恢复服务后重启任务一次，incident 最终 RECOVERED。
出现版本不满足导致的 NOT_APPLICABLE/BLOCKED、只停在隔离、没有实际到达模拟 reset，
或工作流成功却未恢复任务，均为失败。

保留的反向约束由既有目录矩阵和 NVLink5 解码测试验证：产品不适用仍拒绝，
NVLink5 编码所需 driver 信息仍必须提供；移除 linkage 门禁不等于移除所有版本证据要求。

`GF-POL-011` 的旧“低版本应拒绝”期望已修正。`GF-REGIONAL-COLLECT-007` 的定义和
测试选择器随之改变，其旧 PASS 绑定不适用于新定义，目录状态重置为 NOT_RUN；
不得把历史摘要改签为本次通过证明。

## 高风险 3

共同输入是原始 Xid 154 日志：

```text
NVRM: Xid (PCI:0000:81:00): 154, GPU recovery action changed from 0x0 (None) to 0x3 (Drain P2P)
```

### GF-E2E-DRAIN-P2P-001

前置：目标 GPU-a 和非目标 GPU-b 的普通 persistence 均 Enabled，Fabric Manager 活跃。

1. 提交原始日志并关闭关联窗口，确认编译出 RECOVER_DRAIN_P2P。
2. 阻止模拟任务停止完成，执行一轮：必须停在 STOP_WORKLOADS，不能发送节点命令。
3. 允许停止，阻止验证完成，再执行：目标 persistence 关闭后才读取新的驱动 XML。
4. 按下表检查硬件调用；服务恢复后仍不能恢复调度或训练。
5. 重复同一事件，工作流 ID 不变；重放同一签名恢复命令，账本返回原结果，
   不增加查询、reset 或其他主机命令。
6. 允许 GPU/Fabric 验证完成，必须按验证、恢复调度、重启任务顺序成功结束。

| XML 中 GPU Recovery Action | reset 次数 | reset 目标 |
|---|---:|---|
| None | 0 | 无 |
| Reset | 1 | 仅 GPU-a |
| Drain and Reset | 1 | 仅 GPU-a |
| Drain P2P | 1 | 仅 GPU-a |

额外判定：GPU-b persistence 不变；Fabric Manager 不被停止或重启；GPU-a 的
persistence 随服务恢复为 Enabled。任何提前启动训练、重复 reset 或非目标 reset 均失败。

### GF-E2E-DRAIN-P2P-002

步骤：同样从原始 Drain P2P 日志进入工作流，驱动 XML 返回 Reboot。
通过标准：Agent 返回 `gpu_recovery_reboot_required=true`，证据穿过 adapter，
由应用装配的分支升级机制生成并执行一次模拟 RESTART_NODE；GPU reset 为零；
升级后的服务恢复和验证完成后，任务只重启一次。仅检查计划包含 RESTART_NODE 不算通过。

### GF-E2E-DRAIN-P2P-003

每个变体都必须先进入 RECOVER_DRAIN_P2P，不能在前置无关错误处失败就算通过。

| 故障注入 | 预期 |
|---|---|
| unknown-action | XML 为 N/A，不 reset、不自动 reboot |
| missing-gpu | XML 缺少目标 UUID，不猜测其他 GPU 的恢复动作 |
| persistence-stuck | 关闭命令返回但读回仍为 Enabled，不 reset |
| query-timeout | 查询超时，没有 reset，可处于受限重试的 WAITING/RUNNING |
| reset-timeout | reset 调用一次，Agent 标记 outcome_unknown 且不可重试，工作流 BLOCKED |

五个变体均不恢复调度、不重启训练。reset 超时后再次执行该工作流，主机命令不增加。
注意既有 `restoration_refusal` 明确允许已收到终态失败答复后的服务补偿：
可以恢复 persistence/监控服务，但仍保留调度隔离和人工确认。这不是硬件恢复成功，
也不是训练恢复；不得以服务 active 判该用例 PASS。

### GF-E2E-DRAIN-P2P-004

输入 workload_state=IDLE，驱动读回 None。必须执行释放和读回，工作流成功，
不得出现 STOP_WORKLOADS 或 RESTART_WORKLOAD。

### GF-E2E-DRAIN-P2P-005

从 Agent allowlist 移除 RECOVER_DRAIN_P2P。完整事件链不得成功，不得 reset、
自动 reboot 或恢复训练。这里验证的是操作授权拒绝；真正旧版本 Agent 对新枚举的
反序列化行为、滚动升级和回滚还需要独立部署验收。

## 真机验收条件

这些条件是未执行的验收规格；整条 Drain P2P 恢复链没有专用的自动破坏性 runner。
GF-GENERAL-LIVE-024 的 `xid_twin_drain_p2p` 变体（`scripts/e2e/regional/general_live_xid_twins.py`，
需单独的 `GPU_DRAIN_P2P` 审批）在空闲节点上先让 DCGM 字段 230 变为 154，再写入一行经审阅的
XID 154 Drain P2P 内核日志，期望恰好执行一次 RECOVER_DRAIN_P2P、事件 RECOVERED 且不计划 QUARANTINE。
它验证双源配对和这条链在真实节点上的执行；真实 `GPU Recovery Action` 仍由驱动决定，
因此不覆盖下列 Reset/Reboot 分支，也不覆盖训练恢复。

- 使用隔离 GPU 节点、可丢弃训练任务及明确维护窗口；先记录 workload/attempt、
  GPU UUID/PCI、driver/CUDA、boot ID、persistence、服务状态和初始调度状态。
- 先证明独立超时恢复、人工接管和禁止外部调度的机制可用，再授权停止任务/reset/reboot。
  不得为测试旧版本而在生产节点临时降级驱动；没有匹配环境就记录 NOT_RUN。
- 仅写入模拟 Xid 日志不会改变真实 `GPU Recovery Action`。没有可控、可证明的驱动状态，
  就不能宣称验证了真实 None/Reset/Reboot 分支，更不能伪造 nvidia-smi 输出作为真机证明。
- 同步收集事件、关联记录、工作流各 step、签名命令 ID、Agent 账本、nvidia-smi
  原始输出、服务操作、实际 reset/reboot 调用和任务 attempt。按时间排序验证顺序及幂等。
- 成功后校验 GPU/Fabric 健康、persistence 和服务恢复、非目标 GPU 未受损、
  训练从预期 checkpoint 恢复且 step/loss 持续前进；日志显示“重启成功”不够。
- 未知结果时保留隔离和人工确认，禁止直接 uncordon/重启训练作为清理。
  清理只处理本用例拥有的任务与状态，不清空 Agent 幂等账本。

## UVM Persistence Gap

`GF-E2E-DRAIN-P2P-UVM` 当前为 BLOCKED。代码没有独立检测/关闭 UVM persistence；
普通 `persistence_mode=Disabled` 和空的临时 `/proc` 都不能提供这项证明。

后续验收必须包含“普通 persistence 已关闭，但 UVM persistence 仍开启”的反例。
预期系统能识别并完成授权的 UVM 释放，或明确拒绝继续恢复；仅凭普通 Disabled 继续
执行不能通过该项。当前不进行这类故障注入，不把实现缺口缩减为“只差真机测试”。
