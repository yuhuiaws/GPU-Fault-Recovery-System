# GPU 温度、功耗与降频策略复核

## 依据与边界

本复核没有改 NVIDIA Xid catalog，没有部署、改功率上限或执行 GPU 压测。
以下区分接口定义、NVSentinel 实现与本站策略。

1. [NVIDIA nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：
   `SW Power Cap` 是功率管理降频；`HW Power Brake` 是外部供电系统触发；
   `GPU Slowdown Temp` 与 `GPU Max Operating Temp` 分别对应硬件与软件热限制。
   T.Limit 是有符号余量，不是绝对温度；其阈值不能与绝对摄氏温度交叉比较。
2. [NVIDIA NVML clocks event reasons](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html)：
   idle、应用时钟、SW power cap、Sync Boost 不等同硬件坏；
   HW slowdown 总位还可能反映切换状态等因素，不能直接当作 Power Brake。
   Power Brake 使用独立 `0x80` 位；SW/HW thermal 分别是 `0x20/0x40`。
3. NVSentinel 固定版本
   [`10b577655ef4caafc1206d5f2461f762c13b956c`](https://github.com/NVIDIA/NVSentinel/tree/10b577655ef4caafc1206d5f2461f762c13b956c)：
   [实现](https://github.com/NVIDIA/NVSentinel/blob/10b577655ef4caafc1206d5f2461f762c13b956c/health-monitors/gpu-health-monitor/gpu_health_monitor/dcgm_watcher/dcgm.py)
   与[配置](https://github.com/NVIDIA/NVSentinel/blob/10b577655ef4caafc1206d5f2461f762c13b956c/docs/configuration/gpu-health-monitor.md)。
   POWER/THERMAL watch 中普通 clocks-event 错误映射为 NONE，默认抑制；
   不是把这些 watch 的所有错误都抑制。
4. NVSentinel [热余量 runbook](https://github.com/NVIDIA/NVSentinel/blob/10b577655ef4caafc1206d5f2461f762c13b956c/docs/runbooks/gpu-thermal-margin.md)
   与 [Power Brake runbook](https://github.com/NVIDIA/NVSentinel/blob/10b577655ef4caafc1206d5f2461f762c13b956c/docs/runbooks/gpu-power-brake.md)：
   持续硬件热余量违规/Power Brake 映射 CONTACT_SUPPORT；应检查散热或供电故障域，
   不以 reboot 消除根因，不以空闲降温证明负载下已经修复。

NVSentinel 的热余量 watch 默认 enabled + STORE_ONLY，Power Brake 默认 disabled +
STORE_ONLY，二者默认三次确认。开启执行后才进入其隔离流程。GFRS 并非照搬这些默认
开关：继续使用现有 Runtime Profile 对动作的授权，普通可行动性能信号保留只读诊断。
检测器本身不授权越过 Profile、任务归属、节点锁、执行白名单。

## 评估与修改

| 原实现风险 | 本轮修改 |
|---|---|
| 热违规 duration 连续两次增长就 DRAIN | 仅作为 WARNING/只读诊断；不以 counter 单独授权停训 |
| 温度 warning + 热降频重复相关即可 DRAIN | 复合规则不能高于已经确认的温度组件动作 |
| 只识别 `0x60`，遗漏供电 Power Brake | 独立识别 `0x80`，不把 `0x04/0x08/0x100` 等当作 Power Brake |
| 设备 max-operating 当成 GPU HW slowdown | 仅有 GPU 软件最大工作温度时只告警；不凭它推导硬件严重阈值 |
| critical 温度单样本就停训 | 三个同来源、同设备、可比较的严重样本才确认；绝对 GPU shutdown 温度越界例外 |
| 边缘过滤只上传确认边沿，控制面看不到后续严重样本 | 严重候选及退出该状态的样本持续按采集周期上传，控制面独立确认 |
| 新版 exporter 字段名称可能漏采 | 接受 CLOCKS_EVENT_REASONS 与 GPU_TEMP_MARGIN_CELSIUS 的新旧别名；完全相同的别名样本去重，冲突仍为 UNKNOWN |
| nvidia-smi 模式缺少降频位与功率上限 | 增加环境字段，支持十六进制 mask、新旧 clocks 名称与可选显存温度 |
| 停训后冷却 300 秒就自动接纳 | 新严重环境工作流改为 ENVIRONMENT_REVIEW，保留隔离、交人工检查，没有自动恢复训练 |
| 将站点阈值描述为 NVIDIA 官方动作 | duration/功率相关明确标为本站性能策略，不伪称官方统一恢复动作 |

仍采用设备报告的绝对或 T.Limit 阈值，缺失时保留可配置绝对 fallback。
只报 T.Limit 的 GPU（其 XML 不打印绝对阈值）唯一保留的绝对限值是关机温度：DCGM collector
读取设备经 DCGM 字段 159（`DCGM_FI_DEV_SHUTDOWN_TEMP`）报告的值，仅当 XML 没有绝对关机阈值
且该值与此 GPU 的 T.Limit 算式相差不超过 1 °C 时发布为 `gpu_shutdown_temperature_c`
（`src/gpu_fault/collectors/gpu/shutdown_limit.py`），只服务于关机温度的单样本保护，
不换算成绝对读数的严重阈值。GFRS 在 HW
slowdown 边界使用 `>=`（绝对）/`<=`（余量）；NVSentinel 当前余量 watch 使用严格 `<`。
这是有意保留的本站保守边界，不宣称两者完全相同。
显存绝对温度仍独立保留安全阈值，不凭共享 T.Limit 反推 GPU 核心温度或证明即将关机。

三个有效严重观察、相邻不超过 60 秒属于本站确认策略，不是 NVIDIA 官方统一阈值。
`confirmation_count` 随 finding 持久化，控制面服务重建不会丢失确认进度。
warning 的历史次数不计入 critical；重复/旧采样不推进；来源、BDF、boot 标签变化、
过大采样间隔或 collection error 中断确认。GPU collector 为样本附宿主 boot ID。
没有数据不代表健康；未知观察不能授权硬件恢复。

`10000us` 最小增量、`0.1%` 最小占空比和 `105%` 合理性上限仍是本站噪声/有效性边界。
DCGM 的 violation 字段按 ns 转为内部 us；nvidia-smi 文档中的 clocks-event counters
本身为 us，两者不能混用。旧的 `GPU_FAULT_THERMAL_VIOLATION_DRAIN_CONSECUTIVE_SAMPLES` 与
`GPU_FAULT_DCGM_COMPOSITE_CONSECUTIVE_SAMPLES` 已退役：进程仍接受（启动时记一条 warning）但不读取，
duration 计数不能授权 DRAIN。

## 新工作流

```text
普通 power cap / 低主频 / 单独 thermal duration
    -> 不认定硬件损坏；有意义的性能信号只读诊断

设备严重热限值或 Power Brake
    -> 有效样本确认（绝对关机温度越界立即保护）
    -> FREEZE_EVIDENCE
    -> MARK_UNSCHEDULABLE
    -> STOP_WORKLOADS（仅 ACTIVE 且任务归属/授权成立）
    -> QUARANTINE
    -> COLLECT_DIAGNOSTIC_BUNDLE
    -> TRIGGER_HEALTH_SNAPSHOT
    -> ESCALATE_SUPPORT
    -> 保持隔离，人工检查冷却/供电；不自动 reset/reboot/RMA/重启任务
```

人工重新接纳应核对故障根因、同机架/供电域影响范围、对应 GPU 的新鲜温度/余量/原因位，
并在授权维护窗口内验证负载下的行为。本站没有新增自动压力测试或自动提高 power limit。

此次更改面向新编译的工作流，不重写已持久化的旧 THERMAL_COOLDOWN 计划。
上线前应盘点并人工处理这些旧计划，避免仍按旧版规则自动接纳。
finding JSON 增加默认值为 0 的 `confirmation_count`；旧控制面严格模型不认识新字段，
不能未经兼容性审查将已写入新状态的控制面热回滚到旧版本。

## 验证

新增[真机验收规格与受控回放工具](gpu-environment-live-acceptance.md)，
`GF-ENV-LIVE-001..012` 全部保持 manual / NOT_RUN。它们区分物理故障与真机遥测回放，
不从软件结果推断现场验证已完成。

- `tests/metrics/test_gpu_environment_policy.py`：掩码、确认、重启/跨 boot、
  UNKNOWN、绝对 shutdown、max-operating-only、复合不越权。
- `tests/collectors/test_gpu_environment_collection.py`：常量异常及清除都穿过边缘过滤；
  nvidia-smi 新旧名称、hex mask、功率上限和显存温度。
- `tests/execution/test_general_health_policy_e2e.py`：原始 DCGM 文本到真实 API、策略、
  工作流及验证逻辑；补充真实格式XML的绝对/T.Limit/显存/关机阈值，以及热复合只读流程；
  环境故障保持隔离，普通功率限制不停训。
- `scripts/e2e/run_general_fault_injections.py` 的 018/023 组更新为新契约。

软件测试中的设备/工作负载/Kubernetes I/O 是隔离替身；没有实施真实热故障或供电故障，
也没有把 NVSentinel 的其他硬件型号现场结果算作本项目的真机证据。
