# 通用故障自动化注入

## 执行

在仓库根目录使用已安装开发依赖的 Python。脚本只运行本地软件注入，不接受
集群地址，不读取业务 Kubernetes/AWS 凭据，不启用线上 NodeLogCollector。

```bash
.venv/bin/python scripts/e2e/run_general_fault_injections.py --list
.venv/bin/python scripts/e2e/run_general_fault_injections.py
.venv/bin/python scripts/e2e/run_general_fault_injections.py --case 016 --case 017 --output /tmp/gfrs-log-injections
```

默认运行全部 32 类；`--case` 可重复，也接受完整的 `GF-GENERAL-E2E-016`。
每次默认新建 `/tmp/gfrs-general-injection-*`；显式输出目录必须尚不存在，避免覆盖证据。
重复运行使用新的目录，数据库、原始文件和模拟客户端由各测试隔离创建。
执行复用 `tools/run_fault_test_cases.py` 的凭据隔离、pytest 逐项回执和源码身份检查。

输出包括：

- `checks.yaml`：本次实际执行的自动化用例和 pytest 节点。
- `checks.json`：逐检查结果、命令、失败原因、测试回执和源码身份。
- `report.json`：32 类分组结果、原始注入入口、实际覆盖边界、检查回执关联。

新主机和 GPU/日志主场景在检查的 `output` 中还保存
`GENERAL_FAULT_EVIDENCE=` JSON：原始输入、API 响应、持久化工作流/步骤、
模拟动作/最终节点状态和清理结果。分段复用的旧检查不保证具有这份完整轨迹；
不能把某个检查未输出动作回执解释成动作已经成功。

任一检查失败、跳过、未执行或缺回执，不得作为该类 PASS。脚本返回非零退出码。
`--require-continuous` 还要求选中的每类有连续的软件主场景链；分段检查即使全部通过，
也返回非零。它不是宣称该类所有子变体、所有型号均通过。

## 证明边界

映射表在 `tools/general_fault_injections.py`，每项都关联实际可执行的测试函数。
24 类具有连续主场景软件链，另 8 类明确标为 `segmented-checks`：
012、013、014、021、026、028、029、032。
这些类当前复用的采集、策略、适配器测试不等于同一故障完成了整个恢复过程。
028 的 watcher 与恢复服务、029 的预置预算与执行链也不能冒充原始硬件故障闭环。

011 新增原始EFA/CRI文件到API、实际NodeAction、归档及Fabric验证的连续软件链，
共36个参数化实例；原有组件检查仍保留。可用 `--case 011 --require-continuous`
运行。本地链的节点命令输出、时间和通知是测试边界，不能作为真机NCCL恢复证明。
具体范围见 [NCCL hung 端到端覆盖审计](nccl-hung-e2e-coverage.md)。

主机输入使用临时 procfs/sysfs 文件、statvfs 返回值和 SMART/BMC 命令输出。
GPU 主场景使用真实 DCGM Prometheus 文本解析，duration 原始单位是 ns。
日志主场景经过真实 NodeLogCollector，使用 journal JSON 或 CRI 文件，不再直接制造
NodeLogBatch 绕过采集器。训练进度和采集器静默使用各自的服务入口/状态输入。

策略、工作流执行器、GPU/Host/Fabric 验证器保持真实实现。设备命令、工作负载动作、
Kubernetes 客户端和云服务 I/O 是模拟边界；软件测试通过不证明驱动真的完成 reset、
机器真的完成 reboot，也不证明千卡任务完成了现场取证。
原 `GF-GENERAL-E2E-001..032` 真机验收规格继续保持 `manual / NOT_RUN`；
[逐项验收缺口](general-fault-e2e-audit.md)不因这些自动化通过而消失。

## NodeLogCollector 修复

- journal 保存真实 `__CURSOR`，使用 `--after-cursor`，同时间戳分页不再卡住；
  cursor 被 vacuum 后显式报错并回放有界时间窗口，可能重复但不静默跳过。
- journalctl 不存在时报告 UNKNOWN，不再产生健康摘要。
- 文件在消耗本轮预算前统一建立初始位置；新文件被延后也不会在下一轮跳到新的 EOF。
- 文件物理行按剩余字节预算读取；未完成行和 CRI P/F 分片持久化，支持跨轮/重启续接。
  超长内容保持有界截断；分片随 inode 轮转转移，不把后半行误作新记录。
- CRI 保留事件时间。只有可信 pod-log 根目录下的路径才提供 Pod UID/容器名。
  通用文本文件不能仅凭当前节点任务数量推断归属。
- 每个 collector 最多一个文件读取线程，默认等待 2 秒。阻塞的目录查询/open/read
  不阻止 journal 采集；未返回时报告 UNKNOWN，不累积新线程。底层永久阻塞仍需要
  修复挂载或重启 collector，不能在线程内强行中止内核 I/O。
- 文件发现和持久化表有硬上限；超限报告覆盖不完整，要求缩小路径或评估后调整上限。
- uncorrectable/fatal 优先于 corrected，上下文中的 corrected 不会降级真正致命 MCE。
- 使用新鲜 observation 和 Pod/容器/PID 归属；已退出 rank 可保留 EXITED 身份，
  但不对退出进程抓栈，不猜测其他 attempt。
- 有当前 Agent 或 collector boot 身份时，旧 boot/缺 boot 的内核日志不能隔离新 boot；
  超过 900 秒或未来超过 30 秒的内核事件仅保留证据。兼容的旧 API 生产者若双方均无
  boot 元数据，仍采用已认证的来源规则，不应把这种输入当作具备代际验证。

默认仍不自动启用 NodeLogCollector。升级后新写入的 `training_log_pending` 是
有界续读状态；降级到旧版本会忽略片段，必须按降级丢失部分日志证据处理。
超大且无法在 journal 单轮预算内解析的 JSON 会显式报告不可解析/预算问题，保留
已知位置；超过原有 900 秒积压上限时会报告截断窗口，不承诺无限积压零丢失。
