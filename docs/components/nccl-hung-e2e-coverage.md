# NCCL hung 端到端覆盖审计

本轮清单、时序、Fabric 和错误分流修改另增
`test_nccl_review_e2e.py` 的 8 个连续链实例，以及
`test_nccl_review_boundaries.py`、`test_nccl_collection_budget.py` 的边界测试。
下面“历史测量”的覆盖百分比只对应修改前快照，不能作为当前代码覆盖率。
当前行为、RAS 版本边界及未完成的真机验收以
[NCCL hung 改进与验证](nccl-hung-hardening.md)为准。

## 本轮新增

| 测试文件 | 验证内容 | 边界 |
|---|---|---|
| `test_hung_episode_e2e.py` | API、SQLite、第一次取证、噪声不重采、稳定恢复、第二波新轮次 | 软件链，非真实训练 |
| `test_efa_stall_episodes.py` | 冻结基线、fallback、缺样本/尖峰/跨 attempt、旧状态迁移 | 状态机 |
| `test_communication_monitor.py` | 真 loopback RAS 协议、128节点投影、QP绑定、进程阻塞与回收 | RAS服务和GPU响应为测试边界 |
| `test_communication_history_pipeline.py` | 固定epoch历史、PID复用、无RANK关联、三个真实worker入口 | 不执行真实GPU命令 |
| `test_communication_scope.py` | 已认证heartbeat签名兼容、作用域、boot/generation/租约拒绝 | 本地API |
| `test_partial_hung_evidence.py` | 超时部分证据留存，不计完整覆盖率 | 实际工作流适配器 |

第011类自动注入运行器已加入这些关键连续链和边界选择器。
另在独立 PyTorch 2.10.0+cpu 环境运行 `tools.verify_fr_analyzer`，
114 个原生匹配规则对照与 10 个官方 builder 用例通过。没有 CUDA/NCCL 运行；
真机初始化、长 checkpoint、局部组停滞、实际采集开销、框架协同恢复仍需相应环境验收。
本轮未重新计算行覆盖百分比，不能把用例通过数当成代码覆盖率。

## 口径

不能把此前5141项相关回归通过、端到端场景覆盖和真机验收覆盖混为一谈。
本次区分三层：

1. 组件测试：分别验证解析器、分类器、状态机、适配器。
2. 连续软件链：同一输入经过真实采集解析、API、策略/编排、NodeAction、归档、
   Fabric验证和事件收尾；硬件响应、时间及通知外部边界使用隔离夹具。
3. 真机端到端：真实训练/NCCL/驱动/EFA、真实部署与传输，本轮没有执行。

此前测试有良好的分段基础，但并非都满足第二层。例如：
`test_efa_traffic_harness_main.py`只到计划，`test_hung_survey_pipeline.py`直接注入finding
且用成功替身处理Fabric；FR和PID测试多止于分类/后置计划，而非实际归档和事件收尾。
这些测试继续保留，但不能按完整链相加得出一个“总体覆盖百分比”。

## 新增连续链

`tests/execution/test_nccl_continuous_e2e.py`包含36个参数化测试实例。
辅助夹具为 `_nccl_chain_node.py`、`_nccl_chain_support.py`。

```text
临时sysfs计数器 / CRI日志
  -> 真正的HostTelemetryCollector RDMA贡献器 / NodeLogCollector
  -> ASGI API、策略、SQLite、编排器
  -> NodeActionWorkflowAdapter签名命令
  -> 真实NodeActionExecutor：summary、triage、bundle
  -> 压缩证据解码、FR/进程分析、扩采与PID选择
  -> 实际tar.gz及SHA-256、真实GpuValidationAdapter
  -> 事件ESCALATED，禁止凭疑似hung停训或重启
```

sysfs/proc都是测试目录下的文件，不写真实/sys或/proc；py-spy/nvidia-smi/日志工具
返回受控输出，FR响应器通过真实本地FIFO写JSON或pickle，但不是原生PyTorch监控线程。
NodeAction使用进程内sender调用真实签名校验/handler，不覆盖真实HTTP/mTLS网络。
通知只验证执行意图和最终事件状态，不发送真实邮件。
节点取证使用受控时钟和串行sender以稳定场景；此处不能推导千卡并发延迟。
Host Collector仅启用RDMA贡献器，其他健康遥测用于真实Fabric验证的输入，不冒充
全套主机采集器在生产环境均已验收。

## 场景矩阵

以下是本次明确列出的14类链路义务，不是所有NCCL故障的穷尽集合。

| 义务 | 新增实例/断言 | 层级 |
|---|---|---|
| 原始RDMA触发 | normal；实际计数器增量到告警、取证、归档和收尾 | 连续软件链 |
| 不应触发的边界 | always-low、invalid-counter、new-attempt、traffic-recovers、长观测空洞；新attempt不得继承活跃历史 | 连续软件链 |
| 普通日志入口 | CRI NCCL错误保留证据，但不创建 marker 或独立诊断链；退出归 watcher | 连续软件链 |
| 全任务摘要与选点 | 9节点全量summary；可选通信文件把末尾落后rank所在节点提到首轮 | 连续软件链 |
| FR时序 | JSON/pickle、完成落后、正常完成推进、第二份缺失、整组等待 | 连续软件链 |
| 进程与角色 | 无RANK的I/O线索、py-spy非零退出、同角色/不同PP角色、不同采样时长CPU归一化、NSpid映射 | 连续软件链 |
| 扩采与预算 | 首轮4+次轮4、遗漏分母、压缩证据解码；字节预算耗尽不得宣称全采 | 连续软件链 |
| 缺失与损坏 | summary不可达、triage不可达、checksum损坏、旧snapshot epoch | 连续软件链 |
| 身份与范围 | 其他任务PID不attach；无cgroup范围不attach；bundle前PID复用跳过 | 连续软件链 |
| 证据落盘 | 实际归档、manifest中最终判定、SHA-256与执行回执一致 | 连续软件链 |
| Fabric证据不足 | 缺GPU遥测保持等待；补齐后才结束取证，不用进程数据替代 | 连续软件链 |
| 安全收尾 | 健康Fabric不等于训练恢复；bundle部分失败仍为ESCALATED且保留其他节点证据 | 连续软件链 |
| 多节点同一事件 | 多节点同时间窗口原始流量触发，聚合窗口0/15时均只取证一遍 | 连续软件链 |
| 持久化与幂等 | 新SQLite连接/同executor identity恢复；pending超时；响应丢失且Agent重建后不重复采集 | 连续软件链 |

端到端用例还共同禁止STOP_WORKLOADS、RESET_GPU、RESTART_NODE、
RESTART_WORKLOAD、REPLACE_NODE、隔离/禁调度，并让training-progress读取直接失败，
防止无意重新依赖train step。

## 仍有缺口

- 原生CUDA/NCCL真正hung、驱动/固件差异、真实FR版本及异步dump行为未做真机验证。
- 1024进程/128节点及万rank软件规模用例仍是已有模拟测试，不证明真实千卡负载、
  网络并发、CPU/RSS、文件I/O或训练吞吐影响。
- 本次连续链使用SQLite。同worker身份续跑不是PostgreSQL多控制面HA故障转移验证。
- HTTP/mTLS、真实消息队列、对象存储上传和通知实际投递不在新链的外部边界内。
- 更细的非法格式、线程/时钟异常、所有GPU共享方式等仍有组件测试，不全都有连续链。
- checkpoint/长计算引发疑似告警的误报率和长期进程基线，
  不能因这些测试通过就标为解决。
- PP 组识别/send-recv 对端定位明确排除；保留 P2P 原始证据，不做自动归因。
- 普通错误/退出的新增覆盖见 `tests/execution/test_nccl_watcher_ownership.py`；
  watcher 不做 NCCL 文案专用过滤，挂起采集会在已知退出/清理后停止追加。

对应真机规格继续保持manual/NOT_RUN：
[GF-GENERAL-LIVE-011](general-fault-live-acceptance.md#gf-general-live-011)。

## 历史测量

2026-10-23本轮 review 修改前本地运行：36个实例全部通过，无跳过。仅运行这36个实例，
对下列18个NCCL专用模块启用pytest-cov分支统计，结果为：

| 指标 | 已覆盖/总数 | 覆盖率 |
|---|---:|---:|
| 可执行语句 | 1449 / 1681 | 86.2% |
| 分支出口 | 459 / 636 | 72.2% |
| 语句与分支合计 | 1908 / 2317 | 82.35% |

这不是全仓覆盖率，也不是“所有故障场景覆盖率”；剩余232条语句和177个分支出口
没有被这组E2E执行。已有组件测试另行覆盖部分解析错误/旧格式/边界路径，本统计不混入
这些组件测试，也不替代发布时的全量覆盖门禁。没有修改覆盖阈值或排除规则。

测量范围为 `gpu_fault` 下的：

```text
efa_traffic_state
hung_survey_analysis
hung_summary_codec
execution.hung_survey
execution.hung_classification
execution.flight_temporal
execution.process_observations
execution.process_cohorts
execution.process_diagnosis
node_agent.operations.hung_summary
node_agent.operations.hung_triage
node_agent.operations.triage_samples
node_agent.operations.stack_samples
node_agent.operations.process_samples
node_agent.operations.flight_series
node_agent.operations.flight_recorder
node_agent.operations.process_identity
adapters.node_action.hung_survey
```

共享API、通用执行器及HostHealth其他故障类别不计入这个代码统计分母，
但链路用例实际执行了相应路径。
本机原始报告为 `/tmp/gfrs-nccl-coverage-h9itb3/coverage.json` 与同目录 `junit.xml`。
重新统计时对以上模块分别传 `--cov=gpu_fault.<模块>`，加 `--cov-branch`，
并将 `COVERAGE_FILE` 和报告路径指向新的独立目录，避免覆盖其他测试证据。

## 复跑

```bash
.venv/bin/python -m pytest -n 4 tests/execution/test_nccl_continuous_e2e.py
.venv/bin/python -m pytest -n 4 tests/execution/test_nccl_review_e2e.py tests/execution/test_nccl_review_boundaries.py tests/node_agent/test_nccl_collection_budget.py
.venv/bin/python scripts/e2e/run_general_fault_injections.py --case 011 --require-continuous
```

第二条还运行既有组件/规模回归，输出独立checks/report；执行期间不要修改源码，
否则其源码一致性保护会使回执无效。所有输出仍是software，physical_acceptance为NOT_RUN。
