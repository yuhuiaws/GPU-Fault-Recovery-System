# NCCL hung 改进与验证

## 当前流程

```text
有效 EFA RDMA 观测（不依赖 training step）
  |
  +-- EFA: 当前 job/attempt 曾活跃 + 连续近零 + grace/阈值
  |        不读取 step/phase；CPU/写入活动不延后计时
  |        计数无效或长观测空洞打断零流量连续性
  |
  v
同 cluster/job/attempt、同一停滞波次的并发 hung 事件合并
  |
  v
FREEZE_EVIDENCE: 全任务已有遥测摘要及覆盖率，不 attach
  |
  v
COLLECT_HUNG_SUMMARY / BASELINE: 全部任务节点，有界并发，不 attach
  |  workload cgroup -> 全部可发现进程 -> 两次 proc/RDMA
  |  RANK/WORLD_SIZE 可缺失；可选读取通信序号，不读取 step
  |  每 GPU 利用率、每 RDMA port 收发增量；明确缺失与时间跨度
  v
COLLECT_HUNG_SUMMARY / RECHECK: 基线轮全部结束或超时后，再访问全部节点
  |  按 node/PID/starttime 配对两轮通信记录，不跨 attempt、PID 复用配对
  v
COLLECT_HUNG_TRIAGE: 首轮至多 4 节点，结果不足再扩采一轮
  |  cgroup/starttime -> 已配置 FR FIFO 两次触发 -> 各 PG 时序摘要
  |  三次 py-spy JSON/proc/GPU 观察，保留线程和实际采样窗口
  |
  +----------------------------+
  v                            v
有界诊断包                     VALIDATE_FABRIC
可信候选 + 至多一个对照         全部当前任务节点的已有遥测验证
否则保留采样节点               不证明应用恢复
  +----------------------------+
  |
  +-- 人工复核，无自动 reset/reboot/停任务，不凭 step 宣告恢复
```

普通 NCCL/torch.distributed/collective 错误（包括 timeout 文案）归日志证据，
`NO_ACTION` 表示日志本身不触发恢复，不创建硬件 marker、incident 或额外诊断 workflow。
仍保存原始日志、PID/容器/attempt 归属；`COLLECTIVE_TIMEOUT` 等分类只用于解释日志。
watcher 沿用非零退出码和任务终态的通用处理：冻结证据、清理剩余进程、终态后按预算重试；
不增加 NCCL 专用识别或过滤。

只有独立持续停滞证据（目前主要为受管存活任务持续低 RDMA）才触发疑似 hung。
低 RDMA 是嫌疑而非通信根因证明，不依赖 train step，也不把一次 S/D 状态或 futex 等待
当成持续 NCCL 停滞。触发前及每个采集步骤执行前检查已知终态、失败 rank 和清理发起者；
watcher 已接管时跳过后续主动取证。未知观察不冒充已退出。
已经被 Agent 接收的只读命令无法撤销；其后不再发新的采集命令。
旧在途 `APPLICATION_NCCL_ERROR` 采集步骤也跳过，记录 watcher 归属。

## 判定边界

- NCCL 的探测、选点、扩采和结束判定均不读取训练 step/phase；也不要求训练程序
  声明何时开始 train step。删除了 EFA 路径中应用进度查询及相应的延期 helper。
  写日志、checkpoint I/O、CPU busy-loop 可以与 hung 并存，只作为上下文保留。
- EFA 两次有效观测间隔超过 `max(360, zero_warning_seconds)` 时重新开始零流量区间；
  不继承其他 attempt 的活跃基线。始终低流量不能仅靠此规则判 hung。
- EFA 默认最小历史活跃值 1 MiB/s、近零阈值 1 B/s、启动宽限 120 秒、
  warning 60 秒、hung suspicion 600 秒。历史至少一次达标样本即建立活跃标志；
  不要求紧邻低流量前仍活跃。入口是有效 RDMA 收发合计，不包括 TCP。
- 这些条件降低“从未通信”的初始化误报，但不能区别后续长 checkpoint、长计算和
  NCCL hung；600 秒是现场策略，不是 NCCL 故障的数学证明。不应据此自动停训。
- 独立、默认禁用的 training-health monitor 仍可监控应用 reporter；
  新 step-stall finding 使用 `TRAINING_PROGRESS_STALL`，不进入 NCCL summary/triage。
  旧 `TRAINING_HUNG` 输入保留兼容，但新编译流程按人工诊断合同处理。
- 每个 PG 保留自己的最后条目；按名称和成员列表分组，至少三个有效样本、严格多数。
  单次进度差异最多为 `PLAUSIBLE`，从不声称确认根因。缺失 FR 不产生候选。
  栈和行为线索为 `WEAK`，只在角色、硬件和时间窗口可比时做进程间比较；
  不授权定向 attach，缺计数不是零。
- 定向包要求 1～3 个有有效双快照的候选、该 PG 成员完整、无重复身份/不可达节点，
  候选和对照都有 workload cgroup 与 PID starttime 证据。执行时重新验证 starttime。
  无成员表的旧 FR 格式仍可提供线索，但不授权定向采集。

## FR 时序判断

每一轮 `COLLECT_HUNG_TRIAGE` 对同一批已验证 PID 获取两份新 FR dump。
节点内对全部目标先触发第一份，第一轮等待最多使用该节点采样预算的一半；
读取第一份后至少间隔 2 秒，再逐 PID 复核 cgroup/starttime 并触发第二份。
第二份必须比第二次触发前的文件更新，且读取前后的 inode/大小/mtime 一致。
半写入文件在剩余预算内重试；旧文件、第二份缺失或 PID 变化都不能当作停滞。
没有 FR 配置时不增加第二轮等待，不安装或自动开启 FR。

新 plan 的 `triage_timeout_seconds=20`，Agent 上限 25 秒；
`triage_fr_sample_interval_seconds=2`，可在 operation 参数中设为 1～10 秒。
这两个是步骤参数，不是新增环境变量。主动节点和并发预算不变；
全任务摘要新增第二轮，因此不能沿用原来单轮的开销估计。
两轮“节点扩采”与每节点“两次 FR”是两层不同概念，不得把不同节点的两轮数据
伪称同一 rank 的时间序列。原有 10 秒在途 plan 仍使用自身预算，预算不足会保留缺项。

每个 PG 的每份摘要分别保存观察到的 `last_enqueued/last_started/last_completed`，
以及最早未完成记录的序号和稳定操作身份。最高提交序号不等于完成序号；
没有可见的完成记录时保存 null，不能补成零。ring buffer 不保证完整历史，
计数倒退、成员变化或原未完成记录被淘汰时不推断其已经完成或持续卡住。
PyTorch dump version `2.10` 另外读取官方 `pg_status`：
通过 entry 的数值 `pg_id` 定位，不用 PG name/UID 猜键；
支持 pickle 整数和 JSON 十进制字符串的
`last_enqueued_collective/last_started_collective/last_completed_collective`。
合法状态水位优先于环形缓冲区可见最大值；缺失、未知版本、无法映射、倒序计数明确
标记，不把历史完整性设为 true。当前未验证的其他 dump 版本仅使用可见条目。
依据为 [PyTorch 2.10 FlightRecorderDetail.hpp](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/distributed/c10d/FlightRecorderDetail.hpp)。

控制面只配对同一 PID/starttime、同一组名称和成员摘要的有效快照：

- **完成落后**：同一未完成操作在两份 dump 中都存在，本 rank 未观察到完成推进，
  且严格多数同组已采 rank 在两份快照中均已完成到该序号。
- **开始落后**：本 rank 的同一操作两次都尚未开始，而严格多数均已开始到该序号。
- **提交落后**：本 rank 两次提交位置未变，严格多数在更后的同一未完成操作上等待。
- **整组未完成**：成员完整且全部成员都存在持续未完成操作、没有完成推进时，
  记录 `stalled_process_groups`，不能为了生成目标而虚构一个 culprit。

跨 rank 推断还要求至少三个有效 rank，且各自两次观察的时间窗口有交集。
跨节点需有可信的 `clock_uncertainty_seconds` 步骤参数，交集按最坏误差收缩。
没有时钟误差界限不假定为零；同节点组可以用同一时钟的本地跨度。
不重叠时仍保留本地 watermarks、增量与未完成操作证据，但不做跨 rank 比较；
千卡大 PG 在不同批次取样时可能触及该限制，不是全量诊断充分性的保证。
成员大于 256 时通过成员数和 SHA-256 校验所采集合，不能只比较数量。
参考值预先排序，单组比较为 O(R log R)，不逐 rank 对全部成员重复扫描。

`pending_observed_seconds` 仅表示该操作至少跨越了本次观察窗口，默认约 2 秒；
不是从操作 enqueue 起的总等待时间，更不是 EFA 的 600 秒。短窗口不足以单独证明 hung。
任一组内完成推进也不能证明其他组或整个任务恢复。结论最多 `PLAUSIBLE`，
始终 `root_cause_confirmed=false`；旧 Agent 的单快照仍可提供扩采线索，但不能
授权精确 PID 的后置采集。已识别的 P2P 记录不套用 collective 时序规则，
PP 拓扑识别和 send/recv 对端定位明确不在当前范围。P2P 条目保留在原始 FR 中，
`fr_semantics.p2p_analysis=OUT_OF_SCOPE`，不猜测 PP 组、不生成 P2P 异常 rank。
collective 使用 PyTorch 2.10 collective 分支的操作类型、shape、dtype 匹配规则；
不把合法的非对称 collective 张量差异当故障。

工作流保留两份紧凑摘要和采样状态，并在 Agent 私有目录保存不可覆盖的 first/second
原始文件及 SHA-256；单份原始留存上限 8 MiB、节点原始 FR 留存总上限 64 MiB，
超限明确记录，不能声称已经保留。选中节点的原始文件经摘要校验后进入后置包。
最多两次 dump、解析和复制都会增加开销，必须在真机测量。

复杂 collective 可在离线分析机使用官方分析器适配器：
`python -m tools.analyze_nccl_flights manifest.json`。该机器需自行安装 PyTorch 2.10；
manifest 的 `records` 包含 `rank/node_id/pid/process_start_ticks/path/sha256`，
每次分析选择同一轮、不同 rank 的原始文件。适配器拒绝重复 rank、越界路径、
混合版本和摘要不匹配，调用官方 `build_db(..., allow_incomplete_ranks=False)`。
缺依赖返回 UNAVAILABLE，不自动安装、不在训练节点启动。已在独立 CPU 环境安装
PyTorch 2.10.0+cpu 并实际调用官方引擎：114 组原生 `Op.match` 规则对照、
10 组 `build_db` 正反例通过，包含 all_gather/reduce_scatter/coalesced。
可用 `PYTHONPATH=src python -m tools.verify_fr_analyzer --report report.json` 重跑。
运行时轻量匹配是保留版权声明的官方 collective 规则适配，不在控制面引入 torch；
all-to-all 的 pairwise UNDECIDED、跨状态差异仍不强行判错。离线工具保留官方
完整匹配结果与缺失 rank、shape 等细节。这不是实际 CUDA/NCCL 执行验收。
PP/P2P 自动定位不在范围内。

## 停滞波次和流量重新布防

同一 attempt 的近零计时与停滞事件分开保存。大于近零阈值的样本打断 `zero_since`，
但不能凭 2 B/s 或单次流量尖峰结束 `stall_episode`。事件内仅首次达到 hung 门槛发出
hung finding；警告、流量突变仍可独立记录。稳定恢复后下一次近零建立新的 episode ID。

恢复配置均为站点策略初值，不是 NVIDIA 的统一建议：

| 环境变量后缀（统一前缀 `GPU_FAULT_EFA_TRAFFIC_`） | 默认 | 含义 |
|---|---|---|
| `RECOVERY_RATIO` | 0.25 | 可信历史基线的恢复比例 |
| `RECOVERY_FLOOR_BPS` | 65536 | 恢复噪声下限 |
| `RECOVERY_FALLBACK_BPS` | 1048576 | 无可信历史时的绝对恢复门槛 |
| `RECOVERY_SECONDS` | 120 | 连续达到门槛的观察时长，最少允许 60 秒 |
| `RECOVERY_MIN_SAMPLES` | 3 | 恢复窗口最少有效样本数 |
| `RECOVERY_MAX_GAP_SECONDS` | 60 | 相邻恢复样本允许的最大间隔 |

可信历史需要同 job/attempt 至少 3 个有效正常活跃样本；使用进入停滞前的 EMA 基线 B，
事件期间冻结。门槛为 `max(B × ratio, noise_floor, 2 × zero_bps)`；历史不足改用
fallback，绝不把 fallback 写成虚构的历史基线。初始化保护仍依赖曾有活跃 RDMA，
从未活跃不会因为低于 fallback 而启动 hung 事件。

不足门槛、无效观测、过长样本间隔均重开恢复窗口；稀疏遥测可能延后重新布防。
Host Collector 默认正常摘要间隔 300 秒，不足以满足上述恢复窗口。因此观察到流量从
噪声下限以下回升时，短期重发已有采样结果：默认最多每 30 秒一次、持续 180 秒；
不增加 `/proc`/GPU/NIC 查询或主动诊断。窗口结束回到既有摘要节奏，噪声反复切换
也受最小发送间隔限制。实际采集/投递故障仍不能当作恢复；这会增加有限的遥测上报。
持久化状态增加可选字段，旧记录默认无可信历史，跨 attempt 清空。
编排在同一 attempt 事务内验证 Store 中的 episode 和稳定恢复时间，不信任请求自报的
“已恢复”。新波次建立独立 incident/workflow 和生命周期预算；旧只读取证尚在执行时
排在其后，不抢占，也不继承旧事件已耗尽的生命周期。每节点每任务限频仍跨波次生效。
不同节点各有本地停滞区间，同轮并发报告合并，
某节点恢复不等于整个任务恢复。
若第二波首次出现在另一节点，可使用上一轮原触发节点的同 attempt 稳定恢复时间作为
重新采集的依据，要求新节点停滞起点晚于该时间；老波次的迟到报告仍不会开启新轮次。

`RECOVERED` 在 EFA 状态中仅表示流量事件已重新布防，不关闭 NCCL incident，
不触发调度恢复、任务重启或 GPU reset。

## 常驻通信摘要

复用 Agent 生命周期和已认证的 heartbeat 响应，不新增服务或公网接口。
控制面给授权 `COLLECT_HUNG_SUMMARY` 的 Agent 下发最多 90 秒的
`communication-monitor/v1` 或 `v2` 租约（按 Agent heartbeat 的 `communication_monitor_version`）：绑定
cluster/node/boot/generation、当前 job/attempt 和该节点任务容器的 scope（非根 cgroup；v2 另可带运行时容器 ID，见下文）。
无新租约不继续采样；清理或终态任务不再续租。
每节点最多 8 个任务，超限在 healthz 显示实际与期望数，不声称覆盖所有任务。

每任务每 30 秒一轮，扫描任务进程、GPU context、已有新鲜 FR 文件与实际 RDMA QP。
不触发 FR dump、不 attach、不读 training step。节点 SQLite 缓存按 job/attempt/boot
隔离，最多 30 分钟、2048 条、32 MiB 压缩 payload，单条压缩最多 1 MiB。
单条解压最多 8 MiB；结果按真实 epoch/采样时间保存，不伪造准点采样。
新诊断指定统一的两个已结束 epoch，晚到的批次读取相同历史区间，不改读“当前最新”；
缺 epoch 或未知时钟误差仍不能形成可信跨节点完成进度比较。

RAS 针对已验证的 NCCL 2.28 JSON 协议 2。每个任务当前分配中只选一个节点查询，
因为 `VERBOSE STATUS` 本身会向该任务 RAS peers 汇聚。分配变化时旧租约可能短暂重叠，
这不是跨控制面故障的严格分布式 leader lease。监听端口必须由任务 GPU 进程持有，
通过 `/proc/PID/fd` socket inode 验证，进入该 PID 的网络命名空间后仅连接 loopback。
没有 listener、旧 NCCL 无 JSON、无权限、超时、格式损坏均为 UNAVAILABLE。
版本未知不修补或猜解析。一次回复最多 8 MiB。

RAS `collective_counts` 是 NCCL `seqNumber` 提交数，不是 GPU 完成数。
两次完整成员快照中的持续提交落后仅提供 WEAK 选点线索；本地 communicator rank
不当成全局 RANK，按唯一节点地址及已验证 PID/NSpid/starttime 关联。
已有 FR 文件必须更新时间变化才能形成序列；没有启用 FR、不更新 dump 时完成进度仍未知。
后台摘要目前辅助取证选点，不凭 RAS 提交停滞启动另一条自动恢复链。

源码依据：
[NCCL 2.28 RAS 协议与 JSON](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/ras/client_support.cc)、
[NCCL seqNumber 采样](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/ras/collectives.cc)。

## 范围和成本

### 进程与线程证据

`nccl-hung-triage/v4` 在节点预算前半段内最多做三次进程/栈/GPU观察，
相邻轮次等待最多一秒。进程数默认16、最高32；每节点同时最多8个py-spy，
每次工具调用最多2秒。预算不足时保留实际成功样本数，不把两次或失败样本补成三次。
此处的“持续”是本次短窗中的重复观察，不是常驻监控，也不是已有正常训练历史基线。

py-spy使用固定版本0.4.1的 `dump --json --nonblocking --pid`，不启用locals；
格式依据其 [dump实现](https://github.com/benfred/py-spy/blob/v0.4.1/src/dump.rs)
与 [线程结构](https://github.com/benfred/py-spy/blob/v0.4.1/src/stack_trace.rs)。
非零退出码、错误文本、坏JSON、PID不匹配均单独记录失败，不生成可用栈签名。
stderr不拼进JSON。保留Python/native线程ID、线程名、帧文件/函数/行号和角色，
用 `/proc/<pid>/task/<tid>/status` 的NSpid映射容器线程ID到宿主机，
再用线程starttime验证三次观察中是同一线程。
每份栈最多32线程、每线程16帧、摘要8KiB，优先保留显式训练线程与主线程，
截断和实际线程覆盖数明确报告；签名指保留的调用位置，不声称完整原生栈一致。

默认只知道MAIN或UNKNOWN，不把MainThread自动认定为训练线程。应用可声明
TRAINING/COMMUNICATION/BACKGROUND角色；未解析出的后台/原生线程不伪称已覆盖。
只有靠近栈顶、函数名精确匹配且位于torch/distributed路径的调用才标为
`DISTRIBUTED_CALL_CONTEXT`，仍不等于正在阻塞于NCCL。单独出现distributed_c10d、
用户函数名all_reduce或CUDA synchronize都不能证明NCCL hung。

CPU核数使用 `delta(cpu_ticks) / SC_CLK_TCK / actual_elapsed_seconds`，
不是直接横比tick增量。每次最多扫描64个kernel TID，读取失败也占扫描预算；
主线程wchan不代表整个进程，空值、0、0x0不作为等待位置。D状态和等待位置
按同一TID/starttime跨三个样本判断，时长是观察跨度，不是连续阻塞时间。
缺CPU或GPU数据不会丢弃其他有效的I/O/线程证据，分别记录覆盖缺项。
GPU利用率检查有限值和0～100范围，并复核前后PID/GPU映射；
它始终是设备级读数，标记是否与其他进程共享，不冒充该PID的GPU利用率。

所有证据分支独立执行并保存到 `evidence_branches`：
FR时序、FR单快照、线程栈、CPU/I/O/等待状态。FR有完成进度也不抹掉I/O线索，
但后者不能据此推翻通信证据；不同来源各保留自己的窗口，不伪称严格同步。
删除原来的四项等权评分及跨所有rank的80%栈比较。输出
`REPEATED_D_STATE`、`REPEATED_THREAD_WAIT`、`CPU_LOW_ACTIVITY`、
`HOST_WRITES_OBSERVED`、`GPU_DEVICE_IDLE_WITH_LIVE_PROCESS`等具体观察。
同一I/O等待引出的多个现象不累加成故障概率。

诊断身份使用node/PID/starttime，`candidate_processes`可以没有rank。
`culprit_ranks`仅保留FR路径的兼容字段，不把进程症状升级成NCCL根因。
输出最多256个进程详情及候选，同时保留真实总数和截断标志；
原始有界节点证据仍保存在workflow。弱PID线索不授权停任务、reset/reboot，
也不放宽后置精确采集所需的FR双快照与完整成员条件。

### 可比角色声明

没有角色元数据时只分析进程自身的重复观察，不按PID顺序、GPU位置、
rank连续性或栈形状猜TP/PP/EP角色。可选进程快照增加 `diagnostic_context`：

```python
diagnostic_context = {
    "group_name": "tp-stage-0",
    "members": list(range(8)),
    "role": "trainer",
    "pipeline_stage": "0",
    "expert_role": "none",
    "hardware_class": "p5",
    "code_version": "training-image-digest",
    "thread_roles": {"1": "TRAINING"},
}
```

通过下面的 `write_rank_progress(..., diagnostic_context=diagnostic_context)` 写入。
线程ID须来自实际训练线程，不要把负责写快照的后台线程登记为TRAINING。
角色声明需由训练框架适配，不自动注入；无PP/EP时显式填none，
unknown/缺失声明不参与跨进程比较。文件不需要step字段，Agent核对job/attempt、
进程PID、starttime、30秒新鲜度及首末角色声明一致；容器PID通过 NSpid 核对。
这证明声明的身份归属，
不等于平台独立验证了其业务语义。
组成员转成摘要避免千卡列表逐进程重复回传。

仅同任务、同组成员、同计算角色、PP stage、EP角色、代码版本、声明硬件类别，
且实测GPU型号一致且不共享GPU的进程成为可比对象。各自窗口还必须有交集。
只有明确且唯一的TRAINING线程才做角色内栈比较；CPU比较使用归一化后的核数。
角色内80%栈一致或CPU相对中位数的差异仍只是WEAK排查线索，不是故障概率。

全任务已有遥测摘要仅使用一次 collector-status 批量读取。
包含 allocation 时间/节点数、HOST/GPU 新鲜通道数、已分配容器数，
以及最多 256 条 allocation 示例，显式标记截断；不查询应用 heartbeat/step。
这是已有遥测的汇总，不是新采集的全 rank FR/proc 快照。watcher 的 container rank
不等同于容器内全部 torch worker，因此 `rank_inventory_complete=false`。
RDMA 数值只引用触发节点，不能伪称已获取其他所有节点的流量。

新增的主动全任务摘要与上面的已有遥测汇总是两层不同证据：
`COLLECT_HUNG_SUMMARY` 扫描当前 workload cgroup 的进程，从实际进程环境读取
`RANK/WORLD_SIZE`（如果有），没有环境变量也保留 node/PID/starttime 及进程状态，
不把一个容器的 watcher rank 当成它的全部 worker。无 rank 的 helper 进程也保留，
不能由此声称所有训练 rank 都已被识别。

workload cgroup 从哪里来。watcher 只从 Pod 注解 `gpu-fault.io/cgroup-path` 得到容器的 cgroup 路径，
而这个路径含运行时容器 ID，容器启动前不存在，受管 YAML 契约和产品都不写它；于是每个真机受管任务的
hung DAG 都只有 `workload_cgroup_paths_by_node={node: []}`。HyperPod GF-GENERAL-LIVE-011 live3/live4 全部变体
（2026-10-24/25，含 b161 之前的 bf1bb0037847）：两节点的 summary 都是 `SCOPE_UNKNOWN`，`after_summary` 无可选节点，
triage 以 `NO_VALID_SURVEY` 一个节点都不请求（`active_requested_node_count=0`、节点预算 2 未用），
`fr_enabled` 与 `fr_disabled` 都报 "COLLECT_HUNG_TRIAGE sampled no rank process"；VALIDATE_FABRIC 的
`scope_basis=NOT_REPORTED` 也出自同一原因（summary 在计算任务端口绑定之前就返回了）。
现在控制面另按节点下发 kubelet 在 `containerStatuses[].containerID` 报告的容器 ID
（`workload_container_ids_by_node`，只含没有显式路径的运行中容器，只接受 64 位小写十六进制，每节点最多 64 个；
没有可用 ID 时不下发该参数）。Node Agent 在 summary 和 triage 开始时读本机 `/proc/<pid>/cgroup`，
把以该 ID 命名的路径段（`cri-containerd-<id>.scope`、`crio-<id>.scope`、`docker-<id>.scope`、cgroupfs 的 `<id>`）
截到该段为止作为容器 cgroup，与显式路径合并成本节点的 scope；`crio-conmon-<id>.scope` 是运行时监控进程，
不属于容器，嵌套在容器 cgroup 下的子 cgroup 仍属于容器。之后的身份复核（summary、triage、FR 双快照）都读这一个 scope。
结果的 `workload_scope` 记录 `basis`（`EXPLICIT_CGROUP_PATHS`/`RUNTIME_CONTAINER_ID`/`UNAVAILABLE`）、
请求/解析/未解析容器数和最多 8 个未解析 ID；一个都找不到仍是 `SCOPE_UNKNOWN`
（`reason=CONTAINER_CGROUP_NOT_FOUND`，扫描被截断时为 `PROCESS_SCAN_INCOMPLETE`），只找到部分时
`scan_complete=false`。只读 `/proc`，不扩大 attach 范围：其他作业的容器、同一 GPU 上别的进程都不在 scope 内。
旧 Agent 忽略该参数，行为同修复前；旧 workflow 没有该参数，也同修复前。
被动通信监测（communication monitor）原先同样只认显式路径：受管 Pod 的租约里 `expected_target_count=1`
却没有 target（Agent healthz `expected_targets=1`、`assigned_targets=0`），从不采样，hung summary 的
`communication_history` 也就一直是 `UNAVAILABLE`/`INSUFFICIENT_SAMPLES`——只读证据缺失，没有采错进程。
现在 target 可以带 `container_ids`（该节点上没有显式路径的运行中容器的 64 位十六进制运行时 ID，最多 64 个；
有一个容器既无显式路径也无可用 ID、路径为根或含 `..`、或 ID 超过 64 个，该节点仍不分配 target）。
旧 Agent 用 `extra="forbid"` 校验租约，多一个字段就拒收整个计划，所以新字段按 Agent 能力下发：
升级后的 Agent 在 heartbeat 里通告 `communication_monitor_version=communication-monitor/v2`，
控制面按 Agent 通告的版本应答——v1 Agent 收到与修复前逐字节相同的计划（空 `container_ids` 不序列化，
受管 Pod 仍无 target），v2 Agent 收到带 ID 的 v2 计划（计划的 `version` 写明版本），未知版本不给租约。
Agent config digest 里的 `communication_monitor_version` 仍记 v1（它表示「被动监测在运行」，协议版本逐个 heartbeat 协商），
控制面的 digest pin 不变。节点上采样进程用同一解析器把 ID 解析成容器 cgroup，进程扫描、RDMA 绑定、RAS 查询都只在这个 scope 内；
样本文档与结果带 `workload_scope`；只解析到部分容器时 `scan_complete=false`，一个都找不到时结果为 `SCOPE_UNKNOWN`
且不写历史行，绝不记成空任务。兼容：旧控制面不认识 v2，不给升级后的 Agent 租约，在控制面也升级之前被动监测空转
（发布先升级控制面再滚 Agent 波次；旧控制面对受管 Pod 本来也不分配 target）。

QUIESCE_GPU_SERVICES 的设备清扫也读同一个缺失的输入：清扫只可 TERM/KILL 目标 GPU 上位于受影响 workload cgroup
内的持有者，受管 Pod 的 scope 一直为空，停训后仍握着 GPU 的训练进程留在 `unswept_device_holders`
（`reason=outside_workload_cgroup_and_process_whitelist`），VERIFY_NO_GPU_CLIENTS（以及 reset 自己的预检）
以 `GPU device clients are still active` 拒绝 reset——fail-closed，不会带着作业进程 reset，只是该恢复的没有恢复。
现在计划在 QUIESCE 里同样冻结 `workload_container_ids_by_node`，但只在同一计划的 STOP_WORKLOADS 排在该 QUIESCE 之前时才带
（清扫只会遇到已停 attempt 的残留进程，绝不碰计划未停的作业）；Agent 在 quiesce 开始、任何服务停止之前用同一解析器解析，
加进清扫 scope；ID 格式不对（非 64 位十六进制、每节点超过 64 个、不是列表/映射）在武装 fail-safe 和停服务之前就拒绝 quiesce。
没有进程的容器不贡献 scope（STOP_WORKLOADS 之后这是常态），扫描被截断只加入看到的容器；scope 之外的持有者照旧留给 VERIFY 拒绝。
回执带 `workload_scope`；其他作业的容器、CRI-O 的 `crio-conmon-<id>.scope`、主机进程从不进入 scope。

每轮采两次 `stat/io/status/wchan`，不扫描所有线程，不读 Python 栈、不触发 FR，
每节点每轮两次有 2 秒命令上限的 nvidia-smi 查询，两轮总计四次，GPU/NIC 信息按设备去重。
进程退出、starttime 变化、坏 RDMA 计数对、counter reset 都记为不可用而非零。

NodeAction adapter 复用 `GPU_FAULT_NODE_ACTION_MAX_PARALLEL`，最多同时在途 32 节点，
但不截短全任务目标列表。短执行片段持续补充空位，保存已完成结果和 pending 起点；
同一轮续跑不重复已成功节点，第二轮是不同幂等命令。单节点等待上限 30 秒，每轮 survey 根据规模计算预算、
最高 540 秒；上限耗尽后逐项记录缺失，不能宣称全采成功。每节点最多扫描 16384 个
进程、返回 256 个 rank 和 1 MiB 数据。Executor 对节点摘要做有 SHA-256 的无损
JSON/zlib 封装；工作流编码后总预算 8 MiB、解码后总预算 32 MiB，避免包含多个通信组的
万 rank 摘要撑大后续 RPC。控制面逐节点有界解压、验长度和校验和，禁止压缩炸弹。
已采集的结构化证据保留在 workflow，不依赖故障节点仍在线，可用
`gpu_fault.hung_summary_codec.decode_summary` 读取 `node_results` 中的单节点封装。
超出预算标记不完整，不缩小覆盖率分母。除两轮之间的执行屏障，新增上述节点后台历史；
缓存的固定 epoch 序列与当次本地 proc/GPU 观察分别记录，不混成同一时间窗口。

覆盖率分开记录节点、进程身份有效数量、无 rank 进程、有效 rank、通信进度和每 PG 成员。
只有 `WORLD_SIZE` 一致、rank 唯一且覆盖 `0..WORLD_SIZE-1`、全部节点扫描完整，
且没有身份未知的参与者，才设置新的 survey `rank_inventory_complete=true`。
只有进程身份有效，且有本进程 GPU context 或绑定 PID/starttime 的通信注册，
才进入参与者清单；继承 RANK 的 DataLoader/launcher 保留为辅助进程，
不参与 rank 去重，也不因没有 rank 而破坏训练参与者覆盖率。
GPU 查询失败时不能把全部非 GPU 进程直接判为辅助，保留 UNKNOWN。
GPU context 只是参与者候选证据，不证明进程已经创建 NCCL communicator。

`node_coverage_fraction` 分母是任务节点数；`participant_coverage_fraction` 和
`rank_coverage_fraction` 使用一致的 WORLD_SIZE，未知时为 null。
每 PG 单列成员覆盖率、成员 SHA-256 校验、有效时间窗口；
`communication_time_series_coverage_fraction` 仅针对已声明并观察到的组成员，
`communication_coverage_basis=DECLARED_OBSERVED_GROUPS_ONLY`，
不证明发现了任务所有 PG。没有应用通信文件、有效 FR 或 RAS 时，
即使节点响应 100%，通信覆盖仍未知；RAS 不能补齐 FR 完成进度覆盖率。
GPU/RDMA 缺项另外列出。两次均处于 D 状态的进程可提升采集优先级，但仅为
`WEAK` 线索，可能是正常存储 I/O，不是 NCCL 或 Fabric 根因。
两次采样是节点本地的两个时间点，不假设全任务同时采样；全任务起始时间差大于
30 秒时保留 `cross_node_time_comparable=false` 作为节点采样跨度提示。
各 PG 的通信记录必须有正的共同观察交集，扣除时钟误差后仍相交才比较；
全任务两轮配对跨度上限 600 秒，主动 FR 对上限仍为 30 秒。
不能仅因节点响应时间接近就认定通信快照同步。
这是规模化采集的明确限制，需要真机测量批次耗时，不能用软件规模测试代替。

`GPU_FAULT_HUNG_TRIAGE_MAX_NODES` 默认 8，现在约束两轮 FR/栈诊断的合计节点数；
显式配置现在最多 256，旧值 0 保留为 32，而不是无限制。
全任务状态采集不受此总节点数限制。总预算和并发不同：主动两轮也改用持久化滑动
窗口，最多在途 `min(GPU_FAULT_NODE_ACTION_MAX_PARALLEL, 32)` 节点，
未完成命令占据窗口；续跑保留成功节点，逐节点等待 30 秒、整轮最高 540 秒，
主动返回证据也使用相同的无损压缩封装，受 8 MiB 编码预算及 32 MiB 原始预算约束。
控制面兼容旧的未压缩返回，解压预算及失败节点明确记录。
首轮优先全任务摘要中的候选、触发节点，并在有可信 `fabric_partition` 时分散选点；
证据不足时优先补首轮 FR 中已知通信组的未采成员节点，再考虑被动摘要的相关
PG、D 状态线索和剩余节点，不重复首轮。不再因应用 step 前进跳过诊断。
无有效新版本 survey 时仍跳过，不回退到旧 Agent 上全量 attach。
summary、triage、bundle、Fabric 的节点集合分别记录。
无可信候选的后置 bundle 单独限制最多 32 节点，不因主动预算增大而扩大。
每轮主动结果和全任务摘要仍保存在 workflow；后置包的遗漏节点单独列出。
bundle 命令带到节点的 `hung_triage_decision` 是线上形式（`hung_bundle.wire_decision`）：判定、候选、覆盖率与 FR 分支保留，
逐进程逐线程观测 `evidence_branches.process_observations.processes` 换成计数，`wire_form` 记全量的字节数与 SHA-256；
全量判定只在 workflow 记录（bundle step 参数、triage step 执行记录）。节点侧观测请求预算 256 KiB
（`OBSERVATION_REQUEST_BUDGET_BYTES`），超预算先裁掉仅作上下文的 `hung_triage_decision`（占位写明 `omitted`、字节数、SHA-256，
manifest/结果的 `observation_request_trimmed` 列键），bundle 照常采集；裁后仍超才拒绝，拒绝带 `error` 消息、
`failed_component`、`failure_stage`、`worker_started`。真机 `tid_namespace` 曾因 259 KB 全量判定在 5 ms 内整体失败
（`error="ValueError"`，worker 未启动）。
worker 交回的单条结果上限 1 MiB（`OBSERVATION_RESULT_BUDGET_BYTES`，含换行）。真机 HyperPod `fr_enabled`（2026-10-25 23:05Z）：
triage 认全两节点 16 个 rank、每 rank 两次 dump 都在 triage 窗口内写入（PyTorch 日志 32 行 "preparing to dump debug info"，
16 个 29.5 KB 的 dump 在 `/tmp/nccl_trace_rank_*`），但 8 rank × 2 份 dump × 64 条操作摘要被序列化四份
（`last_entry`、`last_entries_by_pg`、series 的 `first` 与 `second`），叠在不开 FR 时 8 rank 约 290 KB 的进程/栈序列之上约 1.2 MB，
worker 把整份结果换成 `{"status":"RESULT_SIZE_LIMIT","ranks":[]}`，两节点的 step `node_failures` 都是
`ObservationIncomplete: RESULT_SIZE_LIMIT`、`node_results={}`，判定一个进程都没有，验收运行器读到 "sampled no rank process"——
rank 发现、容器 scope 与 dump 位置都没有问题，任何开 FR 的 8 GPU 节点都会触发。现在 triage 交出结果前先按
`operations/triage_budget.bounded_triage_result` 的阶梯压到 `TRIAGE_RESULT_BUDGET_BYTES`（1 MiB 减 64 KiB）以内：
先无损地把 `last_entries_by_pg` 中 series 快照已带的操作列表换成 `semantic_entries_in` 指针，再去 `unfinished_entries`、
把每组操作列表缩到前 32/16/8 条（未完成操作在前）、整列去掉（`progress` 水位保留），再把栈/进程序列只留最新一次采样，
最后是地板 `skeleton`：每个 rank 行只剩 pid/rank/身份（完整 `process_identity`）、角色与状态字段、FR 状态与 dump 路径/大小/
`raw_evidence` 引用（不含内容）、各组的组身份、最新操作位置与 `progress` 水位及 `semantic_entries_total`、series 的 `status`
（PAIRED/…）、时间戳与计数、栈/进程/GPU 序列的计数和 `python_stack`/`proc` 的状态，行盖 `skeleton: true`，节点级字段原样；
每个 rank 的身份、FR 状态、`progress`、`raw_evidence` 每一级都保留，结果的 `evidence_trimmed` 记预算、裁前/裁后字节数与各级名称
（用到地板时含 `skeleton` 及其字节数）。读取器的 `last_entry` 不再重复该组的 `semantic_entries`。只有骨架本身就超线的文档才换成占位，
占位带 `error`（字节数、预算、最重字段）、`failed_component=observation_worker`、`failure_stage=WORKER_OUTPUT`、`result_bytes`，
并以 `ranks_summary`（pid、rank、状态，按结果顺序装满 16 KiB 为止）、`ranks_total`、`ranks_summary_truncated` 点名被替换的 rank。
本地复现：真机 8 rank 形状 1,006 KB → 781 KB（只用无损一级），16 rank 2,003 KB → 875 KB，两者的控制面判定与全量结果相同；
64 rank 8.85 MB → 369 KB（地板，5.7 KB 一个 rank），FR 时序/快照分支的判定与全量相同。summary 与 bundle 走同一条 worker 结果线，
但 bundle 结果只是引用（不到 1 KB）、summary 按构造有界（每节点最多 256 行、实测 365 字节一行），都不设地板。
hung 包跳过 DCGM diag、bug-report 与 strace；
FR 文件读取上限 64 MiB，结构遍历有界，环形 pickle 不递归失控。
20 秒是新版 triage 内部采样预算；生产 Agent 另用隔离只读 worker 约束父进程等待，
后台/summary/triage/bundle 分别为 12/20/28/120 秒，另有有限的清理宽限。
有 job/attempt 的 summary/triage/hung诊断包，各允许每节点每 300 秒最多两次新命令，
使用持久化预算及节点文件锁，Agent 重启不清空限额；命令重试先命中原幂等结果。
节点同时只准一个 summary/triage/hung诊断包运行，单节点最多 8 个 py-spy。
该唯一 worker 槽也被每 30 秒的被动通信采样占用：真机 `fr_temporal_completion`（HyperPod，2026-10-26 04:21:31Z）两节点
在采样 epoch 开始后 1.45 s 收到 triage，槽均被占，各 1.2 s 回 `WORKER_STILL_RUNNING`，无 worker 运行、判定零进程。
现在控制面命令等待该槽（自身 timeout 的 0.3、至多 10 s 且在 timeout 内；triage 至多 8.4 s、bundle 至多 10 s，不超控制面单节点 30 s），
占槽者为被动采样时抢占它（停整棵子树，下个 epoch 再采），命令间从不互抢；整段等待仍拿不到才回 `WORKER_STILL_RUNNING`，
且是**可重试**（`worker_started=false`、点名占槽者与等待秒数），survey 适配器在单节点 30 s 内重发。
worker 拿到槽后按"deadline 在规划时刻所剩时间减 2.5 s"规划 triage 预算（默认 20 s；supervisor 交的是 deadline 本身——
`CLOCK_MONOTONIC` 读数——guard 与 worker 的启动在训练占满 CPU 时可达数秒，已在其中扣除，早于本修复的 Agent 不扣），等过槽的命令照样把能采的采完；
8 rank 两份 dump 实测约 4.5 s 配对完（`slot_wait` 0.05 s），远在 20 s 内。triage/bundle 在检查点发部分文档
（triage：RANKS_DISCOVERED→PROCESS_SERIES_SAMPLED→FIRST_FLIGHT_DUMPS_READ，带 dump 的检查点在副本上走裁剪阶梯；
bundle：每采完一项发 `BUNDLE_CAPTURING` 带已采/待采/`elapsed_seconds`），supervisor 的 deadline 或抢占落在最终结果前时
交回最后检查点（`PARTIAL_TIMEOUT`/`PREEMPTED`），慢 triage/bundle 带回已采 rank/dump 与缺项而非空结果；最终结果已到、输出已结束的 worker 树，supervisor 等其退出直至 deadline（原只等 0.1 s，CPU 饥饿时完整结果会被记为 `PARTIAL_TIMEOUT`）。
真机 `fr_second_missing`（2026-10-26 04:46Z）bundle 仅在共享节点 `hyperpod-i-08d8…`（另载一 lane 的 prewarm/训练 Pod、
被动采样目标更多）以 `WORKER_STILL_RUNNING` 整体失败、另一节点正常；本修复令其先抢占等待，真超时交回已采清单。
诊断子进程 stdout+stderr 上限 1 MiB，超时后 kill 等待再限 0.2 秒，
不在 subprocess.run 的无界 wait 上卡死。通信文件读取和 FR 解析使用最多四个
有界等待的读取槽，超时线程仍占槽直到实际退出，防止故障期间无限创建线程。
隔离 worker 的 CPU 时间上限 8 秒，继承 Agent 的 cgroup v2
CPU 至多 2 核、memory.max 至多 1 GiB；无法验证实际配额则返回缺项，不静默无界运行。
内存约束针对实际 cgroup 计费内存，不用低 RLIMIT_AS 限制 GPU 工具的虚拟地址映射。
既有 systemd 部署具有这些配额，非 systemd/cgroup v1 环境需先完成资源封装。
subreaper guard 回收整个工具子树；父进程超时可返回部分证据，但只要子树未退出，
仍占据唯一 worker 槽及 GPU activity，阻止新增采集堆积和并发维护。
这不等于能杀死 D 状态；极端内核故障和真实吞吐开销仍需实测。

并发去重使用已有 attempt-group Store 事务，适用于聚合窗口为 0 的情况；
同一波继续吸收通知；持久化稳定恢复后出现新波次，允许新的有界采集轮次。
硬件故障仍由既有硬件恢复优先级处理，不能由 hung 证据授权重启。
`TRAINING/HUNG_DIAGNOSTIC` 步骤失败以 `ESCALATED` 收尾，而不是通用只读诊断的
`RECOVERED`，避免把未取到证据当成训练恢复。
只读 hung DAG 总会走到 `ESCALATE_SUPPORT`：某个取证/验证步骤 FAILED（含 600 s
单步上限）时，该步骤保留 FAILED 记录并与依赖它输出的步骤一起退役
（`superseded_step_indexes`，`PLAN_REWRITE` 事件 code `HUNG_SUPPORT_HANDOFF`），
不依赖它的取证继续执行，`ESCALATE_SUPPORT` 在依赖解决后照常运行；workflow 以
该失败为 `terminal_failure_reason` 结束为 FAILED，incident 为 `ESCALATED`，从不写
`diagnostic inconclusive`。超过执行期限后不再启动新步骤，只有交接仍运行；
超过 lifetime 则按既有路径交给 `ESCALATE_OPERATOR`。计划里一旦含任何改动节点的
操作（合并后的复位/重启等），仍走原失败语义。普通只读诊断（详细设计 §6.3）不变。
（HyperPod 真机 GF-GENERAL-LIVE-011：VALIDATE_FABRIC 等满 600 s 后 FAILED，
`ESCALATE_SUPPORT` 从未执行，没有工单也没有通知。）
交接通知是专用的 `hung-support-handoff-zh-v1`（`notifications/hung_support.py`，category
`OPERATOR_REVIEW`，每个 workflow 一封），不是硬件升级邮件：后者写「hardware taken out of service」、节点
「unschedulable and quarantined」，对只读 hung 诊断都不成立。通知按各步骤自己的记录
（`execution/hung_handoff_evidence.py`）写出：只读声明、job/attempt 与节点、每步结果与记录键、summary/triage
文档 sha256、诊断包归档引用与 sha256、FAILED 步骤的 error 与被退役步骤依赖的失败步骤（超过执行期限未启动的步骤写明期限）、任务端口 Fabric 结论
（UNKNOWN 或 `PARTIAL_TASK_SCOPE` 时逐节点 `reason` 与复查 summary 的自述）以及运维下一步；缺失或畸形字段读作
UNKNOWN/未记录，每步最多列 8 个节点、其余计数。`ESCALATE_SUPPORT` 的 details 为
`support_disposition=READ_ONLY_HUNG_DIAGNOSTIC`、`nodes_isolated=false`、`workloads_stopped=false`、
`fabric_verdict`，不再写 `hardware_disposition=OFFLINE_QUARANTINED`。
NCCL 流程结束只表示取证完成/转人工，不表示训练已恢复。Fabric 验证是
相关节点已有遥测验证，不是新启动跨节点 NCCL benchmark，也不等待 step 推进。
新 hung 流程要求 `training_fabric_by_node` 中来自可信任务设备绑定的
`basis=TASK_DEVICE_BINDING`、`network_interfaces`、`rdma_ports`。
只验证这些设备，逐端口缺项返回 UNKNOWN；只有管理网卡正常不能通过训练网络检查。
未显式绑定时，summary 可通过 `rdma -j resource show qp` 中任务 GPU 进程的活跃
RTR/RTS QP 自动建立绑定，复查 PID/starttime/cgroup 和 sysfs 端口存在性；
任何参与者缺映射仍 UNKNOWN。它证明任务进程使用这些 RDMA 资源，不证明资源只用于
NCCL；不能从管理网卡或 RAS bootstrap 地址猜训练数据端口。
该表的 `pid` 是**创建 QP 的线程**而不是进程：内核 `fill_res_name_pid` 输出
`task_pid_vnr(res->task)`，用户态 QP 的 restrack task 是创建时的 `current`
（torvalds/linux `drivers/infiniband/core/nldev.c`、`restrack.c`、`verbs.c`）；
NCCL 在 proxy 线程调用网络插件的 listen/connect/accept（`src/transport/net.cc`
的 `recvProxySetup`、`sendProxyConnect`、`recvProxyConnect`），aws-ofi-nccl 按
调用线程创建 endpoint（`get_ep(domain_key, nccl_net_ofi_gettid())`），EFA QP 随之
建在 proxy 线程上。Agent 因此把 QP 的线程号经 `/proc/<pid>/task/<tid>` 归到所属
rank 进程；归不到参与者的线程（其他作业、内核 QP）不计入。早于此修复的 Agent
在真实 NCCL/EFA 任务上从来匹配不到 rank PID，绑定总是 UNKNOWN
（`reason=TASK_RDMA_COVERAGE_INCOMPLETE` 为参与者未全覆盖）。
任务端口范围无法确定时（未绑定或绑定无效），只要本 workflow 之后有
`ESCALATE_SUPPORT`，VALIDATE_FABRIC 立即以显式 UNKNOWN 结束：步骤 SUCCEEDED，
`details.fabric_status=UNKNOWN`、`fabric_evidence_scope=TASK_SCOPE_UNAVAILABLE`
（部分节点已绑定时为 `PARTIAL_TASK_SCOPE`）、`validated_nodes` 不含这些节点、
`unvalidated_nodes`，`task_fabric_unknown_nodes.<node>` 带 `reason`
（`TASK_NIC_PORT_SCOPE_UNAVAILABLE`/`INVALID_TASK_DEVICE_BINDING`）以及该节点复查
summary 自报的 `scope_basis`、`scope_reason`、`scope_source`、进程覆盖计数
（未回报时 `scope_basis=NOT_REPORTED`）。复查 summary 已经跑过，之后没有任何东西
会再填这个范围，等待只会耗尽单步上限。后面没有交接的 workflow 仍等待，
不让 UNKNOWN 以 SUCCEEDED 结束成 `RECOVERED`。
带宽、邻接、GPU P2P 实测需另行协调暂停或终止训练，本流程不自动执行。

### 千卡配置和边界

以 1024 GPU / 128 节点为例：全量被动摘要目标始终是 128 节点。
默认 8 台主动预算只能覆盖少量节点，不保证定位根因；需要全节点可扩采时，
在控制面显式设置 `GPU_FAULT_HUNG_TRIAGE_MAX_NODES=128`，执行侧并发例如 16。
首轮最多 4 台，证据不足再最多补 124 台；首轮已满足严格定向证据条件时可提前结束。
不能将“预算 128”写成“128 台均采集成功”，超时/字节上限/Agent 失败仍计入遗漏。
对更大任务，256 是本次实现的主动负载保护上限，不是定位充分性结论；
全任务摘要仍尝试全部节点，超限或无法定位时明确人工接手。

## 可选应用接口

应用进程可设置 `GPU_FAULT_RANK_PROGRESS_FILE` 指向其容器内每个 rank 独立的 JSON，
并调用 `gpu_fault.rank_progress.write_rank_progress` 原子更新：

```python
from gpu_fault.rank_progress import write_rank_progress

write_rank_progress(
    "/tmp/rank-0-progress.json",
    job_id="example-job", attempt_id="example-attempt",
    rank=0, world_size=8,
    process_groups=[{
        "name": "dp", "members": list(range(8)),
        "enqueued": 100, "started": 100, "completed": 99,
    }],
)
```

省略路径时读取该应用环境变量；目录必须已经存在。接口只写本地文件，不调用 CUDA、
NCCL、barrier 或其他 collective。需把相应模块交付到训练镜像，或按同一
`rank-progress/v1` JSON 契约由应用自行写入；不自动注入客户训练程序。
可选 `process_groups` 每项包含 `name/members/enqueued/started/completed`；
序号必须来自框架自身的可靠计数，不从 EFA/CPU 推断。Agent 验证 job/attempt、
rank、进程 starttime、30 秒新鲜度与 PG 成员；缺失、格式错误或旧文件只标记不可用。
当前 writer 自动写 `process_pid`；手工实现协议也必须提供此字段。
旧版不带 PID 的文件不再作为可信通信证据，避免子进程继承同一路径及相同启动 tick。
成员列表在节点返回时转为名称、成员数及 SHA-256，rank 只引用组键，避免逐 rank
重复传输大成员表；节点线上的组键使用局部短引用，控制面再恢复全局键。
FR 的大于 256 成员的组也只传成员数与 SHA-256，不在每个 rank 的最后条目重复万 rank
名单；这种压缩不允许局部 FR 样本冒充组成员全采。两个时间点均落后且未变化的 rank
只作为诊断优先线索。

该文件接口供取证读取，不会自行向控制面周期上报或触发告警。NCCL reader 忽略
文件内 step/phase，格式异常的 step/phase 也不影响合法通信序号读取。
无该文件、无 FR 的任务仍可由 EFA/日志触发并采集进程、GPU、RDMA；
无法从这些信息证明通信进度时必须保留 `UNDETERMINED`。

## 探测 Review

| 问题 | 本轮处置/剩余边界 |
|---|---|
| 缺 step、step 始终前进或 phase 来回切换影响 NCCL 判断 | NCCL 全链路不再消费这些字段 |
| 写入/CPU 活动掩盖 NCCL hung | 不再延期，只保留上下文 |
| 默认只采少量节点是否能保证千卡定位 | 不能；全任务摘要 + 显式可扩主动预算 + 独立覆盖率 |
| 缺 RANK 环境变量的进程被漏掉 | 保留 PID/starttime，rank 覆盖标未知 |
| 最大提交序号掩盖完成停滞 | 双 FR 快照分别比较提交/开始/完成以及最早持续未完成记录 |
| 全部 rank 同时卡住却没有离群者 | 记录有覆盖证明的整组未完成，不虚构单 rank 根因 |
| 第二份 dump 缺失或时间窗口不相交 | 保留缺项，不授权基于该证据的精确采集 |
| 取证失败被通用收尾标为 RECOVERED | hung/训练路径已改为 ESCALATED |
| 单节点 RDMA 接近零能否证明 NCCL hung | 不能，仍是嫌疑信号，不授权自动停训/reset/reboot |
| 子通信组 hung，但其他组仍有流量 | EFA 聚合规则仍可能漏报，需通信 watchdog 或日志独立触发 |
| EFA 从未活跃、长期低流量 | EFA 规则不单独判 hung；本次未新增常驻框架通信 watchdog |
| 后置全任务采集能否弥补所有前置探测缺口 | 不能；这不是常驻全 rank 通信 watchdog |

## 自动验证

- `tests/execution/test_nccl_review_e2e.py`：本轮继承 rank、绑定注册、pg_status、
  独立原始文件、时钟未知、Fabric 范围未知、显式错误分流及两轮屏障的软件连续链。
- `tests/execution/test_nccl_review_boundaries.py`：128 节点跨批窗口、版本/水位校验、
  P2P 排除、shape/dtype、错误分类及官方分析器缺依赖的负向结果。
- `tests/execution/test_nccl_watcher_ownership.py`：真实日志入口不建诊断链，真实 watcher
  非零退出到 containment/终态重试计划、低流量中的 peer 失败、采集中途退出或清理竞态。
- `tests/node_agent/test_nccl_collection_budget.py`：真实本地子进程超时/输出量、
  阻塞读取槽、持久化频率限制及节点单采集锁。

- `tests/node_agent/test_process_diagnostics_e2e.py`：真实NodeAction处理器、三次JSON采集、
  非零退出码/坏格式、无rank的PID线索进入持久化workflow、无step的角色文件、共享/无效GPU。
- `tests/node_agent/test_stack_samples.py`：线程结构、后台输出顺序、NSpid映射、线程截断、
  错误文本及过宽关键词的负向检查。
- `tests/execution/test_process_diagnosis.py`：时长归一化、PP/EP/硬件/代码/窗口隔离、
  同角色弱线索、wchan=0、线程复用、缺CPU/GPU、FR完成推进不屏蔽I/O证据。

- `tests/execution/test_flight_temporal.py`：同序号完成落后、只提交不完成、
  正常完成推进、整组停滞、PID/成员/计数/历史变化、窗口不重叠与 1024 成员压缩摘要。
- `tests/node_agent/test_flight_temporal_collection.py`：真实本地 FIFO 两次触发，
  半写入文件重试、第二份陈旧、PID 复用，以及实际采集结果进入持久化工作流并选择 PID。

- `tests/execution/test_full_hung_survey.py`：1250 节点/10000 rank 软件规模、并发限制、
  单节点不可达、缺 rank、重复身份、时间差和 PG 成员覆盖。
- `tests/execution/test_hung_survey_pipeline.py`：真实 NodeAction handler、签名命令、
  128 节点/1024 进程全节点采集，无 step/FR、无 RANK 的两种千卡软件场景，
  两轮不重叠选点和有界诊断包。GPU 查询为模拟边界，并非真 GPU 测量。
- `tests/execution/test_nccl_step_independence.py`：旧 step 字段不跳过主动采集、
  首轮 FR 成员驱动扩采、无 rank 的 D-state 仅提供弱线索。
- `tests/node_agent/test_passive_hung_summary.py`：超过原 16-process 限制的节点、
  可选应用文件、跨 attempt/旧文件拒绝、PID 复用和坏 RDMA 计数。
- `tests/execution/test_hung_summary_codec.py`：无损恢复、损坏封装及解压上限；
  万 rank 规模测试同时覆盖每 rank 三个通信组，不能只测空进度摘要。
- `tests/completion/test_hung_detection_review.py`：缺失 step、恢复相同步数、phase 起点、
  失败收尾；PostgreSQL 三种模式另由 native Store contract 验证。

- `tests/host_health/test_hung_progress_hardening.py`：step 缺失/停滞/前进与不同 phase
  均不改变 RDMA 时钟、写入不延期、观测空洞、独立训练 monitor 停滞 latch。
- `tests/execution/test_hung_evidence_boundaries.py`：FR 不可用、严格多数、不同 PG、
  缺成员、重复 rank、无效数值、候选数不截成三个。
- `tests/execution/test_hung_workflow_hardening.py`：真实 API/Store/执行器软件链，
  Fabric 不替代应用恢复；1250 节点规划与并发合并；10000 节点采样边界。
- `tests/node_agent/test_hung_collection_hardening.py`：真实本地 FIFO 与 dump 文件，
  FR 未启用、同卡异任务、PID 复用、侵入性工具排除。
- 32 类软件注入运行器的第 011 类包含持续低流量、普通错误归属及退出竞态的连续软件链/集成用例，从原始RDMA/CRI
  输入到实际NodeAction、归档、Fabric验证及事件收尾；原组件回归仍保留。
  这不是真实网络故障或千卡压测证明，详见 [端到端覆盖审计](nccl-hung-e2e-coverage.md)。

## 真机验收补充

复用 [GF-GENERAL-LIVE-011](general-fault-live-acceptance.md#gf-general-live-011)。
以下均为规格，**NOT_RUN**；本次未操作 GPU、EFA、集群或线上训练。

| 场景 | 安全注入/操作 | 预期与证据 |
|---|---|---|
| 已知 rank 停滞 | 测试程序显式暂停点，至少 3 rank，独立超时释放 | 关闭 step reporter，实测 RDMA/进程/通信状态，不确认单快照根因 |
| checkpoint | 测试任务长 checkpoint 和 busy logging | 允许疑似告警；写入不抑制，不自动停训/重启 |
| FR 开/关 | 测试任务分别预配置/禁用 FR FIFO | 无临时猜造 dump；记录每 PG 成员和缺项 |
| FR 双快照 | 无 step reporter 的隔离任务，在受控暂停点停留并带超时释放 | 核对两次触发、文件更新、watermark 增量、同一操作持续未完成和实际窗口长度；解锁后完成推进 |
| FR 缺项/整组 | 隔离任务模拟不响应第二次 dump；另验整组等待 | 不把缺 dump 当停滞；全组未完成不虚构单 rank；测量双 dump I/O 和吞吐影响 |
| 继承 rank | 真实 DataLoader 多 worker，继承 RANK 及通信文件路径 | 训练 PID 保留；helper 不参与去重，PID/NSpid/starttime 均核对 |
| 跨批采样 | 128 节点、并发16，对部分 Agent 注入可控延迟和时钟误差 | 两轮屏障；记录实际交集和覆盖率，未知时钟不授权定向包 |
| 初始化停滞 | 专用任务初始化受控延迟；独立 watchdog/timeout 释放 | 从未活跃不凭低流量告警；显式超时先保现场，不能认定硬件根因 |
| 健康训练误触发 | 正常训练、长 checkpoint 分别触发只读取证，A/B 多次测量 | 记录 tokens/s、采集 CPU/RSS、工具峰值并发、P50/P99 延迟及字节数，不以 hung 状态开销代替 |
| 局部 TP/EP | 专用子组暂停但其他组维持流量，用独立通信状态确认持续停滞 | EFA 聚合规则可能漏报；普通 timeout 日志不单独建 hung 链，不宣称全任务 RAS 已覆盖 |
| 普通错误与退出 | 独立测试任务输出 NCCL/普通异常并分别继续运行、非零退出 | 只输出日志不建诊断链；真实非零退出由通用 watcher 清理和按预算恢复 |
| 取证退出竞态 | 疑似 hung 建立后，在基线采样前或复查后让测试任务退出 | 后续不再发送主动取证；已发命令单独留痕，不影响 watcher 接管 |
| 角色/线程 | 隔离混合并行任务提供或不提供角色声明；改变后台线程输出顺序 | PP/EP不同角色不互比；无声明只保留本地线索；核对容器/宿主机TID映射 |
| 取栈失败 | 专用测试工具返回非零退出码、错误文本或不完整JSON | 明确采集失败，无签名/无异常rank；其他进程和CPU/I/O证据保留 |
| 采样开销 | 128节点测试任务分别启停有界诊断 | 记录每节点py-spy次数/并发、proc文件读取数、压缩前后字节及外部吞吐影响，不外推软件测试结果 |
| PG 隔离 | DP/TP 各组不同序号；不同成员可复用本地名称 | 不跨组比较；无成员表不定向 attach |
| PID/共享 GPU | 隔离测试容器复用 PID、另任务共享测试 GPU | 不采无关进程；过期目标跳过并留痕 |
| 并发/不可达 | 多节点同窗口触发，隔离一个测试 Agent 管理通道 | 单 workflow、无重复大规模采集，缺失不当故障 rank |
| 千/万卡 | 对独立测试任务运行同场景，记录真实卡数 | 阶段集合、覆盖率、命令数、CPU/RSS/网络/证据体积、由测试工具外部测量的吞吐开销 |
| 恢复与新一轮 | 解除部分/全部 rank 暂停，再制造停滞 | 无 step 也能完成取证并转人工；不把 Fabric 正常当训练恢复 |

升级需交付匹配的控制面、Executor 和 Node Agent。`COLLECT_HUNG_SUMMARY` 是新增独立
operation，需 Profile 的既有 diagnosticBundleCapture 能力以及两端 allowlist，
旧 Agent 拒绝未知操作，不使用旧 triage 加参数降级。遵守 required/compatible
两阶段 pin；回滚前排空或明确终止含新 operation 的在途 workflow。
没有新增生产服务、控制面数据库表、IAM 或公网接口；EFA 已有 JSON 状态增加兼容字段，
Agent 增加私有 SQLite 历史缓存。heartbeat 可选 plan 在旧版缺省时不参与签名编码。
监控版本进入 Agent config digest，部署工具与 Agent 使用相同 payload 构造，
需要匹配版本并按既有双阶段 pin 滚动升级。

框架协同 NCCL 恢复尚未实现，不能由通用 Agent 安全替代框架的 communicator 重建、
一致 checkpoint/状态恢复；需要训练框架接入。千卡正常训练和误触发取证的真实吞吐
验收需要相应集群，本机软件测试不能给出影响百分比。两项均不标记为已完成。
旧 `GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_ENABLED/MAX_SECONDS` 继续接受配置以便
滚动升级，但已不参与新版 NCCL 判断；旧版在途 training-hung 流程应排空后升级，
不能将其含 step 验证的既有 plan 当作新版 no-step 流程。
