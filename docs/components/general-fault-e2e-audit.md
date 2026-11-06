# 通用故障逐项整改与端到端验收

## 结论与范围

核对对象是当前代码。范围是[总表](../故障类别与处置动作总表.md)第一部分的 **32 行**，
不是 32 个底层错误码，也不是所有组合故障的穷举。

**并非全部 32 类建议均已完成。** 第一批修复 Host/EFA 观测有效性、主机诊断绑定与
错误升级；第二批已实现原先 12 类“待改”项的主要策略变更，见
[12 类策略修改](health-policy-hardening.md)。另外 12 类“部分”事项仍有独立缺口。
“已有实现”不等于“本次修改”，“已有单测”不等于“端到端通过”。

本文件给每行一份独立端到端验收规格，统一注册为 `GF-GENERAL-E2E-001..032`。
它们是 **manual / NOT_RUN**，不是 32 个已经运行通过的自动化测试。
需要整改才能满足的目标另列“整改验收缺口”；验证现有行为通过，不能关闭该缺口。
案例和需求矩阵均保留这些缺口。历史区域案例的 PASS 不代表当前代码通过。

新增[32 类自动化软件注入入口](general-fault-injection-runner.md)。运行器区分连续
主场景和分段检查，不能把自动化 PASS 替代本文件仍未执行的完整真机验收。

真机细化规格见[32类通用故障真机验收补充](general-fault-live-acceptance.md)：
其余30类新增 `GF-GENERAL-LIVE-*`，018/023复用12项环境验收。
每类必测变体在 `testcases/general-live-coverage.yaml` 单独列出；
规格映射覆盖不等于现场已执行，全部仍为 NOT_RUN。

## 逐项清单

状态：“部分”指第一批确有相关修改但该类全部建议未完成；“保留”指已有机制未改；
“已改/待真机”指第二批主要策略已实现并有独立软件用例，完整真机验收仍未执行。
测试列描述证明边界，不宣称真机复验。

| 案例 | 故障类别 | 整改状态 | 已有证据与缺口 |
|---|---|---|---|
| 001 | CPU 过载 | 部分 | GF-HOST-003 只到计划；新增 GF-GENERAL-SW-001 执行真实验证；应用阶段关联仍缺 |
| 002 | 内存/交换耗尽 | 部分 | GF-HOST-OBS-001 验证 swap；GF-GENERAL-SW-001 扩展 memory；cgroup/任务归因仍缺 |
| 003 | 低利用率/资源持续异常 (`SITE_HOST_RESOURCE_HEALTH`) | 部分 | 持续窗口、attempt 归属与诊断绑定有本地测试；应用阶段与 MIG 归因仍缺 |
| 004 | 本地磁盘满 | 部分 | GF-HOST-001 只测策略；新增 GF-GENERAL-SW-002 执行隔离；inode 耗尽已按同一 98% 规则判定（`filesystem_inodes_used_percent`，SW-002 `inode` 变体执行真实隔离）；只读、挂载消失与 statvfs 其他错误（含 EIO）报 `filesystem_unavailable`（READ_ONLY / MOUNT_MISSING / PROBE_FAILED，只做 VALIDATE_HOST 诊断，不隔离、不 reset/reboot）；系统盘与任务盘区分仍缺 |
| 005 | 共享存储不可用/满 | 部分 | 有 statvfs 超时与策略测试；新诊断绑定；跨节点故障域记录已有控制面实现与软件测试（`SHARED_STORAGE_FAILURE_DOMAIN`）；真实读写与故障域真机验收仍缺 |
| 006 | 磁盘 IO 饱和 | 部分 | 有持续窗口与限时采集测试；新诊断绑定；与 checkpoint 关联仍缺 |
| 007 | SMART 失败 | 部分 | GF-HOST-004 原为 API 到计划；新增 SW-002/003 分别执行隔离与拒绝误隔离；scan 失败、盘无法打开、单盘超时为 UNKNOWN（`smart_query_failed` + collection error，SW-003 `smart-scan`），工具缺失/无盘/无结论为 NOT_SUPPORTED（`smart_status_unsupported`）；真机回放由 GF-GENERAL-LIVE-007 的宿主来源回放规定（`scripts/e2e/regional/general_live_source_replay.py`），仍为 NOT_RUN |
| 008 | 网卡链路 down | 部分 | 新增 SW-002/003 覆盖接口消失/down/unknown；接口角色、bond 等场景仍缺 |
| 009 | 网卡错误/丢包 | 部分 | 错误/丢包持续窗口有本地测试；新诊断绑定；`statistics` 计数器回退为 UNKNOWN（本轮不发该指标，不再夹成 0 清掉信号）；错误率与业务关联仍缺 |
| 010 | EFA/RDMA 计数错误 | 部分 | GF-HOST-002 只测策略；新诊断绑定；`ethtool -S` 差值只来自真实读数（失败/重置为 UNKNOWN，缺失计数器为 `efa_ethtool_counters_unsupported`，不再报 0）；sysfs 端口计数器回退同样为 UNKNOWN（本轮不发 `rdma_errors_delta`，collection error 点名端口与计数器，基线重建）；按驱动/计数器语义分级仍缺 |
| 011 | EFA 流量状态机 (`SITE_EFA_TRAFFIC`) | 部分 | GF-HOST-OBS-004 跑真实 main 但不执行 DAG；大规模挂起到证据闭环仍缺 |
| 012 | GPU 掉卡 | 部分 | COLLECT-003/004 和采集超时测试；NVML 失败且 `/proc/driver/nvidia/gpus` 不存在/不可读为 UNKNOWN（`INVENTORY_UNAVAILABLE`），不再生成 REBOOT_NODE（SW-003 `gpu-unknown`）；真实掉卡、配置错配仍须分开真机验收 |
| 013 | EFA 掉卡 | 部分 | OBS-003 防读取失败误重启；COLLECT-017 只覆盖部分真实分层恢复；PCI/link 全分支仍缺 |
| 014 | K8s allocatable 少报 | 保留 | COLLECT-017/DESTR-021 有 runner；MIG/time-slicing 与全部归因场景仍缺 |
| 015 | BMC 传感器告警 | 部分 | 新增 SW-002/003 覆盖标准四列和失败查询；告警传感器的名称/读数/阈值随样本标签进入 finding reason、incident、marker 与隔离链完成时的 `quarantined-node-notice-zh-v1` 通知，全 `na` 表为 UNKNOWN，无 `ipmitool`/无 IPMI 设备为 `bmc_status_unsupported`；具体传感器影响级别仍缺 |
| 016 | 日志：MCE / 存储只读 / RDMA 致命 / 内核 lockup | 已改/待真机 | FIX-016：可信内核来源与新鲜任务归属后先停任务再隔离；未归属不猜测停训对象 |
| 017 | 日志：OOM / NCCL / torch.distributed 错误 | 已改/待真机 | FIX-017：保留 PID/容器元数据，区分 LIVE/EXITED/UNKNOWN；取证后人工复核，不以 Fabric 健康宣告应用修复 |
| 018 | GPU 温度 | 已改/待真机 | FIX-018：连续三个有效严重样本后隔离/停任务/人工复核，绝对关机温度越界立即保护；空闲降温不自动接纳 |
| 019 | GPU ECC / 行重映射 / 页退役 | 已改/待真机 | FIX-019：DBE 停训分析，retirement 增量诊断，按架构决定 pending reset；未知架构不盲目 reset |
| 020 | NVLink 错误计数 | 已改/待真机 | FIX-020：counter 单独或多卡相关只诊断；明确致命 Xid/SXid 仍走原 catalog 恢复 |
| 021 | NVLink fabric 未训练 / Fabric Manager 失败 | 保留 | fabric 状态测试与 DESTR-010 可复用部分执行段；状态触发的全链与型号矩阵未全验 |
| 022 | PCIe replay | 已改/待真机 | FIX-022：reset、boot、身份、来源变化不形成有效 delta；保留同卡可信 Xid 联合规则 |
| 023 | 降频/热违规/功耗 | 已改/待真机 | FIX-023：duration只诊断；独立0x80 Power Brake持续确认后隔离复核供电，不以reboot处理 |
| 024 | DCGM 复合规则 (`SITE_DCGM_CORRELATION`, 45s 窗口) | 已改/待真机 | FIX-024：拒绝未来/跨来源组合；重复组件不推进连续次数；相关性不独立授权更强恢复 |
| 025 | DCGM 采集不完整 | 已改/待真机 | FIX-025：无效/重复/空样本为 UNKNOWN；有界新快照及成功观测验证，不升级硬件 |
| 026 | GPU 身份变化 / 期望数未知 | 保留 | COLLECT-020 有身份 runner，采集器有未知机型测试；两种原因须独立验收 |
| 027 | 训练进度 | 已改/待真机 | FIX-027：NaN 仅停归属任务并复核 checkpoint；阶段有界、reporter 失联与挂起区分，恢复要求原 rank 的有效进度 |
| 028 | 训练进程失败 / attempt 终态 | 保留 | GF-TRAIN 系列、COLLECT-016 部分执行段；非硬件失败原因和全部退出路径未全验 |
| 029 | 重启预算耗尽 | 保留 | preflight 执行链测试已有；原总表“所有 adapter 均不调用”与源码不符，已纠正 |
| 030 | 热备节点不健康 | 已改/待真机 | FIX-030：NotReady/旧遥测不单独触发 reboot；只有新鲜明确硬件证据可启动硬件恢复 |
| 031 | 采集器静默 | 已改/待真机 | FIX-031：统一 UNKNOWN 判据、每节点每小时最多一个快照刷新工作流；无任意服务重启权限 |
| 032 | 未知 XID / 未知 GPU 产品 / 缺证据 | 保留 | catalog 回放与安全门禁测试；未知遥测和已确认致命故障区别仍需策略评审 |

## 通用验收协议

每份案例都执行下列步骤，不能只提交预制 finding 或手工创建 workflow 代替检测。

1. 使用独立测试租户、节点和可丢弃训练任务；固定源码/制品、Profile、驱动、
   采集周期、阈值、节点 UID/boot ID、GPU UUID/BDF、job/attempt/rank 清单。
   明确允许动作与维护窗口；没有授权只做本地 fixture 回放。不得对业务集群试注入。
2. 保留健康基线，检查预置异常或历史 marker 不会污染本案例。
   下列涉及原始文件/命令的回放，必须进入真实 parser/collector，而非直接构造判定。
   真实硬件错误不可安全制造时使用原始格式 fixture，并把物理验证记为 NOT_RUN。
3. 注入后跟踪 `batch/event -> finding -> marker/incident -> workflow -> step executions
   -> adapter/provider receipt -> final state/notification`。记录 ID、时间、作用域、
   每步结果和错误。异步入口的 202、计划列表和命令返回 0 都不是整条 PASS。
4. 要求实际执行正确适配器；软件模式仅替换 OS、设备、Kubernetes、云 API I/O，
   不替换策略、工作流执行器或被验收的验证函数为固定成功返回。
   需要真实硬件才能确认的恢复结果单列，不从软件结果推断。
5. 对每个子变体使用干净状态；覆盖阈值前后、恢复、重复/陈旧投递、缺字段/权限失败。
   固定检测期限为采样及确认窗口加有界传输预算，执行期限采用被测 step 的超时。
   超时必须留下明确失败/等待证据，不能略过该变体后宣称整类通过。
6. 检查没有越界节点、无授权 reset/reboot/replace、重复停任务或重复消耗预算。
   读诊断类必须证明任务 UID/attempt 不变、没有隔离副作用，而不只是“没看到重启日志”。
7. 清理注入与独立测试资源；恢复配置、功耗、服务与挂载；重新采集并运行必要的 GPU/
   网络/存储检查。解除隔离须走验证后的恢复流程，不直接删除 taint 作为“修复”。
   动作结果不明时保留隔离和证据。清理失败即本案例失败。

每次输出：`case_id`、`variant`、`proof_scope`（software/live）、源码和配置摘要、
输入/输出证据、覆盖节点/rank 分母与实采数、所有步骤回执、期望/实测结果、
缺失证据、清理结果。不可把观察到的一张卡或一个分支推广到整类。

## 分项案例

### GF-GENERAL-E2E-001

CPU 过载。采集输入使用连续 `/proc/stat` 和 `/proc/loadavg` 快照。
分别令 CPU busy 达 99%、load/CPU 达 2.1；保持 119 秒时不应有该 finding，
达到 120 秒后应产生诊断并实际执行 VALIDATE_HOST。健康基线、零 total delta、
一次短突发、恢复后重发分别运行。原指标仍异常时不能用其他健康指标通过验证，
不能升级为硬件 reset/reboot；恢复后验证成功。清理 CPU 负载并确认任务仍运行。
整改验收缺口：训练阶段与 cgroup CPU 压力归因。软件实现见 SW-001。

### GF-GENERAL-E2E-002

内存/交换耗尽。原始 meminfo 分别令 memory 使用率 99%、swap 使用率 90%。
memory 需持续 120 秒；swap 当前不使用该持续窗口；同时记录独立的低可用内存
60 秒规则，不能误算重复事件。依次执行仍异常/恢复的 VALIDATE_HOST，验证
swap 指标没有被丢弃。另测无 swap、短 checkpoint 突发和不同 attempt。
清理分配并确认没有 OOM 杀死非测试进程。整改缺口：任务/cgroup 和阶段关联。
软件实现见 SW-001 与 GF-HOST-OBS-001。

### GF-GENERAL-E2E-003

资源持续异常。为唯一受管 attempt 注册每 rank 的 GPU UUID/PID，分别回放
GPU/rank CPU ≤5% 持续 300 秒、可用内存 ≤5% 持续 60 秒、page cache ≥90%
持续 300 秒、本地盘 ≥90% 持续 60 秒。执行对应 Host/GPU 诊断。
对 IDLE、两个 attempt、未分配 GPU、换 attempt、持续应用进度各跑负向，
不能借其他 GPU 的健康样本通过原设备的诊断。恢复并检查窗口重置。
整改缺口：checkpoint/初始化显式阶段、MIG/shared GPU 归属与全链验证。

### GF-GENERAL-E2E-004

本地盘满。对测试专用挂载的 statvfs 回放 99% 使用率，经采集到实际 Kubernetes
隔离 patch，检查 cordon、NoSchedule taint、incident QUARANTINED；当前契约不停止
现有任务。97% 不触发该临界规则；重复投递不能重复建链，provider taint 必须保留。
inode 耗尽按同一 98% 规则作为该挂载的独立信号判定，字节剩余不能掩盖 inode 耗尽
（SW-002 `inode` 变体）。只读、挂载消失或 statvfs 其他错误报 `filesystem_unavailable`
（READ_ONLY / MOUNT_MISSING / PROBE_FAILED），只执行 VALIDATE_HOST 诊断，挂载恢复且信号清除后
验证通过，不隔离、不 reset/reboot。清理测试盘并验证后解隔离。
整改缺口：节点系统盘与任务 scratch 区分。软件实现见 SW-002。

### GF-GENERAL-E2E-005

共享存储。用独立 NFS/Lustre 测试挂载分别制造 statvfs 超时、98% 容量和正常响应，
验证采集有界且其他指标仍送达，随后实际执行目标挂载的 VALIDATE_HOST。
另一正常挂载不能覆盖故障挂载；缺失新样本应等待，不得确认恢复。
增加“statvfs 正常但读写/fsync 失败”与“挂载消失”变体作为整改验收：产品对每个共享挂载在
statvfs 之外做一次有界只读目录项读取（`IO_ERROR` 带 errno、超时 `IO_TIMEOUT` 按缺证据记不可用），并记住
节点级共享挂载、消失后报 `MOUNT_MISSING`（记忆跨 collector 重启）；该读取只读、不在客户文件系统上
创建、写入或 fsync 文件，因此读路径仍正常、只有写/fsync 失败的 export 不在其检测范围内；
真机证明仍待独立测试 export。
恢复服务器/挂载，读写探针和相关任务进度均通过后结束。
整改缺口：读路径与挂载消失已由产品关闭——有界只读 I/O 探针与消失挂载记忆由采集器实现（statvfs 之外的有界只读目录项读取、
节点级共享挂载消失的跨重启记忆，样本带 `source`/`fstype` 标签供聚合），只写/fsync 失败的检测仍缺；共享存储故障域聚合已由控制面实现（≥2 节点在 600 s 内报同一
export 不可用时另记一条集群级 `SHARED_STORAGE_FAILURE_DOMAIN` incident 并只发一封通知，节点链不变；
依赖采集器样本标签 `source`/`server`+`export` 给出 export 标识，见
[真机验收 GF-GENERAL-LIVE-005](general-fault-live-acceptance.md#gf-general-live-005)），真机执行仍待证明。

### GF-GENERAL-E2E-006

磁盘 IO。回放 diskstats 基线与增量，让 util ≥98% 或 await ≥100 ms 分别持续
119/120 秒，检查仅后者触发。真正执行验证并区分仍饱和/恢复；另测无完成 IO、
计数器回退、磁盘热插拔、短 checkpoint flush。证据必须绑定设备与有效采样区间。
不得以单次高 util 推导坏盘或 RMA。终止测试 I/O 并确认业务挂载未变化。
整改缺口：按设备能力/请求量归一化和 checkpoint 关联。

### GF-GENERAL-E2E-007

SMART。经真实 JSON parser 回放 `smart_status.passed=false`，执行隔离并保留证据；
再分别回放查询非零退出、损坏 JSON、`--scan-open` 失败、健康磁盘。
查询失败不能生成 `smart_health_failed`；应可重试并表明 UNKNOWN，而不是健康零。
`--scan-open` 失败（非零退出、超时或 smartctl 退出 0 前打印的 `# scan_smart_devices:` 行）
为 `SCAN_FAILED`，扫描无法打开的盘为 `OPEN_FAILED`，单盘 `-H` 挂起或非布尔结论为 `QUERY_FAILED`，
均为 `smart_query_failed` 加 collection error（SW-003 `smart-scan`）；scan 失败须单独运行，
不让 per-device 测试替它背书。
清理命令 fixture，不实际损坏介质。真实设备坏盘/RMA 结果不在软件证明范围。
软件实现见 SW-002/003，整改缺口为分设备影响级别。

### GF-GENERAL-E2E-008

必需网卡。使用完整临时 sysfs 树，分别令 ens5 消失、operstate=down、正常 up、
unknown、目录读取失败、down 但统计字段损坏。确认前两者经过 API 到隔离 patch；
读取失败不伪造成 down，已确认 down 不被统计读取错误吞掉。保留 provider taint。
非必需接口变化不隔离节点。只在独立维护节点进行真实链路实验，禁止关闭管理链路。
清理后验证接口、RDMA 和任务连通性。整改缺口：bond/VF/容器接口角色。见 SW-002/003。

### GF-GENERAL-E2E-009

网卡错误/丢包。真实计数器 parser 读取基线、errors 增量和 drops 增量；
errors ≥1 直接诊断，drops 需持续 60 秒。执行 VALIDATE_FABRIC，检查目标接口与
原错误增量；另跑仅 TCP 重传、低流量分母、counter reset、另一接口健康变体。
故障持续不应自动推导节点硬件坏；清零/恢复须来自新观察而非重新初始化 Store。
整改缺口：错误率、接口角色与训练通信关联。清理注入并验证网络。

### GF-GENERAL-E2E-010

EFA/RDMA 错误。分别回放 RDMA errors、link down、RNR、retry、CQ 增量，经真实
collector 到 Fabric 验证。每个计数器独立建立基线、触发、保持和恢复；缺读数、
计数器回退、非 EFA RDMA 设备不能被混当同一证据。检查没有直接硬件重启。
RNR/retry 并非天然硬件损坏，当前“任一增量”规则保留为现状而非推荐最优策略。
整改缺口：驱动版本语义、错误率和拥塞/资源耗尽分级。恢复链路后跑通信检查。

### GF-GENERAL-E2E-011

EFA 流量/NCCL hung。唯一受管 attempt 先超过 active 门槛且过 startup grace，
再让完整 RX/TX 对维持 ≤zero 门槛；验证 warning/hung 默认 60/600 秒窗口。
必须执行 COLLECT_HUNG_TRIAGE、诊断包及 Fabric 分支，不以计划生成判 PASS。
记录任务总节点/rank 数、选中数、实际完成数、缺失原因；分别测 Flight Recorder
启用/未启用、同时多节点触发、节点不可达、候选 ≤3 和候选过多。
从源码/参数记录 triage cap，不假定全量采集；Fabric scope 与 bundle scope 分开断言。
补充一直低流量、checkpoint 有进度、坏 RX、counter reset、换 attempt 负向。
不得把未知读数当零；整个诊断不 cordon/停任务/reset。恢复通信并核对证据清理。
整改缺口：规模化覆盖、应用阶段、完整执行与真实挂起因果；OBS-004 仅规划证明。

### GF-GENERAL-E2E-012

GPU 掉卡。固定节点真实期望 GPU 数，从完整 UUID 查询 8 卡基线变成 7 卡，
第一有效样本不触发，第二个才进入重启链；实际检查 stop、provider reboot 回执、
新 boot ID、完整 GPU/Host/Fabric 验证与恢复任务。超时、查询失败、错配期望数、
瞬时恢复必须分别跑，不能把“命令没有返回”当真实丢卡证明。
可复用 COLLECT-003/004 的部分夹具，不能用清单只读检查代表 reboot 执行。
查询失败已分类：NVML 失败且 `/proc/driver/nvidia/gpus` 缺失或不可读为 `INVENTORY_UNAVAILABLE`，
不计数、不推进连续确认；配置的期望数与机型自身 GPU 数及驱动绑定数矛盾时只记配置 finding
（`gpu_expected_count_contradicted`，取证并人工复核，不 reboot）。
整改缺口：其余场景更充分的失卡交叉证据。清理期望数与测试任务。

### GF-GENERAL-E2E-013

EFA 掉卡。分别运行 PCI missing、driver unbound、link inactive、excess、unknown、
vendor/driver/port 不可读；只有连续有效读数推进确认。逐项核对 reboot、
定向 bind+plugin、先诊断的不同链。UNKNOWN/excess 验证失败不得授权 reboot。
bind 失败与恢复后验证失败分别验收，确认不执行全局卸载 efa。
COLLECT-017/DESTR-021 只复用适用分支；PCI 真丢失不能由 unbind 测试替代。
清理必须有独立 rebind 保底和原 allocatable 校验。整改缺口：所有模式的执行闭环。

### GF-GENERAL-E2E-014

K8s allocatable。GPU/EFA 分别在物理 inventory 正常时让 Node allocatable 少报，
确认两次后只重启目标 plugin、触发新快照、等待正确 expected_count；Pod UID
变化和重新注册必须有证据，不能只检查 delete 返回 0。训练子场景保留原
Pod/attempt，通信进度在故障和恢复阶段都推进。失败才按授权升级。
另测未知期望数、配置错误、MIG/time-slicing 资源名，后两者不得直接认定物理掉卡。
清理 plugin/annotation，检查 allocatable、节点调度和原训练。
整改缺口：共享 GPU 语义。可复用 COLLECT-017/DESTR-021。

### GF-GENERAL-E2E-015

BMC。标准 `id | value | units | status | ...` 回放 cr/nr/ok，检查第四列判定，
告警执行真实隔离 patch；units 值本身不能触发。命令失败、空输出、三列损坏行、
离散/不可用传感器分别记录观测结果，不以空样本宣布健康。多传感器告警检查去重。
不通过堵风扇或过热制造故障；使用 fixture 或厂商支持的测试接口。
整改缺口：传感器类型/冗余电源/风扇分级。软件实现见 SW-002/003。

### GF-GENERAL-E2E-016

严重主机日志。分别经 NodeLogCollector 回放可信 journal 的 MCE/EDAC uncorrectable、
文件系统只读/Buffer I/O、RDMA fatal、soft/hard lockup；检查分类、作用域、隔离
实际执行与通知。相同文本来自训练 stdout、corrected MCE、重复 journal cursor
分别做负向，不可据不可信文本授权硬件动作。
整改已实现：可信内核日志加新鲜受管 attempt 归属时先 STOP 再 QUARANTINE；
归属未知时只隔离并等待人工，不能猜测任务。保留原日志/cursor，清理后验证节点与任务状态。

### GF-GENERAL-E2E-017

应用错误日志。分别回放容器 OOM、NCCL timeout、torch.distributed error，匹配
job/attempt/PID；实际执行 Host/Fabric 诊断，不能把错误打印所在 rank 当必然根因。
错误已退出与进程仍存活两类独立运行；旧 attempt、无所属 job、重复日志为负向。
整改已增加 journal 结构化身份与 observer 容器归因、存活/退出区别及有界取证；
应用日志工作流保留人工复核，不自动归咎 GPU 硬件。证据缺口仍须逐次记录，
没有 Flight Recorder 或进程已退出时不能声称通信组/调用栈已完整采到。

### GF-GENERAL-E2E-018

GPU/显存温度。分别从 DCGM 文本与 nvidia-smi XML 回放绝对温标和带正负号
T.Limit、warning/critical 边界、缺失限值和 N/A；同卡只采用匹配温标，不能重复
生成 fallback 告警。warning 执行诊断，critical 需三个有效样本确认，验证隔离、
stop与人工复核；冷却后的空闲样本不授权自动恢复调度与任务。
整改已取消热故障自动 RMA/硬件升级，超时保留隔离待人工。不得真实加热损坏 GPU；
物理冷却验收须另有维护方案，清理所有限值 fixture。

### GF-GENERAL-E2E-019

ECC/行重映射/页退役。按受支持架构分别回放 SBE、volatile DBE、aggregate DBE、
新增 retired DBE、pending、row_remap_failure，验证 counter baseline/reset、
一次与重复采样。实际执行对应诊断/reset/DRAIN，核对 Profile OBSERVE 时拒绝
未授权 field diagnostic。reset 后必须重新查询 pending 和驱动 recovery action，
不能以 reset 退出 0 代替恢复。整改已按支持架构区分 pending reset，DBE 停训分析、
普通退役增量诊断；未知架构复核，不因计数器增量自动 RMA。
硬件错误用厂商支持的实验手段或 fixture；不宣称制造并修复真实坏显存。

### GF-GENERAL-E2E-020

NVLink counter。分别对 CRC、replay/recovery 和明确不可恢复错误建立基线再增量，
验证计数器单独触发诊断，counter reset/陈旧批次不重复恢复。整改已取消 counter
单增量和多卡 counter 相关直接 DRAIN，记录速率与类型；明确致命 Xid/SXid 的
独立 catalog 恢复不受影响。多卡同时出错检查无重复链和误扩域。
清理后要求链路、Fabric 及通信验证，不只看总计数器非零。

### GF-GENERAL-E2E-021

Fabric 未训练。受支持 Hopper SXM/HGX Blackwell 节点回放状态 1/2/4 持续
299/300 秒，后者应生成一次节点级 finding，执行完整 GPU/NVSwitch reset、
服务恢复、每卡 Success 和 Fabric 验证；不是仅 restart FM。
未受分级型号、稳定状态、collector 重启重新计时和本系统 reset 窗口为负向。
实际 reset 超时不能重复盲发。DESTR-010 的 Xid 45 触发不替代这里状态触发的检测。
缺口：该触发源到完整执行的型号矩阵验收。清理后验证完整 inventory 和训练。

### GF-GENERAL-E2E-022

PCIe replay。counter 在可比较时间内分别形成恰为规则速率与超过规则速率（batch 233 起规则速率为该卡当前链路的
DCGM 预期速率，Gen5 x16 为 31/min；exporter 不提供链路时为 8/min），前者不越界、后者诊断。再与同一 GPU/BDF、可信时间窗的 Xid 32/79 组合，验证
联合 DRAIN；另一 GPU、过期 Xid、缺 identity 不得误关联。
实际执行诊断/隔离，记录 PCIe link speed/width、AER 和拓扑证据。
整改已拒绝 reset、身份/boot/来源变化形成 delta；流量/链路条件和硬件噪声水平
仍须通过物理验收评估，不宣称固定阈值适合每种平台。
清理输入后验证新采样恢复，不要求累计 counter 回零。

### GF-GENERAL-E2E-023

降频/功耗。原始 DCGM 数据分别包含 thermal bits、正常 power cap、微小 violation
增量、达到配置占空比的持续增量、counter reset 与超出墙钟的非法增量。
记录 collector 过滤阈值与控制面升级阈值，不能把二者混为“增量 1us 就故障”。
实际执行诊断；正常功率限制不能被热故障规则误判，低时钟但无热证据不归咎过热。
若采用真实降 power limit 的实验，先布还原保底，最后逐卡恢复默认值并验证。
默认控制面要求至少 10000us 且占空比至少 0.1%，非法超墙钟增量不判定；
整改已分离临时热故障/RMA。COLLECT-002 只证明投递，性能影响仍需真机验收。

### GF-GENERAL-E2E-024

DCGM 复合。分别构造 memory、correctable memory、单/多 GPU NVLink、thermal、
power 的有效组合，记录 45 秒窗口；correctable 连续三次等条件按当前配置执行。
实际核对只生成必要工作流，component findings 保留而不重复派发。
换 GPU、过窗、乱序、collector 重启、计数器回退和弱相关同时出现为负向。
整改已拒绝未来和跨 metric source 证据，重复组件不累加连续次数；
correctable-only 只诊断，memory 组合保留有依据的组件动作及现场诊断过程。
恢复各组件读数，确认 composite 清除和重复故障重新触发。

### GF-GENERAL-E2E-025

DCGM 不完整。经真实 exporter parser 注入缺必需字段、抓取超时、部分 GPU 消失；
检查错误 finding/观测状态与仍有效的数据同时保留。随后干净完整批次清除错误，
连续失败不得提前恢复；GPU 与温度查询各自故障不能抹掉其他有效数据。
实际执行诊断，不能将不可观测等同 GPU 健康，也不能无硬件证据升级 reset。
整改已统一 UNKNOWN 判据；只请求有界新快照并等待成功观测，不将监控故障升级
为 GPU 故障。真正服务重启仍由既有 systemd 策略或人工处理。
清理 exporter fixture，确认所有期望 GPU 的新样本到达。

### GF-GENERAL-E2E-026

GPU identity/未知期望数。同节点同数量下更换 UUID/BDF 与未知实例类型分开运行：
前者诊断，后者证据收集和人工复核；缺期望数不得偷偷采用 1 或 8 作为可靠值。
实际检查 inventory、incident、通知和无未经授权 reset；旧 boot/陈旧 identity
不能覆盖新库存。合法维护后的换卡要有批准与新 baseline，不能永久重报旧身份。
缺口：完整 identity 变更及未知数量分支验收。可复用 COLLECT-020 的部分流程。

### GF-GENERAL-E2E-027

训练进度。独立作业 reporter 分别产生 nonfinite loss、心跳过期、心跳新鲜但 step
停滞、step 回退、straggler；每个变体从 reporter 输入到实际工作流和证据。
checkpoint、恢复旧 checkpoint、初始化阶段、旧 attempt 和 reporter 自身失联为负向。
整改已按应用信号分流：NaN 停归属任务并人工验证 checkpoint；有效阶段有界抑制；
reporter 失联不等同 NCCL hung，挂起恢复需要原 attempt/rank 的新有效进度。
GFRS 不自动验证 checkpoint 内容或选择 checkpoint，不能把引用存在当恢复授权。
清理后确认 loss 有限且各 rank 进度恢复，不能只验 Pod Ready。

### GF-GENERAL-E2E-028

进程失败/终态。让可丢弃作业的关键容器非零退出，由真实 watcher 观察，而非直接写
FAILED workflow；核对 STOP、日志证据、terminal event、marker 复用/无 marker
预算内重启。用户停止、本系统停止、非关键容器退出、无 allocation 各做负向。
重放旧 attempt 终态不能停止或重启新 attempt。检查重建 UID、训练有效进度和
预算消耗一次。COLLECT-016 的 Xid 触发不替代所有自然退出变体。
整改缺口：应用/硬件失败原因分流；清理测试 Job 与 orphan Pod。

### GF-GENERAL-E2E-029

预算耗尽。先合法消耗 job 的预算，再触发包含 stop/reset/validation/restart 的
故障链。**现源码应继续执行恢复链，只扣住 RESTART_WORKLOAD**，并记录预算失败
和一次通知；不是所有 adapter 零调用。只含 restart 的链则无任务创建。
并发 claim、Store 重启、重复投递和缺 restart context 分开运行，预算不能变成
负数、重复扣减或虚假释放；缺必要 context 仍须预检拒绝。
检查最终 ESCALATED/QUARANTINED 与实际隔离一致。清理测试作业而非增加业务预算。
验收缺口：已有执行器级回归不等于真实故障输入到恢复结束的完整证明；本项未改生产逻辑。

### GF-GENERAL-E2E-030

热备不健康。独立 warm spare 依次报告 Ready=False、硬件 marker、配置错误、
陈旧遥测、保留中的 spare。记录当前 reason 分类与连续两次的 reboot 规则，
验证真实 provider 回执、复查、仍失败 UNAVAILABLE、1h 复查和 24h 告警去重。
整改已将 Ready/provider/HMA/普通告警和缺测从自动硬件依据中分离；
只有新鲜明确的硬件 finding/可信可执行 marker 才能授权 reboot。
不能触及已分配给其他任务的 spare。
清理保留和故障注入，仅验证健康后恢复 AVAILABLE。

### GF-GENERAL-E2E-031

采集静默。保持 Agent lease 有效，只停止一个必需采集通道成功上报，按 inventory
180s、metrics/host 420s、logs 900s 及实际扫描周期检查 advisory 和通知去重。
实际通知发送失败应有有界重试，不重复建立恢复链；disabled/masked、Agent 已过期、
非活跃集群各做负向。重新上报后退出静默，再次静默可重新告警。
整改已增加统一 UNKNOWN 观测样本和每节点每小时最多一个快照刷新工作流，
不授权任意重启采集服务。刷新失败维持 UNKNOWN，不清除已有故障。
清理后恢复采集服务，确认新成功记录而不是单看进程 running。

### GF-GENERAL-E2E-032

未知错误/缺证据。经 kernel collector 分别回放未知 Xid、未知 GPU 型号、
要求的 NVLink 版本/拓扑证据缺失；核对 BLOCKED/SITE_SAFETY 和获 Profile 授权的
隔离实际执行，不能在未知恢复语义下 reset/reboot。没有隔离授权也不能绕过 Profile。
另测普通已知 Xid 的旧驱动，Xid154 linkage 版本不是该 Xid 的最低执行版本；
该变体应保持原动作，不能因缺 linkage 误阻塞。
整改缺口：信息不全和已确认致命故障进一步分流；清理后须人工确认/验证恢复。

## 本地运行与限制

新增软件用例：

- `GF-GENERAL-SW-001`：原始 procfs -> collector -> ASGI API -> SQLite ->
  工作流执行器 -> 真实 GpuValidationAdapter，CPU/memory/swap 各两种结果。
- `GF-GENERAL-SW-002`：filesystem/inode 耗尽/SMART/BMC/net missing/net down -> 同链 ->
  真实 KubernetesWorkflowAdapter。仅 Kubernetes 客户端是本地替身，断言实际 patch。
- `GF-GENERAL-SW-003`：SMART 查询失败/SMART 扫描失败/BMC 查询失败/net unknown/
  GPU 库存双读失败经 API 后不产生隔离或 REBOOT_NODE。

这 17 个参数变体使用临时 OS 数据、命令 runner 与 Kubernetes 客户端，不访问真实
GPU/NIC/Kubernetes/AWS。常规健康 Host 验证基线由 fixture 补齐；未宣称覆盖其他
collector contributor。持续窗口在本地 API 模式使用显式虚拟 `received_at` 接收时钟，
不通过缩短生产阈值或真实等待来加速。没有把 GPU reset、NCCL hung 或其他 32 类
都标成本地已通过。

```bash
.venv/bin/python -m pytest \
  tests/host_health/test_general_fault_observation_e2e.py \
  tests/test_general_fault_coverage.py -q -o addopts=''
```

`tests/test_general_fault_coverage.py` 只检查 32 行与案例/需求的逐项对应、
完整验收规格及诚实的状态，不是故障执行证据。真正运行手工案例需要单独授权的
隔离环境与每次完整回执；本次没有真机执行。

第二批新增软件执行证明、边界和运行入口见
[12 类策略修改与测试](health-policy-hardening.md)。FIX 类本地测试不能替代
`GF-GENERAL-E2E-*` 完整手工验收，也不关闭其他“部分”类别的剩余建议。
