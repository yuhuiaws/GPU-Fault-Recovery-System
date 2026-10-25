# 32 类通用故障真机验收补充

## 覆盖结论

现有覆盖不完整：`GF-GENERAL-E2E-001..032` 是概要规格，软件用例中仍有分段检查；
旧区域测试常只证明采集、投递或一个执行段，不能据此认定当前版本的完整真机闭环通过。
本轮为温度/功耗以外的 **30 类**增加 `GF-GENERAL-LIVE-*` 详细真机规格，
018/023 复用已有 [GPU 环境验收](gpu-environment-live-acceptance.md)，不重复计算覆盖。

机器可读的完整映射和必测变体是
[`testcases/general-live-coverage.yaml`](../../testcases/general-live-coverage.yaml)。
当前列出260个按类别划分的必测变体；跨类别复用不代表新增独立物理实验。
**规格映射覆盖 32/32，实际真机执行覆盖仍未证明。** 本轮没有执行现场实验；
新用例均为 `manual / NOT_RUN`。已有产品缺口没有因补测试而自动修复。

| 类别编号 | 目标 | 真机规格 |
|---|---|---|
| 001 | CPU 过载 | GF-GENERAL-LIVE-001 |
| 002 | 内存/交换耗尽 | GF-GENERAL-LIVE-002 |
| 003 | 低利用率/资源持续异常 | GF-GENERAL-LIVE-003 |
| 004 | 本地磁盘满 | GF-GENERAL-LIVE-004 |
| 005 | 共享存储不可用/满 | GF-GENERAL-LIVE-005 |
| 006 | 磁盘 I/O 饱和 | GF-GENERAL-LIVE-006 |
| 007 | SMART 失败 | GF-GENERAL-LIVE-007 |
| 008 | 网卡链路 down | GF-GENERAL-LIVE-008 |
| 009 | 网卡错误/丢包 | GF-GENERAL-LIVE-009 |
| 010 | EFA/RDMA 错误 | GF-GENERAL-LIVE-010 |
| 011 | EFA 流量/NCCL hung | GF-GENERAL-LIVE-011 |
| 012 | GPU 掉卡 | GF-GENERAL-LIVE-012 |
| 013 | EFA 掉卡 | GF-GENERAL-LIVE-013 |
| 014 | K8s allocatable 少报 | GF-GENERAL-LIVE-014 |
| 015 | BMC 告警 | GF-GENERAL-LIVE-015 |
| 016 | 严重内核日志 | GF-GENERAL-LIVE-016 |
| 017 | 应用错误日志 | GF-GENERAL-LIVE-017 |
| 018 | GPU/显存温度 | GF-ENV-LIVE-001/003/004/005/008/009/010/011/012 |
| 019 | ECC/行重映射/页退役 | GF-GENERAL-LIVE-019 |
| 020 | NVLink 错误计数 | GF-GENERAL-LIVE-020 |
| 021 | NVLink fabric 未训练 | GF-GENERAL-LIVE-021 |
| 022 | PCIe replay | GF-GENERAL-LIVE-022 |
| 023 | 降频/热违规/功耗 | GF-ENV-LIVE-001/002/005/006/007/008/009/010/012 |
| 024 | DCGM 复合 | GF-GENERAL-LIVE-024 |
| 025 | DCGM 采集不完整 | GF-GENERAL-LIVE-025 |
| 026 | GPU 身份/期望数 | GF-GENERAL-LIVE-026 |
| 027 | 训练进度 | GF-GENERAL-LIVE-027 |
| 028 | 进程失败/attempt 终态 | GF-GENERAL-LIVE-028 |
| 029 | 重启预算耗尽 | GF-GENERAL-LIVE-029 |
| 030 | 热备不健康 | GF-GENERAL-LIVE-030 |
| 031 | 采集器静默 | GF-GENERAL-LIVE-031 |
| 032 | 未知错误/缺证据 | GF-GENERAL-LIVE-032 |

## 证明层次与夹具就绪

- `native`：真实系统/应用状态变化，经原安装的 collector/watcher 和真实控制面执行。
  CPU 负载、专用文件系统容量、测试容器退出等可采用此方式。
- `live-replay`：只替换指定原始文件/命令输出/驱动字段的读取结果，collector 的解析、
  传输、判定、数据库、调度和批准的节点动作均真实。只证明处理链，不证明物理故障。
- `vendor-lab`：自然故障，或厂商批准且有带外保护的实验室注入，验证物理检测和修复。
  无此条件保持 NOT_RUN，不能用其他两层的结果顶替。

**没有通用、现成的任意 Host/sysfs/journal/DCGM 故障注入器。** 当前
`tools/gpu_environment_replay.py` 仅允许环境字段，不能用于 ECC、NVLink、PCIe、
inventory 或 Host 指标。本文写到的其他原始源夹具是实施要求；没有经审查的对应
夹具时，该变体 BLOCKED/NOT_RUN，而不是直接 POST 预制 finding/workflow 代替。
不得写入只读 `/proc`、sysfs counter，或声称 `tc netem` 必然增加 NIC 硬件错误计数。
文件/命令回放应隔离在测试 producer 的私有环境，不替换宿主机系统二进制或全局挂载。

旧 runner 只能复用已覆盖的环节，例如 COLLECT-017 的 plugin/rebind、
COLLECT-020 的身份读取，以及 DESTR-001/002 的 reset/reboot 回执；
必须把本次原始输入产生的 incident 接到它的实际工作流，不能拿另一次 Xid 链来拼接。
缺夹具、缺设备能力、缺授权属于未执行；注入有效但判定/处置不正确属于 **FAIL**。

## 通用执行协议

1. 独立测试集群/节点池、可丢弃任务、独立数据卷/存储 export。保存 node UID、
   provider ID、boot ID、GPU UUID/BDF、job/attempt/Pod UID、采集模式、制品/源码/
   配置摘要及 Runtime Profile。HyperPod 试验须确认不会与自动 NodeRecovery 竞争。
2. 按最危险的实际分支审批权限和维护窗口；reset、reboot、热备替换分别审批。
   先验证带外管理、原配置备份、限时停止和独立还原保底，再启动故障。不得关闭管理
   链路、填满系统盘、全局卸载 EFA 驱动，或对业务进程施加停止/内存压力。
3. 为每个变体创建干净的测试状态，保留至少三个正常采样和原任务有效进度。
   禁止重用失败现场而不记录残留 incident/marker/隔离；未知配置不算健康基线。
4. 正向与反向均跟踪：
   `原始源 -> collector/watcher -> accepted batch/event -> finding/marker ->
   incident -> workflow -> step executions -> 真实适配器回执 -> 最终状态/通知`。
   没有 mutation 的诊断链也要证明实际执行了验证器，不能只看计划。
5. 阈值、确认窗口和采集/扫描周期以运行配置和代码为准；文中默认值只用来复核。
   记录预计/实测检测时间、每步 deadline、聚合/队列延迟。不得伪造 live 时钟或将
   提前缩短生产阈值的实验当作默认策略验证。
6. 同时检查无错误作用域、无未授权硬件动作、无重复 STOP/预算扣减。控制面仍处于
   RUNNING/WAITING、只返回 202、只看到命令 exit 0，均不代表完整 PASS。
7. 清理先结束注入并恢复采集/驱动/接口/原配置，再验证新鲜健康证据。只读诊断类
   要证明原任务继续推进；隔离类保留归属并经受控验证/人工流程接纳，不直接删 taint。
   reboot 必须有真实 provider 回执和新 boot ID；热备必须有 reservation/rebinding。
8. 每变体保存 `case_id/variant/proof_mode/source_identity/config_identity`、
   目标与对照清单、原始输入/变换记录、采集覆盖分母与实采数、上述链路 ID/回执、
   预期/实际结果、清理结果、差异和缺口。清理失败整项 FAIL。
9. 每项都执行适用的短暂/阈值前后、持续、恢复、重复/旧数据、缺测/权限错误、
   作用域和处置失败变体。映射表列出的必测变体不得省略；未执行必须留在分母。
   规格存在率、可执行夹具就绪率、实际执行率、通过率分别报告，不合并为一个100%。

## 分项真机规格

### GF-GENERAL-LIVE-001

必测：`cpu_busy`、`load_average`、`short_burst`、`recovery`、`invalid_delta`、`checkpoint_phase`。
使用专属节点的限时 CPU 负载，保留系统/管理资源；实际 `/proc/stat` 与 loadavg
必须越过运行门槛，达不到则不能靠 cgroup 内百分比冒充主机98%。小机型无安全余量时
改用经审查的原始快照回放，记录证明层次。分别覆盖默认120秒前后。
执行 VALIDATE_HOST：持续异常不能因其他指标正常而通过；负载释放后原指标新样本
才证明恢复。短突发/零delta不触发持续故障；checkpoint负载与硬件故障不得混淆。
保存CPU计数/分母、时间、任务进度和诊断回执；清理负载，禁止reset/reboot升级。

### GF-GENERAL-LIVE-002

必测：`memory_pressure`、`swap_pressure`、`no_swap`、`cgroup_only`、`read_failure`、`recovery`。
主机压力只在有足够带外保护的专属节点上做；否则用原始 meminfo 回放。受限容器
OOM 仅证明 cgroup 耗尽，不等于主机 memory_used≥95%。没有swap的节点明确记录能力，
不得动态开启业务swap来凑测试。memory持续默认120秒，swap按其独立规则。
实际验证原指标；读取失败应UNKNOWN而非健康/硬件故障。释放测试分配后核对主机、
collector和原任务均存活，保存内存/cgroup证据；不能用重启掩盖OOM失控。

### GF-GENERAL-LIVE-003

必测：`gpu_idle`、`rank_cpu_idle`、`low_available_memory`、`high_page_cache`、
`scratch_full`、`idle_node`、`attempt_change`、`ambiguous_ownership`。
使用受管测试程序的受控等待阶段，观察真实UUID/PID及 `/proc/<pid>/stat`；另以原始
采样夹具覆盖内存、page cache和专用scratch。默认低GPU/rank CPU、page cache为300秒，
低可用内存/本地盘为60秒。空闲节点、未分配GPU、多个attempt不应猜测任务归属。
旧attempt窗口不能在新attempt继续累计。执行绑定原GPU/指标的诊断，不因低利用率
盲目reset/reboot或未经授权压测。清理等待/压力，核对原rank真实进度；MIG另列能力缺口。

### GF-GENERAL-LIVE-004

必测：`capacity_full`、`below_threshold`、`inode_full`、`read_only`、
`required_mount_missing`、`probe_error`、`duplicate`。
配置专用有容量上限的测试文件系统为监控对象，只填测试卷而非 `/`、`/var` 或日志盘。
98%容量越界后验证真实cordon/taint和QUARANTINED，按当前契约不停止现有任务；
阈值下方不误报，重复采样不重复建链，其他owner的taint保留。
inode耗尽、只读、必需挂载消失、EIO/EACCES分别验，不能用剩余字节判断健康。
只读、消失或statvfs失败的配置挂载按 `filesystem_unavailable`
（READ_ONLY/MOUNT_MISSING/PROBE_FAILED）报为不可用并走 VALIDATE_HOST 诊断：不产生
容量 finding，不 reset/reboot/隔离，不可读状态记为不可用（UNKNOWN）而非健康，挂载
不可用期间 hold 节点、还原后随信号清零受控接纳（批次163修复，此前按 capacity-only
缺口记 FAIL；probe_error 仍因缺故障挂载夹具而 live 延后）。释放文件/还原挂载后验证再
受控解隔离。

### GF-GENERAL-LIVE-005

必测：`stat_timeout`、`capacity_full`、`mount_disappears`、`io_only_failure`、
`healthy_other_mount`、`shared_failure_domain`、`recovery`。
仅使用独立NFS/Lustre export和测试挂载。限制测试服务或测试路径连通性，覆盖statvfs
超时；另用有限测试容量到98%，及statvfs可返回但读写/fsync失败的情形。
确认collector有界返回且其他指标仍上报，VALIDATE_HOST不能用另一个挂载的健康代替。
两个任务节点访问同一export时记录共同故障域，不把每节点都误作坏硬件。
当前若未检测实际I/O或挂载消失，记录FAIL。还原export/连接后验证新采样、真实I/O和任务进度。

### GF-GENERAL-LIVE-006

必测：`sustained_util`、`sustained_latency`、`short_flush`、`zero_completed_io`、
`counter_reset`、`reference_iostat`、`recovery`。
限时I/O负载只写专用测试文件，不访问裸盘；记录挂载到块设备的映射。分别达到util/
latency门槛且跨越默认120秒，另测短flush。用同一设备/同一时间区间的原始diskstats
独立计算读写平均完成时间，并与iostat对照；不得用GFRS自己的值作为唯一oracle。
现有weighted I/O time口径若与标准await不符必须报差异/FAIL。零完成I/O、回退不产生
伪速率。停止负载后实际验证原设备恢复，无reset/reboot；保留原数据卷内容。

### GF-GENERAL-LIVE-007

必测：`health_failed`、`query_failed`、`scan_failed`、`malformed_json`、
`tool_missing`、`unsupported_device`、`healthy`。
先确认云实例是否暴露SMART，使用没有运行训练任务的专属节点。正向使用厂商支持的失败证据或私有命令输出夹具，
经真实 `smartctl` JSON解析到隔离执行；不破坏介质，不替换宿主smartctl。
`--scan-open`失败、设备查询失败、损坏JSON、工具缺失必须与passed=false分开；
能力不支持单独报告，不得当成功健康扫描。现有scan退出码处理缺口应明确FAIL。
保存扫描/逐设备输出、exit status、设备身份及Kubernetes回执；移除夹具并验证原工具。

### GF-GENERAL-LIVE-008

必测：`required_link_down`、`interface_missing`、`unknown_state`、`statistics_failed`、
`optional_interface`、`empty_required_list`、`bond_vf`、`recovery`。
明确管理/存储/训练接口角色，只操作有带外管理的专用数据接口；没有独立接口则用
受检sysfs源夹具。确认配置中的必需接口不是空列表，真实读取operstate和statistics。
down/完整枚举后的缺失应执行隔离；unknown或目录不可读不应当down。down同时统计坏
不能吞掉已知链路证据；非必需接口不隔离。bond/VF语义按实际拓扑独立验证。
恢复原接口配置、地址/路由、RDMA和训练连通性，再按owner受控接纳；不关闭管理链路。

### GF-GENERAL-LIVE-009

必测：`errors_delta`、`sustained_drops`、`short_drops`、`tcp_only`、
`counter_reset`、`wrong_interface`、`recovery`。
记录实际网卡驱动、计数器名称和所属接口。可使用隔离流量实验，但只有目标sysfs
counter确实增长才算原生正向；tc/netem的丢包或应用超时不能自动算NIC硬件错误。
没有可控counter时需受检读取夹具。errors增量和drops默认60秒规则分别验。
执行原接口Fabric验证，TCP重传/短drops/计数重置不误触硬件恢复。恢复后用新零增量
和通信结果确认，而不是清数据库或重置网卡让累计值归零。

### GF-GENERAL-LIVE-010

必测：`rdma_error`、`link_inactive`、`rnr`、`retry`、`cq_error`、`missing_counter`、
`ethtool_failed`、`non_efa`、`counter_reset`。
固定EFA驱动/固件、PCI和RDMA port、可用字段清单；每个错误类别独立基线和注入。
真实故障用厂商路径，否则在读取层回放，不能写sysfs counter。RNR/retry可能是拥塞/
资源压力，只做对应诊断，不推导坏卡。仅支持IB的计数器不能默认视为EFA有效读数。
缺字段/无匹配ethtool名称/命令失败应显式UNKNOWN或NOT_SUPPORTED；如果当前报零，
记录FAIL。核对目标port的VALIDATE_FABRIC、新观察及无未经授权reboot。

### GF-GENERAL-LIVE-011

必测：`active_to_zero`、`always_low`、`write_progress`、`phase_progress`、`broken_pair`、
`attempt_change`、`fr_enabled`、`fr_disabled`、`simultaneous_nodes`、`repeat_after_recovery`、
`unreachable_node`、`large_job`、`few_candidates`、`ambiguous_candidates`、
`different_process_groups`、`no_step_no_rank`、`active_window_resume`、
`fr_temporal_completion`、`fr_second_missing`、`fr_whole_group`、
`pyspy_nonzero`、`role_mismatch`、`tid_namespace`、`wchan_zero`、`cpu_interval`、`fr_plus_process`。
使用至少两节点受管NCCL测试程序，先记录同attempt活跃基线和startup grace。通过
测试程序的可恢复barrier/暂停点让通信停滞，保留带外恢复手段；不得操作无关PID。
必须实际观察完整EFA计数对接近零，否则不能把程序暂停当作流量规则已触发。
默认60/600秒分别核对warning/hung；始终低流量、坏计数对和换attempt为负向。
`repeat_after_recovery` 在首轮取证收尾、释放后任务自身流量持续越过产品重新布防窗口时
再次停滞，第二轮须有自己的incident与工作流（前驱为首轮）并完整执行，不被抑制、合并或重复。
`write_progress/phase_progress` 现在验证持续写入、step增长或phase切换均不延后RDMA时钟；
长checkpoint可能进入怀疑流程，但不能自动停训。`no_step_no_rank` 在128节点/1024进程
测试任务中关闭step reporter和FR，不导出RANK，仍需保留全部可发现进程证据并标记rank未知。
实际执行triage、诊断包和Fabric验证；FR未启用时仍有可用替代证据和明确缺项，不伪造dump。
`fr_temporal_completion` 对同一PID/PG核对两次新dump及提交/开始/完成序号，默认间隔至少2秒；
控制测试进程在collective前的可恢复暂停点，另验正常完成推进和只提交不完成的回放夹具。
`fr_second_missing` 使用隔离测试dump响应器不更新第二份文件，仅证明取证缺项处理；
不能把它当作真实NCCL故障。`fr_whole_group` 用受控等待或源回放确认全组未完成时
不虚构单rank根因。均保存两份摘要、实际间隔、身份及缺项，测量双dump I/O成本；
保留退出/超时释放，不阻断生产通信，不把短观察窗口写成已卡住600秒。
`pyspy_nonzero` 用专用测试工具模拟非零退出码/错误文本/坏JSON，只证明失败分类，
另用真实py-spy成功抓取核对JSON线程结构。`role_mismatch` 在受控混合并行任务中
分别声明不同PP stage、EP角色和无角色，确认不互相归因。`tid_namespace` 核对
容器native TID、宿主机NSpid映射和线程starttime；改变后台线程顺序不能改变训练线程位置。
`wchan_zero/cpu_interval` 验证0不是等待位置、CPU按实际秒数和SC_CLK_TCK归一化，
共享GPU不冒充PID利用率。`fr_plus_process` 在测试任务的正常FR完成推进旁保留
独立I/O等待证据，不互相覆盖。均不需要train step，失败/缺项保留覆盖率。
分别记录任务节点/rank总数、计划集合、实采集合、缺失/超时、可疑rank和对照rank；
summary/两轮triage/Fabric/bundle范围单独比对工作流参数；summary目标必须为全任务，
主动默认8、显式上限256（0仍为32），不限制全任务摘要；128节点可显式设128。
`active_window_resume` 验证未完成命令占据并发窗口、续跑不重复成功节点、
单节点30秒/整轮最高540秒超时、主动证据字节预算和未覆盖范围。
逐项核对PID身份、RANK/WORLD_SIZE缺失、不同PG成员覆盖及跨节点采样时间差。
分别验证1至3个可信候选时的定向bundle及对照rank、候选过多或弱证据时的保守范围；
不同通信组的collective进度不得混比。保存定位依据和置信度，报错rank不能直接当根因。
同时多节点触发时只保留受控工作流/节点命令并发，不能重复大规模取证。规模变体记录
真实卡数、诊断流量/CPU/内存/存储及任务扰动；小任务结果不外推到千卡/万卡。
全程不得因疑似hung自动cordon/STOP/reset；无step也能完成证据采集并交人工复核。
独立training-health告警不再进入NCCL取证链。采样上限、PID复用和PG隔离补充见
[NCCL hung 改进与验证](nccl-hung-hardening.md)。
远端不可达或覆盖不足保持明确不完整，不以健康Fabric宣布应用已修复。

### GF-GENERAL-LIVE-012

必测：`confirmed_missing`、`single_sample`、`query_failed`、`proc_unreadable`、
`expectation_wrong`、`reboot_verified`、`post_reboot_failure`。
固定真实期望数。真掉卡用自然/厂商案例；处理链回放应在完整查询中少一个真实UUID，
不能把改expected_count当实际掉卡。一次不确认，两次有效缺失才允许进入批准的reboot链。
独立回放NVML初始化失败和NVIDIA procfs不可读/不存在，必须UNKNOWN而非零卡。
当前“两个读取都失败仍生成REBOOT_NODE”已知风险应FAIL，不能用重启成功掩盖误判。
正向执行STOP、provider重启、新boot ID、完整GPU/Host/Fabric验证和受控任务恢复。
重启后仍缺失另验升级/保持隔离，不循环盲重启；注入须在重启前按记录解除以防跨boot投毒。

### GF-GENERAL-LIVE-013

必测：`pci_missing`、`driver_unbound`、`link_inactive`、`excess_device`、
`unknown_inventory`、`unreadable_state`、`bind_failure`、`post_repair_failure`。
准备专用节点、PCI/BDF allowlist和独立rebind保底。只允许对测试数据功能做经审批的
定向解绑；这只能证明DRIVER_UNBOUND，不能冒充PCI_DEVICE_MISSING。
分别核对missing的reboot链、unbound的定向bind+plugin+snapshot+验证、
link inactive先诊断；excess/unknown/不可读不能授权reboot。
实际bind失败和动作后验证失败须独立回执并按已批准的分支处置；禁止全局卸载efa。
保存PCI/driver/port、连续有效样本、命令及新库存。还原原绑定/插件/allocatable和通信后再接纳。

### GF-GENERAL-LIVE-014

必测：`gpu_plugin`、`efa_plugin`、`one_sample`、`replacement_uid`、
`expected_unknown`、`mig_timeslicing`、`validation_failure`、`running_training`。
物理库存正常的测试节点，通过受检plugin故障模式造成真实注册不足；
直接patch Node.status或伪造allocatable只算源回放，不能证明kubelet注册故障。
确认两次后重启正确plugin，保存原/新Pod UID、删除回执、重新Ready和资源数。
新快照验证正确expected_count，原训练身份与有效进度保持；失败升级reboot要单独授权。
未知期望值、MIG/time-slicing不能直接按物理卡少报处理。恢复原插件配置并确认资源注册，
不能用手工修Node.status掩盖注册链仍坏。

### GF-GENERAL-LIVE-015

必测：`critical`、`nonrecoverable`、`healthy`、`wrong_column`、
`unavailable_sensor`、`command_failure`、`unsupported_bmc`。
在无运行训练的专属节点先确认BMC权限和传感器支持，云实例无BMC时留能力记录。使用厂商测试接口或私有
命令输出夹具，逐行验证 `id|value|units|status` 的第四列，不用实际过热/停风扇制造告警。
cr/nr到真实隔离；单位字符串、三列损坏、不可用传感器、失败查询不能假健康或假隔离。
多传感器告警需保留名称及范围并去重。清理夹具，读回真实健康状态；不支持的变体不记PASS。

### GF-GENERAL-LIVE-016

必测：`uncorrectable_mce`、`corrected_mce`、`readonly_storage`、`rdma_fatal`、
`kernel_lockup`、`user_text`、`old_boot`、`cursor_pagination`、`unknown_attempt`。
启用并验证真实NodeLogCollector；严重内核输入用受检journal JSON读取夹具或自然事件，
保留来源、boot、cursor和事件时间。普通 `logger` 写入仍是用户态来源，不能伪造可信kernel。
四类严重日志到真实cordon、仅对新鲜归属任务STOP、quarantine与通知；未知归属不能猜任务。
corrected MCE不因上下文丢失而升错级，uncorrectable不被邻近corrected词降级；
训练stdout同文案、旧boot不得隔离当前节点。
同时间戳多cursor及collector重启验证不饿死后续日志。禁止制造kernel panic/真实MCE；
清理夹具与日志路径后保留隔离归属，按真实健康证据人工接纳。

### GF-GENERAL-LIVE-017

必测：`cgroup_oom`、`nccl_error`、`distributed_error`、`live_pid`、`exited_pid`、
`old_pod`、`cri_partial_rotation`、`missing_flight_recorder`。
测试应用在自己的日志中输出错误，或在隔离cgroup里触发自身OOM；不耗尽主机内存。
须经真实CRI/journal源，不直接建NodeLogBatch。NCCL文案回放只证明日志路径，不证明NCCL真hung。
验证LIVE/EXITED/UNKNOWN、Pod UID/container/PID、事件时间，旧Pod不能归到新attempt；
CRI P/F跨采集轮次、文件轮转和投递失败后仍可拼接，退出PID不应再抓栈。
普通 NCCL/分布式错误只保留日志和归属，不建额外诊断 workflow 或硬件 marker；
真实非零退出由 watcher 通用处理并归028验证，不增加 NCCL 专用过滤。
另验未退出且持续低 RDMA 的独立 hung 触发；清理日志夹具并核对没有误停对照任务。

### GF-GENERAL-LIVE-019

必测：`volatile_dbe`、`corrected_ecc`、`aggregate_dbe`、`retired_delta`、
`pending_supported`、`pending_unsupported`、`remap_failure`、`observe_profile`、`post_reset`。
按实际GPU架构列出字段支持；使用厂商批准注入或独立审查的DCGM源夹具，
环境回放工具不支持本类字段。不得实际损坏显存或修改真实retirement数据凑测试。
volatile DBE停归属任务并隔离分析；corrected/aggregate/普通retirement增量只诊断；
pending仅支持架构和授权Profile才走service-window reset；OBSERVE/未知架构不越权。
执行reset的变体须有无GPU客户端证明、实际命令回执、动作后新pending/recovery状态及
GPU/Fabric验证。remap failure的现场诊断/RMA审查不得直接算硬件已更换。清理夹具并人工验卡。

### GF-GENERAL-LIVE-020

必测：`crc`、`replay`、`recovery_counter`、`multiple_gpus`、
`counter_reset`、`stale_sample`、`independent_fatal_event`、`recovered`。
通过受检DCGM源夹具逐个增加目标链路计数，保持UUID/BDF/source/boot和有效分母。
单卡和多卡counter相关都应实际只读诊断，无DRAIN/reset；回退/旧样本不算新故障或修复。
独立致命Xid/SXid必须由对应原始源进入并按其catalog处置，不能被counter诊断吸收丢失。
记录链路、组件/复合finding和各自执行回执；恢复要求新正常增量与通信验证，
不要求累计值归零。自然链路坏损的物理验证另列，不拔链路当通用注入方法。

### GF-GENERAL-LIVE-021

必测：`not_started`、`in_progress`、`failed`、`before_window`、`ungraded_product`、
`monitor_restart`、`full_reset`、`every_gpu_success`、`recovery_failure`。
仅在受支持且已批准完整reset的独占节点测试。状态1/2/4从真实DCGM读取层进入，
默认299/300秒前后分开验；停止FM服务不保证产生目标状态，必须看到真实读数。
未分级型号和稳定状态不误判；collector重启后持有时间按实现重新确认。
工作流必须完整GPU/NVSwitch范围、quiesce、无客户端、一次全reset、恢复服务，
每张卡新Success与Fabric验证后才能接纳。单独重启FM成功不等于GPU完成注册。
失败分支保留隔离并按授权升级；清理状态夹具和服务，NVL5需分别核对FM/NVLSM而非只看一个PID。

### GF-GENERAL-LIVE-022

必测：`at_rate_limit`、`over_rate_limit`、`counter_reset`、`source_boot_change`、
`same_gpu_xid`、`wrong_gpu_xid`、`expired_xid`、`recovery_evidence`。
受检PCIe counter源以有效时间区间形成8/min及超过8/min，两者分别不触发/只读诊断。
对比PCIe链路与AER证据，禁止拔PCIe卡或制造电气错误。
同卡可信Xid32/79窗口内联合才能更强处置，异卡/过期不联合；独立Xid本身的恢复仍要
单独记录，不能误把它的reboot算作PCIe计数器授权。涉及硬件分支须另行批准。
动作后读取新真实状态与通信结果，不能以清计数替代验证；source/boot变化不形成跨代速率。

### GF-GENERAL-LIVE-024

必测：`thermal_warning`、`power_performance`、`corrected_memory`、`memory_failure`、
`nvlink_multiple_gpus`、`fresh_components`、`reused_components`、
`wrong_scope_source`、`independent_xid`。
使用按GPU/来源审查的原始metric/Xid夹具，不直接提交复合finding。热/功耗部分复用
ENV流程，其余字段不在环境代理allowlist内。覆盖45秒窗口内外、异卡、异来源、未来/
重复组件，检查实际工作流去重而非只比较规则名。
热warning、普通power cap、corrected memory、NVLink counter相关不独立授权更强动作；
有依据的内存故障和独立致命Xid不应被弱复合覆盖。保存组件与复合到incident的对应关系，
逐类恢复并验证无残留/重复恢复；危险分支须有独立维护授权。

### GF-GENERAL-LIVE-025

必测：`exporter_timeout`、`missing_field`、`one_gpu_missing`、`conflicting_alias`、
`nan_negative`、`duplicate_slot`、`sustained_unknown`、`post_refresh_success`。
只隔断/暂停测试exporter或用经审查读取夹具，保持Agent和控制面在线。
环境字段可复用现有代理，其他字段或别名冲突需额外夹具；不写坏真实驱动。
持续错误默认180秒后应执行一次有界snapshot和观测验证，不转reset/reboot。
必须逐GPU核对完整性，其他GPU有同名字段不能覆盖缺失GPU；现有全批字段检查若漏掉
目标GPU应FAIL。触发前旧成功不可放行，必须有触发后新完整数据；恢复唯一真实producer。

### GF-GENERAL-LIVE-026

必测：`uuid_changed`、`bdf_changed`、`count_unchanged`、`unknown_expected_count`、
`stale_boot`、`approved_rebaseline`、`no_unapproved_reset`。
身份变更用批准维护后真实库存，或受检inventory读取夹具；不为测试热插拔GPU。
同数量UUID/BDF变化应诊断，未知实例类型/期望数应取证复核，不能猜默认卡数并reboot。
旧boot/旧观察不能覆盖新库存。保存变更前后真实身份、批准记录、最新库存、通知和
实际只读工作流；合法维护建立新baseline后不永久重报旧身份。
夹具不得留下伪造UUID供后续真实reset使用；结束后验证每张真实GPU并恢复原配置。

### GF-GENERAL-LIVE-027

必测：`nan`、`stalled_step`、`reporter_silent`、`step_regression`、`straggler_group`、
`bounded_phase`、`old_attempt`、`checkpoint_unverified`、`resumed_progress`。
测试训练程序原子更新真实进度文件，由启用的reporter上报。分别主动报告非有限loss、
步数停滞、停止reporter、步数回退；这比直接POST心跳更能验证源路径。
NaN只停止匹配attempt并要求人工验证checkpoint，不自动恢复不可信状态；
心跳失联不等同NCCL挂起，straggler只比较同progress_group。
初始化/加载/保存phase有界，超时不能永久抑制；旧attempt/伪归属不授权STOP。
恢复必须原rank的新有效进度，Pod Ready/Fabric正常不够；删除测试文件和任务并核对对照。

### GF-GENERAL-LIVE-028

必测：`nonzero_exit`、`cleanup_timeout`、`one_restart`、`user_stop`、
`controller_stop`、`missing_allocation`、`old_terminal`、`api_unreachable`。
让受管测试关键容器真实非零退出，由completion watcher观察Pod/Job，而非手造终态。
跟踪日志先取证、STOP/suspend/删除、terminal、marker关联及预算内一次恢复；验证
新旧Pod UID、有效训练进度和预算记录。用户/本系统停止不可递归自动重启。
缺allocation和旧attempt终态不能控制新任务。控制面失联变体只阻断测试watcher的目标
访问且有恢复保底，验证紧急停止/重试无重复动作；清理orphan Pod和测试网络隔离。

### GF-GENERAL-LIVE-029

必测：`budget_consumed`、`hardware_chain_continues`、`restart_withheld`、
`concurrent_claim`、`worker_restart`、`duplicate_event`、`missing_context`、`single_notification`。
先通过合法测试失败/恢复真正消耗配置预算，读取持久账本和授权，不直接篡改计数。
随后由原始故障源触发含恢复及restart的链；危险硬件动作只在独占节点另行批准。
预算耗尽只扣住RESTART_WORKLOAD，既定STOP/必要硬件恢复/验证仍有实际回执。
并发claim、worker重建和重复事件不重复扣减/虚假释放，不创建新训练任务；
缺restart context仍须预检拒绝。保存一次通知和最终ESCALATED/QUARANTINED一致性，
不通过给业务job增加预算来清理现场。

### GF-GENERAL-LIVE-030

必测：`not_ready`、`stale_metrics`、`configuration_error`、`confirmed_hardware`、
`reserved_spare`、`no_spare`、`repair_failure`、`bounded_recheck`。
使用无业务任务的独立warm spare，读取实际Ready、保留状态、provider与原始健康源。
NotReady/旧遥测/配置错只阻止分配，不能凭此reboot；硬件正向必须从可信原始故障
进入，不能手工插marker充当检测。只有已授权的新鲜硬件证据才能执行provider恢复。
验证保留中的spare不被另一任务使用、无spare不回落provider replacement。
动作/验证失败转UNAVAILABLE并有界复查/通知，不产生恢复风暴。还原服务/配置后，
先健康验证再回AVAILABLE，保存reservation和node rebinding证明。

### GF-GENERAL-LIVE-031

必测：`inventory_silent`、`metrics_silent`、`host_silent`、`logs_silent`、
`disabled_unit`、`expired_agent`、`inactive_cluster`、`notification_failure`、`new_success`。
先建立必需通道真实成功，再仅暂停该通道且保留Agent有效lease。暂停不是disable；
disabled/masked本来就排除，不能用它验证静默告警。日志通道先按批准配置启用。
按运行阈值（通常180/420/900秒）加扫描延迟观察UNKNOWN、去重通知和每节点小时最多
一个snapshot刷新链；不任意重启服务。通知失败验证有界重试。
Agent过期/非活跃集群单独处理，不伪造成同一种静默。恢复后必须新成功时间推进，
不能仅看进程running或一个子指标成功；保留各通道实采覆盖表。

### GF-GENERAL-LIVE-032

必测：`unknown_xid`、`unknown_product`、`missing_topology`、`missing_version`、
`observe_profile`、`permitted_isolation`、`known_old_driver`、`duplicate`。
用经审查的原始kernel/inventory源夹具，选择当次catalog确实未知的代码或产品；
不直接伪造策略结论，不把用户态logger文字假称真实kernel源。
验证BLOCKED/SITE_SAFETY及Profile允许时的实际隔离；无owner/OBSERVE不得越权，
未知语义不reset/reboot。缺拓扑/版本证据应明确保留缺口。
已知普通Xid的旧驱动变体另验原动作，不能被Xid154 linkage误阻塞；采用仅需应用恢复
的受检变体，不能意外扩大为未批准硬件操作。重复事件不重复隔离，清理后人工验证再接纳。

## 已知缺口和执行入口

本轮补规格，不修改生产检测逻辑。特别需要用测试揭示的已知问题包括：GPU两个库存
读取失败可能被当成零卡、EFA不支持的错误计数可能报零、disk await口径、必需挂载/
真实I/O覆盖、SMART scan失败，以及多GPU字段完整性。这些有正反向验收，但不能
因尚未修复就宣称全部通过。标准Host/journal/非环境DCGM读取夹具、OEM实验条件、
千卡/万卡资源也仍是执行前提，不由本文件自动提供。

安全列出新增规格和复用环境规格（不会执行真机动作）：

```bash
.venv/bin/python tools/run_fault_test_cases.py \
  --category general-fault-live --category gpu-environment-live \
  --include-manual --list
```

`--include-manual` 的运行报告仍是 NOT_RUN，不是现场执行器。每类必须按映射文件逐变体
归档源/配置身份、实际证明层次、完整回执和清理结果，才可讨论该版本的实际覆盖率。

## GF-GENERAL-LIVE 运行器（phase 1）

一个通用运行器 `scripts/e2e/regional/run_general_live.py`，按类别拆分注入/判定
模块（`general_live_host`/`_utilization`/`_storage`/`_network`/`_collectors`/
`_attempts`，共享 `general_live_registry`/`_fixture`/`_common`/`_store`/
`_verdicts`）。主机探针为仅标准库的 `probes/general_live_node_probe.py`
（CPU/内存/page cache 负载、oracle 采样、采集单元与 exporter 的限时 SIGSTOP、
节点日志采集器的 runtime 启用）、`probes/general_live_storage_probe.py`
（loop 文件系统与 loop+device-mapper 测试设备、O_DIRECT 写入）、
`probes/general_live_net_probe.py`（host-local veth/dummy 测试接口、丢包生成器、
009 `tcp_only` 的测试 netns+netem TCP 夹具）、`probes/general_live_deps_probe.py`
（单次运行的临时测试包：仅在缺失时用 dnf 安装经审查的包、加载 `sch_netem`，
并只撤销本次运行装/载的内容）。采集单元清单的只读扩展复用
`probes/collector_window_probe.py`（仅对 host collector 追加一个测试挂载到
`GPU_FAULT_FILESYSTEMS`、或把测试接口追加到 `GPU_FAULT_REQUIRED_INTERFACES`）。

### 计划/执行

每个变体单独 plan/execute，守卫键为 `<case>.<variant>`（例如
`GF-GENERAL-LIVE-004.capacity_full`）：

```bash
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-001 --variant cpu_busy --plan
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-001 --variant cpu_busy --execute \
  --confirm GENERALLIVE001_EXECUTE --maintenance-window-end <UTC>
```

默认 `--plan` 为只读预检：确认目标节点（`--fault-node`，多节点变体另给
`--node-b`）Ready、无 cordon/隔离注解/隔离 taint、无业务或占 GPU 的 Pod、
Node Agent ACTIVE、无未关闭 incident/工作流、必需采集通道新鲜；HyperPod 另确认
`NodeRecovery=None`。`--execute` 需精确确认串 `GENERALLIVE<NNN>_EXECUTE` 和
未过期的维护窗口。危险分支另需 `--approve` 令牌（绑进 `plan.json`）：
`HOST_MEMORY_PRESSURE`（主机内存压力）、`KERNEL_MODULE_LOAD`（加载 dm_delay/
veth/dummy/sch_netem）、`ENABLE_NODE_LOG_COLLECTOR`（临时启用被禁用的节点日志
采集器）、`WATCHER_NETWORK_ISOLATION`、`CONTROL_PLANE_RESTART`、
`DECLARE_WARM_SPARE`（经 admin CLI 声明一台热备并 cordon，清理时释放/解除 cordon）、
`SPARE_KUBELET_STOP`（停热备上的 kubelet 使其 NotReady，带宿主机自动重启定时器）；
缺令牌记 `NOT_RUN(NOT_AUTHORIZED)`。可选令牌只解锁一个附加分支、不影响变体本身是否
可执行（计划里列为 `approvals_optional`，同样经 `--approve` 绑进 `plan.json`，
plan 与 execute 必须给出同一组令牌）：`TEMP_TEST_PACKAGE_INSTALL`（009 `tcp_only`：
节点缺 `tc` 时为本次运行用 dnf 安装 `iproute-tc`，`sch_netem` 无法为运行内核加载时
安装 `kernel-modules-extra-$(uname -r)`；dnf 预演事务必须恰好只安装这些包，不得拉入
内核、升级、降级、删除或其他依赖，否则 `NOT_RUN(FIXTURE_NOT_READY)`；清理与 deadman
只 dnf 卸载本次运行记录在案安装的包）。不给该令牌时节点缺 `tc` 仍记
`NOT_RUN(CAPABILITY_ABSENT)`。

### 安全

每次主机变更先用 `systemd-run --on-active` 布置 deadman，在运行器、探针 Pod、
控制面全部消失后仍把节点恢复基线；每个负载是带 `RuntimeMaxSec` 的瞬态单元；
配置只经 `/run` drop-in 改，关闭即还原；测试卷只在 `/mnt/gfrs-live/<run>/` 下，
backing 文件预分配且限容量上限（永不涨系统盘/`/var`/日志盘）；测试接口只操作
`gfrs*`（拒绝有地址或 `device` 链接的真实网卡，记录前后非测试接口摘要）。
运行器维护一个 LIFO 清理栈并独立还原；隔离类节点只经产品自身的验证式恢复
（VALIDATE_GPU/HOST/FABRIC → RESTORE_SCHEDULING）接纳，不手删 taint。任何清理
不能自证干净即整项 FAIL。

### 证据与判定

判定只读产品自己的记录（accepted batch、health-signal、latest metric、incident、
marker、workflow 的 step execution、remote command、notification、completion
decision、restart 预算账本），绝不 POST 预制 finding/workflow。运行器自带的
原始源读数（`/proc/stat`、statvfs、`/proc/diskstats`、sysfs 计数器、exporter
抓取等）作为有效性 oracle 存入 `cases/<key>/oracle.jsonl`；源没有真正变化时记
`INCONCLUSIVE` 而非 PASS。每变体在 `cases/<case>.<variant>/` 下留 `plan.json`、
`preflight.json`、`result.json`、各阶段 timeline 与 `<case>.<variant>.json`。
判定：有效注入+完整规格链+干净清理为 `PASS`；判定/动作/作用域错误或清理失败为
`FAIL`（规格明示的已知缺口在有效注入且未被检出时即记 `FAIL`）；缺令牌/能力/夹具
为 `NOT_RUN(reason)`；站点前提缺失为 `BLOCKED`。

### phase 1 范围与前提

首批 native 变体：001 CPU 过载、003 低利用率/资源异常、004 本地盘满、
006 磁盘 I/O、008 网卡链路 down、009 网卡错误/丢包、025 DCGM 采集不完整
（native：exporter_timeout/sustained_unknown/post_refresh_success）、
028 进程失败/终态、029 重启预算、031 采集器静默。延后变体留在分母并记
`NOT_RUN`/`BLOCKED`：需产品侧读取夹具的 live-replay（004 probe_error、
008 statistics_failed、009 errors_delta、025 的 5 个字段/别名/NaN 变体、
029 concurrent_claim/missing_context/hardware_chain_continues）、
本站能力缺失（008 bond_vf、031 inactive_cluster/notification_failure）、
需未授批危险分支（028 api_unreachable、029 worker_restart）。

需管理员安排的站点配置：008 `empty_required_list` 是站点 FAIL，修复需把主/EFA
netdev 写进安装器/管理配置的 `GPU_FAULT_REQUIRED_INTERFACES`（默认空）；
031 `logs_silent` 需按批准启用 `GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR`
（生产默认 false）；002/003 的主机内存压力需 `HOST_MEMORY_PRESSURE` 审批；
006 `sustained_latency`、008/009 的 veth/dummy/netem 需 `KERNEL_MODULE_LOAD`
或预装内核模块。运行器只做只读 `--plan` 预检，`--execute` 由运维在授权窗口执行。

009 `tcp_only`（native 负例：TCP 重传不是网卡错误，`general_live_tcp.py`）：网卡探针
建 `gfrs<digest>ns` 测试 netns 与一对 gfrs veth（复用 gfrs-only 建接口与真实网卡守卫，
主机端 `a` 在主机 netns、对端 `b` 直接建在测试 netns 内），两端只配静态 IPv6
link-local 测试地址（不加可路由地址、不改其他接口的路由），`tc netem loss 5%` 只挂在
主机端 veth 的出口；有界 TCP 流的发送端在**主机** netns、接收端在测试 netns（均为带
`RuntimeMaxSec` 的瞬态单元），因此重传计入产品采集器读取的主机 `/proc/net/snmp`。
具名 netns 是 `/run/netns` 下的 bind mount，而探针 Pod 的 `/host` 不传播挂载，所以
netns 相关命令都经 `nsenter` 在 PID 1 的挂载命名空间执行，瞬态单元与 deadman 才看得见；
deadman 在运行器失联时删除流、qdisc、netns 与 veth（临时包由其后的独立 deadman 撤销）。
只把主机端 veth 声明为必需接口（对端在测试 netns 中，主机采集器看不到，声明它会被读成
必需接口缺失 → `network_link_down` → QUARANTINE）。有效性 oracle：流窗口内主机
`RetransSegs` 增量 ≥ 20、netem 丢包 > 0、接收端收到数据，同时测试 veth 的错误/丢包计数
与每个真实接口（管理/VPC-CNI/EFA netdev）的错误计数增量都为 0，否则 `INCONCLUSIVE`。
产品须在流窗口内接收至少一批带正 `tcp_retransmits_delta` 的主机 batch（该指标无规则，
只采集），否则“无 finding”不构成证明，同记 `INCONCLUSIVE`；任何 NIC/TCP finding、硬件
恢复步骤或隔离为 `FAIL`。计划/执行形如（节点缺 `tc` 时）：

```bash
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-009 --variant tcp_only --fault-node <node> \
  --approve KERNEL_MODULE_LOAD --approve TEMP_TEST_PACKAGE_INSTALL --plan
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-009 --variant tcp_only --fault-node <node> \
  --approve KERNEL_MODULE_LOAD --approve TEMP_TEST_PACKAGE_INSTALL --execute \
  --confirm GENERALLIVE009_EXECUTE --maintenance-window-end <UTC>
```

### phase 2 范围（host/device 扩展）

第二批补充 host/device 类别 002 内存/交换、005 共享存储、007 SMART、
010 EFA/RDMA 计数、012 GPU 掉卡、015 BMC（模块
`general_live_memory`/`general_live_shared`/`general_live_devices`，每个必测变体
都在注册表里）。可在本站真机执行的只有不需要原始源回放夹具的变体：

- 002 `memory_pressure`、`recovery`（native，复用节点探针 `memory-start` 的受限
  cgroup 压力和 meminfo oracle，需 `HOST_MEMORY_PRESSURE`；`memory_used_percent`
  ≥95% 持续 120 秒 → VALIDATE_HOST 探针在加载期间失败，无 reset/reboot）；
  `no_swap`（observation：确认 `SwapTotal=0` 时 swap 规则天然不触发）。
- 007 `healthy`（observation：只读确认真实 SMART 判定健康且不产生存储 finding，
  作为失败变体的对照基线）。

其余变体按映射文件留在分母并记 `NOT_RUN`，原因精确可查：

- `FIXTURE_NOT_READY`：007/010/012/015 以及 002 `read_failure`、005 `stat_timeout`
  /`io_only_failure` 需要把 `smartctl`/`ipmitool`/`ethtool`/`nvidia-smi` 的输出或
  `/proc/meminfo`、EFA `hw_counters`/`state`、`/proc/driver/nvidia` 的读取**只在
  host collector 单元私有环境里**替换的共享原始源回放夹具。本阶段不构建该通用
  回放器（由协调方集中审查后统一提供）；不替换宿主二进制、不写全局挂载。
- `CAPABILITY_ABSENT`：005 的共享存储变体（本站无可扰动的独立 NFS/Lustre export）、
  010 `non_efa`（本站只有 EFA RDMA 设备）、015 `healthy`/`unsupported_bmc`
  （EC2 无 BMC）、002 `swap_pressure`（无 swap 且不得为测试开 swap）。
- `VENDOR_LAB_ONLY`：012 `confirmed_missing`/`reboot_verified`/`post_reboot_failure`
  需要真实掉卡与已批准的 provider reboot；本 lane 绝不解绑或移除真实 GPU。
- `cgroup_only`（002）需要一个有界的自身 cgroup OOM 注入模式，节点探针尚未提供。

现有的 `probes/collector_window_probe.py` 的 nvidia-smi shadow 只匹配库存快照查询
（026 身份路径），不覆盖 host collector 的 `uuid,utilization.gpu` 查询，故 012
`single_sample` 仍记 `FIXTURE_NOT_READY` 等待 host 查询 shadow。

### phase 2 范围与前提（工作负载/热备类）

第二批补 011 EFA 流量停滞/NCCL hung、027 训练进度、030 热备不健康；均用受管的
可丢弃测试任务，判定只读产品记录（011 读分组 `SITE_EFA_TRAFFIC` incident 的诊断
DAG、027 读 `SITE_TRAINING_HEALTH` incident/决策、030 读热备分级与注解），永不
POST 预制 finding。类别模块 `general_live_nccl`/`_progress`/`_spares`。

- **011（仅 HyperPod 车道，需 `--node-b`）**：专用受管两节点 NCCL 测试任务
  （`scripts/e2e/regional/general_live_nccl_job.py`，由
  `two-node-replace-pytorchjob.yaml` 的 Pod 模板渲染，
  保留镜像、`FI_PROVIDER=efa`、`FI_EFA_USE_DEVICE_RDMA`、`FI_EFA_FORK_SAFE`、GPU duty
  和每 Pod 8 GPU/16 `vpc.amazonaws.com/efa`，Master/Worker 分别钉到 `--fault-node`/
  `--node-b`）每 2 秒 all-reduce 1.5 MiB fp32。流量按产品规则取每节点 1～4 MiB/s：
  active 阈值 1 MiB/s、零阈值 1 B/s，SPIKE 为基线 4 倍且 SPIKE 不更新基线，低于
  4 MiB/s 时首个只含部分流量的采样不会把基线压低成持续 SPIKE；两节点树/环算法约
  1.5/2.8 MiB/s。原夹具每 2 秒只归约一个数值，整例每节点 EFA 仅约 74 KB，产品不可能
  建立 active 基线。运行器在 Master Pod 建暂停文件后，全局 rank 0 经 heartbeat
  all-reduce 投票，进入其他 rank 未发起的额外 collective 并阻塞；其余 rank 停在其前、
  GPU 维持验收 duty，见到本 Pod 新的 release 文件或有界 watchdog（2700 秒，长于运行器
  最长等待，进程组超时再加 900 秒）后加入，额外 collective 完成、训练继续。有效性
  oracle 是节点探针 `efa-traffic`（只读 `/sys/class/infiniband` hw_counters）每 30 秒
  对两节点收发对的读数：停滞前在 startup grace 内至少 3 个区间全部不低于 active 阈值，
  停滞后至少 600 秒内全部不高于零阈值，且两 Pod 存活、训练程序日志确认按所需方式停滞；
  任一不满足记 `INCONCLUSIVE`，不判 FAIL。产品须在两节点各出 `ZERO_WARNING`（不早于
  60 秒），再出 `HUNG_SUSPECTED`（不早于 600 秒），合并同波为一个工作流，编译只读诊断
  DAG `FREEZE_EVIDENCE→COLLECT_HUNG_SUMMARY×2→COLLECT_HUNG_TRIAGE×2→
  COLLECT_DIAGNOSTIC_BUNDLE→VALIDATE_FABRIC→ESCALATE_SUPPORT`，每步实际执行成功，收尾
  `ESCALATED`（`QUARANTINED` 为 FAIL），全程不 cordon/停任务/reset/reboot；释放后两节点
  流量须恢复。native 实测：`active_to_zero`、`always_low`（只归约 heartbeat、从未达
  1 MiB/s，再静默整个 hung 窗口，不得出 warning/hung）、`write_progress`（被停 rank 每秒
  在 `/dev/shm` 重写 4 MiB）、`phase_progress`（各 rank 在自己的
  `GPU_FAULT_RANK_PROGRESS_FILE` 写 `rank-progress/v1`，停滞中 step 增长、phase 每 30 秒
  切换）、`fr_enabled`（按运维手册注入 trace buffer/pipe/dump-on-timeout，至少一个 PID
  两次真实 dump 为 `PAIRED`）、`fr_disabled`（trace buffer 为 0，只能 `not_configured`）、
  `simultaneous_nodes`（合并 incident 须含两节点各自的 hung 证据）。
  延后留分母：规模类 `large_job`/`no_step_no_rank`/
  `active_window_resume`（`SCALE`，本站无 128 节点任务）、需审查测试 FR dump 响应器
  的 `fr_temporal_completion`/`fr_second_missing`/`fr_whole_group`/`few_candidates`/
  `ambiguous_candidates`/`different_process_groups`/`fr_plus_process`、需 py-spy 故障
  工具的 `pyspy_nonzero`、需角色声明测试任务的 `role_mismatch`/`tid_namespace`/
  `wchan_zero`/`cpu_interval`（均 `FIXTURE_NOT_READY`）、`broken_pair`（需 sysfs 读取
  夹具）、`attempt_change`（需可控换 attempt）、`unreachable_node`（隔离 Agent 通道
  未授批，`NOT_AUTHORIZED`）。
- **027（eks 或 hp 车道）**：产品训练进度 reporter 默认关闭、全局周期 monitor 必须
  保持关闭；仅测试任务 Pod 内以 `gpu-fault-collector training-progress` sidecar 启用
  reporter，读测试应用原子写的进度文件（启用 reporter 的源路径）。训练镜像不含产品，
  sidecar 用本 GPU 集群自身的运行时镜像（取自 `gpu-fault-cluster-executor`
  Deployment，须 digest 固定）运行 `/opt/gpu-fault/executor/bin/gpu-fault-collector`，
  连接取自 `gpu-fault-regional-connection`（URL、cluster id、cluster token，CA 经
  `SSL_CERT_FILE`），每个 reporter 有可写状态目录（outbox）；进度经
  `kubectl exec -i` 写入。扫描类信号用
  execution-token 的 `POST /v1/training-health/<cluster>/scan`（本集群一次），因此执行
  前须确认该测试任务是集群内唯一 RUNNING 受管 attempt（否则整集群扫描会误报正常
  任务 reporter 静默）。native 实测全部 9 个：`nan`（只停归属 attempt+ESCALATE，不
  重启不 cordon，checkpoint 只作证据）、`stalled_step`、`reporter_silent`、
  `step_regression`、`straggler_group`（同 `progress_group`）、`bounded_phase`、
  `old_attempt`、`checkpoint_unverified`、`resumed_progress`（原 rank 新有效进度才
  RECOVERED）。
- **030（仅 HyperPod 车道，`--fault-node` 指向待声明热备）**：经 admin CLI 声明一台
  空闲热备（`DECLARE_WARM_SPARE`），清理必定释放；每个阻断类变体只让该热备对分配
  不合格一种方式，读取产品自身的分级（`HyperPodSpareCoordinator.health_reasons`
  的分配门）并断言按正确 reason class 拒绝、且未产生任何 provider 恢复。native 实测：
  `not_ready`（停 kubelet，需 `SPARE_KUBELET_STOP`）、`stale_metrics`（暂停 metrics
  采集器）、`configuration_error`（注入非 AVAILABLE pool state）、`reserved_spare`
  （注入他 incident 的 reservation，验证不被他任务复用）。本站为区域/多集群控制面
  （`GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false`），周期性 `HyperPodSpareHealth
  Controller` 不运行，故 `confirmed_hardware`/`repair_failure`/`bounded_recheck` 记
  `NOT_RUN(CAPABILITY_ABSENT)`；`no_spare`（无热备不回落 provider 替换）只在危险的
  REPLACE_NODE 失败转路径可见，记 `NOT_RUN(NOT_AUTHORIZED)`。

计划/执行形如 `--case GF-GENERAL-LIVE-011 --variant active_to_zero --site-file <F>
--node-b <N2> --hyperpod-cluster <HP>`（011/030 在 hp 车道，011/030 需 `--site-file`，
011 需 `--node-b`，030 的热备节点用 `--fault-node`）。

### phase 2 范围与前提（内核/应用日志、未知错误、EFA 库存）

phase 2 新增经审查的 `live-replay` 类别，统一经产品自身的读取路径注入：

- 可复用的**内核日志源回放**
  `scripts/e2e/regional/general_live_kernel_replay.py`（主机探针
  `probes/general_live_kernel_source_probe.py`）为单节点/单 boot 安装一组经审查
  的 kernel-transport 记录、arm、读回产品实际摄入内容、清理。两种模式：journal
  模式在节点日志采集器的 PATH 前置一个 `journalctl` 透传包装（转发真实输出，仅在
  arm 后追加经审查记录，有界 emissions、去重为一条 incident），kmsg 模式用 `/run`
  drop-in 把内核采集器的 `GPU_FAULT_KMSG_PATH` 指到经审查夹具文件。`_TRANSPORT=kernel`
  为可信 `dmesg` 来源，其它 transport 保持不可信 `journal` 来源——这正是信任门禁。
  lane R 的“独立可信 XID”变体后续复用此助手。
- 主机采集器**文件源回放** `general_live_source_replay.py`（探针
  `probes/general_live_source_probe.py`）只在主机采集器自身 mount namespace 内用
  `BindReadOnlyPaths` 替换一个允许列表内的 procfs/sysfs 文件（`/proc/stat` 或 gfrs
  测试网卡的 statistics 计数器），或重指 `GPU_FAULT_PCI_DEVICES_ROOT`。
- 所有回放都带 `systemd-run --on-active` deadman，证据标注 `proof_scope: live-replay`，
  只证明处理链，不证明物理故障。

已实现变体：016 严重内核日志（全部 9 个，含 user_text/old_boot/corrected_mce 的
反向门禁）、017 应用错误日志（cgroup_oom 诊断与 nccl_error/distributed_error 仅证据）、
032 `unknown_xid`（判为 SITE_SAFETY）与 `duplicate`、013 `unreadable_state`
（UNKNOWN 而非零卡），以及 phase-1 文件源 001 `invalid_delta`、008 `statistics_failed`、
009 `errors_delta`。

计划/执行守卫键仍为 `<case>.<variant>`，确认串 `GENERALLIVE<NNN>_EXECUTE`。新 `--approve`
令牌：`KERNEL_SOURCE_REPLAY`（kmsg 源重指，032）、`EFA_DRIVER_UNBIND`、
`DEVICE_PLUGIN_RESTART`（013/014 的原生注入，本轮未执行）；016/017 的 journal 模式复用
`ENABLE_NODE_LOG_COLLECTOR`（节点日志采集器生产默认禁用），008/009 复用
`KERNEL_MODULE_LOAD`。每个 016/017/032/013 变体单独 plan/execute，与 phase-1 相同：

```bash
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-016 --variant rdma_fatal --execute \
  --confirm GENERALLIVE016_EXECUTE --approve ENABLE_NODE_LOG_COLLECTOR \
  --maintenance-window-end <UTC>
```

本轮保持 `NOT_RUN` 并留在分母：014 全部（伪造 allocatable/改 Node.status 不算证明，
原生 plugin 故障需 `DEVICE_PLUGIN_RESTART`）、013 多数（唯一 EKS 节点上的 missing→reboot
链、保留的原生解绑、不可重指的 `/sys/class/infiniband`）、032 的目录/版本/Profile 权限
变体（AUTH-015 未配置）、017 的归属变体（需受管 attempt 与 CRI Pod 日志）、004 `probe_error`
（`statvfs` 错误无法由只读文件 bind 产生）。

### phase 2 范围与前提（DCGM 字段 / GPU 库存回放）

第二批覆盖按 DCGM 字段和 GPU 库存判定的类别：019 ECC/行重映射/页退役、
020 NVLink 错误计数、021 NVLink fabric 未训练、022 PCIe replay、024 DCGM 复合、
026 GPU 身份/期望数，以及 phase 1 延后的 025 五个字段变体
（missing_field/one_gpu_missing/conflicting_alias/nan_negative/duplicate_slot）。
注入一律 `live-replay`：只替换单个原始读数，collector 解析、传输、判定、数据库、
调度与已批准的节点动作均真实。

机制（无产品代码改动）：经审查的 DCGM 字段回放工具
`tools/gpu_dcgm_field_replay.py` 带自己的**按类别字段白名单**（与环境工具的
`ALLOWED_FIELDS` 分离，不拓宽），以及经审查的 `nvidia-smi` 透传包装器
`tools/nvidia_smi_inventory_replay.py`（026 身份/BDF/期望数）。两者由主机控制器
`probes/general_live_dcgm_probe.py` 以 base64+sha256 装入 `/run/gfrs-live/<digest>/`，
工具在 loopback 端口 `--serve`（装入即透传，`arm` 前不改任何字节），包装器仅排在
目标单元 PATH 首位；只对一个被允许的采集单元
`gpu-fault-metrics-collector.service` 用 `/run` drop-in（`EnvironmentFile=-`，
覆盖 `GPU_FAULT_DCGM_METRICS_URL` / PATH / 旧 boot / 期望数），并先布置
`systemd-run` deadman 还原；Node Agent 永不在白名单内。计数类故障只能靠产品新采的
零增量批次清除（回退计数是 invalid delta，会保留 finding），故回放在产品接受干净
批次前一直保持抬升值再 disarm。判定只读产品自己的记录
（`general_live_gpu_store`/`general_live_gpu_verdicts`），绝不 POST 预制 finding。

需审批与独占节点的变体：019 `pending_supported`/`post_reset` 单卡 service-window
reset 需 `--approve GPU_RESET`；021 任一 finding 都驱动全 GPU+NVSwitch reset，故
该类变体需 `--approve GPU_FULL_FABRIC_RESET`（`recovery_failure` 另需
`NODE_REBOOT`）并只在 HyperPod 独占节点跑；022 的 XID 联合变体只把 DCGM last-XID
gauge 回放为 XID 32（`--approve DCGM_XID_GAUGE`）；019 `remap_failure` 的 RMA 现场
诊断需 `GPU_FIELD_DIAGNOSTIC`。延后：020 `independent_fatal_event`、024
`independent_xid` 需 K 车道的可信内核 XID/SXid 源，记 `NOT_RUN(FIXTURE_NOT_READY)`；
024 热/功耗复合复用 GF-ENV-LIVE 环境回放；019 `pending_unsupported`（无旧架构卡）、
`observe_profile`（无 OBSERVE reset Profile）记 `CAPABILITY_ABSENT`。

规格明示、注入有效但产品不足的按 `expected_outcome="FAIL"` + `product_gap` 记录，不
掩盖。首批记为预期 FAIL 的四个产品缺口——025 `one_gpu_missing`（完整性按全批 union
计算，漏查单卡）、022 `expired_xid`（PCIe↔XID gauge 回退无时效检查）、026
`bdf_changed`（身份只比 UUID 集）、026 `stale_boot`（库存基线不比 source_boot_id）
——已由批次 160 修复，对应变体改为预期 PASS；产品回退时这些变体判 FAIL。
