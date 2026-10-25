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
未执行原因前缀为 `VENDOR_LAB_ONLY`、`CAPABILITY_ABSENT`、`FIXTURE_NOT_READY`、
`NOT_AUTHORIZED`、`SCALE` 与 `COVERED_BY_REGIONAL: <用例号>`：最后一种指某个
GF-REGIONAL 用例已在本站真机 PASS，且其注入与断言正是该变体要做的（同一真实注入、
同一产品路径），故该变体不另建夹具、在计划/看板上记 `NOT_RUN(COVERED_BY_REGIONAL)`
并带交叉引用；其注册项的车道、令牌与预期保持不变，日后可直接启用。

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
判定按产品规则 id 认持续 finding 的 incident（事件 id 以
`-low_memory_available`/`-high_page_cache`/`-high_local_filesystem_usage` 等小写
规则 id 结尾），不是阈值族名 `memory`/`page_cache`/`local_filesystem`。诊断型
incident 常在两次轮询之间就以 `diagnostic inconclusive` 关闭为 `RECOVERED`，
任何状态都算；但只认本次读取起点之后、本节点本集群创建的 incident，后来的 workflow
（如 validated restore）带进读取结果的旧 incident 不算本次 finding。
`high_page_cache` 以运行器自带的 meminfo oracle（产品公式
(Cached+SReclaimable−Shmem)/MemTotal）确认条件：`page_cache_percent` 未在 300 秒
持续窗口内一直 ≥90% 即 `INCONCLUSIVE`（condition not established），只有条件成立
而产品没有 finding 才记 `FAIL`。
`gpu_idle`/`rank_cpu_idle` 断言产品对空闲 attempt 的处置：FREEZE_EVIDENCE 后
VALIDATE_GPU/VALIDATE_HOST 携带绑定指标的 `host_diagnostic_probe`（≤5%），attempt
仍空闲时探针必须失败并指出 `host_diagnostic:<指标>`，workflow FAILED，incident 以
`diagnostic inconclusive` 关闭为 `RECOVERED`（不得 ESCALATED），无
RUN_DCGM_DIAGNOSTIC/硬件步骤，incident 带 attempt id。验证步骤在更新的证据到达前为
WAITING，所有判定验证的等待都等到该步骤终态（或 workflow 终态）再判，判定所用的
incident/workflow 记录存为 `<wait>-chain.json`。

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
缺口记 FAIL）。`probe_error` 的故障挂载夹具：存储探针的 loop + device-mapper 测试设备改载
FAT16（`dm-create --fstype vfat`），`dm-retarget --target error` 先重新挂载再把表换成 `dm_mod`
内建的 `error` 目标——挂载仍在（非 MOUNT_MISSING）、宿主表为 rw（非 READ_ONLY），而每次
`statvfs` 都返回 EIO（FAT 在挂载后首次 `statfs` 时才从设备读 FAT 计数空簇，ext4 全在内存，
对 dm-error 无感）→ PROBE_FAILED；`--target data` 还原，探针自身读数是有效性 oracle（未读到
EIO 即 `INCONCLUSIVE`，缓存命中时从新挂载重试最多 3 次）。交换前先 sync，不向 error 表写入
（写失败会记 `Buffer I/O error`，节点日志采集器 `log_rules.py` 的存储规则会把它判成
QUARANTINE），因此节点日志采集器处于 active 时记 `NOT_RUN(FIXTURE_NOT_READY)`；缺 `vfat`
模块时在 `KERNEL_MODULE_LOAD` 下加载并在清理时卸载，节点需有 `mkfs.vfat`（dosfstools）。
三个不可用变体另核对 `filesystem_unavailable` 采样的 `failure_mode` 标签（有标签而不符即
FAIL）。只读以宿主 `/proc/1/mountinfo`
中承载该路径挂载的 per-mount/超级块 `ro` 为准，不取 collector 沙箱（`ProtectSystem=strict`
只读绑定）自身的 `ST_RDONLY`；沙箱遮住的路径不上报（批次166）。释放文件/还原挂载后验证再
受控解隔离。

### GF-GENERAL-LIVE-005

必测：`stat_timeout`、`capacity_full`、`mount_disappears`、`io_only_failure`、
`healthy_other_mount`、`shared_failure_domain`、`recovery`。
仅使用独立NFS/Lustre export和测试挂载。限制测试服务或测试路径连通性，覆盖statvfs
超时；另用有限测试容量到98%，及statvfs可返回但读写/fsync失败的情形。
确认collector有界返回且其他指标仍上报，VALIDATE_HOST不能用另一个挂载的健康代替。
两个任务节点访问同一export时记录共同故障域，不把每节点都误作坏硬件。
当前若未检测实际I/O或挂载消失，记录FAIL。还原export/连接后验证新采样、真实I/O和任务进度。
产品行为（批次176）：共享挂载从宿主挂载表 `/proc/1/mountinfo` 发现（沙箱遮住的不上报）；每个挂载在
statvfs 之外再做一次有界、只读的目录项读取（不写、不 fsync），statvfs 正常而读取失败报
`shared_filesystem_unavailable=1`、`failure_mode=IO_ERROR` 并带 `errno`/`source`，读取超时报 `IO_TIMEOUT`
（缺证据，不读作健康）；statvfs 超时/报错为 `PROBE_TIMEOUT`/`PROBE_FAILED`。曾见过的节点级共享挂载（不含
kubelet Pod 卷、容器运行时目录和 autofs 子项）从表中消失而挂载点目录仍在，报 `MOUNT_MISSING`，记忆保存在
`/var/lib/gpu-fault/host-collector-shared-filesystems.json` 并跨 collector 重启；重新挂载清零，挂载点目录被
删除视为退役而遗忘。`mount_disappears`/`io_only_failure` 按这些信号验证，此前登记的产品缺口已修复。

`shared_failure_domain` 的产品形态（批次177，此前因每节点各成一条 NODE_HEALTH incident、无共同记录而记
FAIL）：两节点仍各自产出诊断链（FREEZE_EVIDENCE → VALIDATE_HOST，`RUN_DIAGNOSTICS`，不 cordon/taint、
不停任务）；≥2 个节点在 600 s 内报同一 export 不可用时，控制面另记一条集群级
`event_type=SHARED_STORAGE_FAILURE_DOMAIN` incident（`policy_reference=shared-filesystem:<export>`，
`node_ids` 含两节点，reasons 第一条写明 export 与节点、随后每节点一行写其 incident id，无 workflow、
`effective_action` 为空），两节点 incident 各追加一条 `common failure domain <id>` reason 回指，整个故障域
只发一封 `SHARED_STORAGE_FAILURE_DOMAIN` 通知（正文列 export、节点与各节点 incident）；两节点样本都回 0 后
该记录自动 `RECOVERED`，一节点回 0 而另一节点超过窗口无读数则转 `ESCALATED` 交运维。验收读
`GET /v1/incidents?state=DETECTED,ESCALATED,RECOVERED`（`event_type`、`node_ids`、`first_reason`）与
`GET /v1/incidents/<id>` 的 reasons。export 标识取采集器样本标签 `source`（mountinfo 挂载源）或
`server`+`export`；节点侧采集器需随 `shared_filesystem_unavailable` 样本带该标签，否则标识为 UNKNOWN、
不关联（挂载点不是标识），夹具应对两节点挂载源一致作 `NOT_READY` 前置检查。

本站实现（`general_live_shared.py`）。产品没有共享存储配置项：host collector 的
`_shared_filesystems`（`collectors/host/shared_filesystems.py`）每个采样周期从宿主挂载表
`/proc/1/mountinfo` 按文件系统类型（`nfs`/`nfs4`/`lustre`）发现共享挂载，做有界 `statvfs` 与一次
有界只读目录项读取（每次调用 5 s、每周期 15 s），超时或出错报 `shared_filesystem_unavailable`=1
（CRITICAL RUN_DIAGNOSTICS，`failure_mode` PROBE_TIMEOUT/PROBE_FAILED/IO_TIMEOUT/IO_ERROR），否则报 0 与
`shared_filesystem_used_percent`（98% WARNING RUN_DIAGNOSTICS）；曾见过的节点级共享挂载消失而挂载点目录
仍在报 MOUNT_MISSING（记忆跨 collector 重启）；每个共享样本带 `source`/`fstype`。夹具因此**不**把测试 export
追加进 `GPU_FAULT_FILESYSTEMS`——那会按本地盘规则（`filesystem_used_percent` 98% →
QUARANTINE、MOUNT_MISSING）判定，把共享存储状况变成坏节点结论并掩盖 005 要揭示的缺口。
测试 export 由 `shared_export_fixture.py` 提供：每个 export 一个 Pod
（`gfrs-export-<digest>-<a|b>`，用户态 NFSv4 服务 NFS-Ganesha / VFS FSAL，导出一个 256 MiB 的
内存 `emptyDir`——其 `sizeLimit` 即全部容量，`Root_Squash` + `chmod 1777` **且** `chown 65534:65534`
的 export 根目录（镜像实测：`fs.protected_regular>=1` 下被 squash 的 root 对 root 属主 sticky 目录的
`open(O_WRONLY|O_CREAT)` 在客户端即被拒 EACCES，只留下 0 字节文件），`Attr_Expiration_Time = 1` 写在
EXPORT 块（NFS-Ganesha 4.3 在 MDCACHE 块报 Unknown parameter），`NFS_KRB5 { Active_krb5 = false; }`
消去启动 CRIT；`activeDeadlineSeconds` 为集群侧 deadman，按 UID 前提删除，调度到客户端节点之外或由
`--export-node` 指定）加其 `ganesha.conf` ConfigMap。本站镜像：
`<account-id>.dkr.ecr.us-west-2.amazonaws.com/gfrs-test/nfs-ganesha@sha256:d354efdc8364e4ba17b83c4aa27cdef4001a474fc950b07547e7fcf9d1e14d7f`
（Ubuntu 24.04 + NFS-Ganesha 4.3：`/usr/bin/ganesha.nfsd`、`/usr/lib/x86_64-linux-gnu/ganesha/libfsalvfs.so`，
`/bin/sh` 为 dash，入口脚本按 POSIX sh 书写）。客户端由 `probes/general_live_shared_probe.py` 在
PID 1 的挂载命名空间以 `vers=4.1,soft,timeo=100,retrans=2,actimeo=2` 挂到
`/mnt/gfrs-live/<digest>/shared`（节点作用域；清理 `umount` **并** `rmdir`，产品据此遗忘该挂载）：
deadman 与记录先于挂载，`systemd-run --on-active` deadman 级联 `umount`/`-f`/`-l`，有界
`statvfs`、写/fsync/回读证明与填充读数都经 `nsenter` 在宿主命名空间执行。注意 `soft` 选项**不是**
读数的上界：镜像实测挂起的 export 在 185 s 内未让 statvfs 失败、125 s 内未让写失败（NFSv4.1
会话持续重试），真正限定每次读数的是读取方自己的线程 deadline（产品每次调用 5 s、探针的工作线程
`timed_out`），`soft` 只保证彻底消失的 export 不会永久挂住线程。挂起 = 对 ganesha
子进程 `SIGSTOP`（内核仍接受 TCP 而无应答，statvfs 不返回），`SIGCONT` 以同一组文件句柄恢复、
无需重挂（内核 nfsd 无法这样挂起，故用用户态服务）。判定只读产品记录：
`stat_timeout`/`healthy_other_mount`/`recovery`（`shared_filesystem_unavailable`=1 →
FREEZE_EVIDENCE、VALIDATE_HOST 在该指标上失败、无隔离/硬件步骤；挂起期间其他主机指标仍到达
（≥2 个 batch 带 `memory_used_percent`），另一挂载保持 0 且不建链；恢复后信号清零、新
`used_percent` 采样与真实写/fsync/回读）和 `capacity_full`（`shared_filesystem_used_percent`
≥ 98 → 诊断链，不得 QUARANTINE，释放后低于 98）预期 PASS；`mount_disappears`（`nfs-umount` 不带
`--forget`，挂载点目录保留——目录也删掉会被产品读作退役而遗忘，记 INCONCLUSIVE：产品报
`shared_filesystem_unavailable`=1 `failure_mode=MOUNT_MISSING` 且 `source` 指向测试 export，诊断链后
`nfs-remount` 清零；清理时 `cleanup --kind nfs-shared` 删除目录使产品遗忘）与 `io_only_failure`
（服务端 `chmod 000` export 根：statvfs 正常而 open/read/fsync EACCES，产品的目录项读取报
`IO_ERROR`+errno 或 `IO_TIMEOUT`，`chmod 1777` 后清零）预期 PASS；`shared_failure_domain`（批次 177 后
预期 PASS，原产品缺口已关闭）：挂起前两客户端的基线样本必须带同一个 `source`（缺标签或两节点
`source` 不同记 `NOT_RUN(FIXTURE_NOT_READY)`，无从关联）；两节点都出 finding 后重读两节点记录，要求
恰有一条 `event_type=SHARED_STORAGE_FAILURE_DOMAIN` incident（`policy_reference=shared-filesystem:<source>`，
`node_ids` 含两节点，无 workflow/`effective_action`，首条 reason 写明 export 与两节点、每节点一行
`member <node>: incident <id>`），两节点 incident 各带 `common failure domain <id>:` 回指 reason，恰一封
`SHARED_STORAGE_FAILURE_DOMAIN` 通知；两节点诊断链仍只诊断；恢复后该记录须 `RECOVERED`。全部变体需
`--approve KERNEL_MODULE_LOAD`（缺 `nfsv4` 时加载并在清理时卸载）；`shared_failure_domain`
标志 `X` 并要求 `--node-b`（EKS 的两个 GPU 节点同时作客户端，export 调度到两者之外）。
站点前提：`--export-image` 或 `GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE` 指向带 `ganesha.nfsd` 与
`/bin/sh` 的不可变摘要镜像；节点 AMI 有 `nfs-utils`（`mount.nfs4`）；节点到 Pod IP 的 TCP/2049
放通（GPU 集群内 VPC CNI 的 Pod 共用节点 ENI 安全组，自引用规则通常已覆盖；CPU 集群需 VPC
可达与显式规则）；kubelet 对内存 `emptyDir` 执行 `sizeLimit`（首个客户端 `statvfs` 校验容量，
不符记 `NOT_RUN(FIXTURE_NOT_READY)`）。不起受管训练任务，“任务进度”不在本主机类别范围内。
计划/执行形如：

```bash
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-005 --variant stat_timeout --fault-node <node> \
  --export-image <image@sha256:...> --approve KERNEL_MODULE_LOAD --plan
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-005 --variant shared_failure_domain --fault-node <node-a> \
  --node-b <node-b> --export-image <image@sha256:...> --approve KERNEL_MODULE_LOAD \
  --execute --confirm GENERALLIVE005_EXECUTE --maintenance-window-end <UTC>
```

### GF-GENERAL-LIVE-006

必测：`sustained_util`、`sustained_latency`、`short_flush`、`zero_completed_io`、
`counter_reset`、`reference_iostat`、`recovery`。
限时I/O负载只写专用测试文件，不访问裸盘；记录挂载到块设备的映射。分别达到util/
latency门槛且跨越默认120秒，另测短flush。用同一设备/同一时间区间的原始diskstats
独立计算读写平均完成时间，并与iostat对照；不得用GFRS自己的值作为唯一oracle。
产品`disk_io_await_ms`已改为标准await口径（diskstats字段(4+8)/(1+5)，见故障类别总表
"磁盘 await 口径"），`reference_iostat`预期PASS；若与标准await不符仍必须报差异/FAIL。
零完成I/O、回退不产生伪速率（跨越计数器回退的区间不产生样本）。停止负载后实际验证原设备恢复，无reset/reboot；保留原数据卷内容。

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

夹具（`scripts/e2e/regional/general_live_device_plugin.py` +
`probes/general_live_device_plugin_probe.py`，审批 `DEVICE_PLUGIN_RESTART`）：不移除
DaemonSet、不改 `Node.status`，而是对 plugin 容器自身进程发 SIGSTOP（仅签名匹配该
plugin Pod UID 且基名在允许清单内的进程，deadman 用 SIGCONT 兜底），冻结后 plugin 不再
应答 kubelet 的 `ListAndWatch`，kubelet 注销资源、allocatable 下降而 Pod 仍 Running，由
产品自己删除并替换 Pod。`gpu_plugin`/`efa_plugin` 判 `RESTART_*_DEVICE_PLUGIN` +
`TRIGGER_HEALTH_SNAPSHOT` + `VALIDATE_{GPU,FABRIC}` 均 SUCCEEDED、`replacement_pod_uids`
为新 Ready Pod、allocatable 回基线、incident RECOVERED、无 reboot；`one_sample` 在产品记录
一次失配后立即 SIGCONT，验证两次确认规则不成 finding/不动作；`replacement_uid` 验证新 UID
被记录、异于被删 UID 且不被二次删除；`validation_failure` 跨替换持续冻结使重启的 allocatable
快照不达标，`RESTART_*_DEVICE_PLUGIN` 失败并经 `failure_escalation_action=REBOOT_NODE`
升级到 `RESTART_NODE`（单独 `NODE_REBOOT` 审批，X 标志）。观察到的产品不足：最终步骤回执只
持久化 `replacement_pod_uids`，被删 Pod 的 `deleted_pod_uid` 只在删除那次尝试的瞬时回执里
（`node_operations._restart_device_plugin_node` 终态分支不再带它），故原 UID 需实时抓取或由
Pod 普查（原 Pod 已消失 + 新 Ready Pod）佐证。`expected_unknown`/`mig_timeslicing`
（CAPABILITY_ABSENT）与 `running_training`（FIXTURE_NOT_READY，lane N）保持 NOT_RUN。

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
判定按产品的事件 id 认采集不完整 incident：`inc-gpu-<错误串首批 batch>-dcgm-fields`
（GPU finding 以 `gpu-<finding id>` 进入节点健康链路），id 中不含指标名
`dcgm_field_completeness`。

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
本站实现（`general_live_attempts.py`）：`old_terminal` 用受管任务模式 `exit_once`——首个 attempt
30 s 后非零退出，产品 RESTART_WORKLOAD 复制出的新任务其 Pod 经 downward API 读自己的
`gpu-fault.io/attempt-id` 标签（产品派生的 `-r-` id）而持续运行；待产品的 RESTART_WORKLOAD 执行记下
`restart_attempt_id` 并经过一个 watcher 周期后，`TERMINAL_REPLAY_PROBE` 在 API Pod 内读出产品**自己
记录**的旧 attempt TerminalEvent 并经 `CompletionService.handle_terminal`（`/v1/attempts/terminal`
路由同一入口）重投（不手造任何终态，只对 `gl28` 前缀的测试 attempt）。PASS 要求：重投返回原决策
（status/recovery_plan_id/event_key 相同）且 `duplicate=true`；旧 attempt 恰一个 RESTART_WORKLOAD
工作流、预算只扣一次；新 attempt 无 terminal、decision 与 containment。无决策、无新 attempt 或无已记录
终态记 INCONCLUSIVE。`missing_allocation` 记 `NOT_RUN(FIXTURE_NOT_READY)`：Kubernetes 上 watcher 的
allocation 来自每个已终止关键容器的 `spec.nodeName`（`completion_pod_parsing` →
`watcher._terminal_event`），已终止的容器必有节点，无 Pod 的 FAILED 工作负载到不了终态，故 watcher 观察
到的终态必带 allocation；只有手造 TerminalEvent POST 到 `/v1/attempts/terminal` 才可能缺它，而规格与
本 lane 规则都禁止手造终态；产品路径（`service._decide` → `planner.from_missing_allocation`）由软件测试
覆盖。

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

metrics 采集单元同时产出 GPU_INVENTORY（180 秒）与 GPU_METRICS（420 秒）：暂停该
单元时库存通道先越过阈值，其每节点每小时一次的 snapshot 刷新（Node Agent 的
TRIGGER_HEALTH_SNAPSHOT 会 `systemctl restart` metrics 与 host 采集器）会结束一次裸
SIGSTOP 并启动一个不受暂停的新进程（实测：暂停 183 秒后新进程让 GPU_METRICS 一直
FRESH_SUCCESS）。因此暂停必须跨越产品自身的重启；静默通知判定只认目标通道自己的
`collector-silent/<cluster>/<node>/<channel>/<bucket>` 键（库存通道的通知不能代替
metrics 通道的），同一告警小时内重试时接受该小时已发送、被去重的同通道通知并记入
notes；恢复判定以恢复前的成功时间为基线，要求之后出现新的成功时间。

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
读取失败可能被当成零卡、EFA不支持的错误计数可能报零、disk await口径（已按标准await修正）、必需挂载/
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
（CPU/内存/page cache 负载、小容量测试 cgroup 的自身 OOM（`memory-start
--cgroup-oom-bytes` 与 `memory-status`）、oracle 采样、采集单元与 exporter 的限时
SIGSTOP（采集单元的暂停同时装载一个 `/run/systemd/system/<unit>.d/` drop-in，用
`ExecStartPost` 再次停住被产品自身 TRIGGER_HEALTH_SNAPSHOT 重启出来的新主进程；
`unit-resume` 与 deadman 先移除并 reload 再 SIGCONT，遗留的 drop-in 或已装载的钩子
计为残留）、节点日志采集器的 runtime 启用；三条暂停命令 `unit-pause`/`exporter-pause`/
`log-collector-enable` 本身位于第二个节点侧文件 `probes/general_live_node_pause.py`
——探针是单文件装入节点，运行器在每次运行首次暂停前经探针的 `install-source` 以
sha256 校验装入该模块、探针按路径导入；恢复、`residue` 与 `cleanup` 仍在探针内，
deadman 只需探针本身）、`probes/general_live_storage_probe.py`
（loop 文件系统与 loop+device-mapper 测试设备、O_DIRECT 写入）、
`probes/general_live_net_probe.py`（host-local veth/dummy 测试接口、丢包生成器、
009 `tcp_only` 的测试 netns+netem TCP 夹具）、`probes/general_live_deps_probe.py`
（单次运行的临时测试包：仅在缺失时用 dnf 安装经审查的包、加载 `sch_netem`，
并只撤销本次运行装/载的内容）、`probes/general_live_shared_probe.py`（005 的 NFSv4
测试挂载：宿主命名空间 soft 挂载、有界 statvfs/I/O/填充读数、级联卸载与 deadman；服务端
Pod 由 `shared_export_fixture.py` 提供）。采集单元清单的只读扩展复用
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
`HOST_MEMORY_PRESSURE`（主机内存压力，以及 002 `cgroup_only` 的小容量自身 cgroup
OOM）、`KERNEL_MODULE_LOAD`（加载 dm_delay/
veth/dummy/sch_netem，以及 004 `probe_error` 的 `vfat`、005 的 `nfsv4`；dm-error 目标内建于
dm_mod，无需加载）、`ENABLE_NODE_LOG_COLLECTOR`（临时启用被禁用的节点日志
采集器）、`WATCHER_NETWORK_ISOLATION`（028 `api_unreachable`：在 completion watcher Pod
所在节点，仅以该 Pod IP 为源的 FORWARD REJECT 切断其到控制面端点的 TCP，deadman 不超过
600 s 覆盖期限）、`NODE_AGENT_NETWORK_ISOLATION`（011 `unreachable_node`：在第二节点
INPUT REJECT 其 Node Agent 监听端口——执行器下发节点动作的通道，心跳/采集器/kubelet 不受
影响，带 deadman）、`CONTROL_PLANE_RESTART`（029 `worker_restart`：在 restart claim 打开期间
按 UID/resourceVersion 栅栏删除唯一的 completion watcher Pod，由 Deployment 重建）、
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
不能自证干净即整项 FAIL。受管测试任务的工作负载名为 manifest 名加 job id 的
sha256 前 10 位（不超过 43 字符，产品 `-r-<8位>` 重启副本与训练 operator 的
`-master-0` 副本名仍是合法 DNS-1035 名称），状态读取、Pod/Service 选择和清理都用
该名；清理按重启血缘（本夹具的 owner nonce 与 job id、由本 attempt 派生的 attempt、
链到本夹具创建的 UID 或 RESTART_WORKLOAD 副本）一并删除产品原地重启或复制出的
工作负载并记入日志，清理记录列出删除的每个对象；工作负载与预热 Pod 两部分清理
总是都执行。创建前名称或 job id 已被占用时拒绝，并逐个列出占用者（kind/name、
UID、job id、attempt、owner、状态，是否为其他验收运行的遗留），绝不自动删除。

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
为 `NOT_RUN(reason)`；站点前提缺失为 `BLOCKED`。夹具自身的失败——`HostProbeError`
（探针调用失败或超时）与 `RegionalFixtureError`（如 setup 单元未完成）——意味着条件
从未建立、产品未被判定，记 `INCONCLUSIVE`，`detail` 为 `harness fixture failed: <类>:
<探针诊断>`，`error` 保留原文；不记 `FAIL`，否则 expected FAIL 变体会把夹具失败读成
其登记的产品缺口（2026-10-25 第二台 EKS 节点上的 `reference_iostat`：`dm-create`
`TimeoutExpired`，看板上是一条空备注的 FAIL）。夹具失败之后清理仍未能证明节点干净的，
按清理失败记 `FAIL`。

### phase 1 范围与前提

首批 native 变体：001 CPU 过载、003 低利用率/资源异常、004 本地盘满、
006 磁盘 I/O、008 网卡链路 down、009 网卡错误/丢包、025 DCGM 采集不完整
（native：exporter_timeout/sustained_unknown/post_refresh_success）、
028 进程失败/终态、029 重启预算、031 采集器静默。延后变体留在分母并记
`NOT_RUN`/`BLOCKED`：需产品侧读取夹具的 live-replay（004 probe_error、
008 statistics_failed、009 errors_delta、025 的 5 个字段/别名/NaN 变体；
029 hardware_chain_continues 已由下文的硬件链变体实现）、
本站能力缺失（008 bond_vf、031 inactive_cluster/notification_failure）。028
`api_unreachable` 与 029 `worker_restart` 原为未授批危险分支，现已实现为带令牌的
native 变体（见下文"watcher 失联与重建"）。

需管理员安排的站点配置：008 `empty_required_list` 是站点 FAIL，修复需把主/EFA
netdev 写进安装器/管理配置的 `GPU_FAULT_REQUIRED_INTERFACES`（默认空）；
031 `logs_silent` 需按批准启用 `GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR`
（生产默认 false）；002/003 的主机内存压力需 `HOST_MEMORY_PRESSURE` 审批；
006 `sustained_latency`、008/009 的 veth/dummy/netem 需 `KERNEL_MODULE_LOAD`
或预装内核模块。运行器只做只读 `--plan` 预检，`--execute` 由运维在授权窗口执行。

029 `concurrent_claim`/`missing_context`（native，store 层）：不起受管任务，而是由
`general_live_restart_budget.py` 的 `BUDGET_PROBE` 在 API Pod 内（`cpu_python`，写入调用
`attempts=1` 不重试）用产品自身的接口操作一个**测试 job id**（`gl29cc-…`/`gl29mc-…`，由
run id 与 attempt 派生）的重启预算账本。`concurrent_claim`：4 个 claimer 经 `Barrier` 同时对
同一个 RESTART_WORKLOAD reservation 调用 `store.reserve_job_restart`（预算 1），再让另一步的
reservation 同时争抢最后一个名额，再并发 release 同一 reservation，最后 release 从未持有的
id；判定只读产品账本：恰好一个 reservation 且 `restart_count` 等于 reservation 数、争抢者全部
被拒、release 恰好释放一次、未持有的 release 不改变账本、计数从不为负、不创建任何训练任务。
`missing_context`：把一个只存在于内存、从不保存的 incident/workflow（RESTART_WORKLOAD 步骤缺
`source_gpu_count`）交给产品的 claim 预检 `reserve_restart_budgets` 与适配器侧门禁
`issue_restart_authorization`：预检须以 `RESTART_SAFETY_CONTEXT_MISSING` 拒绝并点名缺失参数，
拒绝前后账本行都不存在（预算读写之前即拒绝）；补全上下文但无 reservation 的步骤在门禁
fail-closed（`RESTART_RESERVATION_MISSING`）；对照组：同一步骤补全上下文后被预留并获授权。
两者都不 POST finding/workflow/命令。清理先经产品 API 释放本次运行的 reservation，再用 store
自身的 `_delete` 原语删除已清空的测试行（产品不提供账本删除接口；与
`command_protocol_probes.cleanup_audit_records` 处理 `restart_budget` 行的做法相同）；仍持有
他人 reservation 的行保留并记为清理失败，非 `gl29` 前缀的 job id 一律拒绝，已有账本行时记
`NOT_RUN(FIXTURE_NOT_READY)` 并由清理移除本次运行的残留。标志 `P`（仅 lane 锁），无需
`--approve`；`--fault-node` 仍是预检要求，但变体本身不触碰节点。

029 `hardware_chain_continues`（`general_live_attempt_hardware.py`；注入为 live-replay，动作为产品
原生节点动作；需 `--approve GPU_RESET --approve KERNEL_SOURCE_REPLAY`，标志 `X`，risk H，两条
lane）：预算耗尽时只扣住 RESTART_WORKLOAD、既定 STOP/硬件恢复/验证仍有实际回执的实证。步骤
（遵守受管任务约定：任务的 `kubectl create` 是变体的第一个集群变更，此前不读任何 store 记录，见
`tests/regional/test_general_live_managed_job_start.py`）：（1）先登记账本清理（LIFO 中位于任务
之后释放），以派生的 `gl29hc-…` 测试 job id 启动受管单节点训练任务（真实 CUDA 负载与心跳、
`restart_budget` 1），等到 Running，使内核 XID 遇到节点上唯一的活动 attempt；（2）只读 GPU
survey 选定最高索引 GPU 及其 PCI BDF，随后 `ExhaustedRestartBudget`（复用 store 层变体的
`SeededRestartBudget`：同一测试 job id、同一清理）经产品自身的 `store.reserve_job_restart`
（claim 预检所用的同一调用）把该 job 的账本预留到 `restart_count == budget`（预算 1），再多预留
一次并要求产品拒绝——耗尽由产品 API 自证，不直接改计数；（3）经 lane K 的 kmsg 内核源回放（`inject_kernel_xid` 配本模块的 reset-grade 目录门禁 `require_reset_grade`）
写入一条审阅过的 `NVRM: Xid (PCI:<bdf>): 46, …` 行，点名最高索引 GPU 的 PCI BDF：XID 46
`ROBUST_CHANNEL_GPU_TIMEOUT_ERROR`（`src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml`
第 462-473 行：`immediateAction: RESET_GPU`、产品族 A100/H100/B100/GB200、无 xid154Linkage；
引擎直接映射为 RESET_GPU、GPU containment、无 pre-action；GF-REGIONAL-DESTR-001 在本硬件上实证
复位链所用的同一代码；行格式即 lane K 已实跑的 `xid_message`，不自创载荷）。产品侧全程真实：
内核采集器解析 → 目录判定 EXECUTABLE RESET_GPU → 按节点唯一活动 attempt 补全 workload →
编译 FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → STOP_WORKLOADS → QUIESCE_GPU_SERVICES →
VERIFY_NO_GPU_CLIENTS → RESET_GPU → RESTORE_GPU_SERVICES → VALIDATE_GPU → RESTORE_SCHEDULING →
RESTART_WORKLOAD；claim 预检发现预算耗尽，只把 RESTART_WORKLOAD 记为
FAILED(`RESTART_BUDGET_EXHAUSTED`，1/1，附一次通知）并 supersede 该步，其余步骤照常执行，
工作流以预算原因结束 FAILED、incident ESCALATED（节点已由 RESTORE_SCHEDULING 解除 cordon）。
决定一出即关闭 kmsg 窗口（窗口内真实 /dev/kmsg 不被读取）。判定只读产品记录：决定为
KERNEL_LOG、RESET_GPU、目标 GPU、唯一且非重复；STOP_WORKLOADS/RESET_GPU/VALIDATE_GPU 必须
SUCCEEDED，计划内的 MARK_UNSCHEDULABLE/QUIESCE/VERIFY/RESTORE_GPU_SERVICES/RESTORE_SCHEDULING
亦须 SUCCEEDED，任何工作流不得出现 RESET_ALL_GPUS_NVSWITCHES/RESTART_NODE/REPLACE_NODE/
QUARANTINE；RESTART_WORKLOAD 恰有一条 FAILED 记录、原因 `RESTART_BUDGET_EXHAUSTED`、计数 1/1、
点名通知、索引已 supersede，远端命令中没有 RESTART_WORKLOAD；与 GF-REGIONAL-ISO-007（软件
XID 11 + 预算 0、整条工作流只因预算 FAILED、从未实跑）的差异显式断言：FAILED 步骤集合恰为
{RESTART_WORKLOAD} 且 RESET_GPU 确有执行记录；该 XID 只有一个 incident、终态 ESCALATED 且节点无
cordon/归属注解/taint（QUARANTINED 或仍隔离即 FAIL，并经产品验证式恢复交还节点）；该 incident
恰有一条 `restart-budget-exhausted` 通知且为扣住记录所点名；attempt 终态 STOPPED（由 incident
发起的停止）、completion 决定 NO_ACTION、无被动 containment incident、账本仍为播种的 1/1 且
reservation 未变（扣住的 restart 不预留）；RESET_GPU 回执之后有新的 GPU_METRICS batch 且无活动
finding；复位若卸载了空闲显示栈必须已装回（RESET_GPU 步骤的 `display_stack_restored`；
`gpu_reset_busy_attempts` 记入结果）。硬件链被跳过或 RESTART_WORKLOAD 实跑为 FAIL；夹具未收到行
或行未成为 KERNEL_LOG 事件为 INCONCLUSIVE；XID 非 reset-grade 在任何变更前记 NOT_RUN，节点无
PCI BDF、已有账本行等在注入故障前记 NOT_RUN（此时测试任务已启动，由清理栈删除）。清理为 LIFO：kmsg 窗口（已提前关闭，自带 `systemd-run --on-active`
deadman）→ 删除测试任务与预热 Pod → 经产品 API 释放播种的 reservation 并删除清空的测试行（行
未证明消失即抛错，整项 FAIL）。账本行与任务是控制面/集群侧测试对象：运行器中途死亡只会留下一行
只扣住该测试 job 的 `gl29hc-…` 账本，下次同 attempt 重跑拒绝（NOT_RUN）并由其清理移除，与 store
层变体同一残留规则。现场前提：目标节点空闲（无 GPU Pod）、`--site-file`、维护窗口（链路期间
节点被 cordon，故标志 `X`）；HyperPod 另需 `NodeRecovery=None`。计划/执行形如：

```bash
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-029 --variant hardware_chain_continues --fault-node <node> \
  --site-file <site.yaml> --approve GPU_RESET --approve KERNEL_SOURCE_REPLAY --plan
run_general_live.py --site-profile <profile> --run-dir <run> \
  --case GF-GENERAL-LIVE-029 --variant hardware_chain_continues --fault-node <node> \
  --site-file <site.yaml> --approve GPU_RESET --approve KERNEL_SOURCE_REPLAY --execute \
  --confirm GENERALLIVE029_EXECUTE --maintenance-window-end <UTC>
```

watcher 失联与重建（028 `api_unreachable`、029 `worker_restart`，native，标志 `X`，
`general_live_network_isolation.py`/`general_live_watcher_restart.py`，节点侧探针
`probes/general_live_isolation_probe.py`）。`api_unreachable`（`--approve
WATCHER_NETWORK_ISOLATION`）：先确认恰好一个 Ready 的 `gpu-fault-completion-watcher` Pod
（非 hostNetwork，有自己的 Pod IP），从 `gpu-fault-regional-connection` 只读
`control-plane-url` 一个键解析端点；受管任务以 `trigger:1` 模式启动（容器等待运行器创建
`/tmp/gfrs-live-exit` 后非零退出），等到全部 Pod Running 且 watcher 已向控制面发布该 attempt
的 RUNNING 观测后，在 watcher 所在节点经探针写入意图记录、布置 `systemd-run --on-active`
deadman（探针自身副本执行 `cleanup --strict`，失败重试）、再在 FORWARD 链顶部插入
`-s <Pod IP>/32 -d <解析地址>/32 -p tcp --dport <端口> -m comment --comment gfrs-<digest>-iso-pod-egress
-j REJECT --reject-with tcp-reset`（其他 Pod、宿主机进程、其他目的地址一律不受影响；deadman
≤ 540 s，低于控制面 600 s 覆盖期限），随后 `kubectl exec` 触发退出。阻断期内 watcher 的
紧急保底（`GPU_FAULT_PASSIVE_STOP_FALLBACK_SECONDS`）须自行 suspend 工作负载并写上
`gpu-fault.io/passive-stop-mode=emergency-fallback` 与 termination-initiator 注解，attempt
的 Pod 被 operator 删除后终态在 watcher 侧产生并缓冲于 outbox；有效性 oracle：阻断至少 150 s
后，控制面的 watcher coverage heartbeat `observed_at` 仍早于阻断时刻，且该 attempt 无
event/decision，否则 `INCONCLUSIVE`（阻断未生效）。恢复后判定只读产品记录：紧急停止注解
指向该 attempt 的 containment incident；event/decision 各恰好一条且 decision 为
`PLAN_CREATED`（被动 containment 发起的 STOPPED 终态仍按预算重启一次）；containment
工作流 STOP_WORKLOADS 恰好成功一次、recovery 工作流至多一个且 RESTART_WORKLOAD 至多成功一次、
账本 `restart_count ≤ 1` 且等于 reservation 数；该 attempt 各 incident 的通知无重复
deduplication key；恢复后出现新的 coverage heartbeat（`observed_at` 晚于恢复时刻）。清理删除
带本次 tag 的全部规则并停 deadman，探针回读链上无残留且 timer 未激活，否则记清理失败。
`worker_restart`（`--approve CONTROL_PLANE_RESTART`）：受管任务 `exit:1`、预算 1；每 5 s
读取 attempt 记录直到 claim 打开——账本已有 recovery 工作流的 reservation 且 RESTART_WORKLOAD
尚无 SUCCEEDED/FAILED 执行（适配器已提交副本、处于 WAITING）；首次读到即已关闭记
`INCONCLUSIVE`。随后以 `DeleteOptions.preconditions{uid,resourceVersion}`、grace 0 删除唯一的
watcher Pod，等待 Deployment 重建出新 UID 且 Ready（≤ 600 s，与 startupProbe 一致），再等
recovery 工作流终结。判定：账本 `restart_count == 1` 且唯一 reservation 属于该 recovery
工作流（未丢失、未重复扣减）；RESTART_WORKLOAD 恰好成功一次、recovery 工作流恰好一个且
SUCCEEDED、STOP_WORKLOADS 恰好成功一次、decision 为 `PLAN_CREATED`；watcher 发布的观测中除源
attempt 外恰好一个重启 attempt；通知无重复；重建后出现新的 coverage heartbeat。两变体都在
`eks`/`hp` 车道、需 `--fault-node`（跑受管任务的 GPU 节点）与 `--site-file`；清理按重启血缘删
除受管任务及其副本。

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

006 磁盘 I/O（native）：存储探针用 loop + device-mapper 测试设备（`dm-N`，即
`/proc/diskstats` 与产品上报的名字）和 `systemd-run` O_DIRECT 写入单元。2026-10-25
两台 plain EKS 节点上所有 006 变体仍在产品证据之前失败（`host probe failed (exit 1;
error TimeoutExpired)`，开始约 290 s，清理栈为空）：错误类来自探针自身 JSON，超时的是
探针内 `dm-create` 的 `dmsetup create`（探针默认 180 s 命令超时），不是运行器的
`kubectl exec`。libdevmapper 的 create/resume/remove 会等主机 systemd-udevd 确认一个
通知 cookie（SysV 信号量，由 `95-dm-notify.rules` 的 `dmsetup udevcomplete $DM_COOKIE`
完成），而 SysV IPC 按 IPC 命名空间隔离；探针 Pod 是 hostPID/hostNetwork 但有自己的
IPC 命名空间，udev 在主机命名空间里找不到 Pod 的信号量，create 一直等下去（004 的 loop
文件系统不调用 `dmsetup`，所以正常）。两节点 journal 已证实这一机制：Pod 内每次
`dm-create` 之后几秒，`(udev-worker)` 记录 `dm-N: Process '/usr/sbin/dmsetup
udevcomplete <cookie>' failed with exit code 1`（ip-10-0-43-5 07:40/07:45/07:51Z，
ip-10-0-32-162 07:38/07:44/07:49Z，各对应一次 loop 附加），随后每个运行的 deadman
`cleanup --kind dm-io` 把映射、loop 与后端文件清掉（`clean: true`）。早先
`sustained_latency` 的 `ProbeError` 符合另一个缺口：变体要求 `KERNEL_MODULE_LOAD`，但
没有任何步骤加载 `dm_delay`，探针因此拒绝 delay 表。两者都与 `dm_delay` 是否已加载
无关，故撤销 `sustained_util`/`sustained_latency` 的 `NOT_RUN(FIXTURE_NOT_READY)`。

现在设备完全在主机命名空间中构建：`dm-create` 先布置 deadman、写记录，只启动一个有界
瞬态 setup 单元（`RuntimeMaxSec`；`fallocate`、`losetup`、`dmsetup create`、
`mkfs.ext4`、`mount`，每步执行前先记录）即返回；运行器轮询 `dm-status`，直到记录为
`ready`，或 `failed`（带步骤、错误类、摘要和探针位置）。设备的其他 `dmsetup`、
`mount`、`umount` 都经 `nsenter` 进入 PID 1 的 IPC 与挂载命名空间（survey 缺 `nsenter`
或 `systemd-run` 记 `NOT_RUN(CAPABILITY_ABSENT)`）。每个 006 探针调用有墙钟预算，探针把
每条命令的超时截到剩余预算之内，运行器的 `kubectl exec` 超时为该预算加 60 s 传输余量，
调用总能先返回带自身错误的 JSON。每个撤销都在对应调用之前压入 LIFO 栈；设备撤销是探针的
`cleanup --kind dm-io`（也是 deadman 的命令）：停 setup 与写入单元、删除映射、只摘除以
本次运行自身文件为后端的 loop、卸载本次运行加载的 `dm_delay`（`--load-module`，加载前
先记录），残留不为空即清理失败。`counter_reset` 先停写入单元再删除并重建映射（写入单元
打开的文件使挂载处于忙状态），之后重新写入。探针失败现在显示
`error <类> <摘要> at <函数:行>`，不再只有类名。修复进入现场代码后首次真机运行
（2026-10-25）：`sustained_util`、`reference_iostat`、`counter_reset`（ip-10-0-43-5）
与 `short_flush`、`zero_completed_io`、`recovery`（ip-10-0-32-162）PASS，setup 单元
约 14 s 建成 `dm-0`；`sustained_latency` 已带 `KERNEL_MODULE_LOAD` 登记为可执行，
等待一次无未关闭 incident 的窗口。该窗口 2026-10-25 14:47Z（R13e）的首次真机运行在恢复阶段
FAIL（"no fresh sample below threshold"）：`io-stop` 之后 dm-delay 设备再无完成的 I/O
（oracle 从 14:53:55Z 到 15:05Z 的每个 2 s 样本 util 0、await 空），产品按零完成 I/O 规则
不产出 `disk_io_await_ms` 样本，信号只能停在 active；而 200 ms 的 delay 表仍在，任何 I/O 的
await 都 ≥ 200 ms——夹具的结构性缺口，不是产品缺口，也与 300 s 健康摘要节拍无关（等待
690 s ≥ 两个摘要周期），设备也是等待结束后才拆除。现在 await 的恢复证明先经 `dm-retarget
--target data --delay-ms 0` 把数据表原地换回 linear（同一 dm-N、挂载与文件不变），再启动
单线程、每 100 ms 一次 64 KiB O_DIRECT 写的节拍写入单元（util 远低于 1 %），让产品在同一
设备上采到阈值以下的新鲜 await 样本，等待结束后停掉它；util 变体无需这些（空闲设备的
field 10 读数即 0 %）。`reference_iostat` 另可选 `--approve KERNEL_MODULE_LOAD`：
测试设备用 200 ms dm-delay 表（`dm_delay` 已加载，或为本次运行加载并在清理卸载），把
await 抬到 5 ms 容差底线之上——无 dm-delay 的线性设备在 NVMe 上 await 不到 1 ms，任何
口径差异都显不出来（2026-10-25 ip-10-0-43-5：产品 0.30–0.44 ms 对 oracle 0.33–0.76 ms，
expected FAIL 变体却 PASS），这种底线之下的读数以及无样本可比的运行现在记 `INCONCLUSIVE`。
`cleanup`/`dm-destroy`/`fs-destroy` 现在也删除本次
运行已空的后端目录与挂载根目录（`residue.directories` 列出仍存在者）：之前只有无人
调用的 `--kind all` 会删，两节点各留下数十个空的 `gfrs-live-<digest>` 目录。

### phase 2 范围（host/device 扩展）

第二批补充 host/device 类别 002 内存/交换、005 共享存储、007 SMART、
010 EFA/RDMA 计数、012 GPU 掉卡、015 BMC（模块
`general_live_memory`/`general_live_shared`/`general_live_devices`（010/012）/
`general_live_smart`（007）/`general_live_bmc`（015），每个必测变体都在注册表里）。
可在本站真机执行的是不需要原始源回放夹具的变体，以及已在共享回放器上接线的 007/015：

- 002 `memory_pressure`、`recovery`（native，复用节点探针 `memory-start` 的受限
  cgroup 压力和 meminfo oracle，需 `HOST_MEMORY_PRESSURE`；`memory_used_percent`
  ≥95% 持续 120 秒 → VALIDATE_HOST 探针在加载期间失败，无 reset/reboot）；
  `no_swap`（observation：确认 `SwapTotal=0` 时 swap 规则天然不触发）。
  2026-10-25 两台 plain EKS 节点上 `memory_pressure` 三次在产品证据之前失败
  （`host probe failed (exit 1; error ProbeError)`，survey 与 oracle 均正常）：运行器按
  2 TiB 主机 14 分钟 ramp 预算请求 1980 秒的 hold（ramp 1080 + 检测 540 + 余量），而
  探针 `memory-start` 当时的上限是 1800 秒，`bounded()` 直接拒绝，错误文本按策略不进日志。
  现在该上限与其他负载单元一致为 3600 秒（deadman 仍为 hold+120 ≤ 5400），运行器在触碰
  节点之前先按探针上限检查请求的秒数（超出即带数字失败），节点探针的失败 JSON 带
  `error_code`/`error_site`，夹具证据显示 `error <类> <摘要> at <函数:行>`
  （如 `memory_start:<行>`）。以上仅改测试夹具，尚未真机复跑。
- 002 `cgroup_only`（native 负例，复用 `HOST_MEMORY_PRESSURE`）：节点探针
  `memory-start --cgroup-oom-bytes 2147483648` 把分配器放进一个上限 2 GiB（探针拒绝超过
  MemTotal 的 1 % 或超过主机在 95 % 规则下余量的上限）、无 swap、`OOMPolicy=kill` 的测试
  cgroup，分配器在 cgroup 内被内核 OOM killer 杀死；该单元不 `--collect`，`memory-status`
  读 systemd 的 `Result=oom-kill`/`ExecMainStatus=9` 作为“OOM 发生在 cgroup 内”的证据，随后
  `memory-stop` 复位单元并撤销 deadman。在长于 120 秒持续窗口加一个 health summary 的窗口内
  重复 3 次；meminfo oracle 读到的主机 `memory_used_percent` 须全程低于 95%（越过即
  `INCONCLUSIVE`：主机本身有压力），窗口内产品须至少接收一个主机内存样本（否则
  `INCONCLUSIVE`：静默通道无法证明判定）。PASS 要求无 `memory_used_percent`/
  `memory_available_percent` finding、无硬件步骤、无 STOP/cordon/隔离、节点仍可调度；若节点
  日志采集器被启用而把 OOM 行判为 finding，只接受 RUN_DIAGNOSTICS 诊断（`log_rules.py`），
  任何 STOP 或隔离为 FAIL。
- 005 全部七个变体（native）：自带的测试 NFSv4 export 夹具（`shared_export_fixture.py`
  服务端 Pod + `probes/general_live_shared_probe.py` 客户端挂载），需
  `KERNEL_MODULE_LOAD`；七个变体均预期 PASS（`mount_disappears`/`io_only_failure` 由批次 176、
  `shared_failure_domain` 由批次 177 关闭缺口），详见 GF-GENERAL-LIVE-005 节。
- 007 `healthy`（observation：只读确认真实 SMART 判定健康且不产生存储 finding，
  作为失败变体的对照基线）。
- 007 SMART 的回放变体 `health_failed`/`query_failed`/`scan_failed`/`malformed_json`/
  `tool_missing`/`unsupported_device`（live-replay，模块 `general_live_smart`，两条车道，需
  `HOST_SOURCE_REPLAY`）：在共享回放器的命令包装模式下只改写产品真实 `smartctl` 调用的一个
  经审查字段——`-H -j <dev>` 的 `smart_status.passed=false`（health_failed）、删除
  `smart_status` 对象（unsupported_device）、整段无判定 JSON + 退出码 2（query_failed）、
  截断 JSON（malformed_json）、`--scan-open` 退出码 1（scan_failed），或只在单元内把
  `smartctl` 设为不可访问（tool_missing）。目标盘取产品自己最新 `smart_health_failed`
  读数的设备（只读产品记录；无判定盘即 `NOT_RUN(CAPABILITY_ABSENT)`）。判定只读产品记录：
  `health_failed` 走 `smart_health_failed=1` → MARK_UNSCHEDULABLE + QUARANTINE（无 STOP/
  硬件步骤）→ 关窗后真实工具重新判健康（recovered）→ 受控恢复；`query_failed`/
  `malformed_json`/`scan_failed` 须是 `smart_query_failed`（`failure_mode`
  QUERY_FAILED/SCAN_FAILED）加 `_smart` 采集错误、无该盘判定、其余盘判定保留、无 finding
  无隔离，`scan_failed` 另禁止 `NO_DEVICES`（规格点名的 scan 退出码缺口）；`tool_missing`/
  `unsupported_device` 须是 `smart_status_unsupported`（TOOL_MISSING/NO_SMART_STATUS）且无
  采集错误、不记健康盘。包装器日志（`audit-read`）是“产品确实问过且规则已生效”的 oracle，
  否则 `INCONCLUSIVE`；每个变体关窗后都要读到真实 `smartctl` 的新鲜判定且无 `_smart` 错误。
- 015 BMC 的回放变体 `critical`/`nonrecoverable`/`wrong_column`/`unavailable_sensor`/
  `command_failure`（live-replay，模块 `general_live_bmc`，两条车道，需 `HOST_SOURCE_REPLAY`；
  `install-wrapper --allow-missing-real`：EC2 无 BMC 也无 `ipmitool`，shim 只按计划回答
  `ipmitool sensor` 的整段输出/退出码，其他调用一律 command not found，绝不编造 BMC）：
  `critical`（两行 cr）/`nonrecoverable`（一行 nr）→ `bmc_critical_sensor` = 处于 cr/nr 的
  传感器数（同名重复行只计一次）→ MARK_UNSCHEDULABLE + QUARANTINE 一条 incident，随后同一
  包装器改供健康表、产品读到 0（recovered），再关窗、受控恢复。`sensor_identity` 阶段要求每个
  告警传感器的名称出现在三处：样本标签 `sensors`（`sensor_alerts` 另带读数、单位、状态和
  ipmitool 打印的阈值列）、该 finding 所开 incident 的 `reasons`（即 QUARANTINE 步骤硬件升级
  邮件所列文字，形如 `BMC reports 2 critical sensors: CPU1 Temp=96.000 degrees C [cr; unc
  90.000, ucr 95.000, unr 100.000]; Fan1=...`）和该 incident marker 的 `raw_reason`；
  旧的裸计数在三处都不命名传感器，即为 FAIL。`wrong_column`（cr/nr 出现在名称/值/单位列、
  三列残行）与 `unavailable_sensor`（一行 na/na）须是 `bmc_critical_sensor=0`、无标签、无采集
  错误、无 finding、无隔离（全部行均为 na 的表在产品里是 UNKNOWN——`_bmc` 采集错误且无样本，
  软件测试覆盖）。`command_failure` 分两段：先让 `ipmitool sensor` 以 BMC 无应答的方式退出
  （退出码 1 “Get Device ID command failed”）→ 须是含 “BMC sensor query failed” 的 `_bmc`
  采集错误、无样本、无 `bmc_status_unsupported`、无 finding；再 replan 为无 IPMI 设备节点
  的退出（退出码 1 “Could not open device at /dev/ipmi0 ...: No such file or directory”，
  即 EC2 上已安装 `ipmitool` 的真实输出）→ 须读到 `bmc_status_unsupported`（`reason=NO_DEVICE`）
  且采集错误停止、仍无样本无 finding。`unsupported_bmc` 是只读 observation（如 002
  `no_swap`）：记录 `ipmitool` 是否安装，等待一个 health summary 周期读产品自己的能力标记——
  无 `ipmitool` 须为 `bmc_status_unsupported` TOOL_MISSING，有 `ipmitool` 无 BMC 须为
  NO_DEVICE——且无 BMC 样本、无 `_bmc` 采集错误、无 finding；缺标记或仍每 tick `_bmc`
  采集错误记 FAIL（永久能力缺口不是采集错误）；若节点真的有 BMC 应答则 BLOCKED（该变体
  不适用）。`healthy` 仍 `NOT_RUN(CAPABILITY_ABSENT)`。

- 010 EFA/RDMA 计数（模块 `general_live_efa`，复用夹具 `general_live_efa_fixture`，8 个变体
  全部 `live-replay`，需 `HOST_SOURCE_REPLAY`，eks/hp 两车道）。每个变体先用共享回放器的
  只读 `inventory` 读取节点真实 RDMA 树——驱动身份、所属 netdev、每端口 `state`/`phys_state`、
  `counters`/`hw_counters` 文件名与值、真实 `ethtool -S` 计数器名——写入证据，再只替换一个读取：
  `rdma_error` 把该端口一个真实计数器文件（优先 `hw_counters/rx_drops`）以真实值 bind 进
  host collector 单元后加 5000，产品记录须出现同端口 `rdma_errors_delta` 与
  `rdma_<counter>_delta` ≥ 5000（计数器在产品读数里确实增长才算正向）→ WARNING
  RUN_DIAGNOSTICS → FREEZE_EVIDENCE + VALIDATE_FABRIC 实际执行（失败须点名该指标）、端口仍被
  观察、无硬件动作/隔离；`link_inactive` bind 端口 `state` 为 `1: DOWN`（phys_state 真实），且只
  持有一个**有界曝光期**——窗口自己的 deadman 在开窗后 1.7 个采集周期（15 s 周期为 26 s，
  `general_live_source_replay.bounded_exposure_seconds`）关窗：第一拍出 `rdma_link_down` WARNING，
  第二拍是其 VALIDATE_FABRIC 判定的新鲜读数（在持有的端口上 FAILED，并点名该指标）同时确认共现的
  `efa_inventory_mismatch` `LINK_INACTIVE`，第三拍已读回真实树。批次 175 起该共现项的诊断按新鲜读数
  判定、对仍不活动的端口按设计升级 REBOOT_NODE，故曝光期把它只留给已恢复的端口：其 VALIDATE_FABRIC
  须 SUCCEEDED、incident RECOVERED、不得等待 `missing_inventory_action_completion`、无 RESTART_NODE；
  产品在恢复后才判定链路（SUCCEEDED 时间晚于恢复）、恢复后仍有不活动读数（上界未守住）或窗口内无
  `rdma_link_down` 样本记 INCONCLUSIVE；关窗后须有新的 `rdma_link_down=0` 读数且 incident 收敛；
  `rnr`/`retry`/`cq_error` 让经审查的 `ethtool`
  透传包装器对 EFA 设备所属 netdev 的 `-S` 输出先以 0 再以 5000 提供一个带对应 token 的
  计数器（真实同 token 计数器优先，否则测试名 `gfrs_<token>_err`），
  `efa_<group>_errors_delta` → WARNING → 只诊断、不推导坏卡；`missing_counter` 先提供后撤回
  该计数器（真实名用 `delete_counter`，测试名改为透传），产品须记
  `efa_ethtool_counters_unsupported` `MISSING_COUNTER` 点名该指标且绝不报零 delta；
  `ethtool_failed` 让 `-S <netdev>` 以 exit 1（I/O error）返回，产品须记 `_efa_network`
  采集错误（UNKNOWN）、无零 delta、不误判 NO_STATISTICS、无 finding，包装器审计日志证明产品
  确实发出该查询并得到该失败；`counter_reset` 在同一窗口让 sysfs 计数器与 ethtool 计数器每半个
  采集周期递增直到各自出 WARNING，再归 1 继续递增：无负 delta、每个读取器只有一个 incident
  （重置不是新故障）、重置那一拍记 UNKNOWN 而不是会清除信号的 0 读数。批次 173 后三个计数读取器
  共用一条重置规则（`collectors/host/network.py` `_monotonic_delta`/`_reset_problem`），`counter_reset`
  预期 PASS 并据此判定：每个读取器一条采集错误点名读取器（`_rdma`/`_efa_network`）、端口或 netdev、
  下降的计数器名与 `delta unknown`，且该拍的 batch 里没有该计数器 delta 及其聚合
  （`rdma_errors_delta`/`rdma_<counter>_delta`/`efa_<group>_errors_delta`）的任何读数，之后从新基线
  续测；旧的夹成 0 读数即 FAIL。009 `counter_reset`（重建丢包接口）同样要求 `_network` 采集错误点名
  接口、计数器与 `delta unknown`、该拍无 `network_drops_delta` 读数、无 finding；接口重建跨越了一拍
  致基线被清（既无错误也无读数）记 INCONCLUSIVE。若节点上没有任何 EFA 功能拥有 netdev，夹具树给第一个 EFA 设备附上节点一个真实
  物理 netdev（`build-infiniband-fixture --netdev`，netdev 本身不改动）以便产品的 ethtool
  路径有读数可回放，证据记 `netdev_source: fixture`。清理关窗后用只读 `read-source` 核对
  单元与 PID 1 读到同一真实 sysfs 值且不等于夹具值。`non_efa` 仍为 CAPABILITY_ABSENT。

其余变体按映射文件留在分母并记 `NOT_RUN`，原因精确可查：

- `FIXTURE_NOT_READY`：本节已无此类变体。007/010/015 的命令/文件回放与 002/012 的
  `/proc/meminfo`、`nvidia-smi`、`/proc/driver/nvidia` 读取回放都已在协调方集中审查提供的
  共享原始源回放夹具（下文“共享主机原始源回放夹具”）上接线（见上及下文）；005 不走回放：
  其七个变体都在自带的测试 NFSv4 export 夹具上 native 执行（见 GF-GENERAL-LIVE-005 节；
  三个按产品缺口预期 FAIL）。
- `CAPABILITY_ABSENT`：010 `non_efa`（本站只有 EFA RDMA 设备）、015 `healthy`（EC2 无 BMC；
  `unsupported_bmc` 本身就是这条能力记录的只读 observation）。

已在共享回放器上接线的 002/012 变体（均需 `--approve HOST_SOURCE_REPLAY`，两个集群
都运行；证据 `proof_scope: live-replay`，判定只读产品自身记录，包装器日志
`audit-read` 为原始源 oracle）：

- 002 `read_failure`：`InaccessiblePaths=/proc/meminfo` 只在 host collector 单元内生效
  （探针经 `nsenter` 核验单元内不可读、PID 1 可读）。PASS 要求窗口内的 batch 带
  `_memory` collection error 且**没有任何**内存/交换样本（UNKNOWN，而不是健康值），
  无 memory/swap finding、无硬件步骤、无隔离；关窗后新样本恢复。
- 002 `swap_pressure`：`build-meminfo-fixture` 复制节点真实 `/proc/meminfo`，只改写
  `SwapTotal=8 GiB`/`SwapFree=1 GiB`（87.5 %），以 `meminfo` bind 进入单元；不建 swap
  设备。规则无 sustain：首个投递 batch 即产出 `swap_used_percent` WARNING
  RUN_DIAGNOSTICS → FREEZE_EVIDENCE + VALIDATE_HOST（`host_diagnostic_probe` 在
  `swap_used_percent` 上，bind 持续期间必须失败）；`memory_used_percent` 保持真实值，不得
  共振；关窗后 2 个新样本低于 80 %。产品样本值须与夹具计算值一致，否则 INCONCLUSIVE。
- 012 `single_sample`：包装器从**一次** `uuid,utilization.gpu` 查询中删去一个真实 UUID
  （`drop_row`，`max_calls=1`；真实 UUID 取自产品 `host_gpu_utilization_percent` latest
  的 device）。PASS：`gpu_inventory_missing_count=1` 仅一次、`gpu_inventory_mismatch`
  始终 0、无 finding、无 REBOOT_NODE、无隔离，随后恢复满数；日志必须恰好记录一次改写。
- 012 `query_failed`：对该查询连续 4 个 tick 回放 NVML 初始化失败（stdout
  `Failed to initialize NVML...`，exit 9）。PASS：collection error
  `GPU inventory query failed`；`gpu_inventory_expected_count` 带
  `failure_mode=DRIVER_QUERY_FAILED` 与驱动侧 `discovered_count`（真实绑定数）；**没有**
  `gpu_inventory_mismatch` 样本、没有 0 计数、无 finding、无 reboot；规则耗尽后恢复满数。
  日志少于两次失败读取为 INCONCLUSIVE（两次失败是规格的确认窗口）。
- 012 `proc_unreadable`：同一失败查询 + `InaccessiblePaths=/proc/driver/nvidia/gpus`
  （两次读取都失败，规格标注的已知风险）。PASS：collection error 含
  `could not be read` 与 `unknown as well`；`failure_mode=INVENTORY_UNAVAILABLE` 且
  **无** `discovered_count`、无 `gpu_inventory_discovered_count` 样本；其余同上。
- 012 `expectation_wrong`（`risk M`，`S`，仅需 `HOST_SOURCE_REPLAY`，预期 PASS——批次 174 关闭了
  原缺口，不再需要 `NODE_REBOOT`）：在单元环境把 `GPU_FAULT_EXPECTED_GPU_COUNT` 覆盖为产品记录的
  期望数 +1；产品对该覆盖产出 finding（矛盾项，或旧行为的 mismatch）或记录两次后立刻关窗。
  控制面 `host_health.gpu_inventory_verdicts` 先按实例类型清单复核：清单内机型、节点列出的 GPU 数等于
  机型数且驱动绑定数一致时，mismatch 复读为 0，另记 `gpu_expected_count_contradicted` WARNING /
  COLLECT_EVIDENCE。PASS 要求：无 `gpu_inventory_mismatch` incident；恰一条
  `gpu_expected_count_contradicted` incident，`effective_action=COLLECT_EVIDENCE`、工作流只有
  FREEZE_EVIDENCE，reason 写明配置数、实例类型及其 GPU 数、列出数与驱动绑定数；恰一封
  `OPERATOR_REVIEW` 通知；无 RESTART_NODE/隔离；关窗后真实期望数读回。产品从未记录该覆盖记
  INCONCLUSIVE。
- `VENDOR_LAB_ONLY`：012 `confirmed_missing`/`reboot_verified`/`post_reboot_failure`
  需要真实掉卡与已批准的 provider reboot；本 lane 绝不解绑或移除真实 GPU。

现有的 `probes/collector_window_probe.py` 的 nvidia-smi shadow 只匹配库存快照查询
（026 身份路径），不覆盖 host collector 的 `uuid,utilization.gpu` 查询；该查询由下文
共享回放器的命令包装模式（`nvidia-smi` 的 `utilization` 查询）覆盖。

#### 共享主机原始源回放夹具（HOST_SOURCE_REPLAY）

`scripts/e2e/regional/general_live_source_replay.py`（主机探针
`probes/general_live_source_probe.py`，夹具构建器 `probes/general_live_source_builders.py`
——与命令包装器 `tools/host_command_replay.py` 一样，由控制端在首次 `build-*` 前经
`install-source` 以 sha256 校验装入节点、探针按路径导入；探针自身是单文件，开关窗与
deadman 不依赖它）是 007/010/011/012/013/015/002/032 共用的原始源回放器。只改动一个允许列表内的
采集单元 `gpu-fault-host-collector.service` 的私有环境：一个 `/run` systemd drop-in，
由清理和 `systemd-run --on-active` deadman（先于 drop-in 布置）移除；夹具内容以
base64 + sha256 装入 `/run/gfrs-live/source-<digest>/`（tmpfs）；Node Agent 永不在
允许列表内；不替换宿主二进制、不写全局挂载、不写真实 `/proc`/sysfs 文件。五种模式
可在一个窗口里组合：

- **bind**：`BindReadOnlyPaths=<夹具文件>:<源文件>`，目标只能是 `/proc/stat`、
  `/proc/meminfo`、`/proc/driver/nvidia/version`、gfrs 测试网卡的 `statistics/<计数器>`，
  或 EFA 端口的 `/sys/class/infiniband/<dev>/ports/<n>/{hw_counters,counters}/<计数器>`
  与 `state`/`phys_state`。
- **tree**：整棵目录的只读 bind，目标只能是 `/sys/class/infiniband`（013 库存变体
  需要全部设备的端口，单文件 bind 表达不了缺卡/多卡/链路不活跃）或 `/proc/driver/nvidia`
  （012/032）。`build-infiniband-fixture` 复制真实树并施加经审查的变异（丢弃/复制
  设备、端口 `state`/`phys_state`、`device/driver` 身份 none/ambiguous/其它驱动、
  计数器改值/删除、清空端口），驱动目录放在树旁而不在树内；
  `build-nvidia-proc-fixture` 复制 `version` 与 `gpus/<bdf>`，可省略 version、限制
  GPU 数或省略 `gpus`。
- **command**：经审查的透传包装器排在该单元 PATH 首位（`bin/<工具>` shim →
  `python -I host_command_replay.py`），转发到真实二进制，只按**按工具、按查询、按字段**
  的允许列表改写：`smartctl --scan-open`（整段输出/退出码）、`smartctl -H -j <dev>`
  （`smart_status.passed`、`smartctl.exit_status`、删除 `smart_status`、整段输出/退出码）、
  `ipmitool sensor`（整段输出/退出码；EC2 无 BMC 时 `--allow-missing-real`）、
  `ethtool -S <iface>`（单个计数器改值/删除、整段输出/退出码）、host collector 的
  `nvidia-smi` 查询 `uuid,utilization.gpu`（按 UUID 删行、`utilization.gpu` 列）、
  `compute-apps`、`driver_version`、`topo`。规则可限次（`max_calls`）、限时
  （`not_after` 后全部透传）、加有界延迟；真实查询失败时按失败透传，绝不编造成功；
  每次被编目的调用都记入 `audit/<工具>.jsonl`（真实/实际输出的 sha256），由
  `audit-read` 读回作为 oracle。`py-spy` 按名拒绝：它只由 Node Agent 以绝对路径运行。
- **inaccessible**：`InaccessiblePaths=` 只在单元内隐藏允许列表内的路径或工具二进制
  （002 `read_failure`、007 `tool_missing`、012 `proc_unreadable`）。
- **env**：`GPU_FAULT_EXPECTED_GPU_COUNT`/`GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT`
  的 0..64 覆盖（012 `expectation_wrong`）；以及 phase-1 的 **pci-root** 重指。

`open-window` 在重启单元后用 `nsenter` **核验**：每个替换在单元 mount namespace /
进程环境里可见，而在 PID 1 的命名空间里不可见（文件内容 sha256 与所在设备号、
`InaccessiblePaths` 的可读性、PATH 首位与 shim、环境变量），核验不过即关闭窗口并报错；
核验结果写入回执。除 phase-1 的 `/proc/stat`、gfrs 统计计数器 bind 与 pci-root 外，
其余模式都需要 `--approve HOST_SOURCE_REPLAY`（plan 与 execute 同时）。运行器 API：
`host_source_replay(ctx)` → `install`/`install_wrapper(tool, plan)`/`replan`/
`build_infiniband(...)`/`build_nvidia_proc(...)`/`open_window(binds, trees, wrappers,
inaccessible, env, pci_root)`/`open_command`/`open_tree`/`set_value`/`set_tree_value`/
`audit`/`verify`/`status`/`close`，计划用 `command_plan(tool, [rule(query, action, **字段)])`
构造并在发送前按包装器自己的允许列表校验。

**内核采集器的命令窗口（`KERNEL_COLLECTOR_SOURCE_REPLAY`）**：同一回放器以
`kernel_collector_source_replay(ctx)`（`unit=gpu-fault-kernel-collector.service`）在内核采集器
单元上只开 **command** 窗口，且只允许一个查询——该单元 CollectorContext 启动时的
`nvidia-smi --query-gpu=driver_version`（`tools/host_command_replay.py` `UNIT_QUERIES`；包装器、
探针、控制端三处各自拒绝其它工具/查询）。drop-in 写在该单元自己的 `.d` 目录（PATH 首位 shim、
审计目录可写，并 `UnsetEnvironment=GPU_FAULT_DRIVER_BRANCH`——`UnsetEnvironment=` 在
`EnvironmentFile=` 之后生效，安装器默认把该变量写空，窗口内不让已配置的分支顶替被扣留的查询），
核验除 shim/PATH 外还确认该变量在单元真实环境中不存在；只重启内核采集器，关窗/deadman 把它
重启回真实二进制，无 state 的残留 drop-in 也按两个单元的路径查找并移除。bind/tree/inaccessible/
env/pci-root 对该单元一律拒绝；`HOST_SOURCE_REPLAY` 不解锁内核采集器，新令牌也不解锁 host
collector 的扩展模式；Node Agent 仍永不在允许列表内。只读的 `journal-read --unit --since` 返回
允许列表内单元自己的 journal 尾部（有界），供变体读采集器对发现过程的记录
（`software version discovery unavailable` / `discovered NVIDIA driver branch`）。

### phase 2 范围与前提（工作负载/热备类）

第二批补 011 EFA 流量停滞/NCCL hung、027 训练进度、030 热备不健康；均用受管的
可丢弃测试任务，判定只读产品记录（011 读分组 `SITE_EFA_TRAFFIC` incident 的诊断
DAG、027 读 `SITE_TRAINING_HEALTH` incident/决策、030 读热备分级与注解），永不
POST 预制 finding。类别模块 `general_live_nccl`/`_nccl_fr`/`_progress`/`_spares`。

- **011（HyperPod 车道，需 `--node-b`；角色/线程/进程/换 attempt 变体同时注册
  plain-EKS 车道，以其第二台 EFA 节点作 `--node-b`）**：专用受管两节点 NCCL 测试任务
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
  两次真实 dump 为 `PAIRED`；现按产品缺口登记为 `expected_outcome="FAIL"`：DLC 镜像的
  PyTorch 2.9 把未钉 `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` 时的默认 dump 文件从
  `/tmp/nccl_trace_rank_<rank>` 移到 `$HOME/.cache/torch/comm_lib_trace_rank_<rank>`
  并优先读 `TORCH_FR_DUMP_TEMP_FILE`，而产品 `flight_recorder.py` 的读取器只认旧变量名与
  旧默认位置，手册的注入又不钉 dump 位置，于是 FIFO 找到、写入成功但全部 rank
  `dump_missing`、无 `PAIRED`；运行器在释放前把两 Pod 内三处候选位置的 `stat` 清单与
  PyTorch 自己的 FR 日志行写入 `<suffix>-fr-forensics.json`，`fr_dumps` 判定指出 dump
  实际所在——写在 2.9 默认位置、写在产品读取位置但读取过早、或根本未写）、
  `fr_disabled`（trace buffer 为 0，只能 `not_configured`；同样留存 forensics 证据）、
  `simultaneous_nodes`（合并 incident 须含两节点各自的 hung 证据）。
  live-replay 实测（`general_live_nccl_fr`，经审查的测试 FR dump 响应器
  `probes/general_live_fr_responder.py`，形状由 `general_live_fr_replay` 渲染）：同一
  两节点停滞任务以 `TORCH_NCCL_TRACE_BUFFER_SIZE=0` 运行（PyTorch 不建 debug pipe、
  不写 dump，`TORCH_NCCL_DEBUG_INFO_PIPE_FILE`/`TEMP_FILE` 仍在 rank 进程环境中，产品
  正是从那里读名字）；容器命令在 `exec torchrun` 前把响应器写到
  `/tmp/gfrs_fr_responder.py` 并在本容器内后台启动，它只拥有本 Pod 8 个 rank 的
  `<pipe base><rank>.pipe` FIFO（路径已存在即拒绝，不与真实 recorder 争抢），对产品
  每次写入的 `1\n` 以 pickle protocol 2（PyTorch 的落盘格式）回写
  `<dump base><rank>`：按组 completion 推进（每份 +2）/ 同一 pending 操作持续 /
  只 enqueue 不完成（每份 +1）/ 指定 rank 落后一个 collective / 第二份请求不回写；
  `--max-seconds 7200` 到期自行退出并只删除自己建的 FIFO/dump（RuntimeMaxSec 的容器内
  对应物，Pod 删除亦一并消失），账本 `/tmp/gfrs-fr-replay/ledger.json` 记录每次请求、
  回写的水位与摘要。FIFO 协议、命名空间映射、解析、双 dump 配对、分类与 bundle 绑定
  均为产品代码；只替换了原始读数，证据写 `proof_scope: live-replay`。判定只读
  COLLECT_DIAGNOSTIC_BUNDLE 步骤绑定的 `hung_triage_decision` 与 bundle 作用域参数
  （`bundle_scope`/`target_pids_by_node`/`max_processes`）、各 rank 的
  `flight_recorder_series`，并以账本为夹具 oracle：账本缺失、响应器出错、FIFO 名字
  不符记 `INCONCLUSIVE`；FIFO 在而产品未请求、未配对、漏报线索或 bundle 作用域错误记
  FAIL。跨节点组的进度比较需要本站 hung DAG 不携带的 `clock_uncertainty_seconds`
  （未知时钟不授权定向包，见 [NCCL hung 改进](nccl-hung-hardening.md)），因此产生
  线索/整组停滞判断的组是节点内组（每节点 8 rank，如 TP 组，按本节点时钟比较），全局
  默认组照样出现在每份 dump 中、其每 rank 水位差仍被读取；产品先采样上报节点，只有
  第一轮证据不足时才采样对端，判定均按实际采样的 rank 表述。
  `fr_temporal_completion`（默认组 +2/+2/+2，第二组只 enqueue +1/0/0 且 pending 持续，
  每 PID 两份 dump 间隔 ≥2 s、PAIRED）、`fr_second_missing`（全部
  `SECOND_UNAVAILABLE`，无 fr_temporal 判定、快照为准、不虚构停滞或线索、保守 bundle）、
  `fr_whole_group`（节点组 `group_stall_observed`、`stalled_process_groups` 点名、
  无 culprit、保守 bundle）、`few_candidates`（每节点 1 个 rank 落后：1-2 个可信候选，
  `targeted_capture_eligible`，bundle `identity_bound_candidates` 恰好绑定 culprit 与
  至多一个对照 rank 的 PID）、`ambiguous_candidates`（两个节点组各 2 个落后：单节点即
  4 个线索，仍报告但 `bounded_triage_nodes`、不定向）、`different_process_groups`
  （两节点组序号 1200/700 互不比较、成员不混、无 culprit）、`fr_plus_process`
  （write 模式停滞 + completion 推进：`HOST_WRITES_OBSERVED` 与进程 I/O 窗口保留在
  决策中，不被 FR 分支覆盖；D 状态等待无法原生制造）。
  `broken_pair`（`live-replay`，需 `HOST_SOURCE_REPLAY`，模块 `general_live_nccl_pair`）：
  两节点任务活跃基线证明后，把目标节点一个 EFA 端口的 `hw_counters/tx_bytes` 在 host
  collector 单元内 bind 成不可解析文本（共享回放器 bind 模式），再停滞任务并持有超过
  warning 阈值加一个 health summary；期间探针自己的收发对 oracle 读到真实零流量，而产品记录
  须为 `efa_traffic_observation_valid=0`、“incomplete traffic counter pair” 采集错误、无速率
  样本、无 ZERO_WARNING/HUNG（无效观察不开始零流量计数，也不是零）；关窗恢复后仍停滞的任务须
  在恢复后不早于 60 s 出 ZERO_WARNING（计数只从真实读数开始），释放后流量恢复、warning
  incident 收敛。
  角色/线程/进程与换 attempt 变体（`general_live_nccl_roles`）在同一训练程序的角色
  模式上做同样的停滞（`general_live_nccl_role_trainer`）：每 rank 经 `rank-progress/v1`
  的 `diagnostic_context` 声明 PP/EP 角色与容器内 native tid 的线程角色，keeper 线程
  每 10 秒重写快照使其保持在产品 30 秒新鲜窗口内（被阻塞的 rank 0 也是），辅助线程
  从启动即存在、随停滞切换行为，子进程受 watchdog、stop 文件与父进程退出约束。判定
  读 triage 各 rank 行（summary codec 信封解码）与产品绑定到 bundle 步骤的
  `hung_triage_decision`（cohort、process_observations），oracle 为训练程序自己打印的
  `ROLES`/`SHARER`/`HOLDER` 行；缺声明、子进程未起、持有程序编译失败记
  `INCONCLUSIVE`，被采样节点缺 rank 记 FAIL。`role_mismatch`：每节点 local 0-3 声明
  pp0/expert 并在停滞中烧 CPU，4-6 声明 pp1/dense 保持安静，7 不声明；cohort 只含同一
  声明角色、未声明 rank 不入 cohort、`ROLE_COHORT_*` 线索不得跨角色（同角色线索合法）。
  `tid_namespace`：容器 native tid 经 NSpid 对应唯一宿主机 tid，训练线程即宿主机 pid，
  `host_thread_roles`、各次 py-spy 样本的角色与稳定线程的 start ticks 一致；奇数 rank
  反序启动辅助线程，停滞中不断起止未声明线程。`wchan_zero`：声明的 COMMUNICATION
  线程自旋、BACKGROUND 线程睡在 futex；wchan 0 不是等待位置，只有睡眠线程有
  `REPEATED_THREAD_WAIT`，`cpu_cores` = tick 增量 / `SC_CLK_TCK` / 实测秒数。
  `cpu_interval`：50 % duty 的 CPU 线程；local 6/7 另起无 RANK 的 CUDA 子进程共享其
  GPU，窗口 elapsed 须与采样时间戳一致、决策 CPU 为窗口加权均值，共享 GPU 读 shared、
  `DEVICE_NOT_PID` 且不入 cohort。`pyspy_nonzero`：local 7 在训练容器内用镜像自带的
  gcc 编译并运行非 Python 的 CUDA 上下文持有程序，Node Agent 自己的 py-spy 对其合法
  失败（"Failed to find python version"，exit 1）；三次栈样本 FAILED、无签名、/proc
  窗口保留，Python rank 的栈解析为线程 JSON；不影子化任何工具，镜像无编译器则
  INCONCLUSIVE。`attempt_change`：attempt A 停滞至两节点出 `ZERO_WARNING`，在 hung
  阈值之下（最晚 540 秒）删除，同一任务以 `--attempt-id <job>-a002` 回到 attempt B
  并转为活跃；A 不得出 `HUNG_SUSPECTED`，B 不得出任何 warning/hung，各节点 EFA 流量
  状态行须记 B、首次观测在切换之后、`had_active_traffic` 为真、无零流量 streak。
  `unreachable_node`（native，`--approve NODE_AGENT_NETWORK_ISOLATION`，标志 `X`，eks/hp，
  `general_live_network_isolation.py` + `probes/general_live_isolation_probe.py`）：Node Agent
  是节点上的 HTTP 服务（`/v1/node-actions/submit|result`），执行器把节点动作推送到其心跳
  宣告的端点；基线证明后、停滞前，在 `--node-b` 上写入意图记录、布置 `systemd-run --on-active`
  deadman（≤ 3600 s，不超过维护窗口），再在 INPUT 链顶部插入 `-p tcp --dport <Agent 端口>
  -m comment --comment gfrs-<digest>-iso-agent-ingress -j REJECT --reject-with tcp-reset`
  （端口取自该节点 Agent 记录的 endpoint，探针核对 Agent 单元 active 且其主进程确在该端口监听；
  探针本机 TCP 连接须被拒才算阻断生效）。Agent 的出向心跳、各采集器与 kubelet 不受影响，所以
  两节点的 EFA 零流量仍由主机采集器上报、控制面仍认为节点在覆盖之内。等 hung 工作流的
  COLLECT_HUNG_SUMMARY/TRIAGE 执行记录出现（或工作流终结）后立即恢复通道，让 bundle/
  VALIDATE_FABRIC/ESCALATE_SUPPORT 面对可达节点运行；再等 incident 稳定。判定只读产品记录：
  DAG 形状与禁止动作同其他 011 变体；incident 为 ESCALATED；summary/triage 执行的
  `undetermined_nodes`/`node_failures`/`hung_triage_decision.undetermined_nodes|not_sampled_nodes`
  或 `survey_coverage.responding_node_count < expected_node_count` 至少一处点名 node-b，
  且 node-b 不出现在任何 summary/triage 的 `node_results`（否则为伪造读数），可达节点的
  `node_results` 存在；node-b 未被 cordon/隔离；清理后探针回读 INPUT 链无残留、timer 未激活、
  Agent 端口再次接受连接，且控制面记录的 Agent `last_seen_at` 晚于恢复时刻；最后释放停滞并
  证明 EFA 流量恢复。
  延后留分母：规模类 `large_job`/`no_step_no_rank`/
  `active_window_resume`（`SCALE`，本站无 128 节点任务）。
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
  REPLACE_NODE 失败转路径可见，其断言已由 GF-REGIONAL-DESTR-008 S1 在本站真机做过
  （hp PASS 2026-10-22：移除全部热备标签后 REPLACE_NODE 以精确的 no-fallback 字面量
  拒绝、CloudTrail 无 BatchReplaceClusterNodes），记
  `NOT_RUN(COVERED_BY_REGIONAL: GF-REGIONAL-DESTR-008 S1)`，不另建夹具。

计划/执行形如 `--case GF-GENERAL-LIVE-011 --variant active_to_zero --site-file <F>
--node-b <N2> --hyperpod-cluster <HP>`（011/030 在 hp 车道，011/030 需 `--site-file`，
011 需 `--node-b`，030 的热备节点用 `--fault-node`）；011 的角色/线程/进程/换 attempt
变体在 eks 车道形如 `--case GF-GENERAL-LIVE-011 --variant role_mismatch --site-file <F>
--node-b <第二台 EFA 节点>`。

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
- 主机采集器**原始源回放** `general_live_source_replay.py`（探针
  `probes/general_live_source_probe.py`）只在主机采集器自身私有环境内替换：
  `BindReadOnlyPaths` 的允许列表 procfs/sysfs 文件或整棵树、PATH 首位的透传命令包装、
  `InaccessiblePaths`、期望数环境覆盖，或重指 `GPU_FAULT_PCI_DEVICES_ROOT`（详见上文
  “共享主机原始源回放夹具”；phase-1 之外的模式需 `HOST_SOURCE_REPLAY`）。
- 可复用的**CRI Pod 日志源夹具** `general_live_applog_fixture.py`（主机探针
  `probes/general_live_applog_probe.py`，受管归属目标 `general_live_applog_job.py`）：
  节点日志采集器只经 `GPU_FAULT_TRAINING_LOG_PATHS` 读训练日志（本站为空），窗口用
  `/run` drop-in 追加第二个 `EnvironmentFile` 把该单元指到本次运行的
  `/var/log/pods/*/<container>/gfrs-live-<digest>.log*`，runtime 启用并重启
  （`ENABLE_NODE_LOG_COLLECTOR`）。夹具文件建在 kubelet 为运行器自建受管 Pod 创建的
  真实目录里（仅 `old_pod` 为已不存在的旧 Pod 以其真实 UID 重建目录，清理即删），按 CRI
  语法 `<RFC3339 ns> <stdout|stderr> <P|F> <message>` 以节点时钟逐条写入，支持改名轮转
  （`.1`，继续写旧 inode）与采集器重启；Pod UID、容器名、attempt 与
  `gpu-fault.io/host-pid`（探针从容器 cgroup 读出 init 进程 PID，运行器为自己的 Pod
  声明）均为真实值，只有日志文本是经审查输入。oracle 只读采集器自身状态文件里本次文件的
  offset/inode/待拼接 CRI 片段；判定读 attempt observation、投递的 NODE_LOGS 条目
  （`pod_uid`/`container_name`/CRI 事件时间）、incident 的 attempt 和诊断 bundle 步骤
  参数（`attribution`/`process_state`/`target_pids_by_node`），不 POST 任何预制记录。
- 所有回放都带 `systemd-run --on-active` deadman，证据标注 `proof_scope: live-replay`，
  只证明处理链，不证明物理故障。

已实现变体：016 严重内核日志（全部 9 个，含 user_text/old_boot/corrected_mce 的
反向门禁）、017 应用错误日志（journal 源的 cgroup_oom 诊断与 nccl_error/distributed_error
仅证据；CRI Pod 日志归属 `live_pid`（LIVE，目标 PID 即声明的 host PID，bundle 实际抓取）、
`exited_pid`（EXITED，不编排 bundle/不抓栈）、`old_pod`（产品 RESTART_WORKLOAD 后以旧 Pod UID
写入，只能归 UNRESOLVED 或旧 attempt）、`cri_partial_rotation`（P 片段跨轮次、改名轮转与
采集器重启后与 F 拼成一条，事件时间取 P 的时间戳；投递失败回滚由采集器单元测试钉住，不在线
诱发），均需 `--site-file`；native 的 `missing_flight_recorder` 复用 011 的两节点 NCCL 任务且
无 flight recorder，需 `--site-file --node-b`，判定独立 RDMA 空闲触发
`detection_basis: ATTEMPT_SCOPED_RDMA_IDLE` 与 triage 的 `not_configured`）、
032 `unknown_xid`（判为 SITE_SAFETY）、`duplicate` 与 `missing_topology`、013
`unreadable_state`（UNKNOWN 而非零卡）与整树回放的 `unknown_inventory`/`excess_device`/
`link_inactive`，phase-1 文件源 001 `invalid_delta`、008 `statistics_failed`、
009 `errors_delta`，以及共享回放器上的 010 全部 8 个 EFA/RDMA 计数变体与 011 `broken_pair`
（见 host/device 与工作负载两节）。

- 032 `missing_topology`（kmsg 回放，`KERNEL_SOURCE_REPLAY`，两个集群）：经审查的
  Fabric Manager 摘要行 `nvidia-nvswitch0: SXid (PCI:...): 11001, Fatal, ...`，**不写**
  `Link N`。内核采集器只转发摘要行（`SXID_SUMMARY_PATTERN`），解析器把链路作用域留为
  UNKNOWN，控制面只能从新鲜可信的 `nvswitch_port_topology`（host collector 的拓扑查询，
  只在设置 `GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND` 时运行；节点安装器默认为空，本站两个
  集群的部署都未设置）或 H100/H200 的“全 access”产品不变量（需要行内有端口号）解析。
  因此本站的“缺拓扑”是真实状态而非伪造：11001 在固定目录里是 FATAL /
  `SCOPE_DEPENDENT_RESET`，无可信作用域 → `BLOCKED_MISSING_EVIDENCE`
  “requires a trusted access/trunk port-scope mapping”，operator-required，
  RESET_ALL_GPUS_AND_NVSWITCHES 被扣留。PASS 要求 FaultPolicyDecision 正是如此且无任何
  reset/reboot 步骤；产品按站点安全对该事件做 MARK_UNSCHEDULABLE + QUARANTINE，变体经
  产品的 validated restore 把节点交回。决策通过节点新 incident 的 `event_id` 读
  `get_xid_policy_decision`（SXID 决策同表）。
- 032 `missing_version`（`KERNEL_COLLECTOR_SOURCE_REPLAY` + `KERNEL_SOURCE_REPLAY`，两个
  集群，`S`）：解码器门禁（`_workflow_nvlink5` 的 `driver_branch is None` →
  BLOCKED_MISSING_EVIDENCE "NVLink5 decode requires driver branch, IntrInfo and Error
  Status"）读的是 XID 事件上由**内核采集器**自身 CollectorContext（单元启动时的
  `nvidia-smi --query-gpu=driver_version`）盖上的版本。变体先在内核采集器单元上开命令窗口
  （包装器以经审查的失败答复 `driver_version` 查询、`GPU_FAULT_DRIVER_BRANCH` 置空），用包装器
  审计（重启后的采集器确实来问、且被拒）和单元自身 journal（无 `discovered NVIDIA driver
  branch`）证明版本确已缺失，**然后**才经 kmsg 回放一条真实 B 系列形态的 XID 149 行
  `NVRM: Xid (PCI:<gpu>): 149, pid=0, name=gfrs-live, NETIR_LINK_EVT Fatal XC0 i0 Link 08
  (0x004505c6 0x00000000)`（固件两寄存器 payload；V1 解码为 RESET_GPU、V2 无匹配——正因如此
  窗口无效时绝不注入）。判定只读产品记录：事件带 `intr_info`/`error_status` 而
  `driver_branch=null`，决策 BLOCKED 且理由点名 driver branch、WORKFLOW_NVLINK5_ERR 被扣留、
  operator-required、无 reset/reboot；站点安全隔离经 validated restore 交回。窗口未生效
  （无被拒查询、journal 出现 discovered、或事件上盖了分支）记 INCONCLUSIVE。**本站结论**：
  `_resolve_xid` 先过目录产品门禁再进工作流解析器，而 XID 144-150 的产品只有 B100/GB200，
  H200 上任何此类 XID 无论分支如何都是 NOT_APPLICABLE（真实解析器 + 引擎已验证）——版本门禁
  在本站不可达，变体从 DCGM survey 读到节点 GPU 型号后、在任何变更之前记
  NOT_RUN(CAPABILITY_ABSENT)；在 Blackwell 节点上照常执行。
- 013 整树回放（`HOST_SOURCE_REPLAY`，两个集群）：`build-infiniband-fixture` 先复制节点
  真实 `/sys/class/infiniband`（选一个端口 ACTIVE/LinkUp 的真实设备），再施加一个经审查的
  变异并以 `--tree` 绑入 host collector 单元；PCI 根保持真实，`pci_discovered` 不会下降，
  故这些变体进不了 PCI_DEVICE_MISSING 的 reboot 链。`unknown_inventory`：该设备
  `device/driver` 身份 ambiguous → collection error
  `EFA inventory observation unavailable: missing or ambiguous RDMA driver identity`，
  无任何 EFA 计数/mismatch 样本、无 finding、无 reboot、无隔离。`excess_device`：复制该设备
  为 `gfrs-excess0` → `failure_mode=EXCESS_DEVICE` → RUN_DIAGNOSTICS（FREEZE_EVIDENCE +
  VALIDATE_FABRIC）；finding 出现后立刻关窗。批次 175 起诊断链的 VALIDATE_FABRIC 按链到达该步后的
  新鲜库存读数判定并把所依据的读数写进 `details.inventory_evidence`：读到恢复后的满数即 SUCCEEDED，
  仍读到多余设备即 FAILED（EXCESS_DEVICE 无升级，`escalate_failed_hardware_remediation` 不规划任何
  步骤，incident 记 diagnostic inconclusive 后 RECOVERED）；PASS 接受两种已结论的判定，并要求
  `inventory_evidence` 在、任何验证不得等待 `missing_inventory_action_completion`、无硬件/隔离步骤、
  incident RECOVERED、恢复满数。`link_inactive`（`risk M`，`S`，仅需 `HOST_SOURCE_REPLAY`，预期
  PASS——原缺口已由批次 175 关闭，不再需要 `NODE_REBOOT`）：该设备一个端口 `state`/`phys_state`
  改为 `1: DOWN`/`3: Disabled` → `failure_mode=LINK_INACTIVE` → RUN_DIAGNOSTICS。因该链的
  VALIDATE_FABRIC 对仍不活动的端口按设计升级 REBOOT_NODE，端口只曝光一个**有界曝光期**：树窗口
  自己的 deadman 在开窗后 1.7 个采集周期（15 s 周期为 26 s）关窗——在确认缺口、产出 finding 的
  第二拍之后、第三拍之前——使诊断只读到恢复后的端口，而不靠运行器轮询（20 s 存储轮询加两次探针
  往返追不上 15 s 的拍）；运行器随后的关窗只是确认（`closed_by: deadman`，deadman 迟到则自己关并
  记 `runner`）。PASS 要求先诊断（首个工作流只有 FREEZE_EVIDENCE + VALIDATE_FABRIC、无硬件/隔离）、
  该 VALIDATE_FABRIC SUCCEEDED 且 `details.inventory_evidence[<node>][efa_inventory_active_count]`
  为恢复后观测到的满数、无验证等待 `missing_inventory_action_completion`、incident RECOVERED、
  无 RESTART_NODE、恢复满数。曝光期内未出 finding（两拍未都落在窗口内），或恢复后仍记录到
  `efa_inventory_missing_count>=1`（上界未守住，此时产品对新鲜不活动证据的升级是设计所为）记
  INCONCLUSIVE，不是对产品的判定。
- 013 `pci_missing`（`risk H`，`X`，两个集群，`--approve HOST_SOURCE_REPLAY NODE_REBOOT`，
  模块 `general_live_efa_pci.py`，`expected_outcome=PASS`）：先做规格要求的反向阶段——
  `build-pci-fixture --unreadable` 把夹具路径建成普通文件并以 `pci-root` 重指入 host collector
  单元（任何权限下列目录都 ENOTDIR），持续 4 个 tick（两倍于 2 样本确认窗）；PASS 要求 collection
  error `EFA inventory observation unavailable`、**没有**任何 EFA 计数/mismatch 样本、无 finding、
  无硬件步骤、无隔离（`_efa_inventory` 在该路径上还会清零确认计数；模块测试用真实采集器钉住
  missing→不可读→missing→missing 只在第 4 个 tick 确认）。反向阶段若产出 finding 记 FAIL、等待
  其链结束且不再执行正向阶段。正向阶段：`inventory` 选一个端口 ACTIVE 的真实 EFA 设备及其 PCI
  功能（新增 `pci` 字段），`build-pci-fixture --source /sys/bus/pci/devices --drop <BDF>` 复制
  节点真实 PCI 根但缺该功能，`build-infiniband-fixture --drop <dev>` 复制 RDMA 树但缺其设备，
  以一个 mixed 窗口绑入单元；finding 出现即关窗（deadman 解除），之后产品的每次读取都是真实的。
  产品链：`failure_mode=PCI_DEVICE_MISSING` 两个连续样本确认 → CRITICAL REBOOT_NODE →
  MARK_UNSCHEDULABLE、QUARANTINE、RESTART_NODE（provider）、VALIDATE_GPU/HOST/FABRIC
  （`EFA_INVENTORY_RECOVERY_OPERATIONS` 含 RESTART_NODE，验证只读重启之后的样本）、
  RESTORE_SCHEDULING。运行器按 COLLECT-004 的方式等重启（`node_snapshot` 新 boot id + Ready、
  Node UID 不变；`wait_node_settled_after_reboot`），再重建随节点一起死亡的探针 Pod，用
  `snapshot` 证明新 boot 上没有 drop-in/deadman/state、单元的 `GPU_FAULT_PCI_DEVICES_ROOT`
  不是夹具路径，`read-source` 证明读回真实 RDMA 树。判定只读产品记录：确认样本的 failure_mode
  与 expected/observed/missing/discovered/driver_bound 计数、恰好一个 SUCCEEDED 的 RESTART_NODE
  执行与节点范围的远程命令、无 REPLACE_NODE、无第二次重启、incident RECOVERED、节点 Ready 且
  可调度、EFA/GPU allocatable 回到基线、重启后新鲜的 `efa_inventory_active_count=期望`/
  `mismatch=0`、CloudTrail 中 provider 的重启回执（EKS `RebootInstances`/HyperPod
  `BatchRebootClusterNodes`）只命名本实例，以及 `--node-b`（同车道另一节点）未被触及。重启安全：
  关窗在 RESTART_NODE 之前完成；若产品比关窗更快，/run 下的 drop-in、deadman transient 单元与
  夹具都不会跨重启存在，deadman 在关机期间最多失败一次 `systemctl restart`，残留由新 boot 上的
  `snapshot` 证伪。Spot 回收导致 Node UID 变化记 INCONCLUSIVE；链结束后节点仍被隔离时清理走
  产品的 validated restore。需要 ≥ 70 分钟剩余维护窗；EKS 先跑，HyperPod 需 `NodeRecovery=None`。
- 013 原生 EFA 驱动解绑 `bind_failure`、`post_repair_failure`（`native`，`risk H`，
  `X`，`--approve EFA_DRIVER_UNBIND NODE_REBOOT`，两个集群，模块
  `general_live_efa_driver.py` + 探针 `probes/general_live_efa_probe.py`，车道最后执行）：
  复用 COLLECT-017 A / DESTR-021 对**一个**允许列表内 EFA 数据功能的真实解绑
  （`/sys/bus/pci/drivers/efa/unbind`）与 `systemd-run --on-active` 重绑保底，另加两个负向分支。
  `bind_failure`：先把该功能的 `driver_override` 置为无模块提供的名字（`gfrs-live-hold`），
  使内核对**这一个功能**到 `efa` 的每次 bind 返回 ENODEV，再解绑；产品判 DRIVER_UNBOUND →
  `REMEDIATE_EFA_DRIVER`（`modprobe efa` + 逐 BDF `printf BDF > .../efa/bind`）对该 BDF 的
  bind 失败并记入独立 `bind_errors` 回执。该步骤带 `failure_escalation_action=REBOOT_NODE`，
  产品升级到 `RESTART_NODE`——这是“bind 确实绑不上”的已批准分支，重启也顺带清掉逐设备
  override；全程**不做**全局 `efa` 卸载（模块保持加载、其余功能保持绑定）。`post_repair_failure`：
  不设 override，解绑后由一个有界一次性 trap 在产品 bind 回执**之后**几秒再次解绑，于是
  `REMEDIATE_EFA_DRIVER` 记 SUCCEEDED（`rebound_pci_bdfs=[BDF]`）而其后的
  `RESTART_EFA_DEVICE_PLUGIN`/`VALIDATE_FABRIC` 在新鲜证据上独立 FAILED，升级到
  `RESTART_NODE`——即动作成功但动作后验证失败所要求的独立回执与升级。两者只按
  `driver_override` 一次写入或重启即可还原、仅作用于一个功能、由保底独立重绑，判定只读产品
  记录（incident 原因、步骤回执、allocatable、boot id、无越界隔离），清理循环把功能绑回
  `efa`、清掉 override，否则大声失败。

计划/执行守卫键仍为 `<case>.<variant>`，确认串 `GENERALLIVE<NNN>_EXECUTE`。新 `--approve`
令牌：`KERNEL_SOURCE_REPLAY`（kmsg 源重指，032）、`KERNEL_COLLECTOR_SOURCE_REPLAY`（内核采集器
单元上仅 `driver_version` 查询的命令窗口，032 `missing_version`）、`EFA_DRIVER_UNBIND`、
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
原生 plugin 故障需 `DEVICE_PLUGIN_RESTART`；其中 `running_training` 的断言——受管训练
任务下移除 EFA 设备插件、Pod UID 集合与训练进度保持、无 STOP/RESTART_WORKLOAD——已由
GF-REGIONAL-COLLECT-017 stage C 在本站真机做过（hp PASS 2026-10-22），记
`NOT_RUN(COVERED_BY_REGIONAL: GF-REGIONAL-COLLECT-017 stage C)`）、013 `driver_unbound`（`pci_missing`、`bind_failure`、`post_repair_failure` 已由上文的变体实现；同一 BDF 的真实 `efa` 解绑与同一
REMEDIATE_EFA_DRIVER 7 步链断言已由 GF-REGIONAL-DESTR-021（eks PASS 2026-10-22）、
GF-REGIONAL-COLLECT-017 stage A（hp PASS 2026-10-22）、GF-REGIONAL-NET-007（eks PASS）
做过，记 `NOT_RUN(COVERED_BY_REGIONAL: ...)`，注册项的 `EFA_DRIVER_UNBIND` 归属与车道
保留）、032 的产品/Profile 权限变体（AUTH-015 未配置）与本站 H200 上的 `missing_version`
（目录产品门禁先于 NVLink5 解码器，记 CAPABILITY_ABSENT；夹具已就绪，Blackwell 节点可执行）、
004 `probe_error`（`statvfs` 错误无法由只读文件 bind 产生）。

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

节点状态规则（2026-10-25 现场：019 `corrected_ecc`/`volatile_dbe` 在 `repoint-collector`
写入 drop-in 并重启采集器后才被判失败，收尾又跳过还原，节点留下 drop-in、代理单元和运行
目录，只剩 deadman 兜底）：

- **开始前**：探针 `survey` 按运行 digest（`gfrs-live-<digest>`）列出节点上尚存的每个回放
  对象——drop-in、代理单元、运行目录（含其 `run.json` 记录的 run id）、deadman 定时器及其
  触发时刻——并单列非本夹具的 drop-in / 目录和仍在回放端口应答的工具。只要有任何残留，
  变体记 `NOT_RUN(FIXTURE_NOT_READY)` 并逐一点名，不装任何文件。仅当残留是本夹具自己的
  对象、且其 deadman 已不再待触发（已触发、从未布置或已停止）时，运行器先经探针
  `cleanup-leftover --digest` 还原再重新 survey；deadman 仍待触发的留给它触发，外来对象
  一律不碰。
- **结束时**：`DcgmReplay.close` 只走探针幂等的 `close-run`，与运行器自身记的
  armed/repointed 标志无关：best-effort disarm 工具 → 删 drop-in 与覆盖文件、重启采集器
  并核对真实端点 → 停工具、删运行目录 → 全部成功才撤 deadman。任一步失败则保留 deadman，
  并把探针仍找到的每个对象逐一写入 `ReplayCleanupError`（整项 FAIL 的原因里带名字），
  `probe_residuals`/`cleanup` 失败项同样写进 "cleanup did not prove the node clean" 之后。
- **重指验证**：`repoint-collector` 读取采集进程环境前先等它真正 exec——`Type=simple`
  单元 fork 即 active，沙箱建立期间 `/proc/<pid>/environ` 仍是 systemd 自身的环境——
  超过等待窗口或读到真实端点才算失败；探针失败行带异常类、消息摘要与 `函数:行` 位点，
  运行器把子命令名前置到 `HostProbeError`。
- **arm 后的有效性门**（2026-10-25 现场：两台 plain-EKS 节点上 019 的 6 个回放变体全部
  INCONCLUSIVE "the replay never served"——运行器在 arm 后约 30 s 即判回放日志，而 15 s
  间隔的采集器只抓了两次，都落在计划开头按设计不改任何读数的 3 次透传基线读里）：
  `arm_and_confirm` 不立即判定，而是轮询 `audit-read` 直到日志出现第一条真正回放的读数
  （`served` 含目标 `<GPU>/<字段>`）；上限 =（基线读数 + 1 + 2 次余量）× 采集间隔 + 90 s
  宽限，间隔取 `repoint-collector` 回执里重指后采集进程真实环境的
  `GPU_FAULT_METRICS_INTERVAL_SECONDS`（默认 15 s，本站上限即 180 s）。回放 abort 立即判
  INCONCLUSIVE；超时的 INCONCLUSIVE 写明日志里的 armed 读数与最后所到的计划步骤，
  以区分"采集器根本没抓回放"与"尚未走到回放步骤"。

需审批与独占节点的变体：019 `pending_supported`/`post_reset` 单卡 service-window
reset 需 `--approve GPU_RESET`；021 任一 finding 都驱动全 GPU+NVSwitch reset，故
该类变体需 `--approve GPU_FULL_FABRIC_RESET`（`recovery_failure` 另需
`NODE_REBOOT`）并只在 HyperPod 独占节点跑；022 的 XID 联合变体只把 DCGM last-XID
gauge 回放为 XID 32（`--approve DCGM_XID_GAUGE`）；019 `remap_failure` 的 RMA 现场
诊断需 `GPU_FIELD_DIAGNOSTIC`。延后：019 `pending_unsupported`（无旧架构卡）、
`observe_profile`（无 OBSERVE reset Profile）记 `CAPABILITY_ABSENT`。

**复用 K 车道的独立可信内核 XID**（020 `independent_fatal_event`、024
`independent_xid`，`general_live_kernel_xid.py`，需 `--approve KERNEL_SOURCE_REPLAY`）：
在 DCGM 回放把 NVLink 计数/弱复合 `CORRECTABLE_MEMORY_DEGRADATION` 保持在目标卡上
并已形成 incident 之后，经 `general_live_kernel_replay` kmsg 模式投递一条经审查的
`NVRM: Xid (PCI:<目标卡 BDF>): <code>` 行，走真实 `KernelLogCollector`、NVIDIA_KERNEL
投递、GPU 身份补全和固定 catalog。只允许 catalog Immediate Action 为 `IGNORE` 的
**monitor 级**条目（`require_monitor_grade`，执行前校验，catalog 变化则记
`NOT_RUN(FIXTURE_NOT_READY)`）：020 用 XID 137 `NVLINK_PRIV_ERR`，024 用 XID 92
`EXCESSIVE_SBE_INTERRUPTS`（与弱复合同属 GPU_MEMORY 类）。引擎把 `IGNORE` 解析为
`NO_ACTION`/`MONITOR_ONLY`（与 workload 状态无关），协调器不会把非 EXECUTABLE 决策并入
其它 incident，其 incident 创建即 `RECOVERED`、无 workflow，因此真机不隔离、不 reset。
判定读产品自己的 `FaultPolicyDecision`：恰一条非 duplicate 决策、disposition
`MONITOR_ONLY`、official_action 等于 catalog、无 action/safety_action/operator/
correlated_event/workflow、incident 为自己的且 `RECOVERED`、不是计数/复合 incident、
GPU 解析到目标卡；复合的 `component_metrics`/`component_finding_ids` 不含 XID；计数链
仍是单条只读诊断。它证明可信源路径独立、未被吸收；**不**证明致命级（RESET/QUARANTINE）
处置在并发诊断 workflow 下存活——该分支需 reset 授权，由 kmsg 致命 XID 的危险车道执行。

**复用 GF-ENV-LIVE 环境回放**（024 `thermal_warning`、`power_performance`，
`general_live_gpu_environment.py` + `general_live_env_replay.py`，无需新令牌）：
`tools/gpu_environment_replay.py` 的 `ALLOWED_FIELDS` 原样覆盖温度/T.Limit、热/功耗
violation、clock reason、功耗/限值/利用率，由 `probes/env_replay_proxy.py` 装入并以
`/run` drop-in 重指 `gpu-fault-metrics-collector.service`（deadman 先布置，透传到 arm）。
计划来自 `env_live_plans.thermal_stress_plan`（warning 区间核心温度，刻度按设备自身
`nvidia-smi` 限值决定，与 003/006 相同；thermal violation 从设备真实计数之上递增；
`0x20` 软件热 reason，绝不 `0x80`）和 `power_limit_throttling_plan`（violation 递增、
功耗钉在真实限值的 ratio 与限值之间、高利用率；不服务任何 clock reason，否则产品按
设计把它当温度 finding 排除出功耗复合）。arm 前校验目标卡真实 T.Limit 高于 10 C 且无
热/brake reason。判定：`THERMAL_STRESS`/`POWER_LIMIT_THROTTLING` 保持 WARNING
`RUN_DIAGNOSTICS`（finding 的 severity/automatic_action 与 incident 的 effective_action
三处一致），workflow 只有 FREEZE_EVIDENCE/VALIDATE，不隔离不 reset；去重按实际
workflow：恰一条复合 incident，组件 finding 被抑制进复合（复合形成后不再单独 mint
组件 incident），复合只由自己的指标组成；代理 `/status` 证明采集器确实消费了全部回放
读数、工具未因真实不安全读数中止，产品接受过回放的 warning 样本；尾部平直/清零后先还原
endpoint 再等待无活动 finding 的新批次（不要求硬件累计计数器归零）。

规格明示、注入有效但产品不足的按 `expected_outcome="FAIL"` + `product_gap` 记录，不
掩盖。首批记为预期 FAIL 的四个产品缺口——025 `one_gpu_missing`（完整性按全批 union
计算，漏查单卡）、022 `expired_xid`（PCIe↔XID gauge 回退无时效检查）、026
`bdf_changed`（身份只比 UUID 集）、026 `stale_boot`（库存基线不比 source_boot_id）
——已由批次 160 修复，对应变体改为预期 PASS；产品回退时这些变体判 FAIL。当前登记的
预期 FAIL 缺口：011 `fr_enabled`（产品 FR 读取器不认 PyTorch 2.9 的默认 dump 位置
`$HOME/.cache/torch/comm_lib_trace_rank_<rank>` 与 `TORCH_FR_DUMP_TEMP_FILE`，手册注入
亦不钉 dump 文件；见该用例的变体说明），产品修复并经该变体 PASS 后改回预期 PASS。
