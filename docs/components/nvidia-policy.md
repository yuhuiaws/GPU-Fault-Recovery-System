# NVIDIA 官方策略实现审计

## 固定上游

- Catalog：NVIDIA Xid Catalog 610
- 官方地址：
  `https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`
- XLSX SHA-256：
  `7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5`
- 生成物 canonical SHA-256：
  `ffb82509abb574577db3c5d759cec94df40ea25edaf4691e7bd4ba0a12ba0462`（即生成物 `metadata.generatedSha256`）

生成物
`src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml`
包含 172 条 XID、95 条 XID 144-150 decode 和 32 个 Resolution Bucket。

重新生成：

```bash
PYTHONPATH=src python tools/generate_nvidia_xid_policy.py \
  Xid-Catalog.xlsx \
  src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml \
  --catalog-version 610 \
  --expected-sha256 \
  7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5
```

摘要不匹配时生成过程直接失败，禁止静默接受上游变化。
`--expected-sha256` 即使省略也默认使用上述固定 XLSX 摘要；输出路径省略时写到上面的生成物。
整数单元格拒绝布尔值，不能把 `TRUE`/`FALSE` 当成 XID 编号；整数、整值浮点和整数文本
保留原规范化行为。只读工作簿在成功及解析失败后都会关闭，避免重复生成时累积文件句柄。

无需 XLSX 的常驻门禁：

```bash
PYTHONPATH=src python tools/generate_nvidia_xid_policy.py --check
```

该门禁检查 canonical 格式、上游摘要、172/95/32 条数和
`metadata.generatedSha256`（`--generated-policy` 可指定别的生成物）。运行时 loader 会重新计算生成物摘要；
`mapping_version` 为 `nvidia-xid-catalog-610/sha256:<生成物摘要前 16 位>`，因此手改规则会直接加载失败，
或在显式更新摘要后触发部署 pin 不匹配。

上游 XLSX 不进入源码树。重新生成前应把已校验文件存入
`artifacts/upstream/nvidia/xid-catalog-610/<source-sha256>/`，并将该目录归档到
受控对象存储；`artifacts/` 只作为本地工作区。

### DCGM Exporter

- 镜像：`nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless`（DCGM 4.6.1 客户端库、exporter 4.8.4、distroless 变体；
  该 release 使用 go-dcgm `v1.4611.2`）
- manifest list SHA-256：
  `sha256:148b0c025e5f2850256816fa33754fbf4733b7a086697a95613fc0ba3db5b003`（`4.6.1-4.8.4`、`4.6.1-4.8.4-distroless`、
  `4.8.4` 与 `latest` 都指向它）
- linux/amd64 manifest SHA-256：
  `sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b`
- 镜像配置：`com.nvidia.dcgm.version=4.6.1`、`org.opencontainers.image.version=4.8.4`，基础镜像
  `nvcr.io/nvidia/distroless/cc:v4.1.4`；层中有 `libdcgm.so.4.6.1` 与 `/etc/ld.so.cache`
- 产品中的位置：唯一来源 `src/gpu_fault/admin/dcgm_exporter_release.py`；`config/release-images.lock.json` 的
  `dcgm_exporter`（`reference` 为 manifest-list digest，`source` 为 tag）；普通 EKS host 模式的主机 DCGM
  `4.6.1-1`（`DCGM_HOST_PACKAGE_VERSION`）由它推导；规则在 `src/gpu_fault/dcgm_versions.py`

NVIDIA 的规则（原文，取自该页 "Last updated on 2026-09-10" 的版本，产品与测试按此版原文钉住）：

- "Install DCGM Exporter"，"Requirements and Version Compatibility"
  （<https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html#requirements-and-version-compatibility>）：
  - "DCGM Exporter releases pair a DCGM version with an exporter version. Container image tags encode the tested pair as `<dcgm-version>-<dcgm-exporter-version>-<image-variant>`."
  - "Run DCGM Exporter with the DCGM version paired with that exporter release. Mismatched combinations might function, but they are not tested or supported. When troubleshooting, first update to matching, current versions of DCGM and DCGM Exporter."
  - "For a separate DCGM host engine, the exporter’s DCGM client library version must also be greater than or equal to the DCGM host engine version. Meeting this connection requirement does not make a mismatched release pair supported."
  - 同页排障："For a separate DCGM host engine, verify the endpoint, transport, socket mount, and client/host-engine version relationship."
- 同页 "Connect to a DCGM Host Engine"（<https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html#connect-to-a-dcgm-host-engine>）：
  "The URI forms require DCGM 4.5 or later."（产品沿用 `host:port` 形式 `127.0.0.1:5555`）
- README（<https://github.com/NVIDIA/dcgm-exporter>，"Version compatibility"）："Run DCGM Exporter with the DCGM version paired with that exporter release."
- GitHub release（<https://github.com/NVIDIA/dcgm-exporter/releases/tag/4.8.4>）："Release DCGM Exporter 4.8.4 with DCGM 4.6.1 and go-dcgm v1.4611.2."
- DCGM release notes（<https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html>）4.6.1 的
  Bug Fixes、Core："Add safeguards against potential memory errors when using mismatched clients and hostengines."

为什么是这个 release：它是最新的 exporter release；DCGM 4.7.0 的软件包已经发布，但 registry 与 GitHub 上都没有与 DCGM 4.7.x
配对的 exporter（go-dcgm 的 `v1.4701.*` 绑定存在，没有 exporter release 使用它）。与 DCGM 4.6.0 配对的
`4.6.0-4.8.3-distroless` 不带 `/etc/ld.so.cache`，启动时对 `libdcgm.so.4` 的检查随即以 "exit status 1" 退出，除非容器运行时的
NVIDIA hook 写入一个（github.com/NVIDIA/dcgm-exporter issue #612，由 4.8.4 的 PR #724 修复）。自 4.8.2 起 NVIDIA 只发布
distroless 变体。各集群的配对（普通 EKS host 模式为受支持的配对；HyperPod 的引擎由 AMI 决定，HyperPod AMI 自带的引擎（当前为 DCGM 4.7.0）比 4.6.1
客户端新，违反连接要求，产品没有补救手段，在 NVIDIA 发布基于 DCGM 4.7.0 或更新版本的 exporter 之前只提示）、只提示不拒绝的理由与升级
步骤见[部署和运维手册](../部署和运维手册.md) §6.5「DCGM exporter 版本与主机 hostengine 配对」。

重新核对（NGC registry v2 API，匿名 pull token）：

```bash
token="$(curl -s 'https://nvcr.io/proxy_auth?scope=repository:nvidia/k8s/dcgm-exporter:pull' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')"
registry=https://nvcr.io/v2/nvidia/k8s/dcgm-exporter
# manifest-list digest of each tag: one value for 4.6.1-4.8.4, 4.6.1-4.8.4-distroless, 4.8.4 and latest
for tag in 4.6.1-4.8.4 4.6.1-4.8.4-distroless 4.8.4 latest; do
  curl -sI -H "Authorization: Bearer ${token}" \
    -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
    "${registry}/manifests/${tag}" | grep -i docker-content-digest
done
# the linux/amd64 manifest of the list, then its image config labels
curl -s -H "Authorization: Bearer ${token}" \
  -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
  "${registry}/manifests/4.6.1-4.8.4-distroless" \
  | python3 -c 'import json, sys; print([m["digest"] for m in json.load(sys.stdin)["manifests"] if m["platform"]["architecture"] == "amd64"][0])'
config="$(curl -s -H "Authorization: Bearer ${token}" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "${registry}/manifests/sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["config"]["digest"])')"
curl -sL -H "Authorization: Bearer ${token}" "${registry}/blobs/${config}" \
  | python3 -c 'import json, sys; labels = json.load(sys.stdin)["config"]["Labels"]; print({key: labels[key] for key in ("com.nvidia.dcgm.version", "org.opencontainers.image.version", "org.opencontainers.image.base.name")})'
```

换 release 时按部署和运维手册 §6.5 的升级步骤同时修改上面的来源、字面值与镜像锁（`tests/regional/test_dcgm_exporter_release_pins.py`），
并按新 release 的 go-dcgm（其 `go.mod`）的 `pkg/dcgm/const_fields.go` 重新读取 counters 字段表
（`tests/regional/test_dataplane_dcgm_counters.py` 的 `PINNED_FIELD_IDS`）。

## 决策语义

策略结果分别保存：

- `official_action`：NVIDIA 原始 Immediate Action/workflow 名称；SXID 记本系统按 Fabric Manager 表选定的分支（Always-Fatal 的
  `RESTART_HOST_OR_SBR` 记为 `RESTART_BM`，scope 相关码记 `RESET_PARTICIPATING_GPUS`、`RESET_ALL_GPUS_AND_NVSWITCHES` 或
  `RESOLVE_TRUSTED_LINK_SCOPE`）。
- `investigatory_action`：NVIDIA 原始 Investigatory Action。
- `action`：当前控制面经过批准且能够表达的执行动作。
- `safety_action`：官方 workflow 因证据或 executor 不完整而阻塞时的站点
  fail-closed 动作，当前仅允许 `QUARANTINE`。
- `source`：`NVIDIA_CATALOG`、`NVIDIA_XID_154`、
  `NVIDIA_FABRIC_MANAGER` 或 `SITE_SAFETY`。
- `disposition`：等待关联（`PENDING_CORRELATION`）、可执行（`EXECUTABLE`）、只监控（`MONITOR_ONLY`）、缺 workflow 阻塞
  （`BLOCKED_WORKFLOW`）、缺证据阻塞（`BLOCKED_MISSING_EVIDENCE`）、不适用（`NOT_APPLICABLE`）或站点安全（`SITE_SAFETY`）；
  后四种且没有执行动作时 `safety_action=QUARANTINE`。

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
`BatchRebootClusterNodes` 执行；`eks`（普通 EKS，`eks-managed`）profile 同样设为 `REBOOT_NODE`，
由 EC2 `RebootInstances` 重启 Node providerID 绑定的实例（`node_provider.restart_vm_effective_action`）；
`kubernetes` profile 不映射，保持引擎的 BLOCKED 判定。该映射只用于 XID（XID 151），SXID 11004/12028 的
`RESTART_VM` 不改写。该映射不表示 EC2 stop/start；stop/start 和
节点 replacement 的宿主机放置、本地存储及身份语义仍是独立动作。

`CONTACT_SUPPORT`、`CHECK_MECHANICALS`、`UPDATE_SWFW` 不会被粗略替换成另一个
“官方动作”：策略保留原始动作并返回 EXECUTABLE，由 workflow 编译器分别落成
`ESCALATE_SUPPORT`（固定模板邮件，不隔离）、`CHECK_MECHANICALS`（通知并等待人工确认）
和含 `UPDATE_SOFTWARE_FIRMWARE` 的停机链（需配置 `GPU_FAULT_TARGET_FIRMWARE_VERSION`，未配置时计划阻塞，只执行安全链
FREEZE_EVIDENCE → MARK_UNSCHEDULABLE → QUARANTINE）。没有注册 resolver 的其他 workflow 名保留
原始动作、返回 `BLOCKED_WORKFLOW`，并通过独立的 `safety_action=QUARANTINE` 阻止
继续调度。逐 XID 的判定与步骤见
[故障类别与处置动作总表](../故障类别与处置动作总表.md)。

未知 XID 使用 `SITE_SAFETY` 来源和 `safety_action=QUARANTINE`，明确表示它是本系统的 fail-closed
安全策略，不是 NVIDIA Catalog 建议。

Catalog 的 `A100/H100/B100/GB200` 列按产品系列解释。具体 SKU 会规范化为
`A* -> A100`、`H*/GH* -> H100`、`B* -> B100`、`GB* -> GB200`，再按每条 XID
允许的系列执行适用性门禁。例如 H200/H800 属于 H100 列，B200 属于 B100 列，
GB300 属于 GB200 列。

## 已实现的官方 workflow

- XID 45：先持久化为 `PENDING_CORRELATION`，等待 30 秒窗口闭合；伴随其他 XID
  时复用最严重伴随 XID 的 incident/workflow，solo 时保留 `RESTART_FM`。属于 catalog 产品系列但未分级的节点——A100
  （NVIDIA 自己的 solo 情形）、PCIe/NVL 变体、GH200、GB200/GB300、B100 等不受 fabric 分级的部件，以及 exporter 不报告
  字段 170（或只报告 NOT_SUPPORTED/UNRECOGNIZED/NVML_TOO_OLD）的 Hopper/Blackwell 节点——不论下述开关如何，都把它编译为 `RESTART_FABRIC_MANAGER`
  （`FREEZE_EVIDENCE → RESTART_FABRIC_MANAGER`），由 Node Agent 的 `fabricManagerRestart` 能力执行。
  在受分级的 NVSwitch 节点（最新 DCGM 样本里 Hopper SXM 或 HGX B200/B300 部件报告字段 170）上，单独重启
  Fabric Manager 会丢失 fabric 状态，本方案因此把那里的 `RESTART_FM` 编译为 NVIDIA 对 H100 及之后的流程，即完整
  GPU + NVSwitch reset；NVIDIA 的 GPU 排障指南则把这种 solo XID 45 判为信息性（见下文「依据」）。
  因此这类节点默认**只观测**（`orchestration/xid45_policy.py`）：判定改为 `MONITOR_ONLY` /
  `NO_ACTION`，`official_action` 仍为 `RESTART_FM`、`investigatory_action` 不变，marker 为 INFO 且
  不 active；incident 以 RECOVERED 记录并带证据，`policy_reference` 为排障指南的 URL，reasons 写明
  原因与开关；照常发送 XID 调查通知；不建 workflow、不下发节点动作、不 cordon。控制面环境变量
  `GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`（与 SXID remediation 列表同一种「catalog 动作的
  站点变体」机制；默认不设置 = 只观测）恢复 catalog 行为：`RESTART_FM` 在被分级的 GPU 上编译为完整
  GPU + NVSwitch reset 链（停止并重启 Fabric Manager 的 quiesce 窗口内的 `RESET_ALL_GPUS_NVSWITCHES`，
  随后 VALIDATE_GPU 与 VALIDATE_FABRIC），使用 `fabricReset` 能力。共享
  Store lease 支持 HA、乱序到达和 Pod 重启。
- XID 48：solo 执行 `RESET_GPU`；30 秒内同 GPU、同 boot 已存有 XID 63/64 时记为 `DRAIN_AND_RESET`（同一条 RESET_GPU 链，
  预先 cordon），只并入 XID 64 的 incident、从不并入 XID 63 的；之后到达的 XID 64 回链到这个 XID 48，同一 GPU 只 reset 一次。
- XID 94/95：分别保存 application/all-applications containment；XID 95 在 reset
  前要求停止受影响工作负载。
- XID 154：驱动报告动作优先。Catalog 的 `Xid 154 linkage` 描述联动上报版本，不是原始 Xid 的最低支持版本；旧驱动或缺少版本信息不因此阻止 Xid 48/63/79/94/95 等原始规则。NVLink5 的 R575 位格式边界和必需解码证据仍保留。DCGM 字段 1523 以状态的形式持有同一个值：取值 1-4 按同标签的 XID 154 判定，XID 154 负责该 GPU 的恢复时只作证据（见下文「依据（Evidence）：DCGM 字段审计补入的规则」）。
- `DRAIN_P2P`：禁止调度、停止关联任务、quiesce、确认无 GPU 客户端后，由独立 `RECOVER_DRAIN_P2P` operation 检查目标 GPU 的 persistence 已关闭并重读 GPU Recovery Action。`None` 不 reset；`Reset`、`Drain and Reset`、仍为 `Drain P2P` 时复用既有受保护的 GPU reset；任一目标要求 `Reboot` 时不先 reset 其他目标，转节点重启。未知动作、缺失/重复 GPU、persistence 未释放等不授权 reset/reboot，交人工处理。独立 operation 让旧 Agent 拒绝不认识的请求，不能静默退化为 `STOP_WORKLOADS` 或无条件 reset。
- XID 159 `CHECK_UVM`：事件显式带 `uvm_in_use=true` 时 reset，`false` 时 ignore；缺少该证据时 `BLOCKED_MISSING_EVIDENCE` +
  `safety_action=QUARANTINE`（本系统的 fail-closed 选择）。产品自带的采集器不提供该字段，因此采集到的 XID 159 目前都按缺证据处理。
- XID 144-150：按驱动 R575 边界选择 V1/V2 `IntrInfo` pattern，同时匹配
  Error Status 和 Action 2。主动作与 Action 2 可以使用不同的位模式，分别匹配；
  RXPIPE 次级模式命中仍须满足同一表项的 Error Status，不能先要求主模式也命中。
  无官方表项匹配时阻塞，不猜测动作。

关联原文达到保留期限后重投同一事件，必须复用尚存的最终决策及 incident 绑定。
原文被清理不代表事件从未处理，不能用未关联的策略缓存结果覆盖最终决策；
仍处于 `PENDING_CORRELATION` 的记录继续按原窗口修复流程处理。

### 依据（Evidence）：受分级 NVSwitch 节点上 solo XID 45 的只观测默认

两处页面均已对照 NVIDIA 在线页面核实：

- NVIDIA GPU Debug Guidelines — GPU Node Triage，章节「Understanding Xid Messages」，XID 表中的行
  「45 Robust Channel Preemptive Removal」；docs version 615，页面 "Last updated on Oct 04, 2026"：
  `https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages`
  原文（英文，逐字）：
  "No action, informative only. Indicates channels affected by another failure. On A100, this error could be seen by itself due to unexpected Fabric Manager shutdown when FM is running in the same OS environment as the GPU. Otherwise, this error is safe to ignore as an informational message."
- NVIDIA XID Catalog，docs version 615，页面 "Last updated on Sep 09, 2026"：
  `https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`；spreadsheet
  `https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx`
  （sha256 `7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5`，与上文固定的 XLSX 逐字节相同）。
  「Xids」sheet 的 Code 45：Immediate Action `WORKFLOW_XID_45`；Investigatory Action
  "Solo: RESTART_FM / Not Solo: IGNORE (follow other Xid)"；Trigger：
  "This event is logged when the user application aborts and the kernel driver tears down the GPU application running on the GPU. Control-C, GPU resets, sigkill are all examples where the application is aborted and this event is created. In many cases, this is not indicative of a bug but rather a user or system action."

结论（本方案的解读与偏离）：NVIDIA 自己的排障指南把 catalog 的「Solo: RESTART_FM」解释为 A100 的情形（Fabric Manager 与 GPU
在同一 OS 中、意外停止），其余 solo XID 45 判为信息性。默认动作只在 NVIDIA 自己的排障指南判为信息性的
地方——受分级的 NVSwitch Hopper/Blackwell 节点——偏离 catalog 的 investigatory action；catalog 数据本身
不变（生成物、摘要与 `mapping_version` 照旧）；A100 保留 `RESTART_FM`（NVIDIA 自己的 solo 情形）。只观测
incident 的 `policy_reference` 就是上面排障指南的 URL；站点需要 catalog 行为时设置
`GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`。

### 依据（Evidence）：复位后的 VALIDATE_GPU 以更新的批次重判 pending 行重映射 / 页退役

以下页面均已对照 NVIDIA 在线页面核实（docs version 615）：

- NVIDIA GPU Memory Error Management，章节「Error Recovery and Response Flags」，条目「Row-remapping
  pending flag」；页面 "Last updated on Sep 09, 2026"：
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/error-recovery-and-response-flags.html#error-recovery-and-response-flags`
  原文（英文，逐字）：
  "This flag indicates that row-remapping will happen at the next GPU reset."
- 同一手册，章节「Row Remapping」：
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html#row-remapping`
  原文："The process of row-remapping requires a GPU reset to take effect and will remain persistent throughout the life of the life of the GPU."
  Table 2「Page Retirement vs. Row-Remapping」的「Application of pending changes」行：Ampere 及更新
  "GPU reset is required."；早期页退役 "Needed a kernel module reload or driver re-initialization or GPU reset"。
- NVIDIA Dynamic Page Retirement，章节「Visibility」>「nvidia-smi」；页面 "Last updated on Sep 09, 2026"：
  `https://docs.nvidia.com/deploy/dynamic-page-retirement/visibility.html#nvidia-smi`
  原文："The nvidia-smi “Pending Page Blacklist” field indicates whether a page has been recently retired and will be blacklisted on the next system reboot or driver load."
  同一文档「Blacklisting and ECC Error Recovery」>「Reattaching the GPU」
  （`https://docs.nvidia.com/deploy/dynamic-page-retirement/blacklisting-and-ecc-error-recovery.html#reattaching-the-gpu`）
  列出应用待退役页的方式，按代价从低到高："Re-attach the GPUs (persistence mode disabled only)"、
  "Reset the GPUs"、"Reload the kernel module (nvidia.ko)"、"Reboot the machine (or VM)"。
- NVIDIA GPU Debug Guidelines — GPU Node Triage，「Understanding Xid Messages」中 63 行
  「A100: Row-remapping recording event」；页面 "Last updated on Oct 04, 2026"：
  `https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#understanding-xid-messages`
  原文节选："until there is a convenient time to reset the GPU (refer to GPU reset capabilities/limitations section below) or reboot for row remapping to activate."

结论：pending 行重映射 / 页退役在下一次 GPU reset（或节点重启）时才生效，所以本 workflow 的
`RESET_GPU`（复位过该 GPU）、`RESET_ALL_GPUS_NVSWITCHES` 或 `RESTART_NODE` 完成后，读数不晚于该完成
时刻的 pending finding 不是复位后 GPU 的判定依据：只要阻塞的 finding 全是 pending 修复标志（`row_remap_pending`/
`retired_pages_pending`），`VALIDATE_GPU` 继续 `WAITING`（`node_pending.<node>`
的 `awaiting_newer_gpu_metrics` 与 `pending_repair_reset`，`policy_references` 为上面对应标志的 URL），
等比复位更新的 GPU 指标批次重判；清除即成功，仍置位即按既有路径失败；上限为执行器单步上限；同时有其他 finding 时照旧立即失败。
`RECOVER_DRAIN_P2P`（可能 `reset_skipped`）、`REMEDIATE_DRIVER`、`UPDATE_SOFTWARE_FIRMWARE`、
`REPLACE_NODE` 不计为应用修复。实现 `src/gpu_fault/adapters/pending_repair_rejudge.py`，引文由
`tests/execution/test_validation_pending_repair_after_reset.py` 钉住；该情形未在真机出现，行为由单元测试覆盖
（详见 [详细设计](../详细设计.md) §2.4.1「GPU Metrics `gpu_metrics.py`」中 pending 修复标志一段）。

### 依据（Evidence）：DCGM 字段审计补入的规则

下列页面均已重新抓取核对；NVML 与 NVIDIA GPU Memory Error Management 为 docs version 615（页面 "Last updated on Sep
09, 2026"），DCGM 页面为 "Version: 4.6 (Latest)"，DCGM 源码为 v4.7.0 标签。完整引文与决定见详细设计 §2.4.1「DCGM 字段审计补入的
字段」；每条引文由 `tests/test_dcgm_gap_field_evidence.py` 钉住。

- GPU 恢复动作（1523）：GPU Memory Error Management，"Error Recovery and Response Flags"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/error-recovery-and-response-flags.html#error-recovery-and-response-flags`
  "Recovery action flag will be triggered for uncorrectable errors."；NVML API Reference，"vGPU Structs"，
  `nvmlDeviceGpuRecoveryAction_t`
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlVgpuStructs.html#_CPPv429nvmlDeviceGpuRecoveryAction_t`
  "NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN Recover IMEX Domain."、"NVML_GPU_RECOVERY_ACTION_BUS_RESET Reset the GPU’s PCIe
  bus."、"NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT Reboot the system."；DCGM Release Notes 4.7.0
  `https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html` "Added DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN for the
  separate case in which driver recovery requires restoring an IMEX domain."；NVML API Reference 变更记录
  `https://docs.nvidia.com/deploy/nvml-api/latest/change-log.html`，"Changes between NVML v610 and v615"："Added
  NVML_GPU_RECOVERY_ACTION_BUS_RESET and NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT to nvmlDeviceGpuRecoveryAction_t."；nvidia-smi 文档
  `https://docs.nvidia.com/deploy/nvidia-smi/index.html`，"-r, --gpu-reset"："Reset triggered without extra arguments, will be a
  default Function Level Reset (FLR). To issue a Bus Reset, use -r bus."
- PCIe 链路（235-238）：nvidia-smi 文档，GPU ATTRIBUTES › PCI › GPU Link information："The current link generation and width.
  These may be reduced when the GPU is not in use."、"The maximum link generation and width possible with this GPU and system
  configuration."；NVIDIA DGX B300 User Guide
  `https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html#key-enhancements-over-previous-dgx-systems`
  "ConnectX-8 networking connects directly to GPUs via PCIe Gen6."
- PCIe replay（202）与 correctable 错误（1501）：DCGM Feature Overview，"Background Health Checks" › "Failure Conditions"
  `https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#background-health-checks` "There were more
  than 8 PCIe replays in the last minute."；DCGM v4.7.0 `modules/health/DcgmHealthWatch.cpp`
  `https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp` "Expected replay rate = Replay rate per lane *
  Num of pcie Lanes. Note that value is ceiled."；DCGM v4.7.0 `nvvs/plugin_src/pcie/Pcie.cpp`
  `https://github.com/NVIDIA/DCGM/blob/v4.7.0/nvvs/plugin_src/pcie/Pcie.cpp` "/* Default is hopper, but fall back for other
  devices */"；NVML API Reference，"Field Value Enums"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html#c.NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS`
  "PCIe Correctable Errors Counter."（没有健康阈值）。
- SRAM 阈值（346）、无备用行（389）、不可纠正重映射（393）：GPU Memory Error Management，"RMA Policy"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html#gpu-l2-memory-rma-policy`
  "Any of the following events will trigger the SRAM Threshold Exceeded flag:"；同一页面 "GPU DRAM Memory RMA Policy"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html#gpu-dram-memory-rma-policy`："For Blackwell GPUs, a third remapping attempt for an uncorrectable memory error on a bank
  invokes HBM channel repair if there is a spare channel available."；DCGM Release Notes 4.6.0 "Health monitoring now warns when
  total uncorrectable row remaps reach 512 on Ampere and newer GPUs, allowing administrators to run field diagnostics before GPU
  memory health degrades further."
- HW slowdown 时间（1423/1424）与时钟事件位：NVML API Reference，"Field Value Enums"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html` "Field values for Clock Throttle Reason
  Counters All counters are in nanoseconds."；NVML API Reference，"Clocks Event Reasons"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html`，`nvmlClocksThrottleReasonHwSlowdown`：
  "May be also reported during PState or clock change This behavior may be removed in a later release."
- ECC 模式（300/301）与 InfoROM：NVML API Reference，"Device Queries"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html` "Requires ECC Mode to be enabled."；
  Dynamic Page Retirement，"Implementation"
  `https://docs.nvidia.com/deploy/dynamic-page-retirement/implementation.html#implementation` "These addresses are stored in the
  InfoROM."

结论：字段 1523 的取值 1-4 按同标签的 XID 154 判定；RECOVER_IMEX_DOMAIN 交运维；BUS_RESET、SYSTEM_REBOOT 与未记载的值 fail
closed（NVIDIA 对 6 和 7 只记载一行枚举含义，且产品的 GPU 复位是 FLR，不是 bus reset）。PCIe replay 阈值按 DCGM health 对当前链路
的公式（只有没有链路读数时为 8/min；配置了不同于 8 的 `GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE` 时以配置为准）；correctable 错误上限取该公式与 DCGM PCIe 诊断上限中较低者。Blackwell（B200/B300）的行为依据
NVIDIA 文档做了单元测试，未经现网验证（没有 B 系列硬件）。

### 依据（Evidence）：HGX B200/B300 的 DCGM 规则

下列页面均已重新抓取核对：NVML API Reference 与 NVIDIA GPU Memory Error Management 为 docs version 615，DCGM
页面为 "Version: 4.6 (Latest)"，DCGM 源码为 v4.7.0 标签，Fabric Manager User Guide 为 "Last updated on Oct 07, 2026"，
nvidia-smi 文档为 v610 时期页面，fabric-health 的位值取自 NVIDIA 的 `nvidia-ml-py` 13.615.71。完整引文与决定见详细设计
§2.4.1「HGX B200/B300 DCGM 字段」；每条引文由 `tests/test_blackwell_dcgm_evidence.py` 钉住。
Blackwell（B200/B300）的行为依据 NVIDIA 文档做了单元测试，未经现网验证（没有 B 系列硬件）。

- NVLink 完整性（1211-1213 作为数据链路 counter 497-499 的备选项）：NVML API Reference，"Field Value Enums"，
  `NVML_FI_DEV_NVLINK_ERROR_DL_CRC`
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html#c.NVML_FI_DEV_NVLINK_ERROR_DL_CRC`
  "This is unsupported for Blackwell+ Please use NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_*"。
- NVLink5 counter（1204-1209、1211-1214、1219、1220）：DCGM Field Identifiers
  `https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html` "Returns aggregate value across all
  links. Not supported on NVLink4 and earlier."；DCGM Feature Overview，"Background Health Checks" › "Failure Conditions"
  `https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#background-health-checks` "An NVLink
  reported 1 or more errors. This includes datalink CRC, recovery, and replay errors (pre-Blackwell GPUs) and link recovery
  events (Blackwell+ GPUs)."；DCGM v4.7.0 `modules/health/DcgmHealthWatch.cpp`
  `https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp`：1212 大于 0 记 FAIL，1211 大于 0 记
  WARN，1204-1209、1213、1214、1219、1220 一分钟内有任何变化记 FAIL，下一步 "Run a field diagnostic on the GPU."；DCGM Release Notes
  4.7.0 `https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html` "Removed NVLink bit-error-rate
  thresholds that caused false health and diagnostic failures on healthy GB200 systems."（BER 不分级）。
- Fabric health（174/175）：NVML API Reference，"Fabric Definitions"
  `https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFabricDefs.html` "Fabric Health Mask: Route is
  unhealthy."、"Fabric Health Mask: Bandwidth degraded."、"Fabric Health Summary: Unhealthy."、"Fabric Health Summary:
  Limited Capacity."；DCGM Field Identifiers
  `https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html#c.DCGM_FI_DEV_FABRIC_HEALTH_MASK` "Use
  DCGM_GPU_FABRIC_HEALTH_TEST macro to check the different health statuses."；DCGM v4.7.0 `MonitorFabricFields` 下一步
  "Check DCGM and system logs for errors. Reset GPU. Restart DCGM. Rerun diagnostics."；nvidia-smi 文档
  `https://docs.nvidia.com/deploy/nvidia-smi/index.html`，"Fabric" › "Health"："Summary - Summary of Fabric Health
  <Healthy, Unhealthy, Limited Capacity>"。
- 内存修复：GPU Memory Error Management，"RAS Repair" › "GPU Memory Repair"
  `https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/hbm-channel-repair.html#gpu-memory-repair` "The process of channel
  repair requires a reboot to take effect."；nvidia-smi 文档，"ECC Errors"："If one of the repair flags is pending, check the
  GPU Recovery action and take the appropriate steps."；DCGM Field Identifiers
  `https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html#c.DCGM_FI_DEV_MEMORY_UNREPAIRABLE`
  "Unrepairable memory flag indicating if memory has unrepairable errors 1=yes, 0=no"；NVML API Reference 变更记录
  `https://docs.nvidia.com/deploy/nvml-api/latest/change-log.html`，"Changes between v580 and v590"："Added
  nvmlDeviceGetUnrepairableMemoryFlag_v1() to report whether or not there is unrepairable memory."

结论：在 HGX B200/B300 上，NVLink5 link-recovery counter 顶替 NVML 不再报告的 NVLink 字段；链路恢复失败（1212）即 drain，
其他 NVLink5 counter 按 NVLink 家族规则判 WARNING，跨产品窗口的增量不判，XID 144-150/155 负责该 GPU 期间的增量只作证据；
DCGM 字段 230 的 XID 变化与同一 GPU 120 s 内的内核行视为同一次发生（见[故障类别与处置动作总表](../故障类别与处置动作总表.md)第四部分），
XID 144-150 的变化最多等 120 s 以待内核行。不健康（summary UNHEALTHY、路由不健康或配置错误）在产品 fabric 窗口之外保持
≥ 60 s 即确认，执行 NVIDIA 的完整 GPU + NVSwitch reset；路由恢复或访问超时恢复进行中只作证据，保持 300 s 后同样处理；
降级的 fabric 运行诊断并通告运维，NOT_SUPPORTED 不判；该规则作用于全部受分级 NVSwitch 部件（含 Hopper SXM）。DCGM 对
NVLink5 错误 counter 与这些 fabric 状态立即记 FAIL，产品的 WARNING 分级、确认等待与降级不 reset 是本方案的偏离。pending 的内存修复交运维 reboot，除非 GPU 恢复动作另有判定；
unrepairable memory 走 row-remap failure 的 RMA drain。逐链路字段、NVSDM 交换芯片字段与 ConnectX bridge 字段延后
（[验证局限](../validation-limitations.md)）。

## SXID

SXID 恢复依据 Fabric Manager User Guide（`https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html`，
生成物 `src/gpu_fault/data/nvidia-fabric-manager-sxid-2025-11-14.yaml` 钉住 "2025-11-14" 版的 Table 21-24）：

- non-fatal：catalog 中 IGNORE 的码与表外的 non-fatal 码为信息性，保持监控；non-fatal 的 19084、11004/12028 与 20012 按下面各自的规则。
- fatal access link：access 范围须来自可信 NVSwitch 端口拓扑（`TRUSTED_NVSWITCH_TOPOLOGY`）或 HGX H100/H200 的产品不变量
  （`NVIDIA_PRODUCT_INVARIANT`），否则 `BLOCKED_MISSING_EVIDENCE` + QUARANTINE；参与 GPU 集合非空时 cordon、停止任务并 reset
  该组 GPU（`RESET_PARTICIPATING_GPUS`），为空时阻塞。集合是可信拓扑给出的端口 GPU、与 NVSwitch PCI 地址同槽的 GPU
  与节点上活动受管任务的全部 GPU 三者之并——这些任务由同一 workflow 停止；没有点名的 GPU 时无法从中分出受影响的任务。
  依据：Fabric Manager User Guide，"Runtime NVSwitch and GPU Errors" > "NVSwitch SXid Errors" > "Fatal SXid Errors"
  （页面 "Last updated on Oct 07, 2026"），`https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#fatal-sxid-errors`：
  "When a fatal SXid error is reported on a NVSwitch port that connects a GPU and an NVSwitch, the corresponding error
  will be propagated to the GPU. The CUDA jobs that are running on that GPU will be aborted" 与 "Reset the specified GPU
  and all the participating GPUs in the affected workload"（由 `tests/app_services/test_access_link_sxid_scope.py` 钉住）。
- fatal trunk 与 10003/19084：要求 fabric_partition 和完整节点 GPU inventory，齐备时
  执行 `RESET_ALL_GPUS_AND_NVSWITCHES`（Node Agent `fabricReset` 能力，多节点 barrier），
  缺一即 `BLOCKED_MISSING_EVIDENCE`，不会退化成单 GPU reset。完整指恰为节点声明的预期
  GPU 数（collector 的 `GPU_FAULT_EXPECTED_GPU_COUNT` 或机型数量，即 inventory snapshot 的
  `expected_gpu_count`）：新鲜的 inventory snapshot（一次 nvidia-smi 枚举）列出的 GPU
  少于其声明数时不算；没有新鲜 snapshot 时，事件前后窗口内的最新逐 GPU 指标行只有恰好覆盖
  节点最新 snapshot 声明的数量才算——这些行不是同一次投递，事件早于最新投递时窗口里留下的
  正是停止刷新的行（例如 collector 重启后空闲 GPU 上的 `pcie_link_downtrain` 判定行停在
  最后一次投递）；可信端口拓扑为 SXID 端口给出的 GPU 也不算。不完整时事件不带
  participating GPU，走既有的证据冻结、cordon、quarantine 与运维审核路径。
- always-fatal（Table 23 钉死的 20 码）：执行 `REBOOT_NODE`，事前 cordon 并停止任务，不需要 fabric_partition
  或 GPU inventory；不在钉死表内的码声明 Always-Fatal 或 Fatal 时一律阻塞（`BLOCKED_MISSING_EVIDENCE`），不凭
  客户端给出的分级授权任何破坏性恢复。
- `RESTART_VM`（11004、12028，Shared NVSwitch/vGPU 场景）：裸金属没有 guest VM 生命周期，`BLOCKED_WORKFLOW` +
  `safety_action=QUARANTINE`（隔离是本系统的 fail-closed 选择）；`CHECK_MECHANICALS`（20012）：`ESCALATE_OPERATOR`，编译为 FREEZE_EVIDENCE → CHECK_MECHANICALS（通知并等待人工确认，不 cordon、不停任务）。
- Blackwell 系列（B100/GB200 产品系列，含 B200/B300、GB200/GB300）：传统 SXID 不适用，判 `NOT_APPLICABLE`（investigatory
  `USE_DCGM_NVSDM_TELEMETRY`）+ `safety_action=QUARANTINE`（本系统的 fail-closed 选择）；产品不属于任何已知系列时
  `BLOCKED_MISSING_EVIDENCE`（`VERIFY_GPU_PRODUCT_GENERATION`）。

SXID 的 `classification_source` 必须是 `NVIDIA_FABRIC_MANAGER`（或 `NVIDIA_FABRIC_MANAGER_CATALOG`、
`NVIDIA_FABRIC_MANAGER_RUNTIME`），否则因证据来源不可信而阻塞；事件观察到的 fatal/non-fatal 与钉死表冲突时同样阻塞。
