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
整数单元格拒绝布尔值，不能把 `TRUE`/`FALSE` 当成 XID 编号；整数、整值浮点和整数文本
保留原规范化行为。只读工作簿在成功及解析失败后都会关闭，避免重复生成时累积文件句柄。

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

NVIDIA 的规则（原文）：

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
distroless 变体。各集群的配对（普通 EKS host 模式为受支持的配对；HyperPod 的引擎由 AMI 决定，现网 Agent 心跳报告 4.7.0，比 4.6.1
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

`CONTACT_SUPPORT`、`CHECK_MECHANICALS`、`UPDATE_SWFW` 不会被粗略替换成另一个
“官方动作”：策略保留原始动作并返回 EXECUTABLE，由 workflow 编译器分别落成
`ESCALATE_SUPPORT`（固定模板邮件，不隔离）、`CHECK_MECHANICALS`（通知并等待人工确认）
和含 `UPDATE_SOFTWARE_FIRMWARE` 的停机链。没有注册 resolver 的其他 workflow 名保留
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
  时复用最严重伴随 XID 的 incident/workflow，solo 时保留 `RESTART_FM`。未分级的节点——A100
  （NVIDIA 自己的 solo 情形）、PCIe/NVL 变体、GH200、GB200/GB300、未知部件，以及 exporter 不报告
  字段 170 的 Hopper/Blackwell 节点——不论下述开关如何，都把它编译为 `RESTART_FABRIC_MANAGER`
  （`FREEZE_EVIDENCE → RESTART_FABRIC_MANAGER`），由 Node Agent 的 `fabricManagerRestart` 能力执行。
  在受分级的 NVSwitch 节点（最新 DCGM 样本里 Hopper SXM 或 HGX B200/B300 部件报告字段 170）上，单独重启
  Fabric Manager 会丢失 fabric 状态，`RESTART_FM` 在那里只能是 NVIDIA 对 H100 及之后的流程，即完整
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
- XID 48：solo 执行 `RESET_GPU`；伴随 XID 63/64 时执行
  `DRAIN_AND_RESET`。
- XID 94/95：分别保存 application/all-applications containment；XID 95 在 reset
  前要求停止受影响工作负载。
- XID 154：驱动报告动作优先并原样映射。
- XID 159 `CHECK_UVM`：显式确认使用 UVM/vGPU 时 reset，否则 ignore；缺少证据阻塞。
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

结论：NVIDIA 自己的排障指南把 catalog 的「Solo: RESTART_FM」解释为 A100 的情形（Fabric Manager 与 GPU
在同一 OS 中、意外停止），其余 solo XID 45 判为信息性。默认动作只在 NVIDIA 自己的排障指南判为信息性的
地方——受分级的 NVSwitch Hopper/Blackwell 节点——偏离 catalog 的 investigatory action；catalog 数据本身
不变（生成物、摘要与 `mapping_version` 照旧）；A100 保留 `RESTART_FM`（NVIDIA 自己的 solo 情形）。只观测
incident 的 `policy_reference` 就是上面排障指南的 URL；站点需要 catalog 行为时设置
`GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`。

## SXID

SXID 恢复依据 Fabric Manager User Guide：

- non-fatal：信息性，保持监控。
- fatal access link：必须已解析 affected GPU 和 workload participating GPUs，
  然后停止任务并 reset 整组 GPU。
- fatal trunk 与 10003/19084：要求 fabric_partition 和完整节点 GPU inventory，齐备时
  执行 `RESET_ALL_GPUS_AND_NVSWITCHES`（Node Agent `fabricReset` 能力，多节点 barrier），
  缺一即 `BLOCKED_MISSING_EVIDENCE`，不会退化成单 GPU reset。
- always-fatal（Table 23 钉死的 20 码）：执行 `REBOOT_NODE`，事前 cordon 并停止任务；
  事件声明 Always-Fatal 但不在表内时阻塞。
- B200/B300：传统 fatal/non-fatal SXID 不适用，要求 DCGM/NVSDM telemetry。

SXID 的 `classification_source` 必须是 `NVIDIA_FABRIC_MANAGER`，否则因证据来源
不可信而阻塞。
