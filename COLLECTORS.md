# GPU Fault Collectors

## 数据路径

`gpu-fault-collector` 是只读采集进程，不执行节点或任务恢复：

| Collector | 数据源 | 控制面入口 | 幂等 ID |
|---|---|---|---|
| `kernel` | 节点 `/dev/kmsg` | `/v1/collector-events/nvidia-kernel` | boot ID + kmsg sequence |
| `kubernetes-hma` | Kubernetes Node list/watch | `/v1/provider-events/hyperpod-hma/kubernetes-node` | Node resourceVersion + fault content |
| CloudWatch Lambda | HMA Logs subscription | `/v1/provider-events/hyperpod-hma/cloudwatch` | CloudWatch log event ID |
| `dcgm` | DCGM Exporter Prometheus endpoint | `/v1/collector-events/gpu-metrics` | node + scrape timestamp |
| `nvidia-smi` | 本机 NVIDIA CLI | `/v1/collector-events/gpu-metrics` | node + sample timestamp |

kernel collector 只发送包含 `NVRM ... Xid` 或 `SXid` 的行，不发送完整内核日志。
HMA watcher 只发送包含 `sagemaker.amazonaws.com/node-health-status`、
`fault-types`、`fault-reasons` 或 `fault-details` 的 Node。

默认HyperPod生产拓扑启用：

- kernel、fabric-manager、dcgm、host节点采集器；
- kubernetes-node-resources集群采集器；
- completion watcher提供attempt/workload observation。

`NodeLogCollector` 当前默认禁用，不进入生产必需集合。安装器只在隔离验证环境显式
传入 `--enable-node-log-collector` 时启用。

`nvidia-smi`保留为DCGM Exporter不可用时的fallback。Kubernetes HMA和
CloudWatch HMA属于optional验证链路，不在默认拓扑中。

## 公共配置

```text
GPU_FAULT_CONTROL_PLANE_URL=http://gpu-fault-api:8080
GPU_FAULT_CONTROL_PLANE_TOKEN=<optional bearer token>
GPU_FAULT_CLUSTER_ID=<cluster>
GPU_FAULT_RUNTIME_PROFILE_VERSION=<registered profile>
GPU_FAULT_GPU_PRODUCT=H100
GPU_FAULT_DRIVER_BRANCH=575
GPU_FAULT_CUDA_VERSION=12.9
```

生产环境必须先向控制面注册对应的 runtime profile。token 应由 Secret 注入，不能写入
ConfigMap。当前应用本身尚未实现 bearer token 校验，生产入口必须由 API Gateway、
service mesh 或反向代理验证 token，并由 NetworkPolicy、安全组或私有负载均衡限制。

## Kubernetes

安装 collector 依赖并构建镜像：

```bash
python3 -m pip install '.[collectors]'
```

历史通用 Kubernetes 模板位于
`examples/legacy/kubernetes/collectors.yaml`，只用于协议和迁移参考，不是受支持的
生产部署入口。区域 HyperPod 数据面使用 `deploy/dataplane/` 清单以及节点
Installer/Reconciler。HMA watcher 只有 Node `get/list/watch` 权限，不具备 patch、
cordon、drain 或 Pod 权限。

当前 HyperPod 方案默认
`GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR=false`，不部署
`gpu-fault-hma-watcher`。现阶段控制面只消费 HMA Node 数据中的 XID/SXID，而节点
Kernel Collector 和 Fabric Manager Collector 已分别覆盖这两类信号。禁用 watcher
不会关闭 SageMaker HMA，也不会删除 Node 上的 HMA labels/annotations。仅在单独验证
该转发链路时显式设置为 `true`。

`deploy/image/Dockerfile` 可用于构建 collector 镜像：

```bash
docker build -f deploy/image/Dockerfile -t gpu-fault-collector:0.6.1 .
```

kernel DaemonSet 读取宿主机 `/dev/kmsg`，通常需要 privileged/CAP_SYSLOG。应通过
nodeSelector 限制到 GPU 节点，并通过准入策略只允许固定镜像 digest。若平台禁止读取
`/dev/kmsg`，可改为受控的 journald/kern.log forwarder，但仍应保留原始 kmsg sequence
作为事件 ID。

本地运行：

```bash
gpu-fault-collector kernel --node-id worker-1
gpu-fault-collector kubernetes-hma
gpu-fault-collector dcgm --node-id worker-1 \
  --metrics-url http://127.0.0.1:9400/metrics
gpu-fault-collector nvidia-smi --node-id worker-1
```

EC2 或 HyperPod Slurm 节点可使用
`deploy/systemd/gpu-fault-kernel-collector.service`。安装 wheel 后，将 CLI 放在
`/opt/gpu-fault/venv/bin`，并把公共环境变量写入
`/etc/gpu-fault/collector.env`。

### GPU 实例一键安装

节点安装脚本同时部署 XID/SXID kernel collector 和 GPU metrics collector：

```bash
sudo deploy/node/install-gpu-fault-collector.sh \
  --control-plane-url https://gpu-fault.example.internal \
  --cluster-id <gpu-hyperpod-cluster> \
  --runtime-profile-version hyperpod-v1 \
  --node-id "$(hostname -f)" \
  --metrics-mode auto \
  --dcgm-exporter existing \
  --enable-node-agent \
  --node-action-secret "${NODE_ACTION_SECRET}" \
  --node-instance-id "${EC2_INSTANCE_ID}" \
  --node-agent-advertise-url "https://${NODE_PRIVATE_IP}:9099"
```

生成可上传到 S3、SSM Distributor 或节点镜像流水线的自包含安装包：

```bash
deploy/node/build-node-installer-bundle.sh
```

输出 `dist/<wheel-sha12>/gpu-fault-node-installer-<version>.tar.gz`，其中包含项目 wheel、安装/卸载/
自检脚本、systemd units 和 DCGM counter 配置。Python 第三方依赖没有重复打入该包；
隔离网络环境应同时准备 `--wheelhouse`。

`execution-token` 和启用 Node Agent 时的 `--node-action-secret` 必须是至少 32
字符且无末尾换行的值；带换行的 execution token 不能作为 HTTP header。使用
Kubernetes Secret 文件时应先规范化，并比较控制面与节点 node-action secret 的
SHA-256 指纹；不要把 token 或 secret 本身写入日志。节点默认 `python3` 低于 3.12
时，显式传入 `--python-command /usr/bin/python3.12`。

Kubernetes/HyperPod 安装应优先使用
`--node-action-secret-file /mounted-secret/node-action-secret`，避免 secret 出现在
命令行参数中。`deploy/node/run-hyperpod-installer-job.sh --node <node-name>` 提供
绑定单节点的 Job transport，可直接配合 fleet deployment 的逐 wave 协调使用。

独立节点安装器和HyperPod生产部署都默认不启用`NodeLogCollector`。
当前生产策略要求保持停用；仅在专项验证时通过
`--enable-node-log-collector`显式启用。启用后collector只发送命中共享
`NODE_LOG_RULES`的行，并每300秒发送空健康摘要，不上传整段journal。

`training-progress` reporter默认关闭。Host collector从procfs生成的rank
liveness是默认无侵入进展信号；只有需要step/loss/numerical-error语义时
才接入training-progress reporter。

采集面统一自检：

```bash
gpu-fault-config collector-readiness \
  --url https://CONTROL_PLANE \
  --cluster-id CLUSTER \
  --execution-token "${GPU_FAULT_EXECUTION_TOKEN}"
```

接口合并Agent心跳、collector systemd状态和各通道last-success；
`unit_state`与`unit_enabled`始终是字符串。

启用 Node Agent 后，Agent 启动时立即向
`POST /v1/fleet/agents/heartbeat` 注册，并默认每 30 秒续租。heartbeat 使用
node-action secret 做 HMAC 签名，内容包括 endpoint、package version、wheel
SHA-256、NVIDIA Catalog policy version、runtime profile、公共配置摘要、boot ID 和
allowed operations。Agent 协议 v3 还包含稳定 Node/实例 UID、incarnation和
五个collector systemd单元状态，并从
heartbeat 响应取得 generation；动作命令 generation 不一致时 fail-closed。安装脚本
优先使用显式 `--node-instance-id`，否则读取 DMI product UUID 或 machine-id。
HyperPod installer Job 自动传入 Kubernetes Node UID。`--node-agent-advertise-url`
必须是控制面实际可达地址；不要依赖 Pod 无法解析的节点本地主机名。
未显式提供 cert/key 时安装器会在节点生成本地 TLS cert/key；签名 heartbeat
携带服务端证书，控制面按证书 pin 访问。生产不得传
`--allow-node-agent-plaintext`。

节点 reboot/replace 不依赖 Agent 自己注销。控制面先用受 execution token 保护的
`/drain` 与 `/revoke` 接口提升 generation、终止 lease 并退休旧 incarnation。新 Agent
必须以新 boot ID 或实例 UID 注册后才能恢复 `ACTIVE`；禁止直接删除 registry 记录。

控制面通过以下接口协调批量升级：

```text
POST /v1/fleet/deployments
POST /v1/fleet/deployments/{id}/next-wave
POST /v1/fleet/deployments/{id}/nodes/{node}/status
GET  /v1/fleet/deployments/{id}
```

`next-wave` 原子占用 `maxUnavailable` 配额。EC2 使用 SSM、EKS 使用 DaemonSet/
Operator、HyperPod 使用 lifecycle script 执行该 wave；安装后的匹配 heartbeat 会
自动将节点标记为 `READY`。当前 wave 未全部 READY 时，控制面拒绝启动后续 wave。
`READY` 不能通过节点状态 API 人工写入，只能由目标 agent version、artifact
SHA-256、policy、runtime profile 和 config digest 全部匹配的签名 heartbeat 写入。

`gpu-fault-fleet run-deployment` 可执行完整的逐 wave rollout。transport command
按当前 wave 并行调用一次；EC2/HyperPod 可封装 SSM 或 lifecycle script，EKS 可封装
Operator/DaemonSet 节点选择逻辑。命令退出 0 只代表部署请求已接受，runner 仍会等待
目标 heartbeat，并在超时或命令失败时将当前节点标为 `FAILED`：

```bash
gpu-fault-fleet \
  --control-plane-url https://gpu-fault.example.internal \
  run-deployment \
  --execution-token "${GPU_FAULT_EXECUTION_TOKEN}" \
  --deployment-id fleet-deployment-... \
  --transport-command \
    '/opt/gpu-fault-fleet/deploy-node --node {node_id} --version {agent_version} --sha256 {artifact_sha256}' \
  --wave-timeout-seconds 900
```

transport command 不通过 shell 执行，支持 `{node_id}`、`{cluster_id}`、
`{deployment_id}`、`{agent_version}` 和 `{artifact_sha256}` 占位符，并同时收到同名
`GPU_FAULT_FLEET_*` 环境变量。全部 wave 完成后 runner 再执行一次 fleet readiness
门禁；任何节点 heartbeat 过期、能力缺失或版本身份不一致都会使 rollout 失败关闭。

`auto` 会先探测 `GPU_FAULT_DCGM_METRICS_URL`，存在 DCGM Exporter 时使用 DCGM。
`NvidiaSmiMetricsCollector` 当前未完成更多生产验证，默认禁用；DCGM 不可用时安装器
失败关闭，不再自动 fallback。若节点没有现成 exporter，可部署一个与 HMA 分离的专用
容器；镜像 tag 必须由集群管理员根据驱动/GPU 支持矩阵显式指定：

```bash
sudo deploy/node/install-gpu-fault-collector.sh \
  --control-plane-url https://gpu-fault.example.internal \
  --cluster-id <gpu-hyperpod-cluster> \
  --runtime-profile-version hyperpod-v1 \
  --dcgm-exporter docker \
  --dcgm-exporter-image nvcr.io/nvidia/k8s/dcgm-exporter:<approved-tag>
```

Docker 模式使用 host network，但 exporter 仅监听 `127.0.0.1:9400`，并挂载
项目的 DCGM counter 列表。它不进入 HMA Pod、不修改 HMA 的 `nv-hostengine`，也不
申请 Kubernetes `nvidia.com/gpu` 资源。安装器检测到已有 `nv-hostengine` 或 DCGM
endpoint 时会拒绝启动第二套 exporter，避免与 HMA、GPU Operator 或现有监控冲突。

安装后执行：

```bash
sudo /opt/gpu-fault/verify
sudo journalctl -u gpu-fault-kernel-collector -u gpu-fault-metrics-collector
```

卸载使用 `sudo /opt/gpu-fault/uninstall`。安装器要求 Python 3.12；离线节点通过
`--python-command` 指定非默认解释器，通过 `--wheel` 和 `--wheelhouse` 指定项目
wheel 及完整依赖目录。访问 token 写入权限为 `0600` 的
`/etc/gpu-fault/collector.env`，不会写入 systemd unit。自检不仅检查本地服务和
exporter，还会等待 `/v1/gpu-metrics/.../latest` 出现该节点的实际样本。

## GPU Metrics

优先使用 `dcgm`，由 `prometheus_client` 官方 parser 读取 DCGM Exporter。支持：

- GPU/memory 温度、功耗、GPU/memory 利用率、显存和时钟。
- volatile/aggregate SBE/DBE ECC。
- retired pages、pending retirement 和 row remap。
- PCIe replay、NVLink CRC/data/replay/recovery counter。
- power/thermal violation duration。
- `DCGM_FI_DEV_XID_ERRORS`。

DCGM Exporter 必须配置 `deploy/dataplane/dcgm-counters.csv` 中的字段。Kubernetes collector 模板
位于 `deploy/dataplane/gpu-metrics-collector.yaml`，默认访问节点
`http://<host-ip>:9400/metrics`；若 exporter 没有 hostPort，应改为对应 Pod IP、Service
或将 collector 作为 exporter sidecar。

不同 DCGM/GPU 代际可能同时暴露旧字段和 aggregate 字段。当前兼容契约为：

- row-remap 使用 `DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS` 与
  `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS`；
- H200 等环境的 NVLink 聚合计数使用
  `DCGM_FI_DEV_NVLINK_ERROR_DL_CRC/RECOVERY/REPLAY`，同时保留旧 total 字段；
- exporter 返回 `N/A`、空值或目标 GPU 不支持字段时忽略该样本，不能转成数值 0，
  也不能据此产生 finding；
- `DCGM_FI_DEV_XID_ERRORS` 是“最后一次 XID”状态，不是事件计数器。

`NvidiaSmiMetricsCollector` 的实现作为后续验证能力保留。只有显式传入
`--enable-nvidia-smi-metrics-collector` 才允许使用；它采集核心、ECC、retired-page
和 row-remap 四组字段。NVLink 与 PCIe counter 仍应使用 DCGM，不能由
`nvidia-smi` 兜底推断。

控制面接口：

```text
POST /v1/collector-events/gpu-metrics
GET /v1/gpu-metrics/{cluster_id}/{node_id}/latest
GET /v1/gpu-health-findings/{cluster_id}/{node_id}
GET /v1/collector-status/{cluster_id}?node_id={node_id}
GET /v1/evidence/{cluster_id}?node_id={node_id}&attempt_id={attempt_id}
```

温度、ECC DBE、row-remap、PCIe/NVLink 增量和 power/thermal violation 生成结构化
finding，默认 `automatic_action=null`。这类指标不能冒充 NVIDIA XID Immediate Action。
新的非零 `DCGM_FI_DEV_XID_ERRORS` 状态会转换为 `XidEvent` 并进入官方策略；相同 GPU
上的相同 XID gauge 值不会在每次 scrape 时重复触发。由于该字段表示持久化的“最后一次
XID”，collector 启动后的首个值仅建立 baseline；之后从 0 或其他 XID 发生变化时才
生成事件。原始 `/dev/kmsg` 仍是捕获新事件及识别同一 XID 重复发生的主要事件源。

## 时间语义与跨源关联

事件保留以下时间字段：

| 字段 | 含义 |
|---|---|
| `source_event_time` | HMA/CloudWatch 原始 RFC3339 时间，保留时区偏移 |
| `source_monotonic_us` | `/dev/kmsg` 启动后单调微秒数 |
| `source_boot_id` | 单调时间所属 Linux boot ID |
| `collected_at` | collector 读取到事件的 UTC 时间 |
| `ingested_at` | 控制面收到事件的 UTC 时间 |

原始 dmesg/kmsg 单调时间没有时区，不能直接与 CloudWatch UTC 时间比较。同一 kernel
源且 boot ID 相同时优先使用单调时间并保持 30 秒窗口；HMA Node、CloudWatch、kernel
和 DCGM 之间使用 5 分钟窗口，以容忍日志投递和 collector 启动延迟。扩大窗口不会放宽
身份条件：XID/SXID、节点必须一致；两侧都有 GPU UUID 或 PCI BDF 时也必须相交。

finding 查询默认返回 active 状态；使用 `?active_only=false` 查询历史。GPU latest、
counter baseline、active/history finding、batch 幂等结果和 collector status 均通过
控制面的 Store 持久化。多副本必须使用 PostgreSQL/Aurora；每个 GPU/metric key 通过
事务锁串行更新，因此请求落到不同 API 副本时仍能正确计算 counter delta。

GPU critical finding（例如 DBE、row-remap failure 或 critical temperature）会进入
站点安全 policy 并生成 quarantine workflow；warning finding 会生成 diagnostics
workflow。GPU/HBM 温度 warning 的节点步骤执行 `dcgmi diag -r 1 -j` 并保存结构化
证据；通过后等待可配置冷却窗口。诊断失败、结果不确定或窗口结束后温度 finding 仍
active 时，控制面升级为 `DRAIN`，保持节点不可调度和隔离，不自动重启训练。
DCGM JSON 会提取 test、status、entity、error code 和消息，并按 thermal、PCIe、
NVLink/NVSwitch、memory、driver/DCGM、GPU client 和 Field Diagnostic 固定规则生成
`recommended_actions`。结果进入 step execution、升级 incident 和固定模板邮件；
未识别测试使用 `DEEP_DIAGNOSTIC_REVIEW`，不会自动猜测修复动作。

完整 batch 还会进入持久化 composite 状态机。状态机在同一GPU事件时间窗口内联合温度、
thermal throttle/violation、power usage/limit、GPU utilization、DBE、row-remap、
PCIe replay、XID和NVLink信号，并在节点级识别多GPU NVLink故障。命中后保留原始
component finding，但只将 composite 状态边沿提交控制面，避免重复 workflow。
`DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` 的 thermal bits 才能证明热降频；SM/memory clock
仅作为上下文，单独变低不告警。

SBE、retired page 和 correctable row-remap counter 统一按相邻样本正增量处理：
SBE、retired SBE、correctable remap 产生 Warning；retired DBE 产生 Critical。
同一GPU窗口内至少两类 corrected-memory 信号持续增长时生成
`CORRECTABLE_MEMORY_DEGRADATION`，默认连续第3次升级 `DRAIN`。累计旧值首次采集只建立
baseline，不告警。
workflow。它们使用 `SITE_NODE_HEALTH` policy source，不能覆盖或冒充 NVIDIA XID
Immediate Action。

GPU、host 和 log 批次同时更新各 collector 的 `last_success_at`、`last_error_at`、
sample count 和 collection errors。验证步骤要求最近两分钟内存在成功采集；Agent
heartbeat 不能替代 collector freshness。

`VALIDATE_HOST` 检查近期 CPU/load、内存和文件系统证据并拒绝仍超过阈值的节点。
`VALIDATE_FABRIC` 同时要求近期 GPU/NVLink、主机网络证据，并检查 RDMA/EFA link/error
counter。HyperPod 清单默认设置 `GPU_FAULT_VALIDATION_REQUIRE_RDMA=true`；非 RDMA
集群保持 false。该步骤属于被动 telemetry validation，维护窗口中的主动 NCCL/EFA
canary 仍应作为后续深度诊断接入。

## Workload 与训练进度

当前生产方案暂不使用 `TrainingProgressCollector`，部署默认
`GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR=false`。控制面不会启动训练健康扫描，也不会
因训练 Pod 未注入 reporter 而生成 `training_hang` finding。以下采集、上报和判定代码
继续保留，只有完成 reporter/sidecar 接入及端到端验证后才能显式启用。

CompletionWatcher 对 managed Pod 的每次 reconciliation 都向
`POST /v1/workload-observations` 发布 allocation。控制面将
`cluster/attempt/workload/Pod/container/rank/node/GPU UUID` 映射写入共享 Store。
GPU、host 以及显式启用的 log collector 未配置 workload 时，控制面根据节点和事件时间自动
补全 workload state、workload IDs 和 runtime profile。

训练容器或 sidecar 可运行：

```bash
gpu-fault-collector training-progress \
  --attempt-id "${GPU_FAULT_ATTEMPT_ID}" \
  --rank "${RANK}" \
  --progress-file /var/run/gpu-fault/progress.json
```

训练应用通过原子 rename 更新 progress 文件：

```json
{
  "step": 1200,
  "samples_per_second": 845.5,
  "loss": 1.73,
  "numerical_error": false,
  "checkpoint_ref": "s3://bucket/checkpoints/step-1200"
}
```

Reporter 默认每 15 秒向 `POST /v1/training-progress` 发送一次。控制面分别记录最后
heartbeat 和最后 step 前进时间，并检测：

- heartbeat timeout 或 step 长时间不前进；
- rank 相对 peer 的 step lag 或 throughput straggler；
- step regression；
- 显式 `numerical_error=true`，用于 JSON 无法表达的 NaN/Inf。

相关环境变量：

```text
GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR=false
GPU_FAULT_TRAINING_HEARTBEAT_TIMEOUT_SECONDS=120
GPU_FAULT_TRAINING_STARTUP_GRACE_SECONDS=300
GPU_FAULT_TRAINING_MAX_STEP_LAG=20
GPU_FAULT_TRAINING_MIN_THROUGHPUT_RATIO=0.5
GPU_FAULT_TRAINING_HEALTH_SCAN_SECONDS=15
```

## Evidence 留存

GPU、host、node log 和 training progress 原始批次写入共享 Store。默认保留 24 小时，
每节点最多 10000 条；写入时自动删除过期记录和超出上限的最旧记录：

```text
GPU_FAULT_EVIDENCE_RETENTION_HOURS=24
GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE=10000
```

该留存用于故障前后短窗口和 Support case 证据，不替代 CloudWatch Logs、S3 或专用
时序数据库的长期归档。启用时，NodeLogCollector 还把 journal 水位和训练日志 offset 原子保存
到 `/var/lib/gpu-fault/log-collector-state.json`，重启后继续读取。

## FSx、Lustre 与 EFA

Host collector 额外采集：

- NFS/FSx/Lustre mount availability 和容量；
- Lustre read/write bytes、dirty-page hit/miss delta；
- RDMA 每类 port/hardware error counter；
- EFA RNR、retry、CQ error；
- NIC PFC pause 和 ECN counter（驱动通过 `ethtool -S` 暴露时）。

共享文件系统不可用、EFA/RDMA 错误增长会进入 node-health incident 和
`VALIDATE_HOST`/`VALIDATE_FABRIC`。

collector通过`nvidia-smi -q -x`读取并缓存设备slowdown、shutdown、
max-operating和memory max-operating温度限制。读取失败时使用固定站点fallback。
PCIe和NVLink默认值来自NVIDIA DCGM Health，覆盖后finding会标记为站点override：

```text
GPU_FAULT_GPU_TEMP_WARNING_C=85
GPU_FAULT_GPU_TEMP_CRITICAL_C=90
GPU_FAULT_MEMORY_TEMP_WARNING_C=90
GPU_FAULT_MEMORY_TEMP_CRITICAL_C=95
GPU_FAULT_GPU_TEMP_WARNING_MARGIN_C=5
GPU_FAULT_GPU_TEMP_SHUTDOWN_MARGIN_C=3
GPU_FAULT_MEMORY_TEMP_WARNING_MARGIN_C=5
GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE=8
GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL=1
GPU_FAULT_POWER_VIOLATION_DELTA_WARNING_US=1
GPU_FAULT_THERMAL_VIOLATION_DELTA_WARNING_US=1
GPU_FAULT_THERMAL_VIOLATION_DRAIN_CONSECUTIVE_SAMPLES=2
```

## CloudWatch Logs Lambda

当前方案暂不建议部署 `CloudWatchHmaCollector`，专用部署脚本默认
`GPU_FAULT_ENABLE_CLOUDWATCH_HMA_COLLECTOR=false` 并直接跳过。控制面目前只处理
CloudWatch HMA 消息中的 XID/SXID，而节点 Kernel Collector 和 Fabric Manager
Collector 已覆盖这两类信号。现有 Lambda、SQS、Subscription 和 consumer 不会被默认
脚本自动删除；仅在隔离环境验证该链路时显式设置为 `true`。

`deploy/aws/lambda/cloudwatch-hma-template.yaml` 创建 HMA log group subscription 和 Lambda。
部署包必须包含本项目及其运行依赖。handler 为：

```text
gpu_fault.collectors.cloudwatch_lambda_handler
```

默认从 `<node>/SagemakerHealthMonitoringAgent` 格式的 log stream 提取节点。格式不同时
设置 `GPU_FAULT_HMA_NODE_REGEX`，并提供命名捕获组 `(?P<node_id>...)`。无法准确解析节点
时函数失败并让 Lambda 重试，禁止把事件随意关联到集群中的某个节点。

CloudWatch subscription 至少一次投递；控制面通过 `logEvent.id` 保持幂等。HTTP 429、
5xx 和网络错误在函数内有限重试，最终失败继续交给 Lambda 重试和告警。

## HMA Metrics

只有 HMA DaemonSet 声明 metrics 端口且存在匹配 Service 时，discovery 才会报告
`metrics_available=true`。缺少端口、Service 或必需字段时，不能从 HMA Pod Ready
推断 DCGM Prometheus 路径可用。当前默认禁用 HMA Node watcher 和 CloudWatch HMA
collector，由 kernel collector 采集 XID，并由 Fabric Manager collector 采集 SXID。
