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

上表的幂等 ID 会作为 HTTP `Idempotency-Key` 头随请求发出，控制面按它去重。sink 只对
**带**这个头的请求重试：默认 4 次尝试并退避，尊重 `Retry-After` 但最多等 30 秒；
一个拿不出幂等 ID 的 payload 只发一次，失败即写 outbox。`snapshot_id`（gpu-inventory
快照）、`log_event_id`（CloudWatch HMA）和 HMA Node 事件（用
`node/<name>/<resourceVersion>`）此前都落在单次发送的分支上，一次 idle-closed 的
keep-alive 连接就把它们推进 outbox；现在三者都有幂等 ID。代价是控制面不可达时
`post()` 阻塞更久（4 次尝试加退避），调用方不能假设它很快返回。

HTTP 2xx 但 body 不是 JSON object 的响应按**结果未知**处理，走与网络失败相同的重试
阶梯，始终无法解析就作为可重放记录进 outbox。重试与重放带同一个 `Idempotency-Key`，
所以确实已被接收的请求会在控制面侧被去重，而不是重复入库。

默认HyperPod生产拓扑启用：

- kernel、fabric-manager、dcgm、host节点采集器；
- kubernetes-node-resources集群采集器；
- completion watcher提供attempt/workload observation。

`NodeLogCollector` 当前默认禁用，不进入生产必需集合。安装器只在隔离验证环境显式
传入 `--enable-node-log-collector` 时启用。

`nvidia-smi`保留为DCGM Exporter不可用时的fallback。Kubernetes HMA和
CloudWatch HMA属于optional验证链路，不在默认拓扑中。

### 新增 Collector

Collector 与 operation、channel、node-action 一样是表驱动的：
`src/gpu_fault/collector_registry.py` 是唯一要改的表，`validate_collector_registry()`
在 import 时校验，缺项在第一次 `import gpu_fault.collectors_cli` 就失败，而不是在
节点上第一次运行时。新增一个 collector 的步骤：

1. **通道**：若数据落在新的控制面入口，先在 `src/gpu_fault/channel_registry.py`
   的 `CHANNEL_REGISTRY` 加一行（processor 路由由 `validate_collector_routes()`
   对账）；复用现有入口则跳过。
2. **工厂**：在 collector 模块里写
   `build_from_environment(sink, context, arguments) -> collector`，把该 collector
   自己的 `os.getenv(...)` 读取放在这里（不需要 context 的 collector 用
   `(sink, arguments)` 两参形式，并在描述符上声明 `needs_context=False`）。
3. **描述符**：在 `COLLECTOR_REGISTRY` 加一个 `CollectorDescriptor`：
   `cli_command`（子命令名）、`kinds`（它产出的 `CollectorKind`，可为空、可与
   `dcgm`/`nvidia-smi` 那样多个子命令共享）、`channel_paths`（它 `post` 的全部
   路径）、`export_name`（`gpu_fault.collectors._EXPORTS` 里的类名）、
   `factory`（`"module:callable"`）、`runs_in`（`node`/`cluster`/`workload`）、
   `needs_product_discovery`。若引入新的 `CollectorKind`，同时在
   `COLLECTOR_KINDS` 加一行：producer 名、systemd unit、
   `silent_threshold("GPU_FAULT_..._SILENT_AFTER_SECONDS", "<秒>")`。
   `telemetry.COLLECTOR_PRODUCER_BY_CHANNEL`、
   `collector_requirements.COLLECTOR_SYSTEMD_UNITS` 和
   `collector_silent_thresholds()` 都由这两张表派生，不再手写。
4. **子命令参数**：需要 argparse 选项时在 `collectors_cli.CLI_ARGUMENTS` 注册一个
   `add_arguments(parser)`；没有则不注册，子命令自动出现。
5. **部署**：`deploy/systemd/` 的 unit 名必须等于 `COLLECTOR_KINDS` 里的
   `systemd_unit`；数据面清单在 `deploy/dataplane/`。

校验器会拒绝：`CollectorKind` 没有描述符产出（或反之）、`channel_paths` 不在
`CHANNEL_REGISTRY`/provider-events 前缀内、`export_name` 未被 `gpu_fault.collectors`
导出、字典键与 `cli_command` 不一致、一个节点子命令跨两个 systemd unit、共享 unit
的 kind 声明不同 producer。第三方 collector 通过 `gpu_fault.collectors` entry-point
组（`PluginGroup.COLLECTORS`）提供一个与 entry point 同名的 `CollectorDescriptor`，
CLI 启动时经 `collector_registry_with_plugins()` 合并并走同一校验；与内建子命令
同名即拒绝。`tests/test_collector_registry.py` 覆盖以上每条规则。

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
ConfigMap。区域模式下控制面自身校验集群 bearer token：请求按 `X-GPU-Fault-Cluster-ID`
查注册表，`Authorization: Bearer` 的 SHA-256 摘要与注册的 token 摘要经
`secrets.compare_digest` 常量时间比对，缺少 bearer 返回 401，集群未注册或不匹配返回
403；执行 token 同样常量时间比对，未声明授权桶的路由默认拒绝（403），见
`src/gpu_fault/app/middleware/auth.py::install_regional_authorization`。API Gateway、
service mesh、NetworkPolicy、安全组或私有负载均衡仍建议作为纵深防御叠加，但不再是
唯一的 token 校验点。`scripts/check-doc-facts.py` 守卫本段与代码一致。

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
  --token-file /etc/gpu-fault/execution-token \
  --wheel-sha256 "${RELEASE_WHEEL_SHA256}" \
  --metrics-mode auto \
  --dcgm-exporter existing \
  --enable-node-agent \
  --node-action-secret "${NODE_ACTION_SECRET}" \
  --node-instance-id "${EC2_INSTANCE_ID}" \
  --node-agent-advertise-url "https://${NODE_PRIVATE_IP}:9099"
```

`--wheel-sha256` 取自受签名发布清单（不是安装包内的 wheel），`--token-file` 指向权限
`0600` 的 token 文件；依赖默认以 `requirements/node-runtime.lock` 的哈希锁通过
`--require-hashes` 安装。三者的完整语义见本节末尾的安全参数说明。

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

journal 是流式读取并受每轮预算约束，不会把整个窗口读进内存：单次 poll 在"批次要保留
的条数"、4 MiB 和 200 ms 三个上限中的第一个处停下，cursor 留在真正读到的最后一条
（历史实现用 `capture_output=True` 一次收完整窗口，高日志量节点会先被 OOM kill，
cursor 未保存，下一次启动重读同一窗口）。批次条数上限（`MAX_ENTRIES_PER_BATCH`，
默认 1000）与 4 MiB 由 journal 与训练日志**共享**：配置了训练日志时先为它保留
`max(1, 上限//4)` 的条目与字节，journal 只用其余额度；没配置时 journal 拿全部额度。被
预算挡下的部分全部计入 discard 计数，不静默丢：

| discard key | 含义 |
| --- | --- |
| `journal-window-capped-seconds` | cursor 比 `MAX_JOURNAL_WINDOW_SECONDS`（默认 900）更旧时放弃的时间跨度（秒），这段 journal 不会再读 |
| `unparseable-journal-entries` | 读到但无法解析的 journal 记录数；cursor 前移时它们已丢失，cursor 保持时下一轮会再读一次 |
| `deferred-training-log-reads` | 本轮因额度耗尽而未读的训练日志文件数；offset 已记录，下一轮从上次停下的文件之后继续 |
| `unreadable-training-logs` | 打开或读取失败（权限、轮转）的训练日志文件数；该文件的 offset 仍被记录，不会被当成新文件重新基线 |

连续第 3 轮训练日志仍只剩推迟（保底份额之外一直拿不到额度）才在批次里额外报一条错误：offset 没丢，但
积压超过一次日志轮转就会真的丢内容。

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

安装器对供应链失败关闭，三个安全参数必须一并提供：

- `--token-file PATH` — 从权限 `0600` 的文件读取 bearer token，避免 token 出现在
  `argv`（`--token` 仍可用但会暴露在进程列表中，仅用于临时调试）。
- `--wheel-sha256 HEX` — 期望的项目 wheel SHA-256，**必须来自受签名的发布清单，而非
  安装包自身**；安装器会把它与 release manifest 中的摘要比对，不一致即失败关闭。
- `--dependency-lock PATH` — 第三方依赖的哈希锁，默认 `requirements/node-runtime.lock`
  （只覆盖节点闭包的窄依赖集，而非控制面的 `runtime.lock`）。安装器始终以
  `pip install --require-hashes --no-deps` 安装，锁内每个分发都带 `--hash=sha256:`，
  因此不会在线解析任何未固定的包。

## GPU Metrics

优先使用 `dcgm`，由 `prometheus_client` 官方 parser 读取 DCGM Exporter。支持：

- GPU/memory 温度、功耗、GPU/memory 利用率、显存和时钟。
- volatile/aggregate SBE/DBE ECC。
- retired pages、pending retirement 和 row remap。
- PCIe replay、NVLink CRC/data/replay/recovery counter。
- power/thermal violation duration。
- `DCGM_FI_DEV_XID_ERRORS`。

DCGM Exporter 必须配置 `deploy/dataplane/dcgm-counters.csv` 中的字段，两条启动路径
（`deploy/dataplane/hyperpod-dcgm-exporter.yaml` 与 `deploy/systemd/gpu-fault-dcgm-exporter.service`）
都以 `-c 15000` 采样并只绑定 `127.0.0.1:9400`；systemd 路径的 `-c` 值来自
`/etc/gpu-fault/dcgm-exporter.env` 的 `GPU_FAULT_DCGM_EXPORTER_COLLECT_INTERVAL_MS`（安装器
参数，默认 15000，须为 ≥ 1000 的整数毫秒），`--dcgm-exporter docker` 时安装器还把换算成秒的
`GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS` 写进 collector.env，采集器据此在启动时校验
「exporter 刷新周期 ≤ 8 × 采集间隔」并在越界时 WARN 一次（`existing` 模式周期未知，不写、
不校验）；GPU metrics collector 由节点 systemd 单元
`gpu-fault-metrics-collector.service` 在本机访问 `http://127.0.0.1:9400/metrics`。历史的
in-cluster DaemonSet 模板 `gpu-metrics-collector.yaml` 已删除，不得重新引入同类 DaemonSet，
否则同一节点会出现双 producer。

只绑定 `127.0.0.1` 的直接后果是：集群外、跨节点或 Prometheus 侧对 `:9400` 的抓取
不再可用，这些计数器只能经 `/v1/collector-events/gpu-metrics` 进入控制面。DaemonSet
不用 nodeSelector，而是用 `nodeAffinity` 按 `node.kubernetes.io/instance-type`
精确列出受支持的 GPU 实例类型（带与不带 `ml.` 前缀各一份），清单由
`regional_release_rendering.render_dcgm_exporter_manifest` 从 `node_installer_reconciler`
的实例类型表渲染；占位符未被替换时渲染报错失败关闭，因此未知实例类型上不会起一个
采不到数的 exporter。

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
