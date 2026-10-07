# 普通 EKS GPU 集群（`eks-managed`）与混合 provider 站点

状态：已实现并在一个真实集群上完成一次 join（2026-10-04，`gpu-eks-cluster`，p5en.48xlarge ×2，
AL2023_x86_64_NVIDIA AMI，加入一个已有 `hyperpod-v1` HyperPod 站点）。**验收 campaign 尚未在混合站点上跑过**；
下文"未验证"一节列出凭一次 join 不能证明的内容。本文是分支 `feature/plain-eks-gpu-cluster`
（基线 `0d151512`，36 个提交，184 个文件）的汇总，面向后续 PR 说明。

## 1. 目标与边界

- 让一个**没有 HyperPod 编排器的普通 EKS 集群**（managed node group）成为站点的 GPU 集群：`join-cluster`/
  `deploy`/`remove-cluster`、release 引擎、数据面清单、节点安装器、控制面 fleet pin 全部按 provider 分支。
- 一个站点可以同时纳管 HyperPod 与普通 EKS 集群（混合 provider），每个集群运行自己的 Runtime Profile 版本。
- HyperPod 集群的一切渲染（选择器、Secret 键、注册表摘要、manifest、安装器环境、CPU pin）逐字节不变；
  以测试固定（`tests/*eks_managed*`、`tests/regional/test_regional_token_rotation.py` 的摘要固定值）。
- 不做：provider 侧节点替换（纯 EKS 没有 `nodeReplace`，profile 为 OBSERVE）、warm-spare、HyperPod
  health-monitoring-agent 相关能力、在节点上安装 DCGM 主机工具。

## 2. 设计

### 2.1 共享契约 `gpu_fault.cluster_scope`
`ClusterProvider`（`hyperpod`|`eks-managed`）、`GPU_FAULT_CLUSTER_PROVIDER`/`GPU_FAULT_NODE_SELECTOR`/
`GPU_FAULT_EKS_CLUSTER_NAME`、连接 Secret 键 `cluster-provider`/`node-selector`/`eks-cluster-name`、
`node_selector(target)`（HyperPod：`sagemaker.amazonaws.com/cluster-name=<name>`；EKS：`nvidia.com/gpu.present=true`
或 `--gpu-nodegroup` 派生的 `eks.amazonaws.com/nodegroup=<name>`）、`selector_matches`、
`PROVIDER_NODE_REBOOT_OWNERS`（`gpu-fault-hyperpod-adapter` / `gpu-fault-eks-adapter`）、
`PROVIDER_PROFILE_SOURCES`/`_VERSIONS`（`config/runtime-profile.regional-eks-safe.example.yaml` / `eks-v1`）。

### 2.2 发现与站点
`join-cluster`/`deploy` 按 ARN 判定 provider：EKS ARN 无 HyperPod 编排器且 `describe-cluster` 为 `ACTIVE` →
`eks-managed`；`site.yaml` 集群条目记录 `provider`、`nodeSelector`，不记录 `hyperpodClusterName`；
Agent endpoint CIDR 取 GPU 节点组子网；`NodeRecovery` 门禁的对应项是 GPU 托管节点组的 EKS node auto repair 必须关闭
（发现对每个承载 GPU 节点的节点组读 `describe-nodegroup`，`nodeRepairConfig.enabled=true` 拒绝，身份记 `node_repair`）；
节点名是 FQDN，安装 Job 名 DNS-1123 截断。

### 2.3 按集群的 Runtime Profile（混合站点）
站点 profile 是缺省 pin；站点 profile 的 `nodeReboot` owner 不能服务目标 provider 时，join 给该集群写入
`spec.clusters[].runtimeProfile`（`source`/`templateSource`/`version`/`agentConfigDigest`），新版本经
`release-deploy/profile-plan.json`（`change_kind: CLUSTER_PROFILE_NEW`）审批：`join-cluster --approve-profile-plan
<sha> --reference <CHG>`；审批消费后写入不可变快照 `profiles/<version>.yaml`，同 provider 的后续集群复用。
release 注册每个在用版本，按集群渲染 Executor/Watcher/Reconciler/安装器的版本 pin 与 Agent 配置摘要；
注册表条目携带 `runtime_profile_version`/`agent_config_digest`（等于站点缺省时不进入摘要）；控制面
`FleetRegistry.cluster_policy` 按集群核对 Agent，`GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION` 仍是站点缺省；
首次 `deploy` 的 GPU 集群仍须同一 provider。

### 2.4 Executor / 节点生命周期
`eks-managed` 的 Executor 构造 `gpu-fault-eks-adapter`：`nodeReboot` 对绑定的 EC2 实例 `RebootInstances`
（IAM 仅 `ec2:DescribeInstances`、按 `aws:ResourceTag/eks:cluster-name` 条件的 `ec2:RebootInstances`、限定
ARN 的 `eks:DescribeCluster/DescribeNodegroup`，无 `sagemaker:*`），确认条件为实例 running + 新 Agent
incarnation/boot_id；离场节点经 `ec2 describe-instances` 证明；失败域标签在实例组之后加入
`eks.amazonaws.com/nodegroup`。

### 2.5 数据面与节点
- DCGM exporter：HyperPod AMI 以主机服务运行 DCGM，DaemonSet 连接 `127.0.0.1:5555`；AL2023 NVIDIA AMI 没有
  nv-hostengine/dcgmi，`eks-managed` 渲染去掉 `DCGM_REMOTE_HOSTENGINE_INFO`，容器内嵌 hostengine
  （`NVIDIA_VISIBLE_DEVICES=all`、`NVIDIA_DRIVER_CAPABILITIES=utility`，无 SYS_ADMIN、不申请 `nvidia.com/gpu`），
  hostNetwork/回环 9400/实例类型亲和不变；容忍改为显式的 `nvidia.com/gpu` 与 `gpu-fault.io/quarantined`
  （不容忍 §2.7 的 quiesce 污点；HyperPod 清单的通配容忍逐字节不变）；混合站点 plan 多出 `dcgm_eks_managed`。
- 节点 collector 从 exporter HTTP 取 DCGM 指标，不依赖 dcgmi；节点 agent 的快速 DCGM 诊断在无 dcgmi 节点记为
  `WARN`/`DCGM_TOOL_UNAVAILABLE`。
- 数据面 Deployment（executor、watcher、reconciler、node-resource collector、ADOT）不钉节点、不容忍 GPU 污点，
  落在 CPU 节点。
- 节点预检 Job 在 `chroot /host` 核对主机前置并一次列全缺失项（`ERROR: host prerequisites unavailable: ...`
  + JSON 判定）；AL2023 需 launch template 安装 `python3.12 strace`（管理员日常运维 §5.0.1）。
- Node Action key：所有节点清单读取都用集群选择器（含 CPU 镜像守卫）。

### 2.6 join 的可恢复性与诊断（本分支修的产品缺口）
- `allowedNamespaces` 缺省为站点现有集群共用的列表、不再强加 `training`；join 在首次读取 GPU 集群时核对每个
  工作负载命名空间存在。
- 敏感命令失败保留脚本自己的 `ERROR:` 行（否则最后一行 stderr，打码）；admin 与 release 两个 runner 一致。
- Runtime Profile 注册探针自述失败（HTTP 状态/30 s 无应答）并重试一次。
- 回滚：从未装过节点运行时的集群其节点清理是"空的已证明目标集"（SKIPPED）；停在 PREFLIGHT 的清理日志在站点升级后
  被留档并按当前 release 重新规划；`ROLLBACK_FAILED` 用同样参数重跑即可恢复并接着开始新 attempt。
- 部署机门禁：测试影响选择与角色拆分渲染测试在部署的 deadline/API-budget 作用域内运行而不继承它。

### 2.7 quiesce 污点（仅 eks-managed）
- 普通 EKS 节点的 quiesce 不停 kubelet，Agent 停掉的 GPU 客户端容器（产品自己的 exporter、device plugin、GFD）几秒内
  被 DaemonSet 拉回，reset 前的 `VERIFY_NO_GPU_CLIENTS` 永远看不到 GPU 空闲（2026-10-05 真机）。Executor 在派发
  QUIESCE_GPU_SERVICES 前给节点打 `gpu-fault.io/quiesce=<incident 值>:NoExecute`（`adapters/node_action/quiesce_taint.py`；
  打不上则该步 FAILED、不发 Agent 命令），在派发 RESTORE_GPU_SERVICES 前摘掉（摘不掉则主机服务照常恢复、该步 FAILED 点名）；
  仅当 `GPU_FAULT_CLUSTER_PROVIDER=eks-managed` 时 `cluster_executor/bootstrap.py` 把 Kubernetes core 客户端接给 Node Action
  适配器，HyperPod 不接、行为不变。
- RESTORE_SCHEDULING（validated restore）连同 cordon/隔离污点一起摘掉残留的 quiesce 污点；`workflow-reconcile` 的节点证据、
  孤儿注解清理与 `config spare` 把它视为仍在隔离。
- 污点的生命周期全部有出口：QUIESCE 的每个结果都记 `quiesce_taint_nodes`，FAILED 的 quiesce 当场摘掉；打了污点未完成的 QUIESCE 同样欠
  RESTORE_GPU_SERVICES 补偿；后继的 MARK_UNSCHEDULABLE 摘掉孤儿污点；RESTART_GPU_DEVICE_PLUGIN 在别人仍持有的污点下 WAITING 并点名、
  孤儿污点先摘；VALIDATE_GPU 的 pending 里写 `quiesce_taint_present`/`quiesce_taint_holders`；`--disposition restore` 对只剩本事件
  quiesce 污点的节点只跑 RESTORE_SCHEDULING（`quiesce_taint_only`）。详见管理员日常运维 §5.0 与 §9.2。
- 排队后继的 plan 就地替换不再抬 fencing token（与 RUNNING 前驱共享世代），RUNNING 的 reset 链吸收自己的污点引起的 EFA/GPU
  allocatable 下降，被 executor 在派发前拒绝的步骤两条梯子都不升级；接管前驱隔离的判定走整条升级谱系
  （`escalation_descends_from`，`/executors/incident-ownership?successor_workflow_id=` 的 `escalation_lineage`），
  不再只认一跳。
- reset 忙碌且 `/proc` 无进程持有设备时，Agent 卸载空闲的 `nvidia_drm`/`nvidia_modeset` 后立刻重试并在 reset 后装回
  （`released_kernel_modules`/`reloaded_kernel_modules`）；QUIESCE details 带 `stopped_services`/`inactive_services`/`protected_services`。
- 不容忍该污点的 DaemonSet Pod 在窗口内被驱逐、窗口后回来；带无 key 通配容忍（`operator: Exists`）的第三方 GPU 客户端会留在
  窗口里，reset 仍会 `In use by another client`——见 §3 第 6 条。

## 3. 操作员前置（普通 EKS 集群）

1. 节点组：`AL2023_x86_64_NVIDIA`（或等价）AMI；launch template user data 安装 `python3.12 strace`
   （管理员日常运维 §5.0.1）；节点组的 EKS node auto repair 关闭（`nodeRepairConfig.enabled=false` 或未设置，
   管理员日常运维 §5.0.1）；NVIDIA device plugin 与 EFA device plugin 运行；GPU 污点可有。
2. 命名空间：`allowedNamespaces` 里的工作负载命名空间（缺省站点列表，如 `training`）必须已存在，或用
   `--allowed-namespace` 指定存在的列表。
3. 网络：GPU VPC 的 NAT EIP 加入 NLB 白名单、Route53 私有区关联、Agent CIDR 来自节点组子网——join 自动完成，
   需要相应 AWS 权限。
4. 审批：首个普通 EKS 集群加入 HyperPod 站点需 `--approve-profile-plan <sha> --reference <CHG>`（eks-v1）。
5. 命令：`gpu-fault-admin join-cluster --state-dir <dir> --gpu-cluster-arn <eks arn> --gpu-nodegroup <ng>`。
6. 第三方 GPU 客户端 DaemonSet（客户自己的 dcgm-exporter、GPU feature discovery、监控 agent 等）**不得容忍**
   `gpu-fault.io/quiesce`（不要用无 key 的 `operator: Exists` 通配容忍；按 key 列出 `nvidia.com/gpu` 等即可），并须容忍在
   quiesce 窗口内被停止/驱逐；NVIDIA device plugin 与 GFD 的默认清单只容忍 `nvidia.com/gpu`，无需改动。

## 4. 限制

- `nodeReplace` 仅 OBSERVE；无 HyperPod 健康代理，`gpuDetection`/`providerHealthIsolation` 不声明。
- 节点上无 DCGM 主机工具：`dcgmi diag` 不可用（WARN），诊断包里记录工具缺失。
- 一个集群只能有一个 profile 版本；站点 profile 变更仍是站点级审批，集群 override 版本固定为示例 profile 的版本名。
- 首次 `deploy` 不接受混合 provider（先部署一种，再 join 另一种）。
- 嵌入式 hostengine 依赖 AMI 的默认 NVIDIA 容器运行时；若集群不是默认 `nvidia` runtime，需为 DaemonSet 加
  `runtimeClassName`（本次集群不需要）。
- quiesce 污点只驱逐不容忍它的 Pod：带通配容忍的 DaemonSet、静态 Pod 与非 Kubernetes 进程不受影响，reset 对它们仍可能
  `In use by another client`（步骤 `details.gpu_client_holders` 点名持有者）。

## 5. 验证状态（诚实版）

- 已验证（真机，一次）：发现、profile 审批、IAM/网络/节点 key 前置、按集群 profile 注册与渲染、DCGM 嵌入式
  hostengine 滚动、DNS/TLS 门、数据面 Deployment 落 CPU 节点、节点预检与安装 Job（2 节点）、collector 就绪、
  VERIFIED、注册表更新、站点提交、激活；多次失败回滚与重跑（含站点升级后）。
- 单元/契约：`tests/*eks_managed*`、`tests/fleet/test_cluster_profile_pins.py`、
  `tests/eks_provider/test_mixed_site_owner_resolution.py`、`tests/deploy/test_node_preflight_host_prerequisites.py`
  等；HyperPod 中立性以摘要固定值与逐字节渲染断言保证。
- **未验证**：quiesce 污点（§2.7）与 RECOVERED 事件的 restore 出口只有单元测试，真机 DESTR-001 第三轮待跑；混合站点上的验收 campaign（故障注入、nodeReboot 实际执行与确认、离场节点、token 轮换、
  remove-cluster/uninstall 对 eks-managed 的真机路径、Agent 升级/回滚、长期稳定性）；多于一个普通 EKS 集群；
  非 AL2023 AMI；`training` 以外的工作负载命名空间；非默认 NVIDIA runtime 的集群。

## 6. 提交（基线 0d151512 → 3dedcd07，摘录）

契约与 runtime：`a36f600f` `b84a033b` `d1244a34` `5fcf6fe6` `3b28ab2e` `2cd0135d`；admin/release/部署：
`d3acba29` `0b6e3823` `daab59e1` `fc3ad683`；按集群 profile：`b9dbf3de` `df71aefe` `114ab02e` `0ced160e`
`2beb24e0` `9c1ec98e`；真机缺陷修复：`3017aba4`（节点 key 清单）、`8ca598db`（注册探针/回滚目标）、
`f7dfb5fa`（回滚日志重规划）、`5a06f7b9`（DCGM 嵌入式 hostengine）、`457829f3`（允许命名空间）、
`3dedcd07`（节点预检判定）；部署机门禁：`3ce79c81` `89153747` `29cdddd1` `526c0a73`。
