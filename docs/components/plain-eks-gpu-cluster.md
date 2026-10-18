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
- 不做：provider 侧节点替换（adapter 拒绝 `REPLACE`，不替换任何 EC2 实例；`nodeReplace` 只作为 warm-spare 交接由
  `gpu-fault-eks-adapter` OWN，见 §2.4）、HyperPod health-monitoring-agent 相关能力、在节点上安装 DCGM 主机工具。

## 2. 设计

### 2.1 共享契约 `gpu_fault.cluster_scope`
`ClusterProvider`（`hyperpod`|`eks-managed`）、`GPU_FAULT_CLUSTER_PROVIDER`/`GPU_FAULT_NODE_SELECTOR`/
`GPU_FAULT_EKS_CLUSTER_NAME`、连接 Secret 键 `cluster-provider`/`node-selector`/`eks-cluster-name`、
`node_selector(target)`（条目记录的选择器；本约定前的 HyperPod 条目缺省 `sagemaker.amazonaws.com/cluster-name=<name>`）、
归属标签约定 `OWNERSHIP_LABEL=gpu-fault.io/control-plane`、`ownership_node_selector`（EKS：`gpu-fault.io/control-plane=<站点ID>`；
HyperPod 另加集群名标签交叉核对）、`NodeSelectorSource`（`label-convention`/`provider-default-legacy`/`flag-legacy`）与
Secret 键 `node-selector-source`、`selector_matches`、
`PROVIDER_NODE_REBOOT_OWNERS`（`gpu-fault-hyperpod-adapter` / `gpu-fault-eks-adapter`）、
`PROVIDER_PROFILE_SOURCES`/`_VERSIONS`（`config/runtime-profile.regional-eks-safe.example.yaml` / `eks-v1`）。

### 2.2 发现与站点
`join-cluster`/`deploy` 按 ARN 判定 provider：EKS ARN 无 HyperPod 编排器且 `describe-cluster` 为 `ACTIVE` →
`eks-managed`；节点范围由节点组归属标签解析（`admin/bootstrap_ownership.py`：带 `gpu-fault.io/control-plane=<站点ID>` 的
托管节点组 / HyperPod 实例组 `KubernetesConfig.CurrentLabels`；无标签、他站点标签或非 GPU 清单实例类型即拒绝并列出证据；
已纳管集群沿用条目记录的选择器，`gpu-fault-admin node-scope --converge` 是带 journal 的收敛步骤）；`site.yaml` 集群条目记录
`provider`、`nodeSelector`、`nodeSelectorSource`，不记录 `hyperpodClusterName`；
Agent endpoint CIDR 取 GPU 节点组子网；`NodeRecovery` 门禁的对应项是 GPU 托管节点组的 EKS node auto repair 必须关闭
（发现对每个承载 GPU 节点的节点组读 `describe-nodegroup`，`nodeRepairConfig.enabled=true` 拒绝，身份记 `node_repair`）；
节点名是 FQDN，安装 Job 名 DNS-1123 截断。

### 2.3 按集群的 Runtime Profile（混合站点）
站点 profile 是缺省 pin；站点 profile 的 `nodeReboot` owner 不能服务目标 provider 时，join 给该集群写入
`spec.clusters[].runtimeProfile`（`source`/`templateSource`/`version`/`agentConfigDigest`），新版本经
`release-deploy/profile-plan.json`（`change_kind: CLUSTER_PROFILE_NEW`）审批：`join-cluster --approve-profile-plan
<sha> --reference <CHG>`；审批消费后写入不可变快照 `profiles/<version>.yaml`，同 provider 的后续集群复用。
批量 join（多个 `--gpu-cluster-arn`、`deploy --gpu-cluster-arn`）与单集群 join 同样审核；同一版本只出一份计划，列出需要它的全部集群，release 比较已注册版本时不比较注册锚点（`cluster_id`），见 [Runtime Profile 变更审批](../管理员Profile变更审批.md) §1.2。
release 注册每个在用版本，按集群渲染 Executor/Watcher/Reconciler/安装器的版本 pin 与 Agent 配置摘要；
注册表条目携带 `runtime_profile_version`/`agent_config_digest`（等于站点缺省时不进入摘要）；控制面
`FleetRegistry.cluster_policy` 按集群核对 Agent，`GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION` 仍是站点缺省；
首次 `deploy` 的 GPU 集群仍须同一 provider。

### 2.4 Executor / 节点生命周期
`eks-managed` 的 Executor 构造 `gpu-fault-eks-adapter`：`nodeReboot` 对绑定的 EC2 实例 `RebootInstances`
（IAM 仅 `ec2:DescribeInstances`、按 `aws:ResourceTag/eks:cluster-name` 条件的 `ec2:RebootInstances`、限定
ARN 的 `eks:DescribeCluster/DescribeNodegroup`，无 `sagemaker:*`），确认条件为实例 running + 新 Agent
incarnation/boot_id；离场节点经 `ec2 describe-instances` 证明；失败域标签在实例组之后加入
`eks.amazonaws.com/nodegroup`。`nodeReplace` 同样由 `gpu-fault-eks-adapter` OWN
（`config/runtime-profile.regional-eks-safe.example.yaml`，observed `warm-spare-v1`），含义是 Kubernetes 层的 warm-spare
交接：executor 在两种 provider 上都构造 spare coordinator，`REPLACE_NODE`（`HEALTHY_WARM_SPARE_ONLY`）把工作负载交接到
用 `gpu-fault-admin config spare --declare` 声明的热备（拓扑按节点组 + 实例类型）；`REPLACE` 从不在 adapter allowlist
里、`submit` 拒绝它，所以没有声明热备时该步骤 fail closed。仍 pin 在 `nodeReplace` 为 OBSERVE 的旧 `eks-v1` 快照上的
集群保持该声明，直到 `deploy` 规划后继版本 `regional-eks-<digest12>`（`CLUSTER_PROFILE_NEW`）并获批准。

### 2.5 数据面与节点
- DCGM exporter：HyperPod AMI 以主机服务运行 DCGM，DaemonSet 连接 `127.0.0.1:5555`；AL2023 NVIDIA AMI 没有
  nv-hostengine/dcgmi，`eks-managed` 渲染去掉 `DCGM_REMOTE_HOSTENGINE_INFO`，容器内嵌 hostengine
  （`NVIDIA_VISIBLE_DEVICES=all`、`NVIDIA_DRIVER_CAPABILITIES=utility`，无 SYS_ADMIN、不申请 `nvidia.com/gpu`），
  hostNetwork/回环 9400/实例类型亲和不变；容忍改为显式的 `nvidia.com/gpu` 与 `gpu-fault.io/quarantined`
  （不容忍 §2.7 的 quiesce 污点；HyperPod 清单的通配容忍逐字节不变）；以上是 embedded 模式（默认），plan 多出 `dcgm_eks_managed`。
- DCGM 主机模式：GPU 节点组带 `gpu-fault.io/dcgm-hostengine=host`（唯一取值，§3 第 7 条）时集群是 host 模式。deploy/join 每次
  对选择器选中的节点组求值——全部 `host` 为 host（来源 `node-group-label`），都没有为 embedded（来源 `default`），混合或其他取值
  在任何变更之前拒绝并逐组列出 `<节点组> gpu-fault.io/dcgm-hostengine=<值|absent>`；Karpenter NodePool 不读，HyperPod 始终是 host
  （来源 `provider`，不记录）。`site.yaml` 的 eks 条目记 `dcgmHostEngine`/`dcgmHostEngineSource`，每次 deploy/join 按节点组改写
  （节点镜像能力，不像 `nodeSelector` 那样沿用）。host 变体的 exporter 保留 `DCGM_REMOTE_HOSTENGINE_INFO=127.0.0.1:5555`、
  注解 `gpu-fault.io/dcgm-hostengine: host`，显式容忍与其余 eks 渲染不变，plan 对应 `dcgm_eks_managed_host`；embedded 变体逐字节不变。
  模式变化是 release diff 输入 `dcgm_hostengines`：规划 DCGM + reconciler + Agent（`DATA_PLANE_COMPATIBLE`），exporter 重新应用、
  该集群的 Agent 分波重装（reconciler 节点注解 `gpu-fault.io/installer-dcgm-hostengine=host`），变更前的节点预检先核对主机 hostengine。
  `status --full` 的 `node_scope.dcgm_hostengine`/`dcgm_hostengine_source`/`dcgm_hostengine_pending_nodes` 与 `node-scope` 的
  `dcgm_hostengine` 显示模式、来源与待重装的节点。源码不变的 deploy 也在 NOOP 捷径之前用同一套发现读取重读标签，声明与
  记录不同就改走应用发布，混合声明在任何变更之前拒绝；`status`（快速与 `--full`）与 `node-scope` 把尚未发布的声明显示为
  `next deploy: DATA_PLANE_COMPATIBLE (dcgm_hostengine: <集群> declared host, recorded embedded)`。exporter 或 reconciler 实际运行的
  模式（DaemonSet 注解/环境，reconciler 的 `GPU_FAULT_DCGM_HOSTENGINE`）与记录不同为 release diff 输入 `dcgm_hostengine_drift`，同样
  规划 DCGM + reconciler + Agent，`status --full`/verify 的 `gpu_cluster:<id>` 检查对这种不一致判失败（`node_scope.dcgm_hostengine_served`）；
  被拒绝并回滚的发布由回滚对齐把记录恢复为实际运行的模式，`sync-state` 按 exporter 实际运行的模式记录摘要，节点预检的拒绝
  逐节点列出完整原因并记下每个节点的 DCGM 状态。
- 节点 collector 从 exporter HTTP 取 DCGM 指标，不依赖 dcgmi；节点 agent 的快速 DCGM 诊断（`dcgmi diag -r 1 -j`）在 embedded 模式下
  对无 dcgmi 或连不上 hostengine 记配置性 `WARN`（`DCGM_TOOL_UNAVAILABLE`/`DCGM_HOSTENGINE_UNAVAILABLE`，不排空）；host 模式下
  （Agent 环境 `GPU_FAULT_DCGM_HOSTENGINE=host`，只写入 eks-managed 节点、不进 config digest）给出真实判定，缺 dcgmi 或连不上主机
  hostengine 按 HyperPod 判 FAIL/`DCGM_EXECUTION_REVIEW`（`DRAIN_AND_QUARANTINE`）。
- 数据面 Deployment（executor、watcher、reconciler、node-resource collector、ADOT）不钉节点、不容忍 GPU 污点，
  落在 CPU 节点。
- 节点预检 Job 在 `chroot /host` 核对主机前置并一次列全缺失项（`ERROR: host prerequisites unavailable: ...`
  + JSON 判定）；AL2023 需 launch template 安装 `python3.12 strace`（管理员日常运维 §5.0.1）；host 模式另需 `dcgmi`、
  `nv-hostengine`、active 的 `nvidia-dcgm` 与成功的 `timeout 30 dcgmi discovery -l`，同一条判定里列出。预检把每个节点的 DCGM 状态
  （`not-installed`/`installed-inactive`/`host-active`/`cuda-mismatch`/`host-failed`/`unmanaged-hostengine`）写进判定的 `dcgm` 对象：host 模式
  只接受 `host-active`，embedded 模式拒绝与内嵌 hostengine 并存（运行中、active/activating 或 enabled）的主机 hostengine
  （`ERROR: DCGM host engine conflict: ...`）；HyperPod 节点始终按 host 模式核对（`dcgmi`、`nv-hostengine`、active 的 `nvidia-dcgm`、
  `dcgmi discovery -l` 缺一即拒绝，CUDA 不符与外来 nv-hostengine 只记 `WARN`）；最近一次预检的逐节点状态记入 release state `node_dcgm_hostengine`，`status --full` 显示为
  `node_scope.dcgm_hostengine_nodes`/`dcgm_hostengine_conflict_nodes`。Agent 始终按集群声明的模式判定。
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
  （`released_kernel_modules`/`reloaded_kernel_modules`；装回以 `/proc/modules` 核对，缺失时暂停 5 s 重试一次，仍未装回
  则 `display_stack_restored=false`：reset 照样成功、不当 GPU 故障处理，GPU reset 完成通知的主题带
  `[display stack not restored]` 并给出补救 `modprobe nvidia_drm` 或重启节点）；窗口保留的 `nvidia-persistenced`（只持有其余 GPU）与窗口保留的
  Fabric Manager（窗口的 `kept_services` 记着它，只持有 `/dev/nvidia-caps/*`/`/dev/nvidia-nvswitch*`/`/dev/nvidia-nvlink`）
  不阻止卸载，记在 `display_release_bystanders`（只豁免 Fabric Manager 本身：NVL5 上同一单元里的 `nvlsm` 若持有 NVIDIA
  设备节点仍算客户端，quiesce 也从不 signal 它）；QUIESCE details 带 `stopped_services`/`inactive_services`/`protected_services`。
- 不容忍该污点的 DaemonSet Pod 在窗口内被驱逐、窗口后回来；带无 key 通配容忍（`operator: Exists`）的第三方 GPU 客户端会留在
  窗口里，reset 仍会 `In use by another client`——见 §3 第 6 条。

## 3. 操作员前置（普通 EKS 集群）

1. 节点组：`AL2023_x86_64_NVIDIA`（或等价）AMI；launch template user data 安装 `python3.12 strace`
   （管理员日常运维 §5.0.1）；节点组的 EKS node auto repair 关闭（`nodeRepairConfig.enabled=false` 或未设置，
   管理员日常运维 §5.0.1）；节点组的 `updateConfig.updateStrategy` 为 `MINIMAL`（`DEFAULT` 滚动先多启动一台 GPU 实例、
   再终止一台；管理员日常运维 §5.0.1）；NVIDIA device plugin 与 EFA device plugin 运行；GPU 污点可有。
   `AL2023_x86_64_NVIDIA` 节点组可用 `gpu-fault-admin prepare-eks-nodegroup` 自动完成本条的 user data、node auto repair
   与更新策略以及第 5 条的归属标签（默认只读计划，`--apply` 执行；管理员日常运维 §5.0.1a）。
2. 命名空间：`allowedNamespaces` 里的工作负载命名空间（缺省站点列表，如 `training`）必须已存在，或用
   `--allowed-namespace` 指定存在的列表。
3. 网络：GPU VPC 的 NAT EIP 加入 NLB 白名单、Route53 私有区关联、Agent CIDR 来自节点组子网——join 自动完成，
   需要相应 AWS 权限。
4. 审批：首个普通 EKS 集群加入 HyperPod 站点需 `--approve-profile-plan <sha> --reference <CHG>`（eks-v1）。
5. 归属标签：`gpu-fault-admin site-id --cpu-cluster-arn <CPU ARN>` 打印站点 ID，
   `aws eks update-nodegroup-config --cluster-name <eks> --nodegroup-name <ng> --labels addOrUpdateLabels={gpu-fault.io/control-plane=<站点ID>}`
   给 GPU 节点组加标签（HyperPod：`update-cluster` 的 `KubernetesConfig.Labels`；Karpenter：NodePool 模板标签）；
   命令：`gpu-fault-admin join-cluster --state-dir <dir> --gpu-cluster-arn <eks arn>`（只需要 ARN）。
   注销即释放：`remove-cluster`/`uninstall` 的最后一步 `OWNERSHIP_RELEASED` 用
   `aws eks update-nodegroup-config --cluster-name <eks> --nodegroup-name <ng> --labels removeLabels=gpu-fault.io/control-plane`
   从站点选中的节点组摘掉该键（HyperPod：`update-cluster` 带实例组完整定义去键并写明当前镜像，有更新/操作进行中或变更待应用即拒绝；Karpenter
   NodePool 不改写、记 `manual`），有界等待 `describe-nodegroup` 显示已消失；操作者身份另需 `eks:UpdateNodegroupConfig`/`sagemaker:UpdateCluster`
   （运行时角色仍只读）；`--keep-ownership-label` 保留并记 `retained`。已注销集群的节点组仍带标签即过期声明，`status`/`node-scope`
   报 finding（重新接入的同一集群是成员，不报）；重新接入前按 `site-id` 打印的命令重新加标签（管理员日常运维 §5.0.0 "释放"行）。
6. 第三方 GPU 客户端 DaemonSet（客户自己的 dcgm-exporter、GPU feature discovery、监控 agent 等）**不得容忍**
   `gpu-fault.io/quiesce`（不要用无 key 的 `operator: Exists` 通配容忍；按 key 列出 `nvidia.com/gpu` 等即可），并须容忍在
   quiesce 窗口内被停止/驱逐；NVIDIA device plugin 与 GFD 的默认清单只容忍 `nvidia.com/gpu`，无需改动。
7. 主机 DCGM（可选，真实的 `dcgmi diag` 判定）：launch template 固定版本安装 `datacenter-gpu-manager-4-core-<ver>` 与
   `datacenter-gpu-manager-4-cuda<CUDA 主版本>-<ver>`（主版本与 `nvidia-smi` 的 "CUDA Version" 一致），写入 `Restart=on-failure` 的 drop-in
   `nvidia-dcgm.service.d/90-gpu-fault-restart.conf`（软件包单元的 `Restart=on-abort` 让一次非零退出或一轮崩溃循环永久停着）并
   `systemctl enable --now nvidia-dcgm`，
   在节点上用 `dcgmi discovery -l`、`dcgmi diag -r 1` 确认，给该集群的**每个** GPU 节点组加 `gpu-fault.io/dcgm-hostengine=host`，随即
   deploy/join（源码不变也照样 deploy，它会重读标签；之前 `status` 显示待生效的 `next deploy`；启用而不加标签的 deploy 会被
   embedded 冲突检查拒绝）；退出时摘标签、把 `nvidia-dcgm` 从 launch template 拿掉并在节点上
   `systemctl disable --now nvidia-dcgm`，再立即 deploy（次序与命令见管理员日常运维 §5.0.1「主机 DCGM」）。
8. 自动节点更新：托管节点组没有计划更新，但 Karpenter NodePool 的 `expireAfter`（默认 720h）、drift 与 consolidation、
   EKS Auto Mode 节点池、ASG `MaxInstanceLifetime` 和按计划调用 `UpdateNodegroupVersion` 的自动化都会替换 GPU 节点：
   NodePool 设 `expireAfter: Never` 并加 `nodes: "0"` 预算，`MaxInstanceLifetime` 保持未设置，自动化移到维护窗口；Spot 节点组的
   Capacity Rebalance 由 EKS 开启，按替换节点的接入窗口处理。`status --full`/`node-scope` 记录并提示（不拒绝），操作者身份需
   `autoscaling:DescribeAutoScalingGroups`（管理员日常运维 §5.0.5）。

## 4. 限制

- 没有 provider 侧节点替换：`nodeReplace` 只是 warm-spare 交接，须先声明热备（§2.4）；无 HyperPod 健康代理，
  `gpuDetection`/`providerHealthIsolation` 不声明。
- embedded 模式（默认，节点组未声明主机 DCGM）：节点上没有主机 hostengine，快速诊断 `dcgmi diag` 只记配置性 WARN，诊断包里
  记录工具缺失；声明 host 模式（§3 第 7 条）后才有真实判定。
- 一个集群只能有一个 profile 版本；站点 profile 变更仍是站点级审批，集群 override 版本固定为示例 profile 的版本名。
- 首次 `deploy` 不接受混合 provider（先部署一种，再 join 另一种）。
- 嵌入式 hostengine 依赖 AMI 的默认 NVIDIA 容器运行时；若集群不是默认 `nvidia` runtime，需为 DaemonSet 加
  `runtimeClassName`（本次集群不需要）。
- quiesce 污点只驱逐不容忍它的 Pod：带通配容忍的 DaemonSet、静态 Pod 与非 Kubernetes 进程不受影响，reset 对它们仍可能
  `In use by another client`（步骤 `details.gpu_client_holders` 点名持有者）。
- 多集群站点（管理员日常运维 §5.0.4）：节点名必须全站唯一，普通 EKS 节点名由私有 IPv4 派生，VPC CIDR 重叠的两个普通
  EKS 集群可能同名（HyperPod 节点名不会）；HyperPod 集群与普通 EKS 集群不得同名（cluster id 与 IAM 角色名由该名字派生，
  `--cluster-id` 不在下一次 `deploy` 中保留）；集群多时发布只有一个 canary、回滚逐集群串行，且共用固定的 6 小时期限与
  2 小时恢复预留。
- 注册集群（`spec.runtimeProfile.registrationClusterId`）被 `remove-cluster` 移除且站点还有其他成员时，改为剩余的同
  provider 成员；没有同 provider 成员时拒绝移除，基线家族不会随移除改变。

## 5. 验证状态（诚实版）

- 已验证（真机，一次）：发现、profile 审批、IAM/网络/节点 key 前置、按集群 profile 注册与渲染、DCGM 嵌入式
  hostengine 滚动、DNS/TLS 门、数据面 Deployment 落 CPU 节点、节点预检与安装 Job（2 节点）、collector 就绪、
  VERIFIED、注册表更新、站点提交、激活；多次失败回滚与重跑（含站点升级后）。
- 单元/契约：`tests/*eks_managed*`、`tests/fleet/test_cluster_profile_pins.py`、
  `tests/eks_provider/test_mixed_site_owner_resolution.py`、`tests/deploy/test_node_preflight_host_prerequisites.py`
  等；HyperPod 中立性以摘要固定值与逐字节渲染断言保证。
- **未验证**：DCGM host 模式（§2.5）只有单元/契约测试，真机待验；quiesce 污点（§2.7）与 RECOVERED 事件的 restore 出口只有单元测试，真机 DESTR-001 第三轮待跑；混合站点上的验收 campaign（故障注入、nodeReboot 实际执行与确认、离场节点、token 轮换、
  remove-cluster/uninstall 对 eks-managed 的真机路径、Agent 升级/回滚、长期稳定性）；多于一个普通 EKS 集群（一批 join 多个
  普通 EKS 集群只有单元测试）；
  非 AL2023 AMI；`training` 以外的工作负载命名空间；非默认 NVIDIA runtime 的集群。
- p6-b200/p6-b300（HGX B200/B300）按与 Hopper SXM 相同的规则分级（fabric 状态窗口、VALIDATE_GPU 的 fabric 与单元检查、
  solo XID 45 默认只观测），但现网没有这类节点，未观察过。

## 6. 提交（基线 0d151512 → 3dedcd07，摘录）

契约与 runtime：`a36f600f` `b84a033b` `d1244a34` `5fcf6fe6` `3b28ab2e` `2cd0135d`；admin/release/部署：
`d3acba29` `0b6e3823` `daab59e1` `fc3ad683`；按集群 profile：`b9dbf3de` `df71aefe` `114ab02e` `0ced160e`
`2beb24e0` `9c1ec98e`；真机缺陷修复：`3017aba4`（节点 key 清单）、`8ca598db`（注册探针/回滚目标）、
`f7dfb5fa`（回滚日志重规划）、`5a06f7b9`（DCGM 嵌入式 hostengine）、`457829f3`（允许命名空间）、
`3dedcd07`（节点预检判定）；部署机门禁：`3ce79c81` `89153747` `29cdddd1` `526c0a73`。
