# GPU多节点分布式训练故障自动化处理

本仓库实现面向大规模 GPU 训练集群的故障采集、策略判定、隔离、恢复编排、训练任务恢复和验收工具。

## 当前支持范围

当前唯一生产交付形态是：

```text
区域 CPU EKS 控制面 + Aurora PostgreSQL + NLB
                    |
                    +-- HyperPod EKS GPU 数据集群 A
                    +-- HyperPod EKS GPU 数据集群 B
                    +-- ...
```

不应从仓库中的历史代码或示例推断额外支持范围：

- HyperPod 单集群一体化只用于迁移、Canary 和历史环境维护。
- 通用 Kubernetes 部署当前未实现为生产交付能力。
- HyperPod Slurm 编排当前不支持。
- GPU 节点上的 systemd Agent 属于区域数据面，不是独立部署架构。

## 部署建议、EC2部署机与已验证环境

- **集群创建**：建议从 SageMaker HyperPod UI 分别创建 GPU 数据面和 CPU 控制面集群，再用 `gpu-fault-admin` 部署本方案。
- **职责分离**：推荐链路为“开发者/Codex工作区 -> GitHub Release CI构建和签名 -> 受控EC2部署机验签、部署和运维”。生产部署机不重新build签名release，也不作为日常源码开发机。
- **EC2部署机**：使用专用CPU EC2，放在私有子网且无公网IP；通过SSM Session Manager访问，并启用IMDSv2、加密EBS、系统审计和受控出口。Python、Docker/Buildx、AWS CLI、Cosign、kubectl、Helm及可选Codex CLI应来自固定AMI或审核安装流程。
- **IAM**：不要给EC2长期绑定`AdministratorAccess`。实例profile只承担SSM和受控制品读取；CI使用GitHub OIDC build角色，部署和运维通过STS临时承担独立的最小权限角色。首次bootstrap所需权限较广，也必须限制账号、Region、集群和方案资源，并保留break-glass审计。
- **Codex边界**：Codex CLI是可选的开发、审阅和runbook助手，不是生产信任根。不得向其提示、日志或工作区写入token、私钥、数据库密码或kubeconfig内容；任何AWS/Kubernetes mutation仍需人工确认、维护窗口、停止条件和回滚方案。一台EC2合并开发/build/deploy只允许用于可销毁的隔离staging验证。
- **Codex安装**：按[官方Codex CLI文档](https://developers.openai.com/codex/cli)安装。认证、endpoint和模型映射由组织批准的OpenAI、AWS Bedrock或内部网关配置决定，README不固定provider或模型命令。

  ```bash
  npm install -g @openai/codex
  ```

  不要把特定provider或模型名作为发布门禁；发布事实仍来自Git commit、测试、Manifest和签名。
- **测试基线**：主要在 AWS `ml.p5en.48xlarge` H200 上验证；其他实例类型需重新验证 GPU/EFA、驱动、DCGM 和互联拓扑。组件不使用GPU算力，节点默认每15秒轻量采样、每60秒上报inventory、每5秒检查Fabric Manager日志。

## 方案硬约束

以下约束不是可调默认值：

1. 受管 HyperPod GPU 集群必须设置 `NodeRecovery=None`。
2. 本方案永不调用 `BatchReplaceClusterNodes`；节点替换只使用已纳管的健康的 warm spare 节点。
3. HyperPod Job Auto Restart、EKS auto-resume 必须禁用。
4. 训练任务恢复由本方案的 Kubernetes Adapter 和 restart budget 独占管理。
5. 未知策略、缺少证据、版本不一致或 workload 状态未知时，破坏性动作必须
   fail closed。

## 核心链路

```text
Collectors / Watcher -> Regional ingress and queue -> Policy / Incident / Workflow
                     -> Per-cluster Executor / signed Node Agent
                     -> Validation / scheduling and workload recovery
```

主要能力：

- NVIDIA XID Catalog 和 Fabric Manager SXID 策略。
- GPU UUID、节点、attempt、workload 和 fabric partition 关联。
- 多副本 processor、lane lease、fencing token 和幂等 workflow。
- GPU 服务 quiesce、GPU reset、节点 reboot 和 warm-spare 故障转移。
- Completion Watcher、训练重启预算和 GPU 数量一致性门禁。
- SES 通知、AMP/Alertmanager/SNS 观测和长期证据归档。

## 文档入口

- [文档索引](docs/README.md) / [贡献指南](CONTRIBUTING.md)
- [管理员快速部署](docs/管理员快速部署.md) / [Runtime Profile变更审批](docs/管理员Profile变更审批.md) / [管理员日常运维](docs/管理员日常运维.md) / [安全与参数参考](docs/安全与参数参考.md)
- [概要设计](docs/概要设计.md) / [概要设计 v2](docs/概要设计-v2.md)
- [详细设计](docs/详细设计.md) / [详细设计 v2](docs/详细设计-v2.md)
- [NVIDIA 策略供应链与实现](docs/components/nvidia-policy.md) / [部署和运维详细参考](docs/部署和运维手册.md) / [逐章解读](docs/部署和运维手册逐章解读.md)
- [EC2源码统一部署流程](docs/EC2源码Staging复现流程.md) / [CI 发布流程](docs/CI发布流程.md) / [开发者部署实现](docs/开发者部署实现.md)
- [环境变量参考](docs/管理员环境变量参考.md) / [性能压测验收方案](docs/性能压测验收方案.md) / [扩展指南](docs/扩展指南.md)
- [Collector说明](COLLECTORS.md) / [训练任务示例](examples/README.md) / [部署目录](deploy/README.md) / [脚本与工具](scripts/README.md)

## 构建和验证

要求 Python 3.12。

```bash
make deploy-host-setup-online
. .venv/bin/activate
make PYTHON=.venv/bin/python check
```

`deploy-host-setup-online`是开发checkout的公共初始化入口，内部封装
`scripts/setup-deploy-host.sh --venv .venv --allow-network`。
生产部署机必须使用CI生成并验签的离线bundle，见[CI 发布流程](docs/CI发布流程.md)。
初始化后，Make默认自动使用`.venv/bin/python`；无本地venv的源码包回退到`python3`，
CI和高级调用仍可用`PYTHON=...`显式覆盖。

需要手工逐项执行时，建议按以下顺序：

```bash
make mypy-check
make architecture-check
make docs-check
make artifact-check
make test-parallel
```

`make check` 的最终全量测试同样使用4个xdist worker。隔离PostgreSQL 16测试库不参与
并行；设置`GPU_FAULT_TEST_POSTGRES_URL`后单独运行`make test-postgres`。

`make artifact-check` 构建三个独立 wheel、Node bundle 和内容寻址
`dist/current-release.json`，并验证模块边界、摘要与重复构建确定性。

## 部署：先区分角色

单EC2源码staging、首次建站、失败续跑和后续升级统一使用四参数
`gpu-fault-admin deploy`，具体见文档入口。

所有生产部署开始前必须满足以下共同前提：

1. 一个已有且至少有3个Ready节点的CPU EKS/HyperPod集群。
2. 至少一个已有GPU EKS/HyperPod集群；GPU HyperPod必须为`NodeRecovery=None`。
3. GPU VPC已有NAT出口。
4. 执行身份具有所需AWS、EKS和Kubernetes管理权限。
5. 部署机已安装Python 3.12项目环境以及`aws`、`cosign`、Docker Buildx、`kubectl`、`helm`、`curl`、`jq`、`openssl`、`sha256sum`和`make`；命令只验证这些前置，不自动安装系统工具。
不要直接 apply `deploy/`。首次部署只使用ARN单命令入口：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-eks-or-hyperpod-arn> \
  --gpu-cluster-arn <gpu-eks-or-hyperpod-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

多个GPU集群重复传`--gpu-cluster-arn`。命令自动判断首次或后续部署，内部管理release、
签名、bundle、venv和site，并完成preflight、deploy/upgrade、verify与stability。

### 开发者：修改代码或Profile后发布

源码staging首次部署、dirty迭代、失败续跑和clean commit验证统一使用：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

dirty工作区由内部准备器转成隔离的`staging_only`快照并执行影响测试；clean commit执行
完整生产门禁。用户不提供release-ref、artifact、site、bundle或venv路径。完整流程见
[EC2源码统一部署流程](docs/EC2源码Staging复现流程.md)。

若Runtime Profile策略变化，首次deploy会生成私有计划并停止。审核后执行：

```bash
gpu-fault-admin approve-profile \
  --state-dir /secure/gpu-fault-staging \
  --plan-sha256 "$(jq -er '.plan_sha256' \
    /secure/gpu-fault-staging/release-deploy/profile-plan.json)" \
  --reference CHG-12345
```

再重跑原四参数deploy。审批绑定计划和live baseline，成功后一次性消费；计划漂移必须
重新审批。

### 管理员：首次部署和日常管理

管理员首次和后续部署都重复同一条四参数命令。Region由集群ARN推导并要求全部一致；
已有站点会先验证CPU/GPU身份集合未变化，再验签并执行upgrade。

#### 1. 首次部署

签名材料由内部准备器在私有state中生成或复用，不进入公共参数。首次站点基础资源统一
由ARN管理员入口创建和纳管。

#### 2. 已有站点升级

继续使用与首次部署完全相同的命令，不提供内部生成的site路径：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

其他生命周期操作仍使用受检管理员子命令：

```bash
# 注册一个已有GPU集群
gpu-fault-admin join-cluster --state-dir /secure/gpu-fault \
  --gpu-cluster-arn <gpu-arn>

# 注销一个GPU集群；保留该GPU EKS/HyperPod和其他集群
gpu-fault-admin remove-cluster --state-dir /secure/gpu-fault \
  --cluster-id <cluster-id> --confirm REMOVE_GPU_CLUSTER

# 卸载整个方案控制面和数据面，但保留底层CPU/GPU集群
gpu-fault-admin uninstall --state-dir /secure/gpu-fault \
  --cpu-cluster keep --confirm UNINSTALL_GPU_FAULT
```

若永久退役并连底层CPU EKS/HyperPod集群一起删除，使用更强确认：

```bash
gpu-fault-admin uninstall --state-dir /secure/gpu-fault \
  --cpu-cluster delete \
  --aurora-final-snapshot retain \
  --confirm DELETE_CPU_CONTROL_PLANE
```

GPU EKS/HyperPod始终保留。`deploy`失败或自动回滚后使用相同四参数命令重跑；
`join-cluster`和`remove-cluster`也通过持久状态幂等续跑。
详细升级、排障和退役规则见
[管理员日常运维](docs/管理员日常运维.md)，逐对象审计和break-glass见
[部署和运维详细参考](docs/部署和运维手册.md)。

## 训练任务提交

支持两种受管提交方式：

```bash
gpu-training-submit --site /path/to/site.yaml customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1
```

或先注入托管元数据，再由客户执行 apply：

```bash
gpu-fault-workload-annotate --site /path/to/site.yaml customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1 \
  --training-container trainer \
  -o /tmp/customer-job.managed.yaml

kubectl apply --dry-run=server -f /tmp/customer-job.managed.yaml
kubectl apply -f /tmp/customer-job.managed.yaml
```

客户训练镜像不是本方案控制面镜像。训练镜像只需满足对应框架、NCCL/EFA、
checkpoint 和训练命令要求。

## 安全

- 不得在生产 GPU 上制造真实硬件损伤。
- `/dev/kmsg` 注入只能在批准的隔离测试节点和维护窗口执行。
- execution token、cluster token、Node Action key、数据库密码和私有 CA
  不得写入源码、文档、命令历史或普通 artifact。
- 测试、Canary、probe 和 fault-injection 清单不属于生产部署资产。
- 任何节点动作都必须经过目标节点、boot/incarnation、fencing token、workload
  和维护窗口门禁。

## License

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。
