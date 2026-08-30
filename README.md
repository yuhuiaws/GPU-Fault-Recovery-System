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
- [管理员快速部署](docs/管理员快速部署.md) / [管理员日常运维](docs/管理员日常运维.md) / [安全与参数参考](docs/安全与参数参考.md)
- [概要设计](docs/概要设计.md) / [概要设计 v2](docs/概要设计-v2.md)
- [详细设计](docs/详细设计.md) / [详细设计 v2](docs/详细设计-v2.md)
- [NVIDIA 策略供应链与实现](docs/components/nvidia-policy.md) / [部署和运维详细参考](docs/部署和运维手册.md) / [逐章解读](docs/部署和运维手册逐章解读.md)
- [开发者发布与测试流程](docs/开发者发布测试流程.md) / [CI 发布流程](docs/CI发布流程.md) / [开发者部署实现](docs/开发者部署实现.md)
- [环境变量参考](docs/管理员环境变量参考.md) / [性能压测验收方案](docs/性能压测验收方案.md) / [扩展指南](docs/扩展指南.md)
- [Collector说明](COLLECTORS.md) / [训练任务示例](examples/README.md) / [部署目录](deploy/README.md) / [脚本与工具](scripts/README.md)

## 构建和验证

要求 Python 3.12。

```bash
scripts/setup-deploy-host.sh --venv .venv --allow-network
. .venv/bin/activate
make check
```

生产部署机必须使用签名离线bundle，见[部署机初始化](docs/部署机初始化.md)。

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

多个GPU集群重复传`--gpu-cluster-arn`。命令自动创建/复用ECR和站点AWS资源、运行release build并push/sign、生成`site.yaml`，随后preflight、deploy和verify。

### 开发者：修改代码或Profile后发布

已有站点的代码候选仍可显式执行build和签名release升级：

```bash
COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python release-build \
  RUNTIME_IMAGE_REPOSITORY=<registry/repository>

make PYTHON=.venv/bin/python release-deploy \
  SITE=/secure/gpu-fault/site.yaml \
  COSIGN_KEY=/secure/release/cosign.pub
```

`release-build`按完整image input digest复用或build/push OCI并签名`dist/current-*`；
`release-deploy`只消费签名制品。CI keyless签名时使用固定certificate identity和issuer。

修改Profile策略时，在上述`release-deploy`命令前追加审批引用：

```bash
PROFILE_APPROVAL=CHG-12345 \
make PYTHON=.venv/bin/python release-deploy ...
```

未提供审批时首次运行只生成`profile-plan.json`并停止。部署机验签后执行
`deploy -> verify -> stability -> release-summary`，保存`verification-report.json`和
`stability-report.json`；`NOOP`记录`SKIPPED_NOOP`并跳过稳定窗口。普通代码发布不需要
`PROFILE_APPROVAL`，且不得手工修改generated Manifest、artifact摘要或Profile版本。
完整实现见[开发者部署实现](docs/开发者部署实现.md)。

### 管理员：首次部署和日常管理

管理员使用`gpu-fault-admin deploy --cpu-cluster-arn ... --gpu-cluster-arn ...`
`--state-dir /secure/gpu-fault --admin-email ...`；`make release-deploy
CPU_CLUSTER_ARN=...`是同一入口的包装。Region由集群ARN推导并要求全部一致。

#### 1. 首次部署

`/secure/gpu-fault/release-signing/`必须预置权限受控的cosign key、password和public
key。首次站点基础资源统一由ARN管理员入口创建和纳管。

#### 2. 已有站点的单命令操作

以下每项都是独立的管理员入口：

```bash
SITE=/secure/gpu-fault/site.yaml

# 预置条件检查：只读，不修改AWS或Kubernetes
gpu-fault-admin preflight -f "${SITE}"

# 首次应用部署或部署升级：自动preflight并按release差异最小滚动
gpu-fault-admin deploy -f "${SITE}"

# 独立验收：只读验证CPU/GPU、Profile、TLS/NLB、Agent和Aurora
gpu-fault-admin verify -f "${SITE}"

# 查看当前健康、release、Profile和各集群状态
gpu-fault-admin status -f "${SITE}"

# 注册一个已有GPU集群
gpu-fault-admin join-cluster -f "${SITE}" --gpu-cluster-arn <gpu-arn>

# 注销一个GPU集群；保留该GPU EKS/HyperPod和其他集群
gpu-fault-admin remove-cluster -f "${SITE}" \
  --cluster-id <cluster-id> --confirm REMOVE_GPU_CLUSTER

# 卸载整个方案控制面和数据面，但保留底层CPU/GPU集群
gpu-fault-admin uninstall -f "${SITE}" \
  --cpu-cluster keep --confirm UNINSTALL_GPU_FAULT
```

若永久退役并连底层CPU EKS/HyperPod集群一起删除，使用更强确认：

```bash
gpu-fault-admin uninstall -f "${SITE}" \
  --cpu-cluster delete \
  --aurora-final-snapshot retain \
  --confirm DELETE_CPU_CONTROL_PLANE
```

GPU EKS/HyperPod始终保留。`deploy`失败或自动回滚后使用相同命令重跑；`join-cluster`和
`remove-cluster`也通过持久状态幂等续跑。已有site的`deploy -f`不会重新构建当前
checkout，只应用site声明的release；代码或Profile变更必须先进入开发者发布入口。
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
