# CI 发布流程

本文描述当前 GitHub Actions `Release` workflow 从源码提交到签名发布制品的完整过程，
面向 Release 工程师、维护发布自动化的开发者，以及负责下载和验签制品的管理员。

本文只描述 CI 构建与发布制品。实际 AWS/Kubernetes 部署继续使用
[管理员快速部署](管理员快速部署.md)、[管理员日常运维](管理员日常运维.md)和
[开发者部署实现](开发者部署实现.md)定义的受检入口。
开发者修改代码后需要一条可直接执行的测试与上线主路径时，使用
[EC2源码统一部署流程](EC2源码Staging复现流程.md)。
单EC2上的dirty源码验证使用
[EC2源码Staging统一部署流程](EC2源码Staging复现流程.md)中的四参数
`gpu-fault-admin deploy`，不属于Release CI。

当前流程的实现事实源是：

- `.github/workflows/release.yml`：触发条件、GitHub 权限、Runner 步骤和 artifact 上传；
- `Makefile` 中的 `release-build`、`deploy-host-bundle`：发布门禁和构建顺序；
- `scripts/build-release-runtime-image.py`、`scripts/build-release-artifacts.py`、
  `scripts/build-release-attestation.py`：运行镜像、Manifest v3 和 attestation；
- `scripts/build-deploy-host-bundle.py`、`scripts/deploy_host_bundle.py`：部署机离线包。

修改上述入口时必须同步本文，不得只修改 workflow 或命令而保留旧流程说明。

## 1. 流程边界

当前 Release CI 的主链路是：

```text
workflow_dispatch 或 v* tag
  -> checkout 完整 Git 历史
  -> 准备 Python 3.12、PostgreSQL 16、Buildx、Cosign
  -> 通过 GitHub OIDC 获取 AWS 临时身份并登录 ECR
  -> make release-build
       -> make check
       -> make test-postgres-stress
       -> 构建或复用并推送不可变 Runtime Image
       -> 重建三个组件 wheel 和 Node bundle
       -> 生成 Manifest v3、attestation 并签名
  -> make deploy-host-bundle
       -> 构建部署机离线 wheelhouse 和项目 wheel
       -> 生成 SHA-256 并签名
  -> 上传 dist/ 为 gpu-fault-release artifact
```

Release CI只接受checkout得到的clean commit，并且只调用`release-build`。因此它生成：

- Manifest中的`staging_only=false`；
- attestation中的`release_tier=production`；
- 完整`make check`和PostgreSQL stress门禁记录。

`release-build-staging`只供统一CLI的内部源码准备器处理dirty隔离快照。该等级执行影响
选择并在不确定时升级全量门禁，但其Manifest固定为`staging_only=true`，普通生产验签
默认拒绝；GitHub Release workflow不会构建或上传该等级。

该 workflow：

- 会查询、构建和推送 ECR Runtime Image，并可读写独立 BuildKit cache repository；
- 不执行 `make release-deploy`；
- 不创建或修改 EKS、HyperPod、Aurora、NLB、Route53 或 Kubernetes 资源；
- 不持有 CPU/GPU kubeconfig，不接触站点 token、Node Action key 或数据库生产密码；
- 不把 GitHub Actions artifact 自动复制到 `/secure/release/`。

首次 ARN 部署可以由 `gpu-fault-admin deploy` 在受控部署机内部复用
`release-build`。这是管理员首次部署路径，不等同于本文描述的 GitHub Release
workflow。

## 2. 触发、权限和输入

### 2.1 触发条件

`.github/workflows/release.yml` 支持两种触发方式：

| 方式 | 用途 |
|---|---|
| `workflow_dispatch` | 经审批后手工构建指定分支或提交的发布候选 |
| 推送 `v*` tag | 为版本标签构建发布候选 |

workflow 使用 `fetch-depth: 0` checkout 完整历史。构建器需要 Git commit、提交时间和
干净工作树状态来计算源码身份、确定性时间戳和 attestation。

### 2.2 GitHub 权限

workflow 只声明：

| 权限 | 用途 |
|---|---|
| `contents: read` | checkout 源码 |
| `id-token: write` | AWS OIDC 临时凭据和 Cosign keyless 签名 |

CI 未配置 `COSIGN_SIGNING_KEY`，因此当前 GitHub workflow 使用 keyless
`cosign sign-blob`。部署侧必须使用审批过的 certificate identity 和 OIDC issuer
验证，不能只检查文件存在或 SHA-256。

### 2.3 Repository variables

Release workflow 消费以下 GitHub repository 或 environment variables：

| 名称 | 必需 | 作用 |
|---|---|---|
| `RELEASE_ROLE_ARN` | 是 | GitHub OIDC 要 assume 的发布角色 |
| `AWS_REGION` | 是 | ECR 登录和镜像操作 Region |
| `RUNTIME_IMAGE_REPOSITORY` | 是 | 不可变 Runtime Image 的 ECR repository URI |
| `RUNTIME_IMAGE_CACHE_REPOSITORY` | 否 | 独立 BuildKit registry cache repository URI |

两个 repository 值都必须匹配 ECR URI。cache repository 只用于加速 layer 构建，
不能作为可信发布制品，也不能替代 Runtime Image digest 校验。

workflow 内部另外创建临时 PostgreSQL 16 service，并设置只指向该 service 的
`GPU_FAULT_TEST_POSTGRES_URL`。该值不是生产数据库连接，也不得替换为生产 Aurora。

## 3. CI 执行步骤

### 3.1 初始化 Runner

CI 按以下顺序准备构建环境：

1. checkout 完整 Git 历史；
2. 安装 Python 3.12 并启用 pip cache；
3. 初始化 Docker Buildx；
4. 安装 Cosign；
5. 使用 `RELEASE_ROLE_ARN` 和 `AWS_REGION` 获取 AWS 临时凭据；
6. 登录 Amazon ECR；
7. 升级 pip，并安装项目的
   `dev,collectors,postgres,performance` extras。

PostgreSQL service 必须先通过 `pg_isready` 健康检查，后续 job 才会继续。

### 3.2 选择 Runtime Image repository

workflow 验证 `RUNTIME_IMAGE_REPOSITORY` 非空且是 ECR URI，然后写入后续步骤的环境。

提供 `RUNTIME_IMAGE_CACHE_REPOSITORY` 时，workflow 生成：

- `RUNTIME_IMAGE_CACHE_FROM`：读取 `buildcache-linux-amd64`；
- `RUNTIME_IMAGE_CACHE_TO`：以 registry cache 模式更新同一 cache tag。

未提供 cache repository 时不启用远端 cache，不影响发布制品身份。

### 3.3 执行 `make release-build`

`release-build` 首先 fail closed 检查：

- `RUNTIME_IMAGE_REPOSITORY` 已设置；
- `GPU_FAULT_TEST_POSTGRES_URL` 已设置；
- Git 工作树干净；
- Cosign 命令可用。

随后严格按以下顺序执行。

#### 3.3.1 完整质量门禁

```bash
make check
```

该门禁覆盖 Ruff、mypy strict、compileall、架构和部署契约、文档检查、配置与生成物检查、
artifact 安全检查、ShellCheck、三个组件 wheel/Node bundle 一致性以及普通全量 pytest。

`make check` 内部的 `artifact-check` 只证明源码可以确定性构建出一致制品。此时尚未绑定
已推送 Runtime Image，因此该中间候选不能替代后续 deployable Manifest。

#### 3.3.2 PostgreSQL stress

```bash
make test-postgres-stress
```

该步骤单独使用 CI PostgreSQL 16 service，覆盖 schema、并发、lease、counter 和
fencing 等不能由普通 xdist 测试代替的路径。任何 skip 或失败都会终止发布。

#### 3.3.3 构建或复用 Runtime Image

`scripts/build-release-runtime-image.py` 根据以下输入计算规范化 image input digest：

- Dockerfile 和固定 base image digest；
- runtime dependency lock；
- 目标 platform 和 build args；
- control-plane/executor wheel SHA-256 与 `module_digest`；
- 构建器定义的其他 Runtime Image 输入。

目标不可变 tag 不存在时，CI 构建并推送镜像；tag 已存在时，只有目标 platform 和全部
`gpu-fault.*` labels 与本次输入完全一致才允许复用。任一 label 不一致都会失败，不能把
已有 tag 当作普通 mutable tag 覆盖。

成功后生成：

```text
dist/release-runtime-image.json
```

该 descriptor 记录最终 OCI digest、平台、构建输入和镜像内组件身份。

#### 3.3.4 构建最终发布制品

`scripts/build-release-artifacts.py` 使用 Runtime Image descriptor 重新构建：

1. Control Plane wheel；
2. Cluster Executor wheel；
3. Node Runtime wheel；
4. 只包含精确 Node Runtime wheel 的 Node installer bundle。

构建器逐项比较 Runtime Image descriptor 中的组件 wheel SHA 和 `module_digest`。
镜像与本次重建结果不一致时不会生成最终 release。

最终 `release_id` 由四个制品 SHA 和 delivery identity 计算，发布目录采用
`dist/<release-id>/`，并生成 `deployable=true` 的 Manifest v3：

```text
dist/<release-id>/release.json
dist/current-release.json
```

`current-release.json` 是同一 Manifest 的稳定入口，不是另一份可独立修改的事实源。

#### 3.3.5 独立制品一致性测试

CI 设置 `GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1`，再次运行
`tests/test_artifact_consistency.py`，验证源码、三个 wheel、Node bundle 和 Runtime
Image 中组件的实际内容一致。该检查失败时不得只保留或发布已推送的 OCI。

#### 3.3.6 生成并签名 release attestation

`scripts/build-release-attestation.py` 生成：

```text
dist/<release-id>/attestation.json
dist/current-attestation.json
```

attestation 绑定：

- `release_id`；
- `dist/current-release.json` 的 SHA-256；
- delivery identity SHA；
- Git commit 和干净工作树状态；
- `release_tier=production`；
- `make check` 与 `make test-postgres-stress` 的 PASS 结论。

随后 `cosign sign-blob` 对 `dist/current-attestation.json` 执行 keyless 签名并生成：

```text
dist/current-attestation.bundle.json
```

部署阶段必须同时校验签名身份、Manifest SHA、release ID 和 delivery identity。

### 3.4 执行 `make deploy-host-bundle`

部署机离线包与 Node installer bundle 是两个独立制品。该步骤不会复用 Node bundle。

`scripts/build-deploy-host-bundle.py`：

1. 再次要求干净源码树；
2. 复制 `requirements/build.lock` 和 `requirements/deploy-host.lock`；
3. 为两个 lock 准备完整离线 wheelhouse；
4. 在临时 venv 中使用该 wheelhouse 构建项目 wheel；
5. 加入部署机工具清单和可选审核后二进制；
6. 记录 Git commit、平台、Python ABI、libc 和每个文件的 SHA-256、大小、权限；
7. 生成确定性 tar.gz 和 `.sha256` sidecar。

当前 CI 没有设置 `DEPLOY_HOST_WHEELHOUSE`，因此只有这一构建阶段可以联网下载 lock
指定的 wheel。部署机安装始终使用 `--no-index`，不能再次在线解析依赖。

默认 Ubuntu x86-64、CPython 3.12 Runner 生成：

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz.sha256
```

随后 Cosign keyless 签名生成：

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json
```

文件名由构建 Runner 的 OS、CPU architecture 和 Python cache tag 计算。其他平台必须在
对应受信环境重新构建和签名，不能手工改名。

### 3.5 上传 GitHub Actions artifact

所有步骤成功后，`actions/upload-artifact` 将整个 `dist/` 上传为：

```text
artifact name: gpu-fault-release
retention: 30 days
```

`if-no-files-found: error` 保证空目录不会形成成功发布。该对象是 GitHub Actions
artifact，不是自动创建的 GitHub Release asset，也不会自动复制到部署机。

## 4. 发布制品清单

| 位置 | 生产者 | 用途 |
|---|---|---|
| ECR 中的不可变 Runtime Image digest | `build-release-runtime-image.py` | CPU Control Plane 和 GPU Executor 运行镜像 |
| `dist/release-runtime-image.json` | 同上 | 绑定 OCI digest、平台、输入和镜像内组件 |
| `dist/current-release.json` | `build-release-artifacts.py` | 当前 deployable Manifest v3 稳定入口 |
| `dist/<release-id>/release.json` | 同上 | 内容寻址 release Manifest |
| `dist/<release-id>/gpu_fault_control_plane-*.whl` | 同上 | CPU Control Plane |
| `dist/<release-id>/gpu_fault_cluster_executor-*.whl` | 同上 | GPU EKS Executor、Watcher、Collector、Reconciler |
| `dist/<release-id>/gpu_fault_node_runtime-*.whl` | 同上 | GPU 节点 Agent/Collector |
| `dist/<release-id>/gpu-fault-node-installer-*.tar.gz` | 同上 | GPU 节点安装脚本、unit 和精确 Node Runtime wheel |
| `dist/<release-id>/attestation.json` | `build-release-attestation.py` | 内容寻址 release attestation |
| `dist/current-attestation.json` | 同上 | 部署入口消费的 attestation |
| `dist/current-attestation.bundle.json` | `cosign sign-blob` | release attestation 的 Sigstore 签名材料 |
| `dist/gpu-fault-deploy-host-<platform>.tar.gz` | `build-deploy-host-bundle.py` | 部署机离线 Python 环境和工具 payload |
| 同名 `.tar.gz.sha256` | 同上 | 部署机离线包 SHA-256 sidecar |
| `dist/gpu-fault-deploy-host-<platform>.sigstore.json` | `cosign sign-blob` | 部署机离线包的 Sigstore 签名材料 |

不要混淆以下三个名称：

- **Node installer bundle**：安装 GPU 节点 Node Runtime，属于 Manifest v3；
- **deploy-host bundle**：初始化部署机 venv，不安装 GPU 节点；
- **Sigstore bundle**：签名和透明日志验证材料，不包含运行软件。

## 5. 信任链

发布链路按以下关系逐层绑定：

1. Runtime Image descriptor 绑定 OCI digest、构建输入和镜像内组件；
2. Manifest v3 绑定三个 wheel、Node bundle、Runtime Image 和 delivery identity；
3. attestation 绑定 Manifest SHA、release ID、delivery identity、源码状态和质量门禁；
4. Sigstore bundle 证明 attestation 来自批准的 GitHub OIDC identity；
5. deploy-host archive 使用独立 Manifest 和签名，绑定平台、源码 commit 和全部 payload。

仅有版本号、tag、文件名、ConfigMap 名或 Kubernetes annotation 均不足以替代上述绑定。
任何一层缺失、摘要不一致、身份不匹配或制品来自不同 workflow run 时都必须停止。

## 6. 下载、验签与部署消费

### 6.1 下载与受控落盘

Release 管理员应：

1. 选择与审批 commit 或 `v*` tag 对应且整体成功的 workflow run；
2. 下载完整 `gpu-fault-release` artifact，不按文件名拼接其他 run 的制品；
3. 将 release 内容恢复为匹配 checkout 下的 `dist/...` 布局；
4. 按需把 deploy-host archive、`.sha256` 和 `.sigstore.json` 复制到
   `/secure/release/` 等权限受控目录；
5. 保持文件内容和相对 Manifest 路径不变，不重新打包、重命名平台或改写 JSON。

`/secure/release/` 是管理员选择的受控落盘目录，不是 CI 生成或自动发现的路径。
GitHub Actions artifact 的 30 天保留期也不是生产制品长期保存策略；需要长期保留的
已批准 release 应在过期前进入受控制品库。

### 6.2 初始化部署机

当前 CI 使用 keyless 签名，因此正常消费方式是：

```bash
make deploy-host-setup \
  DEPLOY_HOST_ARCHIVE=/secure/release/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz \
  DEPLOY_HOST_SIGNATURE_BUNDLE=/secure/release/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json \
  CERTIFICATE_IDENTITY=<approved-release-identity> \
  CERTIFICATE_OIDC_ISSUER=<approved-release-issuer> \
  DEPLOY_HOST_VENV=/secure/gpu-fault/deployer-venv
```

setup 会在安装前验证签名、archive 内容、平台兼容性、源码 commit 和当前 checkout。
bundle Manifest绑定OS、CPU架构、Python实现/3.12 ABI cache tag、sysconfig platform和
libc实现；任一不匹配都fail closed。生产bundle还要求clean源码且Git commit与当前
checkout一致。依赖安装始终使用`--no-index`，在同目录临时venv中完成全部依赖和系统
工具检查后才原子替换目标venv；相同bundle重复执行只验证并复用。初始化器不调用系统
包管理器，报告和venv不得包含AWS凭据、token、私钥、数据库密码或kubeconfig。

### 6.3 消费签名 release

受控制品同步自动化把同一CI artifact恢复到受信checkout后，管理员仍执行统一四参数命令：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

CLI内部解析并验签attestation、Manifest和OCI digest；artifact、certificate identity、
site和低层`release-deploy`参数不进入普通管理员命令。实际Kubernetes/AWS mutation继续
受Profile、维护窗口、回滚和fail-closed门禁约束。

## 7. 失败与重跑

| 失败阶段 | 处理原则 |
|---|---|
| OIDC、AWS 凭据或 ECR 登录失败 | 修复 GitHub environment、角色信任或最小权限后重跑 |
| repository URI 校验失败 | 修正 GitHub variable；不得绕过 ECR URI 检查 |
| `make check` 失败 | 修复源码、测试、文档或生成物后提交新 commit |
| PostgreSQL stress 失败 | 修复并发/schema问题或 CI service；不得以普通 pytest 代替 |
| Runtime Image tag/label不一致 | 调查输入或 registry 漂移；不得覆盖不匹配的不可变 tag |
| wheel、Node bundle 或镜像一致性失败 | 停止发布并调查工具链、lock 或源码闭包漂移 |
| attestation/Cosign失败 | 检查 OIDC identity、`id-token` 权限和签名服务后重跑 |
| deploy-host bundle失败 | 检查 lock wheel、Python 3.12、平台和干净工作树 |
| artifact上传失败 | 整个 run 视为失败，不得只从 Runner 临时目录取文件部署 |

如果 OCI 已推送但后续步骤失败，该 OCI 不能单独视为可发布 release。必须得到完整、
签名且上传成功的 Manifest、attestation 和相关制品后才能交给部署阶段。

同一干净 commit 重跑时可以复用完全匹配的不可变 OCI；仍应把新 run 视为一套独立发布
结果，不能把两个 run 的签名或文件混合。

## 8. 本地受信环境复现

本地复现用于 Release 工程和故障定位，不替代正式 CI 审批。前置条件包括 Python 3.12、
隔离 PostgreSQL 16、ECR 登录、Docker Buildx、Cosign，以及已经安全设置但不打印的
`GPU_FAULT_TEST_POSTGRES_URL`。

```bash
test -n "${GPU_FAULT_TEST_POSTGRES_URL}"

COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python release-build \
  RUNTIME_IMAGE_REPOSITORY=<ecr-repository>

COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python deploy-host-bundle
```

本地示例使用受控私钥；GitHub workflow 使用 keyless OIDC。两种签名模式的部署验签参数
不同，不能把 keyless bundle 当作固定 public key 签名处理。

## 9. 修改与验证

修改发布流程时至少同步：

| 变化 | 必须审阅 |
|---|---|
| workflow trigger、权限、Runner 或 GitHub variables | 本文第 2、3 节 |
| `release-build` 顺序或质量门禁 | 本文第 3.3 节、开发者部署实现和运维手册 |
| `release-build-staging`或两级attestation边界 | EC2源码Staging流程、安全参考和本文第 1 节 |
| Manifest、wheel、Node bundle、attestation | 本文第 4、5 节及对应制品测试 |
| deploy-host bundle、平台或 lock | 本文第 3.4、6.2 节和开发者部署实现 |
| artifact 名称、路径或保留期 | 本文第 3.5、4、6.1 节 |
| CI 与实际部署职责边界 | 本文第 1、6.3 节和管理员文档 |

只修改文档仍必须运行：

```bash
make docs-check
```
