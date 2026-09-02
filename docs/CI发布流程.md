# CI 发布流程

本文描述当前 GitHub Actions `CI` 与 `Release` workflow 从源码提交、质量门禁、
受信候选到签名发布制品的完整过程，面向 Release 工程师、维护发布自动化的开发者，
以及负责下载和验签制品的管理员。

本文只描述 CI 构建与发布制品。实际 AWS/Kubernetes 部署继续使用
[管理员快速部署](管理员快速部署.md)、[管理员日常运维](管理员日常运维.md)和
[开发者部署实现](开发者部署实现.md)定义的受检入口。
开发者修改代码后需要一条可直接执行的测试与上线主路径时，使用
[EC2源码统一部署流程](EC2源码Staging复现流程.md)。
单EC2上的dirty源码验证使用
[EC2源码Staging统一部署流程](EC2源码Staging复现流程.md)中的四参数
`gpu-fault-admin deploy`，不属于Release CI。

当前流程的实现事实源是：

- `.github/workflows/ci.yml`：并行质量门禁、main候选、CI gate和候选签名；
- `.github/workflows/release.yml`：候选解析、验签、ECR晋级和最终artifact上传；
- `Makefile` 中的 `release-build-promoted`、`deploy-host-sign`，以及受信本地环境使用的
  `release-build`、`deploy-host-bundle`；
- `scripts/ci_coverage_gate.py`、`scripts/ci_gate_artifacts.py`、
  `scripts/ci_unit_gate.py`、`scripts/ci_gate.py`、`scripts/resolve_ci_run.py`：
  内容寻址coverage shard、聚合unit门禁、main CI候选身份和Release run选择；
- `scripts/build-release-runtime-image.py`、`scripts/build-release-artifacts.py`、
  `scripts/build-release-attestation.py`：运行镜像、Manifest v3 和 attestation；
- `scripts/build-deploy-host-bundle.py`、`scripts/deploy_host_bundle.py`：部署机离线包。

修改上述入口时必须同步本文，不得只修改 workflow 或命令而保留旧流程说明。

## 1. 流程边界

当前主链路分成main CI和Release晋级两段：

```text
main push
  -> static、artifact、5个非PG coverage shard、1个PG shard并行
  -> 每个coverage shard:
       -> 计算域内容身份
       -> 命中历史成功main CI的同身份签名gate：验签并复用证据
       -> 未命中：执行本域pytest、branch coverage和duration采集
       -> 生成当前run gate并独立签名
  -> unit: 验签六个物理shard
       -> coverage combine + 统一78% floor
       -> 合并pytest结果、生成fault report和duration汇总
       -> 生成并签名聚合unit gate
  -> artifact: source-only canonical组件制品 + 未签名deploy-host bundle
  -> test: 验证unit域gate，汇总本次static/artifact，生成并签名commit CI gate
  -> 上传 gpu-fault-ci-candidate

workflow_dispatch 或 v* tag
  -> checkout目标commit
  -> 解析该commit对应的成功main CI run
  -> 下载候选并验证CI gate签名、源码身份和完整文件清单
  -> 通过GitHub OIDC获取AWS临时身份并登录ECR
  -> make release-build-promoted
       -> 再次验证CI gate
       -> 用候选组件制品构建或复用不可变Runtime Image
       -> 生成deployable Manifest v3和绑定CI gate的attestation
       -> 运行最终制品一致性检查并签名attestation
  -> make deploy-host-sign
       -> 只签名CI已经构建的deploy-host archive，不重复构建
  -> 上传 dist/ 为 gpu-fault-release artifact
```

PR也运行相同的六个fresh shard、统一coverage floor、PostgreSQL stress、`static`和
`artifact`，但shard不作main信任签名、不构建deploy-host bundle，也不生成可供Release
跨run下载的签名候选。最终job仍名为`test`，用于保持分支保护的单一聚合状态。

Release workflow只接受checkout得到的clean commit，并要求其候选来自
`refs/heads/main`上的成功push CI。它生成：

- Manifest中的`staging_only=false`；
- attestation中的`release_tier=production`；
- 绑定签名CI gate的晋级门禁记录。

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
`release-build`。直接`release-build`仍在同一进程内执行`make check`和PostgreSQL
stress；这是管理员首次部署或受信本地复现路径，不等同于本文描述的签名CI候选晋级。

## 2. 触发、权限和输入

### 2.1 触发条件

`.github/workflows/ci.yml`只响应Pull Request和`main` push。普通feature branch push
不重复运行CI；同一PR或分支的新提交会通过concurrency取消旧run。

`.github/workflows/release.yml` 支持两种触发方式：

| 方式 | 用途 |
|---|---|
| `workflow_dispatch` | 经审批后晋级所选commit；可选指定成功CI run ID |
| 推送 `v*` tag | 晋级标签指向commit对应的成功main CI候选 |

两个workflow都使用`fetch-depth: 0`。Release默认按`git rev-parse HEAD`查询最近的
成功main push CI；显式`ci_run_id`只省略查询，后续源码commit、Git tree、repository和
artifact清单仍必须与当前checkout完全一致。

### 2.2 GitHub 权限

两个workflow只声明：

| 权限 | 用途 |
|---|---|
| `actions: read` | Release跨run下载受信CI候选 |
| `contents: read` | checkout 源码 |
| `id-token: write` | CI gate、release attestation、deploy-host archive的keyless签名，以及Release的AWS OIDC |

GitHub workflow不配置`COSIGN_SIGNING_KEY`，因此使用keyless `cosign sign-blob`。
Release在请求AWS身份之前先把CI gate的certificate identity固定为
`ci.yml@refs/heads/main`并验证OIDC issuer。部署侧同样必须验证批准的Release identity
和issuer，不能只检查文件存在或SHA-256。

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

CI测试还支持以下可选repository variables：

| 名称 | 默认 | 作用 |
|---|---|---|
| `CI_TEST_RUNNER` | `ubuntu-latest` | coverage和PostgreSQL shard使用的已配置larger/self-hosted Runner label |
| `CI_PYTEST_WORKERS` | `4` | 非PostgreSQL shard的xdist worker数；应与Runner CPU/IO容量匹配 |

未配置larger runner时不得填写不存在的label。worker数变化会进入shard协议身份；CI固定
使用`--dist=worksteal`减少长尾，但不会以增加worker代替慢测试分析。

只有CI的`postgres` shard创建临时PostgreSQL 16 service，并设置只指向该service的
`GPU_FAULT_TEST_POSTGRES_URL`。Release job不再创建PostgreSQL，也不重复执行已经由
签名CI gate证明的测试。该测试连接不是生产数据库连接，不得替换为生产Aurora。

## 3. CI 执行步骤

### 3.1 并行CI门禁

所有job使用Python 3.12和pip cache，并安装hash固定的`requirements/build.lock`后再
安装项目测试extras。CI的执行单元如下：

| job | 主要职责 |
|---|---|
| `static` | Ruff、mypy strict、compileall、架构、部署契约、文档与CI工具测试、配置、YAML、Shell和artifact安全检查 |
| `coverage-runtime_0..2` | 普通Runtime pytest按稳定nodeid哈希分为三份，分别采集branch coverage和duration证据 |
| `coverage-deployment` | 发布、区域编排、deploy-host管理员pytest及对应coverage |
| `coverage-fault_runner` | fault scheduler/runner测试和duration证据 |
| `coverage-postgres` | 串行PostgreSQL contract coverage，再执行8 workers × 40 rounds stress |
| `unit` | 验签六个物理shard、合并coverage、统一floor、fault report、duration汇总和聚合签名 |
| `artifact` | 构建并验证source-only三个组件wheel与Node bundle；main额外构建未签名deploy-host bundle |
| `test` | 聚合static/unit/artifact并为main生成commit级CI gate |

`config/ci-unit-gate.json`定义测试分区、内容身份组、coverage协议和不重复进入coverage的
文档/CI测试。四个逻辑域按以下边界执行，其中runtime生成三个物理shard：

| shard | 测试边界 | 内容身份要点 |
|---|---|---|
| `runtime_0..2` | 除静态、部署、fault runner和PostgreSQL入口外的普通测试；每个具体nodeid按SHA-256取模只进入一份 | dependencies、Runtime源码、Runtime测试、共享fixture、分区总数和index |
| `deployment` | `tests/admin/`、可执行regional测试和release/deploy/artifact根测试 | dependencies、Runtime共享源码、部署/deploy-host-only源码和部署测试 |
| `fault_runner` | `tests/test_case_scheduler.py`；catalog完整契约仍由static执行 | dependencies、Runtime共享源码、testcases、runner/scheduler和runner测试 |
| `postgres` | 4个`tests/store/test_postgres*.py`/contract入口 | dependencies、Runtime共享源码、PostgreSQL测试、实际PostgreSQL image |

PostgreSQL shard的四个直接入口是：

```text
tests/store/test_postgres_store.py
tests/store/test_postgres_processor_claim.py
tests/store/test_postgres_reconnect.py
tests/store/test_store_contracts.py
```

它主要覆盖`src/gpu_fault/store/postgres/**`、`store/contracts.py`、
`store/shared/**`、schema/DDL、连接池与重连，以及processor claim、lease、counter和
fencing。由于这些测试会消费共享model、Store contract和Runtime辅助代码，其身份保守
绑定Runtime源码，而不是只摘要`store/postgres/`；deploy-host-only管理员源码不在该
身份中。

每个shard身份还包含Python版本/ABI、Runner image、实际worker协议和已安装distribution；
PostgreSQL shard额外包含容器image ID。main push先按
`gpu-fault-coverage-<shard>-<identity>`查找历史artifact，只接受已完成且成功的main
push run。命中后：

1. 验证历史shard gate的Cosign workflow identity、producer run和全部证据SHA；
2. 要求producer commit仍是当前commit的祖先；
3. 复用coverage、pytest和duration证据；
4. 生成带`reused_from`的当前run shard gate并重新签名。

未命中的shard只执行自己的pytest。五个非PostgreSQL shard在独立Runner上使用
`--dist=worksteal`；每个pytest命令同时打印`--durations=50`并写入结构化
`durations.json`。因此fresh run的墙钟由最慢shard决定，不再把约3800条普通测试、
PostgreSQL contract和stress串在同一job中。

pytest nodeid是“测试文件路径 + 测试类/函数 + 参数化case ID”，例如：

```text
tests/collectors/test_xid_kmsg_catalog_replay.py::test_xid_kmsg_catalog_replay_b200[GF-XID-KMSG-B200-143]
```

它不是GPU或Kubernetes节点身份。按nodeid而不是按文件拆分，可以把同一文件中的数百个
参数化case分散到三个runtime Runner；分区算法确定、互斥且并集完整。测试函数或参数ID
变化会改变runtime身份，使三个runtime shard同时失效重跑。

runtime、fault runner和PostgreSQL coverage显式排除deploy-host-only模块；
deployment shard采集完整应用源码覆盖。这样只修改独立管理员CLI代码时，旧Runtime
coverage不会携带变化模块的陈旧行号，只有deployment shard失效。最终`unit` job再次
核对六个shard属于当前run、验签并执行`coverage combine`，只在合并数据达到78% floor
后继续。

六份pytest结果在shard身份校验后合并并改写为当前源码身份。
`make fault-test-cases-ci`直接把这些结果映射到64条unit/component fault case，不再次
启动pytest。文档/CI契约测试由当前commit的`static` job执行，因此docs或`.github/`
变化可以复用六个shard，但不能跳过当前static和artifact门禁。

#### 3.1.1 Fresh实测基线

以下数据来自默认`ubuntu-latest`、4个xdist worker，未配置larger runner；用于回归比较，
不是固定SLA：

| main commit / run | 流程 | 端到端 | 最长测试路径 | 聚合 |
|---|---|---:|---:|---:|
| `863a557` / `33615185401` | 旧单体unit | 18分44秒 | unit 18分19秒；其中普通coverage 14分44秒 | 包含在同一unit |
| `a6dd742` / `33620340339` | 六个签名shard | 6分18秒 | `runtime_2` 4分53秒 | unit 57秒 |

新流程fresh总墙钟减少约66%。同一run中其他参考值为：deployment 3分08秒、
PostgreSQL contract+stress 2分20秒、static 2分40秒、artifact 1分23秒。后续比较应同时
检查artifact的`test-durations.json`，避免只看总时长而遗漏Runner排队、依赖安装或单项
测试长尾。

`artifact-check`只生成一套canonical Control Plane、Executor、Node Runtime wheel和
Node bundle。source-only Manifest为`deployable=false`，但包含物理SHA、
`module_digest`、bundle内嵌wheel和component build identity。component identity还覆盖
Python/平台、build lock、组件源码以及全部Node bundle输入，不能用旧bundle搭配新源码。

deploy-host bundle在main CI artifact job中构建一次。`actions/cache`按两个lock、
Runner OS/architecture和Python 3.12保存wheelhouse；archive此时未签名，留给Release
workflow在候选验签成功后签名。

### 3.2 生成受信CI候选

聚合job `test`依赖三个job并逐一要求`success`。main push时它：

1. 下载artifact job生成的完整`dist/`；
2. 下载unit job本次生成的聚合签名gate，再次验签，并核对其内六个shard gate、
   签名bundle、内容身份和证据；
3. 要求工作树干净，读取source-only schema v3 Manifest；
4. 记录Git commit、Git tree、repository、main CI workflow ref、run ID；
5. 对`dist/`中除CI gate自身外的每个文件记录相对路径、权限、大小和SHA-256；
6. 把unit gate的identity、producer run/commit、SHA和复用shard清单写入
   `domains.unit`；
7. keyless签名`dist/ci-gate.json`；
8. 上传`gpu-fault-ci-candidate`，保留30天。

CI gate只能由`ci.yml@refs/heads/main`生成。PR聚合job只给出分支保护结果，不签名或上传
该候选。

### 3.3 Release先验签后访问云端

Release checkout目标commit后，先解析匹配的成功main CI run并下载
`gpu-fault-ci-candidate`。随后依次：

1. 用精确CI workflow certificate identity和GitHub Actions issuer验证CI gate签名；
2. 验证gate的commit、tree和repository与当前checkout一致；
3. 重新计算候选Manifest SHA和完整文件清单；
4. 再次验证聚合unit gate和六个coverage shard的独立Cosign签名；
5. 只有全部一致后才获取AWS OIDC身份、登录ECR和安装发布依赖。

因此手工提供其他run ID、tag指向未通过main CI的commit、候选缺文件或跨run混合制品都
会在AWS/ECR动作之前失败。

### 3.4 执行 `make release-build-promoted`

`release-build-promoted`要求`RUNTIME_IMAGE_REPOSITORY`、`CI_GATE`、clean工作树和
Cosign。它不重复`make check`或PostgreSQL stress，而是再次执行`ci_gate.py verify`，
然后继续以下步骤。

#### 3.4.1 构建或复用 Runtime Image

`scripts/build-release-runtime-image.py`先验证`artifact-check`留下的source-only
Manifest、delivery identity、wheel SHA、`module_digest`以及Node bundle内嵌wheel，
再根据以下输入计算规范化 image input digest：

- Dockerfile 和固定 base image digest；
- runtime dependency lock；
- 目标 platform 和 build args；
- control-plane/executor wheel SHA-256 与 `module_digest`；
- 构建器定义的其他 Runtime Image 输入。

目标不可变tag不存在时，Release构建并推送镜像；tag已存在时，只有目标platform和全部
`gpu-fault.*` labels 与本次输入完全一致才允许复用。任一 label 不一致都会失败，不能把
已有 tag 当作普通 mutable tag 覆盖。

成功后生成：

```text
dist/release-runtime-image.json
```

该 descriptor 记录最终 OCI digest、平台、构建输入和镜像内组件身份。

#### 3.4.2 构建最终发布制品

`scripts/build-release-artifacts.py` 使用 Runtime Image descriptor 和已验证的
canonical制品生成最终release：

1. 复用 Control Plane wheel；
2. 复用 Cluster Executor wheel；
3. 复用 Node Runtime wheel；
4. 复用只包含该精确 Node Runtime wheel 的 Node installer bundle。

构建器重新计算source identity、四个物理SHA、三个`module_digest`和bundle内嵌wheel，
并逐项比较 Runtime Image descriptor。任何不一致都不会生成最终release。组件构建使用
`requirements/build.lock`中固定的build frontend/backend和`--no-isolation`，不会为
每个组件重复创建隔离构建环境。

最终 `release_id` 由四个制品 SHA 和 delivery identity 计算，发布目录采用
`dist/<release-id>/`，并生成 `deployable=true` 的 Manifest v3：

```text
dist/<release-id>/release.json
dist/current-release.json
```

`current-release.json` 是同一 Manifest 的稳定入口，不是另一份可独立修改的事实源。

#### 3.4.3 独立制品一致性测试

Release设置`GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1`，运行
`tests/test_artifact_consistency.py`，验证源码、三个 wheel、Node bundle 和 Runtime
Image 中组件的实际内容一致。该检查失败时不得只保留或发布已推送的 OCI。

#### 3.4.4 生成并签名 release attestation

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
- `dist/ci-gate.json`的SHA-256；
- `ci_gate.py verify`和最终artifact consistency的PASS结论。

main CI中的static、coverage、artifact和PostgreSQL stress结论保存在被绑定的CI gate中，
不伪装成Release job再次执行了`make check`。

随后 `cosign sign-blob` 对 `dist/current-attestation.json` 执行 keyless 签名并生成：

```text
dist/current-attestation.bundle.json
```

部署阶段必须同时校验签名身份、Manifest SHA、release ID 和 delivery identity。

### 3.5 签名deploy-host archive并上传

部署机离线包与 Node installer bundle 是两个独立制品。该步骤不会复用 Node bundle。

CI artifact job中的`scripts/build-deploy-host-bundle.py`：

1. 再次要求干净源码树；
2. 复制 `requirements/build.lock` 和 `requirements/deploy-host.lock`；
3. 从按两个lock、OS、架构和Python ABI缓存的目录补齐完整离线wheelhouse；
4. 在只安装`build.lock`的临时venv中构建独立
   `gpu_fault_deploy_host-*.whl`；
5. 加入部署机工具清单和可选审核后二进制；
6. 记录 Git commit、平台、Python ABI、libc 和每个文件的 SHA-256、大小、权限；
7. 生成确定性 tar.gz 和 `.sha256` sidecar。

CI workflow使用`actions/cache`持久化`DEPLOY_HOST_WHEELHOUSE`。缓存miss时仍按
hash lock下载，cache hit时只校验和补齐缺失wheel。部署机安装始终使用 `--no-index`，
不能再次在线解析依赖。

deploy-host wheel以`gpu_fault.admin_cli`为根，只安装管理员部署CLI及其Python闭包和
`deploy-host-tools.json`。Control Plane Runtime wheel不再包含`gpu_fault.admin_cli`、
`gpu-fault-admin`入口或deploy-host工具清单。管理员代码变化因此只重建deploy-host
bundle，不改变三个应用组件wheel、Runtime Image或应用release diff。

默认 Ubuntu x86-64、CPython 3.12 Runner 生成：

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz.sha256
```

Release验证完整候选后只运行`make deploy-host-sign`，对上述原始archive执行Cosign
keyless签名，不重新构建项目wheel或wheelhouse：

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json
```

文件名由构建 Runner 的 OS、CPU architecture 和 Python cache tag 计算。其他平台必须在
对应受信环境重新构建和签名，不能手工改名。

所有步骤成功后，`actions/upload-artifact`将整个`dist/`上传为：

```text
artifact name: gpu-fault-release
retention: 30 days
```

`if-no-files-found: error` 保证空目录不会形成成功发布。该对象是 GitHub Actions
artifact，不是自动创建的 GitHub Release asset，也不会自动复制到部署机。

## 4. 发布制品清单

| 位置 | 生产者 | 用途 |
|---|---|---|
| `dist/ci-domains/unit/unit-gate.json` | main CI `unit` job | 聚合六个当前run shard、统一coverage/fault/duration证据 |
| `dist/ci-domains/unit/unit-gate.bundle.json` | main CI `cosign sign-blob` | unit域gate的Sigstore签名 |
| `dist/ci-domains/unit/shards/<shard>/coverage-shard-gate.json` | 对应coverage job | 绑定单个shard内容身份、producer和coverage/pytest/duration证据 |
| 同目录`coverage-shard-gate.bundle.json` | 对应coverage job的`cosign sign-blob` | 单个shard的独立Sigstore签名 |
| `dist/ci-gate.json` | main CI `test` job | 绑定当前源码、source-only候选、unit域gate及组合质量门禁 |
| `dist/ci-gate.bundle.json` | main CI `cosign sign-blob` | CI gate的Sigstore签名材料 |
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

1. 六个coverage shard分别绑定本域内容、测试环境和证据，并由固定main CI identity签名；
2. unit域gate验签并聚合六个当前run shard、统一coverage floor、fault和duration证据；
3. 当前commit CI gate验签并绑定unit域gate，同时绑定当前tree、static和artifact；
4. Runtime Image descriptor绑定OCI digest、构建输入和候选中的镜像组件；
5. Manifest v3绑定三个wheel、Node bundle、Runtime Image和delivery identity；
6. attestation绑定Manifest SHA、release ID、delivery identity、源码状态和CI gate SHA；
7. Release Sigstore bundle证明attestation来自批准的GitHub OIDC identity；
8. deploy-host archive使用独立`gpu-fault-deploy-host` distribution、Manifest和签名，
   绑定平台、源码commit、依赖身份和全部payload。

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
checkout一致。依赖安装始终使用`--no-index`。bundle中的
`dependency_identity_sha256`覆盖两个lock和平台兼容信息；相同依赖身份只创建一次共享
依赖venv，后续不同项目wheel只创建轻量overlay venv并通过`.pth`引用共享
site-packages，不重复安装两个lock。依赖层缺失、损坏或身份不一致时fail closed；旧
bundle没有依赖身份时保留原完整安装路径。完整依赖、项目CLI和系统工具检查通过后才
原子替换目标venv；相同bundle重复执行只验证并复用。初始化器不调用系统包管理器，
报告和venv不得包含AWS凭据、token、私钥、数据库密码或kubeconfig。
bundle还携带经过Manifest摘要校验的`config/admin-config.example.yaml`，setup将其安装
到`<deploy-host-venv>/share/gpu-fault/admin-config.example.yaml`。该文件是管理员首次
部署前准备`0600`配置输入的只读模板，不包含凭据，也不会自动写入任何state-dir。

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
| main CI任一并行job失败 | 修复对应静态、测试、PostgreSQL或artifact问题后提交新commit |
| 单个coverage shard未命中 | 只执行该shard；其他同身份shard继续复用 |
| shard签名、身份、证据或历史run不合法 | fail closed；不得复用该shard，调查artifact后重新执行 |
| coverage combine或78% floor失败 | 检查分片遗漏、陈旧数据或覆盖率回退；不得单独接受某个shard |
| 找不到匹配main CI run | 先让目标commit通过main push CI；不得用其他commit候选代替 |
| CI gate签名、源码或文件清单失败 | 停止晋级；不得获取AWS身份或拼接其他run文件 |
| OIDC、AWS 凭据或 ECR 登录失败 | 修复 GitHub environment、角色信任或最小权限后重跑 |
| repository URI 校验失败 | 修正 GitHub variable；不得绕过 ECR URI 检查 |
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
| CI并行job、CI gate或`release-build-promoted` | 本文第3.1至3.4节、开发者部署实现和运维手册 |
| `release-build-staging`或两级attestation边界 | EC2源码Staging流程、安全参考和本文第 1 节 |
| Manifest、wheel、Node bundle、attestation | 本文第 4、5 节及对应制品测试 |
| deploy-host bundle、共享依赖层、平台或 lock | 本文第3.5、6.2节和开发者部署实现 |
| artifact 名称、路径或保留期 | 本文第3.2、3.5、4、6.1节 |
| CI 与实际部署职责边界 | 本文第 1、6.3 节和管理员文档 |

只修改文档仍必须运行：

```bash
make docs-check
```
