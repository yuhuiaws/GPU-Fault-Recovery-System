# Contributing

修改前按变化类型选择文档：

| 变化 | 必读文档 |
|---|---|
| 新增 operation、channel、Store、路由、插件或指标 | [GPU Fault 扩展指南](docs/扩展指南.md) |
| 修改 Manifest、renderer、systemd、AWS 资源、配置模型或管理员 CLI | [开发者部署实现](docs/开发者部署实现.md) |
| 修改 Release CI、签名、`dist/`制品或部署机离线bundle | [CI 发布流程](docs/CI发布流程.md)、[开发者部署实现](docs/开发者部署实现.md)和[部署机初始化](docs/部署机初始化.md) |
| 完成代码修改并部署到staging或生产验证 | [开发者发布与测试流程](docs/开发者发布测试流程.md) |
| 同时新增代码能力和生产资源 | 两份都读，两套门禁都执行 |
| 只部署已有 release | [管理员快速部署](docs/管理员快速部署.md) |

operation/channel registry、显式授权、生产资源生命周期和管理员入口都不能只在调用侧
或线上增加字面量。扩展指南决定能力登记点，开发者部署实现决定生成、发布、验证、
回滚和卸载方式。

文档职责和权威顺序见 [docs/README.md](docs/README.md)。历史材料不得作为当前
实现依据。

`src/**/*.py`、`tests/**/*.py`、`scripts/**/*.sh` 和生产部署脚本受
[代码与文档影响契约](docs/code-doc-contracts.yaml)约束。修改这些文件时：

- 同时修改契约列出的相关文档；或
- 在 Pull Request 正文中填写 `Documentation-Impact: none`，并在
  `Documentation-Impact-Reason` 中说明为什么公共行为、命令和验收契约均未变化。

不能用空值、`N/A` 或 `TODO` 代替原因。CI 会基于目标分支和当前提交的 Git diff
执行 `scripts/check-doc-impact.py`。

本地确认属于纯内部改动时，可以显式运行：

```bash
GPU_FAULT_DOC_IMPACT=none \
GPU_FAULT_DOC_IMPACT_REASON='仅重构内部实现，公共行为和命令未变化' \
make check
```

日常开发先按相对目标分支的实际差异运行反向影响选择：

```bash
make test-impact BASE=origin/main
make regional-impact-plan BASE=origin/main
```

`test-impact`执行相关静态检查和pytest；`regional-impact-plan`只输出受影响的区域用例，
不会自动执行live或destructive case。无法匹配的文件、共享契约、Python/依赖/runtime
image、schema/事务变化或跨越三个以上影响域时会fail closed并升级为完整门禁。
版本候选和正式发布仍必须运行`make check`；完整区域验收只跟随首次上线、重大架构变化
或影响计划明确要求执行。规则与说明见
[变更影响与测试选择](docs/变更影响与测试选择.md)。

管理员首次部署由一个ARN命令完成基础资源、release build和应用部署：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

开发者或Release CI仍可独立构建签名候选：

```bash
COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make release-build \
  RUNTIME_IMAGE_REPOSITORY=<registry/repository>
```

GitHub Actions从触发、OIDC/ECR、质量门禁、制品签名到`gpu-fault-release` artifact
交付的完整顺序见[CI 发布流程](docs/CI发布流程.md)。
开发者修改代码后的最短测试、staging部署和同制品生产晋级顺序见
[开发者发布与测试流程](docs/开发者发布测试流程.md)。

最后默认消费build生成的`dist/current-attestation.json`和
`dist/current-attestation.bundle.json`，验签后部署：

```bash
make release-deploy \
  SITE=/path/to/site.yaml \
  COSIGN_KEY=/path/to/cosign.pub
```

ARN首次部署内部复用同一`release-build`和持久化部署状态机；已有站点仍可显式执行
`release-build -> release-deploy`。

`make check`的最终全量测试和`make test-parallel`使用4个worker且不连接外部
PostgreSQL。`make coverage`要求设置`GPU_FAULT_TEST_POSTGRES_URL`：先由4个worker
采集非PostgreSQL覆盖率，再串行追加隔离PostgreSQL 16测试库覆盖率，最后统一强制当前
78%的floor。覆盖率可以提高，不能通过调低`COVERAGE_FLOOR`掩盖未测试的新分支。

只修改文档时，仍必须运行：

```bash
make docs-check
```
