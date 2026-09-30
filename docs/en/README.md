# Documentation Index

English edition of `docs/README.md`; the Chinese file remains the source of record until both are maintained together.

This directory separates responsibilities into "current source of truth, operating entry points, acceptance evidence, development conventions, historical material".
When the same fact conflicts, the category listed earlier takes precedence; content under `docs/history/` is used only for tracing back
and must not serve as the basis for the current deployment or implementation.

## English editions

Every document of the Chinese documentation set has an English edition in this
directory. The Chinese file remains the source of record until both are
maintained together; when the two disagree, the Chinese file wins. Alert
`runbook_url` annotations and Grafana panel links point at
[Administrator Operations](administrator-operations.md), whose runbook card
headings are the alert names.

| English edition | Chinese original | What it covers |
|---|---|---|
| [Documentation Index](README.md) | [文档索引](../README.md) | This index: entry points by task, authoritative documents, acceptance, generated files, conventions |
| [Regional E2E Acceptance Test Cases](regional-e2e-acceptance-test-cases.md) | [区域模式端到端验收测试用例](../区域模式端到端验收测试用例.md) | The acceptance specification: every regional case, its preconditions, steps and verdict rules |
| [Regional Case Index](regional-case-index.md) | [区域用例索引](../区域用例索引.md) | Generated execution order and entry anchors of the regional cases |
| [Performance Acceptance Plan](performance-acceptance-plan.md) | [性能压测验收方案](../性能压测验收方案.md) | Capacity model, thresholds and evidence format of the performance acceptance |
| [Fault Simulation Test Manual](fault-simulation-test-manual.md) | [故障模拟测试手册](../故障模拟测试手册.md) | How to inject each fault class and what the system must do |
| [Deployment and Operations Manual](deployment-and-operations-manual.md) | [部署和运维手册](../部署和运维手册.md) | Detailed commands, item-by-item audit, infrastructure and break-glass |
| [Deployment and Operations Manual Walkthrough](deployment-and-operations-manual-walkthrough.md) | [部署和运维手册逐章解读](../部署和运维手册逐章解读.md) | How to choose and execute the manual's chapters |
| [Administrator Operations](administrator-operations.md) | [管理员日常运维](../管理员日常运维.md) | Upgrade, cluster change, rotation, troubleshooting by symptom, alert runbook cards and decommission |
| [Administrator Quick Deploy](administrator-quick-deploy.md) | [管理员快速部署](../管理员快速部署.md) | First and subsequent deployments with the four-argument `gpu-fault-admin deploy` |
| [Administrator Capacity Configuration](administrator-capacity-configuration.md) | [管理员容量配置](../管理员容量配置.md) | control-worker, remediation budget and telemetry spool sizing through plan/apply |
| [Administrator Environment Variables Reference](administrator-environment-variables.md) | [管理员环境变量参考](../管理员环境变量参考.md) | Administrator-facing variables by responsibility, regional values and application defaults |
| [Runtime Profile Change Approval](administrator-profile-change-approval.md) | [管理员Profile变更审批](../管理员Profile变更审批.md) | Reviewing plan SHAs, approving, resuming and keeping evidence of Runtime Profile changes |
| [Detailed Design](detailed-design.md) | [详细设计](../详细设计.md) | Current code modules, protocols and state machines |
| [Detailed Design v2 (implementation view)](detailed-design-v2.md) | [详细设计-v2](../详细设计-v2.md) | How each module runs, how data flows and how errors are handled, in coding detail |
| [High-Level Design](high-level-design.md) | [概要设计](../概要设计.md) | What the system is, its production boundary and hard constraints |
| [High-Level Design v2 (closed-loop view)](high-level-design-v2.md) | [概要设计-v2](../概要设计-v2.md) | Who performs each link from detection to re-admission, and where the gaps are |
| [Fault Categories and Actions](fault-categories-and-actions.md) | [故障类别与处置动作总表](../故障类别与处置动作总表.md) | Detection rules, NVIDIA guidance, policy verdict and executed steps per fault class and XID/SXID |
| [Extension Guide](extension-guide.md) | [扩展指南](../扩展指南.md) | Code registration points for new operations, channels, Stores, adapters, handlers and metrics |
| [Developer Deployment Implementation](developer-deployment-implementation.md) | [开发者部署实现](../开发者部署实现.md) | Internal site, production manifests, renderer, release pin, resource registries and administrator interface requirements |
| [Environment Variables Reference](environment-variables-reference.md) | [环境变量参考](../环境变量参考.md) | The complete machine-generated inventory of `GPU_FAULT_*` variables |
| [Security and Parameters Reference](security-and-parameters-reference.md) | [安全与参数参考](../安全与参数参考.md) | Non-relaxable constraints, destructive capability phases and configuration sources |
| [CI Release Process](ci-release-process.md) | [CI发布流程](../CI发布流程.md) | How main CI produces signed candidates and how Release verifies, promotes and delivers artifacts |
| [Change Impact and Test Selection](change-impact-and-test-selection.md) | [变更影响与测试选择](../变更影响与测试选择.md) | Selecting pytest targets and affected regional cases from changed code paths |
| [EC2 Source Staging Reproduction](ec2-source-staging-reproduction.md) | [EC2源码Staging复现流程](../EC2源码Staging复现流程.md) | First, dirty-iteration and subsequent staging deployments from source on one EC2 host |
| [Workflow and Remote Command State Tables](components/postgres-state-tables.md) | [components/postgres-state-tables.md](../components/postgres-state-tables.md) | Two-phase DDL, migration CLI, validation boundary and the preserved original F2 design |
| [NVIDIA Official Policy Implementation Audit](components/nvidia-policy.md) | [components/nvidia-policy.md](../components/nvidia-policy.md) | Pinned upstream sources, decision semantics, implemented official workflows and SXID handling |

## Administrator Entry Points

| Task | Preferred document |
|---|---|
| Initialize or check the deploy host | The deploy-host bundle chapter of [CI Release Process](ci-release-process.md) |
| First deploy on a single EC2 or continue upgrading staging | [EC2 Source Staging Unified Deployment Process](ec2-source-staging-reproduction.md) |
| First regional deploy or rebuilding the application layer | [Administrator Quick Deploy](administrator-quick-deploy.md) |
| Approve a pending Runtime Profile plan for release | [Runtime Profile Change Approval](administrator-profile-change-approval.md) |
| Adjust control-worker, remediation budget or telemetry spool | [Administrator Capacity Configuration](administrator-capacity-configuration.md) |
| Upgrade, expand clusters, roll back, rotate, troubleshoot by symptom or decommission | [Administrator Operations](administrator-operations.md) |
| Approve destructive capabilities and look up configuration boundaries | [Security and Parameters Reference](security-and-parameters-reference.md) |
| Look up environment variable meanings, regional values and application defaults by category | [Administrator Environment Variables Reference](administrator-environment-variables.md) |
| Item-by-item audit, infrastructure preparation and break-glass | [Deployment and Operations Manual](deployment-and-operations-manual.md) |

The normal path does not require administrators to read the full deployment manual linearly, nor to copy the CPU/REG commands section by section.

## Current Authoritative Documents

| Document | Question it answers |
|---|---|
| [High-Level Design](high-level-design.md) | What the system is, what the production boundary and hard constraints are |
| [High-Level Design v2 (closed-loop view)](high-level-design-v2.md) | Who performs each link from detection to re-onboarding, where the capability boundaries are, what the gaps are |
| [Detailed Design](detailed-design.md) | How the current code modules, protocols and state machines are implemented |
| [Detailed Design v2 (implementation view)](detailed-design-v2.md) | How each module runs, how data flows, how exceptions are handled, enough to code and integrate from it |
| [NVIDIA Policy Supply Chain and Implementation](components/nvidia-policy.md) | Catalog pinned sources, generated digests, action semantics and runtime gates |
| [Fault Categories and Actions](fault-categories-and-actions.md) | For every fault category and every XID/SXID: detection rules, NVIDIA official recommendations, policy decisions and actual execution steps |
| [CI Release Process](ci-release-process.md) | How main CI produces signed candidates, how Release verifies signatures, promotes and delivers artifacts |
| [EC2 Source Staging Unified Deployment Process](ec2-source-staging-reproduction.md) | First deploy, dirty iterations and subsequent upgrades all use the four-parameter `gpu-fault-admin deploy` |
| [Administrator Quick Deploy](administrator-quick-deploy.md) | How to complete first and subsequent deploys with the four-parameter `gpu-fault-admin deploy` |
| [Runtime Profile Change Approval](administrator-profile-change-approval.md) | How administrators review the plan SHA, approve, resume and save evidence |
| [Administrator Capacity Configuration](administrator-capacity-configuration.md) | How to produce a config-only release with plan/apply or the first-deploy configuration file |
| [Administrator Operations](administrator-operations.md) | Choosing upgrade, cluster change, rotation, troubleshooting and decommission entry points by task or symptom |
| [Security and Parameters Reference](security-and-parameters-reference.md) | Constraints that cannot be relaxed, capability phases and the configuration source of truth |
| [Administrator Environment Variables Reference](administrator-environment-variables.md) | Categorized meanings of commonly used administrator variables, regional manifest values and application defaults |
| [Deployment and Operations Manual](deployment-and-operations-manual.md) | Detailed commands, item-by-item audit, infrastructure and break-glass |
| [Walkthrough](deployment-and-operations-manual-walkthrough.md) | How to choose and execute manual chapters |

The only current production form is a regional CPU control plane separated from the HyperPod EKS GPU data plane. The historical
single-cluster all-in-one content is used only for migration and compatibility verification and does not constitute a new production deployment entry point.

## Acceptance and Testing

| Document | Type | Maintenance rule |
|---|---|---|
| [Regional E2E Acceptance Test Cases](regional-e2e-acceptance-test-cases.md) | Specification | Revised with the current implementation; case IDs and title anchors stay stable |
| [Regional Case Index](regional-case-index.md) | Generated artifact | Shows only the execution order and specification entry points, without internal execution state |
| [Change Impact and Test Selection](change-impact-and-test-selection.md) | Development gate | Selects pytest and affected regional cases backwards from code paths |
| [Coverage and Scenario Requirement Matrix](../components/scenario-coverage.md) | Quality statistics | Measures statements, branches, mechanism requirements and real-machine verification separately; keeps unimplemented items |
| [Fault Simulation Test Manual](fault-simulation-test-manual.md) | Operating manual | Kept in sync with the fault catalog and the injection inventory |
| [Performance Acceptance Plan](performance-acceptance-plan.md) | Capacity acceptance | Defines only the model, thresholds and evidence format; real results are kept in the private evidence store |

## Generated Artifacts

- [Environment Variables Reference](environment-variables-reference.md): the complete machine inventory;
  [Administrator Environment Variables Reference](administrator-environment-variables.md): the curated categorized view. Both are
  generated by `scripts/generate-env-reference.py` from the same source extraction.
- [Regional Case Index](regional-case-index.md): generated by
  `scripts/build-regional-case-index.py`.
- `GPU_FAILURE_AUTOMATION_DESIGN.html`: generated by `build-html.sh` from the current document set,
  published as a Release, Pages or CI artifact, and not placed under source version control.
- `diagrams/`: local SVG vector sources referenced by the body text. Markdown uses document-relative paths;
  `build-html.sh` looks up assets from the repository root and `docs/` and embeds them into the HTML, so offline viewing does not depend on S3.
  Dated deployment diagrams represent only the corresponding verification snapshot; when updating them, the dates and observation scope in the body text must be updated in sync.

Generated artifacts must not be edited by hand. After modifying the source of truth, run the corresponding generator's `--check`, and run
`scripts/check-doc-references.py` to verify code and test references and
`scripts/check-doc-anchors.py` to verify the path and anchor links between documents.

## Code and Documentation Synchronization

`code-doc-contracts.yaml` defines the code scopes that require documentation impact review and their related documents:

- `src/**/*.py`: runtime, protocols, configuration and module design;
- `tests/**/*.py`: test behavior, acceptance catalog and limitation notes;
- `scripts/**/*.sh`, `deploy/**/*.sh`: commands executed by administrators and deployment processes.

`scripts/check-doc-impact.py` reads the Git diff in a Pull Request. When code matching a contract
changes, at least one corresponding document must change as well. When a purely internal refactor really does not affect the documentation, the
Pull Request body must state:

```text
Documentation-Impact: none
Documentation-Impact-Reason: 仅重构内部实现，公共行为、命令和验收契约均未变化。
```

Locally, `make check` can use the equivalent `GPU_FAULT_DOC_IMPACT=none` and
`GPU_FAULT_DOC_IMPACT_REASON='concrete reason'`. This declaration does not replace the Pull Request
body audit record in CI.

A source package without Git metadata still runs the contract structure and coverage scope validation, but does not fabricate a diff conclusion.

## Development Conventions

- [CI Release Process](ci-release-process.md): GitHub main CI parallel gates, signed candidates, Release promotion,
  artifact delivery and deployment consumption boundaries. The Runtime logical domain is split into three nodeid sub-shards, signed and reused independently of the deployment,
  fault-runner and PostgreSQL shards, with a final unified execution floor; the offline deploy-host installation isolates the workspace
  Python path, force-installs the bundle project wheel, reuses the shared dependency layer by lock/platform, and automatically rebuilds a venv with the same digest
  that is incomplete; the signing password only enters the final cosign process, never tests or builds.
- [EC2 Source Unified Deployment Process](ec2-source-staging-reproduction.md): the shortest verification after a code change,
  staging deployment, non-destructive testing and the production promotion loop.
- [Developer Deployment Implementation](developer-deployment-implementation.md): internal site, production Manifest, renderer,
  release pin, the three-layer resource registry and administrator interface requirements.
- [Extension Guide](extension-guide.md): code registration points when adding an operation, channel, Store, adapter, handler or
  metrics; production delivery continues to go into Developer Deployment Implementation.
- [Workflow/Remote Command Dedicated Tables Implementation](components/postgres-state-tables.md): two-phase
  DDL, migration CLI, validation boundaries, and the preserved original F2 design.
- [Regional Acceptance Runner Case-by-case Review](../components/regional-acceptance-review.md): implementation analysis of the existing 188 cases,
  repeatability judgment and three subsequent additions; historical full-run results and the current coverage remediation are kept separate and are not live pass evidence.
- [Insufficient Hot Spare Independent Cancellation and Safeguards](../components/destr008-cancellation.md): time-limited fixtures, producer
  withdrawal, command silence, resource custody and cleanup-only resume; service window details are in
  [Service Window](../components/destr008-service-window.md).
- [Independent Recovery Across Restarts](../components/destr014-recovery-safeguard.md): DESTR-014's persisted
  recovery, original runtime/inode binding and expiry limits.
- [Node Key Custody Evidence](../components/node-key-custody-evidence.md): the independently signed
  install-time custody chain, administrator registration, successor rotation and real Agent activation verification.
- [Physical Late-Ownership Acceptance](../components/late-ownership-acceptance.md):
  ownership permission after Agent dequeue, independent execution tracking, three controlled scenarios and the non-atomicity limitation.
- [Atomic Ownership Fence Design](../components/atomic-ownership-fence-design.md):
  Explored but not adopted: external writers remain independent, and the stronger
  atomic ownership guarantee remains unimplemented.
- [Validation limitations](../validation-limitations.md): the current validation boundary.

For the contribution entry point see [CONTRIBUTING.md](../../CONTRIBUTING.md) at the repository root.

## History and Evidence

- Historical review inputs, remediation matrices, architecture batches, security incidents, real-environment execution history and unredacted reports
  are kept in a private evidence store outside the repository and are not part of the public document set.
- `evidence/`: condensed test evidence that needs long-term retention.
  The canonical index of fault evidence is
  [evidence/fault/README.md](../evidence/fault/README.md); the non-authoritative historical narrative
  migrated out of the public specification is
  [evidence/regional-history/README.md](../evidence/regional-history/README.md).
- `reference/`: explanatory reference material that cannot be submitted directly to the runtime API.

## Machine Guard

```bash
make docs-check
```
