# Examples

English edition of [`README.md`](README.md); the Chinese file remains the source of record until both are maintained together.

This directory only keeps sample customer training-job inputs; it is not a production deployment entry point. Production components are
deployed from `deploy/`; the only current production form is described in
[`docs/en/deployment-and-operations-manual.md`](../docs/en/deployment-and-operations-manual.md).

## HyperPod Training Job

`hyperpod/three-node-pytorchjob.yaml` is a SOURCE MANIFEST: it is runnable, but does not yet carry the
`gpu-fault.io/*` managed metadata. It must be submitted through `gpu-training-submit`, or annotated first with
`gpu-fault-workload-annotate` and then `kubectl apply`. Applying it directly starts a training job that is
**not managed by the GPU fault control plane**.

Both commands should read the current Runtime Profile through `--site` instead of relying on the default version:

```bash
gpu-training-submit --site /path/to/site.yaml \
  hyperpod/three-node-pytorchjob.yaml
```

| File | Type | Purpose |
|---|---|---|
| `hyperpod/three-node-pytorchjob.yaml` | SOURCE | Three-node training submission example of a customer YAML; `GF-REGIONAL-WORKLOAD-001/002` |

These files use a digest-pinned AWS PyTorch Training DLC as the repository acceptance baseline. Customers may replace it with
their own training image, but the image must provide the PyTorch, NCCL and EFA runtimes and the shell tools used by the commands
in the manifest.

The manifest is written for p5en-class nodes: every training Pod requests
`nvidia.com/gpu: 8` and `vpc.amazonaws.com/efa: 16`. Other instance types must first adjust the
GPU/EFA counts, CPU, memory and `--nproc-per-node`, otherwise the Pods may stay Pending indefinitely.

## Related E2E Fixtures

Fault-injection, hung, warm-spare, annotated-job and q118 scenarios are all test assets located under
`scripts/e2e/regional/manifests/training/` or `scripts/e2e/regional/manifests/`. They may only be used from the corresponding
test steps and are not customer examples.
