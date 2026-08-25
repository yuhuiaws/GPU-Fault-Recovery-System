# Local artifacts workspace

`artifacts/` is an ignored local workspace. Raw performance logs and release
snapshots must be archived to an access-controlled object store when they
need long-term retention. Only this README belongs in version control.

## Layout

```text
artifacts/
  perf/<case>/<release-id>/<utc>/
    run.json
    status.json
    summary.json
    cgroup-*.json
    postgres-*.json
    aurora.json
    queue-drain.json
    pods/
    executors/
    inflight.jsonl
  perf/_aborted/
  fault/
    fault-tests-<utc>.json
    <case>-<utc>.json
    legacy/
  releases/<release-id>-<utc>/
    inventory.json
    secret-inventory.txt
    control/
    gpu/
  audit/
    code-size-<utc>.json
    code-size-<utc>.md
  upstream/nvidia/xid-catalog-610/<source-sha256>/
    Xid-Catalog.xlsx
```

## Prohibited content

1. Kubernetes `Secret` manifests or `kubectl get secret -o yaml` output.
2. Wheel, installer bundle or other binary payloads encoded in
   `binaryData`.
3. Individual files larger than 5 MiB.

Release evidence must be captured with
`scripts/capture_release_evidence.py`. It stores Secret names and SHA-256
digests only, excludes historical wheel ConfigMaps, and replaces any selected
binary ConfigMap payload with a digest and decoded size.

## Retention

- Completed performance runs: keep the latest five runs per
  `(case, release-id)` plus runs explicitly referenced by documentation.
- Fault runner output: keep the latest five runs per case. Promote only
  redacted, reviewed evidence into `docs/evidence/fault/manifest.yaml`;
  full XID replay dumps remain local/CI artifacts.
- Aborted runs: keep seven days unless linked to an incident.
- Release snapshots: keep until the release leaves the rollback window, then
  archive the compact inventory and delete raw Kubernetes objects.
- Long-term conclusions belong in `docs/evidence/` as compact JSON or
  Markdown summaries, never raw Pod logs or Kubernetes Secrets.

`make artifacts-retention` reports what the first rule above would delete and
never deletes anything; add `--apply` by hand to act on it:

```sh
make artifacts-retention                        # dry run
.venv/bin/python scripts/prune_artifacts.py --apply
```

Each retained run is printed as `KEEP <path>` and counted in the trailing
`protected=<n>`. A run is protected when a Markdown file under `docs/` cites
either its raw `artifacts/perf/<case>/<release>/<utc>/` path or the promoted
flat copy `docs/evidence/perf/<case>-<utc>/`. `protected=0` means the
"plus runs explicitly referenced by documentation" half of the rule is holding
nothing back — confirm that is intended before `--apply`. The count is runs, not
whitelist entries, because prose globs such as
`artifacts/perf/**/registry-baseline.json` enter the whitelist without
protecting any run.
