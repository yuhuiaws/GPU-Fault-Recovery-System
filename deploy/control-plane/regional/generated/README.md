# Generated regional control-plane manifests

This directory is an API between the role-split renderer and the apply
scripts.

- Sole writer:
  `deploy/control-plane/tools/render-control-plane-role-split.sh`
- Exact apply allowlist: `manifest-list.txt`
- ConfigMap glob consumed by the apply path:
  `gpu-fault-*-config-*.yaml`

Do not edit, add or rename matching YAML by hand. The renderer removes stale
`gpu-fault-*.yaml` files and rewrites `manifest-list.txt`. The apply script
compares that list with the directory contents and fails before contacting
the cluster if they differ.
