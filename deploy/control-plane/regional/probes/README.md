# Control-plane probes

Each file here is a standalone Python program that the release engine ships to a
**running** control-plane Pod as `python -c "<source>"`, reading a JSON request
on stdin and printing a JSON response on stdout.

They used to be triple-quoted string literals inside the engine modules. As
strings they were invisible to every gate in the repo: no `ruff`, no `mypy`
against the real `gpu_fault` models, no function-length or code-size audit, and
no way to unit test them except by asserting on substrings of the literal. A
typo in one of them failed in production, in a Pod, mid-release.

Two properties have to be preserved, and they are why the source is *read* at
import time rather than imported:

1. **The engine carries the probe, the Pod does not.** A probe is executed by
   whatever `gpu_fault` version is already deployed. Turning these into
   `python -m gpu_fault.some_probe` would bind probe availability to the
   deployed image, so a new engine could not probe an old Pod -- exactly the
   case a rollback needs.
2. **The wire protocol is unchanged.** stdin JSON in, stdout JSON out, one
   object per line. Moving a probe is a pure relocation; if a diff here changes
   a key name it is a behaviour change, not a refactor.

Some probes do not fit that shape, and each says so in its own docstring. The
exceptions are also registered in `tests/regional/test_release_probes.py`, so a
sixth one cannot appear by accident:

- `store_io_rejection_series.py` takes its parameters as one JSON object in
  `sys.argv[1]`. The `kubectl exec` that runs it has no `-i`, so stdin is not
  attached, and adding `-i` for two scalars would change how that exec handles
  stdin.
- `registry_client.py` takes `method path` in argv, because its stdin is the
  request body being sent to the registry API and the method and path cannot
  travel inside it.
- `control_api_inspect.py` and `executor_tls_healthz.py` take their inputs from
  the **environment**, via `env NAME=value` in front of the interpreter and via
  variables the Pod already carries. Nothing is read from stdin.
- `critical_amp_alerts.py` runs on the **deploy host** rather than in a Pod: it
  SigV4-signs a query to the AMP workspace with the deploy host's credentials,
  which the Pods deliberately do not have. It is here for the same reasons as
  the rest -- lint, types, and a name in the deploy log -- not because of where
  it runs.
- `gpu_endpoint_gate.py` is not exec'd into a running Pod at all: the bootstrap
  puts it in the `command` of a short-lived Pod it creates in the GPU cluster,
  because the question is whether a Pod *there* can resolve, trust and
  authenticate to the control plane. It reads its CA and cluster token from a
  mounted Secret rather than from env, so neither lands in the Pod spec.

Consequences for anything added here:

- Import only the standard library and `gpu_fault`. Never import from the
  deploy tree -- none of it exists inside the Pod. A probe that runs on the
  deploy host instead may use the deploy host's dependencies (`boto3`), but it
  has to say in its docstring that it is not Pod-executed.
- Keep the program at module level with a `main()`-style guard only if it does
  not change what the engine sees on stdout.
- `deploy/` is in `mypy`'s target list, so these are type-checked strictly
  against the same `gpu_fault` the Pod runs.
