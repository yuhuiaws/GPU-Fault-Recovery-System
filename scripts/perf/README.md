# Performance tooling

The supported regional capacity entry point is:

```bash
scripts/perf/regional_capacity_suite.py all \
  --case burst \
  --clusters 32
```

Supported entry points:

- `regional_capacity_suite.py`: registration, mixed/burst/multicluster load,
  metrics, Aurora sampling, teardown and artifacts.
- `regional_action_capacity_suite.py`: regional action execution capacity.
- `seed_regional_action_workflows.py` and
  `benchmark_regional_action_executor.py`: action-side data and load drivers.
- `capture_processor_inflight.py`: optional in-flight sampling attached to a
  suite artifact directory.
- `control_plane_capacity_probe.py`: control-plane capacity probe helper.

`payload-templates.json` and the three benchmark modules selected by
`regional_capacity_suite.py` are implementation inputs, not standalone entry
points.

The supported CLI remains in `regional_capacity_suite.py`; its implementation
is split by responsibility:

- `regional_capacity_job.py`: Indexed Job and start-gate manifest construction.
- `regional_capacity_database.py`: Aurora and PostgreSQL observations.
- `regional_capacity_results.py`: metric parsing, aggregation and artifact
  lifecycle.

Additional focused utilities:

- `benchmark_processor_admission_storm.py`
- `generate_watcher_pod_storm.py`
- `validate_hot_state_defaults.py`

Superseded one-Job-per-benchmark harnesses are retained only in the private
archive and are not part of the public performance tooling.
