# solver_experiments

Runs of PorePy's geothermal reservoir example under systematic parameter variation, to
find out where its nonlinear and linear solvers actually struggle.

The case is `porepy.examples.geothermal_reservoir`: a 1 km cube of basalt at depth, two
elliptic fractures dipping at ±45°, an injection and a production well, thermo-
poromechanics with frictional contact. `CaseSpec()` and `SolverSpec()` with no arguments
reproduce the shipped configuration exactly — they modify the example's own
`set_model_params`/`set_solver_params` rather than restating them, so the baseline
cannot drift away from what PorePy ships.

## Layout

| Path | What |
|---|---|
| `solver_experiments/case.py` | `CaseSpec` (the varied axes), `SolverSpec`, model and solver construction |
| `solver_experiments/linear_solvers.py` | PETSc direct solvers, row equilibration, per-solve residual checking |
| `solver_experiments/instrumentation.py` | Wall-clock breakdown (Jacobian assembly, residual evaluation, linear solve) |
| `solver_experiments/metrics.py` | Metrics derived from PorePy's solver telemetry |
| `solver_experiments/runner.py` | `run_case(case, solver, directory)` → `metrics.json`, `telemetry.jsonl` |
| `solver_experiments/compare.py` | Metrics files → comparison table |
| `scripts/run_case.py` | One run, configured from the command line |
| `scripts/compare.py` | The table over a results directory |
| `results/` | One directory per run (not committed) |

## Running

The runs need a PorePy checkout that carries the solver telemetry, which is not on
`develop` yet:

```bash
cd /workdir/solver_experiments
PYTHONPATH=/workdir/porepy-campaign/src:.:/workdir/petsc_dist/lib python scripts/run_case.py --schedule-length 3
PYTHONPATH=/workdir/porepy-campaign/src:.:/workdir/petsc_dist/lib python scripts/compare.py results
```

Each run writes `metrics.json` (the comparison input), `telemetry.jsonl` (PorePy's
per-iteration records) and `viz/` (the model's own output) into its own directory under
`results/`.

## Axes

In the order they are being explored: injection temperature, injection pressure,
lithostatic stress anisotropy, mesh size (fractures before matrix), dilation angle. Each
axis is varied on its own, from the shipped configuration, in steps small enough that a
failure can be attributed to the step that caused it.

## Environment notes

* The linear solver matters more than usual here: the assembled system is very badly
  row-scaled, and pypardiso returns a wrong answer on it without reporting failure. Every
  solve is therefore residual-checked, and `max_relative_residual` in the metrics is the
  field that says whether a run's linear algebra can be trusted at all.
* `pp_solvers` (the PETSc bridge, for iterative solvers) needs `scikit-learn` installed;
  its `solver_selection` subpackage imports it at module level.

## Where the write-ups live

Interpretive reports go to `/workdir/porepy/.claude/solver-telemetry/`, which is where
this project keeps its diagnosis write-ups. This repository holds the runnable harness
and the machine-readable results.
