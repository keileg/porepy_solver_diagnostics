#!/usr/bin/env bash
# Run a command with the campaign's PorePy and pp_solvers on the path (appended to the
# ambient PYTHONPATH, which carries petsc4py), three threads per process.
export PYTHONPATH=/workdir/pp_solvers-notconverged/src:/workdir/porepy-campaign/src:/workdir/solver_experiments:$PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${OMP_NUM_THREADS:-3}
exec "$@"