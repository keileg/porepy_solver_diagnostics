#!/usr/bin/env bash
# Run a command with the campaign's PorePy and pp_solvers on the path (appended to the
# ambient PYTHONPATH, which carries petsc4py), single-threaded.
#
# POREPY_WORKTREE selects the PorePy checkout (default: /workdir/porepy-campaign).
# Every BLAS/LAPACK library runs on one thread, with MKL's dynamic adjustment off: MKL's
# threaded reductions make a trajectory depend on the thread count, MKL may lower that
# count under load, and MKL also backs PETSc's and hypre's BLAS once pypardiso loaded it.
POREPY_WORKTREE=${POREPY_WORKTREE:-/workdir/porepy-campaign}
export PYTHONPATH=/workdir/pp_solvers-notconverged/src:$POREPY_WORKTREE/src:/workdir/solver_experiments:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
exec "$@"
