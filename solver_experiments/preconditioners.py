"""Preconditioner configurations for `pp_solvers` that are not (yet) part of it.

:func:`thm_factory_system_amg` is copied from the Coso repository (branch
``october-attempt``, commit ``2f95e65``, ``nl_params.py``), and
:data:`SYSTEM_AMG_OPTIONS` from its ``solver_configurations.py``. It differs from
``pp_solvers.thm_factory`` in the last stage of the nested Schur complements: instead
of CPR (AMG on pressure, then ILU on temperature and pressure together), one hypre
BoomerAMG acts on the interleaved temperature–pressure block. Temperature rows are also
scaled by the specific volume. It is here to test whether this handles large
temperature contrasts better than CPR, on cases other than Coso.

"""

from __future__ import annotations

from typing import Any

import pp_solvers
from pp_solvers.equation_variable_groups import (
    DefaultEquationVariableGroups,
    EquationVariableGroup,
)
from pp_solvers.preconditioners import (
    AMG,
    GMRES,
    ILU,
    BlockDiagonalInverter,
    BlockDiagonalPreconditioner,
    DiagonalInverter,
    FixedStressInverter,
    LinearSolverConfiguration,
    PythonPermutationWrapper,
    nested_schur_complements,
)
from pp_solvers.transformations import ContactLinearTransformation, ScaleSpecificVolume

SYSTEM_AMG_OPTIONS: dict[str, Any] = {
    "gmres": {"ksp_max_it": 400},
    "system_amg": {
        "pc_hypre_boomeramg_P_max": 16,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_smooth_type": "ilu",
        "pc_hypre_boomeramg_ilu_level": 0,
    },
}
"""The Coso PETSc options for the system-AMG stage, without the Krylov monitors."""


def thm_factory_system_amg() -> LinearSolverConfiguration:
    """GMRES with nested Schur complements and a system AMG for temperature–pressure."""
    contact_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.contact_mechanics_group
    ]
    interface_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.interface_darcy_flux_group,
        DefaultEquationVariableGroups.interface_enthalpy_flux_group,
        DefaultEquationVariableGroups.interface_fourier_flux_group,
        DefaultEquationVariableGroups.well_flux_group,
        DefaultEquationVariableGroups.well_enthalpy_flux_group,
    ]
    mechanics_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.mechanics_group,
        DefaultEquationVariableGroups.interface_force_balance_group,
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.mass_balance_pressure_matrix_group,
        DefaultEquationVariableGroups.mass_balance_pressure_fractures_group,
        DefaultEquationVariableGroups.mass_balance_pressure_intersections_group,
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.energy_balance_temperature_group,
    ]
    solver = GMRES(
        preconditioner=nested_schur_complements(
            [
                {
                    "subsolver": BlockDiagonalPreconditioner(
                        groups=contact_groups, key="contact"
                    ),
                    "approximate_inverter": BlockDiagonalInverter(),
                },
                {
                    "subsolver": ILU(groups=interface_groups, key="interface_flow"),
                    "approximate_inverter": DiagonalInverter(),
                },
                {
                    "subsolver": AMG(
                        groups=mechanics_groups,
                        key="mechanics_amg",
                        vector_problem=True,
                    ),
                    "approximate_inverter": FixedStressInverter(),
                },
                {
                    "subsolver": PythonPermutationWrapper(
                        permutation_groups=[
                            energy_balance_groups,
                            mass_balance_groups,
                        ],
                        inner_subsolver=AMG(
                            groups=energy_balance_groups + mass_balance_groups,
                            key="system_amg",
                        ),
                    ),
                },
            ]
        )
    )
    return LinearSolverConfiguration(
        transformations=[
            ContactLinearTransformation(),
            ScaleSpecificVolume(groups=energy_balance_groups),
        ],
        solver=solver,
    )


def system_amg_solver(options: dict[str, Any]) -> pp_solvers.IterativeLinearSolver:
    """The system-AMG solver, with ``options`` overriding the Coso ones per key."""
    merged = {
        key: {**SYSTEM_AMG_OPTIONS.get(key, {}), **options.get(key, {})}
        for key in {*SYSTEM_AMG_OPTIONS, *options}
    }
    return pp_solvers.IterativeLinearSolver(
        configuration_factory=thm_factory_system_amg, solver_options=merged
    )
