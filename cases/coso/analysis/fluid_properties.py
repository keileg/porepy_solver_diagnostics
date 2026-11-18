"""Model fluid density and viscosity of Coso against steam-table water, in the wells.

    python cases/coso/analysis/fluid_properties.py \
        results/coso/<run>/rs3/telemetry_main.jsonl 11 35 -1

For each listed attempt (indices as printed by ``attempts.py``; negative counts from
the end), takes the well states at its last iteration and prints per well:
- the temperature and pressure ranges;
- the model density minus the tabulated one, absolute and relative;
- the tabulated viscosity relative to the model's constant one;
- the depths where the pressure is below saturation, i.e. where real water would boil.

First it checks that the density formula below reproduces the density the run
recorded, so that the comparison is against the model actually run.

The model: rho = 998.2 exp(4.559e-10 (p - p_atm) - 7.8745e-4 (T - 293.15)) kg/m^3,
viscosity 1.002e-3 Pa s. The tables are rounded values for saturated liquid water
(about 0.5 %), not an IAPWS-IF97 evaluation; compressed-liquid density is estimated as
rho_sat(T) exp(kappa_T(T) (p - p_sat(T))), with kappa_T accurate to about 10 %.

This is context only: the physical fidelity of the setup is not under study.

"""

from __future__ import annotations

import argparse

import numpy as np
from well_records import read_attempts

T_TABLE = np.array(
    [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300.0]
)
"""Temperatures of the tables [°C]."""
RHO_SAT = np.array(
    [998.2, 992.2, 983.2, 971.8, 958.4, 943.1, 926.1, 907.4, 887.0, 864.7, 840.2,
     813.4, 783.6, 750.3, 712.1]
)  # fmt: skip
"""Density of saturated liquid water [kg/m^3]."""
MU_SAT = 1e-3 * np.array(
    [1.002, 0.653, 0.467, 0.355, 0.282, 0.232, 0.197, 0.170, 0.150, 0.134, 0.122,
     0.111, 0.102, 0.094, 0.086]
)  # fmt: skip
"""Viscosity of saturated liquid water [Pa s]."""
P_SAT = 1e6 * np.array(
    [0.0023, 0.0074, 0.0199, 0.0474, 0.1014, 0.1985, 0.3615, 0.6180, 1.0028, 1.5549,
     2.3196, 3.3469, 4.6923, 6.4166, 8.5879]
)  # fmt: skip
"""Saturation pressure [Pa]."""
KAPPA = 1e-9 * np.array(
    [0.459, 0.442, 0.447, 0.465, 0.490, 0.53, 0.58, 0.65, 0.75, 0.87, 1.03, 1.25,
     1.55, 2.0, 2.7]
)  # fmt: skip
"""Isothermal compressibility of liquid water [1/Pa]."""
MODEL_VISCOSITY = 1.002e-3


def tabulated(values: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    return np.interp(temperature - 273.15, T_TABLE, values)


def table_density(pressure: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    above_saturation = pressure - tabulated(P_SAT, temperature)
    return tabulated(RHO_SAT, temperature) * np.exp(
        tabulated(KAPPA, temperature) * above_saturation
    )


def model_density(pressure: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    return 998.2 * np.exp(
        4.559e-10 * (pressure - 101325.0) - 7.8745e-4 * (temperature - 293.15)
    )


def well_states(iteration: dict) -> list[tuple]:
    """Name, cell depths, pressure [Pa], temperature [K], recorded density [kg/m^3]."""
    states = []
    for w in iteration["wells"]["wells"]:
        values = w["variables"]
        states.append(
            (
                w["well_name"],
                w["cell_z"],
                np.array(values["pressure"]["value"]) * 1e9,
                np.array(values["temperature"]["value"]),
                np.array(w["fluid"]["density"]) * 1e9,
            )
        )
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry")
    parser.add_argument("attempts", type=int, nargs="+", help="Attempt indices")
    arguments = parser.parse_args()

    attempts = read_attempts(arguments.telemetry)
    _, _, p, T, recorded = well_states(attempts[arguments.attempts[0]].iterations[-1])[
        0
    ]
    mismatch = np.max(np.abs(model_density(p, T) / recorded - 1))
    print(
        f"model formula against recorded density, largest relative difference "
        f"{mismatch:.1e}"
    )
    for index in arguments.attempts:
        attempt = attempts[index]
        print(f"\nattempt {attempt.index}, step starting at {attempt.start:.0f} s")
        for name, z, p, T, _ in well_states(attempt.iterations[-1]):
            model, table = model_density(p, T), table_density(p, T)
            mu = tabulated(MU_SAT, T) / MODEL_VISCOSITY
            boiling = [
                round(d) for d, pp, tt in zip(z, p, T) if pp < tabulated(P_SAT, tt)
            ]
            print(
                f"  {name:8s} T {T.min():.0f}-{T.max():.0f} K, p {p.min() / 1e6:.2f}-"
                f"{p.max() / 1e6:.1f} MPa; rho model-table {np.min(model - table):+.0f}"
                f"..{np.max(model - table):+.0f} kg/m3 "
                f"({100 * np.min(model / table - 1):+.1f}.."
                f"{100 * np.max(model / table - 1):+.1f} %); mu table/model "
                f"{mu.min():.2f}-{mu.max():.2f}"
                + (f"; below saturation at z {boiling}" if boiling else "")
            )


if __name__ == "__main__":
    main()
