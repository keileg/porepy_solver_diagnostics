"""Wellhead pressure that would keep the Coso producers liquid, from IAPWS-IF97.

    python cases/coso/analysis/wellhead_pressure.py \
        results/coso/<run>/rs3/telemetry_main.jsonl 11 35 -1

Needs the ``iapws`` package. For each listed attempt (indices as printed by
``attempts.py``), takes the producers' junction states at its last iteration and prints:
- the feed (junction) depth, pressure and temperature;
- the saturation pressure at the feed temperature;
- the closed-in wellhead pressure of a column at the feed temperature, integrated
  upwards without friction, with IAPWS-IF97 density and with the model's density.
  This is the largest wellhead pressure at which the well can flow at all.

Then the driving head of the injector-producer loop at no flow, for three feed
temperatures: the pressure at 2900 m under a 313 K injector column from 0.3 MPa,
against that under a producer column from a wellhead held 0.3 MPa above saturation.

The IAPWS density is that of Region 1 (liquid), also below the saturation pressure,
where it describes metastable liquid: that is what a column not allowed to flash would
hold. This is context only: the physical fidelity of the setup is not under study.

"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
from iapws import IAPWS97
from iapws.iapws97 import _Region1
from well_records import PRODUCERS, junction_of, read_attempts

GRAVITY = 9.80665
INJECTOR_WELLHEAD = 0.3e6
"""Injector wellhead pressure [Pa]."""
INJECTOR_TEMPERATURE = 313.15
"""Injection temperature [K]."""

Density = Callable[[float, float], float]


def iapws_density(pressure: float, temperature: float) -> float:
    """Liquid density [kg/m^3] from IAPWS-IF97 Region 1, also below saturation."""
    return 1.0 / _Region1(temperature, pressure / 1e6)["v"]


def model_density(pressure: float, temperature: float) -> float:
    return 998.2 * np.exp(
        4.559e-10 * (pressure - 101325.0) - 7.8745e-4 * (temperature - 293.15)
    )


def closed_in_wellhead(
    feed_pressure: float, feed_depth: float, temperature: float, density: Density
) -> float:
    """Pressure [Pa] at the surface of a static column above the feed."""
    pressure, dz, steps = feed_pressure, -feed_depth / 600, 600
    for _ in range(steps):
        pressure -= density(pressure, temperature) * GRAVITY * dz
    return pressure


def pressure_at_depth(
    depth: float, temperature: float, top_pressure: float, density: Density
) -> float:
    """Pressure [Pa] at ``depth`` (negative) under a static column from the surface."""
    pressure, dz, steps = top_pressure, -depth / 600, 600
    for _ in range(steps):
        pressure += density(pressure, temperature) * GRAVITY * dz
    return pressure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry")
    parser.add_argument("attempts", type=int, nargs="+", help="Attempt indices")
    arguments = parser.parse_args()

    attempts = read_attempts(arguments.telemetry)
    for index in arguments.attempts:
        attempt = attempts[index]
        print(f"attempt {attempt.index}, step starting at {attempt.start:.0f} s")
        for name in PRODUCERS:
            junction = junction_of(attempt.iterations[-1], name)
            depth = junction["z"]
            pressure = junction["variables"]["pressure"]["value"][0] * 1e9
            temperature = junction["variables"]["temperature"]["value"][0]
            saturation = IAPWS97(T=temperature, x=0).P * 1e6
            iapws = closed_in_wellhead(pressure, depth, temperature, iapws_density)
            model = closed_in_wellhead(pressure, depth, temperature, model_density)
            print(
                f"  {name}: feed z {depth:.0f} m, p {pressure / 1e6:.2f} MPa, "
                f"T {temperature:.0f} K; p_sat(T) {saturation / 1e6:.2f} MPa; "
                f"closed-in wellhead p: IAPWS {iapws / 1e6:.2f}, "
                f"model {model / 1e6:.2f} MPa"
            )

    print("\nDriving head of the injector-producer loop at no flow (IAPWS):")
    for feed_temperature in (460.0, 490.0, 500.0):
        wellhead = IAPWS97(T=feed_temperature, x=0).P * 1e6 + 0.3e6
        cold = pressure_at_depth(
            -2900.0, INJECTOR_TEMPERATURE, INJECTOR_WELLHEAD, iapws_density
        )
        hot = pressure_at_depth(-2900.0, feed_temperature, wellhead, iapws_density)
        print(
            f"  feed {feed_temperature:.0f} K, producer wellhead {wellhead / 1e6:.2f} "
            f"MPa: p(2900 m) under injector {cold / 1e6:.2f} MPa, under producer "
            f"{hot / 1e6:.2f} MPa, surplus {(cold - hot) / 1e6:+.2f} MPa"
        )


if __name__ == "__main__":
    main()
