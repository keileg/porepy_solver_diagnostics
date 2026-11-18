# Analysis of the Coso runs' well telemetry

Offline scripts that read the solver telemetry a Coso run writes
(`results/coso/<run>/rs3/telemetry_main.jsonl`) and print tables about the wells. They
produced the evidence for the diagnosis of the Newton failures at well openings.

They need Python with NumPy only (`wellhead_pressure.py` also needs `iapws`), not
PorePy. Run them from the harness root as `python cases/coso/analysis/<script>.py`;
each script's docstring gives its arguments, and `--help` prints them.

| Script | Question it answers |
|---|---|
| `attempts.py` | Which attempts fall in a time window, and how many failed and iterated: the numbers runs are compared by |
| `failures.py` | Where each failed attempt's temperature updates sit, which grids reach a temperature bound, and how the residual behaves: separates the failure types |
| `flips.py` | Which well fluxes change direction during the iterations of given attempts, with their sign patterns |
| `column_balance.py` | What drives each producer at opening 1: the pressure balance along the column and how its terms change per step |
| `fluid_properties.py` | How the model's fluid compares with steam-table water in the wells (context only) |
| `wellhead_pressure.py` | Which wellhead pressure would keep the producers liquid (context only) |
| `well_records.py` | Shared: reading the telemetry by attempt, and naming the well grids |

## Conventions

- **Attempts** are numbered from 0 in file order; the other scripts take these indices.
  An attempt's start time is that of its step: a failed attempt has no time in its
  record and takes the start time of its retry.
- **Grid names**: wells by their well name (`68-20RD` is the injector, `16A-20` and
  `16B-20` the producers); junctions, the 0-d grids where a well crosses a fracture, as
  `J <well>`, numbered from the top when a well has several; other grids as `g<id>`.
- **Units** are the model's, `pp.Units(kg=1e9)`: pressure in GPa, density in
  1e9 kg/m^3, temperature in K, fluxes in model units. Times are in seconds, or days
  where a name says so.
- **Recorded fluxes** are evaluated after the Newton update, so a recorded sign is the
  upwind direction of the next iteration.
- **Specific to `runscript_3`:** the producer names, the temperature bounds of the cap,
  and in `column_balance.py` the start and length of the opening-1 ramp and the
  producers' wellhead pressures at its start.
