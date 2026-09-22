"""Print the comparison table over every run in a results directory."""

from __future__ import annotations

import sys
from pathlib import Path

from solver_experiments.compare import format_table, load_metrics

if __name__ == "__main__":
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    print(format_table(load_metrics(directory)))
