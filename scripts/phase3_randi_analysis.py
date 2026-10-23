"""Run the Phase 3 Randi target/context/temporal-shape reanalysis."""

from __future__ import annotations

import json

from neurobranch.evaluation.randi_phase3 import write_randi_phase3


def main() -> None:
    summary = write_randi_phase3()
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()
