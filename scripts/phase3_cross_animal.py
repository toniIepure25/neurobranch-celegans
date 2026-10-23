"""Run the Phase 3 cross-animal one-step dynamics probe."""

from __future__ import annotations

import json

from neurobranch.data.baskoylu import load_all
from neurobranch.dynamics.cross_animal import write_cross_animal


def main() -> None:
    recs, _ = load_all()
    summary = write_cross_animal(recs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
