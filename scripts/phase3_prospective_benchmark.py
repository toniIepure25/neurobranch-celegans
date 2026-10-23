"""Run the Phase 3 prospective behavioural benchmark and dataset audit."""

from __future__ import annotations

import json
import subprocess
import sys

from neurobranch.behavior.prospective import write_results
from neurobranch.data.baskoylu import load_all, write_audit


def main() -> None:
    recs, rows = load_all()
    summary = write_audit(recs, rows)
    print(json.dumps(summary, indent=2))
    results = write_results(recs)
    print(json.dumps({k: v.get("aggregate", {}) for k, v in results.items()}, indent=2)[:4000])
    subprocess.run([sys.executable, "scripts/phase3_write_manifest.py"], check=True)


if __name__ == "__main__":
    main()
