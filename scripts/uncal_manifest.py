"""Write the SHA256 manifest for uncertainty-calibration artifacts."""

from __future__ import annotations

from neurobranch.adaptive.manifest import write_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/uncertainty_calibration"


def main() -> None:
    payload = write_manifest(OUT)
    print(f"wrote manifest for {payload['n_files']} files")


if __name__ == "__main__":
    main()
