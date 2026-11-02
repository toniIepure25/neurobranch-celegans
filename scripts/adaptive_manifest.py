"""Write and verify the adaptive-dynamics artifact manifest."""

from __future__ import annotations

from neurobranch.adaptive.manifest import verify_manifest, write_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"


def main() -> int:
    payload = write_manifest(OUT)
    ok, problems = verify_manifest(OUT)
    print(f"manifest files: {payload['n_files']}")
    print("verified" if ok else "FAILED")
    for problem in problems:
        print(problem)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
