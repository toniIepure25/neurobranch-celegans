"""Recover the premotor gap-junction matrix and neuron-name order.

The public Celegans_premotor JLD2 export contains ``A`` and ``d`` but not the
electrical-junction matrix ``W`` or the neuron order.  We rebuild both from the
release's own ``White_data_processed/df5.csv`` and a 205-name order recovered
from the notebook's printed loop.  The script then verifies that every non-zero
entry of ``A`` lies in the chemical-synapse support; this is the check that
caught several wrong candidate orders.

Usage:
    python scripts/build_premotor_inputs.py
    python scripts/build_premotor_inputs.py --premotor-root references/external/Celegans_premotor
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METADATA_NAMES = ROOT / "data" / "metadata" / "premotor_neuron_names.txt"
DEFAULT_PREMOTOR = ROOT / "references" / "external" / "Celegans_premotor"
DEFAULT_OUT = ROOT / "data" / "processed" / "premotor"


def build(premotor_root: Path, out_dir: Path) -> dict[str, object]:
    names_path = METADATA_NAMES
    if not names_path.exists():
        raise FileNotFoundError(f"missing neuron-name metadata: {names_path}")
    names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("neuron-name metadata contains duplicates")
    if len(names) != 205:
        raise ValueError(f"expected 205 names, got {len(names)}")

    df5 = premotor_root / "data" / "White_data_processed" / "df5.csv"
    weights = premotor_root / "data" / "model_weights" / "avg_model_weights.jld2"
    if not df5.exists():
        raise FileNotFoundError(f"missing connectome table: {df5}")
    if not weights.exists():
        raise FileNotFoundError(f"missing fitted weights: {weights}")

    index = {name: i for i, name in enumerate(names)}
    W = np.zeros((len(names), len(names)), dtype=np.float64)
    S = np.zeros_like(W)
    with df5.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            i, j = index[row["Neuron1"]], index[row["Neuron2"]]
            if row["Type"] == "EJ":
                W[i, j] = float(row["Nbr"])
            elif row["Type"] == "S":
                S[i, j] = float(row["Nbr"])

    with h5py.File(weights, "r") as handle:
        # h5py exposes JLD2's column-major array transposed.  Transpose back to
        # the notebook's orientation before checking the chemical support.
        A = np.asarray(handle["A_10takes_avg"], dtype=np.float64).T
    if A.shape != W.shape:
        raise ValueError(f"A shape {A.shape} != W shape {W.shape}")
    # The notebook uses ``S_mat = S_mat'`` before comparing A with the
    # chemical support, so the support matrix is transposed here as well.
    outside = int(((A != 0) & (S.T == 0)).sum())
    if outside:
        raise ValueError(
            f"name order rejected: {outside} non-zero A entries lie outside the chemical support"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "W.npy", W)
    (out_dir / "neuron_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return {
        "n_nodes": len(names),
        "W_nonzero": int((W != 0).sum()),
        "W_sum": float(W.sum()),
        "A_nonzero": int((A != 0).sum()),
        "A_outside_chemical_support": outside,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--premotor-root", type=Path, default=DEFAULT_PREMOTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build(args.premotor_root, args.out)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
