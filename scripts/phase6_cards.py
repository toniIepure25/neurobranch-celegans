"""Write Phase 6 model cards from frozen selection and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, sha256_file
from neurobranch.simulator.train import checkpoint_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = Path(args.reports)
    cards = reports / "model_cards"
    cards.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((reports / "selection_frozen.json").read_text(encoding="utf-8"))
    test = pd.read_csv(reports / "test_summary.csv")
    written = []
    for family in ("var", "stochastic_var", "switching_var"):
        spec = frozen["families"][family]
        for worm, checkpoint_hash in spec.get("checkpoints", {}).items():
            path = root / "phase6" / family / f"worm{worm}.pt"
            rows = test[(test["family"] == family) & (test["worm_id"] == int(worm))]
            metrics = (
                rows.groupby("horizon", as_index=False)
                .agg(
                    mean_skill=("skill_vs_persistence", "mean"),
                    mean_energy=("energy_distance", "mean"),
                    mean_ood=("ood_fraction", "mean"),
                    mean_autocorr=("autocorr_error", "mean"),
                    finite=("finite_fraction", "mean"),
                )
                .to_dict(orient="records")
            )
            card = {
                "schema_version": 1,
                "phase": 6,
                "family": family,
                "worm_id": int(worm),
                "selection": spec.get("label"),
                "config": spec.get("config"),
                "checkpoint": str(path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": checkpoint_hash,
                "data_sha256": sha256_file(cfg.data.path),
                "config_sha256": sha256_file(args.config),
                "split": {"train_blocks": [0, 1, 3], "val_block": 4, "test_block": 2, "purge": 12},
                "history": int(getattr(spec.get("config", {}), "get", lambda k, d=None: d)("lag", 1))
                if isinstance(spec.get("config"), dict)
                else 1,
                "native_stochastic": family != "var",
                "test_metrics": metrics,
                "evidence_tier": "exploratory",
                "test_block_reused": True,
                "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
                "limitations": [
                    "standardised calcium traces, not membrane potential or spikes",
                    "immobilised, unstimulated WT worms",
                    "test block 2 was inspected in Phase 5",
                    "no matched empirical intervention for biological validation",
                ],
            }
            target = cards / f"{family}_worm{worm}.json"
            target.write_text(json.dumps(card, indent=2, default=float) + "\n", encoding="utf-8")
            written.append(str(target.relative_to(PROJECT_ROOT)))
    print(json.dumps({"cards": len(written), "paths": written[:5]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
