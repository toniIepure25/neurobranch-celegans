"""Discover or fit simulator checkpoints and write model cards.

The primary mode reuses the frozen Phase 2 checkpoints after validating their
metadata.  ``--fit-missing`` fits only absent checkpoints with the frozen
configuration; it never overwrites an existing checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.simulator.train import (
    DEFAULT_KINDS,
    DEFAULT_SEEDS,
    checkpoint_root,
    discover_available,
    fit_missing,
    load_datasets,
    select_on_validation,
    write_cards,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator")
    parser.add_argument("--fit-missing", action="store_true")
    parser.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    datasets = load_datasets(cfg)
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    available = discover_available(datasets, root=root, kinds=tuple(args.kinds), seeds=tuple(args.seeds))
    print("available checkpoints:", available)
    fitted = []
    if args.fit_missing:
        fitted = fit_missing(
            datasets,
            cfg,
            root=root,
            kinds=tuple(args.kinds),
            seeds=tuple(args.seeds),
        )
        if fitted:
            pd.DataFrame(fitted).to_csv(reports / "fit_missing.csv", index=False)

    cards = write_cards(
        datasets,
        root=root,
        output=reports,
        kinds=tuple(args.kinds),
        seeds=tuple(args.seeds),
    )
    pd.DataFrame(cards).to_csv(reports / "models.csv", index=False)

    winner, rows, means = select_on_validation(
        datasets,
        root=root,
        kinds=tuple(args.kinds),
        seeds=tuple(args.seeds),
    )
    payload = {
        "schema_version": 1,
        "selection_split": "validation block 4",
        "test_block_used_for_selection": False,
        "criterion": "mean validation R2; tie lower 5-step MSE; tie simpler model",
        "kinds": list(args.kinds),
        "seeds": list(args.seeds),
        "available": available,
        "means": means,
        "selected_kind": winner,
        "selected_means": means.get(winner),
        "n_models": len(cards),
        "n_fitted_missing": len(fitted),
        "checkpoint_root": str(root),
        "rows": rows,
    }
    (reports / "selection.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cards)} model cards; validation-selected model: {winner}")
    print(json.dumps(means, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
