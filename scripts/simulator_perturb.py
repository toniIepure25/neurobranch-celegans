"""Sweep named-neuron virtual perturbations and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.interventions.search import random_control_pool
from neurobranch.simulator.branching import reversal_states
from neurobranch.simulator.perturbation import (
    CANDIDATES,
    baseline_for,
    candidate_specs,
    effect_row,
    initial_state_specs,
)
from neurobranch.simulator.train import checkpoint_root, load_datasets, load_simulator_for
from neurobranch.simulator.types import PerturbationSpec


def _frames(data, n: int, horizon: int) -> list[int]:
    rev = reversal_states(data, horizon=horizon)
    if len(rev) >= n:
        return [int(t) for t in rev[:n]]
    test = [int(t) for t in np.flatnonzero(data.masks["test"]) if t + horizon < data.n_frames]
    return test[:n]


def _run_specs(
    sim,
    frames: list[int],
    specs: list[PerturbationSpec],
    *,
    horizon: int,
    n_rollouts: int,
    noise_std: float,
    base_seed: int,
    model_seed: int,
    control_kind: str,
    baselines: dict[int, Any],
) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        base = baselines[frame]
        for spec in specs:
            rows.append(
                effect_row(
                    sim,
                    frame,
                    spec,
                    base,
                    horizon=horizon,
                    n_rollouts=n_rollouts,
                    noise_std=noise_std,
                    base_seed=base_seed,
                    control_kind=control_kind,
                    model_seed=model_seed,
                )
            )
    return rows


def _aggregate(df: pd.DataFrame, key: str = "target") -> pd.DataFrame:
    if df.empty:
        return df
    out = (
        df.groupby([key, "model_seed"])
        .agg(
            mean_delta_tv=("delta_tv", "mean"),
            abs_delta_tv=("delta_tv", lambda s: float(np.mean(np.abs(s)))),
            mean_delta_js=("delta_js", "mean"),
            abs_delta_js=("delta_js", lambda s: float(np.mean(np.abs(s)))),
            mean_shift_sd=("trajectory_shift_sd", "mean"),
            mean_terminal_sd=("terminal_shift_sd", "mean"),
            mean_ood=("perturbed_ood_fraction", "mean"),
            n=("delta_tv", "size"),
        )
        .reset_index()
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--controls", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[17, 23, 31])
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = args.reports.resolve()
    selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
    selected = str(selection["selected_kind"])
    rng = np.random.default_rng(args.seed)
    datasets = load_datasets(cfg)

    main_rows: list[dict[str, Any]] = []
    initial_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for data in datasets:
        frames = _frames(data, args.frames, args.horizon)
        sims = {seed: load_simulator_for(data, selected, seed, root=root) for seed in args.model_seeds}
        baselines: dict[int, Any] = {}
        for frame in frames:
            baselines[frame] = baseline_for(
                sims[args.seed],
                frame,
                horizon=args.horizon,
                n_rollouts=args.rollouts,
                noise_std=args.noise_std,
                base_seed=args.seed * 10000 + frame,
            )

        main_rows.extend(
            _run_specs(
                sims[args.seed],
                frames,
                candidate_specs(),
                horizon=args.horizon,
                n_rollouts=args.rollouts,
                noise_std=args.noise_std,
                base_seed=args.seed * 10000,
                model_seed=args.seed,
                control_kind="candidate",
                baselines=baselines,
            )
        )
        initial_rows.extend(
            _run_specs(
                sims[args.seed],
                frames,
                initial_state_specs(),
                horizon=args.horizon,
                n_rollouts=args.rollouts,
                noise_std=args.noise_std,
                base_seed=args.seed * 10000,
                model_seed=args.seed,
                control_kind="initial_state",
                baselines=baselines,
            )
        )

        dose_specs: list[PerturbationSpec] = []
        for target in CANDIDATES:
            for amp in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                dose_specs.append(
                    PerturbationSpec(target=target, amplitude=amp, onset=0, duration=10)
                )
        dose_rows.extend(
            _run_specs(
                sims[args.seed],
                frames,
                dose_specs,
                horizon=args.horizon,
                n_rollouts=args.rollouts,
                noise_std=args.noise_std,
                base_seed=args.seed * 10000,
                model_seed=args.seed,
                control_kind="dose",
                baselines=baselines,
            )
        )

        time_specs: list[PerturbationSpec] = []
        for target in CANDIDATES:
            for onset in (0, 5, 10):
                time_specs.append(
                    PerturbationSpec(target=target, amplitude=1.0, onset=onset, duration=5)
                )
        timing_rows.extend(
            _run_specs(
                sims[args.seed],
                frames,
                time_specs,
                horizon=args.horizon,
                n_rollouts=args.rollouts,
                noise_std=args.noise_std,
                base_seed=args.seed * 10000,
                model_seed=args.seed,
                control_kind="timing",
                baselines=baselines,
            )
        )

        for model_seed in args.model_seeds:
            b = {
                frame: baseline_for(
                    sims[model_seed],
                    frame,
                    horizon=args.horizon,
                    n_rollouts=args.rollouts,
                    noise_std=args.noise_std,
                    base_seed=args.seed * 10000 + frame,
                )
                for frame in frames
            }
            seed_rows.extend(
                _run_specs(
                    sims[model_seed],
                    frames,
                    candidate_specs(),
                    horizon=args.horizon,
                    n_rollouts=args.rollouts,
                    noise_std=args.noise_std,
                    base_seed=args.seed * 10000,
                    model_seed=model_seed,
                    control_kind="model_seed",
                    baselines=b,
                )
            )

        pool = random_control_pool(data)
        picks = list(rng.choice(pool, size=min(args.controls, len(pool)), replace=False))
        control_specs = [
            PerturbationSpec(target=str(name), amplitude=1.0, onset=0, duration=10) for name in picks
        ]
        control_rows.extend(
            _run_specs(
                sims[args.seed],
                frames,
                control_specs,
                horizon=args.horizon,
                n_rollouts=args.rollouts,
                noise_std=args.noise_std,
                base_seed=args.seed * 10000,
                model_seed=args.seed,
                control_kind="random",
                baselines=baselines,
            )
        )

    main = pd.DataFrame(main_rows)
    initial = pd.DataFrame(initial_rows)
    dose = pd.DataFrame(dose_rows)
    timing = pd.DataFrame(timing_rows)
    seed = pd.DataFrame(seed_rows)
    controls = pd.DataFrame(control_rows)
    for name, df in [
        ("perturbation_effects.csv", main),
        ("perturbation_initial_state.csv", initial),
        ("perturbation_dose.csv", dose),
        ("perturbation_timing.csv", timing),
        ("perturbation_seed.csv", seed),
        ("perturbation_controls.csv", controls),
    ]:
        df.to_csv(reports / name, index=False)

    # Specificity: candidate |effect| versus the random-neuron control tail.
    control_abs = controls["delta_tv"].abs().to_numpy()
    q95 = float(np.quantile(control_abs, 0.95)) if len(control_abs) else float("nan")
    rows: list[dict[str, Any]] = []
    for target in CANDIDATES:
        for direction in (1, -1):
            sub = main[(main.target == target) & np.isclose(main.amplitude, direction)]
            if sub.empty:
                continue
            worm_turn = sub.groupby("worm_id")["delta_turn"].mean()
            effect_sign = float(np.sign(worm_turn.mean()))
            same_sign = int(np.sum(np.sign(worm_turn) == effect_sign)) if effect_sign != 0 else 0
            abs_mean = float(np.mean(np.abs(sub.groupby("worm_id")["delta_tv"].mean())))
            seed_sub = seed[(seed.target == target) & np.isclose(seed.amplitude, direction)]
            seed_turn = seed_sub.groupby("model_seed")["delta_turn"].mean().to_dict()
            seed_same = int(sum(np.sign(v) == effect_sign for v in seed_turn.values())) if effect_sign != 0 else 0
            rows.append(
                {
                    "target": target,
                    "direction": "activation" if direction == 1 else "inhibition",
                    "abs_mean_delta_tv": abs_mean,
                    "mean_delta_turn": float(worm_turn.mean()),
                    "direction_of_effect": "more TURN" if effect_sign > 0 else ("less TURN" if effect_sign < 0 else "none"),
                    "mean_shift_sd": float(sub["trajectory_shift_sd"].mean()),
                    "worms_same_sign": same_sign,
                    "seed_direction_agreement": seed_same,
                    "seed_effects_delta_turn": json.dumps({str(k): float(v) for k, v in seed_turn.items()}),
                    "specific_vs_random_q95": bool(abs_mean > q95),
                    "robust_across_worms": bool(same_sign >= 4),
                    "robust_across_seeds": bool(seed_same >= 2),
                    "ood_fraction": float(sub["perturbed_ood_fraction"].mean()),
                }
            )
    specificity = pd.DataFrame(rows).sort_values("abs_mean_delta_tv", ascending=False) if rows else pd.DataFrame()
    specificity.to_csv(reports / "perturbation_specificity.csv", index=False)

    # Dose monotonicity: Spearman rho of signed amplitude and delta_tv.
    from scipy.stats import spearmanr

    dose_summary = []
    for target in CANDIDATES:
        sub = dose[dose.target == target]
        if sub.empty:
            continue
        rho, p = spearmanr(sub["amplitude"], sub["delta_tv"])
        dose_summary.append(
            {
                "target": target,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n": int(len(sub)),
            }
        )
    pd.DataFrame(dose_summary).to_csv(reports / "perturbation_dose_summary.csv", index=False)

    top = specificity.iloc[0].to_dict() if len(specificity) else {}
    gate = {
        "status": "pass"
        if len(specificity) and bool((specificity["specific_vs_random_q95"] & specificity["robust_across_worms"] & specificity["robust_across_seeds"]).any())
        else "fail",
        "criterion": (
            "at least one named-neuron direction has mean |delta TV| above the random-neuron "
            "95th percentile, agrees in sign in >=4/5 worms, and agrees in >=2/3 model seeds"
        ),
        "random_control_q95_abs_delta_tv": q95,
        "n_candidates": len(CANDIDATES),
        "top_effect": top,
        "warning": (
            "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. "
            "No empirical stimulation-response model is registered; experimental category C "
            "is not_evaluable on Kato."
        ),
        "empirical_response_status": "not_evaluable",
        "dose_summary": dose_summary,
        "n_rows": {
            "main": int(len(main)),
            "initial_state": int(len(initial)),
            "dose": int(len(dose)),
            "timing": int(len(timing)),
            "seed": int(len(seed)),
            "random_controls": int(len(controls)),
        },
    }
    (reports / "perturbation_summary.json").write_text(
        json.dumps({"schema_version": 1, "gate_5": gate}, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(main)} main perturbation rows; gate_5={gate['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
