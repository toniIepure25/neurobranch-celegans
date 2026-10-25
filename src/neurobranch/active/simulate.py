"""Pseudo-prospective active acquisition benchmark for Phase 7.

The outer test targets are frozen before the simulation.  An acquisition policy
may only reveal targets from the outer-training pool.  Every decision, candidate
pool hash and selected target is logged so that leakage is auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from neurobranch.active.acquisition import STRATEGIES, AcquisitionContext, select_batch
from neurobranch.active.state_features import (
    attach_state_features,
    fit_state_projector,
    load_state_cache,
)
from neurobranch.benchmark.conformal import conformal_interval, conformal_quantile
from neurobranch.benchmark.metrics import (
    coverage,
    event_metrics,
    interval_score,
    pair_metrics,
)
from neurobranch.benchmark.reliability import (
    model_profile_correlation,
    split_half_target_profiles,
)
from neurobranch.benchmark.splits import build_folds
from neurobranch.causal.features import FeatureBundle, build_feature_bundle
from neurobranch.causal.response_operator import (
    CROConfig,
    CROEnsemble,
    fit_cro,
)
from neurobranch.config import PROJECT_ROOT, git_state, resolve_path, sha256_file


@dataclass(frozen=True)
class ActiveConfig:
    trials_path: str = "reports/phase3/randi_trial_table_phase3.csv.gz"
    connectome_path: str = "references/external/aconnectome_witvliet_2020_8_utf8.csv"
    output_dir: str = "reports/phase7"
    outer_split: str = "unseen_targets"
    n_folds: int = 5
    initial_fraction: float = 0.20
    budgets: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50)
    reference_budget: float = 0.60
    primary_budget: float = 0.40
    batch_targets: int = 8
    strategies: tuple[str, ...] = STRATEGIES
    acquisition_seeds: tuple[int, ...] = (101, 202, 303)
    model_seeds: tuple[int, ...] = (17, 23, 31)
    final_model_seeds: tuple[int, ...] = (17, 23, 31, 41, 53)
    rank: int = 16
    hidden: int = 64
    epochs: int = 12
    batch_size: int = 8192
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    patience: int = 8
    response_column: str = "response_mean"
    n_boot: int = 300
    use_state: bool = False
    state_path: str | None = None
    smoke: bool = False

    def cro_config(self, seed: int) -> CROConfig:
        return CROConfig(
            rank=self.rank,
            hidden=self.hidden,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            seed=seed,
            use_state=self.use_state,
        )


def _target_ids(bundle: FeatureBundle, names: set[str] | list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(bundle.target_names)}
    return np.array([lookup[str(n)] for n in names if str(n) in lookup], dtype=np.int64)


def _subset_rows(bundle: FeatureBundle, fold_idx: np.ndarray, targets: set[str]) -> np.ndarray:
    ids = _target_ids(bundle, targets)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64)
    return fold_idx[np.isin(bundle.target_idx[fold_idx], ids)]


def _choose_seed_targets(
    pool: list[str],
    counts: dict[str, int],
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Stratified deterministic seed set covering rare and common targets."""
    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))
    order = sorted(pool, key=lambda t: (-counts.get(t, 0), t))
    strata = [order[i::3] for i in range(3)]
    quotas = [n // 3, n // 3, n - 2 * (n // 3)]
    chosen: list[str] = []
    for stratum, quota in zip(strata, quotas):
        if not stratum or quota <= 0:
            continue
        take = min(quota, len(stratum))
        chosen.extend(rng.choice(stratum, size=take, replace=False).tolist())
    remaining = [t for t in pool if t not in set(chosen)]
    if len(chosen) < n:
        chosen.extend(rng.choice(remaining, size=n - len(chosen), replace=False).tolist())
    return sorted(set(chosen))


def _additive_prediction(
    bundle: FeatureBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Target + responder means minus global mean, fitted on ``train_idx``."""
    global_mean = float(np.mean(bundle.y[train_idx]))
    t_frame = pd.DataFrame({"g": bundle.target_idx[train_idx], "y": bundle.y[train_idx]})
    r_frame = pd.DataFrame({"g": bundle.responder_idx[train_idx], "y": bundle.y[train_idx]})
    t_means = t_frame.groupby("g")["y"].mean()
    r_means = r_frame.groupby("g")["y"].mean()
    t_vals = np.array([t_means.get(int(i), global_mean) for i in bundle.target_idx[test_idx]])
    r_vals = np.array([r_means.get(int(i), global_mean) for i in bundle.responder_idx[test_idx]])
    return t_vals + r_vals - global_mean


def _discrepancy_scores(
    bundle: FeatureBundle,
    *,
    acquired_idx: np.ndarray,
    acquired_targets: set[str],
    candidate_targets: list[str],
) -> dict[str, float]:
    """Predict anatomy-unexplained residual magnitude for unseen targets.

    ``D_t`` is the mean absolute residual of a simple additive anatomical prior
    on already acquired rows.  For candidates it is extrapolated with a ridge on
    target anatomy features.  This is an acquisition heuristic, not a causal
    quantity.
    """
    if not acquired_targets or not candidate_targets:
        return {t: 0.0 for t in candidate_targets}
    acquired_pred = _additive_prediction(bundle, acquired_idx, acquired_idx)
    residual = np.abs(bundle.y[acquired_idx] - acquired_pred)
    frame = pd.DataFrame(
        {
            "target": [bundle.target_names[i] for i in bundle.target_idx[acquired_idx]],
            "residual": residual,
        }
    )
    observed = frame.groupby("target")["residual"].mean()
    acq_names = [t for t in acquired_targets if t in observed.index]
    if len(acq_names) < 3:
        fill = float(observed.mean()) if len(observed) else 0.0
        return {t: fill for t in candidate_targets}
    acq_ids = _target_ids(bundle, acq_names)
    candidate_ids = _target_ids(bundle, candidate_targets)
    model = Ridge(alpha=10.0).fit(
        bundle.target_features[acq_ids], observed.loc[acq_names].to_numpy(dtype=float)
    )
    pred = model.predict(bundle.target_features[candidate_ids])
    return {t: float(max(v, 0.0)) for t, v in zip(candidate_targets, pred)}


def _candidate_context(
    bundle: FeatureBundle,
    *,
    fold_train_idx: np.ndarray,
    acquired_targets: set[str],
    model: CROEnsemble,
    acquisition_seed: int,
    strategy: str,
) -> tuple[AcquisitionContext, dict[str, Any]]:
    train_targets = set(bundle.target_names[i] for i in np.unique(bundle.target_idx[fold_train_idx]))
    candidates = sorted(train_targets - acquired_targets)
    candidate_idx = _subset_rows(bundle, fold_train_idx, set(candidates))
    if not candidates or candidate_idx.size == 0:
        empty = np.zeros((0, bundle.target_features.shape[1]), dtype=np.float32)
        return (
            AcquisitionContext(
                candidates=(),
                features=empty,
                uncertainty={},
                effect={},
                discrepancy={},
                acquired_features=empty,
                rng=np.random.default_rng(acquisition_seed),
            ),
            {},
        )
    mu, _, epistemic = model.predict(bundle, candidate_idx)
    frame = pd.DataFrame(
        {
            "target": [bundle.target_names[i] for i in bundle.target_idx[candidate_idx]],
            "neuron": [bundle.responder_names[i] for i in bundle.responder_idx[candidate_idx]],
            "mu": mu,
            "epistemic": epistemic,
        }
    )
    grouped = frame.groupby("target").agg(
        uncertainty=("epistemic", "mean"),
        effect=("mu", lambda x: float(np.mean(np.abs(x)))),
        n_rows=("mu", "size"),
    )
    grouped = grouped.reindex(candidates).dropna(subset=["n_rows"])
    candidates = [t for t in candidates if t in grouped.index]
    if not candidates:
        empty = np.zeros((0, bundle.target_features.shape[1]), dtype=np.float32)
        return (
            AcquisitionContext(
                candidates=(),
                features=empty,
                uncertainty={},
                effect={},
                discrepancy={},
                acquired_features=empty,
                rng=np.random.default_rng(acquisition_seed),
            ),
            {},
        )
    all_train_targets = sorted(train_targets)
    all_ids = _target_ids(bundle, all_train_targets)
    scaler = StandardScaler().fit(bundle.target_features[all_ids])
    cand_ids = _target_ids(bundle, candidates)
    acq_ids = _target_ids(bundle, sorted(acquired_targets))
    features = scaler.transform(bundle.target_features[cand_ids]).astype(np.float32)
    acquired_features = (
        scaler.transform(bundle.target_features[acq_ids]).astype(np.float32)
        if acq_ids.size
        else np.zeros((0, features.shape[1]), dtype=np.float32)
    )
    acquired_idx = _subset_rows(bundle, fold_train_idx, acquired_targets)
    discrepancy = _discrepancy_scores(
        bundle,
        acquired_idx=acquired_idx,
        acquired_targets=acquired_targets,
        candidate_targets=candidates,
    )
    ctx = AcquisitionContext(
        candidates=tuple(candidates),
        features=features,
        uncertainty=grouped["uncertainty"].astype(float).to_dict(),
        effect=grouped["effect"].astype(float).to_dict(),
        discrepancy=discrepancy,
        acquired_features=acquired_features,
        rng=np.random.default_rng(acquisition_seed + (0 if strategy == "random" else 1)),
    )
    audit = {
        "n_candidates": len(candidates),
        "candidate_pool_hash": hashlib.sha256(
            json.dumps(candidates, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return ctx, audit


def _fit_ensemble(
    bundle: FeatureBundle,
    *,
    acquired_targets: set[str],
    fold_train_idx: np.ndarray,
    fold_val_idx: np.ndarray,
    config: ActiveConfig,
    model_seeds: tuple[int, ...],
) -> CROEnsemble:
    train_idx = _subset_rows(bundle, fold_train_idx, acquired_targets)
    if train_idx.size == 0:
        raise ValueError("no training rows for acquired targets")
    all_targets = set(bundle.target_names)
    held_out = all_targets - acquired_targets
    fits = []
    for seed in model_seeds:
        fits.append(
            fit_cro(
                bundle,
                train_idx=train_idx,
                val_idx=fold_val_idx,
                test_idx=fold_val_idx,
                held_out_targets=held_out,
                held_out_animals=set(),
                config=config.cro_config(seed),
            )
        )
    return CROEnsemble(fits=fits)


def _profile_metrics(
    frame: pd.DataFrame,
    *,
    strategy: str,
    reliability_detail: pd.DataFrame,
) -> tuple[float, float, int]:
    scored = frame.copy()
    scored["model"] = strategy
    profile = model_profile_correlation(scored, model=strategy)
    if profile.empty or reliability_detail.empty:
        return float("nan"), float("nan"), 0
    ceiling = reliability_detail.groupby("target")["attenuation_ceiling"].mean()
    profile = profile.merge(ceiling.rename("ceiling"), on="target", how="left")
    valid = profile["ceiling"] > 1e-6
    normalized = (
        float((profile.loc[valid, "profile_spearman"] / profile.loc[valid, "ceiling"]).mean())
        if valid.any()
        else float("nan")
    )
    return float(profile["profile_spearman"].mean()), normalized, int(len(profile))


def _evaluate(
    bundle: FeatureBundle,
    ensemble: CROEnsemble,
    *,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    strategy: str,
    fold: int,
    budget: float,
    n_targets: int,
    reliability_detail: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    test_mu, test_sigma, _ = ensemble.predict(bundle, test_idx)
    val_mu, val_sigma, _ = ensemble.predict(bundle, val_idx)
    q_raw = float(stats.t.ppf(0.95, df=4.0))
    coverage_raw = coverage(
        pd.DataFrame(
            {
                "y_true": bundle.y[test_idx],
                "lower": test_mu - q_raw * test_sigma,
                "upper": test_mu + q_raw * test_sigma,
            }
        )
    )
    q_cal = conformal_quantile(bundle.y[val_idx], val_mu, val_sigma, level=0.90)
    lower, upper = conformal_interval(test_mu, test_sigma, q_cal)
    frame = pd.DataFrame(
        {
            "split": "active",
            "fold": fold,
            "model": strategy,
            "strategy": strategy,
            "budget": budget,
            "event_id": bundle.event_ids[test_idx],
            "dataset_id": [bundle.animal_names[i] for i in bundle.animal_idx[test_idx]],
            "target": [bundle.target_names[i] for i in bundle.target_idx[test_idx]],
            "neuron": [bundle.responder_names[i] for i in bundle.responder_idx[test_idx]],
            "y_true": bundle.y[test_idx],
            "y_pred": test_mu,
            "lower": lower,
            "upper": upper,
        }
    )
    pm = pair_metrics(frame)
    em = event_metrics(frame)
    profile_spearman, profile_normalized, n_profile_targets = _profile_metrics(
        frame, strategy=strategy, reliability_detail=reliability_detail
    )
    row = {
        "fold": fold,
        "strategy": strategy,
        "budget": budget,
        "n_targets": n_targets,
        "n_events": len(frame),
        "n_pairs": pm["n_pairs"],
        "pair_spearman": pm["spearman"],
        "pair_mae": pm["mae"],
        "event_mae": em["mae"],
        "event_spearman": em["spearman"],
        "profile_spearman": profile_spearman,
        "profile_normalized": profile_normalized,
        "n_profile_targets": n_profile_targets,
        "coverage_raw": coverage_raw,
        "coverage_calibrated": coverage(frame),
        "interval_score": interval_score(frame),
        "conformal_q": q_cal,
        "seed": seed,
    }
    return row, frame


def _run_block(
    bundle: FeatureBundle,
    *,
    fold: Any,
    config: ActiveConfig,
    acquisition_seed: int,
    reliability_detail: pd.DataFrame,
    selection_log: list[dict[str, Any]],
    primary_frames: list[pd.DataFrame],
    metrics: list[dict[str, Any]],
    total_targets: int,
    target_counts: dict[str, int],
) -> None:
    train_targets = sorted(set(bundle.target_names[i] for i in np.unique(bundle.target_idx[fold.train_idx])))
    seed_n = max(1, int(round(config.initial_fraction * total_targets)))
    rng = np.random.default_rng(acquisition_seed + fold.fold * 10_000)
    seed_targets = _choose_seed_targets(train_targets, target_counts, seed_n, rng)
    seed_targets = sorted(set(seed_targets))
    for strategy in config.strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        acquired = set(seed_targets)
        current: CROEnsemble | None = None
        for budget in config.budgets:
            target_n = min(len(train_targets), max(len(seed_targets), int(round(budget * total_targets))))
            while len(acquired) < target_n:
                if current is None:
                    current = _fit_ensemble(
                        bundle,
                        acquired_targets=acquired,
                        fold_train_idx=fold.train_idx,
                        fold_val_idx=fold.val_idx,
                        config=config,
                        model_seeds=config.model_seeds,
                    )
                ctx, ctx_audit = _candidate_context(
                    bundle,
                    fold_train_idx=fold.train_idx,
                    acquired_targets=acquired,
                    model=current,
                    acquisition_seed=acquisition_seed,
                    strategy=strategy,
                )
                if not ctx.candidates:
                    break
                batch_n = min(config.batch_targets, target_n - len(acquired))
                batch = select_batch(ctx, strategy, batch_n)
                if not batch:
                    break
                selection_log.append(
                    {
                        "fold": fold.fold,
                        "acquisition_seed": acquisition_seed,
                        "strategy": strategy,
                        "round": len(selection_log),
                        "budget_target_n": target_n,
                        "selected": batch,
                        "candidate_pool_hash": ctx_audit.get("candidate_pool_hash"),
                        "n_candidates": ctx_audit.get("n_candidates"),
                        "uncertainty": {t: ctx.uncertainty.get(t) for t in batch},
                        "discrepancy": {t: ctx.discrepancy.get(t) for t in batch},
                        "effect": {t: ctx.effect.get(t) for t in batch},
                    }
                )
                acquired.update(batch)
                # Batch acquisition: the model is refit once per budget point,
                # not after every 8-target batch.  This is the preregistered
                # batching constraint and keeps the comparison computationally
                # honest across policies.
            current = _fit_ensemble(
                bundle,
                acquired_targets=acquired,
                fold_train_idx=fold.train_idx,
                fold_val_idx=fold.val_idx,
                config=config,
                model_seeds=config.model_seeds,
            )
            row, frame = _evaluate(
                bundle,
                current,
                test_idx=fold.test_idx,
                val_idx=fold.val_idx,
                strategy=strategy,
                fold=fold.fold,
                budget=budget,
                n_targets=len(acquired),
                reliability_detail=reliability_detail,
                seed=acquisition_seed,
            )
            row["acquisition_seed"] = acquisition_seed
            metrics.append(row)
            if math.isclose(budget, config.primary_budget, abs_tol=1e-9):
                primary_frames.append(frame)

    # Full-atlas reference: all outer-training targets acquired, 5 seeds.
    reference_n = min(len(train_targets), int(round(config.reference_budget * total_targets)))
    reference_targets = set(seed_targets)
    remaining = [t for t in train_targets if t not in reference_targets]
    if len(reference_targets) < reference_n:
        reference_targets.update(remaining[: reference_n - len(reference_targets)])
    reference = _fit_ensemble(
        bundle,
        acquired_targets=reference_targets,
        fold_train_idx=fold.train_idx,
        fold_val_idx=fold.val_idx,
        config=config,
        model_seeds=config.final_model_seeds,
    )
    row, _ = _evaluate(
        bundle,
        reference,
        test_idx=fold.test_idx,
        val_idx=fold.val_idx,
        strategy="full_atlas",
        fold=fold.fold,
        budget=config.reference_budget,
        n_targets=len(reference_targets),
        reliability_detail=reliability_detail,
        seed=acquisition_seed,
    )
    row["acquisition_seed"] = acquisition_seed
    metrics.append(row)


def run_active_benchmark(
    trials: pd.DataFrame,
    *,
    config: ActiveConfig,
    bundle: FeatureBundle | None = None,
) -> dict[str, Any]:
    bundle = bundle or build_feature_bundle(
        trials,
        connectome_path=config.connectome_path,
        response_column=config.response_column,
    )
    folds = build_folds(trials, config.outer_split, n_folds=config.n_folds, seed=17)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection_log: list[dict[str, Any]] = []
    primary_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    total_targets = len(bundle.target_names)
    target_counts = (
        pd.DataFrame({"target": trials["target"].astype(str), "dataset_id": trials["dataset_id"]})
        .drop_duplicates()
        .groupby("target")
        .size()
        .to_dict()
    )
    for fold in folds[: config.n_folds]:
        test_trials = trials.iloc[fold.test_idx]
        _, reliability_detail = split_half_target_profiles(
            test_trials,
            response_column=config.response_column,
            n_splits=10,
            min_events=2,
            seed=17 + fold.fold,
        )
        fold_bundle = bundle
        if config.use_state:
            if not config.state_path:
                raise ValueError("use_state=True requires state_path")
            cache = load_state_cache(resolve_path(config.state_path, PROJECT_ROOT))
            train_events = set(bundle.event_ids[fold.train_idx].astype(str))
            projector = fit_state_projector(cache, train_event_ids=train_events, n_components=16)
            fold_bundle = attach_state_features(bundle, projector, cache)
        for acquisition_seed in config.acquisition_seeds:
            _run_block(
                fold_bundle,
                fold=fold,
                config=config,
                acquisition_seed=acquisition_seed,
                reliability_detail=reliability_detail,
                selection_log=selection_log,
                primary_frames=primary_frames,
                metrics=metrics,
                total_targets=total_targets,
                target_counts=target_counts,
            )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(out / "metrics.csv", index=False)
    if primary_frames:
        pd.concat(primary_frames, ignore_index=True).to_parquet(
            out / "predictions.parquet", index=False
        )
    with (out / "selection_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in selection_log:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    # Explicit leakage audit: within a fold, no outer-test target may be
    # selected.  A target can be training in another fold, so the check must be
    # per fold rather than against the union of all test targets.
    leaks: list[dict[str, Any]] = []
    for fold in folds[: config.n_folds]:
        fold_test = set(trials.iloc[fold.test_idx]["target"].astype(str))
        fold_selected = {
            t
            for row in selection_log
            if int(row["fold"]) == int(fold.fold)
            for t in row["selected"]
        }
        overlap = sorted(fold_test & fold_selected)
        if overlap:
            leaks.append({"fold": int(fold.fold), "targets": overlap})
    selected = {t for row in selection_log for t in row["selected"]}
    def _hash_or_none(path_value: str) -> str | None:
        path = resolve_path(path_value, PROJECT_ROOT)
        return sha256_file(path) if path.exists() else None

    audit = {
        "config": asdict(config),
        "n_targets": total_targets,
        "n_folds": len(folds[: config.n_folds]),
        "n_selection_rows": len(selection_log),
        "n_selected_unique": len(selected),
        "fold_leaks": leaks,
        "no_test_target_selected": not leaks,
        "trials_sha256": _hash_or_none(config.trials_path),
        "connectome_sha256": _hash_or_none(config.connectome_path),
        "git": git_state(PROJECT_ROOT),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {
        "metrics": metrics_frame,
        "audit": audit,
        "selection_log": selection_log,
        "primary_frames": primary_frames,
    }
