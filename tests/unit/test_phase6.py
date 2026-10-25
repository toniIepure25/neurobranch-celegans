from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.benchmark.connectivity import evaluate_connectivity
from neurobranch.benchmark.metrics import pair_metrics, paired_pair_bootstrap_delta
from neurobranch.benchmark.reliability import split_half_target_profiles
from neurobranch.benchmark.splits import build_folds
from neurobranch.causal.design import design_experiments
from neurobranch.causal.export import export_operator
from neurobranch.causal.features import build_feature_bundle
from neurobranch.causal.response_operator import CROConfig, train_cro


@pytest.fixture()
def synthetic_trials(tmp_path):
    connectome = tmp_path / "connectome.tsv"
    connectome.write_text(
        "pre\tpost\ttype\tsynapses\n"
        "T0\tR0\tchemical\t5\n"
        "T0\tR1\telectrical\t3\n"
        "T1\tR0\tchemical\t2\n"
        "T2\tR2\tchemical\t4\n"
        "T3\tR3\tchemical\t1\n",
        encoding="utf-8",
    )
    rows = []
    rng = np.random.default_rng(7)
    for animal in range(3):
        for event in range(2):
            for ti, target in enumerate(["T0", "T1", "T2", "T3"]):
                for ri, neuron in enumerate(["R0", "R1", "R2", "R3"]):
                    rows.append(
                        {
                            "dataset_id": f"a{animal}",
                            "event_id": f"a{animal}:{event}",
                            "target": target,
                            "neuron": neuron,
                            "response_mean": float(ti + ri + 0.1 * animal + 0.01 * event),
                            "pre_activity": float(rng.normal()),
                            "pre_sd": float(abs(rng.normal())),
                            "prev_stim_interval": float(event + 1),
                            "prev_same_target": float(event == 1),
                            "event_order": float(event),
                            "response": float(ti + ri),
                        }
                    )
    return pd.DataFrame(rows), connectome


def test_feature_bundle_excludes_post_stimulus_context(synthetic_trials):
    trials, connectome = synthetic_trials
    bundle = build_feature_bundle(trials, connectome_path=connectome)
    assert len(bundle) == len(trials)
    assert bundle.context_raw.shape[1] == 5
    assert bundle.target_features.shape[0] == 4
    assert bundle.responder_features.shape[0] == 4
    assert bundle.pair_features.shape[1] == 8
    # T0-R0 is a direct chemical edge; T0-R3 is not.
    mask = (bundle.target_idx == bundle.target_names.index("T0")) & (
        bundle.responder_idx == bundle.responder_names.index("R0")
    )
    assert bundle.direct_chem[mask].max() > 0
    mask3 = (bundle.target_idx == bundle.target_names.index("T0")) & (
        bundle.responder_idx == bundle.responder_names.index("R3")
    )
    assert bundle.direct_chem[mask3].max() == 0
    assert np.isfinite(bundle.pair_features).all()


def test_train_cro_handles_unseen_targets(synthetic_trials):
    trials, connectome = synthetic_trials
    bundle = build_feature_bundle(trials, connectome_path=connectome)
    t = bundle.target_names
    train_idx = np.flatnonzero(np.isin(bundle.target_idx, [t.index("T2"), t.index("T3")]))
    val_idx = np.flatnonzero(bundle.target_idx == t.index("T1"))
    test_idx = np.flatnonzero(bundle.target_idx == t.index("T0"))
    result = train_cro(
        bundle,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        held_out_targets={"T0", "T1"},
        held_out_animals=set(),
        config=CROConfig(rank=2, hidden=8, epochs=2, batch_size=32, patience=2, seed=3),
    )
    assert result["test_mu"].shape == (len(test_idx),)
    assert result["test_sigma"].shape == (len(test_idx),)
    assert np.isfinite(result["test_mu"]).all()
    assert (result["test_sigma"] > 0).all()


def test_grouped_split_has_no_target_overlap(synthetic_trials):
    trials, _ = synthetic_trials
    folds = build_folds(trials, "unseen_targets", n_folds=2, seed=11)
    assert folds
    for fold in folds:
        train_t = set(trials.iloc[fold.train_idx]["target"])
        test_t = set(trials.iloc[fold.test_idx]["target"])
        assert not (train_t & test_t)


def test_pair_metrics_and_paired_delta_are_finite(synthetic_trials):
    trials, _ = synthetic_trials
    frame = trials[["target", "neuron", "dataset_id", "response_mean"]].copy()
    frame = frame.rename(columns={"response_mean": "y_true"})
    frame["y_pred"] = frame["y_true"] + 0.1
    frame["y_pred_baseline"] = frame.groupby("target")["y_true"].transform("mean")
    metrics = pair_metrics(frame)
    assert metrics["spearman"] > 0.9
    delta = paired_pair_bootstrap_delta(frame, n_boot=20)
    assert np.isfinite(delta["delta"])


def test_reliability_and_design_emit_warning(synthetic_trials, tmp_path):
    trials, connectome = synthetic_trials
    summary, detail = split_half_target_profiles(
        trials, response_column="response_mean", n_splits=3, min_events=2
    )
    assert summary.n_targets >= 1
    assert not detail.empty
    bundle = build_feature_bundle(trials, connectome_path=connectome)
    pred = trials[["target", "neuron", "dataset_id", "event_id"]].copy()
    pred["model"] = "cro"
    pred["y_true"] = trials["response_mean"]
    pred["y_pred"] = trials["response_mean"] + 0.05
    pred["lower"] = pred["y_pred"] - 1
    pred["upper"] = pred["y_pred"] + 1
    design = design_experiments(pred, output_dir=tmp_path / "design", top_targets=2)
    assert "NOT CAUSALLY VALIDATED" in design["warning"]
    assert (tmp_path / "design" / "target_candidates.csv").exists()
    conn = evaluate_connectivity(bundle, pred, output_dir=tmp_path / "conn")
    assert conn["n_pairs"] > 0
    export_operator(
        bundle,
        pred,
        operator_path=tmp_path / "op.npz",
        uncertainty_path=tmp_path / "u.npz",
    )
    assert (tmp_path / "op.npz").exists()
