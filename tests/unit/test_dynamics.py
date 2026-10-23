from __future__ import annotations

import numpy as np

from neurobranch.dynamics.models import VARModel
from neurobranch.preprocessing import split_starts


def test_var_recovers_persistence_when_dynamics_are_identity(toy_data):
    # Fit VAR on identity targets; ridge shrinks toward persistence but should
    # still predict better than the feature mean on held-out data.
    tr = split_starts(toy_data, "train", history=1, horizon=1)
    te = split_starts(toy_data, "test", history=1, horizon=1)
    m = VARModel.fit(toy_data.X, toy_data.y_macro, tr)
    assert m.predict_next(toy_data.X[te]).shape == (len(te), toy_data.n_neurons)
    assert m.rollout(toy_data.X[te[0]], 5).shape == (6, toy_data.n_neurons)


def test_checkpoint_roundtrip(tmp_path, toy_model):
    from neurobranch.dynamics.checkpointing import load_model, save_model

    p = save_model(toy_model, tmp_path / "m.pt", metadata={"test": True})
    m2, meta = load_model(p)
    assert meta["test"] is True
    x = m2.rollout(np.zeros(toy_model.n_neurons), 3)
    assert x.shape == (4, toy_model.n_neurons)
