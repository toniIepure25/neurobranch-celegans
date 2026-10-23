from __future__ import annotations

import numpy as np
import pytest

from neurobranch.config import SplitConfig
from neurobranch.data.kato import WormRecording
from neurobranch.dynamics.models import BehaviorProbe, MLPDynamics
from neurobranch.dynamics.torch_utils import MLPNet
from neurobranch.preprocessing import make_worm_data


@pytest.fixture(scope="session")
def toy_recording() -> WormRecording:
    rng = np.random.default_rng(3)
    T, N = 600, 8
    names = ("AVEL", "AVER", "RIML", "RIMR", "AIBL", "AIBR", "VA01", "VB01")
    # Slowly varying latent common mode plus independent noise.
    drift = np.cumsum(rng.normal(scale=0.03, size=(T, 1)), axis=0)
    x = drift + rng.normal(scale=0.5, size=(T, N))
    states = []
    behavior = ["fwd"]
    for t in range(T):
        if t % 110 == 0 and t:
            behavior = ["rev1"] if behavior[0] != "rev1" else ["dt"]
        states.append(behavior[0])
    return WormRecording(
        worm_id=0,
        dataset_name="toy",
        neurons=names,
        traces=x.astype(np.float32),
        raw_traces=x.astype(np.float32),
        states=tuple(states),
        fps=3.0,
        time_seconds=np.arange(T) / 3.0,
        source_file="toy",
        source_sha256="0" * 64,
    )


@pytest.fixture(scope="session")
def toy_data(toy_recording):
    cfg = SplitConfig(mode="blocked", n_blocks=5, test_block=2, val_block=4)
    return make_worm_data(toy_recording, cfg, exclude_behavior=True)


@pytest.fixture(scope="session")
def toy_model(toy_data):
    from neurobranch.preprocessing import split_starts

    tr = split_starts(toy_data, "train", history=1, horizon=1)
    probe = BehaviorProbe().fit(toy_data.X[tr], toy_data.y_macro[tr])
    net = MLPNet(toy_data.n_neurons, 32, toy_data.n_neurons, 4, dropout=0.0, residual=True)
    return MLPDynamics(net=net, n_neurons=toy_data.n_neurons, history_len=1, probe=probe, device="cpu")
