from __future__ import annotations

import json

import numpy as np
import pytest

from neurobranch.config import sha256_file
from neurobranch.interventions.safety import require_empirical_validation
from neurobranch.interventions.spec import InterventionSpec, make_perturbation


def test_behavior_label_neuron_is_refused(toy_data):
    spec = InterventionSpec(name="bad", targets=("AVAL",))
    with pytest.raises(ValueError, match="behaviour-label"):
        make_perturbation(toy_data, spec)


def test_unknown_neuron_is_refused(toy_data):
    spec = InterventionSpec(name="bad", targets=("NOTACELL",))
    with pytest.raises(KeyError):
        make_perturbation(toy_data, spec)


def test_perturbation_window(toy_data):
    spec = InterventionSpec(name="ok", targets=("RIML",), amplitude=2.0, onset=2, duration=3)
    fn, inds = make_perturbation(toy_data, spec)
    x = np.zeros(toy_data.n_neurons)
    assert np.allclose(fn(x, 1), x)
    assert fn(x, 2)[inds[0]] == pytest.approx(2.0)
    assert np.allclose(fn(x, 5), x)


def test_safety_gate_rejects_incomplete_manifest(tmp_path):
    p = tmp_path / "evidence.json"
    p.write_text(
        json.dumps(
            {
                "dataset_path": str(tmp_path / "missing.mat"),
                "dataset_sha256": "0" * 64,
                "intervention": "none",
                "target_neurons": [],
                "n_intervention_animals": 1,
                "n_control_animals": 1,
                "provenance_url": "",
                "performed_in_living_animal": False,
                "negative_control": False,
                "preregistered_analysis": False,
            }
        ),
        encoding="utf-8",
    )
    status = require_empirical_validation(p)
    assert not status.validated
    assert status.level == "empirical_rejected"
    assert len(status.reasons) >= 5


def test_empirical_gate_accepts_complete_manifest(tmp_path):
    data = tmp_path / "fake.mat"
    data.write_bytes(b"not really data")
    p = tmp_path / "evidence.json"
    p.write_text(
        json.dumps(
            {
                "dataset_path": str(data),
                "dataset_sha256": sha256_file(data),
                "intervention": "histamine silencing",
                "target_neurons": ["RIML"],
                "n_intervention_animals": 4,
                "n_control_animals": 4,
                "provenance_url": "https://example.org/dataset",
                "performed_in_living_animal": True,
                "negative_control": True,
                "preregistered_analysis": True,
            }
        ),
        encoding="utf-8",
    )
    status = require_empirical_validation(p)
    assert status.validated
