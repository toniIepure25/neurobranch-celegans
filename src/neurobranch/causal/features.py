"""Feature construction for the Causal Response Operator.

Two design choices matter for the unseen-target test:

1. Target and responder latent factors are partly produced by an MLP over
   anatomical features, not only by lookup embeddings.  A held-out target has no
   trained embedding, but it does have a connectome row.
2. The connectome SVD is unsupervised and fitted on the connectome only.  It
   never sees a calcium response, so it cannot leak the test endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from neurobranch.config import PROJECT_ROOT, resolve_path
from neurobranch.data.randi import canonical_neuron_name

DEFAULT_CONNECTOME = "references/external/aconnectome_witvliet_2020_8_utf8.csv"
CONTEXT_COLUMNS = (
    "pre_activity",
    "pre_sd",
    "log_prev_interval",
    "prev_same_target",
    "event_order",
)


@dataclass
class FeatureBundle:
    target_names: list[str]
    responder_names: list[str]
    animal_names: list[str]
    target_idx: np.ndarray
    responder_idx: np.ndarray
    animal_idx: np.ndarray
    target_features: np.ndarray
    responder_features: np.ndarray
    pair_features: np.ndarray
    direct_chem: np.ndarray
    direct_elec: np.ndarray
    twohop_chem: np.ndarray
    twohop_elec: np.ndarray
    pair_known: np.ndarray
    context_raw: np.ndarray
    y: np.ndarray
    event_ids: np.ndarray
    target_seen_possible: np.ndarray
    connectome_coverage: dict[str, int]

    def __len__(self) -> int:
        return int(len(self.y))


def load_connectome_adjacency(
    path: str | Path | None = None,
    *,
    targets: list[str],
    responders: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    p = resolve_path(path or DEFAULT_CONNECTOME, PROJECT_ROOT)
    if not p.exists():
        raise FileNotFoundError(f"connectome not found: {p}")
    table = pd.read_csv(p, sep="\t")
    for col in ("pre", "post"):
        if col not in table.columns:
            raise ValueError(f"connectome missing {col!r}")
        table[col] = table[col].map(canonical_neuron_name)
    table = table.dropna(subset=["pre", "post"])
    nodes = sorted(set(targets) | set(responders) | set(table["pre"]) | set(table["post"]))
    node_idx = {n: i for i, n in enumerate(nodes)}
    chem = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    elec = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for row in table.itertuples(index=False):
        i = node_idx[str(row.pre)]
        j = node_idx[str(row.post)]
        value = float(getattr(row, "synapses", 1.0))
        if str(row.type) == "chemical":
            chem[i, j] += value
        elif str(row.type) == "electrical":
            elec[i, j] += value
    return chem, elec, nodes, targets + [n for n in responders if n not in targets]


def _degree_features(matrix: np.ndarray, axis: int) -> np.ndarray:
    degree = matrix.sum(axis=axis)
    return np.log1p(degree).astype(np.float32)


def _svd_features(chem: np.ndarray, elec: np.ndarray, *, n_components: int = 32) -> np.ndarray:
    combined = np.concatenate([np.log1p(chem), np.log1p(elec)], axis=1)
    n_components = min(n_components, min(combined.shape) - 1)
    if n_components < 1:
        return np.zeros((combined.shape[0], 1), dtype=np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=17)
    return svd.fit_transform(combined).astype(np.float32)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True) + 1.0
    return matrix / denom


def build_feature_bundle(
    trials: pd.DataFrame,
    *,
    connectome_path: str | Path | None = None,
    response_column: str = "response_mean",
) -> FeatureBundle:
    required = {"dataset_id", "target", "neuron", response_column}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"trial table missing {sorted(missing)}")
    data = trials.dropna(subset=["target", "neuron", response_column]).copy()
    data = data[np.isfinite(data[response_column].to_numpy(dtype=float))]
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("no finite response rows")

    target_names = sorted(data["target"].astype(str).unique())
    responder_names = sorted(data["neuron"].astype(str).unique())
    animal_names = sorted(data["dataset_id"].astype(str).unique())
    target_map = {n: i for i, n in enumerate(target_names)}
    responder_map = {n: i for i, n in enumerate(responder_names)}
    animal_map = {n: i for i, n in enumerate(animal_names)}

    target_idx = data["target"].astype(str).map(target_map).to_numpy(dtype=np.int64)
    responder_idx = data["neuron"].astype(str).map(responder_map).to_numpy(dtype=np.int64)
    animal_idx = data["dataset_id"].astype(str).map(animal_map).to_numpy(dtype=np.int64)

    chem, elec, nodes, _ = load_connectome_adjacency(
        connectome_path, targets=target_names, responders=responder_names
    )
    node_map = {n: i for i, n in enumerate(nodes)}
    t_nodes = np.array([node_map.get(n, -1) for n in target_names])
    r_nodes = np.array([node_map.get(n, -1) for n in responder_names])
    node_svd = _svd_features(chem, elec)

    chem_deg = _degree_features(chem, axis=1)
    elec_deg = _degree_features(elec, axis=1)

    def node_features(indices: np.ndarray) -> np.ndarray:
        present = indices >= 0
        out = np.zeros((len(indices), node_svd.shape[1] + 4), dtype=np.float32)
        if present.any():
            out[present, : node_svd.shape[1]] = node_svd[indices[present]]
            out[present, -4] = chem_deg[indices[present]]
            out[present, -3] = elec_deg[indices[present]]
            out[present, -2] = 1.0
        out[:, -1] = (~present).astype(np.float32)
        return out

    target_features = node_features(t_nodes)
    responder_features = node_features(r_nodes)

    # Pair features: direct chemical/electrical, two-hop counts and degrees.
    c = np.log1p(chem)
    e = np.log1p(elec)
    c2 = _normalise_rows(c @ c)
    e2 = _normalise_rows(e @ e)
    pair_feature_dim = 8
    pair_features = np.zeros((len(data), pair_feature_dim), dtype=np.float32)
    t_rows = t_nodes[target_idx]
    r_rows = r_nodes[responder_idx]
    t_safe = np.clip(t_rows, 0, len(nodes) - 1)
    r_safe = np.clip(r_rows, 0, len(nodes) - 1)
    pair_known = (t_rows >= 0) & (r_rows >= 0)
    direct_chem = c[t_safe, r_safe].astype(np.float32)
    direct_elec = e[t_safe, r_safe].astype(np.float32)
    twohop_chem = c2[t_safe, r_safe].astype(np.float32)
    twohop_elec = e2[t_safe, r_safe].astype(np.float32)
    pair_features[:, 0] = direct_chem
    pair_features[:, 1] = direct_elec
    pair_features[:, 2] = twohop_chem
    pair_features[:, 3] = twohop_elec
    pair_features[:, 4] = chem_deg[t_safe]
    pair_features[:, 5] = elec_deg[t_safe]
    pair_features[:, 6] = chem_deg[r_safe]
    pair_features[:, 7] = elec_deg[r_safe]
    # Missing connectome entries are not silently treated as zero evidence.
    pair_features *= pair_known.astype(np.float32)[:, None]

    context = pd.DataFrame(index=data.index)
    context["pre_activity"] = data.get("pre_activity", pd.Series(np.nan, index=data.index))
    context["pre_sd"] = data.get("pre_sd", pd.Series(np.nan, index=data.index))
    # ``global_delta`` is measured *after* stimulation and is therefore not a
    # deployable pre-stimulus covariate.  It is deliberately excluded; keeping
    # it would leak the population response into a "prediction" of one neuron.
    context["log_prev_interval"] = np.log1p(
        data.get("prev_stim_interval", pd.Series(np.nan, index=data.index)).clip(lower=0)
    )
    context["prev_same_target"] = data.get("prev_same_target", pd.Series(0.0, index=data.index)).astype(float)
    context["event_order"] = data.get("event_order", pd.Series(0.0, index=data.index)).astype(float)
    context_raw = context[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float32)

    coverage = {
        "n_targets": len(target_names),
        "n_responders": len(responder_names),
        "n_animals": len(animal_names),
        "targets_in_connectome": int((t_nodes >= 0).sum()),
        "responders_in_connectome": int((r_nodes >= 0).sum()),
        "rows_with_pair_connectome": int(((t_nodes[target_idx] >= 0) & (r_nodes[responder_idx] >= 0)).sum()),
    }

    return FeatureBundle(
        target_names=target_names,
        responder_names=responder_names,
        animal_names=animal_names,
        target_idx=target_idx,
        responder_idx=responder_idx,
        animal_idx=animal_idx,
        target_features=target_features,
        responder_features=responder_features,
        pair_features=pair_features,
        direct_chem=direct_chem,
        direct_elec=direct_elec,
        twohop_chem=twohop_chem,
        twohop_elec=twohop_elec,
        pair_known=pair_known,
        context_raw=context_raw,
        y=data[response_column].to_numpy(dtype=np.float32),
        event_ids=data["event_id"].astype(str).to_numpy(),
        target_seen_possible=np.ones(len(data), dtype=bool),
        connectome_coverage=coverage,
    )
