"""Compact switching VAR for short, noisy neural trajectories.

The model is a sticky Gaussian HMM whose emission mean is a first-order VAR:

    r_t ~ Categorical(P[r_{t-1}])
    x_t ~ N(A_{r_t} x_{t-1} + b_{r_t}, diag(v_{r_t}))

Sequences are formed from contiguous training runs so that a transition is
never invented across a purged split boundary.  EM is initialized from a small
KMeans partition of one-step changes and restarted several times; the fit with
the best training log-likelihood is retained.  This is deliberately a compact
model, not a claim that the worm has discrete hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from neurobranch.dynamics.models import Array, BehaviorProbe, PerturbFn, _initial_history

_LOG_2PI = float(np.log(2.0 * np.pi))


def _contiguous_runs(indices: Array) -> list[Array]:
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    return [part for part in np.split(idx, breaks) if len(part) >= 2]


@dataclass
class SwitchingVARModel:
    n_neurons: int
    K: int
    A: Array
    b: Array
    variances: Array
    P: Array
    pi: Array
    probe: BehaviorProbe
    ridge: float = 1e-2
    log_likelihood: float = float("-inf")
    n_iter: int = 0
    name: str = "switching_var"
    history_len: int = 1
    native_stochastic: bool = True
    diagonal: bool = True
    diagnostics: dict = field(default_factory=dict)

    @property
    def stationary(self) -> Array:
        vals, vecs = np.linalg.eig(self.P.T)
        i = int(np.argmin(np.abs(vals - 1.0)))
        v = np.real(vecs[:, i])
        v = np.maximum(v, 1e-12)
        return v / v.sum()

    @property
    def occupancy(self) -> Array:
        return self.stationary

    def _emission_logprob(self, H: Array, Y: Array) -> Array:
        out = np.empty((len(H), self.K), dtype=np.float64)
        for k in range(self.K):
            mean = H @ self.A[k].T + self.b[k]
            var = np.maximum(self.variances[k], 1e-6)
            out[:, k] = -0.5 * (
                self.n_neurons * _LOG_2PI
                + np.log(var).sum()
                + (((Y - mean) ** 2) / var).sum(axis=1)
            )
        return out

    def regime_proba(self, history: Array) -> Array:
        h = np.asarray(history, dtype=np.float64)
        if h.ndim == 1:
            h = h[None]
        # One-step posterior from the stationary prior.
        logp = np.log(np.maximum(self.stationary, 1e-12))[None, :]
        for k in range(self.K):
            mean = h @ self.A[k].T + self.b[k]
            var = np.maximum(self.variances[k], 1e-6)
            logp[:, k] += -0.5 * self.n_neurons * _LOG_2PI - 0.5 * np.log(var).sum()
            logp[:, k] += -0.5 * (((h - mean) ** 2) / var).sum(axis=1)
        return np.exp(logp - logsumexp(logp, axis=1, keepdims=True))

    def predict_next(self, history: Array) -> Array:
        h = np.asarray(history, dtype=np.float64)
        if h.ndim == 2:
            h = h[:, None, :]
        h = h[:, -1]
        w = self.stationary[None, :]
        means = np.stack([h @ self.A[k].T + self.b[k] for k in range(self.K)], axis=1)
        return (means * w[:, :, None]).sum(axis=1)

    def behavior_proba(self, history: Array) -> Array:
        arr = np.asarray(history)
        states = arr[:, -1] if arr.ndim == 3 else (arr if arr.ndim == 2 else arr[None])
        return self.probe.predict_proba(states)

    def rollout(
        self,
        x0: Array,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb: PerturbFn | None = None,
        initial_history: Array | None = None,
        stochastic: bool = True,
    ) -> Array:
        rng = np.random.default_rng(seed)
        hist = _initial_history(x0, initial_history, self.n_neurons)
        seq = [hist[-1].copy()]
        x = hist[-1].copy()
        state = int(rng.choice(self.K, p=self.stationary))
        for s in range(steps):
            if stochastic:
                mean = self.A[state] @ x + self.b[state]
                sd = np.sqrt(np.maximum(self.variances[state], 1e-8))
                x = mean + rng.normal(scale=sd, size=self.n_neurons)
                state = int(rng.choice(self.K, p=self.P[state]))
            else:
                weights = self.stationary
                mean = sum(
                    weights[k] * (self.A[k] @ x + self.b[k]) for k in range(self.K)
                )
                x = np.asarray(mean)
            if noise_std > 0:
                x = x + rng.normal(scale=noise_std, size=x.shape)
            if perturb is not None:
                x = perturb(x, s)
            seq.append(x.copy())
        return np.stack(seq)

    @classmethod
    def fit(
        cls,
        X: Array,
        y: Array,
        starts: Array,
        *,
        K: int = 2,
        ridge: float = 1e-2,
        seed: int = 17,
        restarts: int = 5,
        n_iter: int = 40,
        tol: float = 1e-5,
        spectral_cap: float = 0.98,
    ) -> SwitchingVARModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # Use the actual train starts; a frame is included only when both the
        # current and next frame have a train start.
        starts = np.asarray(sorted(set(int(t) for t in starts)), dtype=np.int64)
        runs = _contiguous_runs(starts)
        if not runs:
            raise ValueError("no contiguous training runs for switching VAR")
        H = np.concatenate([X[r] for r in runs], axis=0)
        Y = np.concatenate([X[r + 1] for r in runs], axis=0)
        seq_id = np.concatenate([np.full(len(r), i) for i, r in enumerate(runs)])

        best: SwitchingVARModel | None = None
        for restart in range(restarts):
            model = cls._fit_once(
                H, Y, seq_id, K=K, ridge=ridge, seed=seed + restart, n_iter=n_iter, tol=tol,
                spectral_cap=spectral_cap,
            )
            if best is None or model.log_likelihood > best.log_likelihood:
                best = model
        assert best is not None
        best.probe = BehaviorProbe(seed=seed).fit(X[starts], y[starts])
        best.diagnostics = {
            "n_sequences": len(runs),
            "n_transitions": len(H),
            "occupancy": best.stationary.tolist(),
            "min_occupancy": float(best.stationary.min()),
            "restarts": restarts,
            "training_log_likelihood": best.log_likelihood,
            "diagonal_dynamics": best.diagonal,
        }
        return best

    @classmethod
    def _fit_once(
        cls,
        H: Array,
        Y: Array,
        seq_id: Array,
        *,
        K: int,
        ridge: float,
        seed: int,
        n_iter: int,
        tol: float,
        spectral_cap: float = 0.98,
    ) -> SwitchingVARModel:
        n, d = H.shape
        # Weighted regression with an unregularized intercept.
        def m_step(gamma: Array, xi: Array | None = None) -> tuple[Array, Array, Array, Array, Array]:
            A = np.empty((K, d, d))
            b = np.empty((K, d))
            var = np.empty((K, d))
            for k in range(K):
                w = np.maximum(gamma[:, k], 1e-8)
                sw = w.sum()
                Hw = H * np.sqrt(w)[:, None]
                Yw = Y * np.sqrt(w)[:, None]
                Hc = Hw - (w[:, None] * H).sum(0, keepdims=True) / sw
                Yc = Yw - (w[:, None] * Y).sum(0, keepdims=True) / sw
                Ak = np.linalg.solve(Hc.T @ Hc + ridge * np.eye(d), Hc.T @ Yc)
                Ak = np.diag(np.diag(Ak))
                rho = float(np.max(np.abs(np.diag(Ak)))) if d else 0.0
                if rho > spectral_cap:
                    Ak = Ak * (spectral_cap / rho)
                bk = (w[:, None] * (Y - H @ Ak.T)).sum(0) / sw
                resid = Y - (H @ Ak.T + bk)
                vk = (w[:, None] * resid**2).sum(0) / sw
                A[k], b[k], var[k] = Ak, bk, np.maximum(vk, 1e-5)
            if xi is None:
                P = np.full((K, K), 1.0 / K)
            else:
                P = xi + 0.5
                P = P / P.sum(axis=1, keepdims=True)
            pi = np.maximum(gamma[0], 1e-8)
            pi = pi / pi.sum()
            return A, b, var, P, pi

        # KMeans initialization on one-step changes.
        changes = Y - H
        labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(changes)
        gamma = np.zeros((n, K))
        gamma[np.arange(n), labels] = 1.0
        A, b, var, P, pi = m_step(gamma)

        prev_ll = float("-inf")
        ll = float("-inf")
        for _it in range(n_iter):
            model = cls(d, K, A, b, var, P, pi, probe=BehaviorProbe())
            log_emit = model._emission_logprob(H, Y)
            log_alpha = np.empty((n, K))
            log_beta = np.zeros((n, K))
            log_alpha[0] = np.log(np.maximum(pi, 1e-12)) + log_emit[0]
            for t in range(1, n):
                # Reset to the prior at the start of each contiguous sequence.
                prior = np.log(np.maximum(pi, 1e-12)) if seq_id[t] != seq_id[t - 1] else np.log(
                    np.maximum(P.T, 1e-12)
                )
                if seq_id[t] != seq_id[t - 1]:
                    log_alpha[t] = prior + log_emit[t]
                else:
                    log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + prior, axis=0) + log_emit[t]
            for t in range(n - 2, -1, -1):
                if seq_id[t] != seq_id[t + 1]:
                    log_beta[t] = 0.0
                else:
                    log_beta[t] = logsumexp(
                        np.log(np.maximum(P, 1e-12)) + log_emit[t + 1] + log_beta[t + 1], axis=1
                    )
            log_gamma = log_alpha + log_beta
            ends = np.flatnonzero(np.r_[seq_id[1:] != seq_id[:-1], True])
            ll = float(sum(logsumexp(log_alpha[e]) for e in ends))
            gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))

            xi = np.zeros((K, K))
            for t in range(n - 1):
                if seq_id[t] != seq_id[t + 1]:
                    continue
                log_xi = (
                    log_alpha[t][:, None]
                    + np.log(np.maximum(P, 1e-12))
                    + log_emit[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                )
                xi += np.exp(log_xi - logsumexp(log_xi))
            A, b, var, P, pi = m_step(gamma, xi)
            if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        model = cls(d, K, A, b, var, P, pi, probe=BehaviorProbe(), log_likelihood=ll, n_iter=_it + 1)
        return model
