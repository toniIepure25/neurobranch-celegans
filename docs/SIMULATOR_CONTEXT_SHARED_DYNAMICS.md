# Shared latent dynamics across sessions

## 1. Panel

Only **alphabetic, explicitly named neurons present in all five WT_NoStim recordings** are used. Numeric trace IDs are worm-specific and are never treated as common identities. The panel is recorded in `shared_dynamics_summary.json`.

This avoids the invalid operation of concatenating worm-specific numeric columns. It still does not make the five sessions five independent animals.

## 2. Model

- A shared latent transition `A` is fitted on pooled latent transitions from four sessions.
- The held-out session has its own observation map `D` and mean, fitted on a **calibration prefix of its training block**.
- `R` maps the held-out PCA coordinate into the shared coordinate.
  - `R=I` is zero-shot.
  - `R` is fitted on calibration transitions by least squares with a ridge toward identity; this is calibration adaptation.
- `local_calibration` fits a per-session PCA+LDS on exactly the same calibration prefix.
- `local_full_train_reference` fits the per-session model on the full training block; it is a reference, not a fair calibration baseline.

No test frame is used for PCA, `R`, support calibration, or model selection.

## 3. Identifiability caveat

PCA bases are not identifiable across sessions. Even after a deterministic sign convention, the shared `A` lives in a model coordinate, not a biological common coordinate. The calibrated `R` can partially compensate for rotation/scale but cannot create a shared biological basis. A positive result would therefore be model-level transfer evidence, not evidence of a conserved circuit coordinate.

## 4. Results

| Model | Horizon | Mean skill | Median skill |
|---|---:|---:|---:|
| shared zero-shot | 20 | −2.822 | −1.598 |
| shared zero-shot | 100 | −0.980 | −0.341 |
| shared calibrated | 20 | −5.174 | −2.451 |
| shared calibrated | 100 | −7.068 | −7.860 |
| local, same calibration | 20 | −3.818 | −1.972 |
| local, same calibration | 100 | −1.618 | −0.570 |
| local, full training reference | 20 | −1.924 | −0.857 |
| local, full training reference | 100 | −0.515 | −0.139 |

Zero-shot shared dynamics was less bad than the local model fitted on the same short calibration prefix, but still failed badly against persistence. Calibration adaptation was harmful, especially at H100, and several calibrated rollouts fell outside the q95 support envelope. There is no evidence of useful shared dynamics.

## 5. Reproduction

```powershell
python scripts/context_shared_dynamics.py
```

The command is session-disjoint but not animal-disjoint. The result must not be described as cross-animal generalization.
