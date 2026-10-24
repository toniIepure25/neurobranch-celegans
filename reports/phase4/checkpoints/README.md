# Phase 4 model checkpoint note

No trained checkpoint is committed.  The primary benchmark fits a separate StandardScaler + logistic regression for every left-out animal; a single checkpoint would either leak training animals or misrepresent the animal-disjoint protocol.  The exact model specification, seeds, folds and per-event predictions are in this directory and `reports/phase4/turn_direction_predictions.parquet`.
