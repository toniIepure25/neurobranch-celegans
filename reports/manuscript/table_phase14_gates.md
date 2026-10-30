Table 9. Phase 14 frozen gates. Eight of twelve failed.

| gate | requirement | observed | result |
| --- | --- | --- | --- |
| G1a_sbc_coverage | [0.93, 0.97] | 0.9673333333333334 | pass |
| G1b_sbc_rank | KS p > 0.01 | 3.347420875139048e-192 | fail |
| G1c_grid_edge | < 0.05 | 0.08100674492212144 | fail |
| G1d_fit_integrity | <= 0.05 | 0.0 | pass |
| G1e_one_event_stratum | [0.90, 0.98] with n >= 30 | 0.9669505962521294 | pass |
| G2_heldout_ppc | coverage >= 0.90 and MAE <= 0.10; must be run | {'coverage': 0.7697368421052632, 'median_abs_error': 0.39975175226159265, 'n': 304} | fail |
| G3_negative_control | median <= 0.05 and q95 <= 0.15 | {'median': 0.0, 'n': 100, 'q95': 0.35348801675532826} | fail |
| G3b_signflip_control | median <= 0.05 and q95 <= 0.15 | {'median': 0.0, 'n': 100, 'q95': 0.45086823641600177} | fail |
| G4_prior_sensitivity | range <= 0.05 | 0.12197708571916735 | fail |
| G5_event_variance_recovery | [0.90, 0.98] with n >= 100 | 0.9356666666666666 | pass |
| G6_reference_agreement | |R(22) - R_ref(V1)| <= 0.05 | 0.246279373116109 | fail |
| G7_spike_calibration | median P(z=0 | true z=0) >= 0.80 with n >= 100 | 0.35606217323737294 | fail |
