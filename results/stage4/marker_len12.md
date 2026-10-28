# Stage 4b: fine-tuned DNAFormer, marker_len12 (marker ACGTCAGTGCAT, 80 data bases, 37.5% overhead)

Colab T4, run started 2026-10-28 19:21. Locked Stage 4 settings. Best epoch 9 (val NED 0.0039).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.8% [99.3, 99.9] | 0.0001 | 0.9999 | 38/40 |
| 5% | 97.1% [95.9, 98.0] | 0.0008 | 0.9986 | 15/40 |
| 8% | 62.3% [59.3, 65.3] | 0.0119 | 0.9806 | 0/40 |

Validation exact at 8% by epoch: 13.2 (pretrained), 26.8, 30.0, 46.0, 52.8, 55.6, 57.6, 58.0, 57.2, 60.0 (best), 60.0.
