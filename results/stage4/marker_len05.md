# Stage 4b: fine-tuned DNAFormer, marker_len05 (marker TCATG, 108 data bases, 15.6% overhead)

Colab T4, run started 2026-10-28 18:25. Locked Stage 4 settings. Best epoch 10 (val NED 0.0045).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.5% [98.8, 99.8] | 0.0001 | 0.9999 | 35/40 |
| 5% | 95.6% [94.1, 96.7] | 0.0012 | 0.9977 | 12/40 |
| 8% | 45.7% [42.6, 48.8] | 0.0182 | 0.9625 | 0/40 |

Validation exact at 8% by epoch: 9.2 (pretrained), 23.6, 26.8, 38.8, 44.0, 44.8, 48.4, 50.0, 51.6, 50.0, 50.4 (best by val NED).
