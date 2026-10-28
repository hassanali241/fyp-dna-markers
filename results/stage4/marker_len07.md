# Stage 4b: fine-tuned DNAFormer, marker_len07 (marker CTGATCA, 100 data bases, 21.9% overhead)

Colab T4, run started 2026-10-28 19:07. Locked Stage 4 settings. Best epoch 6 (val NED 0.0040).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.8% [99.3, 99.9] | 0.0000 | 1.0000 | 38/40 |
| 5% | 96.1% [94.7, 97.1] | 0.0011 | 0.9978 | 13/40 |
| 8% | 49.9% [46.8, 53.0] | 0.0168 | 0.9661 | 0/40 |

Validation exact at 8% by epoch: 9.6 (pretrained), 24.8, 27.2, 38.4, 44.4, 50.0, 50.8 (best by val NED), 50.8, 48.8, 50.8, 50.0.
