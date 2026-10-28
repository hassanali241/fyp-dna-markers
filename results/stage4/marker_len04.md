# Stage 4b: fine-tuned DNAFormer, marker_len04 (marker CAGT, 112 data bases, 12.5% overhead)

Colab T4, run started 2026-10-28 18:06. Locked Stage 4 settings. Best epoch 8 (val NED 0.0044).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.8% [99.3, 99.9] | 0.0000 | 1.0000 | 38/40 |
| 5% | 95.4% [93.9, 96.5] | 0.0011 | 0.9977 | 15/40 |
| 8% | 44.6% [41.5, 47.7] | 0.0183 | 0.9620 | 0/40 |

Validation exact at 8% by epoch: 10.4 (pretrained), 20.8, 22.0, 34.4, 37.6, 41.2, 42.4, 44.4, 47.6 (best), 46.4, 46.4.
