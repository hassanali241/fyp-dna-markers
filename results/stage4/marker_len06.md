# Stage 4b: fine-tuned DNAFormer, marker_len06 (marker CTGTCA, 104 data bases, 18.75% overhead)

Colab T4, run started 2026-10-28 (resumed once after a disconnect, from the end of epoch 1). Locked Stage 4 settings.
Best epoch 7 (val NED 0.0040).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.9% [99.4, 100.0] | 0.0000 | 1.0000 | 39/40 |
| 5% | 95.3% [93.8, 96.4] | 0.0011 | 0.9978 | 16/40 |
| 8% | 49.6% [46.5, 52.7] | 0.0172 | 0.9659 | 0/40 |

Validation exact at 8% by epoch (from epoch 2, after resume): 28.0, 41.2, 48.0, 48.0, 50.0, 49.2 (best by val NED), 49.6, 48.0, 48.0.
