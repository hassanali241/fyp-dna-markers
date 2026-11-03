# Stage 4d: fine-tuned DNAFormer, marker_len06_n1 (one CTGTCA in the middle: 61 + 6 + 61; 122 data bases, 4.7% overhead)

Colab T4, run started 2026-11-03 21:57. Locked Stage 4 settings, seed 1. Best epoch 9 (val NED 0.0045).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.6% [99.0, 99.8] | 0.0001 | 0.9998 | 36/40 |
| 5% | 95.5% [94.0, 96.6] | 0.0011 | 0.9980 | 14/40 |
| 8% | 46.3% [43.2, 49.4] | 0.0165 | 0.9643 | 0/40 |

Validation exact at 8% by epoch: 9.2 (pretrained), 20.0, 23.2, 34.8, 36.4, 41.2, 42.8, 42.8, 45.2, 44.4 (best by val NED), 44.0.
Data recovered at 8%: 463 x 122 = 56,486 per 1,000 strands (+13.7% vs marker-free; break-even 40.7%).
