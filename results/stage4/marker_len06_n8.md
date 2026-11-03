# Stage 4c: fine-tuned DNAFormer, marker_len06_n8 (CTGTCA x 8 per strand, one every 16 bases; 80 data bases, 37.5% overhead)

Colab T4, run started 2026-11-03 21:46. Locked Stage 4 settings, seed 1. Best epoch 9 (val NED 0.0027).
Test numbers read from Drive results/stage4/marker_len06_n8/results.csv.

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 5% | 97.4% [96.2, 98.2] | 0.0006 | 0.9991 | 23/40 |
| 8% | 69.6% [66.7, 72.4] | 0.0096 | 0.9866 | 0/40 |

Validation exact at 8% by epoch: 12.8 (pretrained), 35.2, 39.6, 56.8, 60.4, 63.6, 68.0, 67.2, 66.4, 68.4 (best), 66.0.
Data recovered at 8%: 696 x 80 = 55,680 data bases per 1,000 strands (+12.1% vs marker-free 49,664; break-even 62.1%).
Same overhead and data bases as marker_len12 (4 x 12 bases): 69.6% vs 62.3% exact at 8%.
