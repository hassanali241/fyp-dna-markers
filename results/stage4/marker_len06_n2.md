# Stage 4c: fine-tuned DNAFormer, marker_len06_n2 (CTGTCA x 2 per strand, one every 64 bases; 116 data bases, 9.4% overhead)

Colab T4, run started 2026-11-03 21:45. Locked Stage 4 settings, seed 1. Best epoch 10 (val NED 0.0048).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.7% [99.1, 99.9] | 0.0001 | 0.9999 | 37/40 |
| 5% | 95.1% [93.6, 96.3] | 0.0010 | 0.9978 | 13/40 |
| 8% | 44.7% [41.6, 47.8] | 0.0177 | 0.9621 | 0/40 |

Validation exact at 8% by epoch: 8.8 (pretrained), 20.0, 23.2, 32.8, 37.6, 38.4, 42.4, 42.8, 44.4, 44.8, 45.6 (best).
Data recovered at 8%: 447 x 116 = 51,852 per 1,000 strands (+4.4% vs marker-free; break-even 42.8%).
