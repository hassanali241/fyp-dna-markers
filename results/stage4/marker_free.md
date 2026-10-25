# Stage 4: fine-tuned DNAFormer, marker_free (Colab T4, 25 Oct 2026, run started 16:52)

Settings: locked Stage 4 settings (see fyp_dna/stage4_finetune.py). Best epoch 9 (val NED 0.0052).
Full CSVs in Drive: `MyDrive/fyp_dna/results/stage4/marker_free/`; log in `checkpoints/stage4/marker_free/train_log.txt`.

## Test (1,000 clusters per level, payload-only)

| deletion | exact [95% CI] | NED | base acc | files | pretrained exact (Stage 3) | pretrained NED |
|---|---|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 | 100.0% | 0.0000 |
| 3% | 99.8% [99.3, 99.9] | 0.0000 | 0.9998 | 38/40 | 99.8% | 0.0000 |
| 5% | 94.1% [92.5, 95.4] | 0.0012 | 0.9977 | 9/40 | 88.5% | 0.0023 |
| 8% | 38.8% [35.8, 41.9] | 0.0202 | 0.9511 | 0/40 | 7.6% | 0.0693 |

## Validation per epoch (exact % at 5% / 8% deletion, overall NED)

| epoch | phase | del05 | del08 | val NED |
|---|---|---|---|---|
| 0 | pretrained | 90.0 | 6.0 | 0.0190 |
| 1 | A (frozen) | 91.2 | 20.0 | 0.0104 |
| 2 | A | 91.2 | 23.6 | 0.0097 |
| 3 | B (full) | 95.2 | 28.8 | 0.0069 |
| 4 | B | 94.4 | 34.8 | 0.0064 |
| 5 | B | 93.6 | 34.8 | 0.0059 |
| 6 | B | 94.8 | 35.6 | 0.0056 |
| 7 | B | 94.8 | 36.8 | 0.0054 |
| 8 | B | 94.8 | 38.0 | 0.0053 |
| 9 | B | 94.8 | 38.4 | 0.0052 (best) |
| 10 | B | 94.4 | 37.2 | 0.0054 |

Speed: 0.21 s/step frozen, ~0.30 s/step full; ~2.5-3.8 min/epoch; peak GPU 3.6 GB.
