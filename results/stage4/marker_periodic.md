# Stage 4: fine-tuned DNAFormer, marker_periodic (Colab T4, 25 Oct 2026, run started 17:02)

Settings: locked Stage 4 settings (identical to marker_free). Best epoch 10 (val NED 0.0041).
Full CSVs in Drive: `MyDrive/fyp_dna/results/stage4/marker_periodic/`; log in `checkpoints/stage4/marker_periodic/train_log.txt`.

## Test (1,000 clusters per level, payload-only, 96 data bases)

| deletion | exact [95% CI] | NED | base acc | files | pretrained exact (Stage 3) | pretrained NED |
|---|---|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 | 100.0% | 0.0000 |
| 3% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 | 100.0% | 0.0000 |
| 5% | 96.1% [94.7, 97.1] | 0.0009 | 0.9983 | 13/40 | 91.2% | 0.0023 |
| 8% | 52.4% [49.3, 55.5] | 0.0148 | 0.9736 | 0/40 | 7.9% | 0.0733 |

## Validation per epoch (exact % at 5% / 8% deletion, overall NED)

| epoch | phase | del05 | del08 | val NED |
|---|---|---|---|---|
| 0 | pretrained | 91.6 | 9.2 | 0.0191 |
| 1 | A (frozen) | 94.0 | 27.2 | 0.0091 |
| 2 | A | 93.2 | 30.8 | 0.0084 |
| 3 | B (full) | 94.8 | 42.4 | 0.0060 |
| 4 | B | 96.4 | 46.0 | 0.0051 |
| 5 | B | 96.8 | 48.4 | 0.0046 |
| 6 | B | 97.2 | 47.2 | 0.0043 |
| 7 | B | 97.2 | 50.8 | 0.0041 |
| 8 | B | 96.4 | 52.8 | 0.0042 |
| 9 | B | 96.4 | 52.4 | 0.0041 |
| 10 | B | 96.4 | 52.4 | 0.0041 (best) |

## First comparison with marker_free (test, 8% deletion)
- exact: 52.4% [49.3, 55.5] vs 38.8% [35.8, 41.9]: CIs do not overlap
- NED: 0.0148 vs 0.0202 (27% fewer base errors)
- data recovered per 1,000 strands: 524 x 96 = 50,304 bases vs 388 x 128 = 49,664 bases (+1.3%, i.e. about break-even)
- Synchronization vs. shorter payload can only be separated with the end-block control (pending).
