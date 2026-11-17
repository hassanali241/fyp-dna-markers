# Stage 4c: fine-tuned DNAFormer, marker_len06_n2 (CTGTCA x 2 per strand, one every 64 bases; 116 data bases, 9.4% overhead)

Colab T4, locked Stage 4 settings, seed 1. Best epoch 7 (val NED 0.0048).

**Official result (2026-11-17).** The first run (2026-11-03, Mahek) finished training, but its final epochs and
test predictions never reached Drive (the last saved resume point was epoch 5). Haad resumed from epoch 5 with
the same settings, seed and optimizer state and retrained epochs 6-10 (torch 2.14.1). This run's `best.pt` and
`predictions.csv` are the ones in Drive and in all Stage 5 analysis. Its numbers are within the normal
run-to-run variation of the first run's log (8%: 43.7% now vs 44.7% then; seed repeats vary by ~1 point).

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.7% [99.1, 99.9] | 0.0001 | 0.9999 | 37/40 |
| 5% | 95.4% [93.9, 96.5] | 0.0009 | 0.9978 | 15/40 |
| 8% | 43.7% [40.7, 46.8] | 0.0179 | 0.9612 | 0/40 |

Validation exact at 8% by epoch: 8.8 (pretrained), 20.0, 23.2, 32.8, 37.6, 38.4 (epochs 1-5, first run),
42.8, 43.6 (best), 45.6, 44.8, 44.8 (epochs 6-10, resumed run).
Data recovered at 8%: 437 x 116 = 50,692 per 1,000 strands (+2.1% vs marker-free; break-even 42.8%).

First-run log (superseded, model not kept): best epoch 10; exact 100 / 99.7 / 95.1 / 44.7%.
