# Stage 4d: fine-tuned DNAFormer, marker_len06_n1 (one CTGTCA in the middle: 61 + 6 + 61; 122 data bases, 4.7% overhead)

Colab T4, run started 2026-11-17 20:41 (Haad), torch 2.14.1. Locked Stage 4 settings, seed 1. Best epoch 9 (val NED 0.0045).

**Official result (2026-11-17).** The first run (2026-11-03) reached 46.3% at 8%, but Google Drive kept only its
epoch-1 best.pt, so that model was lost (old folder kept as checkpoints/stage4/marker_len06_n1_old). This run
was retrained from scratch with identical settings; Drive was flushed before closing.

| deletion | exact [95% CI] | NED | base acc | files |
|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 |
| 3% | 99.6% [99.0, 99.8] | 0.0001 | 0.9998 | 36/40 |
| 5% | 95.1% [93.6, 96.3] | 0.0012 | 0.9978 | 11/40 |
| 8% | 45.8% [42.7, 48.9] | 0.0163 | 0.9639 | 0/40 |

Validation exact at 8% by epoch: 9.2 (pretrained), 19.2, 22.4, 34.0, 36.8, 43.2, 40.8, 44.8, 44.8, 44.4 (best by val NED), 44.0.
Data recovered at 8%: 458 x 122 = 55,876 per 1,000 strands (+12.5% vs marker-free; break-even 40.7%).

First run (superseded, model lost): best epoch 9; exact 100 / 99.6 / 95.5 / 46.3%.
