# Stage 3 results: baselines, no training (Colab T4, 25 Oct 2026)

All 1,000 test clusters per deletion level, payload-only scoring. Full CSVs are in
Google Drive: `MyDrive/fyp_dna/results/stage3/` (results.csv, predictions.csv).

Exact recovery [95% Wilson CI], normalized edit distance (NED), files = 25-strand files fully recovered (of 40).

| method | arm | 1% | 3% | 5% | 8% |
|---|---|---|---|---|---|
| mv | marker_free | 0.6% | 0.0% | 0.0% | 0.0% |
| mv | marker_periodic | 1.4% | 0.0% | 0.0% | 0.0% |
| mv | marker_endblock | 3.6% | 0.0% | 0.0% | 0.0% |
| mv_sync | marker_periodic | 53.4% [50.3, 56.5] | 3.1% | 0.0% | 0.0% |
| dnaformer | marker_free | 100.0% | 99.8% | 88.5% [86.4, 90.3] | 7.6% [6.1, 9.4] |
| dnaformer | marker_periodic | 100.0% | 100.0% | 91.2% [89.3, 92.8] | 7.9% [6.4, 9.7] |
| dnaformer | marker_endblock | 100.0% | 99.9% | 89.3% [87.2, 91.1] | 7.5% [6.0, 9.3] |

NED (lower is better):

| method | arm | 1% | 3% | 5% | 8% |
|---|---|---|---|---|---|
| mv | marker_free | 0.1284 | 0.2074 | 0.2584 | 0.3140 |
| mv | marker_periodic | 0.1263 | 0.2181 | 0.2816 | 0.3491 |
| mv | marker_endblock | 0.1040 | 0.1865 | 0.2402 | 0.2972 |
| mv_sync | marker_periodic | 0.0115 | 0.0620 | 0.1197 | 0.1903 |
| dnaformer | marker_free | 0.0000 | 0.0000 | 0.0023 | 0.0693 |
| dnaformer | marker_periodic | 0.0000 | 0.0000 | 0.0023 | 0.0733 |
| dnaformer | marker_endblock | 0.0000 | 0.0000 | 0.0024 | 0.0865 |

Files fully recovered (of 40), DNAFormer: free 40/38/3/0, periodic 40/40/5/0, end-block 40/39/2/0.
Majority votes: 0/40 everywhere.

Runtime: majority vote 0.5-1.5 ms/cluster (CPU); DNAFormer ~18 ms/cluster, peak GPU 0.70 GB at batch 32.
mv_sync found 100 / 99.7 / 99.1 / 96.8% of markers at 1/3/5/8% deletion.

## Reading
- Plain majority vote fails even at 1% deletion: one deletion shifts the rest of the read.
- Re-aligning at markers (mv_sync) lowers NED about 10x at 1% and ~1.6x at 8%: markers clearly
  help a simple method.
- Pretrained DNAFormer: near-perfect at 1-3%, ~89-91% at 5%, collapses to ~8% at 8%.
  It never saw markers, so the three arms are statistically indistinguishable (CIs overlap).
  At 8% the end-block arm has the highest NED.
- Large headroom at 5-8% deletion for Stage 4 fine-tuning.
