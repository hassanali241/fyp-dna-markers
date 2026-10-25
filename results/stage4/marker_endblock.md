# Stage 4: fine-tuned DNAFormer, marker_endblock (control) (Colab T4, 25 Oct 2026)

Settings: locked Stage 4 settings (identical to the other arms). Best epoch 10.
Full CSVs in Drive: `MyDrive/fyp_dna/results/stage4/marker_endblock/`.

## Test (1,000 clusters per level, payload-only, 96 data bases)

| deletion | exact [95% CI] | NED | base acc | files | pretrained exact (Stage 3) | pretrained NED |
|---|---|---|---|---|---|---|
| 1% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 | 100.0% | 0.0000 |
| 3% | 100.0% [99.6, 100.0] | 0.0000 | 1.0000 | 40/40 | 99.9% | 0.0000 |
| 5% | 95.7% [94.3, 96.8] | 0.0012 | 0.9980 | 13/40 | 89.3% | 0.0024 |
| 8% | 41.0% [38.0, 44.1] | 0.0241 | 0.9484 | 0/40 | 7.5% | 0.0865 |
