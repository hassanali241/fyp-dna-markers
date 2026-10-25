# Stage 4 summary: fine-tuned DNAFormer, all three arms (test sets, 1,000 clusters per level)

Same pretrained start, same data pairs and noise, same settings and budget (10 epochs x 625 steps).
Scoring is payload-only. 95% Wilson CIs in brackets.

## Exact strand recovery

| deletion | marker_free (128 data) | marker_periodic (96 data) | marker_endblock (96 data, control) |
|---|---|---|---|
| 1% | 100.0% | 100.0% | 100.0% |
| 3% | 99.8% | 100.0% | 100.0% |
| 5% | 94.1% [92.5, 95.4] | 96.1% [94.7, 97.1] | 95.7% [94.3, 96.8] |
| 8% | 38.8% [35.8, 41.9] | **52.4% [49.3, 55.5]** | 41.0% [38.0, 44.1] |

## Normalized edit distance (lower is better)

| deletion | marker_free | marker_periodic | marker_endblock |
|---|---|---|---|
| 5% | 0.0012 | 0.0009 | 0.0012 |
| 8% | 0.0202 | **0.0148** | 0.0241 |

## Data recovered per 1,000 strands at 8% deletion (exact strands x data bases)

| arm | bases |
|---|---|
| marker_free | 388 x 128 = 49,664 |
| marker_periodic | 524 x 96 = 50,304 (+1.3% vs free) |
| marker_endblock | 410 x 96 = 39,360 (-20.7% vs free) |

## Gain from fine-tuning (exact recovery at 8%, pretrained -> fine-tuned)
free 7.6 -> 38.8, periodic 7.9 -> 52.4, end-block 7.5 -> 41.0

## Interpretation (preliminary; paired tests in Stage 5)
1. Spacing matters: periodic beats the end-block control at 8% (52.4% vs 41.0%, CIs do not overlap)
   with the same 96 data bases and the same 32 marker bases. The gain is due to the markers'
   placement, i.e. synchronization, not to the shorter payload.
2. Shorter payload alone helps little: end-block vs marker-free at 8% is 41.0% vs 38.8% (CIs overlap),
   and its NED is worse (0.0241 vs 0.0202).
3. At 5% deletion the three arms are within each other's CIs; the effect appears only when deletions are heavy.
4. Storage cost: periodic markers reach break-even with marker-free in data recovered at 8% (+1.3%),
   so they pay for their 25% overhead there but do not yet give a clear net gain.
