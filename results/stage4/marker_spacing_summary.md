# Marker-spacing study: 6-base marker CTGTCA, 1 / 2 / 4 / 8 markers per 128-base strand

Same payloads, error positions and locked settings as all other runs. Test: 1,000 strands per level.
Data recovered = exact strands x data bases (per 1,000 strands). 4 markers = average of seeds 1 and 2.

| markers | layout (data + marker) | data bases | overhead | exact 5% | exact 8% [95% CI] | NED 8% | data 8% | vs free 8% | data 5% | vs free 5% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 128 | 0% | 94.1% | 38.8% [35.8, 41.9] | 0.0202 | 49,664 | - | 120,448 | - |
| 1 | 61 + M + 61 (middle) | 122 | 4.7% | 95.5% | 46.3% [43.2, 49.4] | 0.0165 | **56,486** | **+13.7%** | 116,510 | -3.3% |
| 2 | 2 x (58 + M), last at strand end | 116 | 9.4% | 95.1% | 44.7% [41.6, 47.8] | 0.0177 | 51,852 | +4.4% | 110,316 | -8.4% |
| 4 | 4 x (26 + M), last at strand end | 104 | 18.75% | 95.1% | 50.2% (49.6 / 50.8) | 0.0171 | 52,208 | +5.1% | 98,852 | -17.9% |
| 8 | 8 x (10 + M), last at strand end | 80 | 37.5% | 97.4% | **69.6% [66.7, 72.4]** | **0.0096** | 55,680 | +12.1% | 77,920 | -35.3% |

## Reading
1. Accuracy at 8% rises with the number of markers, steeply from 4 to 8 (50.2 -> 69.6%).
2. Data recovered at 8% is highest for 1 marker (+13.7%) and 8 markers (+12.1%); the two are within each
   other's uncertainty. 2 and 4 markers gain less (+4-5%).
3. At 5% deletion every marker costs stored data; 1 marker costs least (-3.3%), 8 markers most (-35%).
   1 marker in the middle is therefore the most practical all-round design; 8 markers is the
   heavy-deletion specialist (best accuracy).
4. Layout caveat: in the 2/4/8 layouts the last marker sits at the very end of the strand, where it
   gives little synchronization (cf. the end-block control). The 2-marker layout therefore has only one
   useful anchor, like the 1-marker layout, which matches their similar accuracy (44.7 vs 46.3%).
   Spacing markers evenly inside the strand (e.g. 2 markers at 1/3 and 2/3) may do better.
5. Single runs (seed 1) except 4 markers; seed-2 repeats of 1 and 8 markers would confirm the ranking.
