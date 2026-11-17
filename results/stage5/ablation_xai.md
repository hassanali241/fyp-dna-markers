# Stage 5 Parts B + C: marker ablation and XAI (8% deletion unless noted)

Ablation: every marker letter replaced by a different random letter, reads re-simulated with the same noise
seed, same fine-tuned model (best.pt from Drive). "Intact" is recomputed in the same Colab session, so intact vs
scrambled is the fair comparison. p = exact McNemar, 1,000 test strands per level.

XAI: marker share = share of gradient x input importance (data output letters only) on marker positions;
ratio = marker share / space share. marker_free uses the 6-base x4 positions as a placebo.

| run | model epoch in Drive | ablation 5% | ablation 8% | space | grad x input | IG | GradSHAP | attention | ratio |
|---|---|---|---|---|---|---|---|---|---|
| marker_free (placebo) | 9 (ok) | - | - | 18.8% | 18.4% | 17.5% | 17.6% | 21.7% | x0.98 |
| marker_endblock | **7 (tested model was 10)** | 95.4 -> 94.3% (p=0.099) | 40.5 -> 36.7% (-3.8, p=0.0029) | 25.0% | 14.1% | 14.8% | 15.1% | 20.7% | x0.56 |
| marker_periodic (8-base x4) | 10 (ok) | 96.1 -> 92.7% (-3.4, p=3e-6) | 52.4 -> 30.4% (-22.0, p=2e-38) | 25.0% | 20.9% | 20.9% | 21.0% | 25.1% | x0.84 |
| marker_len06 (6-base x4) | 7 (ok) | 95.3 -> 93.1% (-2.2, p=0.005) | 49.6 -> 31.8% (-17.8, p<1e-15) | 18.75% | 18.9% | 18.3% | 18.5% | 19.8% | x1.01 |
| marker_len06_n8 (6-base x8) | 9 (ok) | 97.4 -> 87.4% (-10.0, p=2e-22) | 69.6 -> 18.4% (-51.2, p=9e-135) | 37.5% | 37.3% | 36.4% | 36.5% | 36.3% | x0.99 |
| marker_len06_n1 (6-base x1, middle) | 9 (ok, retrained 2026-11-17) | 95.1 -> 93.3% (-1.8, p=0.0096) | 45.8 -> 33.8% (-12.0, p=1e-20) | 4.7% | 5.8% | 5.7% | 5.7% | 4.0% | **x1.24** |

**Drive problem (fixed).** Google Drive had kept an epoch-1 best.pt for marker_len06_n1 (and an early one for
marker_len06_n2), not the tested models. Both were redone on 2026-11-17 (n1 retrained from scratch: 45.8% at 8%;
n2 resumed: 43.7%) and the n1 row above is from the correct model (intact 45.8% = Stage 4 test). For
marker_endblock Drive kept the epoch-7 model (test model: epoch 10); val scores are nearly the same
(val NED 0.0061 vs 0.0058; 8% test 40.5 vs 41.0%), so its rows are kept with this note.
stage4_finetune.py now refuses to test a best.pt whose epoch does not match, and the notebook flushes Drive.

## Reading
1. Ablation is the clearest result: scrambling evenly spaced markers collapses 8% recovery
   (6-base x8: 69.6 -> 18.4%, below the no-marker model's 38.8%; 8-base x4: -22 pts; 6-base x4: -18 pts;
   one 6-base marker in the middle: -12 pts), while scrambling the 32-letter end-block costs only 3.8 pts.
   6 letters in the middle matter about 3x more than 32 letters at the end: position, not marker amount.
2. Heatmaps: every interior marker shows a bright spot on the diagonal: output letters next to a marker draw
   heavily on the marker letters, and the diagonal stays sharp through the middle of the strand
   (with no markers it widens there).
3. Strand-wide marker share: x1.24 for the single middle marker (the only one clearly above its space;
   IG and GradSHAP agree at 5.7%), about x1 for 4-8 interior markers (x0.84-1.01) and x0.56 for the end-block. The global share therefore does not show extra weight on markers; their effect is
   local (neighbouring letters) and causal (ablation), not a large share of total importance.
