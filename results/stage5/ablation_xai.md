# Stage 5 Parts B + C: marker ablation and XAI (8% deletion unless noted)

Ablation: every marker letter replaced by a different random letter, reads re-simulated with the same noise
seed, same fine-tuned model. "Intact" is recomputed in the same Colab session (torch 2.14), so it can differ
by ~0.5 pt from the Stage 4 numbers; intact vs scrambled is the fair comparison. p = exact McNemar.

XAI: marker share = share of gradient x input importance (data output letters only) that falls on marker
positions; ratio = marker share / space share. marker_free uses the 6-base x4 positions as a placebo.

| run | ablation 5% | ablation 8% | space | grad x input | IG | GradSHAP | attention | ratio |
|---|---|---|---|---|---|---|---|---|
| marker_free (placebo) | - | - | 18.8% | 18.4% | 17.5% | 17.6% | 21.7% | x0.98 |
| marker_endblock | 95.4 -> 94.3% (p=0.099) | 40.5 -> 36.7% (-3.8, p=0.0029) | 25.0% | 14.1% | 14.8% | 15.1% | 20.7% | x0.56 |

Run by Haad, 2026-11-17. Pending: marker_len06, marker_len06_n1 (Hassan); marker_len06_n8, marker_periodic (Mahek).

Reading so far: the end-block model does depend on its markers a little (significant 3.8-pt drop at 8%),
but gives them less importance than their size (x0.56) when decoding data letters: they sit where the
strand is already easy, so they cannot re-synchronize the error-prone middle.
