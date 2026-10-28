# Marker-length sweep: running comparison (test, 1,000 strands per level)

Data recovered = exact strands x data bases per strand (per 1,000 strands).
Break-even at 8% = recovery needed to match marker-free (49,664 data bases).

| version | marker | data bases | overhead | exact 5% | exact 8% [95% CI] | NED 8% | data recovered 8% | break-even 8% |
|---|---|---|---|---|---|---|---|---|
| marker_free | - | 128 | 0% | 94.1% | 38.8% [35.8, 41.9] | 0.0202 | 49,664 | - |
| marker_len04 | CAGT | 112 | 12.5% | 95.4% | 44.6% [41.5, 47.7] | 0.0183 | 49,952 | 44.3% |
| marker_len05 | TCATG | 108 | 15.6% | 95.6% | 45.7% [42.6, 48.8] | 0.0182 | 49,356 | 46.0% |
| marker_len06 | CTGTCA | 104 | 18.8% | 95.3% | 49.6% [46.5, 52.7] | 0.0172 | 51,584 | 47.8% |
| marker_len07 | CTGATCA | 100 | 21.9% | pending | | | | 49.7% |
| marker_periodic (8) | ACGTCAGT | 96 | 25.0% | 96.1% | 52.4% [49.3, 55.5] | 0.0148 | 50,304 | 51.7% |
| marker_len12 | ACGTCAGTGCAT | 80 | 37.5% | pending | | | | 62.1% |
| marker_endblock (8, control) | ACGTCAGT | 96 | 25.0% | 95.7% | 41.0% [38.0, 44.1] | 0.0241 | 39,360 | 51.7% |

Winner rule (agreed before results): most data recovered at 8% deletion; ties go to the shorter marker.
