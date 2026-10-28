# Marker-length sweep: all results (test, 1,000 strands per deletion level)

Same pretrained start, same payloads, error positions and locked settings for every version; one training run each.
Data recovered = exact strands x data bases per strand, per 1,000 strands.
Break-even = recovery needed to match marker-free in data recovered.

## 8% deletion

| version | marker | data bases | overhead | exact 8% [95% CI] | NED 8% | data recovered | vs marker-free | break-even |
|---|---|---|---|---|---|---|---|---|
| marker_free | - | 128 | 0% | 38.8% [35.8, 41.9] | 0.0202 | 49,664 | - | - |
| marker_len04 | CAGT | 112 | 12.5% | 44.6% [41.5, 47.7] | 0.0183 | 49,952 | +0.6% | 44.3% |
| marker_len05 | TCATG | 108 | 15.6% | 45.7% [42.6, 48.8] | 0.0182 | 49,356 | -0.6% | 46.0% |
| **marker_len06** | CTGTCA | 104 | 18.75% | 49.6% [46.5, 52.7] | 0.0172 | **51,584** | **+3.9%** | 47.8% |
| marker_len07 | CTGATCA | 100 | 21.9% | 49.9% [46.8, 53.0] | 0.0168 | 49,900 | +0.5% | 49.7% |
| marker_periodic (8) | ACGTCAGT | 96 | 25.0% | 52.4% [49.3, 55.5] | 0.0148 | 50,304 | +1.3% | 51.7% |
| marker_len12 | ACGTCAGTGCAT | 80 | 37.5% | 62.3% [59.3, 65.3] | 0.0119 | 49,840 | +0.4% | 62.1% |
| marker_endblock (8, control) | ACGTCAGT | 96 | 25.0% | 41.0% [38.0, 44.1] | 0.0241 | 39,360 | -20.7% | 51.7% |

## 5% deletion

| version | exact 5% | data recovered | vs marker-free |
|---|---|---|---|
| marker_free | 94.1% | 120,448 | - |
| marker_len04 | 95.4% | 106,848 | -11.3% |
| marker_len05 | 95.6% | 103,248 | -14.3% |
| marker_len06 | 95.3% | 99,112 | -17.7% |
| marker_len07 | 96.1% | 96,100 | -20.2% |
| marker_periodic (8) | 96.1% | 92,256 | -23.4% |
| marker_len12 | 97.1% | 77,680 | -35.5% |

## Reading (preliminary; paired tests and repeat seeds in Stage 5)
1. Accuracy rises with marker length at 8% deletion: 44.6 (4), 45.7 (5), 49.6 (6), 49.9 (7), 52.4 (8), 62.3% (12).
   Lengths 4-5 and 6-7 form pairs; 12 is clearly above 8 (CIs do not overlap).
2. Stored data at 8% is close to break-even for every length (-0.6% to +1.3%) except length 6 (+3.9%),
   the nominal winner under the agreed rule. Its margin is within single-run noise, so it needs confirmation.
3. At 5% deletion all versions recover 94-97%, so markers cost their overhead in stored data there:
   markers pay for themselves only under heavy deletion.
4. Winner rule (agreed before results): most data recovered at 8% deletion; ties go to the shorter marker.
