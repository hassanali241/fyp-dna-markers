# Repeat runs (seed 2): no markers, length 6, length 8

Same data, same settings, different random start (SEED = 2). Results in Drive
`results/stage4/<arm>_seed2/`.

## Strands recovered perfectly (exact recovery)

| Version | Data letters | 5% seed 1 | 5% seed 2 | 8% seed 1 | 8% seed 2 |
|---|---|---|---|---|---|
| no markers (`marker_free`) | 128 | 94.1% | 93.9% | 38.8% | 38.8% |
| length 6 (`marker_len06`, CTGTCA) | 104 | 95.3% | 94.8% | 49.6% | 50.8% |
| length 8 (`marker_periodic`, ACGTCAGT) | 96 | 96.1% | 96.0% | 52.4% | 52.6% |

Seed-2 NED at 8%: no markers 0.0205, length 6 0.0170, length 8 0.0147. Best epochs: 7, 7, 10.

## Data letters recovered per strand at 8% (exact x data letters)

| Version | Seed 1 | Seed 2 | Average |
|---|---|---|---|
| no markers | 49.7 | 49.7 | 49.7 |
| length 6 | 51.6 | 52.8 | **52.2** |
| length 8 | 50.3 | 50.5 | 50.4 |

## Conclusion

- The two runs agree within about 1 point, so training luck is small and earlier results are reliable.
- Reliability order is unchanged: length 8 > length 6 > no markers.
- Length 6 recovers the most data at 8% in **both** runs (about +2.5 letters, +5%, over no markers).
  The gap is small, so we report it as "best of the tested lengths", not a large win.
- Length 6 (CTGTCA) is kept for the spacing study (2 / 4 / 8 markers per strand).
