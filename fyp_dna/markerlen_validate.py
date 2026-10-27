"""
Marker-length sweep: validate the generated dataset (arms marker_len2 ... marker_len7).

Checks the same things as Stage 1 for every arm: files and counts, strand structure, the
marker at the right places, marker removal, pairing with the Stage 1 payloads (each payload is
a prefix of the same seed's marker-free payload), leakage, deletion levels, reproducibility
and checksums. Then prints the overhead and the chance-collision numbers per marker length.

Usage:
  python markerlen_validate.py --repo /content/Deep-DNA-based-storage --data .../data/markerlen
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_generate as G  # noqa: E402

FAILS = []


def check(ok, msg, quiet=False):
    if not ok or not quiet:
        print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.data, "config.json")))
    sizes, levels, arms = cfg["clusters_per_level"], cfg["deletion_levels"], cfg["arms"]
    print("Dataset: %s  mode=%s  arms=%s" % (args.data, cfg["mode"], ",".join(arms)))
    print("Simulator commit: %s" % cfg["simulator_repo_commit"])
    check(all(a in G.LENGTH_ARMS for a in arms), "config lists only marker-length arms")

    print("\n1. Files and counts")
    subsets = ["train", "val"] + ["test_" + G.level_tag(d) for d in levels]
    data = {}
    for arm in arms:
        data[arm] = {}
        n_before = len(FAILS)
        for sub in subsets:
            folder = os.path.join(args.data, arm, sub)
            ok = all(os.path.isfile(os.path.join(folder, f)) for f in ("clean.fasta", "noisy_reads.fasta", "metadata.json"))
            check(ok, "%s/%s has its 3 files" % (arm, sub), quiet=True)
            if not ok:
                continue
            meta = json.load(open(os.path.join(folder, "metadata.json")))
            data[arm][sub] = meta
            want = sizes["test"] if sub.startswith("test") else sizes[sub] * len(levels)
            check(len(meta) == want, "%s/%s: %d clusters (expected %d)" % (arm, sub, len(meta), want), quiet=True)
        print("  [%s] %-12s %s" % ("PASS" if len(FAILS) == n_before else "FAIL", arm,
              ", ".join("%s=%d" % (k, len(v)) for k, v in data[arm].items())))
    all_c = [c for a in data for s in data[a] for c in data[a][s]]

    print("\n2. Strand structure and markers")
    check(all(len(c["clean_strand"]) == 128 for c in all_c), "every clean strand is 128 bases")
    check(all(len(c["noisy_reads"]) == 10 for c in all_c), "every cluster has exactly 10 noisy reads")
    check(len({c["cluster_id"] for c in all_c}) == len(all_c), "cluster_ids are unique")
    for arm in arms:
        marker, seg = G.periodic_layout(arm)
        L = len(marker)
        cl = [c for s in data[arm] for c in data[arm][s]]
        pos = [[k * G.MARKER_PERIOD + seg, (k + 1) * G.MARKER_PERIOD] for k in range(G.N_MARKERS)]
        ok = (all(all(c["clean_strand"][a:b] == marker for a, b in pos) for c in cl) and
              all(c["marker_positions"] == pos and c["marker_sequence"] == marker for c in cl) and
              all(G.remove_markers(arm, c["clean_strand"]) == c["payload_strand"] for c in cl) and
              all(len(c["payload_strand"]) == 128 - G.N_MARKERS * L for c in cl))
        check(ok, "%s: marker %-8s every %d bases at %s; payload %d bases; removal OK"
              % (arm, marker, G.MARKER_PERIOD, ",".join("%d-%d" % (a, b - 1) for a, b in pos), 128 - 4 * L))

    print("\n3. Pairing with Stage 1 (same seeds)")
    ok = all(c["payload_strand"] == G.make_payload128(c["payload_seed"])[:len(c["payload_strand"])] and
             c["noise_seed"] == c["payload_seed"] + G.NOISE_SEED_OFFSET for c in all_c)
    check(ok, "every payload = prefix of the same seed's 128-base marker-free payload")
    ids = {a: {c["pair_id"] for s in data[a] for c in data[a][s]} for a in arms}
    check(len({frozenset(v) for v in ids.values()}) == 1, "all arms contain the same pair_ids as each other")

    print("\n4. Leakage and deletion levels")
    seeds = collections.defaultdict(set)
    for c in all_c:
        seeds[c["split"]].add(c["payload_seed"])
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        check(not (seeds[a] & seeds[b]), "no shared seeds between %s and %s" % (a, b))
    for d in levels:
        sub = "test_" + G.level_tag(d)
        check({c["deletion_rate"] for a in arms for c in data[a].get(sub, [])} == {d},
              "%s contains only deletion rate %.2f" % (sub, d))

    print("\n5. Reproducibility")
    CG = G.load_simulator(args.repo)
    plan = {c["pair_id"]: c for c in G.plan_clusters(sizes)}
    sample = random.Random(0).sample(all_c, min(60, len(all_c)))
    check(all(G.make_cluster(CG, c["arm"], plan[c["pair_id"]]) == c for c in sample),
          "regenerating %d random clusters from their seeds gives identical clusters" % len(sample))
    sums = json.load(open(os.path.join(args.data, "checksums.json")))
    bad = [p for p, h in sums.items()
           if hashlib.sha256(open(os.path.join(args.data, p), "rb").read()).hexdigest() != h]
    check(not bad, "all %d files match checksums.json" % len(sums))

    print("\n6. Overhead and chance collisions per marker length")
    print("   arm          marker    payload  overhead  payloads containing the marker  expected chance hits in a +/-6 window")
    for arm in arms:
        marker, seg = G.periodic_layout(arm)
        L = len(marker)
        pays = [c["payload_strand"] for s in data[arm] for c in data[arm][s]]
        has = sum(marker in p for p in pays)
        print("   %-12s %-9s %3d      %5.1f%%    %6.1f%%                          %.3f"
              % (arm, marker, 128 - 4 * L, 100 * 4 * L / 128, 100 * has / len(pays), (12 + 1) * 4.0 ** -L))

    print("\n7. Mean read length at 8% deletion (test)")
    for arm in arms:
        cl = data[arm].get("test_del08", [])
        print("   %-12s %.2f" % (arm, float(np.mean([len(r) for c in cl for r in c["noisy_reads"]]))))

    ex = data[arms[0]]["test_del01"][0]
    print("\n8. Example (%s, %s)" % (arms[0], ex["pair_id"]))
    print("   clean : " + ex["clean_strand"])
    print("   read 0: " + ex["noisy_reads"][0])

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else "%d CHECK(S) FAILED:\n  - " % len(FAILS) + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
