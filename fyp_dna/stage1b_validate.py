"""
Stage 1b validation: checks the marker-length datasets against the original Stage 1 data.

Each new arm must use the SAME payloads (first data bases of the marker_free payload),
the SAME seeds and error rates, and get errors at the SAME positions (so every read
has exactly the same length as the matching marker_free read).

Usage:
  python stage1b_validate.py --repo REPO --data NEW_DATA --ref STAGE1_DATA
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_generate as G  # noqa: E402

FAILS = []


def check(ok, msg, quiet=False):
    if not ok or not quiet:
        print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILS.append(msg)


def read_fasta(path):
    out, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif line:
                out[name] = line
    return out


def load(data, arm, sub):
    with open(os.path.join(data, arm, sub, "metadata.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True, help="new marker-length dataset folder")
    ap.add_argument("--ref", required=True, help="original Stage 1 dataset folder (has marker_free)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.data, "config.json")))
    arms, sizes, levels = cfg["arms"], cfg["clusters_per_level"], cfg["deletion_levels"]
    subsets = ["train", "val"] + ["test_" + G.level_tag(d) for d in levels]
    print("New data: %s (%s)\nReference: %s\nArms: %s" % (args.data, cfg["mode"], args.ref, arms))

    print("\n1. Reference marker_free data")
    ref = {}
    for sub in subsets:
        ref.update({c["pair_id"]: c for c in load(args.ref, "marker_free", sub)})
    check(len(ref) == sum(sizes[s if not s.startswith("test") else "test"] * (1 if s.startswith("test") else len(levels))
                          for s in subsets), "loaded %d reference clusters" % len(ref))

    rows = []
    for arm in arms:
        m = G.arm_marker(arm)
        L, d = len(m), G.data_per_block(arm)
        n_data = 128 - 4 * L
        want_pos = [[k * 32 + d, k * 32 + 32] for k in range(4)]
        print("\n2. %s  marker %s (length %d): 4 x (%d data + %d marker) = %d data bases"
              % (arm, m, L, d, L, n_data))
        total, chance = 0, 0
        ok = collections.Counter()
        for sub in subsets:
            folder = os.path.join(args.data, arm, sub)
            meta = load(args.data, arm, sub)
            want = sizes["test"] if sub.startswith("test") else sizes[sub] * len(levels)
            check(len(meta) == want, "%s/%s: %d clusters (expected %d)" % (arm, sub, len(meta), want), quiet=True)
            clean_fa = read_fasta(os.path.join(folder, "clean.fasta"))
            reads_fa = read_fasta(os.path.join(folder, "noisy_reads.fasta"))
            check(len(clean_fa) == len(meta) and len(reads_fa) == 10 * len(meta)
                  and all(clean_fa[c["cluster_id"]] == c["clean_strand"] for c in meta)
                  and all(reads_fa["%s|read_%d" % (c["cluster_id"], j)] == r
                          for c in meta for j, r in enumerate(c["noisy_reads"])),
                  "%s/%s: FASTA files match metadata.json" % (arm, sub), quiet=True)
            for c in meta:
                r = ref[c["pair_id"]]
                s = c["clean_strand"]
                ok["len128"] += len(s) == 128 and set(s) <= set("ACGT")
                ok["markers"] += all(s[a:b] == m for a, b in want_pos) and c["marker_positions"] == want_pos
                ok["payload"] += (c["payload_strand"] == r["payload_strand"][:n_data]
                                  and G.remove_markers(arm, s) == c["payload_strand"]
                                  and len(c["payload_strand"]) == n_data)
                ok["same_seeds_rates"] += (c["payload_seed"] == r["payload_seed"] and c["noise_seed"] == r["noise_seed"]
                                           and c["deletion_rate"] == r["deletion_rate"] and c["subset"] == r["subset"]
                                           and c["insertion_rate"] == r["insertion_rate"]
                                           and c["substitution_rate"] == r["substitution_rate"])
                ok["same_error_positions"] += [len(x) for x in c["noisy_reads"]] == [len(x) for x in r["noisy_reads"]]
                ok["ten_reads"] += len(c["noisy_reads"]) == 10
                chance += m in c["payload_strand"]
                total += 1
        check(all(v == total for v in ok.values()) and len(ok) == 6,
              "all %d clusters: 128 bases, markers at %s, payload = first %d bases of marker_free payload, "
              "same seeds/rates, 10 reads" % (total, [p[0] for p in want_pos], n_data))
        check(ok["same_error_positions"] == total,
              "every read has the same length as the matching marker_free read (same error positions)")
        rows.append((arm, m, L, n_data, 4 * L / 128, chance / total))

    print("\n3. Reproducibility")
    ClusterGenerator = G.load_simulator(args.repo)
    plan = {c["pair_id"]: c for c in G.plan_clusters(sizes)}
    rng = random.Random(0)
    regen = True
    for arm in arms:
        meta = load(args.data, arm, "test_del08") + load(args.data, arm, "train")[:200]
        for c in rng.sample(meta, 10):
            regen &= G.make_cluster(ClusterGenerator, arm, plan[c["pair_id"]]) == c
    check(regen, "regenerating 10 random clusters per arm from their seeds gives identical clusters")
    sums = json.load(open(os.path.join(args.data, "checksums.json")))
    bad = [p for p, h in sums.items()
           if hashlib.sha256(open(os.path.join(args.data, p), "rb").read()).hexdigest() != h]
    check(not bad, "all %d files match checksums.json" % len(sums))

    print("\n4. Summary")
    print("   arm            marker        length  data bases  marker share  payloads containing marker by chance")
    print("   %-14s %-13s %6d  %10d  %11.1f%%  %s" % ("marker_free", "-", 0, 128, 0.0, "-"))
    for arm, m, L, n, ov, ch in sorted(rows, key=lambda r: r[2]):
        print("   %-14s %-13s %6d  %10d  %11.1f%%  %.2f%%" % (arm, m, L, n, 100 * ov, 100 * ch))
    print("   (length 8, ACGTCAGT: marker_periodic in the Stage 1 data, 96 data bases, 25.0%)")

    print("\n" + ("ALL STAGE 1b CHECKS PASSED" if not FAILS else
                  "%d CHECK(S) FAILED:\n  - " % len(FAILS) + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
