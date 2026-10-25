"""
Stage 1 validation: checks the generated dataset and prints a summary to paste back.

Usage:
  python stage1_validate.py --repo /content/Deep-DNA-based-storage --data /content/drive/MyDrive/fyp_dna/data/stage1_pilot
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


def read_fasta(path):
    out, name = collections.OrderedDict(), None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif line:
                out[name] = line
    return out


def align_counts(ref, read):
    """Edit-distance alignment; returns (substitutions, insertions, deletions)."""
    n, m = len(ref), len(read)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != read[j - 1]))
    i, j, s, ins, dl = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != read[j - 1]):
            s += ref[i - 1] != read[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, ins, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--align-sample", type=int, default=30, help="clusters per level for error-rate estimate")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.data, "config.json")))
    sizes = cfg["clusters_per_level"]
    levels = cfg["deletion_levels"]
    print("Dataset: %s  mode=%s  sizes per level=%s" % (args.data, cfg["mode"], sizes))
    print("Simulator commit: %s" % cfg["simulator_repo_commit"])

    # ---------------------------------------------------------------- load
    data = {}  # arm -> subset -> list of clusters
    print("\n1. Files and counts")
    subsets = ["train", "val"] + ["test_" + G.level_tag(d) for d in levels]
    for arm in G.ARMS:
        data[arm] = {}
        for sub in subsets:
            folder = os.path.join(args.data, arm, sub)
            files_ok = all(os.path.isfile(os.path.join(folder, f))
                           for f in ("clean.fasta", "noisy_reads.fasta", "metadata.json"))
            check(files_ok, "%s/%s has clean.fasta, noisy_reads.fasta, metadata.json" % (arm, sub), quiet=True)
            if not files_ok:
                continue
            meta = json.load(open(os.path.join(folder, "metadata.json")))
            data[arm][sub] = meta
            want = sizes["test"] if sub.startswith("test") else sizes[sub] * len(levels)
            check(len(meta) == want, "%s/%s: %d clusters (expected %d)" % (arm, sub, len(meta), want), quiet=True)
            clean_fa = read_fasta(os.path.join(folder, "clean.fasta"))
            reads_fa = read_fasta(os.path.join(folder, "noisy_reads.fasta"))
            fasta_ok = (len(clean_fa) == len(meta) and len(reads_fa) == len(meta) * G.N_READS and
                        all(clean_fa[c["cluster_id"]] == c["clean_strand"] for c in meta) and
                        all(reads_fa["%s|read_%d" % (c["cluster_id"], j)] == r
                            for c in meta for j, r in enumerate(c["noisy_reads"])))
            check(fasta_ok, "%s/%s: FASTA files match metadata.json" % (arm, sub), quiet=True)
        n_fail = len(FAILS)
        print("  [%s] %-16s %s" % ("PASS" if n_fail == 0 else "FAIL", arm,
              ", ".join("%s=%d" % (k, len(v)) for k, v in data[arm].items())))

    all_clusters = [c for arm in data for sub in data[arm] for c in data[arm][sub]]

    # ---------------------------------------------------------------- strands
    print("\n2. Strand structure and markers")
    check(all(len(c["clean_strand"]) == 128 for c in all_clusters), "every clean strand is 128 bases")
    check(all(set(c["clean_strand"]) <= set("ACGT") for c in all_clusters), "clean strands use only A/C/G/T")
    check(all(len(c["noisy_reads"]) == 10 for c in all_clusters), "every cluster has exactly 10 noisy reads")
    check(all(set(r) <= set("ACGT") for c in all_clusters for r in c["noisy_reads"]), "reads use only A/C/G/T")
    check(len({c["cluster_id"] for c in all_clusters}) == len(all_clusters), "cluster_ids are unique")

    per = [c for c in all_clusters if c["arm"] == "marker_periodic"]
    endb = [c for c in all_clusters if c["arm"] == "marker_endblock"]
    free = [c for c in all_clusters if c["arm"] == "marker_free"]
    check(all(all(c["clean_strand"][24 + 32 * k:32 + 32 * k] == G.MARKER for k in range(4)) for c in per),
          "periodic: marker ACGTCAGT at positions 24-31, 56-63, 88-95, 120-127")
    check(all(c["marker_positions"] == [[24, 32], [56, 64], [88, 96], [120, 128]] for c in per),
          "periodic: stored marker_positions are correct")
    check(all(c["clean_strand"][96:] == G.MARKER * 4 for c in endb), "end-block: 32 marker bases at positions 96-127")
    check(all(c["marker_positions"] == [] and not c["marker_present"] for c in free), "marker-free: no markers recorded")
    check(all(G.remove_markers(c["arm"], c["clean_strand"]) == c["payload_strand"] for c in all_clusters),
          "removing markers gives back the stored payload (all arms)")
    check(all(bytes.fromhex(c["payload_bytes_hex"]) == G.payload_to_bytes(c["payload_strand"]) for c in all_clusters),
          "payload_bytes_hex matches payload")

    # ---------------------------------------------------------------- pairing
    print("\n3. Pairing across arms")
    idx = {arm: {c["pair_id"]: c for sub in data[arm] for c in data[arm][sub]} for arm in data}
    pair_ids = set(idx["marker_free"])
    check(all(set(idx[a]) == pair_ids for a in G.ARMS), "all three arms contain the same pair_ids")
    paired_ok = True
    for p in pair_ids:
        f, q, e = idx["marker_free"][p], idx["marker_periodic"][p], idx["marker_endblock"][p]
        same = (f["payload_seed"] == q["payload_seed"] == e["payload_seed"] and
                f["noise_seed"] == q["noise_seed"] == e["noise_seed"] and
                f["deletion_rate"] == q["deletion_rate"] == e["deletion_rate"] and
                f["insertion_rate"] == q["insertion_rate"] == e["insertion_rate"] and
                f["substitution_rate"] == q["substitution_rate"] == e["substitution_rate"] and
                f["subset"] == q["subset"] == e["subset"] and
                q["payload_strand"] == e["payload_strand"] == f["payload_strand"][:96])
        paired_ok &= same
    check(paired_ok, "same seed, error rates, split and payload (marker arms = first 96 bases of marker-free payload)")

    # ---------------------------------------------------------------- leakage
    print("\n4. Leakage between splits")
    split_seeds = collections.defaultdict(set)
    split_payloads = collections.defaultdict(set)
    for c in free:
        split_seeds[c["split"]].add(c["payload_seed"])
        split_payloads[c["split"]].add(c["payload_strand"])
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        check(not (split_seeds[a] & split_seeds[b]), "no shared seeds between %s and %s" % (a, b))
        check(not (split_payloads[a] & split_payloads[b]), "no identical payloads between %s and %s" % (a, b))
    p96 = collections.defaultdict(set)
    for c in per:
        p96[c["split"]].add(c["payload_strand"])
    check(not (p96["train"] & (p96["val"] | p96["test"])), "no identical 96-base payloads leak from train")

    # ---------------------------------------------------------------- error levels
    print("\n5. Deletion levels per subset")
    for sub in ["train", "val"]:
        cnt = collections.Counter(c["deletion_rate"] for c in data["marker_free"].get(sub, []))
        check(sorted(cnt) == sorted(levels) and len(set(cnt.values())) == 1,
              "%s mixes all levels equally: %s" % (sub, dict(sorted(cnt.items()))))
    for d in levels:
        sub = "test_" + G.level_tag(d)
        rates = {c["deletion_rate"] for arm in G.ARMS for c in data[arm].get(sub, [])}
        check(rates == {d}, "%s contains only deletion rate %.2f" % (sub, d))
    check(all(c["insertion_rate"] == 0.01 and c["substitution_rate"] == 0.01 for c in all_clusters),
          "insertion and substitution fixed at 1% everywhere")

    # ---------------------------------------------------------------- reproducibility
    print("\n6. Reproducibility")
    ClusterGenerator = G.load_simulator(args.repo)
    rng = random.Random(0)
    sample = rng.sample(all_clusters, min(60, len(all_clusters)))
    plan_lookup = {c["pair_id"]: c for c in G.plan_clusters(sizes)}
    regen_ok = all(G.make_cluster(ClusterGenerator, c["arm"], plan_lookup[c["pair_id"]]) == c for c in sample)
    check(regen_ok, "regenerating %d random clusters from their seeds gives identical clusters" % len(sample))
    sums = json.load(open(os.path.join(args.data, "checksums.json")))
    bad = [p for p, h in sums.items()
           if hashlib.sha256(open(os.path.join(args.data, p), "rb").read()).hexdigest() != h]
    check(not bad, "all %d files match checksums.json" % len(sums))

    # ---------------------------------------------------------------- statistics
    print("\n7. Read statistics per deletion level (marker_free arm, test subsets)")
    print("   level | mean len | expected | measured sub/ins/del per base | reads DNAFormer would DROP (|len-128|>4) | reads >132")
    for d in levels:
        cl = data["marker_free"].get("test_" + G.level_tag(d), [])
        lens = np.array([len(r) for c in cl for r in c["noisy_reads"]])
        s = i = dl = 0
        for c in cl[:args.align_sample]:
            for r in c["noisy_reads"]:
                a, b, e = align_counts(c["clean_strand"], r)
                s, i, dl = s + a, i + b, dl + e
        nb = 128 * G.N_READS * min(len(cl), args.align_sample)
        print("   %4.0f%% | %8.2f | %8.2f | %.4f / %.4f / %.4f            | %5.1f%%                                   | %4.1f%%"
              % (d * 100, lens.mean(), 128 * (1 - d + 0.01), s / nb, i / nb, dl / nb,
                 100 * np.mean(np.abs(lens - 128) > 4), 100 * np.mean(lens > 132)))

    print("\n8. Marker overhead and payload information rate")
    for arm in G.ARMS:
        c = idx[arm][next(iter(pair_ids))]
        print("   %-16s payload %3d / 128 bases -> info rate %.2f, marker overhead %.2f"
              % (arm, c["payload_length"], c["payload_length"] / 128, c["marker_overhead"]))

    print("\n9. Marker appearing by chance inside the payload")
    for arm in ["marker_free", "marker_periodic"]:
        hits = sum(G.MARKER in c["payload_strand"] for c in idx[arm].values())
        print("   %-16s %d / %d payloads contain ACGTCAGT" % (arm, hits, len(idx[arm])))

    ex = idx["marker_periodic"][sorted(pair_ids)[0]]
    print("\n10. Example (marker_periodic, %s, deletion %.0f%%)" % (ex["pair_id"], ex["deletion_rate"] * 100))
    print("   clean : " + ex["clean_strand"])
    print("   read 0: " + ex["noisy_reads"][0] + "  (len %d)" % len(ex["noisy_reads"][0]))

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else "%d CHECK(S) FAILED:\n  - " % len(FAILS) + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
