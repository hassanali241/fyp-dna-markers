"""
Stage 1: paired dataset generator for the marker vs. marker-free DNAFormer study.

Makes three paired "arms" from the SAME random payloads and the SAME noise seeds:
  marker_free      128 payload bases, no markers
  marker_periodic  (24 payload + 8 marker) x 4   = 96 payload + 32 marker
  marker_endblock  96 payload, then the 4 markers together at the end (control)

Noisy reads come from DNAFormer's own simulator
(Deep-DNA-based-storage/DataGenerator/cluster_generator.py). We only give it our
own error rates; we do not edit the repository.

Usage (Colab or local):
  python stage1_generate.py --repo /content/Deep-DNA-based-storage --out /content/drive/MyDrive/fyp_dna/data/stage1_pilot
  python stage1_generate.py --repo ... --out ... --tiny      # quick smoke test
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Fixed experiment settings (saved to config.json)
# ---------------------------------------------------------------------------
BASES = "ACGT"
STRAND_LEN = 128
MARKER = "ACGTCAGT"
MARKER_INTERVAL = 24          # payload bases between markers
N_MARKERS = 4
N_READS = 10
DELETION_LEVELS = [0.01, 0.03, 0.05, 0.08]
INSERTION_RATE = 0.01
SUBSTITUTION_RATE = 0.01
LONG_DELETION_RATE = 0.0      # all deletions are single-base; change later if wanted

ARMS = ["marker_free", "marker_periodic", "marker_endblock"]

# Seed ranges never overlap between splits, so the same payload cannot appear in two splits.
SPLIT_SEED_BASE = {"train": 10_000_000, "val": 20_000_000, "test": 30_000_000}
LEVEL_SEED_STRIDE = 1_000_000  # each deletion level gets its own block of seeds
NOISE_SEED_OFFSET = 1_000_000_000
SHUFFLE_SEED = 12345

PILOT_SIZES = {"train": 2500, "val": 250, "test": 1000}   # clusters PER deletion level
TINY_SIZES = {"train": 20, "val": 5, "test": 10}


def level_tag(d):
    return "del%02d" % round(d * 100)


# ---------------------------------------------------------------------------
# Strand construction
# ---------------------------------------------------------------------------
def make_payload128(payload_seed):
    rng = np.random.default_rng(payload_seed)
    return "".join(BASES[i] for i in rng.integers(0, 4, STRAND_LEN))


def build_strand(arm, payload128):
    """Return (clean_strand, payload, marker_positions)."""
    if arm == "marker_free":
        return payload128, payload128, []
    payload96 = payload128[: MARKER_INTERVAL * N_MARKERS]  # same seed -> first 96 bases shared
    if arm == "marker_periodic":
        parts, positions = [], []
        for k in range(N_MARKERS):
            parts.append(payload96[k * MARKER_INTERVAL:(k + 1) * MARKER_INTERVAL])
            start = sum(len(p) for p in parts)
            parts.append(MARKER)
            positions.append([start, start + len(MARKER)])
        return "".join(parts), payload96, positions
    if arm == "marker_endblock":
        block = MARKER * N_MARKERS
        start = len(payload96)
        positions = [[start + k * len(MARKER), start + (k + 1) * len(MARKER)] for k in range(N_MARKERS)]
        return payload96 + block, payload96, positions
    raise ValueError(arm)


def remove_markers(arm, strand):
    """Payload-only view of a 128-base strand (used by later stages for scoring)."""
    if arm == "marker_free":
        return strand
    if arm == "marker_periodic":
        step = MARKER_INTERVAL + len(MARKER)
        return "".join(strand[k * step:k * step + MARKER_INTERVAL] for k in range(N_MARKERS))
    if arm == "marker_endblock":
        return strand[: MARKER_INTERVAL * N_MARKERS]
    raise ValueError(arm)


def payload_to_bytes(payload):
    """2 bits per base (A=00 C=01 G=10 T=11). 128 bases -> 32 bytes, 96 bases -> 24 bytes."""
    bits = "".join(format(BASES.index(b), "02b") for b in payload)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


# ---------------------------------------------------------------------------
# DNAFormer simulator wrapper
# ---------------------------------------------------------------------------
def load_simulator(repo):
    gen_dir = os.path.join(repo, "DataGenerator")
    if not os.path.isfile(os.path.join(gen_dir, "cluster_generator.py")):
        sys.exit("ERROR: cluster_generator.py not found in %s. Check --repo." % gen_dir)
    sys.path.insert(0, gen_dir)
    from cluster_generator import ClusterGenerator  # noqa: E402
    return ClusterGenerator


def error_rates(deletion_rate):
    """Rates in the simulator's own format.

    The simulator gives each base one error chance equal to the SUM of the general
    rates, then picks the error type with the per-base weights. Using the same numbers
    for both makes the per-base probabilities exactly sub/ins/del.
    """
    general = {"s": SUBSTITUTION_RATE, "i": INSERTION_RATE, "d": deletion_rate, "ld": LONG_DELETION_RATE}
    per_base = {b: {"s": SUBSTITUTION_RATE, "i": 0.25, "pi": INSERTION_RATE,
                    "d": deletion_rate, "ld": LONG_DELETION_RATE} for b in BASES}
    # 'i' only chooses WHICH base is inserted (uniform here); 'pi' is the insertion rate.
    return general, per_base


def simulate_reads(ClusterGenerator, clean, deletion_rate, noise_seed):
    random.seed(noise_seed)          # the simulator uses Python's global random module
    np.random.seed(noise_seed % (2 ** 32))
    general, per_base = error_rates(deletion_rate)
    gen = ClusterGenerator(general, per_base, clean, min_copies=N_READS, max_copies=N_READS)
    gen.cluster_size = N_READS       # the repo's randint(min, max+1) could give 11; force exactly 10
    gen.generate_cluster(delta=0)    # delta=0: no random jitter of the rates
    assert len(gen.copies) == N_READS
    return list(gen.copies)


# ---------------------------------------------------------------------------
# Cluster plan
# ---------------------------------------------------------------------------
def plan_clusters(sizes):
    """List of dicts: split, subset, deletion_rate, payload_seed, noise_seed, pair_id."""
    plan = []
    for split in ["train", "val", "test"]:
        for li, d in enumerate(DELETION_LEVELS):
            for i in range(sizes[split]):
                seed = SPLIT_SEED_BASE[split] + li * LEVEL_SEED_STRIDE + i
                subset = split if split != "test" else "test_" + level_tag(d)
                plan.append({
                    "pair_id": "%s_%s_%06d" % (split, level_tag(d), i),
                    "split": split,
                    "subset": subset,
                    "deletion_rate": d,
                    "payload_seed": seed,
                    "noise_seed": seed + NOISE_SEED_OFFSET,
                })
    # Mix the deletion levels inside train and val (fixed shuffle seed).
    rng = random.Random(SHUFFLE_SEED)
    for split in ["train", "val"]:
        idx = [k for k, c in enumerate(plan) if c["split"] == split]
        items = [plan[k] for k in idx]
        rng.shuffle(items)
        for k, c in zip(idx, items):
            plan[k] = c
    return plan


def make_cluster(ClusterGenerator, arm, c):
    payload128 = make_payload128(c["payload_seed"])
    clean, payload, marker_pos = build_strand(arm, payload128)
    reads = simulate_reads(ClusterGenerator, clean, c["deletion_rate"], c["noise_seed"])
    n_marker = len(clean) - len(payload)
    return {
        "cluster_id": "%s__%s" % (arm, c["pair_id"]),
        "pair_id": c["pair_id"],
        "arm": arm,
        "split": c["split"],
        "subset": c["subset"],
        "seed": c["payload_seed"],
        "payload_seed": c["payload_seed"],
        "noise_seed": c["noise_seed"],
        "marker_present": arm != "marker_free",
        "marker_layout": {"marker_free": "none", "marker_periodic": "periodic",
                          "marker_endblock": "end_block"}[arm],
        "marker_sequence": MARKER if arm != "marker_free" else "",
        "marker_interval": MARKER_INTERVAL if arm == "marker_periodic" else None,
        "marker_positions": marker_pos,
        "marker_overhead": n_marker / len(clean),
        "payload_length": len(payload),
        "strand_length": len(clean),
        "clean_strand": clean,
        "payload_strand": payload,
        "payload_bytes_hex": payload_to_bytes(payload).hex(),
        "noisy_reads": reads,
        "read_lengths": [len(r) for r in reads],
        "deletion_rate": c["deletion_rate"],
        "insertion_rate": INSERTION_RATE,
        "substitution_rate": SUBSTITUTION_RATE,
        "long_deletion_rate": LONG_DELETION_RATE,
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_subset(folder, clusters):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "clean.fasta"), "w", newline="\n") as f:
        for c in clusters:
            f.write(">%s\n%s\n" % (c["cluster_id"], c["clean_strand"]))
    with open(os.path.join(folder, "noisy_reads.fasta"), "w", newline="\n") as f:
        for c in clusters:
            for j, r in enumerate(c["noisy_reads"]):
                f.write(">%s|read_%d\n%s\n" % (c["cluster_id"], j, r))
    with open(os.path.join(folder, "metadata.json"), "w", newline="\n") as f:
        json.dump(clusters, f, separators=(",", ":"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to Deep-DNA-based-storage")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--tiny", action="store_true", help="very small smoke-test dataset")
    args = ap.parse_args()

    sizes = TINY_SIZES if args.tiny else PILOT_SIZES
    ClusterGenerator = load_simulator(args.repo)
    plan = plan_clusters(sizes)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    manifest_rows = []
    for arm in ARMS:
        by_subset = {}
        for k, c in enumerate(plan):
            cl = make_cluster(ClusterGenerator, arm, c)
            by_subset.setdefault(c["subset"], []).append(cl)
            if (k + 1) % 2000 == 0:
                print("  %s: %d/%d clusters (%.0fs)" % (arm, k + 1, len(plan), time.time() - t0), flush=True)
        for subset, clusters in by_subset.items():
            folder = os.path.join(args.out, arm, subset)
            write_subset(folder, clusters)
            for cl in clusters:
                manifest_rows.append({
                    "cluster_id": cl["cluster_id"], "pair_id": cl["pair_id"], "arm": arm,
                    "split": cl["split"], "subset": subset,
                    "deletion_rate": cl["deletion_rate"], "insertion_rate": cl["insertion_rate"],
                    "substitution_rate": cl["substitution_rate"],
                    "payload_seed": cl["payload_seed"], "noise_seed": cl["noise_seed"],
                    "marker_present": cl["marker_present"], "marker_layout": cl["marker_layout"],
                    "marker_sequence": cl["marker_sequence"], "marker_interval": cl["marker_interval"],
                    "payload_length": cl["payload_length"], "strand_length": cl["strand_length"],
                    "marker_overhead": cl["marker_overhead"], "n_reads": len(cl["noisy_reads"]),
                    "mean_read_length": round(float(np.mean(cl["read_lengths"])), 2),
                    "folder": os.path.join(arm, subset).replace("\\", "/"),
                })
        print("%s done (%.0fs)" % (arm, time.time() - t0), flush=True)

    with open(os.path.join(args.out, "dataset_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    config = {
        "stage": 1, "mode": "tiny" if args.tiny else "pilot",
        "clusters_per_level": sizes, "deletion_levels": DELETION_LEVELS,
        "insertion_rate": INSERTION_RATE, "substitution_rate": SUBSTITUTION_RATE,
        "long_deletion_rate": LONG_DELETION_RATE, "strand_length": STRAND_LEN,
        "marker": MARKER, "marker_interval": MARKER_INTERVAL, "n_markers": N_MARKERS,
        "n_reads": N_READS, "arms": ARMS, "split_seed_base": SPLIT_SEED_BASE,
        "level_seed_stride": LEVEL_SEED_STRIDE, "noise_seed_offset": NOISE_SEED_OFFSET,
        "shuffle_seed": SHUFFLE_SEED, "simulator": "DataGenerator/cluster_generator.py",
        "simulator_repo_commit": repo_commit(args.repo),
        "python": sys.version.split()[0], "numpy": np.__version__,
    }
    with open(os.path.join(args.out, "config.json"), "w", newline="\n") as f:
        json.dump(config, f, indent=2)

    checksums = {}
    for root, _, files in os.walk(args.out):
        for name in files:
            if name in ("checksums.json",):
                continue
            p = os.path.join(root, name)
            checksums[os.path.relpath(p, args.out).replace("\\", "/")] = sha256(p)
    with open(os.path.join(args.out, "checksums.json"), "w", newline="\n") as f:
        json.dump(dict(sorted(checksums.items())), f, indent=1)

    print("\nDONE: %d clusters per arm x %d arms in %.0fs -> %s"
          % (len(plan), len(ARMS), time.time() - t0, args.out))


if __name__ == "__main__":
    main()
