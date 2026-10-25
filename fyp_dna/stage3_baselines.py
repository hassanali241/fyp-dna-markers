"""
Stage 3: baseline evaluation on all test clusters. No training.

Methods
  mv            plain majority vote: line the reads up from the left, vote per position.
                Run on all 3 arms.
  mv_sync       marker-sync majority vote (periodic arm only): find the 4 markers in each
                read, cut the read into the 4 payload segments between them, then vote
                inside each segment. Each marker lets the vote re-align after deletions.
  dnaformer     pretrained DNAFormer, no fine-tuning, relaxed read policy (all reads kept).
                Run on all 3 arms.

Scoring is payload-only: marker bases are removed from the prediction before comparing it
with the true payload (128 bases for marker_free, 96 for the marker arms).

Metrics per (method, arm, deletion level)
  exact recovery rate with 95% Wilson CI
  normalized edit distance (edit distance / payload length), mean with 95% CI
  per-base accuracy (same base at the same payload position), mean with 95% CI
  file-level recovery: 25 consecutive test strands = 1 "file"; a file counts as
    recovered only if all 25 strands are exact (no outer error-correcting code), Wilson CI
  runtime (s total, ms per cluster) and peak GPU memory (DNAFormer only)

Outputs (in --out)
  predictions.csv   one row per (method, arm, cluster)
  results.csv       one row per (method, arm, deletion level)
  stage3_log.json   environment, checkpoint report, settings

Usage
  python stage3_baselines.py --repo REPO --data DATA --ckpt CKPT --out OUT
  python stage3_baselines.py --repo REPO --data DATA --out OUT --no-torch    # majority votes only
"""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dnaf_adapter as A  # noqa: E402
import stage1_generate as G  # noqa: E402

FILE_SIZE = 25          # strands per "file" for file-level recovery
MARKER_MAX_ED = 2       # a marker match may have at most this many edits
MARKER_SEARCH = 6       # search +/- this many bases around where the marker should be
DNAF_BATCH = 32
Z95 = 1.959964


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------
def vote(columns_reads, length):
    """Left-aligned positional majority vote over a list of strings.

    Position j is voted on by every string that is long enough. Ties go to the first base
    in A, C, G, T order. Positions no string reaches get 'A' (counted as a guess)."""
    out = []
    for j in range(length):
        counts = [0, 0, 0, 0]
        for r in columns_reads:
            if j < len(r):
                counts[A.BASE_IDX[r[j]]] += 1
        out.append(A.BASES[int(np.argmax(counts))])
    return "".join(out)


def mv_plain(reads):
    return vote(reads, G.STRAND_LEN)


def find_marker(read, expected_start):
    """Best approximate match of the marker near expected_start.

    Semi-global alignment: the marker must match completely, the read may start and end
    anywhere inside the search window. Returns (start, end, edits) or None if the best
    match has more than MARKER_MAX_ED edits."""
    m = G.MARKER
    lo = max(0, expected_start - MARKER_SEARCH)
    hi = min(len(read), expected_start + len(m) + MARKER_SEARCH)
    text = read[lo:hi]
    if not text:
        return None
    exact = text.find(m)
    if exact >= 0:
        # take the exact hit closest to the expected position
        best, s = None, exact
        while s >= 0:
            if best is None or abs(lo + s - expected_start) < abs(lo + best - expected_start):
                best = s
            s = text.find(m, s + 1)
        return lo + best, lo + best + len(m), 0
    # D[i][j]: edits to align marker[:i] with a substring of text ending at j
    n = len(text)
    prev = [0] * (n + 1)              # row i=0: free start anywhere
    prev_start = list(range(n + 1))   # where that alignment starts in text
    for i in range(1, len(m) + 1):
        cur = [i] + [0] * n
        cur_start = [0] + [0] * n
        for j in range(1, n + 1):
            sub = prev[j - 1] + (m[i - 1] != text[j - 1])
            dele = prev[j] + 1        # marker base missing from read
            ins = cur[j - 1] + 1      # extra base in read
            best = min(sub, dele, ins)
            cur[j] = best
            if best == sub:
                cur_start[j] = prev_start[j - 1]
            elif best == dele:
                cur_start[j] = prev_start[j]
            else:
                cur_start[j] = cur_start[j - 1]
        prev, prev_start = cur, cur_start
    best_j = min(range(1, n + 1), key=lambda j: (prev[j], abs(lo + j - len(m) - expected_start)))
    if prev[best_j] > MARKER_MAX_ED:
        return None
    return lo + prev_start[best_j], lo + best_j, prev[best_j]


def split_at_markers(read):
    """Cut one periodic-arm read into its 4 payload segments.

    Markers are searched one after the other; the expected position of the next marker is
    24 bases after the end of the previous one, so a deletion earlier in the read does not
    throw off the later searches. If a marker is not found, we assume it sits where expected.
    Returns (segments, n_markers_found)."""
    segments, found, pos = [], 0, 0
    for _ in range(G.N_MARKERS):
        expected = pos + G.MARKER_INTERVAL
        hit = find_marker(read, expected)
        if hit is None:
            start, end = min(expected, len(read)), min(expected + len(G.MARKER), len(read))
        else:
            start, end, _ = hit
            found += 1
        segments.append(read[pos:start])
        pos = end
    return segments, found


def mv_sync(reads):
    """Marker-sync majority vote. Returns (payload96, fraction of markers found)."""
    per_read = [split_at_markers(r) for r in reads]
    payload = "".join(vote([segs[k] for segs, _ in per_read], G.MARKER_INTERVAL)
                      for k in range(G.N_MARKERS))
    found = sum(f for _, f in per_read) / (G.N_MARKERS * len(reads))
    return payload, found


# ---------------------------------------------------------------------------
# Scoring and statistics
# ---------------------------------------------------------------------------
def score(pred_payload, true_payload):
    ed = A.edit_distance(pred_payload, true_payload)
    same = sum(a == b for a, b in zip(pred_payload, true_payload))
    n = len(true_payload)
    return {"edit_distance": ed, "ned": ed / n, "base_acc": same / n, "exact": int(pred_payload == true_payload)}


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + Z95 ** 2 / n
    mid = (p + Z95 ** 2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    half = Z95 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return m, m - half, m + half


def summarize(method, arm, level, rows, runtime_s, gpu_peak_gb):
    exact = [r["exact"] for r in rows]
    k, n = sum(exact), len(exact)
    lo, hi = wilson(k, n)
    ned_m, ned_lo, ned_hi = mean_ci([r["ned"] for r in rows])
    acc_m, acc_lo, acc_hi = mean_ci([r["base_acc"] for r in rows])
    files = {}
    for r in rows:
        files.setdefault(r["file_id"], []).append(r["exact"])
    full = [f for f in files.values() if len(f) == FILE_SIZE]
    fk = sum(all(f) for f in full)
    flo, fhi = wilson(fk, len(full))
    payload_len = rows[0]["payload_length"]
    return {
        "method": method, "arm": arm, "deletion_rate": level, "n_clusters": n,
        "payload_length": payload_len, "info_rate": payload_len / G.STRAND_LEN,
        "exact_recovery": k / n, "exact_ci_low": lo, "exact_ci_high": hi,
        "ned_mean": ned_m, "ned_ci_low": ned_lo, "ned_ci_high": ned_hi,
        "base_acc_mean": acc_m, "base_acc_ci_low": acc_lo, "base_acc_ci_high": acc_hi,
        "n_files": len(full), "files_recovered": fk,
        "file_recovery": fk / len(full) if full else float("nan"),
        "file_ci_low": flo, "file_ci_high": fhi,
        "runtime_s": runtime_s, "ms_per_cluster": 1000 * runtime_s / n,
        "gpu_peak_gb": gpu_peak_gb,
        "markers_found": (float(np.mean([r["markers_found"] for r in rows]))
                          if rows[0]["markers_found"] != "" else ""),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
PRED_FIELDS = ["method", "arm", "deletion_rate", "cluster_id", "pair_id", "file_id", "payload_length",
               "true_payload", "pred_payload", "edit_distance", "ned", "base_acc", "exact",
               "markers_found"]


def load_test(data, arm, level, limit):
    with open(os.path.join(data, arm, "test_" + G.level_tag(level), "metadata.json")) as f:
        clusters = json.load(f)
    return clusters[:limit] if limit else clusters


def make_rows(method, arm, level, clusters, preds, markers_found=None):
    rows = []
    for i, (c, p) in enumerate(zip(clusters, preds)):
        s = score(p, c["payload_strand"])
        rows.append({
            "method": method, "arm": arm, "deletion_rate": level,
            "cluster_id": c["cluster_id"], "pair_id": c["pair_id"],
            "file_id": "%s_file%03d" % (G.level_tag(level), i // FILE_SIZE),
            "payload_length": len(c["payload_strand"]),
            "true_payload": c["payload_strand"], "pred_payload": p,
            "markers_found": "" if markers_found is None else round(markers_found[i], 3), **s,
        })
    return rows


def run_mv(method, arm, clusters):
    t0 = time.time()
    if method == "mv":
        preds, found = [G.remove_markers(arm, mv_plain(c["noisy_reads"])) for c in clusters], None
    else:
        out = [mv_sync(c["noisy_reads"]) for c in clusters]
        preds, found = [p for p, _ in out], [f for _, f in out]
    return preds, found, time.time() - t0


def run_dnaformer(model, arm, clusters, device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(clusters), DNAF_BATCH):
            chunk = clusters[i:i + DNAF_BATCH]
            x, _ = A.make_batch(chunk, "relaxed", device)
            logits = model(x)["pred"]
            if not torch.isfinite(logits).all():
                raise RuntimeError("DNAFormer produced NaN/inf outputs")
            preds += [G.remove_markers(arm, p) for p in A.decode(logits)]
    if device == "cuda":
        torch.cuda.synchronize()
    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else ""
    return preds, runtime, peak


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def print_table(results):
    print("\n%-10s %-16s %5s  %-22s %-8s %-8s %-6s %s" % ("method", "arm", "del", "exact recovery [95% CI]",
                                                      "NED", "base acc", "files", "ms/cl"))
    for r in results:
        print("%-10s %-16s %4.0f%%  %5.1f%% [%5.1f, %5.1f]    %.4f   %.4f   %2d/%-3d %.1f" % (
            r["method"], r["arm"], 100 * r["deletion_rate"], 100 * r["exact_recovery"],
            100 * r["exact_ci_low"], 100 * r["exact_ci_high"], r["ned_mean"], r["base_acc_mean"],
            r["files_recovered"], r["n_files"], r["ms_per_cluster"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="clusters per level (0 = all)")
    ap.add_argument("--no-torch", action="store_true", help="majority votes only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    levels = G.DELETION_LEVELS
    log = {"python": platform.python_version(), "file_size": FILE_SIZE, "marker_max_ed": MARKER_MAX_ED,
           "marker_search": MARKER_SEARCH, "read_policy": "relaxed", "limit": args.limit}

    model, device = None, "cpu"
    if not args.no_torch:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log["torch"] = torch.__version__
        log["device"] = device
        if device == "cuda":
            log["gpu"] = torch.cuda.get_device_name(0)
        else:
            print("WARNING: no GPU, DNAFormer will be slow. Runtime > Change runtime type > T4 GPU.")
        if not args.ckpt or not os.path.isfile(args.ckpt):
            sys.exit("ERROR: checkpoint not found: %s" % args.ckpt)
        model, _ = A.build_model(args.repo, device)
        rep = A.load_checkpoint(model, args.ckpt, device)
        model.eval()
        print("Checkpoint loaded strictly: %d tensors, renamed %s" % (rep["n_tensors"], rep["renamed"]))
        log["checkpoint"] = {k: rep[k] for k in ("loader", "n_tensors", "checkpoint_epoch", "renamed")}

    runs = [("mv", arm) for arm in G.ARMS] + [("mv_sync", "marker_periodic")]
    if model is not None:
        runs += [("dnaformer", arm) for arm in G.ARMS]

    all_rows, results = [], []
    for method, arm in runs:
        for level in levels:
            clusters = load_test(args.data, arm, level, args.limit)
            found = None
            if method == "dnaformer":
                preds, runtime, peak = run_dnaformer(model, arm, clusters, device)
            else:
                preds, found, runtime = run_mv(method, arm, clusters)
                peak = ""
            rows = make_rows(method, arm, level, clusters, preds, found)
            res = summarize(method, arm, level, rows, runtime, peak)
            all_rows += rows
            results.append(res)
            print("  done %-10s %-16s %2.0f%%: exact %5.1f%%, NED %.4f (%.1fs)" % (
                method, arm, 100 * level, 100 * res["exact_recovery"], res["ned_mean"], runtime), flush=True)

    write_csv(os.path.join(args.out, "predictions.csv"), all_rows, PRED_FIELDS)
    write_csv(os.path.join(args.out, "results.csv"), results, list(results[0].keys()))
    with open(os.path.join(args.out, "stage3_log.json"), "w") as f:
        json.dump(log, f, indent=1)

    print_table(results)
    if model is not None and device == "cuda":
        peaks = [r["gpu_peak_gb"] for r in results if r["method"] == "dnaformer"]
        print("\nDNAFormer inference peak GPU memory: %.2f GB (batch %d)" % (max(peaks), DNAF_BATCH))
    mf = [r for r in results if r["method"] == "mv_sync"]
    if mf:
        print("Markers found by mv_sync (share of 4 markers x 10 reads): " +
              ", ".join("%.0f%%: %.1f%%" % (100 * r["deletion_rate"], 100 * r["markers_found"]) for r in mf))
    n_expected = len(runs) * len(levels)
    ok = len(results) == n_expected and all(r["n_clusters"] == results[0]["n_clusters"] for r in results)
    print("\nSaved %d prediction rows and %d result rows to %s" % (len(all_rows), len(results), args.out))
    print("STAGE 3 DONE" if ok else "STAGE 3 INCOMPLETE: expected %d result rows" % n_expected)


if __name__ == "__main__":
    main()
