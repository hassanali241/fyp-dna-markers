"""
Stage 5: final evaluation, statistics, marker ablation and explainability (XAI).

Parts (run separately; each writes into <out>/<part>/):
  stats     CPU. Reads every Stage 4 run's predictions.csv and results.csv and makes
            - results_all.csv           every run x deletion level (exact, NED, data recovered)
            - paired_tests.csv          McNemar exact test on the same test strands + paired bootstrap
                                        CI of the data-recovered difference, for the planned comparisons
            - fig_*.png / *.csv         recovery, NED and data recovered vs deletion rate per study
            - error_position_*.png/csv  where along the strand the errors happen (8% deletion)
  ablation  GPU. Rebuilds each test strand with its markers replaced by random bases, re-simulates
            the reads with the SAME noise seed (same error positions), runs the fine-tuned model and
            compares with the normal strands: does the model need the markers?
  xai       GPU. Attribution: which read positions the fine-tuned model relies on.
            - gradient x input for every output letter -> heatmap (output position x strand position)
            - Integrated Gradients and GradientSHAP (Captum) on fewer strands, output in 8 segments
            - "marker attribution share" vs the share of the strand the markers occupy
            - attention: share of transformer attention that goes to marker positions
            - example figure: one read with its letters shaded by importance
Usage:
  python stage5_analysis.py --part stats    --results R --out O
  python stage5_analysis.py --part ablation --repo REPO --project P --out O --arms a,b
  python stage5_analysis.py --part xai      --repo REPO --project P --out O --arms a,b
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dnaf_adapter as A  # noqa: E402
import stage1_generate as G  # noqa: E402

LEVELS = G.DELETION_LEVELS
STRAND = G.STRAND_LEN
B_BOOT = 5000

# Planned comparisons (A, B): "is A different from B?"
COMPARISONS = [
    ("marker_periodic", "marker_endblock", "spacing: periodic vs end-block (same content)"),
    ("marker_periodic", "marker_free", "8-base markers vs no markers"),
    ("marker_endblock", "marker_free", "end-block vs no markers (payload effect)"),
    ("marker_len04", "marker_free", "4-base vs no markers"),
    ("marker_len05", "marker_free", "5-base vs no markers"),
    ("marker_len06", "marker_free", "6-base vs no markers"),
    ("marker_len07", "marker_free", "7-base vs no markers"),
    ("marker_len12", "marker_free", "12-base vs no markers"),
    ("marker_len06", "marker_periodic", "6-base vs 8-base"),
    ("marker_len12", "marker_periodic", "12-base vs 8-base"),
    ("marker_len06_n1", "marker_free", "1 marker vs no markers"),
    ("marker_len06_n2", "marker_free", "2 markers vs no markers"),
    ("marker_len06_n8", "marker_free", "8 markers vs no markers"),
    ("marker_len06_n1", "marker_len06", "1 marker vs 4 markers"),
    ("marker_len06_n8", "marker_len06", "8 markers vs 4 markers"),
    ("marker_len06_n1", "marker_len06_n8", "1 marker vs 8 markers"),
    ("marker_len06_n8", "marker_len12", "8 x 6 vs 4 x 12 (same space)"),
    ("marker_free_seed2", "marker_free", "repeat run: no markers seed 2 vs 1"),
    ("marker_len06_seed2", "marker_len06", "repeat run: 6-base seed 2 vs 1"),
    ("marker_periodic_seed2", "marker_periodic", "repeat run: 8-base seed 2 vs 1"),
]
STUDIES = {
    "main": ["marker_free", "marker_periodic", "marker_endblock"],
    "lengths": ["marker_free", "marker_len04", "marker_len05", "marker_len06", "marker_len07",
                "marker_periodic", "marker_len12"],
    "spacing": ["marker_free", "marker_len06_n1", "marker_len06_n2", "marker_len06", "marker_len06_n8"],
}
LABEL = {"marker_free": "no markers", "marker_periodic": "8-base x4", "marker_endblock": "end-block 8-base x4",
         "marker_len04": "4-base x4", "marker_len05": "5-base x4", "marker_len06": "6-base x4",
         "marker_len07": "7-base x4", "marker_len12": "12-base x4", "marker_len06_n1": "6-base x1 (middle)",
         "marker_len06_n2": "6-base x2", "marker_len06_n8": "6-base x8"}


def arm_of(run):
    return run.split("_seed")[0]


def data_dir(project, arm):
    sub = ("stage1_pilot" if arm in G.ARMS else "stage1_spacing" if arm in G.SPACING_ARMS else
           "stage1_middle" if arm in G.MIDDLE_ARMS else "stage1_markers")
    return os.path.join(project, "data", sub)


def marker_mask(arm):
    """Boolean list over the 128 strand positions: True where the arm has a marker base."""
    _, _, pos = G.build_strand(arm, "A" * STRAND)
    m = [False] * STRAND
    for a, b in pos:
        for i in range(a, b):
            m[i] = True
    return m


def payload_to_strand_positions(arm):
    mm = marker_mask(arm)
    return [i for i in range(STRAND) if not mm[i]]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


# ===========================================================================
# Part 1: statistics and final graphs
# ===========================================================================
def part_stats(args):
    out = os.path.join(args.out, "stats")
    os.makedirs(out, exist_ok=True)
    runs = sorted(d for d in os.listdir(args.results)
                  if os.path.isfile(os.path.join(args.results, d, "predictions.csv"))
                  and arm_of(d) in G.ARM_SPECS)
    print("Runs found (%d): %s" % (len(runs), ", ".join(runs)))
    preds = {}
    for run in runs:
        rows = read_csv(os.path.join(args.results, run, "predictions.csv"))
        by = {}
        for r in rows:
            by.setdefault(float(r["deletion_rate"]), {})[r["pair_id"]] = r
        preds[run] = by

    # ---- results_all.csv
    allrows = []
    for run in runs:
        for d in LEVELS:
            rr = list(preds[run].get(d, {}).values())
            if not rr:
                continue
            n = len(rr)
            k = sum(int(r["exact"]) for r in rr)
            plen = int(rr[0]["payload_length"])
            lo, hi = wilson(k, n)
            allrows.append({"run": run, "arm": arm_of(run), "deletion_rate": d, "n": n, "data_bases": plen,
                            "overhead": 1 - plen / STRAND, "exact": k / n, "exact_ci_low": lo, "exact_ci_high": hi,
                            "ned": float(np.mean([float(r["ned"]) for r in rr])),
                            "data_recovered_per_1000": k * plen * 1000 / n})
    write_csv(os.path.join(out, "results_all.csv"), allrows)

    # ---- paired tests
    tests = []
    rng = np.random.default_rng(0)
    for a, b, what in COMPARISONS:
        if a not in preds or b not in preds:
            continue
        for d in (0.05, 0.08):
            pa, pb = preds[a].get(d, {}), preds[b].get(d, {})
            ids = sorted(set(pa) & set(pb))
            if not ids:
                continue
            ea = np.array([int(pa[i]["exact"]) for i in ids])
            eb = np.array([int(pb[i]["exact"]) for i in ids])
            la, lb = int(pa[ids[0]]["payload_length"]), int(pb[ids[0]]["payload_length"])
            bb, cc = int(((ea == 1) & (eb == 0)).sum()), int(((ea == 0) & (eb == 1)).sum())
            diff = ea * la - eb * lb                      # data bases recovered, per strand
            boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B_BOOT)])
            tests.append({"A": a, "B": b, "comparison": what, "deletion_rate": d, "n_pairs": len(ids),
                          "exact_A": ea.mean(), "exact_B": eb.mean(), "only_A_correct": bb, "only_B_correct": cc,
                          "mcnemar_p": mcnemar_exact(bb, cc),
                          "data_diff_per_1000": diff.mean() * 1000,
                          "data_diff_ci_low": np.percentile(boot, 2.5) * 1000,
                          "data_diff_ci_high": np.percentile(boot, 97.5) * 1000,
                          "data_diff_pct_of_B": diff.mean() / max(1e-9, (eb * lb).mean()) * 100})
    write_csv(os.path.join(out, "paired_tests.csv"), tests)

    # ---- graphs per study
    plt = plt_setup()
    res = {(r["run"], r["deletion_rate"]): r for r in allrows}
    for study, arms in STUDIES.items():
        arms = [a for a in arms if a in preds]
        if len(arms) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        rows_csv = []
        for ai, a in enumerate(arms):
            dodge = (ai - (len(arms) - 1) / 2) * 0.07
            xs = [d * 100 + dodge for d in LEVELS if (a, d) in res]
            ex = [res[(a, d)]["exact"] * 100 for d in LEVELS if (a, d) in res]
            lo = [res[(a, d)]["exact_ci_low"] * 100 for d in LEVELS if (a, d) in res]
            hi = [res[(a, d)]["exact_ci_high"] * 100 for d in LEVELS if (a, d) in res]
            ned = [res[(a, d)]["ned"] for d in LEVELS if (a, d) in res]
            dr = [res[(a, d)]["data_recovered_per_1000"] / 1000 for d in LEVELS if (a, d) in res]
            kw = dict(marker="o", ms=3.5, lw=1.6 if a != "marker_free" else 2.2,
                      ls="--" if a == "marker_free" else "-", label=LABEL.get(a, a))
            axes[0].errorbar(xs, ex, yerr=[np.subtract(ex, lo), np.subtract(hi, ex)], capsize=2, **kw)
            axes[1].plot(xs, ned, **kw)
            axes[2].plot(xs, dr, **kw)
            for d, e, n_, r_ in zip(xs, ex, ned, dr):
                rows_csv.append({"study": study, "arm": a, "deletion_pct": round(d), "exact_pct": e, "ned": n_,
                                 "data_bases_per_strand": r_})
        axes[0].set(title="Strands fully recovered", xlabel="Deletion rate (%)", ylabel="Exact recovery (%)",
                    xticks=[1, 3, 5, 8])
        axes[1].set(title="Letter error (NED, lower is better)", xlabel="Deletion rate (%)", ylabel="NED",
                    xticks=[1, 3, 5, 8])
        axes[2].set(title="Data recovered per strand", xlabel="Deletion rate (%)",
                    ylabel="Data bases recovered / strand", xticks=[1, 3, 5, 8])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(7, len(l)), frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle("Fine-tuned DNAFormer, %s study (1,000 test strands per point, 95%% CI)" % study, fontsize=10, y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_%s_vs_deletion.png" % study))
        plt.close(fig)
        write_csv(os.path.join(out, "fig_%s_vs_deletion.csv" % study), rows_csv)

    # ---- error position along the strand (8% deletion)
    for study, arms in STUDIES.items():
        arms = [a for a in arms if a in preds and 0.08 in preds[a]]
        if not arms:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.2))
        rows_csv = []
        for a in arms:
            rr = list(preds[a][0.08].values())
            pos = payload_to_strand_positions(a)
            err = np.zeros(STRAND)
            cnt = np.zeros(STRAND)
            for r in rr:
                t, p = r["true_payload"], r["pred_payload"]
                for i, s in enumerate(pos):
                    cnt[s] += 1
                    err[s] += t[i] != p[i]
            rate = np.where(cnt > 0, err / np.maximum(cnt, 1), np.nan)
            ax.plot(range(1, STRAND + 1), rate * 100, lw=1.3 if a != "marker_free" else 2,
                    ls="--" if a == "marker_free" else "-", label=LABEL.get(a, a))
            for s in range(STRAND):
                rows_csv.append({"arm": a, "strand_position": s + 1,
                                 "error_pct": None if np.isnan(rate[s]) else rate[s] * 100})
        ref = {"spacing": "marker_len06_n8", "lengths": "marker_len06"}.get(study, "marker_periodic")
        mm = marker_mask(ref)
        for s in range(STRAND):
            if mm[s]:
                ax.axvspan(s + 0.5, s + 1.5, color="#dddddd", lw=0)
        ax.set(xlabel="Position in the 128-base strand (gaps = a version's own marker letters, not scored)",
               ylabel="Wrong letters (%)", xlim=(1, STRAND),
               title="Where the errors are at 8%% deletion, %s study (grey = marker positions of %s)"
                     % (study, LABEL.get(ref, ref)))
        ax.legend(fontsize=7, frameon=False, ncol=3)
        fig.savefig(os.path.join(out, "error_position_%s.png" % study))
        plt.close(fig)
        write_csv(os.path.join(out, "error_position_%s.csv" % study), rows_csv)

    # ---- print summary
    print("\nPAIRED TESTS (McNemar on the same test strands; data difference with 95% bootstrap CI)")
    print("%-42s %4s  %7s %7s  %10s  %22s" % ("comparison", "del", "A", "B", "McNemar p", "data diff / 1000 strands"))
    for t in tests:
        print("%-42s %3.0f%%  %6.1f%% %6.1f%%  %10.2g  %+8.0f [%+.0f, %+.0f]" % (
            t["comparison"][:42], t["deletion_rate"] * 100, t["exact_A"] * 100, t["exact_B"] * 100,
            t["mcnemar_p"], t["data_diff_per_1000"], t["data_diff_ci_low"], t["data_diff_ci_high"]))
    print("\nSaved to %s" % out)
    print("STAGE 5 STATS DONE")


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


# ===========================================================================
# Shared GPU helpers
# ===========================================================================
def load_finetuned(args, run, device):
    import torch
    model, _ = A.build_model(args.repo, device)
    path = os.path.join(args.project, "checkpoints", "stage4", run, "best.pt")
    if not os.path.isfile(path):
        sys.exit("ERROR: fine-tuned model not found: %s" % path)
    st = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(st["model"], strict=True)
    model.eval()
    return model, st.get("epoch")


def load_test(project, arm, level, limit):
    with open(os.path.join(data_dir(project, arm), arm, "test_" + G.level_tag(level), "metadata.json")) as f:
        cl = json.load(f)
    return cl[:limit] if limit else cl


def predict(torch, model, arm, clusters, device, bs=32):
    out = []
    with torch.inference_mode():
        for i in range(0, len(clusters), bs):
            x, _ = A.make_batch(clusters[i:i + bs], "relaxed", device)
            out += [G.remove_markers(arm, p) for p in A.decode(model(x)["pred"])]
    return out


# ===========================================================================
# Part 2: marker ablation
# ===========================================================================
def scrambled_cluster(CG, arm, c, seed):
    """Same payload and noise seed, but every marker base replaced by a different random base."""
    rng = random.Random(seed)
    clean = list(c["clean_strand"])
    for a, b in c["marker_positions"]:
        for i in range(a, b):
            clean[i] = rng.choice([x for x in "ACGT" if x != clean[i]])
    clean = "".join(clean)
    reads = G.simulate_reads(CG, clean, c["deletion_rate"], c["noise_seed"])
    return dict(c, clean_strand=clean, noisy_reads=reads)


def part_ablation(args):
    import torch
    out = os.path.join(args.out, "ablation")
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CG = G.load_simulator(args.repo)
    rows = []
    for run in args.arms:
        arm = arm_of(run)
        if G.ARM_SPECS[arm]["layout"] == "none":
            continue
        model, ep = load_finetuned(args, run, device)
        for d in (0.05, 0.08):
            cl = load_test(args.project, arm, d, args.limit)
            # sanity: re-simulating the unchanged strand must give the stored reads exactly
            chk = [G.simulate_reads(CG, c["clean_strand"], c["deletion_rate"], c["noise_seed"]) == c["noisy_reads"]
                   for c in cl[:20]]
            if not all(chk):
                sys.exit("ERROR: re-simulation does not reproduce the stored reads; ablation would be unfair")
            t0 = time.time()
            scr = [scrambled_cluster(CG, arm, c, 7 + k) for k, c in enumerate(cl)]
            p_norm = predict(torch, model, arm, cl, device)
            p_scr = predict(torch, model, arm, scr, device)
            en = np.array([p == c["payload_strand"] for p, c in zip(p_norm, cl)], dtype=int)
            es = np.array([p == c["payload_strand"] for p, c in zip(p_scr, cl)], dtype=int)
            nn_ = np.mean([A.edit_distance(p, c["payload_strand"]) / len(c["payload_strand"]) for p, c in zip(p_norm, cl)])
            ns = np.mean([A.edit_distance(p, c["payload_strand"]) / len(c["payload_strand"]) for p, c in zip(p_scr, cl)])
            bb, cc = int(((en == 1) & (es == 0)).sum()), int(((en == 0) & (es == 1)).sum())
            r = {"run": run, "deletion_rate": d, "n": len(cl), "exact_normal": en.mean(), "exact_scrambled": es.mean(),
                 "drop_points": (en.mean() - es.mean()) * 100, "ned_normal": nn_, "ned_scrambled": ns,
                 "mcnemar_p": mcnemar_exact(bb, cc), "best_epoch": ep, "seconds": time.time() - t0}
            rows.append(r)
            print("  %-22s %2.0f%%: exact %5.1f%% -> %5.1f%% with markers scrambled (drop %+.1f pts, p=%.2g), "
                  "NED %.4f -> %.4f" % (run, d * 100, r["exact_normal"] * 100, r["exact_scrambled"] * 100,
                                        -r["drop_points"], r["mcnemar_p"], nn_, ns), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_csv(os.path.join(out, "ablation.csv"), rows)
    if rows:
        plt = plt_setup()
        r8 = [r for r in rows if r["deletion_rate"] == 0.08]
        fig, ax = plt.subplots(figsize=(max(4, 1.3 * len(r8)), 3.2))
        x = np.arange(len(r8))
        ax.bar(x - 0.18, [r["exact_normal"] * 100 for r in r8], 0.36, label="markers intact", color="#2a6fd6")
        ax.bar(x + 0.18, [r["exact_scrambled"] * 100 for r in r8], 0.36, label="markers scrambled", color="#bbbbbb")
        ax.set_xticks(x, [LABEL.get(arm_of(r["run"]), r["run"]) for r in r8], rotation=20, ha="right")
        ax.set(ylabel="Strands fully recovered (%)", title="Marker ablation at 8% deletion")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(out, "ablation_8pct.png"))
        plt.close(fig)
    print("\nSaved to %s\nSTAGE 5 ABLATION DONE" % out)


# ===========================================================================
# Part 3: XAI
# ===========================================================================
def align_read_to_strand(strand, read):
    """Edit-distance alignment; returns, for each read position, the strand position it aligns to (or -1)."""
    n, m = len(strand), len(read)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        si = strand[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (si != read[j - 1]))
    mp = [-1] * m
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (strand[i - 1] != read[j - 1]):
            mp[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return mp


def read_importance(att, reads):
    """att: [2, 16, 4, 132] attribution for one cluster (left branch, reversed right branch).
    Returns per read a list of |importance| per read position (both branches combined)."""
    a = np.abs(att.sum(axis=2))            # [2, 16, 132]
    out = []
    for i, r in enumerate(reads[:A.MAX_READS]):
        L = min(len(r), A.INPUT_LEN)
        left = a[0, i, :L]
        right = a[1, i, :L][::-1]          # right branch holds the reversed read
        out.append(left + right)
    return out


def strand_profile(imp_reads, maps):
    prof = np.zeros(STRAND)
    for imp, mp in zip(imp_reads, maps):
        for j, v in enumerate(imp):
            if j < len(mp) and mp[j] >= 0:
                prof[mp[j]] += v
    return prof


def part_xai(args):
    import torch
    out = os.path.join(args.out, "xai")
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plt = plt_setup()
    summary = []
    heatmaps = {}
    for run in args.arms:
        arm = arm_of(run)
        model, ep = load_finetuned(args, run, device)
        for p in model.parameters():
            p.requires_grad_(False)
        cl = load_test(args.project, arm, 0.08, args.n_xai)
        mm = np.array(marker_mask(arm))
        ref_mm = mm if mm.any() else np.array(marker_mask("marker_len06"))   # placebo positions for no markers
        pay_out = [s for s in range(STRAND) if not mm[s]]                    # output positions holding data
        t0 = time.time()
        H = np.zeros((STRAND, STRAND))
        shares, local = [], []
        K = args.k_batch
        band = np.array([[abs(o - s) <= 8 for s in range(STRAND)] for o in pay_out])
        for c in cl:
            x, y = A.make_batch([c], "relaxed", device)                      # x [2, 16, 4, 132]
            maps = [align_read_to_strand(c["clean_strand"], r[:A.INPUT_LEN]) for r in c["noisy_reads"]]
            Hc = np.zeros((STRAND, STRAND))
            for s in range(0, STRAND, K):
                tg = list(range(s, min(STRAND, s + K)))
                k = len(tg)
                X = torch.cat([x[0:1].repeat(k, 1, 1, 1), x[1:2].repeat(k, 1, 1, 1)]).clone().requires_grad_(True)
                with torch.enable_grad():
                    pred = model(X)["pred"]                                   # [k, 4, 128]
                    score = sum(pred[i, int(y[0, j]), j] for i, j in enumerate(tg))
                    g, = torch.autograd.grad(score, X)
                gx = (g * X).detach().float().cpu().numpy()                   # gradient x input
                for i, j in enumerate(tg):
                    att = np.stack([gx[i], gx[k + i]])                       # [2, 16, 4, 132]
                    prof = strand_profile(read_importance(att, c["noisy_reads"]), maps)
                    Hc[j] = prof / max(prof.sum(), 1e-12)
            H += Hc
            rows_pay = Hc[pay_out]
            shares.append(rows_pay[:, ref_mm].sum(axis=1).mean())
            local.append((rows_pay * band).sum(axis=1).mean())
        H /= len(cl)
        heatmaps[run] = (H, mm)
        np.save(os.path.join(out, "heatmap_%s.npy" % run), H)

        # ---- Integrated Gradients and GradientSHAP on fewer strands, 8 output segments
        ig_sh, gs_sh = [], []
        try:
            from captum.attr import GradientShap, IntegratedGradients
            segs = [pay_out[k * len(pay_out) // 8:(k + 1) * len(pay_out) // 8] for k in range(8)]

            def fwd(z, yy):
                p = model(torch.cat([z[:, 0], z[:, 1]]))["pred"]               # [N, 4, 128]
                sc = p.gather(1, yy.unsqueeze(1).expand(p.shape[0], 1, STRAND)).squeeze(1)  # [N, 128]
                return torch.stack([sc[:, sg].sum(1) for sg in segs], 1)      # [N, 8]
            for c in cl[:args.n_ig]:
                x, y = A.make_batch([c], "relaxed", device)
                z = torch.stack([x[0], x[1]]).unsqueeze(0)                    # [1, 2, 16, 4, 132]
                maps = [align_read_to_strand(c["clean_strand"], r[:A.INPUT_LEN]) for r in c["noisy_reads"]]
                ig = IntegratedGradients(lambda zz: fwd(zz, y.expand(zz.shape[0], -1)))
                gs = GradientShap(lambda zz: fwd(zz, y.expand(zz.shape[0], -1)))
                for t in range(8):
                    with torch.enable_grad():
                        a1 = ig.attribute(z, baselines=torch.zeros_like(z), target=t, n_steps=args.ig_steps,
                                          internal_batch_size=8)
                        a2 = gs.attribute(z, baselines=torch.zeros_like(z), target=t,
                                          n_samples=min(args.ig_steps, 16), stdevs=0.0)
                    for a_, lst in ((a1, ig_sh), (a2, gs_sh)):
                        prof = strand_profile(read_importance(a_[0].detach().float().cpu().numpy(), c["noisy_reads"]), maps)
                        lst.append(prof[ref_mm].sum() / max(prof.sum(), 1e-12))
        except ImportError:
            print("  (captum not installed: skipping Integrated Gradients / GradientSHAP)")

        # ---- attention share to marker positions
        att_share = attention_share(torch, model, cl[:args.n_att], ref_mm, pay_out, device)
        r = {"run": run, "best_epoch": ep, "n_strands": len(cl),
             "marker_space_share": ref_mm.mean(), "placebo_positions": not mm.any(),
             "gradxinput_marker_share": float(np.mean(shares)), "gradxinput_local_share_pm8": float(np.mean(local)),
             "ig_marker_share": float(np.mean(ig_sh)) if ig_sh else None,
             "gradshap_marker_share": float(np.mean(gs_sh)) if gs_sh else None,
             "attention_marker_share": att_share, "seconds": time.time() - t0}
        r["enrichment_gradxinput"] = r["gradxinput_marker_share"] / r["marker_space_share"]
        summary.append(r)
        print("  %-22s marker positions = %4.1f%% of strand%s | importance on them: grad x input %4.1f%% "
              "(x%.2f), IG %s, GradSHAP %s, attention %s | %.0fs" % (
                  run, r["marker_space_share"] * 100, " (placebo: no markers)" if not mm.any() else "",
                  r["gradxinput_marker_share"] * 100, r["enrichment_gradxinput"],
                  "%.1f%%" % (r["ig_marker_share"] * 100) if ig_sh else "-",
                  "%.1f%%" % (r["gradshap_marker_share"] * 100) if gs_sh else "-",
                  "%.1f%%" % (att_share * 100) if att_share is not None else "-", r["seconds"]), flush=True)
        if run == args.arms[0] or mm.any():
            example_figure(plt, torch, model, arm, run, cl[0], device, out)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_csv(os.path.join(out, "xai_summary.csv"), summary)
    # heatmaps
    n = len(heatmaps)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3), squeeze=False)
        for ax, (run, (H, mm)) in zip(axes[0], heatmaps.items()):
            ax.imshow(H, cmap="magma", origin="upper", vmax=np.percentile(H, 99.5), aspect="equal")
            for s in np.where(mm)[0]:
                ax.plot([s, s], [-6, -1.5], color="#e8590c", lw=1.2, clip_on=False)
            ax.set_ylim(STRAND - 0.5, -7)
            ax.set(title=LABEL.get(arm_of(run), run), xlabel="Strand position read by the model", ylabel="Output letter")
        fig.suptitle("Where the fine-tuned model looks (gradient x input, 8% deletion; orange ticks = marker positions)",
                     fontsize=9)
        fig.savefig(os.path.join(out, "heatmaps.png"))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(summary)), 3.2))
        x = np.arange(len(summary))
        ax.bar(x - 0.2, [r["marker_space_share"] * 100 for r in summary], 0.4, color="#cccccc", label="share of strand")
        ax.bar(x + 0.2, [r["gradxinput_marker_share"] * 100 for r in summary], 0.4, color="#2a6fd6",
               label="share of model's attention (grad x input)")
        ax.set_xticks(x, [LABEL.get(arm_of(r["run"]), r["run"]) for r in summary], rotation=20, ha="right")
        ax.set(ylabel="%", title="Marker positions: space they take vs importance the model gives them")
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(os.path.join(out, "marker_share.png"))
        plt.close(fig)
    print("\nSaved to %s\nSTAGE 5 XAI DONE" % out)


def attention_share(torch, model, clusters, ref_mm, pay_out, device):
    """Average share of self-attention (all layers, heads averaged) that data positions give to marker positions."""
    store = []
    origs = []
    for layer in model.encoder.layers:
        mha = layer.self_attn
        orig = mha.forward

        def patched(*a, _orig=orig, **kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            o = _orig(*a, **kw)
            store.append(o[1].detach())
            return o
        mha.forward = patched
        origs.append((mha, orig))
    shares = []
    try:
        for c in clusters:
            x, _ = A.make_batch([c], "relaxed", device)
            x = x.clone().requires_grad_(True)       # keeps the transformer off its fused fast path
            store.clear()
            with torch.enable_grad():
                model(x)
            if not store:
                return None
            for w in store:                          # [2, 128, 128] (left and right branch)
                w = w.float().cpu().numpy().mean(0)
                shares.append(w[pay_out][:, ref_mm].sum(1).mean())
    finally:
        for mha, orig in origs:
            mha.forward = orig
    return float(np.mean(shares)) if shares else None


def example_figure(plt, torch, model, arm, run, c, device, out):
    """One read with its letters shaded by importance (gradient x input summed over all output letters)."""
    x, y = A.make_batch([c], "relaxed", device)
    X = x.clone().requires_grad_(True)
    with torch.enable_grad():
        p = model(X)["pred"]
        score = p[0].gather(0, y[0].unsqueeze(0)).sum()
        g, = torch.autograd.grad(score, X)
    att = (g * X).detach().float().cpu().numpy()
    imp = read_importance(att, c["noisy_reads"])[0]
    read = c["noisy_reads"][0][:len(imp)]
    mp = align_read_to_strand(c["clean_strand"], read)
    mm = marker_mask(arm)
    v = imp / max(imp.max(), 1e-12)
    per_line = 66
    lines = math.ceil(len(read) / per_line)
    fig, ax = plt.subplots(figsize=(10, 0.55 * lines + 0.8))
    ax.axis("off")
    for j, ch in enumerate(read):
        r_, c_ = divmod(j, per_line)
        is_m = mp[j] >= 0 and mm[mp[j]]
        ax.add_patch(plt.Rectangle((c_, -r_ - 0.9), 0.95, 0.85, color=plt.cm.Blues(0.15 + 0.85 * v[j])))
        ax.text(c_ + 0.47, -r_ - 0.47, ch, ha="center", va="center", fontsize=7, family="monospace",
                color="white" if v[j] > 0.6 else "black", fontweight="bold" if is_m else "normal")
        if is_m:
            ax.plot([c_, c_ + 0.95], [-r_ - 0.95, -r_ - 0.95], color="#e8590c", lw=1.5)
    ax.set_xlim(0, per_line)
    ax.set_ylim(-lines - 0.1, 0.1)
    ax.set_title("%s: one noisy read, darker = the model relied on it more (orange underline = marker letters)"
                 % LABEL.get(arm, arm), fontsize=8)
    fig.savefig(os.path.join(out, "example_read_%s.png" % run))
    plt.close(fig)


# ===========================================================================
# Part 4: one worked example (same test strand, two models), saved as data for a figure
# ===========================================================================
def explain_cluster(torch, model, c, device):
    """Importance of every read letter (gradient x input over all output letters) + the model's raw output."""
    x, y = A.make_batch([c], "relaxed", device)
    X = x.clone().requires_grad_(True)
    with torch.enable_grad():
        out = model(X)["pred"]
        score = out[0].gather(0, y[0].unsqueeze(0)).sum()
        g, = torch.autograd.grad(score, X)
    att = (g * X).detach().float().cpu().numpy()
    pred = A.decode(out.detach())[0]
    imp = read_importance(att, c["noisy_reads"])
    reads = []
    for i, r in enumerate(c["noisy_reads"][:A.MAX_READS]):
        rr = r[:len(imp[i])]
        reads.append({"read": rr, "importance": [round(float(v), 6) for v in imp[i]],
                      "strand_pos": align_read_to_strand(c["clean_strand"], rr)})
    return pred, reads


def part_example(args):
    import torch
    out = os.path.join(args.out, "example")
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(args.arms) != 2:
        sys.exit("ERROR: --part example needs exactly two runs, e.g. marker_free,marker_len06_n1")
    level = 0.08
    n = args.limit or 300
    runs = args.arms
    models, tests, preds = {}, {}, {}
    for run in runs:
        models[run], ep = load_finetuned(args, run, device)
        tests[run] = load_test(args.project, arm_of(run), level, n)
        preds[run] = predict(torch, models[run], arm_of(run), tests[run], device)
        ok = sum(p == c["payload_strand"] for p, c in zip(preds[run], tests[run]))
        print("  %-18s best epoch %s | %d/%d of the first test strands at 8%% fully recovered" % (run, ep, ok, n))
    a, b = runs
    cands = []
    for i, (ca, cb) in enumerate(zip(tests[a], tests[b])):
        assert ca["pair_id"] == cb["pair_id"]
        if preds[a][i] == ca["payload_strand"] or preds[b][i] != cb["payload_strand"]:
            continue
        with torch.inference_mode():
            x, _ = A.make_batch([ca], "relaxed", device)
            raw = A.decode(models[a](x)["pred"])[0]
        wrong = [k for k in range(STRAND) if raw[k] != ca["clean_strand"][k]]
        mid = sum(30 <= k < 98 for k in wrong)
        cands.append((len(wrong) > 8, -mid / max(1, len(wrong)), len(wrong), i))
    cands.sort()
    print("  %d test strands: %s wrong, %s right" % (len(cands), a, b))
    chosen = [c[3] for c in cands[:3]]
    if not chosen:
        print("  NOTE: no such strand in the first %d; showing the first test strand instead" % n)
        chosen = [0]
    data = []
    for i in chosen:
        item = {"pair_id": tests[a][i]["pair_id"], "deletion_rate": level, "runs": {}}
        for run in runs:
            c = tests[run][i]
            pred, reads = explain_cluster(torch, models[run], c, device)
            item["runs"][run] = {"arm": arm_of(run), "clean_strand": c["clean_strand"], "prediction": pred,
                                 "marker_mask": marker_mask(arm_of(run)), "reads": reads,
                                 "payload_correct": G.remove_markers(arm_of(run), pred) == c["payload_strand"]}
        wa = sum(p != q for p, q in zip(item["runs"][a]["prediction"], item["runs"][a]["clean_strand"]))
        wb = sum(p != q for p, q in zip(item["runs"][b]["prediction"], item["runs"][b]["clean_strand"]))
        print("  example %s: %s gets %d strand letters wrong, %s gets %d wrong" % (item["pair_id"], a, wa, b, wb))
        data.append(item)
    with open(os.path.join(out, "example_data.json"), "w") as f:
        json.dump(data, f)
    print("\nSaved to %s\nSTAGE 5 EXAMPLE DONE" % out)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["stats", "ablation", "xai", "example"])
    ap.add_argument("--results", help="results/stage4 folder (stats)")
    ap.add_argument("--project", help="MyDrive/fyp_dna (ablation, xai)")
    ap.add_argument("--repo", help="Deep-DNA-based-storage (ablation, xai)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="marker_free,marker_endblock,marker_len06,marker_len06_n1,marker_len06_n8",
                    help="comma-separated runs (folder names under checkpoints/stage4)")
    ap.add_argument("--limit", type=int, default=0, help="test strands per level for ablation (0 = all 1,000)")
    ap.add_argument("--n-xai", type=int, default=100, help="test strands (8%% deletion) for gradient x input")
    ap.add_argument("--n-ig", type=int, default=20, help="test strands for Integrated Gradients / GradientSHAP")
    ap.add_argument("--n-att", type=int, default=50, help="test strands for attention")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--k-batch", type=int, default=16, help="output letters explained per backward pass")
    args = ap.parse_args()
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in args.arms if arm_of(a) not in G.ARM_SPECS]
    if bad:
        sys.exit("ERROR: unknown run(s) %s" % bad)
    os.makedirs(args.out, exist_ok=True)
    {"stats": part_stats, "ablation": part_ablation, "xai": part_xai, "example": part_example}[args.part](args)


if __name__ == "__main__":
    main()
