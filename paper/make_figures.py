"""Builds all paper figures (PDF) and the results table (LaTeX) from paper/data/. Run: python3 paper/make_figures.py"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fyp_dna"))
import stage1_generate as G

D, F = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, INK2, GRID, MARK = "#b9b8b3", "#0b0b0b", "#52514e", "#e6e5e1", "#e9e8e3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "legend.frameon": False, "pdf.fonttype": 42})


def mask(arm):
    _, _, pos = G.build_strand(arm, "A" * 128)
    m = np.zeros(128, bool)
    for a, b in pos:
        m[a:b] = True
    return m


def save(fig, name):
    fig.savefig(os.path.join(F, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


res = {}
for r in csv.DictReader(open(os.path.join(D, "results_all.csv"))):
    res[(r["run"], round(float(r["deletion_rate"]) * 100))] = r

# ---------------- Fig 1: strand layouts ----------------
layouts = [("No markers", "marker_free"), ("End-block (control)", "marker_endblock"),
           ("4 × 8-nt, spread", "marker_periodic"), ("1 × 6-nt, middle", "marker_len06_n1"),
           ("8 × 6-nt, spread", "marker_len06_n8")]
fig, ax = plt.subplots(figsize=(3.4, 1.45))
for i, (lab, arm) in enumerate(layouts):
    m = mask(arm)
    y = len(layouts) - 1 - i
    ax.add_patch(Rectangle((0, y + 0.18), 128, 0.64, color="#dbe7f7", lw=0))
    k = 0
    while k < 128:
        if m[k]:
            j = k
            while j < 128 and m[j]:
                j += 1
            ax.add_patch(Rectangle((k, y + 0.18), j - k, 0.64, color=ORANGE, lw=0))
            k = j
        else:
            k += 1
    ax.text(-3, y + 0.5, lab, ha="right", va="center", fontsize=7)
    ax.text(131, y + 0.5, "%d data nt" % (128 - m.sum()), ha="left", va="center", fontsize=6.5, color=INK2)
ax.set_xlim(0, 128); ax.set_ylim(0, len(layouts))
ax.set_xticks([1, 32, 64, 96, 128]); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", labelsize=6.5)
ax.set_xlabel("Position in the 128-nt strand (orange = marker)", fontsize=7)
save(fig, "fig_layouts.pdf")

# ---------------- Fig 2: recovery vs deletion + data at 8% ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
ax = axs[0]
series = [("No markers", "marker_free", BLUE, "--"), ("End-block", "marker_endblock", GREY, "-"),
          ("4 × 8-nt", "marker_periodic", ORANGE, "-"), ("1 × 6-nt, middle", "marker_len06_n1", AQUA, "-"),
          ("8 × 6-nt", "marker_len06_n8", MAGENTA, "-")]
for lab, run, c, ls in series:
    xs = [1, 3, 5, 8]
    ys = [float(res[(run, d)]["exact"]) * 100 for d in xs]
    lo = [float(res[(run, d)]["exact_ci_low"]) * 100 for d in xs]
    hi = [float(res[(run, d)]["exact_ci_high"]) * 100 for d in xs]
    ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=c, ls=ls, lw=1.5, marker="o",
                ms=3.5, capsize=2, label=lab)
ax.set_xticks([1, 3, 5, 8]); ax.set_xlabel("Deletion rate (%)")
ax.set_ylabel("Strands fully recovered (%)"); ax.set_ylim(30, 102)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, loc="lower left")
ax.set_title("(a) Exact strand recovery", loc="left")
ax = axs[1]
order = ["marker_endblock", "marker_len12", "marker_periodic", "marker_len05", "marker_len04", "marker_len07",
         "marker_len06_n2", "marker_len06", "marker_len06_n8", "marker_len06_n1"]
names = {"marker_endblock": "End-block", "marker_len12": "4 × 12", "marker_periodic": "4 × 8", "marker_len05": "4 × 5",
         "marker_len04": "4 × 4", "marker_len07": "4 × 7", "marker_len06_n2": "2 × 6", "marker_len06": "4 × 6",
         "marker_len06_n8": "8 × 6", "marker_len06_n1": "1 × 6 (mid)"}
base8 = float(res[("marker_free", 8)]["data_recovered_per_1000"])
base5 = float(res[("marker_free", 5)]["data_recovered_per_1000"])
v8 = [(float(res[(r, 8)]["data_recovered_per_1000"]) / base8 - 1) * 100 for r in order]
v5 = [(float(res[(r, 5)]["data_recovered_per_1000"]) / base5 - 1) * 100 for r in order]
y = np.arange(len(order))
ax.barh(y + 0.19, v8, 0.36, color=BLUE, label="8% deletion")
ax.barh(y - 0.19, v5, 0.36, color=GREY, label="5% deletion")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y, [names[r] for r in order], fontsize=6.8)
ax.set_xlabel("Data recovered vs. no markers (%)")
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, loc="lower right")
ax.set_title("(b) Net data recovered (markers cost space)", loc="left")
save(fig, "fig_recovery.pdf")


# ---------------- Fig 3: error position ----------------
def load_err(fn):
    d = {}
    for r in csv.DictReader(open(os.path.join(D, fn))):
        d.setdefault(r["arm"], np.full(128, np.nan))[int(r["strand_position"]) - 1] = \
            float(r["error_pct"]) if r["error_pct"] else np.nan
    return d

em, es = load_err("err_main.csv"), load_err("err_spacing.csv")
pos = np.arange(1, 129)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
panels = [(axs[0], "(a) Same 32 marker nt, different placement",
           [("No markers", em["marker_free"], BLUE, "--"), ("4 × 8-nt, spread", em["marker_periodic"], ORANGE, "-"),
            ("End-block", em["marker_endblock"], AQUA, "-")], "marker_periodic"),
          (axs[1], "(b) One vs. eight 6-nt markers",
           [("No markers", es["marker_free"], BLUE, "--"), ("1 × 6-nt, middle", es["marker_len06_n1"], ORANGE, "-"),
            ("8 × 6-nt", es["marker_len06_n8"], AQUA, "-")], "marker_len06_n8")]
for ax, title, ser, ref in panels:
    for i in np.where(mask(ref))[0]:
        ax.axvspan(i + 0.5, i + 1.5, color=MARK, lw=0)
    for lab, yy, c, ls in ser:
        ax.plot(pos, yy, color=c, lw=1.3, ls=ls, label=lab)
    ax.set_title(title, loc="left"); ax.set_xlabel("Strand position")
    ax.set_xlim(1, 128); ax.legend(fontsize=6.5, loc="upper right")
axs[0].set_ylabel("Wrong output nt at 8% (%)")
save(fig, "fig_errors.pdf")

# ---------------- Fig 4: ablation ----------------
abl = [("End-block\n(32 nt, end)", 40.5, 36.7), ("1 × 6-nt\n(middle)", 45.8, 33.8), ("4 × 6-nt", 49.6, 31.8),
       ("4 × 8-nt", 52.4, 30.4), ("8 × 6-nt", 69.6, 18.4)]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
x = np.arange(len(abl)); w = 0.38
b1 = ax.bar(x - w / 2 - 0.01, [a[1] for a in abl], w, color=BLUE, label="markers intact")
b2 = ax.bar(x + w / 2 + 0.01, [a[2] for a in abl], w, color=GREY, label="markers scrambled")
for i, a in enumerate(abl):
    ax.text(i, max(a[1], a[2]) + 2.5, "−%.1f" % (a[1] - a[2]), ha="center", fontsize=6.8, fontweight="bold")
ax.axhline(38.8, color=INK2, lw=0.8, ls=(0, (4, 3)), label="no-marker model")
ax.set_xticks(x, [a[0] for a in abl], fontsize=6.3)
ax.set_ylabel("Strands fully recovered at 8% (%)", fontsize=7)
ax.set_ylim(0, 85); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.2, loc="upper left", ncol=2)
save(fig, "fig_ablation.pdf")

# ---------------- Fig 5: heatmaps ----------------
hms = [("No markers", "hm_free.npy", None), ("End-block", "hm_endblock.npy", "marker_endblock"),
       ("4 × 8-nt", "hm_periodic.npy", "marker_periodic"), ("1 × 6-nt, middle", "hm_n1.npy", "marker_len06_n1"),
       ("8 × 6-nt", "hm_n8.npy", "marker_len06_n8")]
fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.75), gridspec_kw={"wspace": 0.3})
for ax, (title, fn, arm) in zip(axs, hms):
    H = np.load(os.path.join(D, fn))
    ax.imshow(H, cmap="Blues", norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(H, 99.7)))
    if arm:
        for i in np.where(mask(arm))[0]:
            ax.plot([i, i], [-8, -2], color=ORANGE, lw=0.8, clip_on=False)
    ax.set_ylim(127.5, -9); ax.set_title(title, fontsize=7.5, pad=5)
    ax.set_xticks([0, 127], ["1", "128"], fontsize=6); ax.set_yticks([0, 127], ["1", "128"], fontsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_ylabel("Output position", fontsize=7)
fig.text(0.5, -0.04, "Aligned read position", ha="center", fontsize=7)
save(fig, "fig_heatmaps.pdf")

# ---------------- Table: all designs ----------------
p8 = {"marker_endblock": "0.12", "marker_periodic": r"$6\times10^{-14}$", "marker_len04": "0.001",
      "marker_len05": r"$1\times10^{-4}$", "marker_len06": r"$8\times10^{-10}$", "marker_len07": r"$3\times10^{-10}$",
      "marker_len12": r"$3\times10^{-35}$", "marker_len06_n1": r"$1\times10^{-5}$", "marker_len06_n2": "0.003",
      "marker_len06_n8": r"$1\times10^{-56}$"}
rows = [("No markers", "marker_free", "--", "--"), ("End-block (control)", "marker_endblock", "ACGTCAGT", r"$4\times8$, end"),
        (r"Spread, 8-nt", "marker_periodic", "ACGTCAGT", r"$4\times(24+8)$"),
        ("Spread, 4-nt", "marker_len04", "CAGT", r"$4\times(28+4)$"), ("Spread, 5-nt", "marker_len05", "TCATG", r"$4\times(27+5)$"),
        ("Spread, 6-nt", "marker_len06", "CTGTCA", r"$4\times(26+6)$"), ("Spread, 7-nt", "marker_len07", "CTGATCA", r"$4\times(25+7)$"),
        ("Spread, 12-nt", "marker_len12", "ACGTCAGTGCAT", r"$4\times(20+12)$"),
        ("Middle, 1 marker", "marker_len06_n1", "CTGTCA", r"$61+6+61$"), ("Spread, 2 markers", "marker_len06_n2", "CTGTCA", r"$2\times(58+6)$"),
        ("Spread, 8 markers", "marker_len06_n8", "CTGTCA", r"$8\times(10+6)$")]
out = []
for name, run, mk, lay in rows:
    r5, r8 = res[(run, 5)], res[(run, 8)]
    dn = int(r8["data_bases"])
    e5 = float(r5["exact"]) * 100; e8 = float(r8["exact"]) * 100
    lo, hi = float(r8["exact_ci_low"]) * 100, float(r8["exact_ci_high"]) * 100
    d8 = (float(r8["data_recovered_per_1000"]) / base8 - 1) * 100
    d5 = (float(r5["data_recovered_per_1000"]) / base5 - 1) * 100
    pv = p8.get(run, "--")
    ds8 = "--" if run == "marker_free" else "%+.1f" % d8
    ds5 = "--" if run == "marker_free" else "%+.1f" % d5
    out.append(r"%s & \texttt{\footnotesize %s} & %s & %d & %.1f & %.1f\,[%.1f, %.1f] & %s & %.4f & %s & %s \\" %
               (name, mk if mk != "--" else "--", lay, dn, e5, e8, lo, hi, pv, float(r8["ned"]), ds8, ds5))
HEAD = (r"\begin{tabular}{llccccccrr}" "\n" r"\hline" "\n"
        r"Design & Marker & Layout (data + marker) & Data nt & Exact 5\% & Exact 8\% & $p$ (8\%) & NED 8\% & $\Delta D$ 8\% & $\Delta D$ 5\% \\" "\n" r"\hline" "\n")
import re
open(os.path.join(HERE, "table_results.tex"), "w").write(HEAD + re.sub(r"& -(\d)", r"& $-$\1", "\n".join(out)) + "\n\\hline\n\\end{tabular}\n")
print("figures and table written")

# ---------------- Extra (IEEE) Fig: pipeline ----------------
fig, ax = plt.subplots(figsize=(7.0, 1.25))
ax.axis("off")
boxes = [("Random\npayload", "#dbe7f7"), ("Insert\nmarkers", "#fbe0d4"), ("Simulate 10 reads\n(del 1–8%, ins/sub 1%)", "#eeeeea"),
         ("DNAFormer\n(fine-tuned)", "#dbe7f7"), ("Remove\nmarkers", "#fbe0d4"), ("Score data nt\n+ XAI", "#eeeeea")]
wbox, gap = 1.0, 0.22
for i, (t, c) in enumerate(boxes):
    x0 = i * (wbox + gap)
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x0, 0.1), wbox, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                                   fc=c, ec=INK2, lw=0.7))
    ax.text(x0 + wbox / 2, 0.5, t, ha="center", va="center", fontsize=7)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x0 + wbox + gap - 0.02, 0.5), xytext=(x0 + wbox + 0.02, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
ax.set_xlim(-0.05, len(boxes) * (wbox + gap)); ax.set_ylim(0, 1)
save(fig, "fig_pipeline.pdf")

# ---------------- Extra (IEEE) Fig: length and number ----------------
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=True, gridspec_kw={"wspace": 0.12})
L = [(4, "marker_len04"), (5, "marker_len05"), (6, "marker_len06"), (7, "marker_len07"), (8, "marker_periodic"), (12, "marker_len12")]
N = [(1, "marker_len06_n1"), (2, "marker_len06_n2"), (4, "marker_len06"), (8, "marker_len06_n8")]
for ax, pts, xl, tt in [(axs[0], L, "Marker length (nt), 4 markers", "(a) Length"), (axs[1], N, "Number of 6-nt markers", "(b) Number")]:
    xs = [p[0] for p in pts]
    ex = [float(res[(r, 8)]["exact"]) * 100 for _, r in pts]
    lo = [float(res[(r, 8)]["exact_ci_low"]) * 100 for _, r in pts]
    hi = [float(res[(r, 8)]["exact_ci_high"]) * 100 for _, r in pts]
    ax.errorbar(xs, ex, yerr=[np.subtract(ex, lo), np.subtract(hi, ex)], color=BLUE, marker="o", ms=3, lw=1.3, capsize=2)
    ax.axhline(38.8, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(xs); ax.set_xlabel(xl, fontsize=6.8); ax.set_title(tt, loc="left", fontsize=7.5)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=6.5)
axs[0].set_ylabel("Exact recovery at 8% (%)", fontsize=6.8)
axs[0].text(4, 40.5, "no markers", fontsize=6, color=INK2)
save(fig, "fig_length_number.pdf")
print("extra figures written")

# ---------------- Short table for LNCS ----------------
short = [r"\begin{tabular}{lrrrrrr}", r"\hline",
         r"Design & Data nt & Exact 5\% & Exact 8\% [95\% CI] & $p$ (8\%) & $\Delta D$ 8\% & $\Delta D$ 5\% \\", r"\hline"]
for name, run, mk, lay in rows:
    r5, r8 = res[(run, 5)], res[(run, 8)]
    d8 = (float(r8["data_recovered_per_1000"]) / base8 - 1) * 100
    d5 = (float(r5["data_recovered_per_1000"]) / base5 - 1) * 100
    nm = name if run == "marker_free" else "%s (%s)" % (name, lay)
    short.append(r"%s & %d & %.1f & %.1f\,[%.1f, %.1f] & %s & %s & %s \\" % (
        nm, int(r8["data_bases"]), float(r5["exact"]) * 100, float(r8["exact"]) * 100,
        float(r8["exact_ci_low"]) * 100, float(r8["exact_ci_high"]) * 100, p8.get(run, "--"),
        "--" if run == "marker_free" else "%+.1f" % d8, "--" if run == "marker_free" else "%+.1f" % d5))
short += [r"\hline", r"\end{tabular}"]
open(os.path.join(HERE, "table_results_short.tex"), "w").write(re.sub(r"& -(\d)", r"& $-$\1", "\n".join(short)) + "\n")
print("short table written")
