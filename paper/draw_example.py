import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PAIR = sys.argv[1] if len(sys.argv) > 1 else "test_del08_000005"
LO, HI = 40, 76                      # strand window (0-based, HI exclusive)
INK, INK2 = "#0b0b0b", "#52514e"
RED, REDBG, GREENBG, ORANGE = "#b3261e", "#f6d3d0", "#d8efdf", "#eb6834"
LABEL = {"marker_free": "No markers", "marker_len06_n1": "1 × 6-nt marker (middle)", "marker_len06_n8": "8 markers"}

data = {it["pair_id"]: it for it in json.load(open(os.path.join(HERE, "data", "example_data.json")))}[PAIR]
runs = list(data["runs"])
cmap = plt.cm.Blues
W = HI - LO
nrows = 10 + 2


def panel(ax, run):
    r = data["runs"][run]
    cs, pr, mm = r["clean_strand"], r["prediction"], r["marker_mask"]
    # importance per read, per strand column (deleted = None)
    grid = []
    vmax = 0
    for rd in r["reads"][:10]:
        row = {}
        for j, k in enumerate(rd["strand_pos"]):
            if k >= 0:
                row[k] = (rd["read"][j], rd["importance"][j])
                if LO <= k < HI:
                    vmax = max(vmax, rd["importance"][j])
        grid.append(row)
    wrong = [k for k in range(LO, HI) if pr[k] != cs[k]]
    # true strand row
    y = 0
    ax.text(-0.6, y + 0.5, "True", ha="right", va="center", fontsize=6, color=INK, fontweight="bold")
    for c, k in enumerate(range(LO, HI)):
        ax.text(c + 0.5, y + 0.5, cs[k], ha="center", va="center", fontsize=5.6, family="DejaVu Sans Mono",
                color=ORANGE if mm[k] else INK, fontweight="bold")
        if mm[k]:
            ax.plot([c + 0.08, c + 0.92], [y + 0.97, y + 0.97], color=ORANGE, lw=1.6)
    # reads
    for i, row in enumerate(grid):
        y = 1.25 + i
        ax.text(-0.6, y + 0.5, "Read %d" % (i + 1), ha="right", va="center", fontsize=5.5, color=INK2)
        for c, k in enumerate(range(LO, HI)):
            if k in row:
                ch, v = row[k]
                t = min(1.0, v / vmax) if vmax else 0
                ax.add_patch(Rectangle((c + 0.04, y + 0.04), 0.92, 0.92, color=cmap(0.06 + 0.84 * t), lw=0))
                ax.text(c + 0.5, y + 0.5, ch, ha="center", va="center", fontsize=5.4, family="DejaVu Sans Mono",
                        color=("white" if t > 0.6 else (RED if ch != cs[k] else INK)),
                        fontweight="bold" if ch != cs[k] else "normal")
            else:
                ax.add_patch(Rectangle((c + 0.04, y + 0.04), 0.92, 0.92, color=REDBG, lw=0))
                ax.text(c + 0.5, y + 0.5, "–", ha="center", va="center", fontsize=5.5, color=RED, fontweight="bold")
    # output row
    y = 1.5 + 10
    ok = not wrong
    ax.text(-0.6, y + 0.5, "Output", ha="right", va="center", fontsize=6, color=INK, fontweight="bold")
    for c, k in enumerate(range(LO, HI)):
        bad = pr[k] != cs[k]
        ax.add_patch(Rectangle((c + 0.04, y + 0.04), 0.92, 0.92, color=RED if bad else GREENBG, lw=0))
        ax.text(c + 0.5, y + 0.5, pr[k], ha="center", va="center", fontsize=5.6, family="DejaVu Sans Mono",
                color="white" if bad else INK, fontweight="bold")
    for k in wrong:
        c = k - LO
        ax.add_patch(Rectangle((c, -0.05), 1, y + 1.1, fill=False, ec=RED, lw=1.4))
    ax.set_xlim(-0.2, W + 0.2)
    ax.set_ylim(y + 1.3, -0.4)
    ax.axis("off")
    verdict = ("1 wrong nt → strand lost" if len(wrong) == 1 else
               "%d letters wrong  →  strand lost" % len(wrong)) if wrong else "all correct → strand recovered"
    ax.set_title("%s:  %s" % (LABEL.get(run, run), verdict), loc="left", fontsize=6.6, fontweight="bold",
                 color=RED if wrong else "#0f7a3d", pad=4)
    # position ruler
    for c, k in enumerate(range(LO, HI)):
        if (k + 1) % 10 == 0:
            ax.text(c + 0.5, y + 1.2, str(k + 1), ha="center", va="top", fontsize=5.3, color=INK2)


fig, axs = plt.subplots(1, len(runs), figsize=(7.2, 2.9))
for ax, run in zip(np.atleast_1d(axs), runs):
    panel(ax, run)
plt.subplots_adjust(left=0.08, right=0.995, wspace=0.16, top=0.92, bottom=0.03)
fig.savefig(os.path.join(HERE, "figures", "fig_example.pdf"), facecolor="white")
print("saved")
