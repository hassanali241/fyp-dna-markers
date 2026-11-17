"""Builds Stage5_analysis.ipynb from the .py files in this folder (run after editing them)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "/content/drive/MyDrive/fyp_dna/code"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def writefile(name):
    with open(os.path.join(HERE, name)) as f:
        return code("%%%%writefile %s/%s\n%s" % (CODE, name, f.read()))


cells = [
    md("""
# Stage 5: statistics + explainability (XAI)
Uses the **fine-tuned models from Stage 4**. Nothing is trained here.

| Part | What it does | Needs | Time |
|---|---|---|---|
| **A. Stats** | Tables, 95% CIs, paired tests (McNemar + bootstrap), figures for all runs | CPU is fine | a few minutes |
| **B. Ablation** | Scrambles the marker letters and checks how much recovery drops | T4 GPU | ~10 min per model |
| **C. XAI** | Gradient x input, Integrated Gradients, GradientSHAP, attention: *where does the model look?* | T4 GPU | ~30-45 min per model |

1. Runtime > Change runtime type > **T4 GPU**.
2. In **Cell 1**, set `PARTS` and `ARMS` (split the models between team members). Change nothing else.
3. Runtime > **Run all**. Allow Google Drive access.
4. Paste back everything from each `STAGE 5 ... DONE` block (and the table printed above it).

Suggested split for Parts B + C:

| Person | ARMS |
|---|---|
| Member 1 | `marker_free,marker_endblock` |
| Member 2 | `marker_len06,marker_len06_n1` |
| Member 3 | `marker_len06_n8,marker_periodic` |

Part A only needs to be run **once** (by anyone): it reads every run in `results/stage4`.
Parts B + C save into your own folder `results/stage5/<your ARMS>/`, so all three can run at the same time.
"""),
    md("### Cell 1: choose what to run (the ONLY thing you change)"),
    code("""
PARTS = ['stats', 'ablation', 'xai']        # any of 'stats', 'ablation', 'xai'
ARMS  = 'marker_free,marker_endblock'       # models for ablation + XAI (comma separated, no spaces)
                                            # e.g. marker_len06,marker_len06_n1   or   marker_len06_n8,marker_periodic

ALL_RUNS = ('marker_free', 'marker_periodic', 'marker_endblock',
            'marker_len04', 'marker_len05', 'marker_len06', 'marker_len07', 'marker_len12',
            'marker_len06_n1', 'marker_len06_n2', 'marker_len06_n8')
for a in ARMS.split(','):
    assert a in ALL_RUNS, f'{a} is misspelled'
assert set(PARTS) <= {'stats', 'ablation', 'xai'}, 'PARTS is misspelled'
print('Parts:', PARTS, '| models for B/C:', ARMS)
"""),
    md("### Cell 2: mount Drive, set paths, check the Stage 4 models"),
    code("""
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
PROJECT = '/content/drive/MyDrive/fyp_dna'
CODE    = f'{PROJECT}/code'
REPO    = '/content/Deep-DNA-based-storage'
OUT     = f'{PROJECT}/results/stage5'
OUT_BC  = f'{OUT}/' + ARMS.replace(',', '__')   # own folder per person, so parallel runs never overwrite each other
os.makedirs(CODE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

runs = sorted(d for d in os.listdir(f'{PROJECT}/results/stage4')
              if os.path.isfile(f'{PROJECT}/results/stage4/{d}/predictions.csv'))
print('Stage 4 runs with test predictions:', ', '.join(runs))
missing = []
if 'ablation' in PARTS or 'xai' in PARTS:
    for a in ARMS.split(','):
        p = f'{PROJECT}/checkpoints/stage4/{a}/best.pt'
        print('found' if os.path.isfile(p) else 'MISSING', ' ', p)
        if not os.path.isfile(p):
            missing.append(p)
if missing:
    raise RuntimeError('STOP: fine-tuned model(s) missing from Drive:\\n  ' + '\\n  '.join(missing))
print('OK to continue.')
!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'no GPU (fine for Part A only)'
"""),
    md("### Cell 3: install packages (einops for DNAFormer, captum for Integrated Gradients / SHAP)"),
    code("!pip install -q einops captum"),
    md("### Cell 4: write the Stage 1 helpers"),
    writefile("stage1_generate.py"),
    md("### Cell 5: write the DNAFormer adapter"),
    writefile("dnaf_adapter.py"),
    md("### Cell 6: write the Stage 3 scoring code"),
    writefile("stage3_baselines.py"),
    md("### Cell 7: write the Stage 5 analysis script"),
    writefile("stage5_analysis.py"),
    md("### Cell 8: get the DNAFormer repo (same pinned commit as before)"),
    code("""
!rm -rf {REPO}
!git clone -q https://github.com/itaiorr/Deep-DNA-based-storage {REPO}
!git -C {REPO} checkout -q 167a68261719ea458e79c40fcc485ae768c1a64a
!git -C {REPO} rev-parse HEAD
"""),
    md("""
### Cell 9: Part A, statistics (a few minutes)
**Paste back** the tables printed above `STAGE 5 STATS DONE`.
Saves to `results/stage5/stats/`: `results_all.csv`, `paired_tests.csv`, `fig_*_vs_deletion.png`, `error_position_*.png`.
"""),
    code("""
if 'stats' in PARTS:
    !cd {CODE} && python stage5_analysis.py --part stats --results {PROJECT}/results/stage4 --out {OUT} 2>&1 | grep -v Warning
else:
    print('skipped (stats not in PARTS)')
"""),
    md("""
### Cell 10: Part B, marker ablation (~10 min per model)
Same reads, same noise, but every marker letter is replaced by a different random letter.
If recovery drops, the model **uses** the markers. **Paste back** the table above `STAGE 5 ABLATION DONE`.
"""),
    code("""
if 'ablation' in PARTS:
    !cd {CODE} && python stage5_analysis.py --part ablation --project {PROJECT} --repo {REPO} --arms {ARMS} --out {OUT_BC} 2>&1 | grep -v Warning
else:
    print('skipped (ablation not in PARTS)')
"""),
    md("""
### Cell 11: Part C, XAI (~30-45 min per model)
Gradient x input on 100 test strands (8% deletion), Integrated Gradients + GradientSHAP on 20, attention on 50.
**Paste back** the table above `STAGE 5 XAI DONE`.
Figures (`heatmaps.png`, `marker_share.png`, `example_read_*.png`) are saved in `results/stage5/<your ARMS>/xai/`.
"""),
    code("""
if 'xai' in PARTS:
    !cd {CODE} && python stage5_analysis.py --part xai --project {PROJECT} --repo {REPO} --arms {ARMS} --out {OUT_BC} 2>&1 | grep -v Warning
else:
    print('skipped (xai not in PARTS)')
"""),
    md("### Cell 12: show the figures"),
    code("""
import glob
from IPython.display import Image, display
pngs = glob.glob(f'{OUT}/stats/*.png') if PARTS == ['stats'] else glob.glob(f'{OUT_BC}/**/*.png', recursive=True)
for p in sorted(pngs):
    print(p.replace(PROJECT + '/', '')); display(Image(p, width=900))
"""),
    md("""
---
**How to read the XAI numbers.** *Marker share* = how much of the model's attention (importance) lands on marker
letters. *Space share* = how much of the strand the markers take up. **Share above space share** means the model
looks at the markers more than their size alone explains, i.e. it really uses them as anchors.
"""),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(os.path.join(HERE, "Stage5_analysis.ipynb"), "w", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote Stage5_analysis.ipynb with %d cells" % len(cells))
