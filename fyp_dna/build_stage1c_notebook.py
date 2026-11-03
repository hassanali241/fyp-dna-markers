"""Rebuilds Stage1c_marker_spacing.ipynb from the .py files (run: python build_stage1c_notebook.py)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "/content/drive/MyDrive/fyp_dna/code"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")}


def embed(name):
    return code("%%%%writefile %s/%s\n" % (CODE, name) + open(os.path.join(HERE, name), encoding="utf-8").read())


cells = [
    md("""# Stage 1c: marker-spacing datasets
Builds 2 new versions of the **same** Stage 1 dataset into `MyDrive/fyp_dna/data/stage1_spacing`.
Same 6-letter marker `CTGTCA`; only the number of markers (the gap between them) changes.
Your other data is not touched.

| Arm | Markers | Each block | Data letters | Marker share |
|---|---|---|---|---|
| marker_len06_n2 | 2 | 58 data + 6 marker (64) | 116 | 9.4% |
| marker_len06 (done) | 4 | 26 data + 6 marker (32) | 104 | 18.8% |
| marker_len06_n8 | 8 | 10 data + 6 marker (16) | 80 | 37.5% |

Same payloads, same seeds, same error positions as Stage 1. **No GPU needed.**
Runtime > **Run all** (about 5 minutes). Paste the output of the last cell."""),
    md("### Cell 1: mount Drive and set paths"),
    code("""
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
PROJECT = '/content/drive/MyDrive/fyp_dna'
CODE    = f'{PROJECT}/code'
REPO    = '/content/Deep-DNA-based-storage'
REF     = f'{PROJECT}/data/stage1_pilot'
OUT     = f'{PROJECT}/data/stage1_spacing'
os.makedirs(CODE, exist_ok=True); os.makedirs(f'{PROJECT}/logs', exist_ok=True)
print('Stage 1 reference data found:', os.path.isfile(f'{REF}/marker_free/train/metadata.json'))
print('Output folder already exists:', os.path.isdir(OUT))
"""),
    md("### Cell 2: write the dataset generator (now supports 2, 4 or 8 markers)"),
    embed("stage1_generate.py"),
    md("### Cell 3: write the validation script"),
    embed("stage1b_validate.py"),
    md("### Cell 4: get DNAFormer repo (same pinned commit as Stage 1)"),
    code("""
!rm -rf {REPO}
!git clone -q https://github.com/itaiorr/Deep-DNA-based-storage {REPO}
!git -C {REPO} checkout -q 167a68261719ea458e79c40fcc485ae768c1a64a
!git -C {REPO} rev-parse HEAD
"""),
    md("### Cell 5: build the 2 spacing datasets in Drive (about 4 minutes, about 120 MB)"),
    code("""
!python {CODE}/stage1_generate.py --repo {REPO} --out {OUT} --arms spacing 2>&1 | tail -3
"""),
    md("### Cell 6: validate against Stage 1. Paste the whole output of this cell back to Claude."),
    code("""
!python {CODE}/stage1b_validate.py --repo {REPO} --data {OUT} --ref {REF} 2>&1 | tee {PROJECT}/logs/stage1c_validation.txt
!du -sh {OUT}
"""),
]
nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(os.path.join(HERE, "Stage1c_marker_spacing.ipynb"), "w", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote Stage1c_marker_spacing.ipynb with %d cells" % len(cells))
