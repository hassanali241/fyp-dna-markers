"""Builds Stage3_baselines.ipynb from the .py files in this folder (run after editing them)."""
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
# Stage 3: baselines (no training)
**Use a GPU:** Runtime > Change runtime type > **T4 GPU** before running.

Run the cells in order (or Runtime > Run all). Nothing to upload.
Scores majority vote, marker-sync majority vote and the pretrained DNAFormer on all
1,000 test clusters per deletion level, payload-only.
"""),
    md("### Cell 1: mount Drive, set paths, check data, checkpoint and GPU"),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT = '/content/drive/MyDrive/fyp_dna'
CODE    = f'{PROJECT}/code'
REPO    = '/content/Deep-DNA-based-storage'
DATA    = f'{PROJECT}/data/stage1_pilot'
CKPT    = f'{PROJECT}/checkpoints/DNAFormer_siamese.pth'
OUT     = f'{PROJECT}/results/stage3'
os.makedirs(CODE, exist_ok=True); os.makedirs(OUT, exist_ok=True); os.makedirs(f'{PROJECT}/logs', exist_ok=True)
print('Stage 1 data on Drive:', os.path.isfile(f'{DATA}/config.json'))
print('Checkpoint on Drive:  ', os.path.isfile(CKPT), '(made by the Stage 2 notebook)')
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("### Cell 2: install the small package DNAFormer needs"),
    code("!pip install -q einops"),
    md("### Cell 3: write the Stage 1 helpers (marker layout, marker removal)"),
    writefile("stage1_generate.py"),
    md("### Cell 4: write the DNAFormer adapter (same as Stage 2)"),
    writefile("dnaf_adapter.py"),
    md("### Cell 5: write the Stage 3 script"),
    writefile("stage3_baselines.py"),
    md("### Cell 6: get the DNAFormer repo (same pinned commit as Stages 1 and 2)"),
    code("""
!rm -rf {REPO}
!git clone -q https://github.com/itaiorr/Deep-DNA-based-storage {REPO}
!git -C {REPO} checkout -q 167a68261719ea458e79c40fcc485ae768c1a64a
!git -C {REPO} rev-parse HEAD
"""),
    md("### Cell 7: quick smoke test (25 clusters per level, about 1 minute)\nShould end with `STAGE 3 DONE`. If it fails, paste the error to Claude."),
    code("""
!cd {CODE} && python stage3_baselines.py --repo {REPO} --data {DATA} --ckpt {CKPT} --out /content/stage3_smoke --limit 25 2>&1 | grep -v Warning | tail -8
"""),
    md("### Cell 8: full run (all 1,000 test clusters per level, about 5 minutes)\n**Paste the whole output of this cell back to Claude.**"),
    code("""
!cd {CODE} && python stage3_baselines.py --repo {REPO} --data {DATA} --ckpt {CKPT} --out {OUT} 2>&1 | grep -v Warning | tee {PROJECT}/logs/stage3_baselines.txt
!ls -la {OUT}
"""),
    md("""
---
**Results are saved to** `MyDrive/fyp_dna/results/stage3/`:
- `results.csv`: one row per method, version and deletion level (the summary)
- `predictions.csv`: one row per test strand (true payload, prediction, scores)
- `stage3_log.json`: settings and environment

**Methods:** `mv` = plain majority vote, `mv_sync` = majority vote that re-aligns at each marker
(periodic version only), `dnaformer` = pretrained DNAFormer without fine-tuning.
"""),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(os.path.join(HERE, "Stage3_baselines.ipynb"), "w", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote Stage3_baselines.ipynb with %d cells" % len(cells))
