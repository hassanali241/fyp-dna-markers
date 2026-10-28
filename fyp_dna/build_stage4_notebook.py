"""Builds Stage4_finetune.ipynb from the .py files in this folder (run after editing them)."""
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
# Stage 4: fine-tune DNAFormer on ONE version
Each team member trains a different version with **exactly the same settings**.

1. **Use a GPU:** Runtime > Change runtime type > **T4 GPU**.
2. In **Cell 1**, set `ARM` to your version. **Change nothing else.**
3. Runtime > **Run all**. Allow Google Drive access.

| Person | ARM |
|---|---|
| Member 1 | `marker_free` |
| Member 2 | `marker_periodic` |
| Member 3 | `marker_endblock` |

**Marker-length runs** (data made by `Stage1b_marker_lengths.ipynb`): `marker_len04`, `marker_len05`,
`marker_len06`, `marker_len07`, `marker_len12`. Same settings; one run per notebook.

Training takes about **1.5 hours**. Keep the tab open.
**If Colab disconnects:** reconnect (T4 GPU) and **Run all** again. It continues from the last finished epoch.
"""),
    md("### Cell 1: choose your version (the ONLY thing you change)"),
    code("""
ARM = 'marker_periodic'   # one of: 'marker_free', 'marker_periodic', 'marker_endblock',
                          #         'marker_len04', 'marker_len05', 'marker_len06', 'marker_len07', 'marker_len12'

ARMS_STAGE1  = ('marker_free', 'marker_periodic', 'marker_endblock')
ARMS_LENGTHS = ('marker_len04', 'marker_len05', 'marker_len06', 'marker_len07', 'marker_len12')
assert ARM in ARMS_STAGE1 + ARMS_LENGTHS, 'ARM is misspelled'
print('This notebook will fine-tune:', ARM)
"""),
    md("### Cell 2: mount Drive, set paths, check data, checkpoint and GPU"),
    code("""
from google.colab import drive
drive.mount('/content/drive', force_remount=True)   # fresh view of Drive (sees new uploads)

import os
PROJECT  = '/content/drive/MyDrive/fyp_dna'
CODE     = f'{PROJECT}/code'
REPO     = '/content/Deep-DNA-based-storage'
DATA     = f'{PROJECT}/data/' + ('stage1_pilot' if ARM in ARMS_STAGE1 else 'stage1_markers')
CKPT     = f'{PROJECT}/checkpoints/DNAFormer_siamese.pth'
CKPT_DIR = f'{PROJECT}/checkpoints/stage4'
OUT_DIR  = f'{PROJECT}/results/stage4'
os.makedirs(CODE, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
need = {f'{DATA}/{ARM}/{s}/metadata.json': f'Stage 1 data {ARM}/{s}' for s in
        ['train', 'val', 'test_del01', 'test_del03', 'test_del05', 'test_del08']}
need[CKPT] = 'pretrained model DNAFormer_siamese.pth'
missing = [what + '  ->  ' + path for path, what in need.items() if not os.path.isfile(path)]
for path, what in need.items():
    print('found' if os.path.isfile(path) else 'MISSING', ' ', what)
print('Earlier progress to resume: ', os.path.isfile(f'{CKPT_DIR}/{ARM}/last.pt'))
if missing:
    raise RuntimeError('STOP: these files are missing from Google Drive:\\n  ' + '\\n  '.join(missing) +
                     '\\nPut them back in MyDrive/fyp_dna (data/stage1_pilot or data/stage1_markers, and checkpoints), '
                     'then Runtime > Disconnect and delete runtime, and Run all again.')
print('All files found. OK to continue.')
!nvidia-smi --query-gpu=name,memory.total --format=csv
!df -h /content/drive | tail -1
"""),
    md("### Cell 3: install the small package DNAFormer needs"),
    code("!pip install -q einops"),
    md("### Cell 4: write the Stage 1 helpers"),
    writefile("stage1_generate.py"),
    md("### Cell 5: write the DNAFormer adapter"),
    writefile("dnaf_adapter.py"),
    md("### Cell 6: write the Stage 3 scoring code (reused for the test scores)"),
    writefile("stage3_baselines.py"),
    md("### Cell 7: write the Stage 4 training script"),
    writefile("stage4_finetune.py"),
    md("### Cell 8: get the DNAFormer repo (same pinned commit as before)"),
    code("""
!rm -rf {REPO}
!git clone -q https://github.com/itaiorr/Deep-DNA-based-storage {REPO}
!git -C {REPO} checkout -q 167a68261719ea458e79c40fcc485ae768c1a64a
!git -C {REPO} rev-parse HEAD
"""),
    md("""
### Cell 9: 3-minute test (tiny run, also tests stop-and-resume)
Should end with `STAGE 4 DONE`. Nothing is written to your real checkpoint folder.
If it fails, **stop here** and paste the error to Claude.
"""),
    code("""
!rm -rf /content/stage4_smoke
!cd {CODE} && python stage4_finetune.py --repo {REPO} --data {DATA} --ckpt {CKPT} --arm {ARM} --ckpt-dir /content/stage4_smoke/ckpt --out-dir /content/stage4_smoke/out --smoke --stop-after 2 2>&1 | grep -v Warning | tail -4
!cd {CODE} && python stage4_finetune.py --repo {REPO} --data {DATA} --ckpt {CKPT} --arm {ARM} --ckpt-dir /content/stage4_smoke/ckpt --out-dir /content/stage4_smoke/out --smoke 2>&1 | grep -v Warning | tail -12
"""),
    md("""
### Cell 10: full fine-tuning (about 1.5 hours) + test scores
Progress prints every ~2 minutes. **When it finishes, paste everything from
`Training finished` to `STAGE 4 DONE` back to Claude.**
The full log is also saved to `MyDrive/fyp_dna/checkpoints/stage4/<ARM>/train_log.txt`.
"""),
    code("""
!cd {CODE} && python stage4_finetune.py --repo {REPO} --data {DATA} --ckpt {CKPT} --arm {ARM} --ckpt-dir {CKPT_DIR} --out-dir {OUT_DIR} 2>&1 | grep -v Warning
"""),
    md("""
---
**Saved to Drive**
- `checkpoints/stage4/<ARM>/best.pt`: the fine-tuned model (best validation epoch), about 0.4 GB
- `checkpoints/stage4/<ARM>/last.pt`: resume point, about 1.2 GB (can be deleted after Stage 5)
- `checkpoints/stage4/<ARM>/history.json` and `train_log.txt`: validation scores per epoch
- `results/stage4/<ARM>/results.csv` and `predictions.csv`: test scores, same format as Stage 3

**Settings are locked.** If you change anything except `ARM`, the run refuses to resume and the
comparison between versions is no longer fair.
"""),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(os.path.join(HERE, "Stage4_finetune.ipynb"), "w", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote Stage4_finetune.ipynb with %d cells" % len(cells))
