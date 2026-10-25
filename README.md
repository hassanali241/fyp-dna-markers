# FYP: synchronization markers for deletion-heavy DNA storage

Research question: at the same strand length (128), read count (10) and error
conditions, do periodic synchronization markers improve DNAFormer recovery
compared with a marker-free baseline? An end-block layout (same 32 marker bases,
not spread out) is the control.

Model: pretrained DNAFormer (https://github.com/itaiorr/Deep-DNA-based-storage,
commit 167a682), fine-tuned, architecture unchanged.

## Files (`fyp_dna/`)

| File | Stage | What it does |
|---|---|---|
| `stage1_generate.py` | 1 | Paired dataset: 3 arms, 4 deletion levels (1/3/5/8%), 10 reads per strand, DNAFormer's simulator |
| `stage1_validate.py` | 1 | Checks markers, pairing, leakage, reproducibility |
| `Stage1_dataset.ipynb` | 1 | Colab notebook (self-contained) |
| `dnaf_adapter.py` | 2+ | Feeds our data to the unchanged DNAFormer; strict checkpoint loading |
| `stage2_check.py` | 2 | Compatibility check, sanity inference, GPU memory test |
| `Stage2_compatibility.ipynb` | 2 | Colab notebook (self-contained, T4 GPU) |
| `stage3_baselines.py` | 3 | Majority vote, marker-sync majority vote, pretrained DNAFormer; payload-only metrics |
| `Stage3_baselines.ipynb` | 3 | Colab notebook (self-contained, T4 GPU) |
| `build_stage3_notebook.py` | 3 | Rebuilds the Stage 3 notebook from the .py files |
| `DNA_Storage_FYP_Overview.pdf` | – | 3-page printable overview |

Data and checkpoint live in Google Drive (`MyDrive/fyp_dna`), not in this repo.

## Status

- Stage 1 (dataset): done, all checks passed in Colab.
- Stage 2 (compatibility check): done, all checks passed in Colab (T4, Python 3.13, torch 2.11).
  - All 274 checkpoint tensors load strictly. The checkpoint names two fusion vectors
    `pred_fusion` / `pred_fusion_flip`; the code calls them `pred_fusion_left` / `_right`.
    The repo's strict=False load skips them; we map them (difference is tiny: mean NED 0.0182 vs 0.0188).
  - Repo data loader is unusable here: crashes on Python >= 3.11, drops deletion-heavy reads,
    swaps 41% of 8%-deletion test clusters, zeroes length-132 reads. We use `dnaf_adapter.py`.
  - Pretrained sanity check (50 clusters/level, exact recovery): 1% 100%, 3% 100%, 5% ~92%, 8% ~3%.
    Markers give no gain before fine-tuning (expected). Original loader at 8%: NED 0.358 vs 0.072 with adapter.
  - Fine-tune step on T4 (full model, Adam, fp32): batch 16 = 4.7 GB, 0.83 s/step; batch 32 = 8.0 GB, 1.64 s/step.
- Stage 3 (baselines on all 1,000 test clusters/level): done in Colab, see `results/stage3/summary.md`.
  - Pretrained DNAFormer exact recovery 100 / ~100 / 88-91 / ~8% at 1/3/5/8%; no difference between arms yet.
  - Majority votes tested locally on the full test sets (identical data): plain vote is 0-4% exact
    even at 1% deletion (one deletion shifts the rest of the read); marker-sync vote gets 53% at 1%,
    3% at 3%, 0% at 5-8%, and finds 97-100% of markers.
  - Pretrained DNAFormer part not run yet (no PyTorch locally); it reuses the Stage 2 code path.
- Stages 4–6: fine-tuning, final graphs, analysis.
