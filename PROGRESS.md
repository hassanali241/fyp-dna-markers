# FYP Progress: Synchronization markers for DNA storage (DNAFormer)

Shared status file for the team. Update it every time you finish or start something.

## Research question
At the same strand length (128), same reads (10) and same errors, do periodic
synchronization markers (`ACGTCAGT` after every 24 data bases) improve DNAFormer
recovery under heavy deletions, compared with no markers?

## The 3 dataset versions (arms)
| Arm | Layout | Data bases |
|---|---|---|
| marker_free | 128 data | 128 |
| marker_periodic | 4 x (24 data + 8 marker) | 96 |
| marker_endblock | 96 data + 32 marker bases at the end (control) | 96 |

End-block is the control: same data amount and same markers as periodic, only the
position differs. Periodic > end-block means the spacing (synchronization) helps.

## Where things live
- Code: this repo, folder `fyp_dna/`
- Dataset: Google Drive `MyDrive/fyp_dna/data/stage1_pilot` (181 MB). Add a shortcut
  to "My Drive" if it was shared with you, so notebook paths work.
- DNAFormer checkpoint: downloaded to Drive by the Stage 2 notebook.

## Stage status
| Stage | Status | Owner | Notes |
|---|---|---|---|
| 1. Dataset | DONE | Hassan | All checks passed in Colab; identical to local run |
| 2. Compatibility check | DONE | Hassan | All checks passed (T4, Python 3.13, torch 2.11). See README for numbers |
| 3. Baselines | NEXT, not written | ? | Majority vote + pretrained DNAFormer, metrics CSVs |
| 4. Fine-tuning | Not written | ? | T4: batch 16 = 4.7 GB (0.83 s/step), batch 32 = 8.0 GB (1.64 s/step) |
| 5. Final eval + graphs | Not written | | Recovery rate and edit distance vs deletion rate |
| 6. Analysis | Not started | | |

## Key facts so far
- Dataset per arm: 10,000 train (mixed 1/3/5/8% deletion), 1,000 val, 1,000 test per
  deletion level. 10 reads per cluster. Insertion and substitution fixed at 1%.
- Measured deletion rates: 1.0%, 2.9%, 4.9%, 7.6%.
- DNAFormer's original read filter would drop 59% of reads at 5% deletion, 92% at 8%.
- DNAFormer's original loader also crashes on Python 3.11+, swaps clusters with no
  reads left (41% of 8% test clusters), blanks 132-length reads, and loads weights
  with strict=False. `dnaf_adapter.py` replaces it; model architecture unchanged.
- Marker appears by chance in ~0.2% of payloads (30/15000 marker-free).
- Scoring: payload only (markers removed before scoring).
- Stage 2: all 274 checkpoint tensors load strictly. Two fusion weights were renamed in the
  checkpoint (`pred_fusion`/`pred_fusion_flip` -> `pred_fusion_left`/`_right`); the original
  strict=False load silently skipped them. Now mapped.
- Pretrained DNAFormer, no fine-tuning (50 clusters/level, exact recovery):
  1% 100%, 3% 100%, 5% ~92%, 8% ~3%. Markers give no gain before fine-tuning (expected).
- At 8% deletion the original loader gives NED 0.358 vs 0.072 with our adapter.

## Team rules
1. Pull and read this file before starting work.
2. Claim a stage here (set Owner) before running it.
3. Don't run notebooks that write to the same Drive folder at the same time.
4. When done: save outputs/CSVs/graphs in `results/stageN/`, update this file, push.

## Log (newest first)
- 2026-10-25: Teammate added to GitHub and Drive. PROGRESS.md created.
- Stage 2 completed and verified in Colab (Hassan).
- Stage 1 completed and verified in Colab.
