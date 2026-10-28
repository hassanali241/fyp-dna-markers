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
| 3. Baselines | DONE | Hassan's teammate | `results/stage3/summary.md`; CSVs in Drive `results/stage3` | Majority vote + pretrained DNAFormer, metrics CSVs |
| 4. Fine-tuning | DONE (all 3 arms) | 3 members, one ARM each | `Stage4_finetune.ipynb`; ~1.5 h on T4; paste output from `Training finished` |
| 1b. Marker-length datasets | DONE | Hassan | Drive `data/stage1_markers` (300 MB); all checks passed in Colab, identical to local run |
| 4b. Fine-tune marker lengths | READY TO RUN | | `Stage4_finetune.ipynb` with ARM = marker_len04 / 05 / 06 / 07 / 12; ~1.5 h each on T4 |
| 5. Final eval + graphs | NEXT | | Recovery rate and edit distance vs deletion rate |
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
- 2026-10-28: Stage 1b run in Colab: all checks passed (same payloads, seeds, error positions as Stage 1; 92 files match checksums).
- 2026-10-28: Stage 4b marker_len12 DONE: test exact 100/99.8/97.1/62.3%.
- 2026-10-28: Stage 4b marker_len06 DONE: test exact 100/99.9/95.3/49.6% (resumed once; resume worked).
- 2026-10-28: Stage 4b marker_len05 DONE: test exact 100/99.5/95.6/45.7%.
- 2026-10-28: Stage 4b marker_len04 DONE: test exact 100/99.8/95.4/44.6%. See results/stage4/marker_lengths_summary.md.
- 2026-10-28: Stage 1b built: marker lengths 4 (CAGT), 5 (TCATG), 6 (CTGTCA), 7 (CTGATCA), 12 (ACGTCAGTGCAT); 4 markers, one per 32-base block, same payloads/seeds/error positions as Stage 1. Original 3 arms rebuild byte-identical. Marker sequences chosen by design rules (no AA, no ACAC-type repeats, GC 40-60%, no self-overlap): occurrence ranking was re-tested on 2 new random datasets and does not repeat (rank agreement ~0), so only length matters.
- 2026-10-25: Teammate added to GitHub and Drive. PROGRESS.md created.
- Stage 4 marker_endblock DONE: test exact 100/100/95.7/41.0%. Summary: results/stage4/summary.md.
- Stage 4 marker_periodic DONE: test exact 100/100/96.1/52.4%. See results/stage4/marker_periodic.md.
- Stage 4 marker_free DONE: test exact 100/99.8/94.1/38.8% (pretrained 100/99.8/88.5/7.6%). See results/stage4/marker_free.md.
- Stage 4 notebook built and tested locally; merged to main.
- Stage 3 completed in Colab: DNAFormer 100/~100/88-91/~8% exact at 1/3/5/8%, arms equal before fine-tuning.
- Stage 3 notebook built; majority votes tested locally on full test sets.
- Stage 2 completed and verified in Colab (Hassan).
- Stage 1 completed and verified in Colab.
