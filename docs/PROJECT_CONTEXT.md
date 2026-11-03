# FYP Project Context: Synchronization Markers for Deletion-Heavy DNA Storage (DNAFormer)

Handover document for anyone (or any Claude chat) joining the project, especially for writing the
**SRS** and **SDS**. It summarizes the whole project discussion up to 29 October 2026.
For live status see `PROGRESS.md`; for numbers see `results/` and `reports/`.

---

## 1. Project in one paragraph
DNA data storage writes digital data as short synthetic DNA strands. Reading them back gives ~10 noisy
copies ("reads") per strand with substitutions, insertions and **deletions** (the hardest error, because
everything after a deletion shifts). **DNAFormer** (Bar-Lev et al., Nature Machine Intelligence 2025,
repo `itaiorr/Deep-DNA-based-storage`) is a deep-learning model that reconstructs the original strand from
its reads. This FYP tests whether inserting short known **synchronization markers** into each strand helps a
**fine-tuned** DNAFormer recover data under deletion-heavy noise, and which **marker length** gives the best
trade-off between accuracy and storage overhead.

**Research question:** At the same strand length (128 bases), same number of reads (10) and same error
conditions, do periodic synchronization markers improve deletion-heavy recovery compared with a marker-free
DNAFormer baseline, and is the gain due to synchronization (marker placement) rather than a shorter payload?

**Inspiration:** Inria PhD posting RobDNA (coding for DNA storage). Markers are a classic idea
("watermark codes", Davey & MacKay 2001).

## 2. Team and tools
| Member | Role in practice |
|---|---|
| Hassan (repo/Drive owner, `hassanali241`, akhundhassan84@gmail.com) | Stage 1-2, Stage 1b datasets, runs, reports |
| Teammate (haadjunejo@gmail.com) | Stage 3, Stage 4 runs, reports |
| Mahek (mahakkalani29@gmail.com) | Stage 4 end-block / length-12 runs; **SRS and SDS documents** |

- **Code:** private GitHub repo `hassanali241/fyp-dna-markers` (folder `fyp_dna/`).
- **Data and models:** Google Drive folder `MyDrive/fyp_dna` (owned by Hassan, shared as Editor).
  Team members add it to their My Drive as a **shortcut**.
- **Compute:** Google Colab, free **T4 GPU**. Every stage is a self-contained notebook (scripts are embedded
  with `%%writefile`, so nothing needs uploading).
- **Assistant:** Claude Code (cloud session) wrote and tested code; team runs notebooks and pastes outputs.

### Drive layout (do not move or delete anything here)
```
fyp_dna/
  code/                       scripts written by the notebooks
  data/stage1_pilot/          Stage 1 dataset (3 arms)            ~181 MB
  data/stage1_markers/        Stage 1b dataset (marker lengths)
  checkpoints/DNAFormer_siamese.pth   pretrained model (1.24 GB)
  checkpoints/stage4/<ARM>/   best.pt, last.pt, history.json, train_log.txt
  results/stage3/, results/stage4/<ARM>/   results.csv, predictions.csv
  logs/                       validation and run logs
```
Incident (25 Oct): folders were twice removed from `fyp_dna` by a teammate account; restored. Rule: never
delete/move inside `fyp_dna`.

## 3. Pipeline (stages) and status
| Stage | What | Status |
|---|---|---|
| 1. Dataset | Paired datasets, 3 arms, DNAFormer's own simulator | DONE |
| 2. Compatibility | Load pretrained checkpoint, adapter, GPU memory | DONE |
| 3. Baselines | Majority vote, marker-sync vote, pretrained DNAFormer | DONE (CSV files lost in Drive incident; numbers in `results/stage3/summary.md`) |
| 4. Fine-tuning | 3 arms fine-tuned with identical settings | DONE |
| 1b. Marker-length datasets | lengths 4, 5, 6, 7, 12 | DONE |
| 4b. Marker-length fine-tuning | 5 arms | DONE |
| 5. Final evaluation | paired tests (McNemar), final graphs, maybe repeat seeds | NEXT |
| 6. Analysis/report | thesis chapters, SRS/SDS, slides | in progress |

### Stage 1: dataset (`stage1_generate.py`, `stage1_validate.py`, `Stage1_dataset.ipynb`)
- Strand length **128**; 10 reads per strand; DNAFormer simulator (`DataGenerator/cluster_generator.py`,
  pinned commit `167a682`); insertion and substitution fixed at **1%**; deletion **1, 3, 5, 8%**.
- Arms: `marker_free` (128 data bases), `marker_periodic` (4 x [24 data + `ACGTCAGT`], 96 data),
  `marker_endblock` (96 data + 4 markers at the end; the **control** that separates synchronization from
  "less payload to recover").
- Splits per arm: train 10,000 (levels mixed equally), val 1,000 (mixed), test **1,000 per level**.
- Arms are **paired**: same payload seeds (marker arms use the first bases of the marker-free payload), same
  noise seeds; seed ranges never overlap between splits (no leakage). Reproducible byte-for-byte, checksums.
- Files per subset: `clean.fasta`, `noisy_reads.fasta`, `metadata.json`; plus `dataset_manifest.csv`,
  `config.json`, `checksums.json`.

### Stage 2: compatibility (`dnaf_adapter.py`, `stage2_check.py`)
Problems found in DNAFormer's original loader (so we replaced it with `dnaf_adapter.py`, model unchanged):
- crashes on Python 3.11+; its length filter drops deletion-heavy reads (59% at 5%, 92% at 8% deletion);
  silently swaps empty clusters (41% of 8% test clusters); blanks length-132 reads; loads weights with
  `strict=False` (two renamed fusion weights were silently skipped; now mapped explicitly, 274/274 tensors).
- Model format: input 132 bases (reads padded), output 128 bases. GPU: batch 16 = 4.7 GB fp32.

### Stage 3: baselines (`stage3_baselines.py`), 1,000 test strands per level
| method | 1% | 3% | 5% | 8% |
|---|---|---|---|---|
| majority vote (all arms) | 1-4% | 0% | 0% | 0% |
| marker-sync vote (periodic) | 53.4% | 3.1% | 0% | 0% |
| pretrained DNAFormer (free / periodic / end-block) | 100 | ~100 | 88.5 / 91.2 / 89.3 | 7.6 / 7.9 / 7.5 |

### Stage 4: fine-tuning (`stage4_finetune.py`, `Stage4_finetune.ipynb`; one notebook, set `ARM` in Cell 1)
Locked settings (identical for every arm):
| setting | value |
|---|---|
| start | pretrained `DNAFormer_siamese.pth`, architecture unchanged (103.4M params) |
| loss | DNAFormer's own: cross-entropy + 0.25 x consistency (forward/reverse branches) |
| optimizer | Adam (0.9, 0.999, eps 1e-8), no weight decay |
| batch / epochs / seed | 16 / 10 x 625 steps / 1 |
| phase A (epochs 1-2) | transformer frozen; CNN alignment + embedding, output, fusion train (2.6M params); lr 1e-5 |
| phase B (epochs 3-10) | whole model; 100-step warm-up to 3e-6, cosine decay to 1e-7 |
| other | grad clip 1.0, fp16 mixed precision, all 10 reads kept, best epoch by payload-only val NED |
| resume | `last.pt` saved every epoch; rerun continues; settings mismatch refused |
| time | ~35-40 min per arm on T4 |

### Results: Stage 4 + 4b (test, 8% deletion unless noted)
| version | marker | data bases | overhead | exact 5% | exact 8% [95% CI] | NED 8% | data recovered 8% | vs free |
|---|---|---|---|---|---|---|---|---|
| marker_free | - | 128 | 0% | 94.1% | 38.8% [35.8, 41.9] | 0.0202 | 49,664 | - |
| len 4 | CAGT | 112 | 12.5% | 95.4% | 44.6% [41.5, 47.7] | 0.0183 | 49,952 | +0.6% |
| len 5 | TCATG | 108 | 15.6% | 95.6% | 45.7% [42.6, 48.8] | 0.0182 | 49,356 | -0.6% |
| **len 6** | CTGTCA | 104 | 18.75% | 95.3% | 49.6% [46.5, 52.7] | 0.0172 | **51,584** | **+3.9%** |
| len 7 | CTGATCA | 100 | 21.9% | 96.1% | 49.9% [46.8, 53.0] | 0.0168 | 49,900 | +0.5% |
| len 8 (periodic) | ACGTCAGT | 96 | 25% | 96.1% | 52.4% [49.3, 55.5] | 0.0148 | 50,304 | +1.3% |
| len 12 | ACGTCAGTGCAT | 80 | 37.5% | 97.1% | 62.3% [59.3, 65.3] | 0.0119 | 49,840 | +0.4% |
| end-block 8 (control) | ACGTCAGT | 96 | 25% | 95.7% | 41.0% [38.0, 44.1] | 0.0241 | 39,360 | -20.7% |

Data recovered = exact strands x data bases per 1,000 strands. 1% and 3%: all versions ~100%.

**Findings**
1. Spacing (synchronization) is what helps: periodic 52.4% vs end-block 41.0% with identical content.
2. Accuracy rises with marker length (44.6 -> 62.3%).
3. On stored data nearly every length breaks even; length 6 is the nominal winner (+3.9%) under the rule
   agreed before results ("most data recovered at 8%; ties go to shorter"). Single runs: needs Stage 5
   paired tests / repeat seeds to confirm.
4. Markers pay off only under heavy deletion; at 5% the marker-free strand stores the most data.
5. Fine-tuning raised 8% recovery for all arms from ~8% to 39-62%; no harmful overfitting (val matches test).

### Marker selection (Stage 1b; `reports/marker_occurrence_table.xlsx`)
- Layout: 4 markers, one at the end of each 32-base block; data per block = 32 - L.
- Rules: no repeated letter, no self-overlap, GC 40-60%, lowest occurrence in the dataset among eligible.
  Occurrence ranking within a length is random (did not repeat on new datasets): **length** is what matters
  (each extra base = ~4x rarer accidental matches).
- Not tested: length 2 (in 99.96% of strands), length 3 (in ~86%, and cannot meet GC 40-60%).
  9-11 not run (8 already rare at 0.2%); 12 run once as an upper-bound check.

### Real-world context
- Typical nanopore deletion ~2% per base (1-3% levels realistic); 5-8% = stress test (homopolymers, read
  ends, deletion-prone enzymatic synthesis). DNAFormer reports ~85% exact recovery on Microsoft's real ONT
  dataset (failure rate 0.146, as cited by TReconLM, arXiv 2507.12927) ~ our 5% condition.

## 4. Model architecture (DNAFormer, unchanged)
```
10 reads (one-hot 4 x 132, padded; 16 slots) + same reads reversed (siamese, shared weights)
 1. Alignment (CNN, per read): 4 parallel conv branches k=1,3,5,7 (8 conv layers) + linear block (3)
 2. Embedding (CNN): sum over reads -> 4 conv branches (8 conv) + linear 132->128 => 128 x 1024 tokens
 3. Transformer encoder: 12 layers, 32 heads, d_model 1024, FF 2048 (~100.8M params, no positional enc.)
 4. Output: 3 x 1x1 conv 1024->1024->1024->4  (per-position classifier)
 5. Fusion: learned per-position weights merge forward/reversed + 3 x 1x1 conv -> 128 letters
```

## 5. Repository map (`fyp_dna/`)
| file | purpose |
|---|---|
| `stage1_generate.py` / `stage1_validate.py` / `Stage1_dataset.ipynb` | dataset generation + checks |
| `Stage1b_marker_lengths.ipynb`, `stage1b_validate.py`, `build_stage1b_notebook.py` | marker-length datasets |
| `dnaf_adapter.py` | safe loader: encoding, strict checkpoint load, scoring helpers |
| `stage2_check.py` / `Stage2_compatibility.ipynb` | compatibility check |
| `stage3_baselines.py` / `Stage3_baselines.ipynb` | baselines and shared metrics (Wilson CI, NED, file-level) |
| `stage4_finetune.py` / `Stage4_finetune.ipynb` | fine-tuning + test scoring |
| `build_stage*_notebook.py` | rebuild notebooks from the .py files |

Reports (`reports/`): `Stage4_results_one_page.pdf`, `Stage4_results_detailed.pdf` (incl. fine-tuning page),
`Stage4b_marker_length_results.pdf`, `Marker_Length_Insights.pdf`, `Marker_Length_Results_Supervisor.pdf`,
`marker_occurrence_table.xlsx`. Per-run results: `results/stage4/*.md`.

## 6. Material for the SRS and SDS
**Scope:** a reproducible research pipeline (not an end-user product) to generate DNA-storage datasets with
optional synchronization markers, fine-tune a pretrained DNAFormer, and evaluate recovery.

**Functional requirements (suggested)**
- FR1 Generate paired datasets for any arm (marker-free, periodic, end-block, marker length L) at configurable
  deletion/insertion/substitution rates, 10 reads per strand, fixed seeds.
- FR2 Validate datasets (structure, marker positions, pairing, leakage, reproducibility, checksums).
- FR3 Load the pretrained DNAFormer strictly and refuse mismatched weights.
- FR4 Encode reads without dropping or swapping clusters (relaxed read policy).
- FR5 Compute baselines (majority vote, marker-sync vote, pretrained DNAFormer).
- FR6 Fine-tune one arm with locked settings; save best/last checkpoints each epoch; resume after disconnect.
- FR7 Evaluate payload-only metrics: exact recovery + Wilson 95% CI, NED, per-base accuracy, file-level
  recovery (25 strands/file), runtime, GPU memory; write `predictions.csv`, `results.csv`.
- FR8 Stop with a clear message if data or checkpoint files are missing.
- FR9 (Stage 5) Paired statistical tests and final graphs.

**Non-functional requirements**
- Reproducibility (fixed seeds, pinned simulator commit, checksums, byte-identical regeneration).
- Fairness (identical settings across arms; settings stored in checkpoints; resume refuses changes).
- Runs on free Colab T4 (peak ~3.6 GB GPU; ~40 min per arm); resumable.
- Data safety (no deletions; Drive as single source of data; GitHub for code).
- Maintainability (one notebook per stage, generated from versioned .py files).

**Design (SDS) notes**
- Components: Data generator -> Validator -> Adapter (I/O + model loading) -> Baseline evaluator ->
  Fine-tuner -> Evaluator/Reporter. Shared modules: `stage1_generate` (layouts, marker removal),
  `dnaf_adapter` (encoding, decode, edit distance), `stage3_baselines` (metrics).
- Data flow: seeds -> payload -> strand (arm layout) -> simulator reads -> FASTA/JSON -> encoder tensors
  [2B,16,4,132] -> model -> logits [B,4,128] -> decode -> remove markers -> metrics.
- Storage: Drive paths above; outputs per arm folder; logs.
- Deployment: Colab notebooks; GitHub repo; Drive shortcut per member.

## 7. Open items / next steps
1. Stage 5: McNemar paired tests (same strands across arms), final graphs, regenerate Stage 3 CSVs.
2. Repeat runs (seed 2) for lengths 6, 7, 8 and marker-free to confirm length 6.
3. Optional: end-block controls for short markers; error-position analysis; marker ablation; SHAP /
   Integrated Gradients ("why markers help"); coverage (fewer reads) test.
4. Thesis chapters, SRS, SDS, defense slides.

## 8. How to continue in a new Claude chat
Start a new Claude Code session with this repo selected and say:
> "Read docs/PROJECT_CONTEXT.md, PROGRESS.md, README.md and results/stage4/marker_lengths_summary.md.
> I am Mahek. Help me write the SRS (and later the SDS) for this project."
