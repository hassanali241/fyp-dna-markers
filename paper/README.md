# Phase 1 conference paper (two versions)

- `paper_ieee.pdf`: IEEE conference format (two columns), extended version with pipeline, baselines,
  length/number figure and a worked example.
- `paper_lncs.pdf`: Springer LNCS format (one column), core version.

Both use the same text (`body.tex`, `abstract.tex`), figures (`figures/`) and references (`refs.bib`).
IEEE-only parts are inside `\ifextended ... \fi` in `body.tex`.

## Edit on Overleaf
1. Zip this `paper/` folder (or download it from GitHub).
2. Overleaf > New Project > Upload Project > choose the zip.
3. Menu > Main document: `paper_ieee.tex` or `paper_lncs.tex`. Compiler: pdfLaTeX.

## Rebuild figures and tables from the data
`python3 paper/make_figures.py` (figures + tables) and `python3 paper/draw_example.py` (worked example).
Data in `paper/data/` comes from Google Drive `fyp_dna/results/stage5/`.

## Authors
Hassan Ali, Muhammad Haad, Mahek Kalani (equal contribution); supervisor Dr. Muhammad Ismail Mangrio;
Department of Computer Science, Sukkur IBA University, Sukkur, Pakistan.
