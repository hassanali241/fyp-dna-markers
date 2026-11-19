# Phase 1 conference paper (two versions)

- `paper_ieee.pdf`: IEEE conference format (two columns), extended version with pipeline diagram, strand-layout
  figure, baselines, marker-selection details and the length/number figure (7 pages).
- `paper_lncs.pdf`: Springer LNCS format (one column), core version (10 pages + references).

Both use the same text (`body.tex`, `abstract.tex`), figures (`figures/`) and references (`refs.bib`).
IEEE-only parts are inside `\ifextended ... \fi` in `body.tex`.

## Edit on Overleaf
1. Zip this `paper/` folder (or download it from GitHub).
2. Overleaf > New Project > Upload Project > choose the zip.
3. Menu > Main document: `paper_ieee.tex` or `paper_lncs.tex`. Compiler: pdfLaTeX.

## Rebuild figures and tables from the data
`python3 paper/make_figures.py` (all figures and tables).
Data in `paper/data/` comes from Google Drive `fyp_dna/results/stage5/`.

## Authors
Hassan Ali, Muhammad Haad, Mahek Kalani (equal contribution); supervisor Dr. Muhammad Ismail Mangrio;
Department of Computer Science, Sukkur IBA University, Sukkur, Pakistan.

## Checks done
- All 18 references verified against publisher/arXiv records (authors, title, venue, year, pages, DOI).
- Citations, figure, table and section references are clickable (blue).
- Figures and tables are numbered in the order the text first cites them.

## Word versions
`paper/word/paper_ieee.docx` and `paper/word/paper_lncs.docx` are generated from the same LaTeX sources by
`python3 paper/make_word.py` (needs pandoc). They are single-column and meant for editing and comments; the PDFs
from LaTeX remain the formatted versions for submission.
