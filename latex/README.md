# LaTeX package (ACM acmart) — gitignored

This folder holds the camera draft of the final report in ACM `acmart` format.
It is intentionally listed in the repo `.gitignore` and is **not** pushed to GitLab.

## Contents
- `main.tex` — full paper, `\documentclass[sigconf,nonacm]{acmart}`.
- `references.bib` — verified bibliography (authors checked against arXiv/DBLP 2026-06-16).
- `fig/` — all figures copied from the project `fig/` directory.

## Compile (Overleaf recommended; no local TeX toolchain on this machine)
1. Upload this whole `latex/` folder to a new Overleaf project (or zip it).
2. Set the compiler to **pdfLaTeX**.
3. Build order: pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX.

## Before submission
- Fill the repository URL in Appendix A (`[insert GitHub/GitLab URL]`).
- Team confirms the Appendix B contribution split.
- Check the paper lands in the 8–12 page range (excluding appendices); trim if over.
