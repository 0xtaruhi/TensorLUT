# latexmk configuration for the ASP-DAC paper.
# Usage (after `module load texlive/2026`):
#   latexmk aspdac2027.tex      # build PDF (runs pdflatex+bibtex as many times as needed)
#   latexmk -pvc aspdac2027.tex # watch mode: rebuild on save
#   latexmk -c                  # clean aux files (keep PDF)
#   latexmk -C                  # clean everything including PDF

$pdf_mode = 1;                 # generate PDF via pdflatex
$bibtex_use = 2;               # always run bibtex; clean .bbl on -C
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';

# Warn (don't silently pass) if a run leaves overfull/undefined issues.
$warnings_as_errors = 0;

# Extra byproducts to remove on cleanup.
$clean_ext = 'bbl nav out snm synctex.gz run.xml fdb_latexmk fls';

# Keep the build in-place next to the sources.
$out_dir = '.';
