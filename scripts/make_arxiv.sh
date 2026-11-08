#!/bin/sh
# Build an arXiv-ready source archive. arXiv does not run BibTeX, so the generated .bbl is included.
set -e
cd "$(dirname "$0")/../paper"
latexmk -pdf -interaction=nonstopmode paper.tex
rm -rf arxiv arxiv-source.zip && mkdir -p arxiv/sections arxiv/tables arxiv/figures
cp paper.tex paper.bbl refs.bib arxiv/
cp sections/*.tex arxiv/sections/ && cp tables/*.tex arxiv/tables/ && cp figures/*.pdf arxiv/figures/
(cd arxiv && zip -qr ../arxiv-source.zip .)
rm -rf arxiv
echo "wrote paper/arxiv-source.zip"
