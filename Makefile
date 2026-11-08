PY ?= .venv/bin/python

.PHONY: setup test quick study figures paper web banner poster arxiv hf hf-upload all clean

setup:          ## create venv and install the package (needs uv)
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev,paper]"

test:           ## 19 unit tests, about 1 s
	$(PY) -m pytest -q

quick:          ## 15-second smoke run of the study -> results-quick/
	$(PY) run_study.py --quick --out results-quick

study:          ## full Monte Carlo study from master seed 20261108 (~5 min, 1 core; resumable)
	$(PY) run_study.py

figures:        ## paper tables, number macros and vector figures (with the overlap check)
	$(PY) paper/analysis.py
	$(PY) paper/make_figures.py

paper: figures  ## compile paper/paper.pdf (latexmk + pdfLaTeX, needs TeX Live)
	cd paper && latexmk -pdf -interaction=nonstopmode paper.tex

web:            ## web reader -> docs/paper/index.html (pandoc; copies the current paper.pdf)
	$(PY) scripts/paper_web.py

banner:         ## README banner -> docs/assets/banner.png
	$(PY) scripts/make_banner.py

poster:         ## LinkedIn poster -> docs/assets/linkedin_poster.png (real intervals from the coverage study)
	$(PY) scripts/make_poster.py

arxiv:          ## arXiv source archive, .bbl included -> paper/arxiv-source.zip
	sh scripts/make_arxiv.sh

hf:             ## Hugging Face dataset -> hf/ (regenerated from the seeds, checked against results/)
	$(PY) scripts/export_hf.py

hf-upload: hf   ## upload hf/ to huggingface.co/datasets/thanatchon36/finality-research (needs hf auth login)
	hf upload thanatchon36/finality-research hf . --repo-type dataset --commit-message "Update dataset"

all: study paper web

clean:
	rm -rf results-quick hf paper/arxiv paper/arxiv-source.zip
	cd paper && rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.bbl *.blg *.synctex.gz
