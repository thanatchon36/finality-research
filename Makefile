.PHONY: test study quick paper arxiv
test:
	python -m pytest -q
quick:
	python run_study.py --quick --out results-quick
study:
	python run_study.py
paper:
	python paper/analysis.py && python paper/make_figures.py && cd paper && latexmk -pdf -interaction=nonstopmode paper.tex
arxiv:
	sh paper/make_arxiv.sh
