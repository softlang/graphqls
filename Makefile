all: evolution star_fork querykind vennrepos

.PHNOY: prep
prep:
	@mkdir -p out
	@mkdir -p figures

evolution: prep
	@./src/activity.py
	@mv evolution.pdf figures

star_fork: prep
	@./src/dataset.py
	@./src/star.sc
	@./src/grid.py
	@mv star_fork.pdf figures

querykind: prep
	@./src/barplot_kinds.py
	@mv querykind.pdf figures

vennrepos: prep
	@./src/dataset.py
	@./src/vdiag.py
	@mv vennrepos.pdf figures

.PHONY: clean
clean:
	@rm -rf lib/__pycache__/
	@rm -rf out

.PHONY: purge
purge: clean
	@rm -rf figures
