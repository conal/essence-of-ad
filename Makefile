TARG = essence-of-ad

.PRECIOUS: %.tex %.pdf %.web

# all: $(TARG).pdf

# This target for a second view
all: other.pdf

other.pdf: $(TARG).pdf
	cp $? $@

see: $(TARG).see

dots = $(wildcard Figures/*.dot)
pdfs = $(addsuffix .pdf, $(basename $(dots))) $(wildcard Figures/circuits/*-scaled.pdf)

#latex=pdflatex
latex=latexmk -pdf

%.pdf: %.tex $(pdfs) bib.bib Makefile
	$(latex) $*.tex

%.tex: %.lhs macros.tex formatting.fmt Makefile
	lhs2TeX -o $*.tex $*.lhs

showpdf = open -a Skim.app

%.see: %.pdf
	${showpdf} $*.pdf

# Cap the size so that LaTeX doesn't choke.
%.pdf: %.dot # Makefile
	dot -Tpdf -Gmargin=0 -Gsize=10,10 $< -o $@

pdfs: $(pdfs)

clean:
	rm -f $(TARG).{tex,pdf,aux,nav,snm,ptb,log,out,toc,bbl,blg,fdb_latexmk,fls}

web: web-token

STASH=conal@conal.net:/home/conal/web/talks
web: web-token

web-token: $(TARG).pdf
	scp $? $(STASH)/simple-essence-of-automatic-differentiation.pdf
	touch $@
