PAPER = essence-of-ad

ICFP = $(PAPER)-icfp
ARXIV = $(PAPER)-arxiv

.PRECIOUS: %.tex %.pdf

all: $(ICFP).pdf
# all: $(ARXIV).pdf

other.pdf: $(EXTENDED).pdf
	cp $? $@

texdeps = formatting.fmt Makefile

$(ICFP).tex: $(PAPER).lhs $(texdeps)
	lhs2TeX -o $*.tex $(PAPER).lhs

$(ARXIV).tex: $(PAPER).lhs $(texdeps)
	lhs2TeX --set=extended --set=arXiv -o $*.tex $(PAPER).lhs

%.tex: %.lhs $(texdeps)
	lhs2TeX -o $*.tex $*.lhs

pdfdeps = $(pdfs) macros.tex bib.bib acmart.cls ACM-Reference-Format.bst

see: $(ICFP).see

dots = $(wildcard Figures/*.dot)
pdfs = $(addsuffix .pdf, $(basename $(dots)))

#latex=pdflatex
latex=latexmk -pdf

icfp.zip: $(ICFP).tex $(ICFP).bbl macros.tex $(pdfs) acmart.cls ACM-Reference-Format.bst
	zip $@ $^

arxiv.zip: $(ARXIV).tex $(ARXIV).bbl macros.tex $(pdfs)
	zip $@ $^

%.pdf: %.tex $(pdfdeps)
	$(latex) $*.tex

showpdf = open -a Skim.app

%.see: %.pdf
	${showpdf} $*.pdf

# Cap the size so that LaTeX doesn't choke.
%.pdf: %.dot # Makefile
	dot -Tpdf -Gmargin=0 -Gsize=10,10 $< -o $@

pdfs: $(pdfs)

clean:
	rm -f $(PAPER)*.{tex,pdf,aux,nav,snm,ptb,log,out,toc,bbl,blg,fdb_latexmk,fls}

web: web-token

STASH=conal@conal.net:/home/conal/web/papers/essence-of-ad
web: web-token

web-token: $(ICFP).pdf
	scp $? $(STASH)/
	touch $@
