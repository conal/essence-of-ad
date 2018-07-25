PAPER = essence-of-ad

SHORT = $(PAPER)-short
LONG = $(PAPER)-long
ICFP = $(PAPER)-icfp

# $(ICFP).pdf is the final camera-ready version from the publisher.

.PRECIOUS: %.tex %.pdf

# ALL += $(SHORT).pdf
ALL += $(LONG).pdf

all: $(ALL)

other.pdf: $(EXTENDED).pdf
	cp $? $@

texdeps = formatting.fmt Makefile

$(SHORT).tex: $(PAPER).lhs $(texdeps)
	lhs2TeX -o $*.tex $(PAPER).lhs

$(LONG).tex: $(PAPER).lhs $(texdeps)
	lhs2TeX --set=extended --set=long -o $*.tex $(PAPER).lhs

%.tex: %.lhs $(texdeps)
	lhs2TeX -o $*.tex $*.lhs

pdfdeps = $(pdfs) macros.tex bib.bib acmart.cls ACM-Reference-Format.bst

short: $(SHORT).pdf
long: $(LONG).pdf

see: $(SHORT).see

short.see: $(SHORT).see
long.see:  $(LONG).see
icfp.see:  $(ICFP).see

dots = $(wildcard Figures/*.dot)
pdfs = $(addsuffix .pdf, $(basename $(dots)))

#latex=pdflatex
latex=latexmk -pdf -halt-on-error

short.zip: $(SHORT).tex $(SHORT).bbl macros.tex $(pdfs) acmart.cls ACM-Reference-Format.bst
	zip $@ $^

long.zip: $(LONG).tex $(LONG).bbl macros.tex $(pdfs)
	zip $@ $^

zips: short.zip long.zip

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
# web: web-token

web: $(ALL)
	scp $? $(STASH)/
	touch $@

# How does make know that the web target is up to date without the
# web-token trick?
