PAPER = essence-of-ad

EXTENDED = $(PAPER)-extended

EXTENDED_ARXIV = $(PAPER)-arxiv

# ANON = $(PAPER)-anon
# EXTENDED_ANON = $(EXTENDED)-anon

.PRECIOUS: %.tex %.pdf

# all: $(PAPER).pdf

# # This target for a second view
# all: other.pdf

all: $(PAPER).pdf
# all: $(EXTENDED).pdf
# all: $(EXTENDED_ARXIV).pdf

# all: $(ANON).pdf

other.pdf: $(EXTENDED).pdf
	cp $? $@

moredeps = formatting.fmt macros.tex acmart.cls ACM-Reference-Format.bst Makefile

# $(ANON).tex: $(PAPER).lhs $(moredeps)
# 	lhs2TeX --set=anonymous -o $*.tex $(PAPER).lhs

$(EXTENDED).tex: $(PAPER).lhs $(moredeps)
	lhs2TeX --set=extended --set=draft -o $*.tex $(PAPER).lhs

$(EXTENDED_ARXIV).tex: $(PAPER).lhs $(moredeps)
	lhs2TeX --set=extended --set=arXiv -o $*.tex $(PAPER).lhs

$(EXTENDED_ANON).tex: $(PAPER).lhs $(moredeps)
	lhs2TeX --set=extended --set=anonymous -o $*.tex $(PAPER).lhs

see: $(PAPER).see

dots = $(wildcard Figures/*.dot)
pdfs = $(addsuffix .pdf, $(basename $(dots)))

#latex=pdflatex
latex=latexmk -pdf

# Loses embedded links :(
arXiv.pdf: arXiv.tex $(EXTENDED_ARXIV).pdf Makefile
	$(latex) arXiv.tex

arXiv.zip: $(EXTENDED_ARXIV).tex $(EXTENDED_ARXIV).bbl $(moredeps) $(pdfs)
	zip $@ $^

%.pdf: %.tex $(pdfs) bib.bib Makefile
	$(latex) $*.tex

%.tex: %.lhs formatting.fmt
	lhs2TeX -o $*.tex $*.lhs

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

web-token: $(EXTENDED).pdf
	scp $? $(STASH)/
	touch $@
